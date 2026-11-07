"""Deterministic calibration-policy checks; synthetic cases are NOT accuracy evidence."""
import importlib.util
import json
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "work/kaggle_test_deps"))
import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location("research_calibration_part", ROOT / "work/fovea_v3/research_calibration_part.py")
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)
old_spec = importlib.util.spec_from_file_location("legacy_calibration_checks", ROOT / "work/test_research_calibration.py")
old = importlib.util.module_from_spec(old_spec)
old_spec.loader.exec_module(old)
old_config = old.config
# Keep all 14 prior checks unchanged. Their deliberately tiny fixtures exercise
# rule mechanics with an explicit low minimum; new checks verify production 12.
old.config = lambda **kwargs: old_config(**{"FALLBACK_MIN_GROUPS": 2, **kwargs})


def fixture(n=16, coarse_error=2.0, fine_error=20.0):
    truth = pd.DataFrame({"image_id": [str(i) for i in range(n)], "group": [f"g{i}" for i in range(n)],
                          "x": np.full(n, 400.), "y": np.full(n, 300.),
                          "width": np.full(n, 1000), "height": np.full(n, 800)})
    coarse = truth[["image_id", "x", "y"]].copy()
    coarse.x += coarse_error
    coarse["uncertainty"] = np.linspace(1, 10, n)
    fine = truth[["image_id", "x", "y"]].copy()
    fine.x += fine_error
    fine["crop_edge_fraction"] = .01
    return truth, coarse, fine


class Round4Checks(unittest.TestCase):
    def test_default_small_group_guard_and_reason(self):
        truth, coarse, fine = fixture(16)
        truth.group = [f"g{i % 3}" for i in range(16)]
        rule = m.fit_fallback_rule(coarse, fine, truth, old_config())
        self.assertFalse(rule["enabled"])
        self.assertEqual(rule["selection_reason"], "insufficient_calibration_groups")
        self.assertEqual(rule["evaluated_candidate_count"], 0)
        self.assertEqual(rule["n_groups"], 3)

    def test_default_three_candidates_material_gain_and_json(self):
        truth, coarse, fine = fixture()
        rule = m.fit_fallback_rule(coarse, fine, truth, old_config())
        self.assertTrue(rule["enabled"])
        self.assertEqual(rule["candidate_count"], 3)
        self.assertEqual(rule["evaluated_candidate_count"], 3)
        self.assertGreaterEqual(rule["baseline_calibration_metrics"]["rmse"] - rule["selected_calibration_metrics"]["rmse"], 1.)
        json.dumps(rule, allow_nan=False)

    def test_absolute_small_gain_is_rejected(self):
        truth, coarse, fine = fixture(coarse_error=9., fine_error=10.)
        rule = m.fit_fallback_rule(coarse, fine, truth, old_config())
        self.assertFalse(rule["enabled"])
        self.assertEqual(rule["required_gain_px"], 1.)

    def test_relative_small_gain_is_rejected(self):
        truth, coarse, fine = fixture(coarse_error=98., fine_error=100.)
        rule = m.fit_fallback_rule(coarse, fine, truth, old_config(FALLBACK_MIN_RELATIVE_GAIN=.03))
        self.assertFalse(rule["enabled"])
        self.assertGreater(rule["required_gain_px"], 2.)

    def test_rule_invariant_to_all_id_permutations(self):
        truth, coarse, fine = fixture()
        a = m.fit_fallback_rule(coarse, fine, truth, old_config())
        b = m.fit_fallback_rule(coarse.sample(frac=1, random_state=3), fine.iloc[::-1], truth.sample(frac=1, random_state=4), old_config())
        for key in ("enabled", "edge_threshold", "disagreement_threshold", "selected_calibration_metrics"):
            self.assertEqual(a[key], b[key])

    def test_invalid_fallback_settings_and_nonfinite_data(self):
        truth, coarse, fine = fixture()
        for change in ({"FALLBACK_MIN_GROUPS": 1}, {"FALLBACK_MIN_GROUPS": 12.5},
                       {"FALLBACK_MIN_GAIN_PX": np.inf}, {"FALLBACK_MIN_RELATIVE_GAIN": np.nan},
                       {"FALLBACK_RULE_CANDIDATES": ((.6, 4),)}, {"FALLBACK_RULE_CANDIDATES": ((.1, np.inf),)}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                m.fit_fallback_rule(coarse, fine, truth, old_config(**change))
        fine.loc[0, "crop_edge_fraction"] = np.nan
        with self.assertRaises(ValueError):
            m.fit_fallback_rule(coarse, fine, truth, old_config())

    def test_numerical_edge_tolerance_does_not_trigger_zero(self):
        truth, coarse, fine = fixture(1)
        fine["crop_edge_fraction"] = -5e-7
        result = m.apply_fallback(coarse, fine, {"enabled": True, "edge_threshold": 0., "disagreement_threshold": 1e9})
        self.assertFalse(result.fallback_used.any())
        fine["crop_edge_fraction"] = -.01
        with self.assertRaises(ValueError):
            m.apply_fallback(coarse, fine, {"enabled": True, "edge_threshold": 0., "disagreement_threshold": 1e9})

    def test_diagnostic_clamp_degeneracy_permutation_and_no_mutation(self):
        truth, coarse, _ = fixture()
        radius = m.fit_radius_calibration(coarse, truth, old_config())
        roi = m.make_roi_spec(coarse, truth, old_config(), "calibrated", radius)
        copies = [v.copy(deep=True) for v in (truth, coarse, roi)]
        a = m.research_calibration_diagnostics(coarse, truth, roi, radius, old_config())
        b = m.research_calibration_diagnostics(coarse.iloc[::-1], truth.sample(frac=1, random_state=2), roi.sample(frac=1, random_state=3), radius, old_config())
        self.assertEqual(a, b)
        self.assertEqual(a["sample_role"], "calibration_fit")
        self.assertEqual(a["roi_min_clamp_rate"], 1.)
        self.assertTrue(a["adaptive_roi_nearly_constant"])
        self.assertIsNone(a["dispersion_error_spearman"])
        self.assertFalse(a["changes_parameters"])
        for current, previous in zip((truth, coarse, roi), copies):
            pd.testing.assert_frame_equal(current, previous)
        json.dumps(a, allow_nan=False)

    def test_diagnostic_held_out_and_positive_negative_association(self):
        truth, coarse, _ = fixture()
        coarse.uncertainty += 2.0  # Keep every value above the configured floor.
        radius = m.fit_radius_calibration(coarse, truth, old_config())
        truth.image_id = "outer_" + truth.image_id
        truth.group = "outer_" + truth.group
        coarse.image_id = truth.image_id
        coarse.x = truth.x + coarse.uncertainty * 4
        roi = m.make_roi_spec(coarse, truth, old_config(), "calibrated", radius)
        a = m.research_calibration_diagnostics(coarse, truth, roi, radius, old_config())
        self.assertEqual(a["sample_role"], "held_out_from_radius_calibration")
        self.assertAlmostEqual(a["dispersion_error_spearman"], 1.)
        coarse.x = truth.x + (14 - coarse.uncertainty) * 4
        roi = m.make_roi_spec(coarse, truth, old_config(), "calibrated", radius)
        b = m.research_calibration_diagnostics(coarse, truth, roi, radius, old_config())
        self.assertAlmostEqual(b["dispersion_error_spearman"], -1.)

    def test_one_group_zero_error_radius_is_finite_and_no_fallback(self):
        truth, coarse, fine = fixture(1, coarse_error=0.)
        radius = m.fit_radius_calibration(coarse, truth, old_config())
        self.assertEqual(radius["alpha"], 0.)
        roi = m.make_roi_spec(coarse, truth, old_config(), "calibrated", radius)
        report = m.research_calibration_diagnostics(coarse, truth, roi, radius, old_config())
        self.assertEqual(report["bounded_roi_image_coverage"], 1.)
        self.assertIsNone(report["dispersion_error_spearman"])
        self.assertFalse(m.fit_fallback_rule(coarse, fine, truth, old_config())["enabled"])
        json.dumps(report, allow_nan=False)

    def test_diagnostic_boundary_and_group_coverage(self):
        truth, coarse, _ = fixture(3)
        truth.group = ["same", "same", "other"]
        roi = m.make_roi_spec(coarse, truth, old_config(), "fixed")
        truth.x = roi.roi_x1
        truth.loc[1, "x"] += .01
        report = m.research_calibration_diagnostics(coarse, truth, roi, {}, old_config())
        self.assertAlmostEqual(report["bounded_roi_image_coverage"], 2 / 3)
        self.assertEqual(report["bounded_roi_group_coverage"], .5)

    def test_diagnostic_rejects_nonfinite_and_id_mismatch(self):
        truth, coarse, _ = fixture()
        roi = m.make_roi_spec(coarse, truth, old_config(), "fixed")
        invalid = roi.copy()
        invalid["roi_at_min"] = invalid.roi_at_min.astype(float)
        invalid.loc[0, "roi_at_min"] = np.nan
        with self.assertRaises(ValueError):
            m.research_calibration_diagnostics(coarse, truth, invalid, {}, old_config())
        with self.assertRaises(ValueError):
            m.research_calibration_diagnostics(coarse, truth.iloc[:-1], roi, {}, old_config())
        with self.assertRaises(ValueError):
            m.research_calibration_diagnostics(coarse, truth, roi, {"target_coverage": np.nan}, old_config())
        invalid = roi.copy()
        invalid.loc[0, "crop_side"] = np.inf
        with self.assertRaises(ValueError):
            m.research_calibration_diagnostics(coarse, truth, invalid, {}, old_config())


if __name__ == "__main__":
    suite = unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromTestCase(old.CalibrationChecks),
                               unittest.defaultTestLoader.loadTestsFromTestCase(Round4Checks)])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report = {"round": 4, "tests_run": result.testsRun, "failures": len(result.failures), "errors": len(result.errors),
              "legacy_tests_preserved": 14, "new_policy_tests": 12,
              "legacy_fixture_override": "FALLBACK_MIN_GROUPS=2 for the existing four-image/three-group mechanics fixture only; default 12 is tested separately.",
              "accuracy_evidence": False, "source": "work/fovea_v3/research_calibration_part.py"}
    (Path(__file__).parent / "round4_calibration_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    raise SystemExit(0 if result.wasSuccessful() else 1)
