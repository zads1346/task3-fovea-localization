"""Small deterministic scientific checks for calibration/ROI evaluation."""
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent / "kaggle_test_deps"))
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import pandas as pd
from research_calibration_part import (
    fit_radius_calibration, make_roi_spec, roi_contains_truth,
    localization_metrics, fit_fallback_rule, apply_fallback,
    uncertainty_diagnostics, paired_group_bootstrap,
)


def config(**overrides):
    return SimpleNamespace(**{
        "ROI_COVERAGE": 0.9, "UNCERTAINTY_FLOOR_PX": 2.0,
        "ROI_MIN_FRACTION": 0.15, "ROI_MAX_FRACTION": 0.60,
        "ROI_MARGIN_PX": 16.0, "FIXED_ROI_FRACTION": 0.30,
        "RAW_RADIUS_MULTIPLIER": 2.0, **overrides,
    })


def examples():
    truth = pd.DataFrame({"image_id": ["a", "b", "c", "d"],
                          "x": [100., 200., 300., 400.], "y": [200.] * 4,
                          "width": [1000] * 4, "height": [800] * 4,
                          "group": ["g0", "g0", "g1", "g2"]})
    coarse = truth[["image_id", "x", "y"]].copy()
    coarse["x"] += [2, 8, 4, 6]
    coarse["uncertainty"] = [2., 2., 2., 2.]
    return truth, coarse


class CalibrationChecks(unittest.TestCase):
    def test_id_alignment_is_permutation_invariant(self):
        truth, pred = examples()
        a = localization_metrics(pred, truth)
        b = localization_metrics(pred.sample(frac=1, random_state=1), truth.iloc[::-1])
        self.assertEqual(a, b)
        self.assertAlmostEqual(a["rmse"], np.sqrt((4 + 64 + 16 + 36) / 8))

    def test_group_max_not_row_weighted_quantile(self):
        truth, pred = examples()
        fitted = fit_radius_calibration(pred, truth.iloc[::-1], config(ROI_COVERAGE=0.5))
        # Per-group ratios [4,2,3] -> middle 3. Per-row ratios would be [1,4,2,3].
        self.assertEqual(fitted["alpha"], 3.0)
        self.assertEqual(fitted["n_images"], 4)
        self.assertEqual(fitted["n_groups"], 3)
        json.dumps(fitted, allow_nan=False)

    def test_reject_nan_uncertainty_and_missing_ids(self):
        truth, pred = examples()
        pred.loc[0, "uncertainty"] = np.nan
        with self.assertRaises(ValueError):
            fit_radius_calibration(pred, truth, config())
        with self.assertRaises(ValueError):
            localization_metrics(pred.iloc[1:], truth)

    def test_reject_duplicate_prediction_and_negative_spread(self):
        truth, pred = examples()
        with self.assertRaises(ValueError):
            localization_metrics(pd.concat([pred, pred.iloc[:1]]), truth)
        pred.loc[0, "uncertainty"] = -1
        with self.assertRaises(ValueError):
            fit_radius_calibration(pred, truth, config())

    def test_uncertainty_floor_and_exact_predictions(self):
        truth, pred = examples()
        pred[["x", "y"]] = truth[["x", "y"]]
        pred.uncertainty = 0.
        fitted = fit_radius_calibration(pred, truth, config())
        self.assertEqual(fitted["alpha"], 0.)
        roi = make_roi_spec(pred, truth, config(), "calibrated", fitted)
        self.assertTrue(np.allclose(roi.crop_side, 120.))
        self.assertTrue(roi_contains_truth(roi, truth).all())

    def test_roi_prediction_does_not_require_labels(self):
        truth, pred = examples()
        dimensions_only = truth[["image_id", "width", "height"]].iloc[::-1]
        roi = make_roi_spec(pred, dimensions_only, config(), "fixed")
        self.assertTrue(np.allclose(roi.crop_side, 800 * .3 + 32))
        self.assertTrue(np.allclose(roi.crop_x, pred.x))
        different_labels = truth.copy()
        different_labels.x = 999.
        identical_roi = make_roi_spec(pred, different_labels, config(), "fixed")
        pd.testing.assert_frame_equal(roi, identical_roi)

    def test_margin_then_clamp_applies_to_all_modes(self):
        truth, pred = examples()
        pred.uncertainty = 60.
        cfg = config(ROI_MAX_FRACTION=.31)
        fixed = make_roi_spec(pred, truth, cfg, "fixed")
        raw = make_roi_spec(pred, truth, cfg, "raw")
        calibrated = make_roi_spec(pred, truth, cfg, "calibrated", {"alpha": 2.0})
        for roi in (fixed, raw, calibrated):
            self.assertTrue(np.allclose(roi.crop_side, 248.))
            self.assertTrue(roi.roi_at_max.all())

    def test_clamp_can_defeat_nominal_coverage_and_reports_it(self):
        truth, pred = examples()
        pred.x = 900.
        fitted = fit_radius_calibration(pred, truth, config())
        roi = make_roi_spec(pred, truth, config(), "calibrated", fitted)
        metrics = localization_metrics(pred, truth, roi)
        self.assertEqual(fitted["fit_image_coverage_before_margin_and_clamp"], 1.)
        self.assertEqual(metrics["roi_coverage"], 0.)
        self.assertEqual(metrics["roi_max_clamp_rate"], 1.)
        self.assertTrue(roi.roi_has_padding.all())
        self.assertTrue((roi.roi_x1 <= 999).all())

    def test_fallback_is_prediction_rule_and_improves_calibration(self):
        truth, coarse = examples()
        fine = truth[["image_id", "x", "y"]].copy()
        fine.loc[0, "x"] += 100
        fine["crop_edge_fraction"] = [.01, .3, .3, .3]
        rule = fit_fallback_rule(coarse, fine.iloc[::-1], truth.iloc[::-1], config())
        self.assertTrue(rule["enabled"])
        result = apply_fallback(coarse.iloc[::-1], fine, rule)
        indexed = result.set_index("image_id")
        self.assertTrue(indexed.loc["a", "fallback_used"])
        self.assertEqual(indexed.loc["a", "x"], coarse.loc[0, "x"])
        self.assertLess(localization_metrics(result, truth)["rmse"], localization_metrics(fine, truth)["rmse"])
        json.dumps(rule, allow_nan=False)

    def test_no_fallback_when_fine_is_perfect(self):
        truth, coarse = examples()
        fine = truth[["image_id", "x", "y"]].copy()
        fine["crop_edge_fraction"] = .3
        rule = fit_fallback_rule(coarse, fine, truth, config())
        self.assertFalse(rule["enabled"])
        result = apply_fallback(coarse, fine, rule)
        self.assertEqual(localization_metrics(result, truth)["rmse"], 0.)

    def test_fallback_covariance_follows_coarse_with_shuffled_ids(self):
        truth, coarse = examples()
        coarse[["cov_xx", "cov_xy", "cov_yy"]] = np.array([
            [4., 1., 9.], [16., 2., 25.], [36., 3., 49.], [64., 4., 81.]])
        fine = truth[["image_id", "x", "y"]].copy()
        fine["uncertainty"] = .5
        fine["crop_edge_fraction"] = [.01, .3, .3, .3]
        fine[["cov_xx", "cov_xy", "cov_yy"]] = [.25, 0., .25]
        rule = {"enabled": True, "edge_threshold": .03,
                "disagreement_threshold": 1e9, "uncertainty_floor_px": 2.}
        result = apply_fallback(coarse.iloc[[2, 0, 3, 1]], fine.iloc[::-1], rule).set_index("image_id")
        self.assertTrue(result.loc["a", "fallback_used"])
        np.testing.assert_allclose(result.loc["a", ["cov_xx", "cov_xy", "cov_yy"]].to_numpy(float), [4., 1., 9.])
        self.assertEqual(result.loc["a", "uncertainty"], 2.)
        self.assertEqual(result.loc["a", "x"], coarse.loc[0, "x"])
        self.assertFalse(result.loc["b", "fallback_used"])
        np.testing.assert_allclose(result.loc["b", ["cov_xx", "cov_xy", "cov_yy"]].to_numpy(float), [.25, 0., .25])
        self.assertEqual(result.loc["b", "uncertainty"], .5)

    def test_risk_coverage_ranks_predicted_uncertainty_not_truth(self):
        truth, pred = examples()
        pred.x = truth.x + [100., 10., 2., 1.]
        pred.uncertainty = [1., 3., 5., 7.]
        result = uncertainty_diagnostics(pred, truth, cfg=config())
        first = result["risk_coverage"][0]
        self.assertGreater(first["rmse_by_predicted_uncertainty"], first["oracle_rmse_by_true_error"])
        self.assertAlmostEqual(result["full_coverage_rmse"], localization_metrics(pred, truth)["rmse"])
        self.assertAlmostEqual(result["uncertainty_error_spearman"], -1.)

    def test_bootstrap_preserves_alignment_and_paired_differences(self):
        truth, pred = examples()
        methods = {"baseline": pred, "same_reordered": pred.iloc[::-1],
                   "perfect": truth[["image_id", "x", "y"]]}
        result = paired_group_bootstrap(methods, truth.iloc[::-1], n_bootstrap=300, seed=7).set_index("method")
        self.assertEqual(result.loc["same_reordered", "delta_ci_low"], 0.)
        self.assertEqual(result.loc["same_reordered", "delta_ci_high"], 0.)
        self.assertLess(result.loc["perfect", "delta_ci_high"], 0.)
        self.assertTrue((result.n_groups == 3).all())

    def test_diagnostics_json_and_duplicate_group_coverage(self):
        truth, pred = examples()
        fitted = fit_radius_calibration(pred, truth, config())
        diagnostics = uncertainty_diagnostics(pred.iloc[::-1], truth, fitted, config())
        self.assertIsNone(diagnostics["uncertainty_error_spearman"])
        self.assertEqual(diagnostics["bounded_roi_calibration_curve"][-1]["actual_bounded_roi_group_coverage"], 1.)
        json.dumps(diagnostics, allow_nan=False)


if __name__ == "__main__":
    unittest.main(verbosity=2)
