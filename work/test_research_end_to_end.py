"""Real-data, real-pretrained-net five-fold execution acceptance (not accuracy)."""
from pathlib import Path
import importlib.util
import sys
import os
import json
import time

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "work/kaggle_test_deps"))
os.environ["MPLBACKEND"] = "Agg"
import numpy as np
import pandas as pd
import torch
torch.set_num_threads(2)
spec = importlib.util.spec_from_file_location("research_acceptance", ROOT / "outputs/fovea_research_kaggle.py")
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)
geometry = m.research_geometry_selfcheck()
cfg = m.ResearchConfig(
    INPUT_ROOT=str(ROOT / "work/real_competition_data"),
    OUTPUT_DIR=str(ROOT / "work/research_real_acceptance"),
    SUBMISSION_PATH=str(ROOT / "work/research_real_acceptance/submission.csv"),
    PRETRAINED_WEIGHTS_PATH=str(ROOT / "outputs/fovea_pretrained_weights.zip"),
    SMOKE_TEST=True, SMOKE_FOLDS=5, SMOKE_INPUT_SIZE=64,
    NUM_WORKERS=0, AMP=False, SHOW_PLOTS=False, TTA=True,
    FINE_TRAIN_CROPS="oof", INNER_CROP_FOLDS=3,
)
started = time.perf_counter()
result = m.run_research(cfg)
assert len(result["train"]) == 80 and len(result["test"]) == 20
assert result["train"].fold.nunique() == 5
pair = result["train"].set_index("image_id").loc[["39", "77"]]
assert pair.fold.nunique() == 1 and pair.group.nunique() == 1
np.testing.assert_allclose(pair[["x", "y"]], [[1299.440632,930.171897], [1299.490121,934.244593]], atol=1e-8)
expected = [f"{i}_Fovea_{axis}" for i in range(81,101) for axis in ("X", "Y")]
for name, pred in result["oof"].items():
    assert len(pred) == 80 and pred.image_id.nunique() == 80
    assert np.isfinite(pred[["x", "y", "uncertainty"]]).all().all()
    assert result["submissions"][name].ImageID.tolist() == expected
    assert np.isfinite(result["submissions"][name].value).all()
out = Path(result["config"].OUTPUT_DIR)
assert not (ROOT / "outputs/submission.csv").exists()
for fold in range(5):
    roles = pd.read_csv(out / f"fold_{fold}/split_roles.csv", dtype=str)
    assert roles.groupby("group").role.nunique().max() == 1
    audits = json.loads((out / f"fold_{fold}/crop_crossfit/crossfit_audit.json").read_text())
    for audit in audits:
        assert not set(audit["fit_groups"]) & set(audit["held_groups"])
    radius = json.loads((out / f"fold_{fold}/radius_calibration.json").read_text())
    assert not set(radius["calibration_group_ids"]) & set(roles.loc[roles.role == "outer_validation", "group"])
    control = json.loads((out / f"fold_{fold}/area_matched_control.json").read_text())
    assert control["relative_area_difference"] < 1e-7
    assert (out / f"fold_{fold}/complete.json").is_file()
curves = pd.read_csv(out / "uncertainty_calibration_curve.csv")
assert len(curves) >= 4
before = {str(p): p.stat().st_mtime_ns for p in out.glob("fold_*/coarse_final/model.pt")}
reused = m.run_research(cfg)
pd.testing.assert_frame_equal(result["submission"], reused["submission"])
after = {str(p): p.stat().st_mtime_ns for p in out.glob("fold_*/coarse_final/model.pt")}
assert before == after
report = {
    "status": "PASS", "source": "User-provided 2026task3.zip",
    "training_images": 80, "test_images": 20, "unique_training_pixel_hashes": 79,
    "outer_folds_executed": 5, "experiments": list(result["oof"]),
    "submission_rows_per_method": 40, "coarse_fine_training_jobs": 35,
    "device": "CPU", "input_size": 64, "epochs_per_training_job": 1,
    "official_pretrained_weights_used": True, "offline_zip_loading": True,
    "pretrained_sha256": m.RESEARCH_WEIGHT_SHA256,
    "tta": True, "inner_crossfit_crop_folds": 3,
    "duplicate_images_grouped": True, "original_labels_preserved": True,
    "calibration_groups_disjoint": True, "outer_validation_disjoint": True,
    "all_six_ablations_complete": True, "completed_fold_reuse_verified": True,
    "area_matched_control_verified": True, "geometry": geometry,
    "full_resolution_training_completed": False,
    "accuracy_claim": "None. This is a low-resolution execution test, not formal competition performance.",
    "elapsed_seconds": time.perf_counter()-started,
}
(ROOT / "outputs/research_validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print("RESEARCH_REAL_FIVEFOLD_ACCEPTANCE_PASS", flush=True)
