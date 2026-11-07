"""Real-data smoke run; never present its predictions as a trained submission."""
from pathlib import Path
import os
import sys
import json
import importlib.util

ROOT = Path(__file__).resolve().parent.parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "work/kaggle_test_deps"))
os.environ["MPLBACKEND"] = "Agg"
import numpy as np
import pandas as pd
import torch
torch.set_num_threads(2)

spec = importlib.util.spec_from_file_location("real_fovea", ROOT / "outputs/fovea_kaggle.py")
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)
m.quick_sanity_checks()
cfg = m.Config(
    INPUT_ROOT=str(ROOT / "work/real_competition_data"),
    OUTPUT_DIR=str(ROOT / "work/real_competition_acceptance"),
    SUBMISSION_PATH=str(ROOT / "work/real_competition_acceptance/submission.csv"),
    SMOKE_TEST=True, NUM_WORKERS=0, PRETRAINED=False,
    ALLOW_WEIGHT_DOWNLOAD=False, SHOW_PLOTS=False, AMP=False,
)
result = m.run_pipeline(cfg)
assert len(result["train"]) == 80
assert len(result["test"]) == 20
assert len(result["submission"]) == 40
assert len(result["oof"]) == 80
pair = result["train"].set_index("image_id").loc[["39", "77"]]
assert pair.fold.nunique() == 1 and pair.group.nunique() == 1
np.testing.assert_allclose(pair[["x", "y"]].to_numpy(), [
    [1299.440632, 930.171897], [1299.490121, 934.244593]], atol=1e-8)
assert result["train"].path.map(lambda p: Path(p).name).tolist() == [f"{i:04d}.jpg" for i in range(1, 81)]
assert result["test"].path.map(lambda p: Path(p).name).tolist() == [f"{i:04d}.jpg" for i in range(81, 101)]
assert result["submission"].ImageID.tolist() == [
    f"{i}_Fovea_{axis}" for i in range(81, 101) for axis in ("X", "Y")]
assert np.isfinite(result["submission"].value).all()
out = Path(result["config"].OUTPUT_DIR)
xml = pd.read_csv(out / "xml_box_audit.csv")
report = {
    "status": "PASS",
    "data_source": "User-provided 2026task3.zip",
    "training_images": len(result["train"]),
    "test_images": len(result["test"]),
    "submission_rows": len(result["submission"]),
    "visible_training_images": int(result["train"].visible.sum()),
    "invisible_training_images": int((~result["train"].visible).sum()),
    "unique_pixel_hashes": int(result["train"].pixel_hash.nunique()),
    "image_dimensions": result["train"][["width", "height"]].drop_duplicates().to_dict("records"),
    "xml_boxes_parsed": len(xml),
    "duplicate_label_pair": {
        "ids": [39, 77], "distance_px": 4.072997,
        "original_labels_preserved": True, "same_validation_fold": True,
    },
    "test_type": "CPU real-data smoke run: 2 folds, 2 epochs per fold, 128 input, random initialization",
    "full_5fold_training_completed": False,
    "tests": ["numeric CSV IDs matched to four-digit jpg filenames",
              "real coordinates validated", "training and test disjoint",
              "two-fold OOF covers all 80 labels",
              "duplicate images with small label differences remain in the same fold",
              "checkpoints saved and reloaded", "horizontal flip TTA",
              "submission preserved all 40 original IDs and row order",
              "no nonfinite output values"],
}
(ROOT / "work/real_data_acceptance_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print("REAL_DATA_ACCEPTANCE_PASS")
