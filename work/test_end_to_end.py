"""Synthetic acceptance run. Never uses Kaggle competition data."""
from pathlib import Path
import sys
import os
import importlib.util
import json

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "work/kaggle_test_deps"))
os.environ["MPLBACKEND"] = "Agg"

import numpy as np
import pandas as pd
from PIL import Image
import torch

torch.set_num_threads(2)
spec = importlib.util.spec_from_file_location("fovea_pipeline", ROOT / "outputs/fovea_kaggle.py")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
module.quick_sanity_checks()

test_root = ROOT / "work/synthetic_acceptance"
data_root = test_root / "input/detection"
for directory in ("train", "test", "train_location"):
    (data_root / directory).mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(16)
labels = []
for i in range(10):
    width, height = (181 + i * 2, 137 + i * 4) if i % 2 else (135 + i * 2, 181 + i * 3)
    pixels = rng.integers(0, 80, (height, width, 3), dtype=np.uint8)
    if i < 8:
        x, y = int(width * (0.35 + 0.03 * i)), int(height * (0.55 - 0.02 * i))
        pixels[y-4:y+5, x-4:x+5] = [250, 100, 80]
        xml = (f"<annotation><filename>train__{i:03d}.png</filename><object><name>fovea</name>"
               f"<bndbox><xmin>{x-10}</xmin><ymin>{y-10}</ymin><xmax>{x+10}</xmax>"
               f"<ymax>{y+10}</ymax></bndbox></object></annotation>")
        (data_root / "train_location" / f"train__{i:03d}.xml").write_text(xml)
    else:
        x, y = 0, 0
    filename = f"train__{i:03d}.png"
    Image.fromarray(pixels).save(data_root / "train" / filename)
    labels.append({"data": filename, "Fovea_X": x, "Fovea_Y": y})
pd.DataFrame(labels).to_csv(data_root / "fovea_localization_train_GT.csv", index=False)
for i, name in enumerate(["test left__001", "test__002"]):
    pixels = rng.integers(0, 100, (159 + 16*i, 203 - 24*i, 3), dtype=np.uint8)
    Image.fromarray(pixels).save(data_root / "test" / f"{name}.PNG")
sample = pd.DataFrame({"ImageID": ["test__002.PNG_Fovea_Y", "test left__001.PNG_Fovea_X",
                                   "test__002.PNG_Fovea_X", "test left__001.PNG_Fovea_Y"],
                       "value": [0, 0, 0, 0]})
sample.to_csv(data_root / "sample_submission.csv", index=False)

cfg = module.Config(
    INPUT_ROOT=str(test_root / "input"),
    OUTPUT_DIR=str(test_root / "run"),
    SUBMISSION_PATH=str(test_root / "submission.csv"),
    N_FOLDS=2, INPUT_SIZE=128, HEATMAP_SIZE=32, BATCH_SIZE=2,
    NUM_WORKERS=0, EPOCHS=2, PATIENCE=2, PRETRAINED=False,
    ALLOW_WEIGHT_DOWNLOAD=False, SHOW_PLOTS=False, AMP=False,
)
result = module.run_pipeline(cfg)
assert len(result["oof"]) == 10
assert set(result["oof"].fold) == {0, 1}
assert result["submission"].ImageID.tolist() == sample.ImageID.tolist()
assert len(result["submission"]) == 4
assert np.isfinite(result["submission"].value).all()
for group, frame in result["train"].groupby("group"):
    assert frame.fold.nunique() == 1
truth = result["oof"][["x_true", "y_true"]].to_numpy()
pred = module.apply_visibility(result["oof"], 0.5)
rmse = float(np.sqrt(np.mean((pred - truth)**2)))
assert np.isclose(result["scores"]["oof_rmse_fixed_threshold"], rmse)

cfg.REUSE_CHECKPOINTS = True
resumed = module.run_pipeline(cfg)
np.testing.assert_allclose(resumed["submission"].value, result["submission"].value, atol=1e-6)

# Dataset negative-path checks.
index = module.build_image_index([data_root / "a/same.png", data_root / "b/same.png"])
try:
    module.resolve_image(index, "same.png", "train")
    raise AssertionError("Ambiguous image names must fail.")
except ValueError:
    pass
try:
    module.parse_submission_id("bad_without_an_axis")
    raise AssertionError("Invalid sample suffix must fail.")
except ValueError:
    pass

report = {
    "status": "PASS",
    "synthetic_train_images": 10, "synthetic_test_images": 2,
    "folds": 2, "epochs_per_fold": 2, "actual_model": "ResNet18 + U-Net decoder",
    "tests": ["actual training", "both visibility classes", "grouped CV", "best checkpoint reload",
              "horizontal TTA", "long-format sample ordering", "RMSE recomputation",
              "completed-fold reuse equivalence", "ambiguous filename rejection", "invalid ID rejection"],
    "real_competition_training": False,
}
(test_root / "acceptance_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print("END_TO_END_ACCEPTANCE_PASS")
