"""Regression: numeric CSV IDs, padded/prefixed filenames, full two-fold pipeline."""
from pathlib import Path
import sys
import os
import importlib.util
import shutil

ROOT = Path(__file__).resolve().parent.parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "work/kaggle_test_deps"))
os.environ["MPLBACKEND"] = "Agg"
import pandas as pd
import numpy as np
import torch
torch.set_num_threads(2)

spec = importlib.util.spec_from_file_location("numeric_pipeline", ROOT / "outputs/fovea_kaggle.py")
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)

fixture = ROOT / "work/synthetic_acceptance/input/detection"
root = ROOT / "work/numeric_id_acceptance"
data = root / "input/detection"
for folder in ("train", "test", "train_location"):
    (data / folder).mkdir(parents=True, exist_ok=True)
labels = pd.read_csv(fixture / "fovea_localization_train_GT.csv")
for i, row in enumerate(labels.itertuples(), start=1):
    name = f"retina_{i:03d}.png"
    shutil.copyfile(fixture / "train" / row.data, data / "train" / name)
    xml_path = fixture / "train_location" / Path(row.data).with_suffix(".xml").name
    if xml_path.exists():
        content = xml_path.read_text().replace(row.data, name)
        (data / "train_location" / f"retina_{i:03d}.xml").write_text(content)
labels["data"] = np.arange(1, len(labels) + 1)
labels.to_csv(data / "fovea_localization_train_GT.csv", index=False)
for source, number in [("test left__001.PNG", 81), ("test__002.PNG", 82)]:
    shutil.copyfile(fixture / "test" / source, data / "test" / f"retina_{number:03d}.png")
sample = pd.DataFrame({"ImageID": ["82_Fovea_Y", "81_Fovea_X", "82_Fovea_X", "81_Fovea_Y"],
                       "value": [0, 0, 0, 0]})
sample.to_csv(data / "sample_submission.csv", index=False)
cfg = m.Config(INPUT_ROOT=str(root / "input"), OUTPUT_DIR=str(root / "run"),
               SUBMISSION_PATH=str(root / "submission.csv"), N_FOLDS=2, EPOCHS=1,
               INPUT_SIZE=64, HEATMAP_SIZE=16, BATCH_SIZE=2, NUM_WORKERS=0,
               PRETRAINED=False, ALLOW_WEIGHT_DOWNLOAD=False, AMP=False, SHOW_PLOTS=False)
result = m.run_pipeline(cfg)
assert result["train"].iloc[0].source_id == "1"
assert Path(result["train"].iloc[0].path).name == "retina_001.png"
assert result["submission"].ImageID.tolist() == sample.ImageID.tolist()
assert np.isfinite(result["submission"].value).all()
assert len(pd.read_csv(root / "run/xml_box_audit.csv")) == 8
mapping = pd.read_csv(root / "run/filename_mapping.csv")
assert len(mapping) == 12
assert set(result["oof"].fold) == {0, 1}
print("NUMERIC_ID_END_TO_END_PASS: 1 -> retina_001.png; 2-fold training, XML audit and submission verified.")
