"""Fixed, small inner-development training diagnostic; no outer evaluation."""
from pathlib import Path
import hashlib
import importlib.util
import json
import os
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "baseline_development"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "work/kaggle_test_deps"))
os.environ["MPLBACKEND"] = "Agg"
import pandas as pd
import torch
torch.set_num_threads(2)
source = Path(__file__).resolve().parent / "v2_baseline_source.py"
if not source.exists():
    source.write_bytes((ROOT / "outputs/fovea_research_kaggle.py").read_bytes())
spec = importlib.util.spec_from_file_location("v2_baseline_development", source)
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)
cfg = m.ResearchConfig(
    DATA_DIR=str(ROOT / "work/real_competition_data/detection"),
    INPUT_ROOT=str(ROOT / "work/real_competition_data"),
    PRETRAINED_WEIGHTS_PATH=str(ROOT / "outputs/fovea_pretrained_weights/resnet18-f37072fd.pth"),
    OUTPUT_DIR=str(OUT), REQUIRE_GPU=False,
    NUM_WORKERS=0, BATCH_SIZE=4, AMP=False, INPUT_SIZE=128, HEATMAP_SIZE=32,
    COARSE_EPOCHS=6, PATIENCE=6, TTA=True, SEED=2026,
)
started = time.perf_counter()
m.resolve_research_weights(cfg)
train, *_ = m.prepare_tables(cfg)
train = m.assign_folds(train, cfg)
outer_train = train.loc[train.fold != 0].reset_index(drop=True)
development, calibration = m.research_group_holdout(outer_train, cfg.CALIBRATION_FRACTION, 2026)
fitting, stopping = m.research_group_holdout(development, cfg.EARLY_STOP_FRACTION, 2027)
assert not set(fitting.group) & set(stopping.group)
assert not set(development.group) & set(calibration.group)
assert not set(development.group) & set(train.loc[train.fold == 0, "group"])
assignments = pd.concat([fitting[["image_id", "group"]].assign(role="model_fit"),
                         stopping[["image_id", "group"]].assign(role="early_stop"),
                         calibration[["image_id", "group"]].assign(role="calibration_excluded"),
                         train.loc[train.fold == 0, ["image_id", "group"]].assign(role="outer_excluded")])
assignments.to_csv(OUT / "split_roles.csv", index=False)
model, info = m.fit_research_model(fitting, stopping, cfg, "cpu", "coarse", OUT / "model.pt", seed=2026)
prediction = m.predict_research(model, stopping, cfg, "cpu", "coarse")
metric = m.localization_metrics(prediction, stopping)
prediction.to_csv(OUT / "early_stop_tta_predictions.csv", index=False)
pd.DataFrame(info["history"]).to_csv(OUT / "history.csv", index=False)
report = {
    "status": "PASS", "protocol": "Fixed development diagnostic only, no parameter selection or outer-fold accuracy claim",
    "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    "fold_reserved": 0, "train_ids": fitting.image_id.tolist(), "early_stop_ids": stopping.image_id.tolist(),
    "n_fit": len(fitting), "n_stop": len(stopping), "n_calibration_excluded": len(calibration),
    "outer_validation_evaluated": False, "calibration_evaluated": False,
    "seed": 2026, "input_size": 128, "heatmap_size": 32, "batch_size": 4,
    "max_epochs": 6, "schedule_horizon_epochs": 6, "best_epoch": info["best_epoch"],
    "best_early_stop_no_tta_rmse": info["best_rmse"], "early_stop_tta_metrics": metric,
    "history": info["history"], "device": "cpu", "elapsed_seconds": time.perf_counter()-started,
    "accuracy_scope": "Small reused early-stop set at low resolution; cannot establish Kaggle accuracy or unbiased performance.",
}
(OUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
