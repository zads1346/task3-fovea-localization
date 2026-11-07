"""Fixed inner-development comparisons of frozen round snapshots.

No calibration or outer-validation accuracy is computed. Short-budget scores
are diagnostic only and must not determine release defaults or claim benefit.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
TESTS = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "work/kaggle_test_deps"))
os.environ["MPLBACKEND"] = "Agg"
import pandas as pd
import torch
torch.set_num_threads(2)


def load_round(snapshot):
    source = TESTS / "v2_baseline_source.py"
    name = "diagnostic_" + snapshot.name
    spec = importlib.util.spec_from_file_location(name, source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    hashes = {"v2_base": hashlib.sha256(source.read_bytes()).hexdigest()}
    for filename in ("research_config_part.py", "research_model_part.py"):
        path = snapshot / filename
        text = path.read_text(encoding="utf-8")
        exec(compile(text, str(path), "exec"), module.__dict__, module.__dict__)
        hashes[filename] = hashlib.sha256(path.read_bytes()).hexdigest()
    return module, hashes


def configuration(m):
    return m.ResearchConfig(
        DATA_DIR=str(ROOT / "work/real_competition_data/detection"),
        INPUT_ROOT=str(ROOT / "work/real_competition_data"),
        PRETRAINED_WEIGHTS_PATH=str(ROOT / "outputs/fovea_pretrained_weights/resnet18-f37072fd.pth"),
        REQUIRE_GPU=False, NUM_WORKERS=0, BATCH_SIZE=4, AMP=False,
        INPUT_SIZE=128, HEATMAP_SIZE=32, FINE_INPUT_SIZE=128, FINE_HEATMAP_SIZE=32,
        COARSE_EPOCHS=6, FINE_EPOCHS=6, PATIENCE=6, TTA=True, SEED=2026,
    )


def fixed_split(m, cfg):
    m.resolve_research_weights(cfg)
    train, *_ = m.prepare_tables(cfg)
    roles = pd.read_csv(TESTS / "baseline_development/split_roles.csv", dtype=str)
    assert roles.image_id.is_unique
    fit_ids = roles.loc[roles.role == "model_fit", "image_id"].tolist()
    stop_ids = roles.loc[roles.role == "early_stop", "image_id"].tolist()
    fitting = train.set_index("image_id").loc[fit_ids].reset_index()
    stopping = train.set_index("image_id").loc[stop_ids].reset_index()
    assert not set(fitting.group) & set(stopping.group)
    return fitting, stopping, roles


def run_one(snapshot, name, overrides):
    m, hashes = load_round(snapshot)
    cfg = configuration(m)
    for key, value in overrides.items():
        if not hasattr(cfg, key):
            raise ValueError(f"Unknown configuration field: {key}")
        setattr(cfg, key, value)
    out = TESTS / (snapshot.name + "_development") / name
    out.mkdir(parents=True, exist_ok=True)
    cfg.OUTPUT_DIR = str(out)
    started = time.perf_counter()
    fitting, stopping, roles = fixed_split(m, cfg)
    roles.to_csv(out / "split_roles.csv", index=False)
    model, info = m.fit_research_model(fitting, stopping, cfg, "cpu", "coarse", out / "model.pt", seed=2026)
    prediction = m.predict_research(model, stopping, cfg, "cpu", "coarse")
    metrics = m.localization_metrics(prediction, stopping)
    prediction.to_csv(out / "early_stop_tta_predictions.csv", index=False)
    pd.DataFrame(info["history"]).to_csv(out / "history.csv", index=False)
    report = {
        "status": "PASS", "round_snapshot": snapshot.name, "variant": name,
        "protocol": "Fixed low-resolution inner-development diagnostic only; no parameter selection or outer-fold accuracy claim",
        "source_hashes": hashes, "overrides": overrides,
        "fit_ids": fitting.image_id.tolist(), "early_stop_ids": stopping.image_id.tolist(),
        "n_fit": len(fitting), "n_stop": len(stopping),
        "outer_validation_evaluated": False, "calibration_evaluated": False,
        "release_defaults_selected_from_results": False,
        "seed": 2026, "input_size": 128, "heatmap_size": 32, "batch_size": 4,
        "max_epochs": 6, "best_epoch": info["best_epoch"],
        "early_stop_selection_rmse": info["best_rmse"], "early_stop_tta_metrics": metrics,
        "history": info["history"], "elapsed_seconds": time.perf_counter()-started,
        "accuracy_scope": "Small reused early-stop set at low resolution; cannot establish generalization benefit or Kaggle performance.",
    }
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in ("history", "fit_ids", "early_stop_ids")}, indent=2), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("snapshot")
    parser.add_argument("--name", default="default")
    parser.add_argument("--overrides", default="{}")
    args = parser.parse_args()
    run_one(TESTS / args.snapshot, args.name, json.loads(args.overrides))
