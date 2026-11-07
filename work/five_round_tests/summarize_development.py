from pathlib import Path
import json

root = Path(__file__).resolve().parent
files = [
    ("baseline", "coarse", root / "baseline_development/report.json"),
    ("R1", "coarse", root / "round_01_development/default/report.json"),
    ("R2-fixed", "fine", root / "round_02_development/fixed_train/report.json"),
    ("R2-multiscale", "fine", root / "round_02_development/multiscale_train/report.json"),
    ("R3-lr1-noEMA", "coarse", root / "round_03_development/lr_1_no_ema/report.json"),
    ("R3-lr025-noEMA", "coarse", root / "round_03_development/lr_025_no_ema/report.json"),
    ("R3-lr025-EMA", "coarse", root / "round_03_development/lr_025_ema/report.json"),
]
rows = []
for label, stage, path in files:
    d = json.loads(path.read_text())
    rows.append(dict(
        round=label, stage=stage, best_epoch=d["best_epoch"],
        metric="early_stop_TTA_RMSE" if stage == "coarse" else "early_stop_mean_three_scale_TTA_RMSE",
        rmse=d["early_stop_tta_metrics"]["rmse"] if stage == "coarse" else d["early_stop_mean_scale_rmse"],
        source=str(path)))
result = dict(
    status="COMPLETED_DIAGNOSTICS", same_fit_count=38, same_stop_count=10,
    input_size=128, heatmap_size=32, training_epochs_per_job=6,
    calibration_labels_evaluated=False, outer_validation_labels_evaluated=False,
    release_defaults_selected_from_results=False, rows=rows,
    limitations="Single seed, short budget, low resolution, reused early-stopping set; no generalization or Kaggle improvement claim. All requested R1-R3 development diagnostics completed serially; none evaluated calibration or outer validation labels.")
(root / "development_results_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result, indent=2))
