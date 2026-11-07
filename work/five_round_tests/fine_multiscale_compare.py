"""Fixed versus multiscale fine training with shared genuine OOF crop centers."""
import gc
import json
import time
import development_compare as helper
import pandas as pd
import numpy as np

TESTS = helper.TESTS
CROPS = TESTS / "development_oof_crops"
reports = []
for multiscale in (False, True):
    m, hashes = helper.load_round(TESTS / "round_02")
    cfg = helper.configuration(m)
    cfg.FINE_MULTISCALE_TRAIN = multiscale
    cfg.FINE_MULTISCALE_EARLY_STOP = True
    name = "multiscale_train" if multiscale else "fixed_train"
    out = TESTS / "round_02_development" / name
    out.mkdir(parents=True, exist_ok=True)
    cfg.OUTPUT_DIR = str(out)
    fitting, stopping, roles = helper.fixed_split(m, cfg)
    fitting_crop = pd.read_csv(CROPS / "fitting_oof_crop.csv", dtype={"image_id": str})
    stopping_crop = pd.read_csv(CROPS / "stopping_roi.csv", dtype={"image_id": str})
    roles.to_csv(out / "split_roles.csv", index=False)
    started = time.perf_counter()
    model, info = m.fit_research_model(fitting, stopping, cfg, "cpu", "fine", out / "model.pt",
                                     train_crop=fitting_crop, val_crop=stopping_crop, seed=2026)
    evaluations = []
    for fraction in cfg.FINE_VALIDATION_ROI_FRACTIONS:
        roi = stopping_crop.copy()
        short_by_id = stopping.set_index("image_id")[["width", "height"]].min(axis=1)
        roi["crop_side"] = m.research_roi_side(short_by_id.loc[roi.image_id].to_numpy(float), fraction, cfg)
        prediction = m.predict_research(model, stopping, cfg, "cpu", "fine", roi)
        prediction.to_csv(out / f"early_stop_prediction_{fraction:g}.csv", index=False)
        evaluations.append({"roi_fraction": fraction, **m.localization_metrics(prediction, stopping)})
    pd.DataFrame(info["history"]).to_csv(out / "history.csv", index=False)
    report = {
        "status": "PASS", "source_hashes": hashes, "variant": name,
        "fine_multiscale_train": multiscale, "same_multiscale_tta_validation_for_both": True,
        "validation_roi_fractions": list(cfg.FINE_VALIDATION_ROI_FRACTIONS),
        "train_roi_fractions": list(cfg.ROI_FRACTIONS),
        "best_epoch": info["best_epoch"], "early_stop_selection_rmse": info["best_rmse"],
        "early_stop_mean_scale_rmse": float(np.mean([r["rmse"] for r in evaluations])),
        "early_stop_scale_metrics": evaluations, "history": info["history"],
        "seed": 2026, "input_size": 128, "heatmap_size": 32, "epochs": 6,
        "n_fit": len(fitting), "n_stop": len(stopping), "fit_ids": fitting.image_id.tolist(),
        "early_stop_ids": stopping.image_id.tolist(),
        "crop_source": "R1 3-fold coarse crossfit within fitting only; held image absent from its crop model",
        "outer_validation_evaluated": False, "calibration_evaluated": False,
        "release_defaults_selected_from_results": False,
        "accuracy_scope": "Low-resolution 6-epoch inner-development diagnostic; insufficient to claim generalization benefit or choose defaults.",
        "seconds": time.perf_counter()-started,
    }
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    reports.append({k: v for k, v in report.items() if k not in ("history", "fit_ids", "early_stop_ids")})
    print(json.dumps(reports[-1], indent=2), flush=True)
    del model, m
    gc.collect()
(TESTS / "round_02_development/comparison.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")
