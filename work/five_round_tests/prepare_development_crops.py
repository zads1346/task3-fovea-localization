"""Create genuine held-image-out training crop centers for fine A/B diagnostics."""
from pathlib import Path
import json
import time
import development_compare as helper
import pandas as pd

ROOT, TESTS = helper.ROOT, helper.TESTS
OUT = TESTS / "development_oof_crops"
OUT.mkdir(parents=True, exist_ok=True)
m, hashes = helper.load_round(TESTS / "round_01")
cfg = helper.configuration(m)
cfg.OUTPUT_DIR = str(OUT)
fitting, stopping, roles = helper.fixed_split(m, cfg)
selection = json.loads((TESTS / "round_01_development/default/report.json").read_text())
epochs = int(selection["best_epoch"])
started = time.perf_counter()
crop = m.research_crossfit_crops(fitting, cfg, "cpu", OUT / "crossfit", epochs, 2126)
crop.to_csv(OUT / "fitting_oof_crop.csv", index=False)
model = m.research_build_model(cfg, "coarse", "cpu")
m.load_model_checkpoint(model, TESTS / "round_01_development/default/model.pt", "cpu")
stop_prediction = m.predict_research(model, stopping, cfg, "cpu", "coarse")
stop_prediction.to_csv(OUT / "stopping_coarse_prediction.csv", index=False)
stop_roi = m.make_roi_spec(stop_prediction, stopping, cfg, mode="fixed")
stop_roi.to_csv(OUT / "stopping_roi.csv", index=False)
audits = json.loads((OUT / "crossfit/crossfit_audit.json").read_text())
for audit in audits:
    assert not set(audit["fit_groups"]) & set(audit["held_groups"])
    assert not set(audit["fit_ids"]) & set(stopping.image_id)
assert set(crop.image_id) == set(fitting.image_id)
report = {
    "status": "PASS", "source_hashes": hashes, "coarse_epochs": epochs,
    "n_fitting_crops": len(crop), "n_stopping_crops": len(stop_roi), "inner_folds": len(audits),
    "held_image_never_trained_in_its_crop_model": True,
    "early_stop_groups_never_train_crop_models": True,
    "calibration_or_outer_evaluation": False,
    "coarse_training_budget_selected_only_on_development_early_stop": True,
    "seconds": time.perf_counter()-started,
    "purpose": "Shared fixed crop inputs for multiscale fine-training A/B; no generalization claim",
}
(OUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
