from common import ROOT, load_module, small_config, save_round
import numpy as np
import torch

m = load_module()
cfg = small_config(m)
assert m.RESEARCH_VERSION == "3.0.0"
assert cfg.OUTPUT_DIR == "/kaggle/working/fovea_optimized"
# A selected 2-epoch refit must retain the prefix of the 6-epoch search schedule.
full = [m.research_schedule_factor(s, 4, 6, 1) for s in range(25)]
prefix = [m.research_schedule_factor(s, 4, 6, 1) for s in range(8)]
assert prefix == full[:8]
assert full[0] == 0.25 and full[3] == 1 and full[-1] == 0
assert full[8] > 0.8  # A truncated horizon would have already decayed to zero.
train, *_ = m.prepare_tables(cfg)
fit, validation = train.iloc[:4].copy(), train.iloc[4:6].copy()
out = ROOT / "work/five_round_tests/round_01"
out.mkdir(parents=True, exist_ok=True)
model, selected = m.fit_research_model(fit, validation, cfg, "cpu", "coarse", out / "selection.pt", seed=71)
comparisons = []
for tta in [False, True]:
    check_cfg = m.replace(cfg, TTA=tta)
    loader = m.make_research_loader(validation, check_cfg, False, "coarse", seed=71)
    score = m.research_validation_rmse(model, loader, check_cfg, torch.device("cpu"))
    pred = m.predict_research(model, validation, check_cfg, "cpu")
    truth = validation[["x", "y"]].to_numpy(np.float32).astype(float)
    expected = float(np.sqrt(np.mean((pred[["x", "y"]].to_numpy() - truth) ** 2)))
    assert abs(score - expected) < 1e-9
    comparisons.append({"tta": tta, "validation_rmse": score, "prediction_rmse": expected})
class ValidationBomb:
    def __len__(self):
        raise AssertionError("Fixed refit accessed validation")
refit, fixed = m.fit_research_model(fit, ValidationBomb(), cfg, "cpu", "coarse",
                                    out / "refit.pt", epochs=1, seed=71)
assert fixed["schedule_horizon_epochs"] == 2
assert fixed["requested_training_epochs"] == 1
assert selected["history"][0]["learning_rate"] == fixed["history"][0]["learning_rate"]
assert fixed["history"][0]["learning_rate"] > 0
save_round(1, {"changes": ["stage-budget LR horizon retained during shorter refit", "same TTA distribution decoder for early stopping and inference"],
               "schedule_first_eight_factors": prefix, "validation_matches_prediction": comparisons,
               "selection": selected, "refit": fixed, "geometry": m.research_geometry_selfcheck()})
