from common import ROOT, load_module, small_config, save_round
import numpy as np
import pandas as pd
import torch

m = load_module()
cfg = small_config(m, OOF_CENTER_JITTER=0.0)
train, *_ = m.prepare_tables(cfg)
fit, validation = train.iloc[:4].copy(), train.iloc[4:6].copy()
def crops(frame):
    pred = pd.DataFrame({"image_id": frame.image_id, "x": frame.x + 30,
                         "y": frame.y - 20, "uncertainty": 20.0})
    return m.make_roi_spec(pred, frame, cfg, mode="fixed")
fit_crop, val_crop = crops(fit), crops(validation)
one, one_crop = fit.iloc[:1], fit_crop.iloc[:1]
dataset = m.ResearchDataset(one, cfg, True, "fine", one_crop)
# Cache one real image for repeated geometric sampling, leaving production IO unchanged.
original_reader = m.read_rgb
cached = original_reader(one.path.iloc[0])
m.read_rgb = lambda path: cached.copy()
sampled, inverse_errors = set(), []
m.seed_everything(72)
for _ in range(60):
    item = dataset[0]
    fraction = round(float(item["crop_sampled_fraction"]), 4)
    sampled.add(fraction)
    expected = m.make_roi_spec(pd.DataFrame({"image_id": one.image_id,
                                            "x": one_crop.crop_x, "y": one_crop.crop_y}),
                               one, cfg, mode="fixed", fixed_fraction=fraction).crop_side.iloc[0]
    assert abs(float(item["crop_sampled_side"]) - expected) < 1e-3
    np.testing.assert_allclose(item["crop"][:2], one_crop[["crop_x", "crop_y"]].iloc[0], atol=1e-4)
    recovered = item["input_to_orig"].numpy() @ np.r_[item["target_xy"].numpy() * (cfg.FINE_INPUT_SIZE - 1), 1]
    error = float(np.abs(recovered - one[["x", "y"]].iloc[0].to_numpy()).max())
    inverse_errors.append(error)
    assert error < 1e-3
m.read_rgb = original_reader
assert sampled == set(cfg.ROI_FRACTIONS)
# Switch-off preserves original supplied training side; inference ignores this switch.
single_cfg = m.replace(cfg, FINE_MULTISCALE_TRAIN=False, FINE_MULTISCALE_EARLY_STOP=False)
sample = m.ResearchDataset(one, single_cfg, True, "fine", one_crop)[0]
assert abs(float(sample["crop_sampled_side"]) - one_crop.crop_side.iloc[0]) < 1e-3
assert len(m.research_validation_loaders(validation, single_cfg, "fine", val_crop, 72)) == 1
loaders = m.research_validation_loaders(validation, cfg, "fine", val_crop, 72)
assert len(loaders) == len(cfg.FINE_VALIDATION_ROI_FRACTIONS)
assert all(set(loader.dataset.frame.image_id) == set(validation.image_id) for _, loader in loaders)
out = ROOT / "work/five_round_tests/round_02"
out.mkdir(parents=True, exist_ok=True)
model, info = m.fit_research_model(fit, validation, cfg, "cpu", "fine", out / "fine.pt",
                                   train_crop=fit_crop, val_crop=val_crop, seed=72)
scores = {name: m.research_validation_rmse(model, loader, cfg, torch.device("cpu")) for name, loader in loaders}
assert abs(np.mean(list(scores.values())) - info["best_rmse"]) < 1e-9
prediction = m.predict_research(model, validation, cfg, "cpu", "fine", val_crop)
poison = validation.copy()
poison["x"], poison["y"], poison["visible"] = "not-a-coordinate", np.nan, False
other = m.predict_research(model, poison, single_cfg, "cpu", "fine", val_crop)
fields = ["x", "y", "uncertainty", "cov_xx", "cov_xy", "cov_yy", "crop_edge_fraction"]
np.testing.assert_allclose(prediction[fields], other[fields], atol=0, rtol=0)
assert all("crop_sampled_fraction_counts" in row and "crop_enlargement_rate" in row for row in info["history"])
save_round(2, {"changes": ["multiscale OOF-centred training with common margin and clamp", "prespecified inner stopping scale-mean RMSE"],
               "sampled_fractions": sorted(sampled), "geometry_trials": len(inverse_errors),
               "maximum_inverse_error_px": max(inverse_errors), "label_poisoning_inference_equal": True,
               "train_scale_toggle_does_not_change_inference": True, "stopping_scores_recomputed": scores,
               "all_stopping_loaders_use_supplied_inner_ids_only": True, "training": info})
