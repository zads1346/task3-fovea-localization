from common import ROOT, load_module, small_config, save_round
import gc
import numpy as np
import torch

m = load_module()
cfg = small_config(m, USE_EMA=True, BACKBONE_LR_MULTIPLIER=0.25)
assert m.ResearchConfig().USE_EMA is False
model = m.research_build_model(cfg, "coarse", "cpu")
groups = m.research_optimizer_groups(model, cfg)
ids = [id(p) for group in groups for p in group["params"]]
assert len(ids) == len(set(ids))
assert set(ids) == {id(p) for p in model.parameters() if p.requires_grad}
optimizer = torch.optim.AdamW(groups, lr=cfg.LR)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: m.research_schedule_factor(s, 2, 5, 1))
rate_samples = []
for step in range(11):
    rates = {g["name"]: g["lr"] for g in optimizer.param_groups}
    if rates["decoder"] > 0:
        assert abs(rates["encoder"] / rates["decoder"] - 0.25) < 1e-12
    else:
        assert rates["encoder"] == 0
    rate_samples.append(rates)
    optimizer.step()
    scheduler.step()
initial_buffers = {name: value.clone() for name, value in model.named_buffers()}
del model, optimizer, scheduler, groups
gc.collect()

# Hand-calculated EMA, including integer buffers and skipped optimizer updates.
tiny = torch.nn.Sequential(torch.nn.Linear(1, 1, bias=False), torch.nn.BatchNorm1d(1))
ema = m.ResearchEMA(tiny, decay=0.99, warmup_updates=10)
with torch.no_grad():
    tiny[0].weight.fill_(2)
    tiny[1].running_mean.fill_(7)
    tiny[1].num_batches_tracked.fill_(3)
assert ema.update(tiny)
assert float(ema.model[0].weight) == 2.0
with torch.no_grad():
    tiny[0].weight.fill_(4)
    tiny[1].running_mean.fill_(9)
    tiny[1].num_batches_tracked.fill_(5)
assert ema.update(tiny)
assert ema.last_decay == 0.25
assert float(ema.model[0].weight) == 3.5
assert float(ema.model[1].running_mean) == 9
assert int(ema.model[1].num_batches_tracked) == 5
before = {k: v.clone() for k, v in ema.model.state_dict().items()}
with torch.no_grad():
    tiny[0].weight.fill_(100)
    tiny[1].running_mean.fill_(100)
assert not ema.update(tiny, optimizer_stepped=False)
assert ema.updates == 2
for key, value in ema.model.state_dict().items():
    torch.testing.assert_close(value, before[key], rtol=0, atol=0)
assert not any(p.requires_grad for p in ema.model.parameters())

train, *_ = m.prepare_tables(cfg)
fit, validation = train.iloc[:4].copy(), train.iloc[4:6].copy()
out = ROOT / "work/five_round_tests/round_03"
out.mkdir(parents=True, exist_ok=True)
model, info = m.fit_research_model(fit, validation, cfg, "cpu", "coarse", out / "ema.pt", seed=73)
assert info["ema_updates"] == info["successful_optimizer_steps"] > 0
assert info["weights_kind"] == "ema"
checkpoint = m._load_torch_file(out / "ema.pt")
assert checkpoint["weights_kind"] == "ema"
assert not any(name.startswith("module.") for name in checkpoint["model"])
for name, value in model.state_dict().items():
    torch.testing.assert_close(value, checkpoint["model"][name], rtol=0, atol=0)
for name, value in model.named_buffers():
    torch.testing.assert_close(value, initial_buffers[name], rtol=0, atol=0)
score = m.research_validation_rmse(model, m.make_research_loader(validation, cfg, False, "coarse"), cfg, torch.device("cpu"))
assert abs(score - info["best_rmse"]) < 1e-9
prediction = m.predict_research(model, validation, cfg, "cpu")
loaded = m.research_build_model(cfg, "coarse", "cpu")
m.load_model_checkpoint(loaded, out / "ema.pt", "cpu")
reloaded_prediction = m.predict_research(loaded, validation, cfg, "cpu")
fields = ["x", "y", "uncertainty", "cov_xx", "cov_xy", "cov_yy"]
np.testing.assert_allclose(prediction[fields], reloaded_prediction[fields], atol=0, rtol=0)
for row in info["history"]:
    weighted = (cfg.HEATMAP_LOSS_WEIGHT * row["train_heatmap_kl"] +
                cfg.COORD_LOSS_WEIGHT * row["train_coordinate_mse"] +
                cfg.VIS_LOSS_WEIGHT * row["train_visibility_bce"])
    assert abs(weighted - row["train_loss"]) < 1e-5
class ValidationBomb:
    def __len__(self):
        raise AssertionError("EMA fixed refit accessed validation")
refit, fixed = m.fit_research_model(fit, ValidationBomb(), cfg, "cpu", "coarse", out / "ema_fixed.pt", epochs=1, seed=73)
assert fixed["weights_kind"] == "ema" and fixed["best_rmse"] is None
save_round(3, {"changes": ["encoder/head differential LR", "optional dynamic EMA", "heatmap and coordinate component logging"],
               "optimizer_tensor_coverage": len(ids), "optimizer_groups_disjoint_and_complete": True,
               "group_lr_samples": rate_samples,
               "ema_hand_calculation": {"first_weight": 2.0, "second_weight": 3.5, "second_decay": 0.25,
                                        "skipped_update_unchanged": True, "buffers_copied": True},
               "ema_disabled_by_default": True, "checkpoint_prediction_roundtrip_exact": True,
               "frozen_BN_buffers_unchanged": True, "ema_validation_matches_saved_checkpoint": True,
               "loss_component_reconstruction": True, "training": info, "fixed_epoch_no_validation": fixed})
