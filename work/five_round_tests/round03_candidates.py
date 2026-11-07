"""Three preregistered coarse candidates on the same inner-development split."""
import gc
import json
import development_compare as helper

variants = [
    ("lr_1_no_ema", {"BACKBONE_LR_MULTIPLIER": 1.0, "USE_EMA": False}),
    ("lr_025_no_ema", {"BACKBONE_LR_MULTIPLIER": 0.25, "USE_EMA": False}),
    ("lr_025_ema", {"BACKBONE_LR_MULTIPLIER": 0.25, "USE_EMA": True}),
]
reports = []
for name, overrides in variants:
    report = helper.run_one(helper.TESTS / "round_03", name, overrides)
    reports.append({k: v for k, v in report.items() if k not in ("history", "fit_ids", "early_stop_ids")})
    gc.collect()
(helper.TESTS / "round_03_development/comparison.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")
