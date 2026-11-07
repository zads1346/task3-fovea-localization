"""Full-resolution functional checks; no claims about trained accuracy."""
from pathlib import Path
import gc
import importlib.util
import json
import os
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "work/kaggle_test_deps"))
os.environ["MPLBACKEND"] = "Agg"
import numpy as np
import pandas as pd
import torch
torch.set_num_threads(1)
spec = importlib.util.spec_from_file_location("research_fullsize", ROOT / "outputs/fovea_research_kaggle.py")
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)
cfg = m.ResearchConfig(
    DATA_DIR=str(ROOT / "work/real_competition_data/detection"),
    PRETRAINED_WEIGHTS_PATH=str(ROOT / "outputs/fovea_pretrained_weights/resnet18-f37072fd.pth"),
    NUM_WORKERS=0, BATCH_SIZE=1, AMP=False, TTA=True,
    INPUT_SIZE=512, HEATMAP_SIZE=128, FINE_INPUT_SIZE=384, FINE_HEATMAP_SIZE=96,
)
train, *_ = m.prepare_tables(cfg)
frame = train.iloc[:1].copy()
report = {"status": "PASS", "device": "cpu", "torch_threads": 1,
          "torch_version": torch.__version__, "pretrained": True,
          "meaning": "Full-size forward/backward and TTA inference functional checks, not training-performance evaluation",
          "stages": {}, "geometry": m.research_geometry_selfcheck()}
for stage in ["coarse", "fine"]:
    started = time.perf_counter()
    m.seed_everything(2026)
    model = m.research_build_model(cfg, stage, "cpu")
    loader = m.make_research_loader(frame, cfg, True, stage, seed=2026)
    batch = next(iter(loader))
    model.train()
    output = model(batch["image"])
    loss, components = m.fovea_loss(output, batch, cfg, use_visibility=False)
    assert torch.isfinite(loss)
    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    assert float(model.heatmap_head.weight.grad.abs().sum()) > 0
    crop = None
    if stage == "fine":
        # Prediction-like shifted centre, not target-centred evaluation crops.
        centre = pd.DataFrame({"image_id": frame.image_id, "x": frame.x + 25,
                               "y": frame.y - 20, "uncertainty": 20.0})
        crop = m.make_roi_spec(centre, frame, cfg, mode="fixed")
    pred = m.predict_research(model, frame, cfg, "cpu", stage, crop)
    assert np.isfinite(pred[["x", "y", "uncertainty", "cov_xx", "cov_xy", "cov_yy"]]).all().all()
    covariance = np.array([[pred.cov_xx.iloc[0], pred.cov_xy.iloc[0]],
                           [pred.cov_xy.iloc[0], pred.cov_yy.iloc[0]]])
    eigvals = np.linalg.eigvalsh(covariance)
    assert eigvals.min() >= -1e-4
    if stage == "fine":
        assert pred.crop_edge_fraction.between(0, 0.5).all()
    report["stages"][stage] = {
        "input_shape": list(batch["image"].shape), "heatmap_shape": list(output["logits"].shape),
        "loss": float(loss.detach()), "loss_components": components,
        "gradient_tensors_checked": len(gradients), "all_gradients_finite": True,
        "tta": True, "prediction": pred.drop(columns=["crop_edge_fraction"] if stage == "coarse" else []).to_dict("records"),
        "covariance_eigenvalues": eigvals.tolist(), "seconds": time.perf_counter() - started,
    }
    del model, gradients, batch, output, loss, loader
    gc.collect()
    print(stage, "full-size PASS", flush=True)

# The calibration ROI builder requires every centre to lie within its image.
# Input-space expectations lie inside their ROI; clipping to the image moves
# them toward the in-image centre, never outside the crop. Test extreme corners.
minimum, maximum, cases = 1.0, 0.0, 0
for width, height in [(2992, 2000), (1956, 1934)]:
    for cx, cy in [(0, 0), (width - 1, 0), (0, height - 1), (width - 1, height - 1),
                   ((width - 1) / 2, (height - 1) / 2)]:
        for fraction in [0.15, 0.30, 0.60]:
            side = min(width, height) * fraction
            inverse = torch.tensor(np.linalg.inv(m.research_crop_transform(cx, cy, side, 384))[:2], dtype=torch.float32)[None]
            for iy, ix in [(0, 0), (0, 95), (95, 0), (95, 95), (47, 47)]:
                logits = torch.full((1, 1, 96, 96), -50.0)
                logits[0, 0, iy, ix] = 50.0
                mean, covariance = m.research_distribution_moments(logits, inverse, 384)
                mean = torch.minimum(mean.clamp_min(0), torch.tensor([[width - 1, height - 1]]))
                edge = float(0.5 - ((mean - torch.tensor([[cx, cy]])) / side).abs().amax(1))
                assert -1e-6 <= edge <= 0.5 + 1e-6, edge
                minimum, maximum, cases = min(minimum, edge), max(maximum, edge), cases + 1
report["crop_edge_boundary"] = {"cases": cases, "minimum": minimum, "maximum": maximum,
                                 "valid_range": [0, 0.5], "calibration_builder_rejects_outside_image_centres": True,
                                 "conclusion": "Compatible for all pipeline-generated ROIs, including padded corner crops"}
out = ROOT / "work/research_fullsize_check.json"
out.write_text(json.dumps(report, indent=2), encoding="utf-8")
print("PASS", out, flush=True)
