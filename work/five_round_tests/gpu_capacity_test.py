"""Local GPU compatibility/VRAM check, no fitted accuracy claim."""
from pathlib import Path
import gc
import importlib.util
import json
import os
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "work/gpu_test_deps"))
os.environ["MPLBACKEND"] = "Agg"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import numpy as np
import pandas as pd
import torch
import torchvision
import cv2
import sklearn
import nbformat
torch.set_num_threads(2)
torch.backends.cudnn.enabled = os.environ.get("GPUCHECK_CUDNN", "1") == "1"
assert torch.cuda.is_available()
assert Path(np.__file__).resolve().is_relative_to(Path(sys.prefix))
assert Path(torch.__file__).resolve().is_relative_to(Path(sys.prefix))
nbformat.validate(nbformat.read(ROOT / "outputs/fovea_research_kaggle.ipynb", as_version=4))
spec = importlib.util.spec_from_file_location("gpu_capacity_fovea", ROOT / "outputs/fovea_research_kaggle.py")
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)
cfg = m.ResearchConfig(
    DATA_DIR=str(ROOT / "work/real_competition_data/detection"),
    INPUT_ROOT=str(ROOT / "work/real_competition_data"),
    PRETRAINED_WEIGHTS_PATH=str(ROOT / "outputs/fovea_pretrained_weights/resnet18-f37072fd.pth"),
    OUTPUT_DIR=str(OUT / "gpu_capacity_outputs"), NUM_WORKERS=0,
    BATCH_SIZE=int(os.environ.get("GPUCHECK_BATCH", "4")), AMP=os.environ.get("GPUCHECK_AMP", "1") == "1",
    INPUT_SIZE=int(os.environ.get("GPUCHECK_SIZE", "256")),
    HEATMAP_SIZE=int(os.environ.get("GPUCHECK_SIZE", "256")) // 4,
    FINE_INPUT_SIZE=int(os.environ.get("GPUCHECK_SIZE", "256")),
    FINE_HEATMAP_SIZE=int(os.environ.get("GPUCHECK_SIZE", "256")) // 4,
)
m.resolve_research_weights(cfg)
train, *_ = m.prepare_tables(cfg)
frame = train.iloc[:cfg.BATCH_SIZE].copy()
report = {
    "status": "PASS", "python_executable": sys.executable, "python": sys.version,
    "device": torch.cuda.get_device_name(0), "device_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
    "versions": {"torch": str(torch.__version__), "torchvision": str(torchvision.__version__),
                 "numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__, "opencv": cv2.__version__},
    "extra_dependencies": str(ROOT / "work/gpu_test_deps"),
    "existing_numpy_torch_torchvision_preserved": True,
    "nbformat_validation": "PASS", "stages": [],
    "accuracy_claim": "None. One batch update and inference only; no full training."
}
for stage in ("coarse", "fine"):
    m.seed_everything(2026)
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    model = m.research_build_model(cfg, stage, "cuda")
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, foreach=False)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.AMP)
    cpu_batch = next(iter(m.make_research_loader(frame, cfg, True, stage, seed=2026)))
    batch = m._to_device(cpu_batch, "cuda")
    with m._autocast_context(torch.device("cuda"), cfg):
        output = model(batch["image"])
    loss, _ = m.fovea_loss(output, batch, cfg, False)
    print({"before_backward_free_total": torch.cuda.mem_get_info(),
           "allocated": torch.cuda.memory_allocated(), "stage": stage}, flush=True)
    assert torch.isfinite(loss)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
    scaler.step(optimizer)
    scaler.update()
    torch.cuda.synchronize()
    crop = None
    if stage == "fine":
        centres = pd.DataFrame({"image_id": frame.image_id, "x": frame.x + 20,
                                "y": frame.y - 20, "uncertainty": 20.})
        crop = m.make_roi_spec(centres, frame, cfg, mode="fixed")
    prediction = m.predict_research(model, frame, cfg, "cuda", stage, crop)
    assert np.isfinite(prediction[["x", "y", "uncertainty"]]).all().all()
    entry = {"stage": stage, "input_shape": list(batch["image"].shape),
             "amp": cfg.AMP, "loss": float(loss.detach()), "all_gradients_finite": True,
             "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
             "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
             "seconds": time.perf_counter()-started, "tta_inference": "PASS"}
    report["stages"].append(entry)
    print(entry, flush=True)
    del model, optimizer, scaler, cpu_batch, batch, output, loss, grads, prediction
    gc.collect()
    torch.cuda.empty_cache()
(OUT / "gpu_capacity_test.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
