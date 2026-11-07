"""Small real-image acceptance checks for separately snapshotted optimization rounds."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import sys
import types

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "work/kaggle_test_deps"))
os.environ["MPLBACKEND"] = "Agg"
import torch
torch.set_num_threads(1)


def load_module():
    module = types.ModuleType("fovea_v3_checks")
    sys.modules[module.__name__] = module
    parts = ["work/data_part.py", "work/model_training_part.py",
             "work/fovea_v3/research_config_part.py", "work/fovea_v3/research_model_part.py",
             "work/research_calibration_part.py"]
    for relative in parts:
        path = ROOT / relative
        exec(compile(path.read_text(encoding="utf-8-sig"), str(path), "exec"), module.__dict__)
    return module


def small_config(m, **kwargs):
    settings = dict(DATA_DIR=str(ROOT / "work/real_competition_data/detection"),
                    PRETRAINED_WEIGHTS_PATH=str(ROOT / "outputs/fovea_pretrained_weights/resnet18-f37072fd.pth"),
                    INPUT_SIZE=64, HEATMAP_SIZE=16, FINE_INPUT_SIZE=64, FINE_HEATMAP_SIZE=16,
                    COARSE_EPOCHS=2, FINE_EPOCHS=2, NUM_WORKERS=0, BATCH_SIZE=2,
                    AMP=False, TTA=True, WARMUP_EPOCHS=0, PATIENCE=1)
    settings.update(kwargs)
    return m.ResearchConfig(**settings)


def save_round(number, report):
    out = ROOT / "work/five_round_tests" / f"round_{number:02d}"
    out.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for name in ["research_model_part.py", "research_config_part.py"]:
        source = ROOT / "work/fovea_v3" / name
        shutil.copyfile(source, out / name)
        hashes[name] = hashlib.sha256(source.read_bytes()).hexdigest()
    report = {"status": "PASS", "round": number, "device": "cpu", "torch_threads": 1,
              "pretrained": True, "test_type": "Functional real-image checks; not accuracy evidence",
              "source_sha256": hashes, **report}
    (out / "evidence.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("ROUND", number, "PASS:", out, flush=True)
    return report
