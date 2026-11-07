"""Train a real five-fold submission; preserve an independent calibration split."""
from pathlib import Path
import importlib.util
import json
import os
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
os.environ['MPLBACKEND'] = 'Agg'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
sys.path.insert(0, str(ROOT / 'work/gpu_test_deps'))
import torch
torch.set_num_threads(2)
spec = importlib.util.spec_from_file_location('submission_engine', ROOT / 'work/current_notebook_review/submission_engine.py')
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)
cfg = m.ResearchConfig(
 DATA_DIR=str(ROOT / 'work/real_competition_data/detection'),
 INPUT_ROOT=str(ROOT / 'work/real_competition_data'),
 PRETRAINED_WEIGHTS_PATH=str(ROOT / 'outputs/fovea_pretrained_weights/resnet18-f37072fd.pth'),
 OUTPUT_DIR=str(ROOT / 'outputs/trained_submission_20261025_fixed'),
 SUBMISSION_PATH=str(ROOT / 'outputs/submission.csv'),
 NUM_WORKERS=0, BATCH_SIZE=8, AMP=True, INPUT_SIZE=256, HEATMAP_SIZE=64,
 FINE_INPUT_SIZE=256, FINE_HEATMAP_SIZE=64, COARSE_EPOCHS=40, FINE_EPOCHS=30,
 PATIENCE=7, N_FOLDS=5, SMOKE_TEST=False, REQUIRE_GPU=True,
 SELECT_FINE_CROP_POLICY=True, BACKBONE_LR_MULTIPLIER=1.0, USE_EMA=False,
 SUBMISSION_EXPERIMENT='B4', BOOTSTRAP_REPEATS=1000,
)
if __name__ == '__main__':
 started=time.perf_counter()
 print('TRAINING A NEW MODEL: 5 folds, 256/256 inputs, 40/30 maximum epochs, B4 preselected.',flush=True)
 print(m.research_preflight(cfg),flush=True)
 result=m.run_research(cfg)
 m.json_save({'elapsed_seconds':time.perf_counter()-started,'cuda_peak_memory_mb':torch.cuda.max_memory_allocated()/2**20,
              'metric_scope':'Internal out-of-fold validation; not a Kaggle leaderboard score.',
              'origin':'Newly trained on the actual 80 labeled competition images; no smoke checkpoint reused.'},
             Path(cfg.OUTPUT_DIR)/'execution_report.json')
 print(result['summary'].to_string(index=False),flush=True)
