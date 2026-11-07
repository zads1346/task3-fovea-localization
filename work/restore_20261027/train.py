from pathlib import Path
import os,sys,json,time,importlib.util
ROOT=Path(__file__).resolve().parents[2]
os.environ['MPLBACKEND']='Agg'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
sys.path.insert(0,str(ROOT/'work/gpu_test_deps'))
import torch
torch.set_num_threads(2)
spec=importlib.util.spec_from_file_location('restored_engine',Path(__file__).with_name('engine.py'))
m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
cfg=m.ResearchConfig(**json.loads((ROOT/'outputs/trained_submission_20261025_fixed/run_config.json').read_text())['config'])
cfg.MODEL_SEED_OFFSET=10000
cfg.OUTPUT_DIR=str(ROOT/'outputs/restored_20261027/seed_10000')
cfg.SUBMISSION_PATH=str(Path(cfg.OUTPUT_DIR)/'submission.csv')
if __name__=='__main__':
 started=time.perf_counter()
 print('Original 8.9 pipeline, new model RNG only; split seed unchanged. GPU:',torch.cuda.get_device_name(0),flush=True)
 print(m.research_preflight(cfg),flush=True)
 result=m.run_research(cfg)
 m.json_save({'elapsed_seconds':time.perf_counter()-started,'gpu':torch.cuda.get_device_name(0),'peak_cuda_mb':torch.cuda.max_memory_allocated()/2**20},Path(cfg.OUTPUT_DIR)/'execution_report.json')
 print(result['summary'].to_string(index=False),flush=True)
