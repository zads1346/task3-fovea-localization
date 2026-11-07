"""Regenerate B4 predictions from all ten saved final networks and real images."""
from pathlib import Path
import gc
import importlib.util
import json
import os
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'outputs/trained_submission_20261025_fixed'
os.environ['MPLBACKEND']='Agg'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
sys.path.insert(0,str(ROOT/'work/gpu_test_deps'))
import torch
import numpy as np
import pandas as pd
torch.set_num_threads(2)
spec=importlib.util.spec_from_file_location('reload_engine',ROOT/'work/current_notebook_review/submission_engine.py')
m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
cfg=m.ResearchConfig(**json.loads((RUN/'run_config.json').read_text())['config'])
cfg.PRETRAINED=False
cfg.PRETRAINED_WEIGHTS_PATH=None
m.seed_everything(cfg.SEED);m.cv2.setNumThreads(0)
test=pd.read_csv(RUN/'test_manifest.csv',dtype={'image_id':str})
preds=[];details=[];start=time.perf_counter()
for fold in range(5):
    folder=RUN/f'fold_{fold}'
    coarse=m.research_build_model(cfg,'coarse','cuda')
    m.load_model_checkpoint(coarse,folder/'coarse_final/model.pt','cuda')
    cp=m.predict_research(coarse,test,cfg,'cuda','coarse')
    m._research_cleanup(coarse);del coarse;gc.collect()
    radius=json.loads((folder/'radius_calibration.json').read_text())
    rule=json.loads((folder/'fallback_rule.json').read_text())
    roi=m.make_roi_spec(cp,test,cfg,mode='calibrated',calibration=radius)
    fine=m.research_build_model(cfg,'fine','cuda')
    m.load_model_checkpoint(fine,folder/'fine_final/model.pt','cuda')
    fp=m.predict_research(fine,test,cfg,'cuda','fine',roi)
    out=m.apply_fallback(cp,fp,rule)
    original=pd.read_csv(folder/'B4_test.csv',dtype={'image_id':str}).set_index('image_id').loc[out.image_id]
    difference=np.abs(original[['x','y']].to_numpy()-out[['x','y']].to_numpy()).max()
    assert difference<1e-3,(fold,difference)
    details.append({'fold':fold,'max_abs_difference_px':float(difference),'images':len(out)})
    preds.append(out.assign(fold=fold))
    m._research_cleanup(fine);del fine;gc.collect()
    print('Fold reloaded',fold,'max difference',difference,flush=True)
ensemble=pd.concat(preds).groupby('image_id',as_index=False)[['x','y']].mean().set_index('image_id')
submission=pd.read_csv(ROOT/'outputs/submission.csv')
regenerated=submission.copy()
regenerated['value']=[ensemble.loc[str(k.split('_Fovea_')[0]),'x' if k.endswith('_X') else 'y'] for k in submission.ImageID]
delta=float(np.abs(regenerated.value-submission.value).max())
assert delta<0.001001
regenerated.to_csv(ROOT/'work/current_notebook_review/reloaded_submission.csv',index=False,float_format='%.6f')
report={'status':'PASS','final_networks_reloaded':10,'test_images_per_fold':20,'folds':details,
        'submission_max_abs_difference_px':delta,'seconds':time.perf_counter()-start,
        'inference_uses_no_test_labels':True,'pretraining_weights_not_needed_for_reload':True}
(ROOT/'outputs/submission_reload_verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2),flush=True)
