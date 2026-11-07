"""Read-only lineage, split, checkpoint and independent inference audit."""
from pathlib import Path
import json,os,sys,time,hashlib,importlib.util
ROOT=Path(__file__).resolve().parents[2];WORK=Path(__file__).resolve().parent;OUT=ROOT/'outputs/restored_20261027'
os.environ['MPLBACKEND']='Agg';os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
sys.path.insert(0,str(ROOT/'work/gpu_test_deps'))
import torch,numpy as np,pandas as pd
torch.set_num_threads(2)
spec=importlib.util.spec_from_file_location('verify_engine',WORK/'engine.py')
m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
js=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
read=lambda p:pd.read_csv(p,dtype={'image_id':str,'group':str})
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
started=time.perf_counter();protocol=js(OUT/'protocol.json')
assert sha(ROOT/'outputs/submission.csv')==sha(OUT/'submission_restored_8p9.csv')==protocol['baseline_submission_sha256']
assert sha(WORK/'engine.py')==protocol['engine_sha256']
old=ROOT/'outputs/trained_submission_20261025_fixed';new=OUT/'seed_10000'
old_config=js(old/'run_config.json')['config'];new_config=js(new/'run_config.json')['config']
config_changes={k for k in old_config.keys()|new_config.keys() if old_config.get(k)!=new_config.get(k)}
assert config_changes=={'MODEL_SEED_OFFSET','OUTPUT_DIR','SUBMISSION_PATH'},config_changes
train=read(new/'folds.csv');test=read(new/'test_manifest.csv');old_train=read(old/'folds.csv')
assert len(train)==80 and len(test)==20
assert set(train.image_id)==set(map(str,range(1,81))) and set(test.image_id)==set(map(str,range(81,101)))
assert not set(train.pixel_hash)&set(test.pixel_hash)
for col in ['fold','group','pixel_hash','x','y']:
 assert np.array_equal(train[col],old_train[col]),col
assert train.loc[train.image_id.isin(['39','77']),'fold'].nunique()==1
stage_count=0;split_checks=[];different_final_checkpoints=0
for f in range(5):
 folder=new/f'fold_{f}'
 roles=read(folder/'split_roles.csv');roles_old=read(old/f'fold_{f}'/'split_roles.csv')
 pd.testing.assert_frame_equal(roles,roles_old)
 assert roles.image_id.is_unique and len(roles)==80
 assert (roles.groupby('group').role.nunique()==1).all()
 excluded=set(roles.loc[roles.role.isin(['calibration_only','outer_validation']),'image_id'])|set(test.image_id)
 development=set(roles.loc[roles.role.isin(['model_fit','early_stop']),'image_id'])
 stages=list(folder.rglob('actual_training_inputs.csv'));assert len(stages)==8
 for p in stages:
  stage_count+=1;actual=read(p)
  assert not set(actual.image_id)&excluded
  assert set(actual.image_id)<=development
  stop_path=p.parent/'actual_early_stop_inputs.csv'
  if stop_path.exists():
   stop=read(stop_path)
   assert not set(stop.image_id)&excluded
   assert not set(actual.group)&set(stop.group)
  m.research_verify_cache(p.parent,js(p.parent/'stage_complete.json')['files'])
  info=js(p.parent/'training_summary.json');assert info['successful_optimizer_steps']>0
  hist=pd.read_csv(p.parent/'history.csv');assert np.isfinite(hist.train_loss).all()
 for stage in ['coarse_final','fine_final']:
  assert set(read(folder/stage/'actual_training_inputs.csv').image_id)==development
  old_digest=js(old/f'fold_{f}'/stage/'stage_complete.json')['files']['model.pt']['sha256']
  new_digest=js(folder/stage/'stage_complete.json')['files']['model.pt']['sha256']
  assert old_digest!=new_digest
  different_final_checkpoints+=1
 assert js(folder/'crop_crossfit/crossfit_audit.json')==js(old/f'fold_{f}'/'crop_crossfit/crossfit_audit.json')
 for a in js(folder/'crop_crossfit/crossfit_audit.json'):
  assert not set(a['fit_ids'])&set(a['held_ids']) and not set(a['fit_groups'])&set(a['held_groups'])
  assert not (set(a['fit_ids'])|set(a['held_ids']))&excluded
 split_checks.append({'fold':f,'training_stages':len(stages),'final_training_images':len(development),
                     'same_all_split_roles_as_baseline':True,'test_calibration_outer_overlap':0})
all_reloaded=[];reload_checks=[]
for label,run in [('baseline',old),('additional_seed',new)]:
 cfg=m.ResearchConfig(**js(run/'run_config.json')['config'])
 cfg.PRETRAINED=False;cfg.PRETRAINED_WEIGHTS_PATH=None
 m.seed_everything(cfg.SEED)
 m.cv2.setNumThreads(0);parts=[]
 for f in range(5):
  folder=run/f'fold_{f}'
  # Verify all original fold files too before trusting the baseline component.
  m.research_verify_cache(folder,js(folder/'complete.json')['files'])
  coarse=m.research_build_model(cfg,'coarse','cuda')
  m.load_model_checkpoint(coarse,folder/'coarse_final/model.pt','cuda')
  cp=m.predict_research(coarse,test,cfg,'cuda','coarse')
  m._research_cleanup(coarse);del coarse
  roi=m.make_roi_spec(cp,test,cfg,mode='calibrated',calibration=js(folder/'radius_calibration.json'))
  fine=m.research_build_model(cfg,'fine','cuda')
  m.load_model_checkpoint(fine,folder/'fine_final/model.pt','cuda')
  fp=m.predict_research(fine,test,cfg,'cuda','fine',roi)
  bp=m.apply_fallback(cp,fp,js(folder/'fallback_rule.json'))
  m._research_cleanup(fine);del fine
  saved=read(folder/'B4_test.csv').set_index('image_id').loc[bp.image_id]
  delta=float(np.max(np.abs(saved[['x','y']].to_numpy()-bp[['x','y']].to_numpy())))
  assert delta<.001,(label,f,delta)
  parts.append(bp)
  reload_checks.append({'component':label,'fold':f,'maximum_coordinate_difference_px':delta})
  print('Reload verified',label,f,'max delta',delta,flush=True)
 all_reloaded.append(pd.concat(parts).groupby('image_id')[['x','y']].mean())
ensemble=.75*all_reloaded[0]+.25*all_reloaded[1]
sub=pd.read_csv(OUT/'submission_candidate.csv');baseline=pd.read_csv(OUT/'submission_restored_8p9.csv')
assert sub.ImageID.tolist()==baseline.ImageID.tolist() and list(sub.columns)==['ImageID','value'] and len(sub)==40
expected=[];baseline_expected=[]
for row in sub.itertuples():
 iid,axis=row.ImageID.split('_Fovea_');col='x' if axis=='X' else 'y'
 expected.append(ensemble.loc[iid,col]);baseline_expected.append(all_reloaded[0].loc[iid,col])
 assert np.isfinite(row.value) and 0<=row.value<=test.set_index('image_id').loc[iid,'width' if axis=='X' else 'height']-1
maxdiff=float(np.max(np.abs(sub.value.to_numpy()-expected)))
old_diff=float(np.max(np.abs(baseline.value.to_numpy()-baseline_expected)))
assert maxdiff<.001 and old_diff<.001
report={'status':'PASS','new_training_stages':stage_count,'reloaded_final_networks':20,
 'new_final_checkpoints_different_from_old':different_final_checkpoints,
 'new_submission_sha256':sha(OUT/'submission_candidate.csv'),'baseline_sha256':sha(OUT/'submission_restored_8p9.csv'),
 'baseline_unchanged':True,'engine_sha256':sha(WORK/'engine.py'),'rows':40,
 'reloaded_candidate_max_difference_px':maxdiff,'reloaded_baseline_max_difference_px':old_diff,
 'all_training_holdouts_unchanged':True,'test_images_in_training_or_selection':0,
 'crossfit_assignments_identical_to_baseline':True,'configuration_changes':sorted(config_changes),
 'split_checks':split_checks,'reload_checks':reload_checks,'elapsed_seconds':time.perf_counter()-started,
 'scope':'Exact duplicate and recorded input audit; patient identity unavailable. No claim of independent development OOF or ensemble coverage.'}
(OUT/'validation_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2),flush=True)
