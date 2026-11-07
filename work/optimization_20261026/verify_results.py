"""Verify actual batch-input manifests and independently regenerate the new submission."""
from pathlib import Path
import json,os,sys,importlib.util,hashlib,time
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/optimized_20261026'
os.environ['MPLBACKEND']='Agg';os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
sys.path.insert(0,str(ROOT/'work/gpu_test_deps'))
import torch,numpy as np,pandas as pd
torch.set_num_threads(2)
spec=importlib.util.spec_from_file_location('verified_engine',Path(__file__).with_name('engine.py'))
m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
def js(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def frame(p):return pd.read_csv(p,dtype={'image_id':str,'group':str})
cfg=m.OptimizedConfig(**js(OUT/'run_config.json')['config'])
assert js(OUT/'run_summary.json')['status']=='COMPLETE'
old_hash=js(ROOT/'outputs/submission_validation.json')['submission_sha256']
assert m.research_file_digest(ROOT/'outputs/submission.csv')==old_hash
assert m.research_file_digest(Path(__file__).with_name('engine.py'))==js(OUT/'execution_report.json')['engine_sha256']
train=frame(OUT/'folds.csv');test=frame(OUT/'test_manifest.csv')
assert set(train.image_id)==set(map(str,range(1,81)))
assert set(test.image_id)==set(map(str,range(81,101)))
assert not set(train.pixel_hash)&set(test.pixel_hash)
old=frame(ROOT/'outputs/trained_submission_20261025_fixed/folds.csv').set_index('image_id')
assert np.array_equal(train.set_index('image_id').loc[old.index,'fold'],old.fold)
assert train.set_index('image_id').loc['39','fold']==train.set_index('image_id').loc['77','fold']
cfg.PRETRAINED=False;cfg.PRETRAINED_WEIGHTS_PATH=None
m.seed_everything(cfg.SEED);m.cv2.setNumThreads(0)
predictions=[];folds=[];start=time.perf_counter()
for f in range(5):
    folder=OUT/f'fold_{f}';outer=set(train.loc[train.fold!=f,'image_id']);held=set(train.loc[train.fold==f,'image_id'])
    roles=frame(folder/'split_roles.csv')
    assert set(roles.image_id)==set(train.image_id)
    assert (roles.groupby('group').role.nunique()==1).all()
    stages=list(folder.rglob('actual_training_inputs.csv'));assert len(stages)==7
    for p in stages:
        actual=frame(p)
        assert set(actual.image_id)<=outer
        assert not set(actual.image_id)&(held|set(test.image_id))
        if (p.parent/'actual_early_stop_inputs.csv').exists():
            stop=frame(p.parent/'actual_early_stop_inputs.csv')
            assert set(stop.image_id)<=outer
            assert not set(actual.group)&set(stop.group)
        m.research_verify_cache(p.parent,js(p.parent/'stage_complete.json')['files'])
    for stage in ['coarse_final','fine_final']:
        assert set(frame(folder/stage/'actual_training_inputs.csv').image_id)==outer
    coarse_info=js(folder/'coarse_select/training_summary.json');fine_info=js(folder/'fine_select/training_summary.json')
    assert coarse_info['best_epoch']>=cfg.COARSE_MIN_EPOCHS
    assert fine_info['best_epoch']>=cfg.FINE_MIN_EPOCHS
    for p in folder.rglob('training_summary.json'):
        info=js(p);assert info['successful_optimizer_steps']>0
        hist=pd.read_csv(p.parent/'history.csv');assert np.isfinite(hist.train_loss).all()
    for a in js(folder/'crop_crossfit/crossfit_audit.json'):
        assert not set(a['fit_ids'])&set(a['held_ids'])
        assert not set(a['fit_groups'])&set(a['held_groups'])
    coarse=m.research_build_model(cfg,'coarse','cuda')
    m.load_model_checkpoint(coarse,folder/'coarse_final/model.pt','cuda')
    cp=m.predict_research(coarse,test,cfg,'cuda','coarse')
    m._research_cleanup(coarse);del coarse
    fine=m.research_build_model(cfg,'fine','cuda')
    m.load_model_checkpoint(fine,folder/'fine_final/model.pt','cuda')
    fp,bp,views=m.optimized_inference(fine,test,cp,cfg,'cuda')
    m._research_cleanup(fine);del fine
    saved=frame(folder/'BLEND_test.csv').set_index('image_id').loc[bp.image_id]
    delta=float(np.abs(saved[['x','y']].to_numpy()-bp[['x','y']].to_numpy()).max())
    assert delta<1e-3,(f,delta)
    predictions.append(bp.assign(fold=f))
    folds.append({'fold':f,'final_training_images':len(outer),'test_overlap':0,'outer_validation_overlap':0,
                  'actual_training_stages':len(stages),'reloaded_prediction_max_difference_px':delta,
                  'coarse_epochs':coarse_info['best_epoch'],'fine_epochs':fine_info['best_epoch']})
    print('Verified fold',f,'reloaded delta',delta,flush=True)
ensemble=pd.concat(predictions).groupby('image_id')[['x','y']].mean()
submission=pd.read_csv(OUT/'submission.csv');template=pd.read_csv(ROOT/'work/real_competition_data/detection/sample_submission.csv')
assert list(submission.columns)==['ImageID','value']
assert submission.ImageID.tolist()==template.ImageID.tolist() and len(submission)==40
assert np.isfinite(submission.value).all()
check=[]
for row in submission.itertuples():
    image_id,axis=row.ImageID.split('_Fovea_');coordinate='x' if axis=='X' else 'y'
    check.append(float(ensemble.loc[image_id,coordinate]))
    bound=float(test.set_index('image_id').loc[image_id,'width' if axis=='X' else 'height'])-1
    assert 0<=row.value<=bound
delta=float(np.max(np.abs(np.asarray(check)-submission.value.to_numpy())))
assert delta<1e-3
oof=frame(OUT/'oof_BLEND.csv')
assert set(oof.image_id)==set(train.image_id) and oof.image_id.is_unique
score=float(np.sqrt(np.mean((oof[['x','y']].to_numpy()-oof[['x_true','y_true']].to_numpy())**2)))
report={'status':'PASS','backup_verified':js(ROOT/'backups/backup_20261026_verification.json')['status']=='PASS',
 'old_submission_unchanged':True,'old_submission_sha256':old_hash,
 'source_engine_sha256':m.research_file_digest(Path(__file__).with_name('engine.py')),
 'submission_sha256':m.research_file_digest(OUT/'submission.csv'),'new_submission_rows':40,
 'test_images':20,'actual_training_stages':35,'reloaded_final_models':10,
 'same_outer_folds_as_previous_version':True,'test_images_in_training_or_early_stopping':0,
 'folds':folds,'oof_coordinate_rmse_px':score,'reloaded_submission_max_difference_px':delta,
 'seconds':time.perf_counter()-start,'limitation':'Same data and folds were used for iterative development; no new independent test score is available.'}
m.json_save(report,OUT/'validation_report.json')
print(json.dumps(report,indent=2),flush=True)
