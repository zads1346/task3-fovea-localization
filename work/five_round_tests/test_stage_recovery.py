from pathlib import Path
import sys, json, importlib.util, os, shutil
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'work/kaggle_test_deps'))
os.environ['MPLBACKEND']='Agg'
import torch
torch.set_num_threads(2)
spec=importlib.util.spec_from_file_location('cache_acceptance',ROOT/'outputs/fovea_optimized_5rounds.py')
m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
out=ROOT/'work/five_round_tests/stage_recovery'
cfg=m.ResearchConfig(DATA_DIR=str(ROOT/'work/real_competition_data/detection'),
    PRETRAINED_WEIGHTS_PATH=str(ROOT/'outputs/fovea_pretrained_weights/resnet18-f37072fd.pth'),
    OUTPUT_DIR=str(out),INPUT_SIZE=64,HEATMAP_SIZE=16,NUM_WORKERS=0,BATCH_SIZE=2,
    COARSE_EPOCHS=2,FINE_EPOCHS=2,AMP=False,REUSE_CHECKPOINTS=True,REQUIRE_GPU=False)
train,*_=m.prepare_tables(cfg)
fit=train.iloc[:4].copy();val=train.iloc[4:6].copy()
model,info=m._research_stage(fit,val,cfg,'cpu','coarse',out,2026)
before=m.predict_research(model,val,cfg,'cpu')
del model
mtime=(out/'model.pt').stat().st_mtime_ns
model,reused=m._research_stage(fit,val,cfg,'cpu','coarse',out,2026)
assert reused['reused_completed_stage'] is True
after=m.predict_research(model,val,cfg,'cpu')
m.np.testing.assert_allclose(before[['x','y','uncertainty']],after[['x','y','uncertainty']],atol=0,rtol=0)
assert mtime==(out/'model.pt').stat().st_mtime_ns
del model
checks=['real_stage_trained','completed_stage_reused_without_retraining','identical_predictions_after_reload']
def reject(call,fragment):
    try:call()
    except ValueError as error:assert fragment in str(error),str(error)
    else:raise AssertionError('Expected cache rejection')
path=out/'history.csv';saved=path.read_bytes();path.write_bytes(saved+b'\n')
reject(lambda:m._research_stage(fit,val,cfg,'cpu','coarse',out,2026),'integrity')
path.write_bytes(saved);checks.append('tampered_history_rejected')
path=out/'model.pt'
with path.open('r+b') as stream:
    first=stream.read(1);stream.seek(0);stream.write(bytes([first[0]^1]))
reject(lambda:m._research_stage(fit,val,cfg,'cpu','coarse',out,2026),'integrity')
with path.open('r+b') as stream:stream.write(first)
checks.append('same_size_corrupt_model_rejected')
reject(lambda:m._research_stage(fit,val,m.replace(cfg,NUM_WORKERS=1),'cpu','coarse',out,2026),'changed')
checks.append('worker_rng_config_change_rejected')
changed=fit.copy();changed.loc[0,'x']+=1
reject(lambda:m._research_stage(changed,val,cfg,'cpu','coarse',out,2026),'changed')
checks.append('training_label_change_rejected')
record=json.loads((out/'stage_complete.json').read_text())
incomplete=record['files'].copy();incomplete.pop('model.pt')
reject(lambda:m.research_verify_cache(out,incomplete,('model.pt','history.csv','training_summary.json')),'missing required')
checks.append('manifest_missing_required_entry_rejected')
files=record['files'].copy();files['../escape.pt']={'bytes':0,'sha256':'x'}
reject(lambda:m.research_verify_cache(out,files),'escapes')
checks.append('path_escape_rejected')
assert m.research_stage_signature(fit,object(),cfg,'coarse',1,1,None,object())==m.research_stage_signature(fit,None,cfg,'coarse',1,1,None,None)
checks.append('fixed_epoch_cache_signature_ignores_validation')
original=m.research_geometry_selfcheck
exec('def research_geometry_selfcheck():\n    return {"changed": True}\n',m.__dict__)
reject(lambda:m._research_stage(fit,val,cfg,'cpu','coarse',out,2026),'changed')
m.research_geometry_selfcheck=original
checks.append('notebook_function_edit_rejected')
report={'round':5,'status':'PASS','checks':checks,'source_sha256':m.RESEARCH_SOURCE_HASH,
        'real_images':6,'input_size':64,'max_epochs':2,'accuracy_claim':False}
(ROOT/'work/five_round_tests/stage_recovery_report.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
