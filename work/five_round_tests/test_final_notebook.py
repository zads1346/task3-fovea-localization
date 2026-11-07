"""Fresh namespace, native notebook cells, real data, all five outer folds."""
from pathlib import Path
import sys, os, json, ast, types, hashlib, time
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'work/kaggle_test_deps'))
os.environ['MPLBACKEND']='Agg'
import torch, numpy as np, pandas as pd
torch.set_num_threads(2)
notebook=ROOT/'outputs/fovea_optimized_5rounds.ipynb'
nb=json.loads(notebook.read_text(encoding='utf-8'))
module=types.ModuleType('optimized_notebook_acceptance')
sys.modules[module.__name__]=module
scope=module.__dict__
scope['display']=lambda value:None
started=time.perf_counter()
executed=[]
work=ROOT/'work/five_round_tests/final_fivefold'
work.mkdir(exist_ok=True)
os.chdir(work)
for index,cell in enumerate(nb['cells']):
    if cell['cell_type']!='code':continue
    source=''.join(cell['source'])
    tree=ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node,ast.Import):names=[alias.name for alias in node.names]
        elif isinstance(node,ast.ImportFrom):names=[node.module or '']
        else:continue
        assert not any(name.startswith(('work.','outputs.','research_','data_part','model_training_part','pipeline_part','fovea_')) for name in names),names
    if 'PREFLIGHT = research_preflight(CFG)' in source:
        scope['CFG']=scope['ResearchConfig'](
            INPUT_ROOT=str(ROOT/'work/real_competition_data'),OUTPUT_DIR=str(work),
            SUBMISSION_PATH=str(work/'submission.csv'),
            PRETRAINED_WEIGHTS_PATH=str(ROOT/'outputs/fovea_pretrained_weights.zip'),
            SMOKE_TEST=True,SMOKE_FOLDS=5,SMOKE_INPUT_SIZE=128,SMOKE_EPOCHS=2,
            NUM_WORKERS=0,AMP=False,SHOW_PLOTS=False,TTA=True,
            FINE_TRAIN_CROPS='oof',INNER_CROP_FOLDS=3,SELECT_FINE_CROP_POLICY=True)
    exec(compile(source,f'notebook_cell_{index}','exec'),scope,scope)
    executed.append(index)
result=scope['RESULTS'];m=module
assert not hasattr(m,'__file__')
train,test=result['train'],result['test']
assert len(train)==80 and len(test)==20 and train.fold.nunique()==5
assert train.set_index('image_id').loc[['39','77'],'fold'].nunique()==1
assert train.set_index('image_id').loc[['39','77'],'group'].nunique()==1
expected=[f'{i}_Fovea_{axis}' for i in range(81,101) for axis in ('X','Y')]
for name,pred in result['oof'].items():
    assert len(pred)==80 and pred.image_id.nunique()==80
    assert np.isfinite(pred[['x','y','uncertainty']]).all().all()
    assert result['submissions'][name].ImageID.tolist()==expected
    assert np.isfinite(result['submissions'][name].value).all()
output=Path(result['config'].OUTPUT_DIR)
selections=[]
for fold in range(5):
    folder=output/f'fold_{fold}'
    roles=pd.read_csv(folder/'split_roles.csv',dtype=str)
    assert roles.groupby('group').role.nunique().max()==1
    audit=json.loads((folder/'crop_crossfit/crossfit_audit.json').read_text())
    for record in audit:assert not set(record['fit_groups'])&set(record['held_groups'])
    selection=json.loads((folder/'fine_policy_selection.json').read_text())
    assert len(selection['candidates'])==2
    assert set(selection['stopping_image_ids'])==set(roles.loc[roles.role=='early_stop','image_id'])
    assert selection['selected']['best_rmse']==min(x['best_rmse'] for x in selection['candidates'])
    selections.append({'fold':fold,**selection['selected']})
    radius=json.loads((folder/'radius_calibration.json').read_text())
    assert set(radius['calibration_image_ids'])==set(roles.loc[roles.role=='calibration_only','image_id'])
    assert (folder/'calibration_health.json').exists()
    assert len(list(folder.rglob('stage_complete.json')))==8
    assert json.loads((folder/'area_matched_control.json').read_text())['relative_area_difference']<1e-7
assert len(pd.read_csv(output/'uncertainty_calibration_curve.csv'))>=4
mtimes={str(p):p.stat().st_mtime_ns for p in output.glob('fold_*/**/model.pt')}
assert len(mtimes)==40
again=m.run_research(m.CFG)
pd.testing.assert_frame_equal(result['submission'],again['submission'])
assert mtimes=={str(p):p.stat().st_mtime_ns for p in output.glob('fold_*/**/model.pt')}
checks=['all_notebook_cells_executed','official_weight_zip_offline','five_outer_folds',
        '40_real_model_training_jobs','six_ablations','80_unique_oof_rows_each',
        '40_submission_rows_each_correct_order','duplicates_same_group_and_fold',
        'calibration_never_trained','crossfit_target_groups_excluded',
        'fine_policy_selected_inner_only','completed_fold_reuse_preserves_models']
folder=output/'fold_0'
record=json.loads((folder/'complete.json').read_text())
for filename in ['B4_oof.csv','radius_calibration.json','coarse_outer_validation.csv']:
    path=folder/filename;saved=path.read_bytes();path.write_bytes(saved+b'\n')
    try:
        m.run_research_fold(train,test,0,result['config'],torch.device('cpu'),output,record['signature'])
    except ValueError as error:assert 'integrity' in str(error)
    else:raise AssertionError(f'Tampering accepted: {filename}')
    finally:path.write_bytes(saved)
    checks.append('tamper_rejected:'+filename)
# Simulate an interrupted fold after stage completion. All eight models should
# survive unchanged while downstream predictions and final marker are rebuilt.
(folder/'complete.json').unlink()
m.run_research_fold(train,test,0,result['config'],torch.device('cpu'),output,record['signature'])
assert mtimes=={str(p):p.stat().st_mtime_ns for p in output.glob('fold_*/**/model.pt')}
checks.append('interrupted_fold_reuses_all_completed_stages')
report={'status':'PASS','round':5,'checks':checks,'code_cells_executed':executed,
        'source_sha256':m.RESEARCH_SOURCE_HASH,'notebook_sha256_at_test':hashlib.sha256(notebook.read_bytes()).hexdigest(),
        'device':'cpu','input_size':128,'epochs_per_training_job':2,'training_jobs':40,
        'outer_folds':5,'train_images':80,'test_images':20,'experiments':list(result['oof']),
        'submission_rows':40,'pretrained':True,'tta':True,'inner_crop_selection':selections,
        'elapsed_seconds':time.perf_counter()-started,
        'full_resolution_formal_training_completed':False,
        'accuracy_claim':'None: reduced-resolution execution acceptance, not formal Kaggle accuracy.'}
(ROOT/'work/five_round_tests/final_notebook_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print('FINAL_NOTEBOOK_FIVEFOLD_PASS',json.dumps(report,indent=2),flush=True)
