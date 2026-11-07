"""Independent submission, split, checkpoint, and artifact consistency audit."""
from pathlib import Path
import ast
import csv
import hashlib
import json
import math
from collections import Counter

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'outputs/trained_submission_20261025_fixed'
def rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))
def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))
def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

summary=read_json(RUN/'run_summary.json')
assert summary['smoke_test'] is False and summary['n_folds']==5
assert summary['n_train']==80 and summary['n_test']==20
assert summary['submission_experiment']=='B4'
pred=rows(ROOT/'outputs/submission.csv')
template=rows(ROOT/'work/real_competition_data/detection/sample_submission.csv')
assert len(pred)==40
assert list(pred[0])==list(template[0])==['ImageID','value']
assert [r['ImageID'] for r in pred]==[r['ImageID'] for r in template]
assert len(set(r['ImageID'] for r in pred))==40
assert all(math.isfinite(float(r['value'])) for r in pred)
ensemble={r['image_id']:r for r in rows(RUN/'test_ensemble_B4.csv')}
assert set(ensemble)==set(str(i) for i in range(81,101))
for r in pred:
    image_id,axis=r['ImageID'].split('_Fovea_')
    e=ensemble[image_id]; v=float(r['value']); coordinate='x' if axis=='X' else 'y'
    assert abs(v-float(e[coordinate]))<=0.00000051
    bound=float(e['width' if axis=='X' else 'height'])-1
    assert 0<=v<=bound
all_test=rows(RUN/'test_all_folds_B4.csv')
assert len(all_test)==100
assert Counter(r['image_id'] for r in all_test)==Counter({k:5 for k in ensemble})
for image_id,e in ensemble.items():
    pp=[r for r in all_test if r['image_id']==image_id]
    assert set(r['fold'] for r in pp)==set(map(str,range(5)))
    for coordinate in ['x','y']:
        assert abs(sum(float(r[coordinate]) for r in pp)/5-float(e[coordinate]))<1e-6

folds=rows(RUN/'folds.csv')
id_to_fold={r['image_id']:r['fold'] for r in folds}
assert id_to_fold['39']==id_to_fold['77']
oof=rows(RUN/'oof_B4.csv')
assert len(oof)==80 and len(set(r['image_id'] for r in oof))==80
rmse=math.sqrt(sum((float(r['x'])-float(r['x_true']))**2+(float(r['y'])-float(r['y_true']))**2 for r in oof)/160)
fold_reports=[]
checkpoint_hashes={}
for fold in range(5):
    folder=RUN/f'fold_{fold}'
    roles=rows(folder/'split_roles.csv')
    group_roles={}
    for r in roles:group_roles.setdefault(r['group'],set()).add(r['role'])
    assert all(len(v)==1 for v in group_roles.values())
    held_ids={r['image_id'] for r in roles if r['role']=='outer_validation'}
    assert held_ids=={r['image_id'] for r in oof if int(r['fold'])==fold}
    for audit in read_json(folder/'crop_crossfit/crossfit_audit.json'):
        assert not set(audit['fit_groups']) & set(audit['held_groups'])
    stages=list(folder.rglob('training_summary.json'))
    assert len(stages)==8
    for stage in stages:
        info=read_json(stage)
        assert info['best_epoch']>=1
        history=rows(stage.parent/'history.csv')
        assert len(history)>=info['best_epoch']
        assert all(math.isfinite(float(h['train_loss'])) for h in history)
    for stage in ('coarse_final','fine_final'):
        p=folder/stage/'model.pt'
        assert p.stat().st_size>1000000
        checkpoint_hashes[str(p.relative_to(RUN))]=digest(p)
    fold_reports.append({'fold':fold,'roles':dict(Counter(r['role'] for r in roles)),
                         'training_stages':len(stages),
                         'fine_policy':read_json(folder/'fine_policy_selection.json')['selected']})

nb=read_json(ROOT/'outputs/fovea_submission_20261025.ipynb')
engine=ROOT/'work/current_notebook_review/submission_engine.py'
def definitions(text):
    return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(text).body
            if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
nbdefs={}
for cell in nb['cells']:
    if cell['cell_type']=='code':nbdefs.update(definitions(''.join(cell['source'])))
assert nbdefs==definitions(engine.read_text(encoding='utf-8'))
report={'status':'PASS','submission_rows':len(pred),'test_images':20,'folds':5,
        'actual_training_stages':40,'oof_coordinate_rmse_px':rmse,
        'source_notebook':'notebook3 (1).ipynb',
        'source_notebook_sha256':digest(Path(r'C:/Users/lenovo/Downloads/notebook3 (1).ipynb')),
        'submission_sha256':digest(ROOT/'outputs/submission.csv'),
        'trained_engine_sha256':digest(engine),
        'notebook_sha256':digest(ROOT/'outputs/fovea_submission_20261025.ipynb'),
        'notebook_definition_count':len(nbdefs),'notebook_matches_trained_functions':True,
        'submission_matches_five_fold_mean':True,'template_ids_order_columns_match':True,
        'all_coordinates_finite_and_in_image':True,'duplicate_group_kept_together':True,
        'role_group_disjoint':True,'fold_details':fold_reports,
        'final_checkpoint_sha256':checkpoint_hashes,
        'score_scope':'Internal OOF score. No Kaggle test labels or leaderboard score available.'}
(ROOT/'outputs/submission_validation.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
print(json.dumps({k:v for k,v in report.items() if k not in ['fold_details','final_checkpoint_sha256']},indent=2))
