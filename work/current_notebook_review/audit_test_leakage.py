"""Read-only audit of the actual submission's data lineage and train/test separation."""
from pathlib import Path
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'work/kaggle_test_deps'))
import numpy as np
from PIL import Image
RUN=ROOT/'outputs/trained_submission_20261025_fixed'
DATA=ROOT/'work/real_competition_data/detection'
def rows(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def js(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

train=rows(RUN/'folds.csv');test=rows(RUN/'test_manifest.csv')
tid={r['image_id'] for r in train};qid={r['image_id'] for r in test}
assert tid==set(map(str,range(1,81))) and qid==set(map(str,range(81,101)))
assert not tid&qid
assert not {str(Path(r['path']).resolve()) for r in train}&{str(Path(r['path']).resolve()) for r in test}
assert all(Path(r['path']).parent.name=='train' for r in train)
assert all(Path(r['path']).parent.name=='test' for r in test)
labels={str(int(r['data'])):r for r in rows(DATA/'fovea_localization_train_GT.csv')}
assert set(labels)==tid
assert all(float(r['x'])==float(labels[r['image_id']]['Fovea_X']) and float(r['y'])==float(labels[r['image_id']]['Fovea_Y']) for r in train)
template=rows(DATA/'sample_submission.csv')
assert all(r['value']=='' for r in template)
assert not ({'x','y','Fovea_X','Fovea_Y'}&set(test[0]))
pixel_groups={}
for role,frame in [('train',train),('test',test)]:
    groups=defaultdict(list)
    for r in frame:
        with Image.open(r['path']) as im:rgb=np.asarray(im.convert('RGB'))
        digest=hashlib.sha256(str(rgb.shape).encode()+rgb.tobytes()).hexdigest()
        assert digest==r['pixel_hash'],r['image_id']
        groups[digest].append(r['image_id'])
    pixel_groups[role]=groups
assert not set(pixel_groups['train'])&set(pixel_groups['test'])

fold_checks=[];stage_files=0
for f in range(5):
    folder=RUN/f'fold_{f}'
    roles=rows(folder/'split_roles.csv');group_roles=defaultdict(set)
    role_ids=defaultdict(set)
    for r in roles:
        role_ids[r['role']].add(r['image_id']);group_roles[r['group']].add(r['role'])
    assert set().union(*role_ids.values())==tid
    assert all(len(v)==1 for v in group_roles.values())
    assert not set().union(*role_ids.values())&qid
    dev=role_ids['model_fit']|role_ids['early_stop'];cal=role_ids['calibration_only'];val=role_ids['outer_validation']
    assert not dev&cal and not dev&val and not cal&val
    assert val=={r['image_id'] for r in train if int(r['fold'])==f}
    split=js(folder/'split_audit.json')
    assert set(split['development_ids'])==dev
    assert set(split['calibration_ids'])==cal and set(split['validation_ids'])==val
    for a in js(folder/'crop_crossfit/crossfit_audit.json'):
        fit,held=set(a['fit_ids']),set(a['held_ids'])
        assert fit|held==role_ids['model_fit'] and not fit&held
        assert not (fit|held)&(qid|cal|val|role_ids['early_stop'])
        assert not set(a['fit_groups'])&set(a['held_groups'])
    assert {r['image_id'] for r in rows(folder/'crop_crossfit/training_oof_centres.csv')}==role_ids['model_fit']
    for name in ['calibration_coarse.csv','calibration_fine.csv','calibration_roi.csv']:
        assert {r['image_id'] for r in rows(folder/name)}==cal
    for name in ['radius_calibration.json','fallback_rule.json']:
        assert set(js(folder/name)['calibration_image_ids'])==cal
    assert set(js(folder/'fine_policy_selection.json')['stopping_image_ids'])==role_ids['early_stop']
    for name in ['B4_test.csv','B0_test.csv']:
        assert {r['image_id'] for r in rows(folder/name)}==qid
    assert {r['image_id'] for r in rows(folder/'B4_oof.csv')}==val
    markers=list(folder.rglob('stage_complete.json'))
    assert len(markers)==8
    for marker in markers:
        for name,entry in js(marker)['files'].items():
            path=marker.parent/name
            assert path.stat().st_size==entry['bytes'] and sha(path)==entry['sha256']
            stage_files+=1
    assert js(folder/'coarse_final/training_summary.json')['training_images']==len(dev)
    assert js(folder/'fine_final/training_summary.json')['training_images']==len(dev)
    fold_checks.append({'fold':f,'network_final_fit_images':len(dev),'early_stop_images':len(role_ids['early_stop']),
                       'calibration_images':len(cal),'outer_validation_images':len(val),
                       'test_ids_in_any_training_or_selection_role':[],'training_stages':8})

saved=js(ROOT/'outputs/submission_validation.json')
engine=ROOT/'work/current_notebook_review/submission_engine.py'
assert sha(engine)==saved['trained_engine_sha256']
submission=rows(ROOT/'outputs/submission.csv')
assert sha(ROOT/'outputs/submission.csv')==saved['submission_sha256']
assert [r['ImageID'] for r in submission]==[r['ImageID'] for r in template]
predictions=defaultdict(list)
for r in rows(RUN/'test_all_folds_B4.csv'):predictions[r['image_id']].append(r)
max_delta=0
for r in submission:
    ident,axis=r['ImageID'].split('_Fovea_');field='x' if axis=='X' else 'y'
    pp=predictions[ident]
    assert len(pp)==5 and {int(p['fold']) for p in pp}==set(range(5))
    value=sum(float(p[field]) for p in pp)/5
    delta=abs(value-float(r['value']));assert delta<0.00000051
    max_delta=max(max_delta,delta)
oof=rows(RUN/'oof_B4.csv')
assert {r['image_id'] for r in oof}==tid
assert not {r['image_id'] for r in oof}&qid
duplicates=[ids for ids in pixel_groups['train'].values() if len(ids)>1]
report={'status':'PASS','scope':'Actual delivered outputs/submission.csv and its saved training pipeline; no retraining performed.',
 'training_label_ids':'1-80','test_ids':'81-100','id_overlap':[],'physical_path_overlap':[],
 'fresh_rgb_hashes_verified':100,'exact_train_test_pixel_duplicates':[],
 'training_duplicate_groups':duplicates,'all_template_values_empty':True,
 'training_labels_match_original_csv':True,'test_manifest_has_no_label_columns':True,
 'fold_checks':fold_checks,'training_artifact_files_sha256_verified':stage_files,
 'delivered_code_matches_training_audit_hash':True,'delivered_submission_unchanged':True,
 'submission_sha256':sha(ROOT/'outputs/submission.csv'),'submission_rows':40,
 'submission_matches_B4_five_fold_mean_max_difference_px':max_delta,
 'oof_evaluated_only_on_80_training_ids':True,
 'test_usage':'Read RGB sizes/hashes for data integrity and clipping; frozen-model forward inference; mean ensemble and export. No test training, pseudo-label fitting, early stopping or calibration found.',
 'limitations':['No patient/subject IDs: patient-level independence cannot be established.','Exact RGB comparison does not rule out transformed or near-duplicate images.','This is a code, record, hash and output lineage audit, not an independent record of every historical training batch.']}
(ROOT/'outputs/test_leakage_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(report,ensure_ascii=True,indent=2))
