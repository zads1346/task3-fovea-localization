"""Prepare a new isolated experiment and preserve hashes of previous versions."""
from pathlib import Path
import ast,hashlib,json,shutil,time
ROOT=Path(__file__).resolve().parents[2]
WORK=Path(__file__).resolve().parent
OUT=ROOT/'outputs/dual_context_20261027'
BASE=ROOT/'work/restore_20261027/engine.py'
PARTS=[BASE,WORK/'model.py',WORK/'risk.py',WORK/'pipeline.py']
def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''): h.update(block)
    return h.hexdigest()
def js(path):return json.loads(Path(path).read_text(encoding='utf-8'))
assert digest(BASE)=='afb9077d1a30947902030cfba666c4c5c367e917597c58670edb16798fd733cd'
source='\n\n'.join(p.read_text(encoding='utf-8') for p in PARTS)
compile(source,'dual_context_engine','exec');ast.parse(source)
OUT.mkdir(parents=True,exist_ok=True)
if (OUT/'run_protocol.json').exists():raise RuntimeError('Training already initialized; do not reassemble frozen sources.')
(WORK/'engine.py').write_text(source,encoding='utf-8')
manifest={'version':'6.0.0-dual-context-20261027','engine_sha256':digest(WORK/'engine.py'),
          'parts':{p.relative_to(ROOT).as_posix():digest(p) for p in PARTS}}
(WORK/'source_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
for name,key in [('model_cpu_validation.json','model_source_sha256'),('risk_cpu_validation.json','risk_source_sha256')]:
    report=js(WORK/name);assert report['status']=='PASS'
    if key in report: assert report[key]==digest(WORK/('model.py' if name.startswith('model') else 'risk.py'))
baseline=ROOT/'outputs/restored_20261027/submission_candidate.csv'
assert digest(baseline)=='be3cba8f60f96325c7730480db2f740522e79ac550368984174a73396342176f'
shutil.copy2(baseline,OUT/'baseline_8p2.csv')
shutil.copy2(ROOT/'outputs/research82_precision_20261027/submission.csv',OUT/'previous_precision.csv')
backup=js(ROOT/'outputs/research82_20261027/backup_verification.json')
assert backup['status']=='PASS' and Path(backup['backup']).is_file()
shutil.copy2(ROOT/'outputs/research82_20261027/backup_verification.json',OUT/'existing_baseline_backup.json')
protected_path=OUT/'protected_previous.json'
if not protected_path.exists():
    started=time.perf_counter();paths=[]
    for path in (ROOT/'outputs').rglob('*'):
        if path.is_file() and OUT not in path.parents and path.suffix.lower() in ('.pt','.pth','.csv','.ipynb','.py'):
            paths.append(path)
    for directory in ('restore_20261027','research82','research82_precision','current_notebook_review','optimized_20261026'):
        parent=ROOT/'work'/directory
        if parent.exists():paths.extend(parent.rglob('*.py'))
    records={str(p.relative_to(ROOT).as_posix()):{'bytes':p.stat().st_size,'sha256':digest(p)} for p in sorted(set(paths))}
    protected_path.write_text(json.dumps({'status':'CAPTURED_BEFORE_NEW_TRAINING','files':records,
       'seconds':time.perf_counter()-started,'scope':'All existing output model/checkpoint/CSV/Notebook/Python files plus selected previous source trees; no old files modified.'},indent=2),encoding='utf-8')
cfg=js(ROOT/'outputs/restored_20261027/seed_10000/run_config.json')['config']
cfg.update(OUTPUT_DIR=str(OUT),SUBMISSION_PATH=str(OUT/'submission.csv'),SEED=2026,FINE_VARIANT='single',CONTEXT_RATIO=2.0,
           HEATMAP_SIGMA=2.5,FINE_MULTISCALE_TRAIN=False,SELECT_FINE_CROP_POLICY=False,FINE_MULTISCALE_EARLY_STOP=False,
           FINE_VALIDATION_ROI_FRACTIONS=[.15],FIXED_ROI_FRACTION=.15,ROI_MARGIN_PX=0.,REUSE_CHECKPOINTS=True)
(OUT/'prepared_config.json').write_text(json.dumps({'config':cfg},indent=2),encoding='utf-8')
planned={'status':'FROZEN_BEFORE_TRAINING','split_seeds':[2026,2027],'primary_new_fine_stages':20,
         'secondary_new_coarse_stages':25,'secondary_new_fine_stages':20,'total_new_stages':65,
         'architecture_control':'single and dual matched sigma1.5, local side .15, input256/hm64, deployment-scale early stop',
         'gate':'dual inner RMSE improvement >=0.5px AND >=2% relative to matched single; rejected primary fold stays original8.2',
         'accepted_primary_blend':'.8 original8.2 fold + .2 dual_risk fold','risk_constants_version':'1.0.0-fixed-four-policy-loog',
         'secondary':'fivefold audit only; no test inference and no final submission influence',
         'external_test_reference_used':False,'new_notebook_total_stages':145,'source_manifest':manifest}
(OUT/'planned_experiment.json').write_text(json.dumps(planned,indent=2),encoding='utf-8')
print(json.dumps({'status':'PASS','engine_sha256':manifest['engine_sha256'],'protected_files':len(js(protected_path)['files']),'output':str(OUT)},indent=2))
