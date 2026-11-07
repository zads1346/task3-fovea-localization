from pathlib import Path
import hashlib,json,zipfile,shutil,time
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/research82_20261027'
OUT.mkdir(parents=True,exist_ok=True)
OLD=ROOT/'outputs/trained_submission_20261025_fixed';SEED=ROOT/'outputs/restored_20261027/seed_10000'
CURRENT=ROOT/'outputs/restored_20261027'
def digest_file(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()
files=[]
for base in [OLD,SEED]:
 for name in ['run_config.json','execution_report.json','environment.json','folds.csv','test_manifest.csv','oof_B4.csv','test_all_folds_B4.csv','test_ensemble_B4.csv']:
  files.append(base/name)
 for k in range(5):
  folder=base/f'fold_{k}'
  for name in ['complete.json','radius_calibration.json','fallback_rule.json','split_roles.csv','split_audit.json','fine_policy_selection.json','B4_oof.csv','B4_test.csv']:
   files.append(folder/name)
  for stage in ['coarse_final','fine_final']:
   for name in ['model.pt','stage_complete.json','training_summary.json','history.csv']:
    files.append(folder/stage/name)
files+=list(p for p in CURRENT.iterdir() if p.is_file())
files+=[ROOT/'work/restore_20261027/engine.py',ROOT/'work/restore_20261027/combine.py',ROOT/'work/current_notebook_review/submission_engine.py',ROOT/'outputs/submission.csv']
files+=list((ROOT/'outputs/fovea_pretrained_weights').iterdir())
files=sorted(set(files));assert all(p.is_file() for p in files)
target=ROOT/'backups/baseline_8p2_before_research_20261027.zip'
assert not target.exists(), 'Do not overwrite a backup.'
manifest={p.relative_to(ROOT).as_posix():{'bytes':p.stat().st_size,'sha256':digest_file(p)} for p in files}
started=time.perf_counter()
with zipfile.ZipFile(target,'x',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
 for p in files:z.write(p,p.relative_to(ROOT).as_posix())
 z.writestr('backup_manifest.json',json.dumps(manifest,indent=2))
with zipfile.ZipFile(target) as z:
 for name,expected in manifest.items():
  h=hashlib.sha256()
  with z.open(name) as f:
   for block in iter(lambda:f.read(4*1024**2),b''):h.update(block)
  assert h.hexdigest()==expected['sha256'],name
report={'status':'PASS','backup':str(target),'bytes':target.stat().st_size,'files':len(manifest),
 'trained_final_models':20,'all_archived_bytes_verified':True,'seconds':time.perf_counter()-started,
 'submission_sha256':digest_file(CURRENT/'submission_candidate.csv'),'score_user_reported':8.2}
(OUT/'backup_verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
shutil.copy2(CURRENT/'submission_candidate.csv',OUT/'submission_baseline_8p2.csv')
(OUT/'baseline_identity.json').write_text(json.dumps({'score':8.2,'source':'User reported Kaggle score; not read by API.',
 'baseline_submission':str(CURRENT/'submission_candidate.csv'),'sha256':report['submission_sha256'],
 'components':[{'run':str(OLD),'weight':.75},{'run':str(SEED),'weight':.25}]},indent=2),encoding='utf-8')
print(json.dumps(report,indent=2),flush=True)
