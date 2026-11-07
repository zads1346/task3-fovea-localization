from pathlib import Path
import hashlib, json, shutil, zipfile, ast
ROOT=Path(__file__).resolve().parents[2]
WORK=Path(__file__).resolve().parent
OUT=ROOT/'outputs/restored_20261027'
OUT.mkdir(parents=True,exist_ok=True)
files=[ROOT/'outputs/submission.csv', ROOT/'outputs/fovea_submission_20261025.ipynb',
 ROOT/'work/current_notebook_review/submission_engine.py',
 ROOT/'outputs/optimized_20261026/submission.csv', ROOT/'outputs/optimized_20261026/fovea_optimized_20261026.ipynb',
 ROOT/'work/optimization_20261026/engine.py']
backup=ROOT/'backups/before_restore_20261027.zip'
assert not backup.exists(), 'Preserve existing backup; do not overwrite.'
manifest={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
with zipfile.ZipFile(backup,'w',zipfile.ZIP_DEFLATED) as z:
 for p in files:z.write(p,p.relative_to(ROOT).as_posix())
 z.writestr('manifest.json',json.dumps(manifest,indent=2))
with zipfile.ZipFile(backup) as z:
 assert z.testzip() is None
 for p in files:assert hashlib.sha256(z.read(p.relative_to(ROOT).as_posix())).hexdigest()==manifest[str(p.relative_to(ROOT))]
shutil.copy2(ROOT/'outputs/submission.csv',OUT/'submission_restored_8p9.csv')
shutil.copy2(ROOT/'outputs/fovea_submission_20261025.ipynb',OUT/'baseline_8p9.ipynb')
source=(ROOT/'work/current_notebook_review/submission_engine.py').read_text(encoding='utf-8')
source=source[:source.index('if __name__ == "__main__":')]
source=source.replace('    REQUIRE_GPU: bool = True\n','    REQUIRE_GPU: bool = True\n    MODEL_SEED_OFFSET: int = 0  # Changes model/augmentation RNG only; all data splits remain unchanged.\n',1)
needle='    folder.mkdir(parents=True, exist_ok=True)\n    marker = folder / "stage_complete.json"'
replacement='''    folder.mkdir(parents=True, exist_ok=True)
    seed = int(seed) + int(cfg.MODEL_SEED_OFFSET)
    cols = [c for c in ("image_id", "group", "pixel_hash") if c in training]
    training[cols].to_csv(folder / "actual_training_inputs.csv", index=False)
    if validation is not None:
        validation[cols].to_csv(folder / "actual_early_stop_inputs.csv", index=False)
    if train_crop is not None:
        train_crop.to_csv(folder / "actual_training_crops.csv", index=False)
    marker = folder / "stage_complete.json"'''
assert source.count(needle)==1
source=source.replace(needle,replacement)
ast.parse(source)
(WORK/'engine.py').write_text(source,encoding='utf-8')
protocol={'date':'2026-10-27','baseline_score_user_reported':8.9,'rejected_score_user_reported':12.4,
 'score_direction':'lower_is_better','baseline_submission_sha256':hashlib.sha256((OUT/'submission_restored_8p9.csv').read_bytes()).hexdigest(),
 'candidate_weights':{'baseline':0.75,'additional_seed':0.25},'model_seed_offset':10000,
 'split_seed':2026,'changes':'Only model/augmentation random seed changed; same folds, holdouts, architecture, resolution, AMP, ROI, learning rates, early stop and B4 pipeline.',
 'promotion':'Candidate stays separately named. Baseline remains active until leaderboard evidence supports improvement.',
 'engine_sha256':hashlib.sha256((WORK/'engine.py').read_bytes()).hexdigest(),
 'backup':str(backup),'backup_hashes_verified':True,'prior_training_weights':'Retained in original version directories; not duplicated in this lightweight backup.'}
(OUT/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
print(json.dumps(protocol,indent=2))
