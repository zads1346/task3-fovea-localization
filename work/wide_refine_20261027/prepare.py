"""Create an isolated, frozen experiment without modifying old artifacts."""
from pathlib import Path
import ast, hashlib, json, shutil, time

ROOT = Path(__file__).resolve().parents[2]
WORK = Path(__file__).resolve().parent
OUT = ROOT / 'outputs/wide_refine_20261027'
PREVIOUS = ROOT / 'outputs/dual_context_20261027'
BASE = ROOT / 'work/restore_20261027/engine.py'

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4*1024*1024), b''): h.update(block)
    return h.hexdigest()

def js(path): return json.loads(Path(path).read_text(encoding='utf-8'))

def main():
    assert sha(BASE) == 'afb9077d1a30947902030cfba666c4c5c367e917597c58670edb16798fd733cd'
    assert js(PREVIOUS/'validation_report.json')['status'] == 'PASS'
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT/'run_protocol.json').exists(): raise RuntimeError('Formal run already initialized. Frozen sources may not be rebuilt.')
    parts = [BASE, WORK/'pipeline.py']
    source = '\n\n'.join(p.read_text(encoding='utf-8') for p in parts)
    compile(source, 'wide_refine_engine', 'exec'); ast.parse(source)
    (WORK/'engine.py').write_text(source, encoding='utf-8')
    manifest = {'engine_sha256': sha(WORK/'engine.py'), 'parts': {p.relative_to(ROOT).as_posix(): sha(p) for p in parts},
                'previous_engine_sha256': sha(ROOT/'work/dual_context_20261027/engine.py')}
    assert manifest['previous_engine_sha256'] == '8403d5bd3ba4d5c728d1bac2ea61f475d6145bc0df5d8658197ffd47d4cf741e'
    (WORK/'source_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    (OUT/'source_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    baseline = PREVIOUS/'submission.csv'
    assert sha(baseline) == '7e40d7de452b984e2c7d899c210698fe2506311fb92d459ab10f2e632acdcc5c'
    shutil.copy2(baseline, OUT/'previous_submission.csv')
    shutil.copy2(PREVIOUS/'baseline_8p2.csv', OUT/'baseline_8p2.csv')
    oldcfg = js(PREVIOUS/'prepared_config.json')['config']
    for key in ('FINE_VARIANT', 'CONTEXT_RATIO'): oldcfg.pop(key)
    oldcfg.update(OUTPUT_DIR=str(OUT), SUBMISSION_PATH=str(OUT/'submission.csv'), SEED=2026,
                  FINE_INPUT_SIZE=384, FINE_HEATMAP_SIZE=96, HEATMAP_SIGMA=1.5,
                  FIXED_ROI_FRACTION=.25, ROI_MIN_FRACTION=.25, ROI_MARGIN_PX=0.,
                  ROI_FRACTIONS=[.25], FINE_VALIDATION_ROI_FRACTIONS=[.25],
                  FINE_MULTISCALE_TRAIN=False, SELECT_FINE_CROP_POLICY=False,
                  FINE_MULTISCALE_EARLY_STOP=False, REUSE_CHECKPOINTS=True)
    (OUT/'prepared_config.json').write_text(json.dumps({'config': oldcfg}, indent=2), encoding='utf-8')
    protection = OUT/'protected_previous.json'
    if not protection.exists():
        start = time.perf_counter()
        suffixes = {'.pt','.pth','.csv','.ipynb','.py','.md','.json','.zip'}
        paths = [p for p in (ROOT/'outputs').rglob('*') if p.is_file() and OUT not in p.parents and p.suffix.lower() in suffixes]
        paths += [p for p in (ROOT/'work').rglob('*') if p.is_file() and WORK not in p.parents and p.suffix.lower() in {'.py','.ipynb'} and 'gpu_test_deps' not in p.parts]
        files = {p.relative_to(ROOT).as_posix(): {'bytes': p.stat().st_size, 'sha256': sha(p)} for p in sorted(set(paths))}
        protection.write_text(json.dumps({'status':'CAPTURED_BEFORE_NEW_TRAINING', 'files':files,
                              'seconds':time.perf_counter()-start}, indent=2), encoding='utf-8')
    protocol = {
        'status':'FROZEN_BEFORE_TRAINING', 'split_seeds':[2026,2027], 'new_training_stages':30,
        'configuration':'single ResNet18, window .25 shortside, input384 heat96 sigma1.5, fixed scale, deployment-scale early stop',
        'per_fold_stages':['wide_select: model_fit vs early_stop','wide_dev: refit original development','wide_full: refit all non-outer images'],
        'gate':'WIDE_DEV vs frozen SINGLE on original calibration selection set: RMSE gain>=0.5px AND>=2%; P90 not worse; fixed before FULL refit',
        'submission':'accepted primary fold .75 previous submission fold + .25 WIDE_FULL; rejected retains previous; equal mean of five primary folds',
        'secondary':'fivefold training audit only, no test prediction, no submission influence',
        'risk':'No new learned fallback or uncertainty calibration. Original calibration becomes FULL training after architecture selection.',
        'ablation_limit':'DEV/FULL same epoch count and initialization; FULL has more images and optimizer updates, not a data-only causal comparison.',
        'reference_scores':'External reference scores permitted only after complete candidate freeze and verification; no test labels in training/model selection.',
        'previous_submission_sha256':sha(baseline), 'source_manifest':manifest}
    (OUT/'planned_experiment.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    print(json.dumps({'status':'PASS','output':str(OUT),'engine_sha256':manifest['engine_sha256'],
                      'protected_files':len(js(protection)['files'])}, indent=2))

if __name__ == '__main__': main()
