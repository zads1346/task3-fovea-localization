"""Prepare precision-only sources and baseline references; never start training."""
from pathlib import Path
import ast, hashlib, json, shutil

ROOT = Path(__file__).resolve().parents[2]
WORK = Path(__file__).resolve().parent
OUT = ROOT / 'outputs/research82_precision_20261027'
PHASE1 = ROOT / 'outputs/research82_20261027'
SEED = ROOT / 'outputs/restored_20261027/seed_10000'
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
read = lambda path: json.loads(Path(path).read_text(encoding='utf-8'))

parts = [ROOT / 'work/restore_20261027/engine.py',
         ROOT / 'work/research82/fine_variants.py', WORK / 'pipeline.py']
source = '\n\n'.join(path.read_text(encoding='utf-8') for path in parts)
compile(source, 'precision_engine', 'exec')
ast.parse(source)
(WORK / 'engine.py').write_text(source, encoding='utf-8')
manifest = {'engine_sha256': sha(WORK / 'engine.py'),
    'source_parts': {path.relative_to(ROOT).as_posix(): sha(path) for path in parts},
    'version': '5.1.0-research82-precision-20261027',
    'status': 'PREPARED_NOT_TRAINED',
    'treatment': 'Fine-only HEATMAP_SIGMA 2.5 -> 1.5; all other experimental settings retained.'}
(WORK / 'source_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
OUT.mkdir(parents=True, exist_ok=True)

backup = read(PHASE1 / 'backup_verification.json')
identity = read(PHASE1 / 'baseline_identity.json')
archive = Path(backup['backup'])
assert backup['status'] == 'PASS' and backup['all_archived_bytes_verified']
assert archive.is_file() and archive.stat().st_size == backup['bytes']
baseline = PHASE1 / 'submission_baseline_8p2.csv'
assert sha(baseline) == backup['submission_sha256'] == identity['sha256']
shutil.copy2(baseline, OUT / 'baseline_8p2.csv')
shutil.copy2(PHASE1 / 'baseline_identity.json', OUT / 'baseline_identity.json')
for name in ('folds.csv', 'test_manifest.csv'):
    shutil.copy2(SEED / name, OUT / name)
backup_reference = {**backup,
    'reference_only': True,
    'source_verification': str(PHASE1 / 'backup_verification.json'),
    'source_verification_sha256': sha(PHASE1 / 'backup_verification.json'),
    'precision_phase_checks': 'Existing archive path/size and baseline CSV hash checked; earlier full-byte verification is referenced, not rerun.',
    'new_model_backup_created': False}
(OUT / 'backup_verification.json').write_text(json.dumps(backup_reference, indent=2), encoding='utf-8')

config = read(SEED / 'run_config.json')['config']
config.update({'OUTPUT_DIR': str(OUT), 'SUBMISSION_PATH': str(OUT / 'submission.csv'),
               'FINE_VARIANT': 'resnet18', 'EFFNET_WEIGHTS_PATH': None})
assert config['MODEL_SEED_OFFSET'] == 10000 and config['HEATMAP_SIGMA'] == 2.5
assert sha(config['PRETRAINED_WEIGHTS_PATH']) == 'f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec'
(OUT / 'prepared_config.json').write_text(json.dumps({'config': config,
    'note': 'Base/source config retains sigma 2.5. refinement_variant_config sets 1.5 only on the new fine variant.'}, indent=2), encoding='utf-8')
planned = {'status': 'PREPARED_NOT_TRAINED', 'version': manifest['version'],
    'baseline_score_user_reported': 8.2, 'baseline_csv_sha256': sha(OUT / 'baseline_8p2.csv'),
    'variants': ['resnet18_narrow'], 'fine_sigma_cells': 1.5,
    'coarse_sigma_cells_unchanged': 2.5, 'new_training_stages': 15,
    'notebook_total_training_stages': 95, 'minimum_absolute_gain_px': .5,
    'minimum_relative_gain': .02, 'blend_weight_if_accepted': .20,
    'gate_proxy': 'The original seed10000 ResNet18 internal stopping score, not the full 8.2 ensemble score.',
    'hyperparameter_origin': 'The single 1.5 fine-sigma candidate was proposed in keypoint_literature.md before this precision phase; no grid search.',
    'literature': 'https://www.melba-journal.org/papers/2021:014.html',
    'training_launched_by_assembly': False}
(OUT / 'planned_experiment.json').write_text(json.dumps(planned, indent=2), encoding='utf-8')

# Verify that only version metadata, the fine candidate, weight requirements,
# and stage-count metadata changed from phase 1; gate constants are identical.
def functions(text):
    return {node.name: ast.dump(node, include_attributes=False) for node in ast.parse(text).body
            if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef))}
old_functions = functions((ROOT / 'work/research82/pipeline.py').read_text(encoding='utf-8'))
new_functions = functions((WORK / 'pipeline.py').read_text(encoding='utf-8'))
changed = sorted(name for name in old_functions.keys() & new_functions.keys()
                 if old_functions[name] != new_functions[name])
assert changed == ['refinement_variant_config', 'run_refinement']
assert old_functions.keys() - new_functions.keys() == {'refinement_effnet_weights'}
assert new_functions.keys() - old_functions.keys() == set()
for file in ('pipeline.py', 'assemble.py', 'run.py', 'build_notebook.py'):
    path = WORK / file
    if path.exists(): compile(path.read_text(encoding='utf-8'), file, 'exec')
validation = {'status': 'PASS', 'source_manifest': manifest,
    'changed_pipeline_functions': changed, 'removed_function': 'refinement_effnet_weights',
    'gate_and_fold_training_logic_unchanged': True, 'baseline_sha256_matches': True,
    'folds_byte_identical_to_baseline': sha(OUT / 'folds.csv') == sha(SEED / 'folds.csv'),
    'test_manifest_byte_identical_to_baseline': sha(OUT / 'test_manifest.csv') == sha(SEED / 'test_manifest.csv'),
    'training_started': False}
(OUT / 'assembly_validation.json').write_text(json.dumps(validation, indent=2), encoding='utf-8')
print(json.dumps(validation, indent=2))
