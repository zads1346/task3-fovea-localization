"""Offline CPU Notebook audit; explicit training stubs are NOT real results."""
from pathlib import Path
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['MPLBACKEND'] = 'Agg'
import ast, hashlib, json, shutil, sys, types, urllib.request

ROOT = Path(__file__).resolve().parents[2]
WORK = Path(__file__).resolve().parent
OUT = ROOT / 'outputs/dual_context_20261027'
SIM = WORK / 'notebook_cpu_sim'
sys.path.insert(0, str(ROOT / 'work/gpu_test_deps'))
import torch
torch.set_num_threads(2)
assert not torch.cuda.is_available()


def main():
    attempts = []
    def no_network(*args, **kwargs):
        attempts.append(str(args[0]) if args else 'urlopen')
        raise AssertionError('No network is allowed in the offline Notebook audit.')
    urllib.request.urlopen = no_network
    path = OUT / 'fovea_dual_context_kaggle.ipynb'
    notebook_bytes = path.read_bytes()
    notebook = json.loads(notebook_bytes)
    cells = [cell['source'] for cell in notebook['cells'] if cell['cell_type'] == 'code']
    cells = [''.join(cell) if isinstance(cell, list) else cell for cell in cells]
    source = cells[1]
    assert source == (WORK / 'engine.py').read_text(encoding='utf-8')
    module = types.ModuleType('dual_notebook_cpu_audit')
    module.__file__ = str(WORK / 'engine.py')
    sys.modules[module.__name__] = module
    exec(compile(source, module.__file__, 'exec'), module.__dict__)
    m = module
    config_tree = ast.parse(cells[2])
    first = config_tree.body[0]
    assert isinstance(first, ast.Assign) and first.targets[0].id == 'BASE_CFG0'
    base_config = ast.literal_eval(first.value.keywords[0].value)
    original_cfg = json.loads((ROOT / 'outputs/restored_20261027/seed_10000/run_config.json').read_text())['config']
    portable_changes = {'INPUT_ROOT', 'DATA_DIR', 'LABEL_CSV', 'SAMPLE_CSV', 'PRETRAINED_WEIGHTS_PATH',
                        'OUTPUT_DIR', 'SUBMISSION_PATH', 'MODEL_SEED_OFFSET', 'FINE_VARIANT', 'CONTEXT_RATIO'}
    assert {key for key in set(base_config) | set(original_cfg) if base_config.get(key) != original_cfg.get(key)} <= portable_changes
    assert base_config['HEATMAP_SIGMA'] == 2.5 and base_config['MODEL_SEED_OFFSET'] == 0
    assert base_config['SELECT_FINE_CROP_POLICY'] and base_config['FINE_MULTISCALE_EARLY_STOP']
    assert base_config['FINE_MULTISCALE_TRAIN'] and base_config['FINE_VALIDATION_ROI_FRACTIONS'] == [.2,.3,.4]
    SIM.mkdir(parents=True, exist_ok=True)
    cfg = m.DualConfig(**base_config)
    local_cfg = m.replace(cfg, INPUT_ROOT=str(ROOT / 'work/real_competition_data'), DATA_DIR=None,
                          OUTPUT_DIR=str(SIM / 'data_discovery'),
                          PRETRAINED_WEIGHTS_PATH=str(ROOT / 'outputs/fovea_pretrained_weights'))
    train, test, sample, layout, metadata = m.prepare_tables(local_cfg)
    assert len(train) == 80 and len(test) == 20 and len(sample) == 40
    folded = m.assign_folds(train, local_cfg)
    source_folds = m.dc_align(m.dc_read(ROOT / 'outputs/restored_20261027/seed_10000/folds.csv'), folded.image_id)
    for column in ('fold', 'group', 'pixel_hash', 'x', 'y'):
        assert m.np.array_equal(folded[column], source_folds[column]), column
    unpacked_weight = m.resolve_research_weights(local_cfg)
    assert m.research_file_digest(unpacked_weight) == m.RESEARCH_WEIGHT_SHA256
    zip_cfg = m.replace(local_cfg, OUTPUT_DIR=str(SIM / 'zip_verification'),
                        PRETRAINED_WEIGHTS_PATH=str(ROOT / 'outputs/fovea_pretrained_weights.zip'))
    zip_weight = m.resolve_research_weights(zip_cfg)
    assert m.research_file_digest(zip_weight) == m.RESEARCH_WEIGHT_SHA256
    assert not attempts

    calls = []
    captured = {}
    def stub_baseline(config):
        calls.append(['baseline', config.MODEL_SEED_OFFSET])
        source_run = ROOT / ('outputs/trained_submission_20261025_fixed' if config.MODEL_SEED_OFFSET == 0
                             else 'outputs/restored_20261027/seed_10000')
        destination = Path(config.OUTPUT_DIR)
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_run / 'test_ensemble_B4.csv', destination / 'test_ensemble_B4.csv')
        Path(config.SUBMISSION_PATH).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_run / 'submission_B4.csv', config.SUBMISSION_PATH)
        return {'config': config, 'test': test.copy()}
    def stub_dual(config, old_run, source_run, baseline_csv):
        calls.append(['dual_context', config.MODEL_SEED_OFFSET])
        assert Path(old_run).name == 'baseline_seed0'
        assert Path(source_run).name == 'baseline_seed10000'
        assert Path(baseline_csv).name == 'submission_retrained_8p2_method.csv'
        captured['dual_config'] = m.asdict(config)
        expected_reference = m.dc_blend(m.dc_read(Path(old_run) / 'test_ensemble_B4.csv'),
                                        m.dc_read(Path(source_run) / 'test_ensemble_B4.csv'), .25)
        verify_csv = SIM / 'verify_reference.csv'
        m.dc_export(baseline_csv, expected_reference, test, verify_csv)
        assert verify_csv.read_bytes() == Path(baseline_csv).read_bytes()
        destination = Path(config.OUTPUT_DIR)
        destination.mkdir(parents=True, exist_ok=True)
        # A known placeholder exercises export identity only, not candidate quality.
        shutil.copyfile(baseline_csv, destination / 'submission.csv')
        m.json_save({'status': 'COMPLETE', 'completed_folds': 10, 'new_training_stages': 65,
                     'scope': 'EXPLICIT TEST STUB; NOT A TRAINING RESULT'}, destination / 'execution_report.json')
        return {'summary': m.pd.DataFrame([{'scope': 'EXPLICIT TEST STUB'}]), 'submission': m.pd.read_csv(baseline_csv)}
    m.run_research = stub_baseline
    m.run_dual_context = stub_dual
    m.display = lambda *args, **kwargs: None
    # Map writable Kaggle output locations into this isolated CPU simulation.
    def remap(text):
        return text.replace('/kaggle/working', SIM.as_posix())
    initial = remap(ast.get_source_segment(cells[2], first))
    exec(initial, m.__dict__)
    # Weight/data checks above were real CPU operations. Training below is stubbed.
    for source_cell in cells[3:]:
        exec(compile(remap(source_cell), 'notebook_simulated_orchestration', 'exec'), m.__dict__)
    assert calls == [['baseline', 0], ['baseline', 10000], ['dual_context', 10000]]
    prepared = json.loads((OUT / 'prepared_config.json').read_text())['config']
    actual = json.loads(json.dumps(captured['dual_config']))  # tuple/list normalize
    exempt = {'INPUT_ROOT', 'DATA_DIR', 'LABEL_CSV', 'SAMPLE_CSV', 'OUTPUT_DIR',
              'SUBMISSION_PATH', 'PRETRAINED_WEIGHTS_PATH'}
    changes = {key for key in set(prepared)|set(actual) if prepared.get(key) != actual.get(key)}
    assert changes <= exempt, changes-exempt
    assert (SIM / 'submission.csv').read_bytes() == (SIM / 'dual_context_results/submission.csv').read_bytes()
    assert path.read_bytes() == notebook_bytes
    report = {'status': 'PASS', 'notebook_sha256': hashlib.sha256(notebook_bytes).hexdigest(),
              'engine_sha256': m.research_file_digest(WORK / 'engine.py'),
              'scope': 'CPU data/weight checks plus explicit training stubs for orchestration; not GPU training or new predictions.',
              'gpu_available': False, 'network_attempts': len(attempts),
              'competition_data_discovery': {'train_images': len(train), 'test_images': len(test), 'submission_rows': len(sample)},
              'original_fold_assignment_exact': True, 'baseline_hyperparameters_restored': True,
              'new_configuration_matches_local_prepared_except_paths': True,
              'offline_resnet18_directory_discovery_and_sha': True,
              'offline_original_zip_and_sha': True,
              'orchestration_stub_call_order': calls,
              'baseline_stage_count': 80, 'new_stage_count': 65, 'total_training_stages': 145,
              'reference_from_actual_baseline_prediction_files': True,
              'final_export_byte_identity': True, 'notebook_or_training_source_modified_by_audit': False,
              'blocking_issues': []}
    (OUT / 'notebook_terminal_audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
