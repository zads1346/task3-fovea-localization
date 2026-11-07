"""Independent audit and 25-network reload after the precision run completes.

Importing this module does not run inference. Run it only after run.py finishes.
The original submissions, checkpoints and training records are read-only here.
"""
from pathlib import Path
from dataclasses import replace
import json
import math
import time
import traceback
import zipfile

# run.py establishes the exact local dependency path and environment, but its
# __main__ guard prevents training when imported.
from run import ROOT, WORK, OUT, OLD, SEED, m, cfg
import numpy as np
import pandas as pd
import torch

TOLERANCE_PX = 1e-3
VARIANTS = ('resnet18_narrow',)
METHODS = ('BASELINE8P2',) + VARIANTS + ('GUARDED20',)


def js(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def frame(path):
    return pd.read_csv(path, dtype={'image_id': str, 'group': str})


def require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def close(actual, expected, label, atol=1e-9):
    require(math.isfinite(float(actual)) and math.isfinite(float(expected)), f'{label}: nonfinite value')
    require(abs(float(actual) - float(expected)) <= atol, f'{label}: {actual!r} != {expected!r}')


def compare_configs(a, b, allowed, label):
    changes = {key for key in set(a) | set(b) if a.get(key) != b.get(key)}
    require(changes <= set(allowed), f'{label}: unexpected config changes {sorted(changes-set(allowed))}')
    return sorted(changes)


def verify_files(folder, manifest):
    # The base helper validates relative paths, lengths and streaming SHA256.
    m.research_verify_cache(folder, manifest)


def coordinate_difference(actual, saved, label):
    require(actual.image_id.is_unique and saved.image_id.is_unique, f'{label}: repeated IDs')
    require(set(actual.image_id) == set(saved.image_id), f'{label}: ID sets differ')
    expected = saved.set_index('image_id').loc[actual.image_id, ['x', 'y']].to_numpy(float)
    values = actual[['x', 'y']].to_numpy(float)
    require(np.isfinite(values).all() and np.isfinite(expected).all(), f'{label}: nonfinite coordinates')
    delta = float(np.max(np.abs(values - expected)))
    require(delta < TOLERANCE_PX, f'{label}: max difference {delta:.12g}px exceeds {TOLERANCE_PX}px')
    return delta


def blend(a, b, weight):
    # Independent coordinate implementation; do not call the producer's blend helper.
    require(a.image_id.is_unique and b.image_id.is_unique, 'blend: duplicate IDs')
    require(set(a.image_id) == set(b.image_id), 'blend: ID mismatch')
    result = a[['image_id', 'x', 'y']].copy()
    right = b.set_index('image_id').loc[result.image_id, ['x', 'y']].to_numpy(float)
    result[['x', 'y']] = result[['x', 'y']].to_numpy(float) * (1.0-weight) + right * weight
    return result


def metrics(pred):
    residual = pred[['x', 'y']].to_numpy(float) - pred[['x_true', 'y_true']].to_numpy(float)
    require(np.isfinite(residual).all(), 'OOF residual contains nonfinite values')
    distance = np.hypot(residual[:, 0], residual[:, 1])
    return {'rmse': float(np.sqrt(np.mean(residual**2))),
            'median_distance_px': float(np.median(distance)),
            'p90_distance_px': float(np.quantile(distance, .9)),
            'max_distance_px': float(np.max(distance))}


def verify_actual_inputs(path, expected, train_by_id, label):
    actual = frame(path)
    require(actual.image_id.is_unique, f'{label}: repeated training IDs')
    require(actual.image_id.tolist() == expected.image_id.tolist(), f'{label}: IDs/order differ from intended role')
    truth = train_by_id.loc[actual.image_id]
    for column in ('group', 'pixel_hash'):
        require(np.array_equal(actual[column].to_numpy(), truth[column].to_numpy()), f'{label}: wrong {column}')
    return actual


def audit_records(protocol, train, test, backup_manifest):
    source_cfg = js(SEED/'run_config.json')['config']
    original_cfg = js(OLD/'run_config.json')['config']
    base_changes = compare_configs(original_cfg, source_cfg,
        {'OUTPUT_DIR', 'SUBMISSION_PATH', 'MODEL_SEED_OFFSET'}, 'two baseline configs')
    require(original_cfg.get('MODEL_SEED_OFFSET', 0) == 0, 'Original seed offset must be zero')
    require(source_cfg['MODEL_SEED_OFFSET'] == 10000, 'Source seed offset changed')
    new_cfg = protocol['config']
    new_changes = compare_configs(source_cfg, new_cfg,
        {'OUTPUT_DIR', 'SUBMISSION_PATH', 'FINE_VARIANT', 'EFFNET_WEIGHTS_PATH'}, 'new protocol config')
    require(new_cfg['FINE_VARIANT'] == 'resnet18', 'Protocol base variant must remain ResNet18')
    close(source_cfg['HEATMAP_SIGMA'], 2.5, 'Inherited reference Gaussian sigma')
    close(new_cfg['HEATMAP_SIGMA'], 2.5, 'Protocol base Gaussian sigma')
    for key in ('TTA', 'FINE_MULTISCALE_EARLY_STOP', 'FINE_VALIDATION_ROI_FRACTIONS',
                'FIXED_ROI_FRACTION', 'ROI_MIN_FRACTION', 'ROI_MAX_FRACTION', 'ROI_MARGIN_PX',
                'ROI_COVERAGE', 'COORDINATE_ORIGIN', 'FINE_INPUT_SIZE', 'BATCH_SIZE',
                'COORD_LOSS_WEIGHT', 'HEATMAP_LOSS_WEIGHT', 'COORD_SCALE'):
        require(new_cfg[key] == source_cfg[key], f'Selection/deployment comparator changed: {key}')
    close(protocol['blend_weight'], .20, 'protocol blend weight')
    close(protocol['minimum_absolute_gain_px'], .5, 'absolute gate')
    close(protocol['minimum_relative_gain'], .02, 'relative gate')
    require(tuple(protocol['variants']) == VARIANTS, 'Variant order changed')
    train_by_id = train.set_index('image_id')
    selections = []; audits = []; stages = 0; old_models = 0; source_files = 0
    for fold in range(5):
        folder = OUT/f'fold_{fold}'; source = SEED/f'fold_{fold}'
        completed = js(folder/'complete.json')
        require(completed['status'] == 'COMPLETE' and completed['fold'] == fold, f'Fold {fold} incomplete')
        roles = frame(folder/'split_roles.csv')
        pd.testing.assert_frame_equal(roles, frame(source/'split_roles.csv'))
        pd.testing.assert_frame_equal(roles, frame(OLD/f'fold_{fold}'/'split_roles.csv'))
        require(roles.image_id.is_unique and set(roles.image_id) == set(train.image_id), 'Invalid role partition')
        require((roles.groupby('group').role.nunique() == 1).all(), 'Group crosses roles')
        require(set(roles.role) == {'model_fit', 'early_stop', 'calibration_only', 'outer_validation'}, 'Unknown roles')
        def get_role(name):
            ids = set(roles.loc[roles.role == name, 'image_id'])
            return train.loc[train.image_id.isin(ids)].reset_index(drop=True)
        fitting = get_role('model_fit'); stopping = get_role('early_stop')
        calibration = get_role('calibration_only'); outer = get_role('outer_validation')
        development = train.loc[train.image_id.isin(set(fitting.image_id) | set(stopping.image_id))].reset_index(drop=True)
        require(set(outer.image_id) == set(train.loc[train.fold == fold, 'image_id']), 'Wrong outer fold')
        excluded = set(calibration.image_id) | set(outer.image_id) | set(test.image_id)
        require(not set(development.image_id) & excluded, 'Development overlaps held-out/test roles')
        lineage = js(folder/'coarse_lineage.json')
        manifest = js(source/'complete.json')['files']
        require(Path(lineage['inherited_source']).resolve() == source.resolve(), 'Wrong inherited source')
        required = {'coarse_final/model.pt', 'coarse_select/model.pt', 'calibration_coarse.csv',
                    'radius_calibration.json', 'B0_oof.csv', 'B0_test.csv', 'fine_policy_selection.json',
                    'fine_final/actual_training_crops.csv', 'split_roles.csv', 'split_audit.json',
                    'crop_crossfit/crossfit_audit.json'}
        require(required <= set(lineage['verified_source_files']), 'Incomplete coarse lineage')
        for name, entry in lineage['verified_source_files'].items():
            require(entry == manifest[name], f'Source manifest mismatch: fold {fold}/{name}')
        verify_files(source, lineage['verified_source_files']); source_files += len(lineage['verified_source_files'])
        crop = frame(source/'fine_final/actual_training_crops.csv')
        require(crop.image_id.is_unique and set(crop.image_id) == set(development.image_id), 'Wrong crop membership')
        for inner in js(source/'crop_crossfit/crossfit_audit.json'):
            require(not set(inner['fit_ids']) & set(inner['held_ids']), 'Crossfit image overlap')
            require(not set(inner['fit_groups']) & set(inner['held_groups']), 'Crossfit group overlap')
            require((set(inner['fit_ids']) | set(inner['held_ids'])) <= set(fitting.image_id), 'Crossfit leaks nonfitting images')
        reference_policy = js(source/'fine_policy_selection.json')
        require(set(reference_policy['stopping_image_ids']) == set(stopping.image_id), 'Reference proxy uses different stopping IDs')
        proxy = float(reference_policy['selected']['best_rmse'])
        close(lineage['baseline_proxy_inner_score'], proxy, 'lineage proxy')
        require(lineage['proxy_is_8p2_ensemble_score'] is False, 'Proxy incorrectly described as ensemble')
        candidate_records = []
        for variant in VARIANTS:
            destination = folder/variant
            variant_cfg = js(destination/'variant_config.json')['config']
            changes = compare_configs(new_cfg, variant_cfg,
                {'OUTPUT_DIR', 'FINE_VARIANT', 'FINE_HEATMAP_SIZE', 'HEATMAP_SIGMA'}, f'{fold}/{variant} config')
            require(variant_cfg['FINE_VARIANT'] == 'resnet18', 'Narrow target must retain the original ResNet18 architecture')
            expected_heatmap = 64
            expected_sigma = 1.5
            require(variant_cfg['FINE_HEATMAP_SIZE'] == expected_heatmap, 'Wrong heatmap resolution')
            close(variant_cfg['HEATMAP_SIGMA'], expected_sigma, 'Physical Gaussian sigma')
            policy = js(destination/'policy_selection.json')
            require(policy['selection_role'] == 'inner_early_stop_only', 'Wrong policy selection role')
            require(set(policy['stopping_ids']) == set(stopping.image_id), 'Candidate stopping IDs changed')
            require([p['policy'] for p in policy['candidates']] == ['fixed', 'multiscale'], 'Unexpected policy candidates/order')
            selected = min(policy['candidates'], key=lambda p: p['best_rmse'])
            require(policy['selected'] == selected, 'Recorded policy is not the inner winner')
            final_cfg = js(destination/'final_config.json')['config']
            compare_configs(variant_cfg, final_cfg, {'FINE_MULTISCALE_TRAIN'}, 'final config')
            require(final_cfg['FINE_MULTISCALE_TRAIN'] == selected['multiscale'], 'Final policy mismatch')
            for stage in ('fine_select_fixed', 'fine_select_multiscale', 'fine_final'):
                stage_dir = destination/stage; stages += 1
                stage_record = js(stage_dir/'stage_complete.json')
                verify_files(stage_dir, stage_record['files'])
                require({'model.pt', 'history.csv', 'training_summary.json'} <= set(stage_record['files']), 'Incomplete stage manifest')
                info = js(stage_dir/'training_summary.json'); history = pd.read_csv(stage_dir/'history.csv')
                require(info['successful_optimizer_steps'] > 0, 'No successful optimizer updates')
                require(int(history.successful_optimizer_steps.sum()) == info['successful_optimizer_steps'], 'Update count differs')
                require(np.isfinite(history.train_loss.to_numpy(float)).all(), 'Nonfinite training loss')
                require(info['validation_tta'] == source_cfg['TTA'], 'Stage TTA differs from proxy')
                require(info['schedule_horizon_epochs'] == source_cfg['FINE_EPOCHS'], 'Schedule horizon changed')
                expected = development if stage == 'fine_final' else fitting
                actual = verify_actual_inputs(stage_dir/'actual_training_inputs.csv', expected, train_by_id, f'{fold}/{variant}/{stage}')
                require(not set(actual.image_id) & excluded, 'Training consumes held-out/test images')
                require(info['training_images'] == len(expected), 'Training image count mismatch')
                actual_crops = frame(stage_dir/'actual_training_crops.csv')
                expected_crops = crop.loc[crop.image_id.isin(expected.image_id)].reset_index(drop=True)
                cols = ['image_id', 'crop_x', 'crop_y', 'crop_side']
                pd.testing.assert_frame_equal(actual_crops[cols].reset_index(drop=True), expected_crops[cols],
                                              check_exact=False, atol=1e-10, rtol=1e-12)
                if stage == 'fine_final':
                    require(info['fixed_epochs'] is True, 'Final stage unexpectedly performs early stopping')
                    require(info['best_epoch'] == selected['best_epoch'] == len(history), 'Final epoch count mismatch')
                    require(not (stage_dir/'actual_early_stop_inputs.csv').exists(), 'Final refit has stale early-stop inputs')
                else:
                    stop = verify_actual_inputs(stage_dir/'actual_early_stop_inputs.csv', stopping, train_by_id, 'stopping inputs')
                    require(not set(actual.group) & set(stop.group), 'Training/stopping group overlap')
                    require(info['fixed_epochs'] is False, 'Selection stage marked final')
                    require(np.isfinite(history.val_rmse.to_numpy(float)).all(), 'Nonfinite validation score')
                    best_row = history.loc[history.val_rmse.idxmin()]
                    close(info['best_rmse'], best_row.val_rmse, 'history best score', atol=1e-8)
                    require(info['best_epoch'] == int(best_row.epoch), 'Wrong selected epoch')
                    option = next(p for p in policy['candidates'] if p['policy'] == stage.removeprefix('fine_select_'))
                    close(option['best_rmse'], info['best_rmse'], 'policy score')
                    require(option['best_epoch'] == info['best_epoch'], 'Policy epoch mismatch')
            fine_cal = frame(destination/'calibration_fine.csv')
            require(fine_cal.image_id.is_unique and set(fine_cal.image_id) == set(calibration.image_id), 'Calibration fine membership differs')
            candidate_records.append({'variant': variant, **selected,
                'model_sha256': m.research_file_digest(destination/'fine_final/model.pt')})
        selection = js(folder/'selection.json')
        require(selection == completed['selection'], 'Fold selection record differs from completion record')
        require(selection['candidates'] == candidate_records, 'Candidate summary differs from actual inner selection')
        winner = min(candidate_records, key=lambda p: p['best_rmse'])
        gain = proxy - winner['best_rmse']
        accepted = bool(gain >= .5 and gain >= .02 * proxy)
        require(selection['selected'] == winner['variant'], 'Wrong architecture selected')
        require(selection['proxy_is_8p2_ensemble'] is False, 'Proxy scope incorrectly stated')
        close(selection['proxy_resnet18_inner_rmse'], proxy, 'selection proxy')
        close(selection['gain_px'], gain, 'selection gain')
        require(selection['accepted'] is accepted, 'Gate decision cannot be reproduced from inner scores')
        close(selection['blend_weight'], .2 if accepted else 0, 'per-fold blend weight')
        selections.append(selection)
        for run in (OLD, SEED):
            run_folder = run/f'fold_{fold}'
            old_manifest = js(run_folder/'complete.json')['files']
            verify_files(run_folder, {name: old_manifest[name] for name in ('B4_oof.csv', 'B4_test.csv')})
            for name in ('complete.json', 'radius_calibration.json', 'fallback_rule.json',
                         'split_roles.csv', 'fine_policy_selection.json', 'B4_oof.csv', 'B4_test.csv'):
                path = run_folder/name
                historical = backup_manifest[path.relative_to(ROOT).as_posix()]
                require(path.stat().st_size == historical['bytes'] and
                        m.research_file_digest(path) == historical['sha256'],
                        f'Baseline source record changed since backup: {path}')
            for stage in ('coarse_final', 'fine_final'):
                path = run_folder/stage/'model.pt'
                historical = backup_manifest[path.relative_to(ROOT).as_posix()]
                digest = m.research_file_digest(path)
                require(path.stat().st_size == historical['bytes'] and digest == historical['sha256'], 'Old final model changed since backup')
                require(digest == old_manifest[f'{stage}/model.pt']['sha256'], 'Old model completion hash differs')
                old_models += 1
        audits.append({'fold': fold, 'new_training_stages': 3, 'fitting_images': len(fitting),
                       'stopping_images': len(stopping), 'calibration_images': len(calibration),
                       'outer_images': len(outer), 'test_or_held_out_training_overlap': 0,
                       'accepted': accepted, 'selected_variant': winner['variant'], 'inner_gain_px': gain})
        print(f'Audit fold {fold+1}/5 PASS; gate={accepted}, winner={winner["variant"]}', flush=True)
    require(stages == 15 and old_models == 20, 'Incomplete model/stage audit')
    require(len(list(OUT.glob('fold_*/*/fine_*/actual_training_inputs.csv'))) == 15, 'Unexpected training-input manifest count')
    return {'stages': stages, 'old_models_unchanged': old_models, 'source_files_verified': source_files,
            'baseline_config_changes': base_changes, 'new_protocol_config_changes': new_changes,
            'folds': audits}, selections


def reload_config(data):
    loaded = type(cfg)(**data)
    loaded.PRETRAINED = False
    loaded.PRETRAINED_WEIGHTS_PATH = None
    if hasattr(loaded, 'EFFNET_WEIGHTS_PATH'):
        loaded.EFFNET_WEIGHTS_PATH = None
    return loaded


def reload_predictions(test, selections):
    checks = []; counts = 0; all_results = {name: [] for name in METHODS}
    for fold, selection in enumerate(selections):
        sources = {}
        for label, run in (('original', OLD), ('seed10000', SEED)):
            folder = run/f'fold_{fold}'
            config = reload_config(js(run/'run_config.json')['config'])
            m.seed_everything(config.SEED)
            coarse = m.research_build_model(config, 'coarse', 'cuda')
            m.load_model_checkpoint(coarse, folder/'coarse_final/model.pt', 'cuda'); counts += 1
            cp = m.predict_research(coarse, test, config, 'cuda', 'coarse')
            m._research_cleanup(coarse); del coarse
            delta_c = coordinate_difference(cp, frame(folder/'B0_test.csv'), f'{label} coarse fold {fold}')
            radius = js(folder/'radius_calibration.json')
            roi = m.make_roi_spec(cp, test, config, mode='calibrated', calibration=radius)
            fine = m.research_build_model(config, 'fine', 'cuda')
            m.load_model_checkpoint(fine, folder/'fine_final/model.pt', 'cuda'); counts += 1
            fp = m.predict_research(fine, test, config, 'cuda', 'fine', roi)
            bp = m.apply_fallback(cp, fp, js(folder/'fallback_rule.json'))
            m._research_cleanup(fine); del fine
            delta_b = coordinate_difference(bp, frame(folder/'B4_test.csv'), f'{label} B4 fold {fold}')
            sources[label] = {'coarse': cp, 'prediction': bp, 'radius': radius}
            checks.append({'fold': fold, 'component': label, 'coarse_max_difference_px': delta_c,
                           'final_max_difference_px': delta_b})
        baseline = blend(sources['original']['prediction'], sources['seed10000']['prediction'], .25)
        coordinate_difference(baseline, frame(OUT/f'fold_{fold}'/'BASELINE8P2_test.csv'), '8.2 fold reconstruction')
        results = {'BASELINE8P2': baseline}
        for variant in VARIANTS:
            destination = OUT/f'fold_{fold}'/variant
            config = reload_config(js(destination/'final_config.json')['config'])
            m.seed_everything(config.SEED)
            roi = m.make_roi_spec(sources['seed10000']['coarse'], test, config,
                                 mode='calibrated', calibration=sources['seed10000']['radius'])
            fine = m.research_build_model(config, 'fine', 'cuda')
            m.load_model_checkpoint(fine, destination/'fine_final/model.pt', 'cuda'); counts += 1
            fp = m.predict_research(fine, test, config, 'cuda', 'fine', roi)
            pred = m.apply_fallback(sources['seed10000']['coarse'], fp, js(destination/'fallback_rule.json'))
            m._research_cleanup(fine); del fine
            delta = coordinate_difference(pred, frame(OUT/f'fold_{fold}'/f'{variant}_test.csv'), f'{variant} fold {fold}')
            results[variant] = pred
            checks.append({'fold': fold, 'component': variant, 'final_max_difference_px': delta})
        results['GUARDED20'] = blend(baseline, results[selection['selected']], selection['blend_weight'])
        delta = coordinate_difference(results['GUARDED20'], frame(OUT/f'fold_{fold}'/'GUARDED20_test.csv'), f'guarded fold {fold}')
        checks.append({'fold': fold, 'component': 'GUARDED20', 'final_max_difference_px': delta})
        for name, pred in results.items():
            all_results[name].append(pred[['image_id', 'x', 'y']].assign(fold=fold))
        print(f'Reload fold {fold+1}/5 PASS; networks={counts}/25, candidate delta={delta:.9g}px', flush=True)
    require(counts == 25, 'Did not reload all 25 final networks')
    ensembles = {}
    for name, parts in all_results.items():
        combined = pd.concat(parts, ignore_index=True)
        require((combined.groupby('image_id').size() == 5).all(), 'Every test image needs all five folds')
        for fold in range(5):
            saved = frame(OUT/f'test_all_folds_{name}.csv')
            coordinate_difference(combined.loc[combined.fold == fold], saved.loc[saved.fold == fold], f'{name} all-fold record')
        ensembles[name] = combined.groupby('image_id', as_index=False)[['x', 'y']].mean()
        coordinate_difference(ensembles[name], frame(OUT/f'test_ensemble_{name}.csv'), f'{name} five-fold ensemble')
    return ensembles, checks, counts


def verify_submission(path, prediction, test, expected_ids):
    sub = pd.read_csv(path)
    require(list(sub.columns) == ['ImageID', 'value'], 'Wrong submission columns')
    require(len(sub) == 40 and sub.ImageID.is_unique, 'Wrong submission row count/duplicate IDs')
    require(sub.ImageID.tolist() == expected_ids, 'Submission ID order differs from baseline/template')
    coords = prediction.set_index('image_id'); sizes = test.set_index('image_id')
    expected = []
    for row in sub.itertuples():
        iid, axis = row.ImageID.split('_Fovea_')
        require(axis in ('X', 'Y') and iid in sizes.index, 'Malformed or unexpected submission ID')
        require(math.isfinite(float(row.value)), 'Nonfinite submission coordinate')
        limit = float(sizes.loc[iid, 'width' if axis == 'X' else 'height']) - 1
        require(0 <= row.value <= limit, 'Submission coordinate outside image')
        expected.append(float(coords.loc[iid, 'x' if axis == 'X' else 'y']))
    difference = float(np.max(np.abs(sub.value.to_numpy(float) - np.asarray(expected))))
    require(difference < TOLERANCE_PX, f'Submission reload mismatch: {difference}px')
    return difference


def verify_oof(train, selections):
    truth = train.set_index('image_id'); summaries = []; per_fold = []
    all_frames = {}
    stored_summary = pd.read_csv(OUT/'ablation_summary.csv').set_index('method')
    for name in METHODS:
        pred = frame(OUT/f'oof_{name}.csv')
        require(pred.image_id.is_unique and set(pred.image_id) == set(train.image_id), 'Incomplete OOF predictions')
        expected = truth.loc[pred.image_id]
        for column, reference in (('fold', 'fold'), ('group', 'group'), ('x_true', 'x'), ('y_true', 'y')):
            if column in ('x_true', 'y_true'):
                require(np.allclose(pred[column], expected[reference], atol=1e-9, rtol=0), 'OOF truth differs')
            else:
                require(np.array_equal(pred[column].to_numpy(), expected[reference].to_numpy()), 'OOF fold/group differs')
        measured = metrics(pred)
        for key, value in measured.items():close(stored_summary.loc[name, key], value, f'{name} summary {key}', atol=1e-8)
        summaries.append({'method': name, **measured}); all_frames[name] = pred
    for fold, selection in enumerate(selections):
        current = {name: df.loc[df.fold == fold].copy() for name, df in all_frames.items()}
        original = frame(OLD/f'fold_{fold}'/'B4_oof.csv'); source = frame(SEED/f'fold_{fold}'/'B4_oof.csv')
        expected_base = blend(original, source, .25)
        coordinate_difference(expected_base, current['BASELINE8P2'], 'Baseline OOF aggregation')
        expected_final = blend(expected_base, current[selection['selected']], selection['blend_weight'])
        coordinate_difference(expected_final, current['GUARDED20'], 'Guarded OOF gate aggregation')
        stored = pd.read_csv(OUT/f'fold_{fold}'/'metrics.csv').set_index('method')
        result = {'fold': fold}
        for name in METHODS:
            measured = metrics(current[name])
            for key, value in measured.items():close(stored.loc[name, key], value, 'Per-fold metric', atol=1e-8)
            result[name] = measured['rmse']
            coordinate_difference(current[name], frame(OUT/f'fold_{fold}'/f'{name}_oof.csv'), 'Combined versus per-fold OOF')
        per_fold.append(result)
    return summaries, per_fold


def validate():
    started = time.perf_counter()
    protocol = js(OUT/'run_protocol.json'); execution = js(OUT/'execution_report.json')
    require(execution['status'] == 'COMPLETE' and execution['new_training_stages'] == 15, 'Training has not completed')
    require(torch.cuda.is_available(), 'GPU is needed for independent reload')
    m.seed_everything(cfg.SEED); m.cv2.setNumThreads(0)
    source_manifest = js(WORK/'source_manifest.json')
    require(m.research_file_digest(WORK/'engine.py') == source_manifest['engine_sha256'], 'Combined engine changed')
    for name, digest in source_manifest['source_parts'].items():
        require(m.research_file_digest(ROOT/Path(name)) == digest, f'Source part changed: {name}')
    require(m.research_runtime_code_digest() == protocol['runtime_digest'], 'Runtime functions differ from training')
    identity = js(OUT/'baseline_identity.json'); backup = js(OUT/'backup_verification.json')
    require(backup['status'] == 'PASS' and backup['trained_final_models'] == 20, 'Baseline backup not verified')
    for path in (Path(identity['baseline_submission']), OUT/'baseline_8p2.csv'):
        require(m.research_file_digest(path) == identity['sha256'] == protocol['baseline_csv_sha256'], '8.2 baseline changed')
    with zipfile.ZipFile(backup['backup']) as archive:
        backup_manifest = json.loads(archive.read('backup_manifest.json'))
    for run in (OLD, SEED):
        for name in ('run_config.json', 'folds.csv', 'test_manifest.csv'):
            path = run/name
            historical = backup_manifest[path.relative_to(ROOT).as_posix()]
            require(path.stat().st_size == historical['bytes'] and
                    m.research_file_digest(path) == historical['sha256'],
                    f'Baseline configuration/data manifest changed since backup: {path}')
    require(identity['components'][0]['weight'] == .75 and identity['components'][1]['weight'] == .25, 'Baseline component weights changed')
    require(Path(identity['components'][0]['run']).resolve() == OLD.resolve(), 'Wrong original baseline run')
    require(Path(identity['components'][1]['run']).resolve() == SEED.resolve(), 'Wrong additional-seed run')
    # This single-variable branch uses only the original ResNet18 ImageNet
    # initialization; no EfficientNet weight is needed or implicitly loaded.
    weight_path = Path(protocol['config']['PRETRAINED_WEIGHTS_PATH'])
    expected_weights = {m.RESEARCH_WEIGHT_FILENAME: m.RESEARCH_WEIGHT_SHA256}
    require(weight_path.is_file(), 'ResNet18 initialization file is missing')
    require(m.research_file_digest(weight_path) == m.RESEARCH_WEIGHT_SHA256,
            'Official ResNet18 initialization hash mismatch')
    train = frame(OUT/'folds.csv'); test = frame(OUT/'test_manifest.csv')
    require(len(train) == 80 and train.image_id.is_unique and set(train.image_id) == set(map(str, range(1,81))), 'Wrong training set')
    require(len(test) == 20 and test.image_id.is_unique and set(test.image_id) == set(map(str, range(81,101))), 'Wrong test set')
    require(not set(train.pixel_hash) & set(test.pixel_hash), 'Exact train/test pixel overlap')
    require(not set(train.image_id) & set(test.image_id), 'Train/test ID overlap')
    require(train.loc[train.image_id.isin(['39','77']), 'fold'].nunique() == 1, 'Duplicate images split across folds')
    for run in (OLD, SEED):
        previous = frame(run/'folds.csv').set_index('image_id').loc[train.image_id]
        for column in ('fold', 'group', 'pixel_hash', 'x', 'y'):
            require(np.array_equal(train[column].to_numpy(), previous[column].to_numpy()), f'{run}: data/fold changed')
        previous_test = frame(run/'test_manifest.csv').set_index('image_id').loc[test.image_id]
        for column in ('pixel_hash', 'width', 'height'):
            require(np.array_equal(test[column].to_numpy(), previous_test[column].to_numpy()), 'Test inputs changed')
    record_audit, selections = audit_records(protocol, train, test, backup_manifest)
    require(execution['selected_folds'] == selections, 'Execution selections differ from folds')
    summaries, per_fold = verify_oof(train, selections)
    ensembles, reload_checks, networks = reload_predictions(test, selections)
    reference = pd.read_csv(OUT/'baseline_8p2.csv')
    baseline_delta = verify_submission(OUT/'baseline_8p2.csv', ensembles['BASELINE8P2'], test, reference.ImageID.tolist())
    final_delta = verify_submission(OUT/'submission.csv', ensembles['GUARDED20'], test, reference.ImageID.tolist())
    require(m.research_file_digest(OUT/'submission.csv') == execution['submission_sha256'], 'Final CSV changed since training')
    comparison = {'rmse_difference_candidate_minus_baseline': summaries[-1]['rmse'] - summaries[0]['rmse'],
                  'median_difference_px': summaries[-1]['median_distance_px'] - summaries[0]['median_distance_px'],
                  'p90_difference_px': summaries[-1]['p90_distance_px'] - summaries[0]['p90_distance_px']}
    return {'status': 'PASS', 'new_training_stages': 15, 'reloaded_final_networks': networks,
            'baseline_final_networks_unchanged': record_audit['old_models_unchanged'],
            'baseline_submission_unchanged': True, 'baseline_sha256': identity['sha256'],
            'submission_sha256': execution['submission_sha256'], 'submission_rows': 40, 'test_images': 20,
            'test_images_in_training_or_selection': 0, 'coarse_lineage_verified': True,
            'pretraining_files_sha256_verified': expected_weights,
            'reload_uses_pretrained_initialization': False,
            'source_engine_sha256': source_manifest['engine_sha256'],
            'independent_reload_tolerance_px': TOLERANCE_PX,
            'baseline_max_reload_difference_px': baseline_delta,
            'submission_max_reload_difference_px': final_delta,
            'record_audit': record_audit, 'reload_checks': reload_checks,
            'recomputed_oof_summary': summaries, 'recomputed_per_fold_rmse': per_fold,
            'comparison': comparison, 'accepted_folds': [s['fold'] for s in selections if s['accepted']],
            'elapsed_seconds': time.perf_counter()-started,
            'limitations': 'Repeated development OOF; no independent leaderboard result. Role/input manifests and exact pixel hashes checked; patient identity and approximate duplicates unavailable. Ensemble uncertainty coverage is not claimed.'}


if __name__ == '__main__':
    started = time.perf_counter()
    try:
        report = validate()
    except Exception as error:
        failure = {'status': 'FAIL', 'error_type': type(error).__name__, 'error': str(error),
                   'elapsed_seconds': time.perf_counter()-started, 'traceback': traceback.format_exc(),
                   'policy': 'No tolerance change, model modification or fallback was performed.'}
        (OUT/'validation_report.json').write_text(json.dumps(failure, indent=2), encoding='utf-8')
        raise
    (OUT/'validation_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({'status': report['status'], 'reloaded_networks': report['reloaded_final_networks'],
        'candidate_max_difference_px': report['submission_max_reload_difference_px'],
        'accepted_folds': report['accepted_folds'], 'seconds': report['elapsed_seconds']}, indent=2), flush=True)
