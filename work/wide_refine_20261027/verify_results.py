"""Independent audit of the frozen wide-window experiment; never trains.

Only original training labels are read. The main entry point requires all folds
complete and reloads 20 new final networks plus 10 frozen SINGLE controls.
Calling fold_record_audit alone is CPU-only and supports completed-fold checks.
"""
from pathlib import Path
from dataclasses import asdict, replace
import hashlib
import json
import time
import traceback

from run import ROOT, WORK, OUT, PREVIOUS, m, old_engine, cfg
import numpy as np
import pandas as pd
import torch

PREVIOUS_SHA = '7e40d7de452b984e2c7d899c210698fe2506311fb92d459ab10f2e632acdcc5c'
METHODS = ('COARSE', 'SINGLE', 'WIDE_DEV', 'WIDE_FULL', 'MATCHED_DEPLOY')
PRIMARY_EXTRA = ('PREVIOUS_SUBMISSION', 'SUBMISSION_CANDIDATE')
TOLERANCE_PX = 1e-3
ROI_SCOPE = 'current .25 proposal coverage; blends have no single physical ROI'


def require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def js(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def frame(path):
    # Old operands were consumed by wr_read's default parser. New serialized
    # operands must round-trip exactly for stage signatures and model replay.
    precision = 'round_trip' if Path(path).resolve().is_relative_to(OUT.resolve()) else None
    return pd.read_csv(path, dtype={'image_id': str, 'group': str}, float_precision=precision)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def close(a, b, name, tolerance=1e-7):
    require(np.isfinite(float(a)) and np.isfinite(float(b)), f'{name}: nonfinite')
    require(abs(float(a)-float(b)) <= tolerance, f'{name}: {a} != {b}')


def align(value, ids):
    ids = list(ids)
    require(value.image_id.is_unique and len(ids) == len(set(ids)), 'Repeated image IDs')
    require(set(value.image_id) == set(ids), 'Mismatched image IDs')
    return value.set_index('image_id').loc[ids].reset_index()


def compare_xy(actual, expected, label, tolerance=TOLERANCE_PX):
    expected = align(expected, actual.image_id)
    a, b = (v[['x', 'y']].to_numpy(float) for v in (actual, expected))
    require(len(a) > 0 and np.isfinite(a).all() and np.isfinite(b).all(), f'{label}: invalid xy')
    difference = float(np.abs(a-b).max())
    require(difference < tolerance, f'{label}: delta {difference:.12g} >= {tolerance}')
    return difference


def blend(a, b, weight=.25):
    out = a[['image_id', 'x', 'y']].copy()
    right = align(b, out.image_id)[['x', 'y']].to_numpy(float)
    out[['x', 'y']] = (1-weight)*out[['x', 'y']].to_numpy(float) + weight*right
    return out


def metric(prediction, truth, coarse=None, roi=None, source_roi=None):
    prediction = align(prediction, truth.image_id)
    y = truth[['x', 'y']].to_numpy(float)
    residual = prediction[['x', 'y']].to_numpy(float)-y
    require(len(y) > 0 and np.isfinite(residual).all(), 'Invalid metric operands')
    distance = np.sqrt(np.sum(residual**2, axis=1))
    result = {'rmse': float(np.sqrt(np.mean(residual**2))), 'mean_distance_px': float(distance.mean()),
              'median_distance_px': float(np.median(distance)), 'p90_distance_px': float(np.quantile(distance, .9)),
              'p95_distance_px': float(np.quantile(distance, .95)), 'max_distance_px': float(distance.max()),
              'images': len(y)}
    if coarse is not None:
        coarse = align(coarse, truth.image_id)
        error = np.sqrt(np.sum((coarse[['x', 'y']].to_numpy(float)-y)**2, axis=1))
        result['worse_than_coarse_fraction'] = float(np.mean(distance > error+1e-9))
    for spec, key in ((roi, 'roi_coverage'), (source_roi, 'old_single_roi_coverage')):
        if spec is not None:
            spec = align(spec, truth.image_id)
            delta = np.abs(y-spec[['crop_x', 'crop_y']].to_numpy(float))
            result[key] = float((delta <= spec.crop_side.to_numpy(float)[:, None]/2).all(axis=1).mean())
    if roi is not None:
        result['roi_coverage_scope'] = ROI_SCOPE
    return result


def verify_metric_record(saved, expected, label, tolerance=1e-6):
    for key, value in expected.items():
        if isinstance(value, str):
            require(saved[key] == value, f'{label}.{key}')
        else:
            close(saved[key], value, f'{label}.{key}', tolerance)


def verify_manifest(folder, records):
    folder = Path(folder).resolve()
    require(isinstance(records, dict) and records, f'{folder}: empty manifest')
    for relative, entry in records.items():
        path = (folder/relative).resolve()
        require(path.is_relative_to(folder), f'Manifest escape: {relative}')
        require(path.is_file() and path.stat().st_size == entry['bytes'], f'File size: {path}')
        require(sha(path) == entry['sha256'], f'File digest: {path}')


def verify_roi(roi, centres, truth, label, fraction=.25):
    roi, centres = align(roi, truth.image_id), align(centres, truth.image_id)
    require(np.isfinite(roi[['crop_x', 'crop_y', 'crop_side']].to_numpy(float)).all(), label)
    side = fraction*truth[['width', 'height']].min(axis=1).to_numpy(float)
    require(np.max(np.abs(roi.crop_side.to_numpy(float)-side)) < 1e-8, f'{label}: wrong side')
    columns = ['crop_x', 'crop_y'] if 'crop_x' in centres else ['x', 'y']
    require(np.max(np.abs(roi[['crop_x', 'crop_y']].to_numpy(float)-centres[columns].to_numpy(float))) < 1e-8,
            f'{label}: prediction centres changed')


def stage_check(folder, fitting, stopping, stage_cfg, seed, epochs, train_crop, val_crop=None):
    record = js(folder/'stage_complete.json')
    verify_manifest(folder, record['files'])
    inputs = frame(folder/'actual_training_inputs.csv')
    require(inputs.image_id.tolist() == fitting.image_id.tolist(), f'{folder}: training row order')
    for column in ('group', 'pixel_hash'):
        require(inputs[column].astype(str).tolist() == fitting[column].astype(str).tolist(), f'{folder}: training {column}')
    # Target coordinates/dimensions are bound by the exact stage signature;
    # actual_training_inputs intentionally records only IDs/groups/pixel hashes.
    if stopping is None:
        require(not (folder/'actual_early_stop_inputs.csv').exists(), f'{folder}: unexpected stop records')
    else:
        stopped = frame(folder/'actual_early_stop_inputs.csv')
        require(stopped.image_id.tolist() == stopping.image_id.tolist(), f'{folder}: stopping identity/order')
        require(not set(inputs.group) & set(stopped.group), f'{folder}: group overlap')
    saved = align(frame(folder/'actual_training_crops.csv'), train_crop.image_id)
    require(np.array_equal(saved[['crop_x', 'crop_y', 'crop_side']], train_crop[['crop_x', 'crop_y', 'crop_side']]),
            f'{folder}: training crop operands changed')
    signature = m.research_stage_signature(fitting, stopping, stage_cfg, 'fine', seed, epochs, train_crop, val_crop)
    require(signature == record['signature'], f'{folder}: stage signature')
    info, history = js(folder/'training_summary.json'), pd.read_csv(folder/'history.csv')
    require(len(history) > 0 and history.epoch.tolist() == list(range(1, len(history)+1)), f'{folder}: epoch sequence')
    require(np.isfinite(history.train_loss.to_numpy(float)).all(), f'{folder}: nonfinite loss')
    require(info['successful_optimizer_steps'] == int(history.successful_optimizer_steps.sum()) > 0, f'{folder}: no updates')
    require(info['training_images'] == len(fitting) and info['fixed_epochs'] == (epochs is not None), f'{folder}: refit role')
    if epochs is not None:
        require(info['best_epoch'] == epochs == len(history), f'{folder}: fixed epochs')
    else:
        require(np.isfinite(history.val_rmse.to_numpy(float)).all(), f'{folder}: invalid stop loss')
        best = int(history.val_rmse.to_numpy(float).argmin())
        require(info['best_epoch'] == best+1, f'{folder}: best epoch')
        close(info['best_rmse'], history.iloc[best].val_rmse, f'{folder}: best score')
    require(info['validation_metric'] == 'inner RMSE' and info['fine_multiscale_train'] is False,
            f'{folder}: deployment-scale selection')
    return info


def verify_source_and_protection():
    source = js(WORK/'source_manifest.json')
    require(source == js(OUT/'source_manifest.json'), 'Copied frozen source manifest differs')
    require(source['engine_sha256'] == sha(WORK/'engine.py'), 'Frozen engine changed')
    for name, digest in source['parts'].items():
        require(sha(ROOT/name) == digest, f'Source changed: {name}')
    require(source['previous_engine_sha256'] == sha(ROOT/'work/dual_context_20261027/engine.py'), 'Previous engine changed')
    previous_validation = js(PREVIOUS/'validation_report.json')
    require(previous_validation['status'] == 'PASS' and previous_validation['submission_sha256'] == PREVIOUS_SHA,
            'Previous experiment validation')
    protected = js(OUT/'protected_previous.json')
    require(protected['status'] == 'CAPTURED_BEFORE_NEW_TRAINING', 'Protection status')
    verify_manifest(ROOT, protected['files'])
    for split in (2026, 2027):
        for fold in range(5):
            for relative in ('complete.json', 'single/fine_final/model.pt', 'SINGLE_oof.csv', 'training_centres.csv'):
                path = PREVIOUS/f'split_{split}'/f'fold_{fold}'/relative
                require(path.relative_to(ROOT).as_posix() in protected['files'], f'Unprotected inherited file: {path}')
    require(sha(OUT/'previous_submission.csv') == sha(PREVIOUS/'submission.csv') == PREVIOUS_SHA, 'Previous submission changed')
    planned = js(OUT/'planned_experiment.json')
    require(planned['status'] == 'FROZEN_BEFORE_TRAINING' and planned['source_manifest'] == source and
            planned['new_training_stages'] == 30 and planned['split_seeds'] == [2026, 2027], 'Frozen plan changed')
    return len(protected['files'])


def role_frames(train, folder):
    roles = frame(folder/'split_roles.csv')
    require(roles.image_id.is_unique and set(roles.image_id) == set(train.image_id), 'Role image identities')
    require(set(roles.role) == {'model_fit', 'early_stop', 'selection_then_final_fit', 'outer_validation'}, 'Role semantics')
    require(roles.groupby('group').role.nunique().max() == 1, 'Group crosses roles')
    keyed = align(roles, train.image_id)
    require(keyed.group.tolist() == train.group.tolist(), 'Role group mapping changed')
    def take(role):
        return train.loc[train.image_id.isin(roles.loc[roles.role == role, 'image_id'])].reset_index(drop=True)
    fit, stop, selection, outer = [take(name) for name in ('model_fit', 'early_stop', 'selection_then_final_fit', 'outer_validation')]
    development = train.loc[train.image_id.isin(set(fit.image_id)|set(stop.image_id))].reset_index(drop=True)
    full = train.loc[~train.image_id.isin(outer.image_id)].reset_index(drop=True)
    return fit, stop, development, selection, full, outer


def audit_gate(selection, single, wide, truth, tolerance=1e-6):
    single_metrics, wide_metrics = metric(single, truth), metric(wide, truth)
    verify_metric_record(selection['single_metrics'], single_metrics, 'Gate SINGLE', tolerance)
    verify_metric_record(selection['wide_dev_metrics'], wide_metrics, 'Gate WIDE_DEV', tolerance)
    gain = single_metrics['rmse']-wide_metrics['rmse']
    close(selection['gain_px'], gain, 'Gate gain', tolerance)
    accepted = bool(gain >= .5 and gain >= .02*single_metrics['rmse'] and
                    wide_metrics['p90_distance_px'] <= single_metrics['p90_distance_px']+1e-9)
    require(selection['accepted'] == accepted, 'Gate outcome differs')
    require(selection['minimum_gain_px'] == .5 and selection['minimum_relative_gain'] == .02 and
            selection['p90_must_not_increase'] is True and selection['weight_if_accepted'] == .25, 'Gate constants changed')
    require(selection['selection_ids'] == truth.image_id.tolist() and
            selection['selection_groups'] == truth.group.astype(str).unique().tolist(), 'Gate image/group roles')
    return {'accepted': accepted, 'gain_px': gain, 'single_rmse': single_metrics['rmse'], 'wide_dev_rmse': wide_metrics['rmse'],
            'single_p90': single_metrics['p90_distance_px'], 'wide_dev_p90': wide_metrics['p90_distance_px']}


def fold_record_audit(split, fold, train, test):
    """Read only one completed fold; no CUDA model, inference, or training."""
    folder, source = OUT/f'split_{split}'/f'fold_{fold}', PREVIOUS/f'split_{split}'/f'fold_{fold}'
    complete = js(folder/'complete.json')
    require(complete['status'] == 'COMPLETE' and complete['split_seed'] == split and complete['fold'] == fold,
            'Fold completion identity')
    require(complete['full_refit_has_no_independent_calibration'] is True, 'FULL calibration claim')
    verify_manifest(folder, complete['files'])
    fold_cfg = m.ResearchConfig(**js(folder/'fold_config.json')['config'])
    expected_cfg = replace(cfg, SEED=split, OUTPUT_DIR=str(OUT/f'split_{split}'))
    require(asdict(fold_cfg) == asdict(expected_cfg), 'Fold configuration changed')
    require(fold_cfg.FINE_INPUT_SIZE == 384 and fold_cfg.FINE_HEATMAP_SIZE == 96 and fold_cfg.HEATMAP_SIGMA == 1.5 and
            fold_cfg.FIXED_ROI_FRACTION == .25 and fold_cfg.ROI_MARGIN_PX == 0 and fold_cfg.MODEL_SEED_OFFSET == 10000 and
            not fold_cfg.FINE_MULTISCALE_TRAIN and not fold_cfg.FINE_MULTISCALE_EARLY_STOP, 'Treatment changed')
    fit, stop, development, selection_group, full, outer = role_frames(train, folder)
    require(set(outer.image_id) == set(train.loc[train.fold == fold, 'image_id']), 'Outer role assignment')
    require(len(full) + len(outer) == 80 and set(full.image_id) == set(development.image_id)|set(selection_group.image_id),
            'FULL data union')
    require(not set(full.group) & set(outer.group) and not set(development.group) & set(selection_group.group), 'Group role leak')
    for partition in (fit, stop, development, selection_group, full, outer):
        require(not set(partition.image_id) & set(test.image_id) and not set(partition.pixel_hash) & set(test.pixel_hash), 'Test use')
    original = frame(source/'split_roles.csv')
    copied = align(frame(folder/'source_roles.csv'), original.image_id)
    require(copied.equals(original), 'Source role snapshot changed')
    renamed = original.copy(); renamed.loc[renamed.role == 'calibration_only', 'role'] = 'selection_then_final_fit'
    require(align(frame(folder/'split_roles.csv'), renamed.image_id).equals(renamed), 'Role relabeling changed')
    lineage = js(folder/'coarse_lineage.json')
    require(Path(lineage['source_folder']).resolve() == source.resolve() and
            lineage['source_complete_sha256'] == sha(source/'complete.json'), 'Source lineage')
    verify_manifest(source, lineage['verified_source_files'])
    parent = Path(lineage['coarse_parent'])
    verify_manifest(parent, lineage['verified_coarse_files'])
    require(Path(lineage['coarse_checkpoint']).resolve() == (parent/'coarse_final/model.pt').resolve(), 'Coarse checkpoint lineage')
    for path, key in ((source/'single/fine_final/actual_training_inputs.csv', 'old_single_training_ids'),
                      (parent/'coarse_final/actual_training_inputs.csv', 'old_coarse_training_ids')):
        actual = frame(path)
        require(actual.image_id.tolist() == lineage[key] and set(actual.image_id) == set(development.image_id), 'Inherited fitting roles')
        require(not set(actual.image_id) & (set(selection_group.image_id)|set(outer.image_id)|set(test.image_id)), 'Inherited selection leak')
    crossfit = js(folder/'crop_crossfit_audit.json')
    require(crossfit == js(source/'crop_crossfit_audit.json'), 'Crossfit source changed')
    held = []
    for row in crossfit:
        fit_ids, held_ids = set(row['fit_ids']), set(row['held_ids'])
        require(fit_ids|held_ids == set(fit.image_id) and not fit_ids & held_ids, 'Fitting-only crossfit partition')
        require(not set(row['fit_groups']) & set(row['held_groups']), 'Crossfit group leak')
        held.extend(row['held_ids'])
    require(len(held) == len(set(held)) == len(fit) and set(held) == set(fit.image_id), 'Crossfit centre coverage')
    centres_dev, centres_full = frame(folder/'training_centres_development.csv'), frame(folder/'training_centres_full.csv')
    require(centres_dev.image_id.tolist() == development.image_id.tolist() and centres_full.image_id.tolist() == full.image_id.tolist(),
            'Training centre order')
    verify_roi(centres_dev, frame(source/'training_centres.csv'), development, 'Development centres')
    coarse = {kind: frame(folder/f'coarse_{kind}.csv') for kind in ('selection', 'oof')+(('test',) if split == 2026 else ())}
    for kind, oldkind in [('selection', 'calibration'), ('oof', 'oof')]+([('test', 'test')] if split == 2026 else []):
        compare_xy(coarse[kind], frame(source/f'coarse_{oldkind}.csv'), 'Inherited coarse '+kind, 1e-8)
    full_dev = centres_full.loc[centres_full.image_id.isin(development.image_id)]
    full_extra = centres_full.loc[centres_full.image_id.isin(selection_group.image_id)]
    verify_roi(full_dev, centres_dev, development, 'FULL old centres')
    verify_roi(full_extra, coarse['selection'], selection_group, 'FULL added held centres')
    fit_crop = centres_dev.loc[centres_dev.image_id.isin(fit.image_id)].reset_index(drop=True)
    stop_crop = centres_dev.loc[centres_dev.image_id.isin(stop.image_id)].reset_index(drop=True)
    base_seed = split + fold*1000 + fold_cfg.MODEL_SEED_OFFSET
    selected = stage_check(folder/'wide_select', fit, stop, fold_cfg, base_seed+20, None, fit_crop, stop_crop)
    epochs = int(selected['best_epoch'])
    dev_info = stage_check(folder/'wide_dev', development, None, fold_cfg, base_seed+40, epochs, centres_dev)
    full_info = stage_check(folder/'wide_full', full, None, fold_cfg, base_seed+40, epochs, centres_full)
    require(full_info['successful_optimizer_steps'] >= dev_info['successful_optimizer_steps'], 'FULL updates unexpectedly smaller')
    stopping = frame(folder/'selected_stop_predictions.csv')
    close(metric(stopping, stop)['rmse'], selected['best_rmse'], 'Selected deployment RMSE', 1e-3)
    decision = js(folder/'selection.json')
    require(decision == complete['selection'] and decision['fold'] == fold and decision['split_seed'] == split, 'Gate record identity')
    require(decision['selected_epoch'] == epochs and decision['stopping_ids'] == stop.image_id.tolist() and
            decision['full_training_ids'] == full.image_id.tolist(), 'Selected epoch/refit identities')
    close(decision['early_stop_best_rmse'], selected['best_rmse'], 'Gate selected score')
    # Local filesystem ordering corroborates the frozen source's gate-before-refit
    # order. All data needed for the gate is also independently recomputed here.
    require((folder/'selection.json').stat().st_mtime_ns <= (folder/'wide_full/actual_training_inputs.csv').stat().st_mtime_ns,
            'Gate was not saved before FULL training inputs')
    roi_selection = frame(folder/'calibration_roi.csv')
    verify_roi(roi_selection, coarse['selection'], selection_group, 'Selection wide ROI')
    verify_roi(frame(folder/'calibration_single_roi.csv'), coarse['selection'], selection_group, 'Selection SINGLE ROI', .15)
    old_cfg = old_engine.DualConfig(**js(source/'single/config.json')['config'])
    require(old_cfg.FINE_VARIANT == 'single' and old_cfg.SEED == split and old_cfg.FINE_INPUT_SIZE == 256 and
            old_cfg.FINE_HEATMAP_SIZE == 64 and old_cfg.HEATMAP_SIGMA == 1.5, 'Old SINGLE model configuration')
    gate = audit_gate(decision, frame(folder/'calibration_single.csv'), frame(folder/'calibration_wide_dev.csv'), selection_group)
    require(not list(folder.glob('*risk*')) and not list(folder.glob('*radius*')), 'Unexpected FULL calibration artifacts')
    metric_rows = frame(folder/'metrics.csv')
    expected_methods = METHODS + (PRIMARY_EXTRA if split == 2026 else ())
    require(set(metric_rows.method) == set(expected_methods) and len(metric_rows) == len(expected_methods), 'Method list')
    for kind, truth in [('oof', outer)]+([('test', test)] if split == 2026 else []):
        roi, source_roi = frame(folder/f'{kind}_roi.csv'), frame(folder/f'source_single_{kind}_roi.csv')
        verify_roi(roi, coarse[kind], truth, kind+' proposal ROI')
        verify_roi(source_roi, coarse[kind], truth, kind+' old SINGLE ROI', .15)
        names = [name for name in expected_methods if kind == 'oof' or name != 'WIDE_DEV']
        predictions = {name: frame(folder/f'{name}_{kind}.csv') for name in names}
        compare_xy(predictions['COARSE'], coarse[kind], 'Saved COARSE', 1e-8)
        compare_xy(predictions['SINGLE'], frame(source/f'SINGLE_{kind}.csv'), 'Saved inherited SINGLE', 1e-8)
        matched = blend(predictions['SINGLE'], predictions['WIDE_FULL']) if gate['accepted'] else predictions['SINGLE']
        compare_xy(predictions['MATCHED_DEPLOY'], matched, 'Matched blend', 1e-8)
        if split == 2026:
            compare_xy(predictions['PREVIOUS_SUBMISSION'], frame(source/f'SUBMISSION_CANDIDATE_{kind}.csv'), 'Previous fold', 1e-8)
            candidate = blend(predictions['PREVIOUS_SUBMISSION'], predictions['WIDE_FULL']) if gate['accepted'] else predictions['PREVIOUS_SUBMISSION']
            compare_xy(predictions['SUBMISSION_CANDIDATE'], candidate, 'Submission fold blend', 1e-8)
        for name, prediction in predictions.items():
            require(prediction.image_id.tolist() == truth.image_id.tolist(), f'{name}/{kind} row order')
            require((prediction.fold == fold).all() and (prediction.experiment == name).all(), f'{name}/{kind} annotations')
            if kind == 'oof':
                paired = align(prediction, truth.image_id)
                require(np.array_equal(paired[['x_true', 'y_true']].to_numpy(), truth[['x', 'y']].to_numpy()), 'OOF truth changed')
                saved = metric_rows.loc[metric_rows.method == name].iloc[0]
                verify_metric_record(saved, metric(prediction, truth, coarse[kind], roi, source_roi), f'{name} fold metrics')
            else:
                require(not {'x_true', 'y_true'} & set(prediction.columns), 'Test truth present')
    if split == 2027:
        require(not list(folder.rglob('*test*.csv')), 'Secondary test predictions')
    return {'split_seed': split, 'fold': fold, 'stages': 3, 'selected_epoch': epochs, **gate,
            'development_images': len(development), 'selection_then_refit_images': len(selection_group),
            'full_training_images': len(full), 'outer_images': len(outer),
            'development_updates': dev_info['successful_optimizer_steps'], 'full_updates': full_info['successful_optimizer_steps'],
            'gate_saved_before_full_training': True, 'full_refit_independent_calibration': False}


def record_audit():
    execution, protocol = js(OUT/'execution_report.json'), js(OUT/'run_protocol.json')
    require(execution['status'] == 'COMPLETE' and execution['completed_folds'] == 10 and execution['new_training_stages'] == 30,
            'Experiment incomplete')
    require(execution['split_seeds'] == protocol['split_seeds'] == [2026, 2027] and protocol['new_training_stages'] == 30,
            'Split/stage protocol')
    require(protocol['roi_short_side_fraction'] == .25 and protocol['gate_min_px'] == .5 and protocol['gate_min_relative'] == .02 and
            protocol['new_branch_weight'] == .25 and protocol['gate_p90_must_not_increase'] is True, 'Treatment/gate protocol')
    require(protocol['test_reference_labels_used'] is False and execution['secondary_submission_influence'] is False and
            execution['full_refit_has_no_independent_calibration'] is True, 'Evaluation boundaries')
    require(protocol['runtime_digest'] == m.research_runtime_code_digest(), 'Runtime code changed')
    require(protocol['previous_submission_sha256'] == PREVIOUS_SHA, 'Previous protocol hash')
    test = frame(OUT/'test_manifest.csv')
    require(len(test) == 20 and test.image_id.is_unique and set(test.image_id) == set(map(str, range(81, 101))), 'Test IDs')
    require(not {'x', 'y', 'x_true', 'y_true'} & set(test.columns), 'Test labels present')
    input_audit = OUT/'verification_input_audit'; input_audit.mkdir(exist_ok=True)
    prepared_train, prepared_test, _, _, _ = m.prepare_tables(replace(cfg, OUTPUT_DIR=str(input_audit)))
    require(set(prepared_train.image_id) == set(map(str, range(1, 81))) and len(prepared_train) == 80, 'Training IDs')
    require(not set(prepared_train.image_id) & set(test.image_id) and not set(prepared_train.pixel_hash) & set(test.pixel_hash) and
            not set(prepared_train.path) & set(test.path), 'Train/test overlap')
    require(align(prepared_test, test.image_id).pixel_hash.tolist() == test.pixel_hash.tolist(), 'Test bytes changed')
    protected_count = verify_source_and_protection()
    records, frames = [], {}
    summaries = frame(OUT/'ablation_summary.csv')
    for split in (2026, 2027):
        split_folder = OUT/f'split_{split}'
        train = frame(split_folder/'folds.csv'); frames[split] = train
        expected = align(m.assign_folds(prepared_train, replace(cfg, SEED=split)), train.image_id)
        inherited = align(frame(PREVIOUS/f'split_{split}'/'folds.csv'), train.image_id)
        for col in ('group', 'fold', 'pixel_hash', 'x', 'y', 'width', 'height'):
            require(np.array_equal(train[col], expected[col]) and np.array_equal(train[col], inherited[col]), f'Split changed {col}')
        require(train.groupby('group').fold.nunique().max() == 1, 'Group crosses folds')
        if split == 2027:
            require(not list(split_folder.rglob('*test*.csv')), 'Secondary has test predictions')
        for fold in range(5):
            records.append(fold_record_audit(split, fold, train, test))
            require(js(split_folder/f'fold_{fold}'/'selection.json') == execution['decisions'][str(split)][fold], 'Execution gate differs')
            print(f'Record split {split} fold {fold+1}/5 PASS; stages={len(records)*3}/30', flush=True)
        names = METHODS + (PRIMARY_EXTRA if split == 2026 else ())
        split_summary = frame(split_folder/'summary.csv')
        require(len(split_summary) == len(names) and set(split_summary.method) == set(names), 'Summary method coverage')
        coarse = frame(split_folder/'oof_COARSE.csv')
        roi = pd.concat([frame(split_folder/f'fold_{fold}'/'oof_roi.csv') for fold in range(5)], ignore_index=True)
        source_roi = pd.concat([frame(split_folder/f'fold_{fold}'/'source_single_oof_roi.csv') for fold in range(5)], ignore_index=True)
        for name in names:
            combined = frame(split_folder/f'oof_{name}.csv')
            pieces = pd.concat([frame(split_folder/f'fold_{fold}'/f'{name}_oof.csv') for fold in range(5)], ignore_index=True)
            require(len(combined) == 80 and combined.image_id.is_unique and combined.image_id.tolist() == pieces.image_id.tolist(), 'OOF coverage/order')
            compare_xy(combined, pieces, 'OOF concatenation', 1e-8)
            expected_metrics = metric(combined, train, coarse, roi, source_roi)
            verify_metric_record(split_summary.loc[split_summary.method == name].iloc[0], expected_metrics, f'{split}/{name} summary')
            summary = summaries.loc[(summaries.split_seed == split) & (summaries.method == name)]
            require(len(summary) == 1, 'Global summary duplicates/missing')
            verify_metric_record(summary.iloc[0], expected_metrics, f'{split}/{name} global summary')
            if split == 2026 and name != 'WIDE_DEV':
                combined_test = frame(split_folder/f'test_folds_{name}.csv')
                pieces = pd.concat([frame(split_folder/f'fold_{fold}'/f'{name}_test.csv') for fold in range(5)], ignore_index=True)
                require(len(combined_test) == 100 and (combined_test.groupby('image_id').size() == 5).all(), 'Test-fold counts')
                require(combined_test.image_id.tolist() == pieces.image_id.tolist() and combined_test.fold.tolist() == pieces.fold.tolist(), 'Test-fold order')
                require(np.max(np.abs(combined_test[['x', 'y']].to_numpy()-pieces[['x', 'y']].to_numpy())) < 1e-8, 'Test-fold values')
                average = combined_test.groupby('image_id', as_index=False)[['x', 'y']].mean()
                compare_xy(frame(split_folder/f'test_ensemble_{name}.csv'), average, 'Five-fold mean', 1e-8)
    require(len(summaries) == 12 and len(records) == 10, 'All-method/all-fold counts')
    require(not np.array_equal(align(frames[2026], frames[2027].image_id).fold, frames[2027].fold), 'Secondary split identical')
    stage_markers = list(OUT.glob('split_*/fold_*/wide_*/stage_complete.json'))
    require(len(stage_markers) == 30 and sum(row['stages'] for row in records) == 30, 'Actual new stage count')
    return {'stages': 30, 'protected_files': protected_count, 'folds': records,
            'secondary_split_differs': True, 'secondary_test_predictions': 0}, frames, test


def reload_check(frames, test):
    require(torch.cuda.is_available(), 'GPU required for final independent reload')
    m.seed_everything(cfg.SEED); m.cv2.setNumThreads(0)
    count, maximum, records, submission_folds = 0, 0., [], []
    for split, train in frames.items():
        for fold in range(5):
            folder, source = OUT/f'split_{split}'/f'fold_{fold}', PREVIOUS/f'split_{split}'/f'fold_{fold}'
            fold_cfg = m.ResearchConfig(**js(folder/'fold_config.json')['config'])
            _, _, _, selection_group, _, outer = role_frames(train, folder)
            decision = js(folder/'selection.json')
            fresh = {}
            for stage in ('wide_dev', 'wide_full'):
                load_cfg = replace(fold_cfg, PRETRAINED=False)
                model = m.research_build_model(load_cfg, 'fine', 'cuda')
                m.load_model_checkpoint(model, folder/stage/'model.pt', 'cuda')
                kinds = ([('selection', selection_group)] if stage == 'wide_dev' else []) + [('oof', outer)]
                if stage == 'wide_full' and split == 2026:
                    kinds.append(('test', test))
                for kind, truth in kinds:
                    roi = frame(folder/('calibration_roi.csv' if kind == 'selection' else f'{kind}_roi.csv'))
                    # Each call preserves original frame order and batch shape.
                    prediction = m.predict_research(model, truth, load_cfg, 'cuda', 'fine', roi)
                    saved = frame(folder/('calibration_wide_dev.csv' if kind == 'selection' else f'{stage.upper()}_{kind}.csv'))
                    maximum = max(maximum, compare_xy(prediction, saved, f'Reload {stage} {kind}'))
                    fresh[(stage, kind)] = prediction
                m._research_cleanup(model); del model; count += 1
            old_cfg = old_engine.DualConfig(**js(source/'single/config.json')['config'])
            old_cfg = replace(old_cfg, PRETRAINED=False)
            model = old_engine.research_build_model(old_cfg, 'fine', 'cuda')
            old_engine.load_model_checkpoint(model, source/'single/fine_final/model.pt', 'cuda')
            for kind, truth in [('selection', selection_group), ('oof', outer)]+([('test', test)] if split == 2026 else []):
                if kind == 'selection':
                    # This prediction was newly produced by the wide pipeline.
                    roi = frame(folder/'calibration_single_roi.csv')
                else:
                    # These predictions were inherited from the previous run,
                    # not recomputed using the wide pipeline's default-parsed
                    # copy of its ROI CSV. Replay the actual original operands.
                    # .15*1934 is 290.09999999999997; default parsing of its CSV
                    # can yield 290.1 and change crop resampling numerics.
                    roi = pd.read_csv(source/f'{kind}_roi.csv', dtype={'image_id': str, 'group': str},
                                      float_precision='round_trip')
                prediction = old_engine.predict_research(model, truth, old_cfg, 'cuda', 'fine', roi)
                saved = frame(folder/('calibration_single.csv' if kind == 'selection' else f'SINGLE_{kind}.csv'))
                maximum = max(maximum, compare_xy(prediction, saved, 'Reload old SINGLE '+kind))
                fresh[('single', kind)] = prediction
            old_engine._research_cleanup(model); del model; count += 1
            # Gate outcomes must remain identical with replayed checkpoints.
            audit_gate(decision, fresh[('single', 'selection')], fresh[('wide_dev', 'selection')], selection_group,
                       tolerance=TOLERANCE_PX)
            for kind in ('oof',) + (('test',) if split == 2026 else ()):
                wide, single = fresh[('wide_full', kind)], fresh[('single', kind)]
                matched = blend(single, wide) if decision['accepted'] else single
                maximum = max(maximum, compare_xy(matched, frame(folder/f'MATCHED_DEPLOY_{kind}.csv'), 'Reload matched blend'))
                if split == 2026:
                    previous = frame(folder/f'PREVIOUS_SUBMISSION_{kind}.csv')
                    candidate = blend(previous, wide) if decision['accepted'] else previous
                    maximum = max(maximum, compare_xy(candidate, frame(folder/f'SUBMISSION_CANDIDATE_{kind}.csv'), 'Reload primary blend'))
                    if kind == 'test':
                        submission_folds.append(candidate.assign(fold=fold))
            records.append({'split_seed': split, 'fold': fold, 'reloaded_models': 3,
                            'new_final_models': ['wide_dev', 'wide_full'], 'frozen_controls': ['single/fine_final'],
                            'gate_replayed': True})
            print(f'Reload split {split} fold {fold+1}/5 PASS; networks={count}/30', flush=True)
    require(count == 30, 'Expected 30 final/control network reloads')
    average = pd.concat(submission_folds).groupby('image_id', as_index=False)[['x', 'y']].mean()
    return {'networks': count, 'new_final_networks': 20, 'frozen_single_controls': 10,
            'max_prediction_difference_px': maximum, 'folds': records}, average


def submission_check(fresh, test, tolerance=TOLERANCE_PX):
    submitted, previous = pd.read_csv(OUT/'submission.csv'), pd.read_csv(OUT/'previous_submission.csv')
    template = pd.read_csv(OUT/'sample_template.csv')
    require(list(submitted.columns) == ['ImageID', 'value'] and len(submitted) == 40, 'Submission format')
    require(submitted.ImageID.is_unique and submitted.ImageID.tolist() == previous.ImageID.tolist() == template.ImageID.tolist(),
            'Submission identifier order')
    coords, sizes, expected, seen = fresh.set_index('image_id'), test.set_index('image_id'), [], set()
    for name, value in zip(submitted.ImageID, submitted.value):
        image_id, axis = name.split('_Fovea_')
        require(axis in ('X', 'Y') and image_id in coords.index, 'Submission coordinate ID')
        require(np.isfinite(value) and 0 <= value <= sizes.loc[image_id, 'width' if axis == 'X' else 'height']-1, 'Submission range')
        expected.append(float(coords.loc[image_id, 'x' if axis == 'X' else 'y'])); seen.add((image_id, axis))
    require(seen == {(image_id, axis) for image_id in test.image_id for axis in ('X', 'Y')}, 'Submission coordinate coverage')
    delta = float(np.max(np.abs(np.asarray(expected)-submitted.value.to_numpy(float))))
    require(delta < tolerance, 'CSV does not match fivefold inference')
    require(sha(OUT/'submission.csv') == js(OUT/'execution_report.json')['submission_sha256'], 'Submission digest changed')
    reconstructed = pd.read_csv(OUT/'previous_reconstructed.csv')
    require(reconstructed.ImageID.tolist() == previous.ImageID.tolist() and
            np.max(np.abs(reconstructed.value.to_numpy()-previous.value.to_numpy())) <= 1.1e-6, 'Previous submission reconstruction')
    return delta


def main():
    started = time.perf_counter()
    try:
        audited, frames, test = record_audit()
        submission_check(frame(OUT/'split_2026/test_ensemble_SUBMISSION_CANDIDATE.csv'), test, 5.1e-7)
        reloads, fresh = reload_check(frames, test)
        csv_delta = submission_check(fresh, test)
        protected = verify_source_and_protection()
        report = {'status': 'PASS', 'new_training_stages': 30, 'completed_outer_folds': 10,
                  'reloaded_final_networks': reloads['networks'], 'reloaded_new_final_networks': 20,
                  'reloaded_frozen_single_controls': 10, 'protected_previous_files_unchanged': protected,
                  'protected_previous_verified_files': protected, 'previous_submission_sha256': PREVIOUS_SHA,
                  'submission_sha256': sha(OUT/'submission.csv'), 'source_engine_sha256': sha(WORK/'engine.py'),
                  'test_images_in_training_or_selection': 0, 'secondary_submission_influence': False,
                  'full_refit_has_no_independent_calibration': True, 'submission_rows': 40, 'test_images': 20,
                  'independent_reload_tolerance_px': TOLERANCE_PX, 'submission_max_reload_difference_px': csv_delta,
                  'max_reload_difference_px': max(csv_delta, reloads['max_prediction_difference_px']),
                  'accepted_primary_folds_zero_based': [r['fold'] for r in audited['folds'] if r['split_seed'] == 2026 and r['accepted']],
                  'accepted_folds_by_split': {str(split): [r['fold'] for r in audited['folds'] if r['split_seed'] == split and r['accepted']]
                                            for split in (2026, 2027)},
                  'selection_records': audited['folds'], 'record_audit': audited, 'reload_audit': reloads,
                  'input_replay_note': 'Inherited SINGLE OOF/test predictions are replayed using the protected previous producer ROI CSV with round-trip parsing. Newly produced SINGLE selection predictions use this run\'s selection ROI. Each call preserves the original full frame and batch partition; tolerance remains 0.001 px.',
                  'seconds': time.perf_counter()-started,
                  'limitations': 'Only original training labels were used for this audit. Secondary is repeated development, not an independent patient test. FULL reuses the selection group for fitting and has no independent uncertainty calibration. DEV/FULL change data and optimizer-update counts. No official leaderboard claim is verified.'}
    except Exception as error:
        report = {'status': 'FAIL', 'error': str(error), 'traceback': traceback.format_exc(), 'seconds': time.perf_counter()-started}
        (OUT/'validation_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        raise
    (OUT/'validation_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'status': 'PASS', 'new_training_stages': 30, 'reloaded_networks': 30,
                      'candidate_max_difference_px': csv_delta, 'accepted_folds': report['accepted_folds_by_split'],
                      'seconds': report['seconds']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
