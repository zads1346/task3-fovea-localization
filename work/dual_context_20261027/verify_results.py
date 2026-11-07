"""Independent record/math audit and 30-final-network reload after completion.

No training is launched. This file never opens external test-reference labels.
Only this new output directory receives the validation report.
"""
from pathlib import Path
from dataclasses import asdict, replace
import hashlib
import json
import math
import time
import traceback

from run import ROOT, WORK, OUT, OLD, SEED, m, cfg
import numpy as np
import pandas as pd
import torch

BASELINE_SHA = 'be3cba8f60f96325c7730480db2f740522e79ac550368984174a73396342176f'
METHODS = ('COARSE', 'SINGLE', 'DUAL_RAW', 'DUAL_RISK', 'MATCHED_DEPLOY')
PRIMARY_EXTRA = ('BASELINE8P2', 'SUBMISSION_CANDIDATE')
TOLERANCE_PX = 1e-3
POLICIES = ('fine_only', 'risk_half', 'risk_coarse', 'global_quarter')


def require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def js(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def frame(path):
    # New CSVs serialize the actual in-memory training operands. Read them
    # round-trip exactly (e.g. .15*1956 is 293.40000000000003, not 293.4).
    # Inherited old files were originally read with the base default parser;
    # preserve that parsing path when reconstructing inherited operands.
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
    require(value.image_id.is_unique, 'Repeated image IDs')
    require(set(value.image_id) == set(ids), 'Mismatched image IDs')
    return value.set_index('image_id').loc[list(ids)].reset_index()


def compare_xy(actual, expected, label, tolerance=TOLERANCE_PX):
    expected = align(expected, actual.image_id)
    a, b = (v[['x', 'y']].to_numpy(float) for v in (actual, expected))
    require(np.isfinite(a).all() and np.isfinite(b).all(), f'{label}: nonfinite xy')
    difference = float(np.abs(a-b).max())
    require(difference < tolerance, f'{label}: delta {difference:.12g} >= {tolerance}')
    return difference


def blend(a, b, weight):
    out = a[['image_id', 'x', 'y']].copy()
    right = align(b, out.image_id)[['x', 'y']].to_numpy(float)
    out[['x', 'y']] = (1-weight)*out[['x', 'y']].to_numpy(float) + weight*right
    return out


def metric(prediction, truth, coarse=None, roi=None):
    prediction = align(prediction, truth.image_id)
    residual = prediction[['x', 'y']].to_numpy(float)-truth[['x', 'y']].to_numpy(float)
    require(np.isfinite(residual).all(), 'Nonfinite residual')
    distance = np.hypot(residual[:, 0], residual[:, 1])
    result = {'rmse': float(np.sqrt(np.mean(residual**2))), 'mean_distance_px': float(distance.mean()),
              'median_distance_px': float(np.median(distance)), 'p90_distance_px': float(np.quantile(distance, .9)),
              'p95_distance_px': float(np.quantile(distance, .95)), 'max_distance_px': float(distance.max()),
              'images': len(truth)}
    if coarse is not None:
        coarse = align(coarse, truth.image_id)
        error = np.linalg.norm(coarse[['x', 'y']].to_numpy(float)-truth[['x', 'y']].to_numpy(float), axis=1)
        result['worse_than_coarse_fraction'] = float(np.mean(distance > error+1e-9))
    if roi is not None:
        roi = align(roi, truth.image_id)
        difference = np.abs(truth[['x', 'y']].to_numpy(float)-roi[['crop_x', 'crop_y']].to_numpy(float))
        result['roi_coverage'] = float((difference <= roi.crop_side.to_numpy(float)[:, None]/2).all(axis=1).mean())
    return result


def verify_metric_record(saved, expected, label):
    for key, value in expected.items():
        close(saved[key], value, f'{label}.{key}', 1e-6)


def verify_manifest(folder, records):
    folder = Path(folder).resolve()
    require(isinstance(records, dict) and records, f'{folder}: empty manifest')
    for relative, entry in records.items():
        path = (folder/relative).resolve()
        require(path.is_relative_to(folder), f'Manifest escape: {relative}')
        require(path.is_file() and path.stat().st_size == entry['bytes'], f'File size: {path}')
        require(sha(path) == entry['sha256'], f'File digest: {path}')


def verify_roi(roi, centres, truth, label):
    roi, centres = align(roi, truth.image_id), align(centres, truth.image_id)
    require(np.isfinite(roi[['crop_x', 'crop_y', 'crop_side']].to_numpy(float)).all(), label)
    side = .15*truth[['width', 'height']].min(axis=1).to_numpy(float)
    require(np.max(np.abs(roi.crop_side.to_numpy(float)-side)) < 1e-8, f'{label}: side not exact .15')
    columns = ['crop_x', 'crop_y'] if 'crop_x' in centres else ['x', 'y']
    require(np.max(np.abs(roi[['crop_x', 'crop_y']].to_numpy(float)-centres[columns].to_numpy(float))) < 1e-8,
            f'{label}: prediction centres changed')
    # ROI definition contains no target-derived correction; coverage is evaluation only.
    if not {'x', 'y'}.issubset(truth.columns):
        return {'geometry_verified': True, 'label_coverage_unavailable': True}
    delta = np.abs(truth[['x', 'y']].to_numpy(float)-roi[['crop_x', 'crop_y']].to_numpy(float))
    return {'local_coverage': float((delta <= side[:, None]/2).all(axis=1).mean()),
            'context_coverage': float((delta <= side[:, None]).all(axis=1).mean())}


def risk_arrays(coarse, fine):
    coarse = align(coarse, fine.image_id)
    features = fine[['scale_disagreement_norm', 'tta_disagreement_norm', 'secondary_peak_ratio']].to_numpy(float)
    require(np.isfinite(features).all() and (features >= 0).all(), 'Invalid risk features')
    require((features[:, 2] <= 1+1e-7).all(), 'Secondary-peak ratio >1')
    scale, tta, peak = features.T
    trigger = (scale > .05) | ((peak > .80) & (scale > .025)) | (tta > .025)
    c, f = coarse[['x', 'y']].to_numpy(float), fine[['x', 'y']].to_numpy(float)
    weights = {'fine_only': np.zeros(len(f)), 'risk_half': trigger.astype(float)*.5,
               'risk_coarse': trigger.astype(float), 'global_quarter': np.full(len(f), .25)}
    arrays = {name: weights[name][:, None]*c+(1-weights[name][:, None])*f for name in POLICIES}
    return arrays, weights, trigger


def risk_metric(pred, target):
    delta = np.asarray(pred)-np.asarray(target)
    dist = np.linalg.norm(delta, axis=1)
    return {'rmse': float(np.sqrt(np.mean(delta**2))), 'p90_distance_px': float(np.quantile(dist, .9)),
            'max_distance_px': float(dist.max()), 'rows': len(delta), 'sse': float((delta**2).sum())}


def audit_risk(coarse, fine, truth, rule):
    """Recompute fixed policies and group leave-one-out without producer helpers."""
    truth = align(truth, fine.image_id)
    arrays, weights, trigger = risk_arrays(coarse, fine)
    target = truth[['x', 'y']].to_numpy(float)
    groups = truth.group.astype(str).to_numpy()
    unique = sorted(set(groups))
    constants = rule['feature_constants']
    fixed = {'probe_side_ratio': 1.25, 'secondary_peak_exclusion_radius_cells': 3, 'scale_threshold': .05,
             'peak_ratio_threshold': .80, 'peak_scale_threshold': .025, 'tta_threshold': .025,
             'minimum_absolute_gain_px': .5, 'minimum_relative_gain': .02,
             'policies_in_tie_break_order': list(POLICIES)}
    require(constants == fixed, 'Risk constants changed')
    require(rule['calibration_ids'] == truth.image_id.tolist() and rule['calibration_group_ids'] == unique,
            'Risk calibration identity')
    require(rule['calibration_images'] == len(truth) and rule['calibration_groups'] == len(unique), 'Risk calibration counts')
    require(rule['minimum_groups'] == 12, 'Risk minimum groups changed')
    full = [risk_metric(arrays[name], target) for name in POLICIES]
    require([r['policy'] for r in rule['candidate_scores']] == list(POLICIES), 'Risk policy order')
    for name, actual, expected in zip(POLICIES, rule['candidate_scores'], full):
        verify_metric_record(actual, expected, 'risk candidate '+name)
        require(actual['risk_trigger_rows'] == int(trigger.sum()), 'Risk trigger count')
        require(actual['nonzero_coarse_weight_rows'] == int((weights[name] > 0).sum()), 'Risk weight count')
        require(actual['changed_prediction_rows'] == int(np.any(arrays[name] != arrays['fine_only'], axis=1).sum()), 'Risk changed count')
    selected = min(range(4), key=lambda i: (full[i]['rmse'], i))
    require(rule['selected_policy'] == POLICIES[selected], 'Full calibration policy')
    require(len(rule['crossfit_audit']) == len(unique), 'Risk group crossfit count')
    held_predictions = np.zeros_like(target)
    for group, record in zip(unique, rule['crossfit_audit']):
        held = groups == group
        fit = ~held
        require(record['held_group'] == group and record['fit_groups'] == sorted(set(groups[fit])), 'Risk held-group identity')
        require(record['held_image_ids'] == truth.image_id[held].tolist() and record['fit_image_ids'] == truth.image_id[fit].tolist(),
                'Risk image-role identity')
        candidates = [risk_metric(arrays[name][fit], target[fit]) for name in POLICIES]
        for actual, expected in zip(record['fitting_candidate_scores'], candidates):
            verify_metric_record(actual, expected, 'risk crossfit fit')
        winner = min(range(4), key=lambda i: (candidates[i]['rmse'], i))
        require(record['selected_policy'] == POLICIES[winner], 'Risk crossfit selected policy')
        held_predictions[held] = arrays[POLICIES[winner]][held]
        verify_metric_record(record['held_metrics'], risk_metric(held_predictions[held], target[held]), 'risk crossfit held')
    cv = risk_metric(held_predictions, target)
    verify_metric_record(rule['crossfit_selected_procedure_metrics'], cv, 'risk crossfit combined')
    gain = full[0]['rmse']-cv['rmse']
    close(rule['crossfit_gain_px'], gain, 'risk gain')
    enabled = bool(len(unique) >= 12 and gain >= .5 and gain >= .02*full[0]['rmse'] and
                   cv['p90_distance_px'] <= full[0]['p90_distance_px']+1e-9 and selected != 0)
    require(rule['enabled'] == enabled, 'Risk enabled decision')
    require(rule['effective_policy'] == (POLICIES[selected] if enabled else 'fine_only'), 'Risk effective policy')
    return {'enabled': enabled, 'policy': rule['effective_policy'], 'calibration_groups': len(unique),
            'calibration_risk_trigger_fraction': float(trigger.mean()), 'crossfit_gain_px': gain}


def risk_apply_independent(coarse, fine, rule):
    arrays, weights, trigger = risk_arrays(coarse, fine)
    name = rule['selected_policy'] if rule['enabled'] else 'fine_only'
    out = fine[['image_id', 'x', 'y']].copy()
    out[['x', 'y']] = arrays[name]
    return out, weights[name], trigger


def stage_check(folder, fitting, stopping, stage_cfg, stage, seed, epochs, train_crop=None, val_crop=None):
    record = js(folder/'stage_complete.json')
    verify_manifest(folder, record['files'])
    inputs = frame(folder/'actual_training_inputs.csv')
    require(inputs.image_id.tolist() == fitting.image_id.tolist(), f'{folder}: training row order')
    for column in ('group', 'pixel_hash'):
        require(inputs[column].astype(str).tolist() == fitting[column].astype(str).tolist(), f'{folder}: training {column}')
    if stopping is None:
        require(not (folder/'actual_early_stop_inputs.csv').exists(), f'{folder}: unexpected stop records')
    else:
        stopped = frame(folder/'actual_early_stop_inputs.csv')
        require(stopped.image_id.tolist() == stopping.image_id.tolist(), f'{folder}: stopping identity/order')
        require(not set(inputs.group) & set(stopped.group), f'{folder}: group overlap')
    if train_crop is not None:
        saved = align(frame(folder/'actual_training_crops.csv'), train_crop.image_id)
        require(np.max(np.abs(saved[['crop_x', 'crop_y', 'crop_side']].to_numpy(float)-train_crop[['crop_x', 'crop_y', 'crop_side']].to_numpy(float))) < 1e-8,
                f'{folder}: training crop mismatch')
    signature = m.research_stage_signature(fitting, stopping, stage_cfg, stage, seed, epochs, train_crop, val_crop)
    require(signature == record['signature'], f'{folder}: stage signature')
    info, history = js(folder/'training_summary.json'), pd.read_csv(folder/'history.csv')
    require(len(history) > 0 and history.epoch.tolist() == list(range(1, len(history)+1)), f'{folder}: epoch sequence')
    require(np.isfinite(history.train_loss.to_numpy(float)).all(), f'{folder}: nonfinite loss')
    require(info['successful_optimizer_steps'] == int(history.successful_optimizer_steps.sum()) > 0, f'{folder}: no updates')
    require(info['training_images'] == len(fitting), f'{folder}: sample count')
    require(info['fixed_epochs'] == (epochs is not None), f'{folder}: refit flag')
    if epochs is not None:
        require(info['best_epoch'] == epochs == len(history), f'{folder}: fixed epochs')
    else:
        require(np.isfinite(history.val_rmse.to_numpy(float)).all(), f'{folder}: invalid stop loss')
        best = int(history.val_rmse.to_numpy(float).argmin())
        require(info['best_epoch'] == best+1, f'{folder}: incorrect best epoch')
        close(info['best_rmse'], history.iloc[best].val_rmse, f'{folder}: best score')
    if stage == 'fine':
        require(info['validation_metric'] == 'inner RMSE' and info['fine_multiscale_train'] is False,
                f'{folder}: scale selection mismatch')
    return info


def verify_source_and_protection():
    source = js(WORK/'source_manifest.json')
    require(source['engine_sha256'] == sha(WORK/'engine.py'), 'Frozen engine changed')
    parts = source.get('parts', source.get('source_parts'))
    require(isinstance(parts, dict) and parts, 'Source part manifest missing')
    for name, digest in parts.items():
        require(sha(ROOT/name) == digest, f'Source changed: {name}')
    protected = js(OUT/'protected_previous.json')['files']
    verify_manifest(ROOT, protected)
    for run in (OLD, SEED):
        for fold in range(5):
            for relative in ('coarse_final/model.pt', 'fine_final/model.pt', 'B4_oof.csv', 'B4_test.csv'):
                path = run/f'fold_{fold}'/relative
                require(path.relative_to(ROOT).as_posix() in protected, f'Critical old file not protected: {path}')
    require(sha(OUT/'baseline_8p2.csv') == BASELINE_SHA, '8.2 baseline altered')
    return len(protected)


def record_audit():
    execution, protocol = js(OUT/'execution_report.json'), js(OUT/'run_protocol.json')
    require(execution['status'] == 'COMPLETE' and execution['completed_folds'] == 10 and execution['new_training_stages'] == 65,
            'Experiment incomplete')
    require(execution['split_seeds'] == protocol['split_seeds'] == [2026, 2027], 'Wrong split seeds')
    require(protocol['local_short_side_fraction'] == .15 and protocol['context_ratio'] == 2 and
            protocol['gate_min_px'] == .5 and protocol['gate_min_relative'] == .02 and protocol['new_branch_weight'] == .2,
            'Frozen treatment or gate changed')
    require(protocol['test_reference_labels_used'] is False, 'Unexpected external labels flag')
    test = frame(OUT/'test_manifest.csv')
    require(len(test) == 20 and test.image_id.is_unique and set(test.image_id) == set(map(str, range(81, 101))), 'Test IDs')
    require(not {'x', 'y', 'x_true', 'y_true'} & set(test.columns), 'Test manifest contains targets')
    input_audit = OUT/'verification_input_audit'
    input_audit.mkdir(exist_ok=True)
    prepared_train, prepared_test, _, _, _ = m.prepare_tables(replace(cfg, OUTPUT_DIR=str(input_audit)))
    require(set(prepared_train.image_id) == set(map(str, range(1, 81))), 'Train IDs')
    require(not set(prepared_train.image_id) & set(test.image_id) and
            not set(prepared_train.pixel_hash) & set(test.pixel_hash) and
            not set(prepared_train.path) & set(test.path), 'Train/test overlap')
    require(align(prepared_test, test.image_id).pixel_hash.tolist() == test.pixel_hash.tolist(), 'Test bytes changed')
    protected_count = verify_source_and_protection()
    records, stage_count = [], 0
    frames = {}
    for split in (2026, 2027):
        split_folder = OUT/f'split_{split}'
        train = frame(split_folder/'folds.csv'); frames[split] = train
        expected = m.assign_folds(prepared_train, replace(cfg, SEED=split))
        expected = align(expected, train.image_id)
        for col in ('group', 'fold', 'pixel_hash', 'x', 'y', 'width', 'height'):
            require(np.array_equal(train[col].to_numpy(), expected[col].to_numpy()), f'Split {split}: changed {col}')
        require(train.groupby('group').fold.nunique().max() == 1, 'Group crosses folds')
        if split == 2027:
            require(not list(split_folder.rglob('*test*.csv')), 'Secondary has test predictions')
        for fold in range(5):
            folder = split_folder/f'fold_{fold}'
            complete = js(folder/'complete.json')
            require(complete['status'] == 'COMPLETE', 'Missing fold completion')
            verify_manifest(folder, complete['files'])
            fold_cfg = m.DualConfig(**js(folder/'fold_config.json')['config'])
            require(fold_cfg.SEED == split and fold_cfg.CONTEXT_RATIO == 2 and fold_cfg.HEATMAP_SIGMA == 2.5,
                    'Fold coarse configuration mismatch')
            require(fold_cfg.FINE_MULTISCALE_TRAIN is False and fold_cfg.FINE_MULTISCALE_EARLY_STOP is False,
                    'Fine scale flags changed')
            roles = frame(folder/'split_roles.csv')
            require(roles.image_id.is_unique and set(roles.image_id) == set(train.image_id), 'Role IDs')
            require(roles.groupby('group').role.nunique().max() == 1, 'Group crosses roles')
            def take(name):
                return train.loc[train.image_id.isin(roles.loc[roles.role == name, 'image_id'])].reset_index(drop=True)
            fit, stop, cal, outer = [take(r) for r in ('model_fit', 'early_stop', 'calibration_only', 'outer_validation')]
            development = train.loc[train.image_id.isin(set(fit.image_id) | set(stop.image_id))].reset_index(drop=True)
            require(set(outer.image_id) == set(train.loc[train.fold == fold, 'image_id']), 'Outer role assignment')
            if split == 2026:
                old_roles = frame(SEED/f'fold_{fold}'/'split_roles.csv')
                require(roles.set_index('image_id').loc[old_roles.image_id, 'role'].tolist() == old_roles.role.tolist(), 'Primary roles changed')
            else:
                avail = train.loc[train.fold != fold].reset_index(drop=True)
                dev_expected, cal_expected = m.research_group_holdout(avail, fold_cfg.CALIBRATION_FRACTION, split+fold*1000)
                fit_expected, stop_expected = m.research_group_holdout(dev_expected, fold_cfg.EARLY_STOP_FRACTION, split+fold*1000+1)
                for actual, expected_role in ((fit, fit_expected), (stop, stop_expected), (cal, cal_expected)):
                    require(actual.image_id.tolist() == expected_role.image_id.tolist(), 'Secondary roles changed')
            audit = js(folder/'crop_crossfit_audit.json')
            held_ids = []
            for a in audit:
                fit_ids, held = set(a['fit_ids']), set(a['held_ids'])
                require(fit_ids | held == set(fit.image_id) and not fit_ids & held, 'Crossfit image partition')
                require(not set(a['fit_groups']) & set(a['held_groups']), 'Crossfit group overlap')
                require(not (fit_ids | held) & set(stop.image_id), 'Stop data in crop crossfit')
                held_ids.extend(a['held_ids'])
            require(len(held_ids) == len(set(held_ids)) == len(fit) and set(held_ids) == set(fit.image_id), 'Crossfit coverage')
            centres = frame(folder/'training_centres.csv')
            require(centres.image_id.tolist() == development.image_id.tolist(), 'Training centre order')
            lineage = js(folder/'coarse_lineage.json')
            coarse = {name: frame(folder/f'coarse_{name}.csv') for name in ('calibration', 'oof') + (('test',) if split == 2026 else ())}
            if split == 2026:
                require(lineage['mode'] == 'verified_primary_reuse', 'Primary coarse lineage')
                verify_manifest(Path(lineage['source']), lineage['files'])
                inherited = frame(SEED/f'fold_{fold}'/'fine_final/actual_training_crops.csv')
                verify_roi(centres, inherited, development, 'primary centres')
                for name, original in [('calibration', 'calibration_coarse.csv'), ('oof', 'B0_oof.csv'), ('test', 'B0_test.csv')]:
                    compare_xy(coarse[name], frame(SEED/f'fold_{fold}'/original), 'Inherited coarse '+name, 1e-8)
            else:
                require(lineage['mode'] == 'new_secondary_models', 'Secondary lineage')
                base_seed = split+fold*1000+fold_cfg.MODEL_SEED_OFFSET
                info = stage_check(folder/'coarse_select', fit, stop, fold_cfg, 'coarse', base_seed+10, None)
                stage_count += 1
                for a in audit:
                    inner_cfg = replace(fold_cfg, N_FOLDS=fold_cfg.INNER_CROP_FOLDS, SEED=split+fold*1000+100)
                    inner = m.assign_folds(fit, inner_cfg)
                    inner_id = a['inner_fold']
                    fitting = inner.loc[inner.fold != inner_id].reset_index(drop=True)
                    stage_check(folder/'crop_crossfit'/f'inner_{inner_id}', fitting, None, fold_cfg, 'coarse',
                                base_seed+100+inner_id, int(info['best_epoch']))
                    stage_count += 1
                stage_check(folder/'coarse_final', development, None, fold_cfg, 'coarse', base_seed+30, int(info['best_epoch']))
                stage_count += 1
                fitting_pred = frame(folder/'crop_crossfit/training_oof_centres.csv')
                verify_roi(centres.loc[centres.image_id.isin(fit.image_id)], fitting_pred, fit, 'secondary fitting centres')
                require(lineage['coarse_epoch'] == info['best_epoch'], 'Secondary coarse epoch')
            fit_crop = centres.loc[centres.image_id.isin(fit.image_id)]
            stop_crop = centres.loc[centres.image_id.isin(stop.image_id)]
            selection = js(folder/'selection.json')
            require(complete['selection'] == selection == execution['decisions'][str(split)][fold], 'Fold selection identity')
            arms = {}
            for arm in ('single', 'dual'):
                arm_folder = folder/arm
                arm_cfg = m.DualConfig(**js(arm_folder/'config.json')['config'])
                expected_cfg = replace(fold_cfg, FINE_VARIANT=arm, HEATMAP_SIGMA=1.5, OUTPUT_DIR=str(arm_folder))
                require(asdict(arm_cfg) == asdict(expected_cfg), 'Arm config changed beyond declared treatment')
                require(arm_cfg.FINE_INPUT_SIZE == 256 and arm_cfg.FINE_HEATMAP_SIZE == 64 and arm_cfg.BATCH_SIZE == 8, 'Arm sizes')
                actual_seed = split+fold*1000+20+arm_cfg.MODEL_SEED_OFFSET
                info = stage_check(arm_folder/'fine_select', fit, stop, arm_cfg, 'fine', actual_seed, None, fit_crop, stop_crop)
                selected = selection['arms'][arm]
                require(selected['best_epoch'] == info['best_epoch'] and selected['stopping_ids'] == stop.image_id.tolist(), 'Arm selection')
                close(selected['best_rmse'], info['best_rmse'], 'Arm best RMSE')
                stop_prediction = frame(arm_folder/'selected_stop_predictions.csv')
                close(metric(stop_prediction, stop)['rmse'], info['best_rmse'], 'Saved stopping predictions', 1e-3)
                stage_check(arm_folder/'fine_final', development, None, arm_cfg, 'fine', actual_seed+20,
                            int(info['best_epoch']), centres)
                stage_count += 2
                arms[arm] = arm_cfg
            gain = selection['arms']['single']['best_rmse']-selection['arms']['dual']['best_rmse']
            accepted = bool(gain >= .5 and gain >= .02*selection['arms']['single']['best_rmse'])
            close(selection['gain_px'], gain, 'Dual gate gain')
            require(selection['accepted'] == accepted and selection['minimum_gain_px'] == .5 and
                    selection['minimum_relative_gain'] == .02 and selection['weight_if_accepted'] == .2, 'Architecture gate')
            cal_features = frame(folder/'calibration_dual_features.csv')
            roi_cal = frame(folder/'calibration_roi.csv')
            verify_roi(roi_cal, coarse['calibration'], cal, 'Calibration ROI')
            rule = js(folder/'risk_rule.json')
            risk_record = audit_risk(coarse['calibration'], cal_features, cal, rule)
            metrics = frame(folder/'metrics.csv').set_index('method')
            coverage = None
            for kind, truth in [('oof', outer)] + ([('test', test)] if split == 2026 else []):
                preds = {name: frame(folder/f'{name}_{kind}.csv') for name in METHODS+(PRIMARY_EXTRA if split == 2026 else ())}
                roi = frame(folder/f'{kind}_roi.csv')
                if kind == 'oof':
                    coverage = verify_roi(roi, coarse[kind], truth, 'Outer ROI')
                else:
                    verify_roi(roi, coarse[kind], truth, 'Test ROI geometry')
                compare_xy(preds['COARSE'], coarse[kind], 'Coarse copy', 1e-8)
                risk_pred, weights, trigger = risk_apply_independent(coarse[kind], preds['DUAL_RAW'], rule)
                compare_xy(preds['DUAL_RISK'], risk_pred, 'Independent risk application', 1e-8)
                require(np.array_equal(preds['DUAL_RISK'].coarse_weight.to_numpy(float), weights), 'Saved coarse weights')
                require(np.array_equal(preds['DUAL_RISK'].risk_trigger.to_numpy(bool), trigger), 'Saved risk flags')
                expected_deploy = blend(preds['SINGLE'], risk_pred, .2) if accepted else preds['SINGLE']
                compare_xy(preds['MATCHED_DEPLOY'], expected_deploy, 'Matched deployment blend', 1e-8)
                if split == 2026:
                    baseline = blend(frame(OLD/f'fold_{fold}'/f'B4_{kind}.csv'), frame(SEED/f'fold_{fold}'/f'B4_{kind}.csv'), .25)
                    compare_xy(preds['BASELINE8P2'], baseline, '8.2 fold composition', 1e-8)
                    expected_submission = blend(baseline, risk_pred, .2) if accepted else baseline
                    compare_xy(preds['SUBMISSION_CANDIDATE'], expected_submission, 'Submission fold composition', 1e-8)
                if kind == 'oof':
                    for name, pred in preds.items():
                        labels = align(truth, pred.image_id)
                        require(np.array_equal(pred[['x_true', 'y_true']].to_numpy(float), labels[['x', 'y']].to_numpy(float)), 'OOF truth changed')
                        verify_metric_record(metrics.loc[name], metric(pred, truth, coarse[kind], roi), name+' OOF metrics')
            records.append({'split_seed': split, 'fold': fold, 'accepted': accepted, 'risk': risk_record,
                            'coverage': coverage, 'fine_training_images': len(development), 'calibration_images': len(cal)})
            print(f'Audit split {split} fold {fold+1}/5 PASS; new stages checked={stage_count}/65', flush=True)
    require(stage_count == 65 and len(list(OUT.rglob('stage_complete.json'))) == 65, 'Training stage count')
    require(np.any(align(frames[2026], frames[2027].image_id).fold.to_numpy() != frames[2027].fold.to_numpy()), 'Secondary split identical')
    summary = frame(OUT/'ablation_summary.csv').set_index(['split_seed', 'method'])
    for split, train in frames.items():
        folder = OUT/f'split_{split}'
        coarse = frame(folder/'oof_COARSE.csv')
        roi = pd.concat([frame(folder/f'fold_{k}'/'oof_roi.csv') for k in range(5)], ignore_index=True)
        for name in METHODS+(PRIMARY_EXTRA if split == 2026 else ()):
            saved = frame(folder/f'oof_{name}.csv')
            combined = pd.concat([frame(folder/f'fold_{k}'/f'{name}_oof.csv') for k in range(5)], ignore_index=True)
            require(len(saved) == 80 and saved.image_id.is_unique, 'OOF row count')
            compare_xy(saved, combined, 'Combined OOF', 1e-8)
            verify_metric_record(summary.loc[(split, name)], metric(saved, train, coarse, roi), 'Combined metric')
            if split == 2026:
                folds = frame(folder/f'test_folds_{name}.csv')
                combined_test = pd.concat([frame(folder/f'fold_{k}'/f'{name}_test.csv') for k in range(5)], ignore_index=True)
                require(len(folds) == 100 and (folds.groupby('image_id').size() == 5).all(), 'Test-fold row count')
                require(folds[['image_id', 'fold']].values.tolist() == combined_test[['image_id', 'fold']].values.tolist(), 'Test-fold order')
                require(np.max(np.abs(folds[['x', 'y']].to_numpy(float)-combined_test[['x', 'y']].to_numpy(float))) < 1e-8, 'Test-fold values')
                average = combined_test.groupby('image_id', as_index=False)[['x', 'y']].mean()
                compare_xy(frame(folder/f'test_ensemble_{name}.csv'), average, 'Five-fold average', 1e-8)
    return {'stages': stage_count, 'protected_files': protected_count, 'folds': records,
            'secondary_split_differs': True, 'secondary_test_predictions': 0}, frames, test


def reload_check(frames, test):
    require(torch.cuda.is_available(), 'GPU required for independent reload')
    m.seed_everything(cfg.SEED); m.cv2.setNumThreads(0)
    count, maximum, records, submission_folds = 0, 0., [], []
    for split, train in frames.items():
        for fold in range(5):
            folder = OUT/f'split_{split}'/f'fold_{fold}'
            fold_cfg = m.DualConfig(**js(folder/'fold_config.json')['config'])
            roles = frame(folder/'split_roles.csv')
            def take(role):
                return train.loc[train.image_id.isin(roles.loc[roles.role == role, 'image_id'])].reset_index(drop=True)
            cal, outer = take('calibration_only'), take('outer_validation')
            kinds = [('calibration', cal), ('oof', outer)] + ([('test', test)] if split == 2026 else [])
            both = pd.concat([value for _, value in kinds], ignore_index=True)
            coarse_cfg = replace(fold_cfg, PRETRAINED=False)
            model = m.research_build_model(coarse_cfg, 'coarse', 'cuda')
            checkpoint = Path(js(folder/'coarse_lineage.json')['coarse_checkpoint'])
            m.load_model_checkpoint(model, checkpoint, 'cuda')
            # Keep each producer call's batch partition. This avoids changing
            # the final short-batch convolution shape during reproducibility checks.
            fresh_coarse = pd.concat([m.predict_research(model, value, coarse_cfg, 'cuda', 'coarse')
                                      for _, value in kinds], ignore_index=True)
            m._research_cleanup(model); del model; count += 1
            for kind, truth in kinds:
                part = fresh_coarse.loc[fresh_coarse.image_id.isin(truth.image_id)]
                maximum = max(maximum, compare_xy(part, frame(folder/f'coarse_{kind}.csv'), 'Reload coarse'))
            rois = pd.concat([frame(folder/('calibration_roi.csv' if kind == 'calibration' else f'{kind}_roi.csv')) for kind, _ in kinds], ignore_index=True)
            fresh = {}
            for arm in ('single', 'dual'):
                arm_cfg = m.DualConfig(**js(folder/arm/'config.json')['config'])
                arm_cfg = replace(arm_cfg, PRETRAINED=False)
                model = m.research_build_model(arm_cfg, 'fine', 'cuda')
                m.load_model_checkpoint(model, folder/arm/'fine_final/model.pt', 'cuda')
                prediction_parts = []
                for kind, truth in kinds:
                    if arm == 'single' and kind == 'calibration':
                        continue
                    part_roi = rois.loc[rois.image_id.isin(truth.image_id)]
                    prediction_parts.append(m.predict_research(model, truth, arm_cfg, 'cuda', 'fine', part_roi)
                                            if arm == 'single' else
                                            m.dual_predict_features(model, truth, arm_cfg, 'cuda', part_roi))
                prediction = pd.concat(prediction_parts, ignore_index=True)
                if arm == 'dual':
                    # Independent base decoder agrees with diagnostic inference x/y.
                    # Keep the same complete calibration frame and batch shape.
                    # Taking only two images changes AMP convolution numerics and
                    # tests batch-size invariance instead of decoder equivalence.
                    diagnostic_frame = cal
                    diagnostic_roi = rois.loc[rois.image_id.isin(diagnostic_frame.image_id)]
                    simple = m.predict_research(model, diagnostic_frame, arm_cfg, 'cuda', 'fine', diagnostic_roi)
                    compare_xy(simple, prediction.loc[prediction.image_id.isin(diagnostic_frame.image_id)], 'Diagnostic decoder agreement')
                m._research_cleanup(model); del model; count += 1
                fresh[arm] = prediction
                for kind, truth in kinds:
                    if arm == 'single' and kind == 'calibration':
                        continue
                    part = prediction.loc[prediction.image_id.isin(truth.image_id)]
                    saved = frame(folder/('calibration_dual_features.csv' if kind == 'calibration' else f'{"SINGLE" if arm == "single" else "DUAL_RAW"}_{kind}.csv'))
                    maximum = max(maximum, compare_xy(part, saved, 'Reload '+arm+' '+kind))
                    if arm == 'dual':
                        saved = align(saved, part.image_id)
                        for col in ('scale_disagreement_norm', 'tta_disagreement_norm', 'secondary_peak_ratio'):
                            require(np.max(np.abs(part[col].to_numpy(float)-saved[col].to_numpy(float))) < 1e-5, 'Reload risk feature '+col)
            rule, selection = js(folder/'risk_rule.json'), js(folder/'selection.json')
            # Reevaluate calibration decisions using freshly reloaded model outputs.
            cal_coarse = fresh_coarse.loc[fresh_coarse.image_id.isin(cal.image_id)]
            cal_dual = fresh['dual'].loc[fresh['dual'].image_id.isin(cal.image_id)]
            audit_risk(cal_coarse, cal_dual, cal, rule)
            for kind, truth in kinds[1:]:
                cp = fresh_coarse.loc[fresh_coarse.image_id.isin(truth.image_id)]
                fp = fresh['dual'].loc[fresh['dual'].image_id.isin(truth.image_id)]
                sp = fresh['single'].loc[fresh['single'].image_id.isin(truth.image_id)]
                rp, _, _ = risk_apply_independent(cp, fp, rule)
                maximum = max(maximum, compare_xy(rp, frame(folder/f'DUAL_RISK_{kind}.csv'), 'Reload risk result'))
                deployed = blend(sp, rp, .2) if selection['accepted'] else sp
                maximum = max(maximum, compare_xy(deployed, frame(folder/f'MATCHED_DEPLOY_{kind}.csv'), 'Reload matched result'))
                if split == 2026:
                    base = frame(folder/f'BASELINE8P2_{kind}.csv')
                    result = blend(base, rp, .2) if selection['accepted'] else base
                    maximum = max(maximum, compare_xy(result, frame(folder/f'SUBMISSION_CANDIDATE_{kind}.csv'), 'Reload submission result'))
                    if kind == 'test':
                        submission_folds.append(result.assign(fold=fold))
            records.append({'split_seed': split, 'fold': fold, 'reloaded_models': 3})
            print(f'Reload split {split} fold {fold+1}/5 PASS; networks={count}/30', flush=True)
    require(count == 30, 'Expected 30 final networks reloaded')
    average = pd.concat(submission_folds).groupby('image_id', as_index=False)[['x', 'y']].mean()
    return {'networks': count, 'max_prediction_difference_px': maximum, 'folds': records}, average


def submission_check(fresh, test):
    submitted = pd.read_csv(OUT/'submission.csv')
    baseline = pd.read_csv(OUT/'baseline_8p2.csv')
    require(list(submitted.columns) == ['ImageID', 'value'] and len(submitted) == 40, 'Submission format')
    require(submitted.ImageID.is_unique and submitted.ImageID.tolist() == baseline.ImageID.tolist(), 'Submission identifier order')
    coords = fresh.set_index('image_id'); sizes = test.set_index('image_id')
    expected = []
    for name, value in zip(submitted.ImageID, submitted.value):
        image_id, axis = name.split('_Fovea_')
        require(axis in ('X', 'Y') and image_id in coords.index, 'Unknown submission coordinate')
        require(np.isfinite(value) and 0 <= value <= sizes.loc[image_id, 'width' if axis == 'X' else 'height']-1, 'Submission range')
        expected.append(float(coords.loc[image_id, 'x' if axis == 'X' else 'y']))
    delta = float(np.abs(np.asarray(expected)-submitted.value.to_numpy(float)).max())
    require(delta < TOLERANCE_PX, 'Reloaded CSV mismatch')
    decisions = js(OUT/'execution_report.json')['decisions']['2026']
    if not any(row['accepted'] for row in decisions):
        require(sha(OUT/'submission.csv') == BASELINE_SHA, 'All rejected but baseline not retained byte-for-byte')
    require(sha(OUT/'submission.csv') == js(OUT/'execution_report.json')['submission_sha256'], 'Submission digest changed')
    return delta


def main():
    started = time.perf_counter()
    try:
        audited, frames, test = record_audit()
        reloads, fresh = reload_check(frames, test)
        csv_delta = submission_check(fresh, test)
        protected = verify_source_and_protection()
        report = {'status': 'PASS', 'new_training_stages': 65, 'completed_outer_folds': 10,
                  'reloaded_final_networks': reloads['networks'], 'baseline_final_networks_unchanged': 20,
                  'protected_previous_files_unchanged': protected, 'protected_previous_verified_files': protected,
                  'baseline_sha256': BASELINE_SHA,
                  'submission_sha256': sha(OUT/'submission.csv'), 'source_engine_sha256': sha(WORK/'engine.py'),
                  'test_images_in_training_or_selection': 0, 'secondary_submission_influence': False,
                  'submission_rows': 40, 'test_images': 20, 'independent_reload_tolerance_px': TOLERANCE_PX,
                  'submission_max_reload_difference_px': csv_delta,
                  'max_reload_difference_px': max(csv_delta, reloads['max_prediction_difference_px']),
                  'accepted_primary_folds_zero_based': [r['fold'] for r in audited['folds'] if r['split_seed'] == 2026 and r['accepted']],
                  'accepted_folds_by_split': {str(split): [r['fold'] for r in audited['folds'] if r['split_seed'] == split and r['accepted']]
                                               for split in (2026, 2027)},
                  'calibration_rule_records': [{'split_seed': r['split_seed'], 'fold': r['fold'], **r['risk']}
                                               for r in audited['folds']],
                  'record_audit': audited, 'reload_audit': reloads, 'seconds': time.perf_counter()-started,
                  'limitations': 'Saved source/records and actual checkpoint inference were verified; no official private labels or leaderboard were used. Secondary is repeated development, not an independent patient test.'}
    except Exception as error:
        report = {'status': 'FAIL', 'error': str(error), 'traceback': traceback.format_exc(), 'seconds': time.perf_counter()-started}
        (OUT/'validation_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        raise
    (OUT/'validation_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'status': report['status'], 'new_training_stages': 65, 'reloaded_networks': 30,
                      'candidate_max_difference_px': csv_delta, 'accepted_folds': report['accepted_primary_folds_zero_based'],
                      'seconds': report['seconds']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
