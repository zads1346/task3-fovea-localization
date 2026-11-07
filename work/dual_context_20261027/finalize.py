"""Report completed two-split experiments; never train, tune or score test labels.

Importing this module is read-only. Only main() writes the new experiment
directory, and it requires completed training, model reload, and notebook
validation before producing final deliverables.
"""
from pathlib import Path
import hashlib
import json
import shutil
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[2]
WORK = Path(__file__).resolve().parent
OUT = ROOT / 'outputs/dual_context_20261027'
NOTEBOOK = 'fovea_dual_context_kaggle.ipynb'
SPLITS = (2026, 2027)
METHODS = ('COARSE', 'SINGLE', 'DUAL_RAW', 'DUAL_RISK', 'MATCHED_DEPLOY')
PRIMARY_EXTRA = ('BASELINE8P2', 'SUBMISSION_CANDIDATE')
EXPECTED_BASELINE = 'be3cba8f60f96325c7730480db2f740522e79ac550368984174a73396342176f'


def require(condition, message):
    if not bool(condition):
        raise RuntimeError(message)


def js(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def main():
    sys.path.insert(0, str(ROOT / 'work/gpu_test_deps'))
    import numpy as np
    import pandas as pd

    def read(path):
        return pd.read_csv(path, dtype={'image_id': str, 'group': str, 'ImageID': str})

    def align(reference, candidate):
        require(reference.image_id.is_unique and candidate.image_id.is_unique, 'Repeated comparison IDs')
        require(set(reference.image_id) == set(candidate.image_id), 'Comparison ID sets differ')
        return candidate.set_index('image_id').loc[reference.image_id].reset_index()

    def metrics(prediction, truth):
        residual = prediction - truth
        require(residual.shape == (80, 2) and np.isfinite(residual).all(), 'Expected 80 finite OOF coordinate pairs')
        distance = np.linalg.norm(residual, axis=1)
        return {'rmse': float(np.sqrt(np.mean(residual ** 2))),
                'mean_distance_px': float(distance.mean()), 'median_distance_px': float(np.median(distance)),
                'p90_distance_px': float(np.quantile(distance, .9)),
                'p95_distance_px': float(np.quantile(distance, .95)),
                'max_distance_px': float(distance.max()), 'images': 80}

    execution = js(OUT / 'execution_report.json')
    validation = js(OUT / 'validation_report.json')
    protocol = js(OUT / 'run_protocol.json')
    planned = js(OUT / 'planned_experiment.json')
    notebook_validation = js(OUT / 'notebook_validation.json')
    manifest = js(WORK / 'source_manifest.json')
    protected = js(OUT / 'protected_previous.json')
    backup = js(OUT / 'existing_baseline_backup.json')
    require(execution['status'] == 'COMPLETE' and execution['completed_folds'] == 10,
            'Both complete fivefold experiments are required')
    require(execution['split_seeds'] == protocol['split_seeds'] == list(SPLITS), 'Split protocol changed')
    require(execution['new_training_stages'] == protocol['new_training_stages'] == planned['total_new_stages'] == 65,
            'Expected 40 new fine stages plus 25 secondary coarse stages')
    require(validation['status'] == 'PASS', 'Independent model reload validation did not pass')
    require(validation['reloaded_final_networks'] == 30, 'Expected independent reload of 30 final networks')
    require(validation['new_training_stages'] == 65, 'Independent training-stage audit differs')
    require(validation['protected_previous_verified_files'] == len(protected['files']),
            'Previous-version preservation verification is incomplete')
    require(validation['test_images_in_training_or_selection'] == 0, 'Test data entered training/selection')
    require(validation['source_engine_sha256'] == manifest['engine_sha256'] == sha(WORK / 'engine.py'),
            'Frozen engine identity differs from verified training')
    for relative, digest in manifest['parts'].items():
        require(sha(ROOT / relative) == digest, f'Source changed after assembly: {relative}')
    require(protocol['gate_min_px'] == .5 and protocol['gate_min_relative'] == .02 and
            protocol['new_branch_weight'] == .20 and protocol['context_ratio'] == 2.0 and
            protocol['local_short_side_fraction'] == .15, 'Frozen gate or architecture settings changed')
    require(protocol['test_reference_labels_used'] is False and planned['external_test_reference_used'] is False,
            'External reference exclusion was not asserted in the frozen protocol')
    require(notebook_validation['status'] == 'PASS' and notebook_validation['total_training_stages'] == 145,
            'Notebook must pass validation of its 145-stage standalone workflow')
    require(notebook_validation['notebook_sha256'] == sha(OUT / NOTEBOOK) and
            notebook_validation['engine_sha256'] == manifest['engine_sha256'], 'Notebook identity changed')
    require(backup['status'] == 'PASS', 'The historical backup has not been verified')
    baseline_sha, candidate_sha = sha(OUT / 'baseline_8p2.csv'), sha(OUT / 'submission.csv')
    require(baseline_sha == EXPECTED_BASELINE == backup['submission_sha256'], 'Original 8.2 CSV changed')
    require(candidate_sha == execution['submission_sha256'] == validation['submission_sha256'],
            'Submission changed after training or validation')
    require(not list((OUT / 'split_2027').glob('test_*.csv')),
            'Secondary split must not contain test ensembles used for output')

    summary = read(OUT / 'ablation_summary.csv')
    require(not summary[['split_seed', 'method']].duplicated().any(), 'Duplicate summary rows')
    expected_pairs = {(seed, method) for seed in SPLITS for method in METHODS + (PRIMARY_EXTRA if seed == 2026 else ())}
    require(set(zip(summary.split_seed, summary.method)) == expected_pairs, 'Incomplete or unexpected ablation summary')
    summary_index = summary.set_index(['split_seed', 'method'])
    arrays, truth_frames, per_image_parts, tail_parts = {}, {}, [], []
    fold_rows, risk_rows, decisions, training_crop_rows = [], [], {}, []

    for seed in SPLITS:
        folder = OUT / f'split_{seed}'
        truth = read(folder / 'folds.csv').sort_values('image_id', key=lambda s: s.map(int)).reset_index(drop=True)
        require(len(truth) == 80 and truth.image_id.is_unique and truth.group.notna().all(), 'Invalid train manifest')
        require((truth.groupby('group').fold.nunique() == 1).all(), 'Training groups cross validation folds')
        truth_frames[seed] = truth
        y = truth[['x', 'y']].to_numpy(float)
        roi = pd.concat([read(folder / f'fold_{fold}/oof_roi.csv') for fold in range(5)], ignore_index=True)
        roi = align(truth, roi)
        inside = (np.abs(y - roi[['crop_x', 'crop_y']].to_numpy(float)) <= roi.crop_side.to_numpy(float)[:, None] / 2).all(axis=1)
        require(np.allclose(roi.crop_side, .15 * truth[['width', 'height']].min(axis=1)), 'OOF ROI size changed')
        method_list = METHODS + (PRIMARY_EXTRA if seed == 2026 else ())
        arrays[seed] = {}
        for method in method_list:
            saved = align(truth, read(folder / f'oof_{method}.csv'))
            require(np.array_equal(saved[['x_true', 'y_true']].to_numpy(float), y), 'OOF truth mismatch')
            require(np.array_equal(saved.fold.to_numpy(int), truth.fold.to_numpy(int)), 'OOF fold mismatch')
            prediction = saved[['x', 'y']].to_numpy(float)
            arrays[seed][method] = prediction
            computed = metrics(prediction, y)
            for key, value in computed.items():
                require(np.isclose(value, summary_index.loc[(seed, method), key], rtol=1e-10, atol=1e-8),
                        f'Saved metric differs: {seed}/{method}/{key}')
        coarse_distance = np.linalg.norm(arrays[seed]['COARSE'] - y, axis=1)
        for method in method_list:
            prediction = arrays[seed][method]
            residual, distance = prediction - y, np.linalg.norm(prediction - y, axis=1)
            per_image = truth[['image_id', 'group', 'fold', 'width', 'height']].copy()
            per_image['split_seed'], per_image['method'] = seed, method
            per_image['dx_px'], per_image['dy_px'] = residual[:, 0], residual[:, 1]
            per_image['error_px'], per_image['squared_error_px2'] = distance, distance ** 2
            per_image['coarse_error_px'] = coarse_distance
            per_image['worse_than_coarse'] = distance > coarse_distance + 1e-9
            per_image['local_roi_contains_truth'] = inside
            total_sse = float(np.sum(distance ** 2))
            per_image['total_sse_share_percent'] = (distance ** 2 / total_sse * 100 if total_sse > 0 else 0.)
            require(np.isclose(per_image.worse_than_coarse.mean(), summary_index.loc[(seed, method), 'worse_than_coarse_fraction']),
                    'Saved fine-worsening fraction differs')
            require(np.isclose(inside.mean(), summary_index.loc[(seed, method), 'roi_coverage']), 'Saved ROI coverage differs')
            per_image_parts.append(per_image)
            tail = per_image.sort_values(['error_px', 'image_id'], ascending=[False, True]).head(10).copy()
            tail['error_rank'] = np.arange(1, len(tail) + 1)
            tail_parts.append(tail)
        decisions[str(seed)] = []
        for fold in range(5):
            sub = folder / f'fold_{fold}'
            decision = js(sub / 'selection.json')
            risk = js(sub / 'risk_rule.json')
            roles = read(sub / 'split_roles.csv')
            fitting = set(roles.loc[roles.role == 'model_fit', 'image_id'])
            stopping = set(roles.loc[roles.role == 'early_stop', 'image_id'])
            calibration = set(roles.loc[roles.role == 'calibration_only', 'image_id'])
            outer = set(roles.loc[roles.role == 'outer_validation', 'image_id'])
            require(set(risk['calibration_ids']) == calibration, 'Risk calibration used the wrong images')
            require(not calibration & (fitting | stopping | outer), 'Risk calibration overlaps other roles')
            require(risk['feature_constants']['policies_in_tie_break_order'] ==
                    ['fine_only', 'risk_half', 'risk_coarse', 'global_quarter'], 'Risk-policy search changed')
            single, dual = decision['arms']['single'], decision['arms']['dual']
            require(set(single['stopping_ids']) == set(dual['stopping_ids']) == stopping,
                    'Matched arms used different early-stop images')
            gain = single['best_rmse'] - dual['best_rmse']
            accept = gain >= .5 and gain >= .02 * single['best_rmse']
            require(decision['accepted'] == accept and np.isclose(gain, decision['gain_px']), 'Architecture gate changed')
            decisions[str(seed)].append(decision)
            fold_rows.append({'split_seed': seed, 'fold': fold, 'fold_display': fold + 1,
                              'single_best_epoch': single['best_epoch'], 'dual_best_epoch': dual['best_epoch'],
                              'single_inner_rmse': single['best_rmse'], 'dual_inner_rmse': dual['best_rmse'],
                              'inner_gain_px': gain, 'dual_accepted': accept,
                              'risk_enabled': risk['enabled'], 'risk_policy': risk['effective_policy'],
                              'risk_calibration_groups': risk['calibration_groups']})
            risk_rows.append({'split_seed': seed, 'fold': fold, 'enabled': risk['enabled'],
                              'selected_policy': risk['selected_policy'], 'effective_policy': risk['effective_policy'],
                              'reason': risk['reason'], 'calibration_images': risk['calibration_images'],
                              'calibration_groups': risk['calibration_groups'],
                              'baseline_fine_rmse': risk['baseline_fine_metrics']['rmse'],
                              'crossfit_rmse': risk['crossfit_selected_procedure_metrics']['rmse'],
                              'crossfit_gain_px': risk['crossfit_gain_px']})
            for arm in ('single', 'dual'):
                for stage in ('fine_select', 'fine_final'):
                    stage_dir = sub / arm / stage
                    history = read(stage_dir / 'history.csv')
                    training = js(stage_dir / 'training_summary.json')
                    n_images = int(training['training_images'])
                    presentations = n_images * len(history)
                    enlarged = int(history.crop_was_enlarged.sum())
                    initially_missed = int(history.initial_crop_miss.sum())
                    require(presentations > 0 and 0 <= enlarged <= presentations and
                            0 <= initially_missed <= presentations, 'Invalid training-crop counters')
                    require(np.isfinite(history[['crop_sampled_side_mean_px', 'crop_final_side_mean_px']].to_numpy()).all(),
                            'Nonfinite training-crop side means')
                    require(np.allclose(history.crop_enlargement_rate, history.crop_was_enlarged / n_images),
                            'Training-crop enlargement rate differs from actual counters')
                    training_crop_rows.append({'split_seed': seed, 'fold': fold, 'arm': arm, 'stage': stage,
                        'training_images': n_images, 'completed_epochs': len(history),
                        'image_presentations': presentations, 'enlarged_presentations': enlarged,
                        'initially_missed_presentations': initially_missed,
                        'enlargement_fraction': enlarged / presentations,
                        'mean_sampled_side_px': float(history.crop_sampled_side_mean_px.mean()),
                        'mean_final_side_px': float(history.crop_final_side_mean_px.mean()),
                        'scope': 'Training augmentation presentations, not unique images or validation ROIs'})
        require(decisions[str(seed)] == execution['decisions'][str(seed)], 'Recorded decisions differ from execution')

    pd.concat(per_image_parts, ignore_index=True).to_csv(OUT / 'oof_per_image_diagnostics.csv', index=False)
    pd.concat(tail_parts, ignore_index=True).to_csv(OUT / 'oof_top10_errors.csv', index=False)
    pd.DataFrame(fold_rows).to_csv(OUT / 'fold_decisions.csv', index=False)
    pd.DataFrame(risk_rows).to_csv(OUT / 'risk_calibration_summary.csv', index=False)
    require(len(training_crop_rows) == 40, 'Expected 40 fine-stage crop histories')
    pd.DataFrame(training_crop_rows).to_csv(OUT / 'training_crop_summary.csv', index=False)
    crop_presentations = sum(row['image_presentations'] for row in training_crop_rows)
    crop_enlarged = sum(row['enlarged_presentations'] for row in training_crop_rows)
    crop_summary = {'fine_training_stages': 40, 'image_presentations': crop_presentations,
        'enlarged_presentations': crop_enlarged, 'enlargement_fraction': crop_enlarged / crop_presentations,
        'mean_sampled_side_px': sum(row['mean_sampled_side_px'] * row['image_presentations'] for row in training_crop_rows) / crop_presentations,
        'mean_final_side_px': sum(row['mean_final_side_px'] * row['image_presentations'] for row in training_crop_rows) / crop_presentations,
        'scope': 'Counts include repeated epochs and both matched arms; only training augmentation can use training labels to enlarge a crop.'}
    write_json(OUT / 'training_crop_summary.json', crop_summary)

    # Paired group bootstrap resamples the fixed OOF predictions, not training.
    comparison_specs = [('SINGLE', 'DUAL_RAW', 'architecture'),
                        ('DUAL_RAW', 'DUAL_RISK', 'risk_rule'),
                        ('SINGLE', 'MATCHED_DEPLOY', 'matched_deployment')]
    bootstrap_rows, pair_rows = [], []
    for seed in SPLITS:
        truth = truth_frames[seed]
        groups = sorted(truth.group.unique())
        group_indices = {group: np.flatnonzero(truth.group.to_numpy() == group) for group in groups}
        rng = np.random.default_rng(20261027 + seed)
        draws = [np.concatenate([group_indices[group] for group in rng.choice(groups, len(groups), replace=True)])
                 for _ in range(2000)]
        y = truth[['x', 'y']].to_numpy(float)
        specs = comparison_specs + ([('BASELINE8P2', 'SUBMISSION_CANDIDATE', 'actual_submission')] if seed == 2026 else [])
        for baseline, candidate, purpose in specs:
            a, b = arrays[seed][baseline], arrays[seed][candidate]
            ea, eb = a - y, b - y
            da, db = np.linalg.norm(ea, axis=1), np.linalg.norm(eb, axis=1)
            ma, mb = metrics(a, y), metrics(b, y)
            rmse_deltas, p90_deltas = [], []
            for indices in draws:
                rmse_deltas.append(float(np.sqrt(np.mean(eb[indices] ** 2)) - np.sqrt(np.mean(ea[indices] ** 2))))
                p90_deltas.append(float(np.quantile(db[indices], .9) - np.quantile(da[indices], .9)))
            entry = {'split_seed': seed, 'baseline': baseline, 'candidate': candidate, 'purpose': purpose,
                     'baseline_rmse': ma['rmse'], 'candidate_rmse': mb['rmse'], 'delta_rmse': mb['rmse'] - ma['rmse'],
                     'delta_p90_distance_px': mb['p90_distance_px'] - ma['p90_distance_px'],
                     'delta_p95_distance_px': mb['p95_distance_px'] - ma['p95_distance_px'],
                     'delta_max_distance_px': mb['max_distance_px'] - ma['max_distance_px'],
                     'images_improved': int((db < da-1e-9).sum()), 'images_worsened': int((db > da+1e-9).sum()),
                     'images_unchanged': int((np.abs(db-da) <= 1e-9).sum()),
                     'rmse_delta_group_bootstrap_95pct': np.quantile(rmse_deltas, [.025, .975]).tolist(),
                     'p90_delta_group_bootstrap_95pct': np.quantile(p90_deltas, [.025, .975]).tolist(),
                     'bootstrap_repeats': 2000, 'groups': len(groups),
                     'scope': 'Paired resampling of fixed development OOF predictions; excludes retraining randomness, patient dependence and adaptive selection bias.'}
            bootstrap_rows.append(entry)
            changed = truth[['image_id', 'group', 'fold']].copy()
            changed['split_seed'], changed['baseline'], changed['candidate'] = seed, baseline, candidate
            changed['baseline_error_px'], changed['candidate_error_px'], changed['error_change_px'] = da, db, db-da
            changed['baseline_sse'], changed['candidate_sse'] = da ** 2, db ** 2
            pair_rows.append(changed)
    write_json(OUT / 'paired_group_bootstrap.json', {'comparisons': bootstrap_rows,
               'interpretation': 'These are repeated-development OOF comparisons, not independent test confidence intervals. No comparison changes the predeclared submission.'})
    pd.concat(pair_rows, ignore_index=True).to_csv(OUT / 'paired_oof_changes.csv', index=False)

    primary = OUT / 'split_2026'
    base_test = read(primary / 'test_ensemble_BASELINE8P2.csv')
    candidate_test = align(base_test, read(primary / 'test_ensemble_SUBMISSION_CANDIDATE.csv'))
    require(len(base_test) == len(candidate_test) == 20, 'Expected 20 test images')
    displacement = np.linalg.norm(candidate_test[['x','y']].to_numpy() - base_test[['x','y']].to_numpy(), axis=1)
    test_changes = base_test.rename(columns={'x':'baseline_x', 'y':'baseline_y'}).copy()
    test_changes['candidate_x'], test_changes['candidate_y'], test_changes['displacement_px'] = candidate_test.x, candidate_test.y, displacement
    test_changes.to_csv(OUT / 'test_prediction_changes.csv', index=False)
    base_csv, new_csv = read(OUT / 'baseline_8p2.csv'), read(OUT / 'submission.csv')
    require(list(base_csv.columns) == list(new_csv.columns) == ['ImageID', 'value'], 'Submission columns differ')
    require(len(base_csv) == len(new_csv) == 40 and base_csv.ImageID.is_unique and new_csv.ImageID.is_unique,
            'Expected 40 unique submitted coordinates')
    require(set(base_csv.ImageID) == set(new_csv.ImageID), 'Submission identifiers differ')
    new_csv = new_csv.set_index('ImageID').loc[base_csv.ImageID].reset_index()
    csv_changes = base_csv.rename(columns={'value':'baseline_value'})
    csv_changes['candidate_value'] = new_csv.value
    csv_changes['delta_px'] = csv_changes.candidate_value - csv_changes.baseline_value
    require(np.isfinite(csv_changes[['baseline_value', 'candidate_value', 'delta_px']].to_numpy()).all(), 'Nonfinite submission values')
    csv_changes.to_csv(OUT / 'submission_value_changes.csv', index=False)
    accepted = {str(seed): [row['fold'] for row in decisions[str(seed)] if row['accepted']] for seed in SPLITS}
    if not accepted['2026']:
        require(candidate_sha == baseline_sha, 'All primary gates rejected but baseline changed')
    actual = next(row for row in bootstrap_rows if row['purpose'] == 'actual_submission')
    comparison = {'primary_actual_submission': actual,
                  'accepted_fold_ids_zero_based': accepted,
                  'risk_enabled_folds': {str(seed): [row['fold'] for row in risk_rows if row['split_seed'] == seed and row['enabled']] for seed in SPLITS},
                  'submission_sha256': candidate_sha, 'baseline_sha256': baseline_sha,
                  'submission_bytes_changed': candidate_sha != baseline_sha,
                  'submitted_coordinate_rows_changed': int((csv_changes.delta_px != 0).sum()),
                  'max_submitted_coordinate_change_px': float(csv_changes.delta_px.abs().max()),
                  'mean_test_displacement_px': float(displacement.mean()), 'max_test_displacement_px': float(displacement.max()),
                  'secondary_test_predictions_used': False, 'new_online_score_known': False,
                  'new_training_stages': 65, 'independently_reloaded_final_networks': 30,
                  'protected_previous_files_verified': len(protected['files']),
                  'training_crop_augmentation': crop_summary,
                  'baseline_online_score_user_reported': 8.2,
                  'limitations': 'Training OOF was repeatedly used for development. Neither split is an untouched independent test. No external test reference labels or private scores were read.'}
    write_json(OUT / 'comparison.json', comparison)

    optional_names = []
    for name in ('gpu_probe.json', 'model_cpu_validation.json', 'risk_cpu_validation.json'):
        path = WORK / name
        require(path.is_file(), f'Missing actual functional evidence: {name}')
        require(js(path)['status'] == 'PASS', f'Functional evidence failed: {name}')
        shutil.copy2(path, OUT / name)
        optional_names.append(name)
    for name in ('notebook_terminal_audit.json', 'independent_cpu_audit.json'):
        path = OUT / name
        if path.exists():
            evidence = js(path)
            require(evidence['status'] == 'PASS', f'Optional evidence failed: {name}')
            if 'notebook_sha256' in evidence:
                require(evidence['notebook_sha256'] == notebook_validation['notebook_sha256'], 'Notebook audit is stale')
            optional_names.append(name)
    for name in ('protocol_review.md', 'verifier_preflight.md', 'primary_record_precheck_0_3.json', 'secondary_record_precheck_0_2.json'):
        path = WORK / name
        if path.exists():
            shutil.copy2(path, OUT / name)
            optional_names.append(name)
    if (OUT / 'verification_attempt1.json').exists():
        # Preserve the failed verifier attempt as history, never as PASS evidence.
        optional_names.append('verification_attempt1.json')
    shutil.copy2(WORK / 'source_manifest.json', OUT / 'source_manifest.json')

    labels = {'COARSE':'粗定位', 'SINGLE':'匹配的单窗口精修', 'DUAL_RAW':'双尺度精修，未回退',
              'DUAL_RISK':'双尺度精修＋校准回退', 'MATCHED_DEPLOY':'按内部门槛融合的匹配对照',
              'BASELINE8P2':'原 8.2 基线', 'SUBMISSION_CANDIDATE':'本轮实际提交方案'}
    tables = {}
    for seed in SPLITS:
        rows = summary.loc[summary.split_seed == seed]
        table_rows = '\n'.join(f"| {labels[row.method]} | {row.rmse:.4f} | {row.p90_distance_px:.4f} | {row.p95_distance_px:.4f} | {row.max_distance_px:.4f} | {100*row.worse_than_coarse_fraction:.1f}% |"
                               for row in rows.itertuples(index=False))
        tables[seed] = '| 方法 | 坐标 RMSE | 欧氏 P90 | 欧氏 P95 | 最大欧氏误差 | 比粗定位更差 |\n|---|---:|---:|---:|---:|---:|\n' + table_rows
    decision_table = '\n'.join(f"| {row['split_seed']} | {row['fold_display']} | {row['single_best_epoch']}/{row['dual_best_epoch']} | {row['inner_gain_px']:+.4f} | {'通过' if row['dual_accepted'] else '未通过'} | {row['risk_policy']} |"
                               for row in fold_rows)
    stability_table = '\n'.join(f"| {row['split_seed']} | {labels[row['candidate']]} 对 {labels[row['baseline']]} | {row['delta_rmse']:+.4f} | {row['delta_p90_distance_px']:+.4f} | {row['delta_max_distance_px']:+.4f} |"
                                for row in bootstrap_rows)
    changed = comparison['submission_bytes_changed']
    outcome = (f"主划分有 {len(accepted['2026'])}/5 折通过预先固定的内部对照门槛，按固定 20% 权重接入双尺度分支；提交中 {comparison['submitted_coordinate_rows_changed']}/40 个坐标值改变。"
               if accepted['2026'] else '主划分五折均未通过预设门槛，最终 submission.csv 与原 8.2 文件逐字节相同。')
    local_conclusion = (f"实际提交对应的主划分 OOF RMSE 比原 8.2 基线降低 {-actual['delta_rmse']:.6f}px。"
                        if actual['delta_rmse'] < -1e-9 else
                        f"实际提交对应的主划分 OOF RMSE 比原 8.2 基线增加 {actual['delta_rmse']:.6f}px，本地证据不支持将其称为更好的提交。"
                        if actual['delta_rmse'] > 1e-9 else
                        '实际提交对应的主划分 OOF RMSE 与原 8.2 基线相同。')
    primary_arch = next(row for row in bootstrap_rows if row['split_seed'] == 2026 and row['purpose'] == 'architecture')
    secondary_arch = next(row for row in bootstrap_rows if row['split_seed'] == 2027 and row['purpose'] == 'architecture')
    stable = primary_arch['delta_rmse'] < 0 and secondary_arch['delta_rmse'] < 0
    stability_sentence = ('双尺度 raw 分支相对匹配单窗口对照，在两套划分上的整体 RMSE 方向均为降低；仍需结合尾部与逐折结果评估收益大小。'
                          if stable else '双尺度 raw 分支没有在两套划分上都降低整体 RMSE，不能据此宣称改进具有跨划分稳定性。')
    roi_coverage = {seed: float(summary_index.loc[(seed, 'COARSE'), 'roi_coverage']) for seed in SPLITS}
    report = f'''# 双尺度精修、校准回退与第二套五折审计

{outcome}

{local_conclusion} 没有新的 Kaggle 线上分数，也没有读取外部测试参考坐标或私榜得分。8.2 是用户反馈的历史成绩，不是本文计算值。

## 本轮具体改变

保留原 ImageNet ResNet18 和 KL＋DSNT＋坐标损失。新的 single 对照与 dual 候选都使用 256×256 输入、64×64 热图、fine sigma=1.5、相同训练图/随机种子/选轮流程。默认局部窗口为原图短边的 15%，无附加 margin；dual 同时读取边长两倍的上下文，通过共享编码器和按物理像素中心对齐的多层特征融合完成精修。

两路图像共享几何变换；local 保留原光度增强，context 不另加光度扰动。single 和 dual 使用同一个六通道数据加载过程，single 仅消费局部三通道，因此不会因选择模型类型改变样本的随机增强顺序。训练期沿用标签驱动的裁剪扩张增强；验证、校准与测试窗口不根据标签修复。

40 个 fine 训练阶段共记录 {crop_presentations} 次图像呈现（含重复轮次与两组模型），其中 {crop_enlarged} 次扩张局部窗口，占 {crop_summary['enlargement_fraction']*100:.2f}%。按呈现次数加权的采样窗口平均边长 {crop_summary['mean_sampled_side_px']:.3f}px，扩张后为 {crop_summary['mean_final_side_px']:.3f}px。因此“15%窗口”描述默认采样与部署，不能解释为所有训练输入都保持 15%。明细见 `training_crop_summary.csv`。

早停尺度改为真实部署的局部窗口，不再用 20%/30%/40% 多尺度平均分数选一个最终在 15% 窗口推理的模型。两个新候选共享此变更，所以架构收益要看 DUAL_RAW 对 SINGLE；不能把它与历史 8.2 的全部差异都归因于上下文。

## 回退规则

正常窗口预测是默认输出；同模型在 1.25 倍窗口上再次推理，只提取尺度稳定性，不直接替换默认坐标。TTA 的两个方向按原概率混合方式解码。多峰代理是抑制主峰半径 3 个热图格点后，次峰与主峰概率之比。

预设风险条件为尺度位移/窗口边长 >0.05，或次峰比 >0.80 且尺度位移比 >0.025，或翻转 TTA 位移比 >0.025。候选只有 fine 原样、风险时粗细各半、风险时回退 coarse、全局 25% coarse＋75% fine 四种。

独立 calibration 组按 group 留一：每次只用其他组选择候选，拼回留出组预测后，要求相对 fine 的 RMSE 至少降低 0.5px 且 2%，P90 不恶化，并满足最少 12 个组才启用。最后在完整 calibration 组选择一个离散策略。它是低自由度经验规则，不能宣称有严格覆盖率或小样本无偏保证；融合协方差也只是描述性混合矩。

## 两套划分与提交规则

原始 80 张有标签训练图按精确重复组隔离。主划分 seed=2026，粗模型和图像级交叉预测中心来自经过身份校验的旧运行；第二套划分 seed=2027 自行重训粗模型和中心。两套都完整五折，新训练共 65 个阶段：主划分 20 个 fine、第二划分 25 个 coarse 加 20 个 fine。两个最终 fine 模型都不使用 calibration 或 outer 标签训练。

每折只根据内部 stopping 数据比较匹配的 single/dual，dual 至少改善 0.5px 且 2% 才接入。匹配对照 MATCHED_DEPLOY 是 80% SINGLE＋20% DUAL_RISK；实际主划分提交是 80% 原 8.2 折预测＋20% DUAL_RISK。未过门槛时分别保留 SINGLE 或原 8.2。最后对主划分五折等权平均。

第二套划分用于稳定性审计，不做测试推理，不用于选择门槛、权重或最终提交。没有为了报告更好看的 OOF 临时切换提交方案。

## 主划分 seed=2026 的真实 OOF

{tables[2026]}

默认局部窗口覆盖率为 {roi_coverage[2026]*100:.2f}%。宽上下文有助于提供额外信息，但最终热图仍在局部窗口坐标内；窗口遗漏并不会因读取上下文而自动获得修复。

## 第二套划分 seed=2027 的稳定性审计

{tables[2027]}

默认局部窗口覆盖率为 {roi_coverage[2027]*100:.2f}%。{stability_sentence}

| 划分 | 对照 | RMSE 差值 | P90 差值 | 最大误差差值 |
|---|---|---:|---:|---:|
{stability_table}

差值统一为候选减去基线，负数表示降低。两套划分仍使用同一批训练图，第二套不是新独立样本集。

## 逐折接入与校准结果

| 划分 | 折 | single/dual 轮数 | 内部 RMSE 降低 | 架构门槛 | 实际回退策略 |
|---|---:|---:|---:|---|---|
{decision_table}

`risk_calibration_summary.csv` 和各折 `risk_rule.json` 提供完整校准候选、留一组记录和启用原因。回退是否有效应比较 DUAL_RISK 与 DUAL_RAW；校准交叉预测中改善，仍可能在外层样本上变差，不能仅从“启用过规则”推断泛化改善。

## 实际提交变化与尾部诊断

实际提交主划分：RMSE {actual['baseline_rmse']:.6f} → {actual['candidate_rmse']:.6f}；逐图欧氏误差 {actual['images_improved']} 张改善、{actual['images_worsened']} 张恶化、{actual['images_unchanged']} 张不变。

基线 SHA256：`{baseline_sha}`。新文件 SHA256：`{candidate_sha}`。文件字节是否变化：{str(changed).lower()}；实际导出坐标变化 {comparison['submitted_coordinate_rows_changed']}/40 行；坐标最大绝对变化 {comparison['max_submitted_coordinate_change_px']:.6f}px。测试预测相对基线平均移动 {comparison['mean_test_displacement_px']:.6f}px、最大移动 {comparison['max_test_displacement_px']:.6f}px。预测移动不是测试误差。

逐图诊断保存于 `oof_per_image_diagnostics.csv`，各方法最大的 10 个 OOF 误差及平方误差贡献保存于 `oof_top10_errors.csv`，严格配对变化保存于 `paired_oof_changes.csv`。没有删除或更改大误差样本标签。

实际提交对基线的固定预测 group bootstrap RMSE 差值 95% 分位区间为 {actual['rmse_delta_group_bootstrap_95pct']}（2000 次）。完整架构/回退/融合比较见 `paired_group_bootstrap.json`。这些区间仅重采样已有开发预测，不包括重训随机性、未知患者相关性或反复开发选择偏差，不能解释为独立测试显著性或私榜提升证明。

## 真实运行、旧版本保留与复现

GPU：{execution['gpu']}；两套划分的新训练与推理耗时 {execution['elapsed_seconds']/60:.2f} 分钟；记录峰值 CUDA 分配内存 {execution['peak_cuda_mb']:.2f}MB。独立验证 PASS，重载 30 个最终网络；校验了训练前记录的 {len(protected['files'])} 个旧版本文件，保留原代码、模型、Notebook 和 CSV。既有历史备份继续位于 `{backup['backup']}`，本轮没有声称重新创建该备份。

核验过程记录见 `verifier_preflight.md`。首次额外解码器对照检查因使用不同批次大小而失败，已保留 `verification_attempt1.json`；统一为原 calibration 的完整行序与批次后重新执行全套核验。模型、提交坐标与原误差容差均未因此改动。

原生 Notebook：`{NOTEBOOK}`。添加比赛数据和已有 `3pretrained` 中的官方 ResNet18 ImageNet 权重即可，Internet 可关闭。从头运行先重训两套历史基线 80 阶段，再完成本轮 65 阶段，合计 145；本机本轮复用历史粗模型，只新增 65 阶段。Notebook 已通过静态验证；这不代表另外完成过一遍 Kaggle 云端训练。不同运行库/GPU可能产生数值差异。双尺度特征对齐所用 CUDA grid_sample 反向传播在当前库中不保证逐位确定，训练按原 warn_only 设置继续；独立核验检查的是保存权重的前向预测可复现性，不能据此保证重新训练后每一位权重一致。

`dual_context_documents.zip` 包含 Notebook、原始 8.2 CSV、新提交、前一轮 precision CSV、源码、指标、校准审计和完整对比报告，不含训练权重或 ImageNet 初始化权重。所有模型仍保留在新实验目录，旧版不被覆盖。

实验实施与审计不读取外部测试参考坐标或私榜评分文件。后续若继续使用本次外层结果调方案，应仍称为开发验证，而不是新的独立泛化证据。
'''
    (OUT / 'report.md').write_text(report, encoding='utf-8')
    readme = f'''# 双尺度精修实验交付

{outcome}

{local_conclusion} 新的线上成绩未知。请先读 `report.md`，结构化比较见 `comparison.json`。

- `submission.csv`：按冻结协议产生的主划分新候选。
- `baseline_8p2.csv`：原 8.2 提交的原样副本。
- `previous_precision.csv`：上一轮 precision 提交的原样副本，不参与本轮选型。
- `{NOTEBOOK}`：完整 Kaggle Notebook，添加比赛数据与现有 `3pretrained`，GPU 运行；从头 145 个训练阶段。
- `ablation_summary.csv`、`paired_group_bootstrap.json`、`oof_top10_errors.csv`：两套划分的真实对照与尾部审计。
- `validation_report.json`：65 阶段、30 最终网络独立重载、数据角色和历史文件保留核验。

文档 ZIP 不含模型权重。新增模型在 `split_2026/` 与 `split_2027/` 下，旧版文件均保留。第二划分只作稳定性审计，不参与测试预测或提交融合。没有使用外部测试参考标签或私榜成绩进行调参。
'''
    (OUT / 'README.md').write_text(readme, encoding='utf-8')

    top_names = ['README.md', 'report.md', 'submission.csv', 'baseline_8p2.csv', 'previous_precision.csv', NOTEBOOK,
                 'comparison.json', 'ablation_summary.csv', 'paired_group_bootstrap.json', 'paired_oof_changes.csv',
                 'oof_per_image_diagnostics.csv', 'oof_top10_errors.csv', 'fold_decisions.csv', 'risk_calibration_summary.csv',
                 'training_crop_summary.csv', 'training_crop_summary.json',
                 'test_prediction_changes.csv', 'submission_value_changes.csv', 'execution_report.json', 'validation_report.json',
                 'notebook_validation.json', 'run_protocol.json', 'planned_experiment.json', 'prepared_config.json',
                 'protected_previous.json', 'existing_baseline_backup.json', 'source_manifest.json', 'test_manifest.csv'] + optional_names
    archive_files = {name: OUT / name for name in top_names}
    for name in ('engine.py', 'model.py', 'risk.py', 'pipeline.py', 'run.py', 'assemble.py', 'build_notebook.py',
                 'verify_results.py', 'finalize.py', 'test_model_cpu.py', 'test_risk_cpu.py', 'test_notebook_cpu.py'):
        path = WORK / name
        if path.is_file():
            archive_files[f'reproduction_source/{name}'] = path
    # Preserve all small experiment evidence while deliberately excluding .pt.
    for seed in SPLITS:
        for path in (OUT / f'split_{seed}').rglob('*'):
            if path.is_file() and path.suffix.lower() in ('.json', '.csv'):
                archive_files[path.relative_to(OUT).as_posix()] = path
    require(all(path.is_file() for path in archive_files.values()), 'Required report artifact is missing')
    archive_manifest = {name: {'sha256': sha(path), 'bytes': path.stat().st_size}
                        for name, path in sorted(archive_files.items())}
    archive_path = OUT / 'dual_context_documents.zip'
    with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name, path in sorted(archive_files.items()):
            archive.write(path, name)
        archive.writestr('manifest.json', json.dumps(archive_manifest, ensure_ascii=False, indent=2))
    with zipfile.ZipFile(archive_path) as archive:
        require(archive.testzip() is None, 'ZIP integrity test failed')
        require(set(archive.namelist()) == set(archive_manifest) | {'manifest.json'}, 'ZIP manifest entries differ')
        for name, expected in archive_manifest.items():
            data = archive.read(name)
            require(len(data) == expected['bytes'] and hashlib.sha256(data).hexdigest() == expected['sha256'],
                    f'ZIP content differs from source: {name}')
    write_json(OUT / 'package_verification.json', {'status': 'PASS', 'archive': str(archive_path),
               'sha256': sha(archive_path), 'files': len(archive_manifest), 'all_archived_bytes_verified': True,
               'training_weights_included': False, 'pretrained_weights_included': False,
               'previous_files_modified_by_finalizer': False, 'new_training_launched_by_finalizer': False})
    print(json.dumps(comparison, ensure_ascii=False, indent=2), flush=True)
    print('PACKAGED:', archive_path, flush=True)


if __name__ == '__main__':
    main()
