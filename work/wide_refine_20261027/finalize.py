"""Package completed wide-refinement evidence without training or score leakage.

This module never imports either training engine, opens external evaluation
references, changes predictions, or selects models. main() only writes the new
output directory after independent validation reports PASS.
"""
from pathlib import Path
import hashlib
import json
import shutil
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[2]
WORK = Path(__file__).resolve().parent
OUT = ROOT / 'outputs/wide_refine_20261027'
PREVIOUS = ROOT / 'outputs/dual_context_20261027'
SPLITS = (2026, 2027)
METHODS = ('COARSE', 'SINGLE', 'WIDE_DEV', 'WIDE_FULL', 'MATCHED_DEPLOY')
PRIMARY_EXTRA = ('PREVIOUS_SUBMISSION', 'SUBMISSION_CANDIDATE')
BASELINE_SHA = 'be3cba8f60f96325c7730480db2f740522e79ac550368984174a73396342176f'


def require(condition, message):
    if not bool(condition):
        raise RuntimeError(message)


def js(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def main():
    sys.path.insert(0, str(ROOT / 'work/gpu_test_deps'))
    import numpy as np
    import pandas as pd

    def read(path):
        return pd.read_csv(path, dtype={'image_id': str, 'group': str, 'ImageID': str})

    def align(frame, ids):
        require(frame.image_id.is_unique and len(ids) == len(set(ids)), 'Duplicate comparison IDs')
        require(set(frame.image_id) == set(ids), 'Mismatched comparison IDs')
        return frame.set_index('image_id').loc[list(ids)].reset_index()

    def metrics(prediction, truth, coarse):
        residual = prediction - truth
        require(residual.shape == (80, 2) and np.isfinite(residual).all(), 'Expected 80 finite OOF pairs')
        distance = np.linalg.norm(residual, axis=1)
        coarse_distance = np.linalg.norm(coarse - truth, axis=1)
        return {'rmse': float(np.sqrt(np.mean(residual ** 2))),
                'mean_distance_px': float(distance.mean()), 'median_distance_px': float(np.median(distance)),
                'p90_distance_px': float(np.quantile(distance, .9)),
                'p95_distance_px': float(np.quantile(distance, .95)),
                'max_distance_px': float(distance.max()),
                'worse_than_coarse_fraction': float(np.mean(distance > coarse_distance + 1e-9))}

    execution, validation = js(OUT/'execution_report.json'), js(OUT/'validation_report.json')
    protocol, planned = js(OUT/'run_protocol.json'), js(OUT/'planned_experiment.json')
    manifest, protected = js(WORK/'source_manifest.json'), js(OUT/'protected_previous.json')
    geometry, probe = js(WORK/'geometry_cpu_validation.json'), js(WORK/'gpu_probe.json')
    require(execution['status'] == 'COMPLETE' and execution['completed_folds'] == 10, 'Two complete fivefold runs required')
    require(validation['status'] == geometry['status'] == probe['status'] == 'PASS', 'Required validation is incomplete')
    require(execution['new_training_stages'] == validation['new_training_stages'] == protocol['new_training_stages'] == planned['new_training_stages'] == 30,
            'Expected 30 newly trained stages')
    require(validation['reloaded_final_networks'] == 30, 'Expected 20 new final and 10 old SINGLE checkpoint reloads')
    require(validation['protected_previous_verified_files'] == len(protected['files']), 'Old artifact preservation audit incomplete')
    require(validation['test_images_in_training_or_selection'] == 0 and protocol['test_reference_labels_used'] is False,
            'Training/selection isolation was not verified')
    require(execution['split_seeds'] == protocol['split_seeds'] == planned['split_seeds'] == list(SPLITS), 'Split protocol changed')
    require(execution['secondary_submission_influence'] is False, 'Secondary split affected the submission')
    require(execution['full_refit_has_no_independent_calibration'] is True, 'Full-refit calibration status is incorrect')
    require(validation['source_engine_sha256'] == manifest['engine_sha256'] == sha(WORK/'engine.py'), 'Frozen engine changed')
    require(geometry['engine_sha256'] == manifest['engine_sha256'], 'Geometry checks refer to another engine')
    for relative, digest in manifest['parts'].items():
        require(sha(ROOT/relative) == digest, f'Frozen source changed: {relative}')
    require(sha(ROOT/'work/dual_context_20261027/engine.py') == manifest['previous_engine_sha256'], 'Previous engine changed')
    require(protocol['gate_min_px'] == .5 and protocol['gate_min_relative'] == .02 and
            protocol['gate_p90_must_not_increase'] is True and protocol['new_branch_weight'] == .25 and
            protocol['roi_short_side_fraction'] == .25, 'Frozen gate/ROI settings changed')
    config = protocol['config']
    require(config['FINE_INPUT_SIZE'] == 384 and config['FINE_HEATMAP_SIZE'] == 96 and config['HEATMAP_SIGMA'] == 1.5,
            'Unexpected wide-refinement configuration')
    previous_sha, candidate_sha = sha(OUT/'previous_submission.csv'), sha(OUT/'submission.csv')
    require(previous_sha == validation['previous_submission_sha256'] == protocol['previous_submission_sha256'] ==
            planned['previous_submission_sha256'] == sha(PREVIOUS/'submission.csv'), 'Previous submission identity changed')
    require(candidate_sha == validation['submission_sha256'] == execution['submission_sha256'], 'Candidate changed after verification')
    require(sha(OUT/'baseline_8p2.csv') == BASELINE_SHA, 'Original 8.2 copy changed')

    summary = read(OUT/'ablation_summary.csv')
    require(not summary.duplicated(['split_seed','method']).any(), 'Duplicate summary methods')
    require(set(zip(summary.split_seed,summary.method)) == {(seed,name) for seed in SPLITS for name in METHODS + (PRIMARY_EXTRA if seed == 2026 else ())},
            'Unexpected ablation methods')
    summaries, paired, tail_rows, decisions, accepted = [], [], [], [], {}
    prediction_maps = {}
    for split in SPLITS:
        folder = OUT/f'split_{split}'
        truth_frame = read(folder/'folds.csv')
        require(len(truth_frame) == 80 and truth_frame.image_id.is_unique, 'Expected 80 training images')
        require(set(truth_frame.image_id) == {str(i) for i in range(1,81)}, 'Unexpected training IDs')
        truth = truth_frame[['x','y']].to_numpy(float)
        coarse_frame = align(read(folder/'oof_COARSE.csv'), truth_frame.image_id)
        coarse = coarse_frame[['x','y']].to_numpy(float)
        predictions = {}
        for name in METHODS + (PRIMARY_EXTRA if split == 2026 else ()):
            frame = align(read(folder/f'oof_{name}.csv'), truth_frame.image_id)
            require(np.array_equal(frame[['x_true','y_true']].to_numpy(float), truth), f'{name}: OOF truth differs')
            prediction = frame[['x','y']].to_numpy(float)
            measured = metrics(prediction, truth, coarse)
            stored = summary.loc[(summary.split_seed == split)&(summary.method == name)].iloc[0]
            for key, value in measured.items():
                require(np.isclose(value, float(stored[key]), atol=1e-8, rtol=1e-10), f'Metric mismatch {split}/{name}/{key}')
            predictions[name] = prediction
            summaries.append({'split_seed': split, 'method': name, **measured,
                              'proposed_roi_coverage': float(stored.roi_coverage),
                              'old_single_roi_coverage': float(stored.old_single_roi_coverage)})
        prediction_maps[split] = (truth_frame, truth, predictions)
        groups = truth_frame.group.astype(str).to_numpy()
        unique_groups = np.unique(groups)
        group_members = [np.flatnonzero(groups == group) for group in unique_groups]
        comparisons = [('SINGLE','WIDE_DEV'),('WIDE_DEV','WIDE_FULL'),('SINGLE','WIDE_FULL'),('SINGLE','MATCHED_DEPLOY')]
        if split == 2026:
            comparisons.append(('PREVIOUS_SUBMISSION','SUBMISSION_CANDIDATE'))
        for baseline, candidate in comparisons:
            old_sq = ((predictions[baseline]-truth)**2).sum(axis=1)
            new_sq = ((predictions[candidate]-truth)**2).sum(axis=1)
            old_distance, new_distance = np.sqrt(old_sq), np.sqrt(new_sq)
            actual_delta = float(np.sqrt(new_sq.mean()/2)-np.sqrt(old_sq.mean()/2))
            rng = np.random.default_rng(8109 + split)
            bootstrap = []
            for _ in range(2000):
                indexes = np.concatenate([group_members[i] for i in rng.integers(0,len(group_members),size=len(group_members))])
                bootstrap.append(float(np.sqrt(new_sq[indexes].mean()/2)-np.sqrt(old_sq[indexes].mean()/2)))
            paired.append({'split_seed':split, 'baseline':baseline, 'candidate':candidate,
                           'rmse_delta_px':actual_delta, 'improved_images':int((new_distance<old_distance-1e-9).sum()),
                           'worsened_images':int((new_distance>old_distance+1e-9).sum()),
                           'unchanged_images':int((np.abs(new_distance-old_distance)<=1e-9).sum()),
                           'fixed_prediction_group_bootstrap_95pct_delta_interval':np.quantile(bootstrap,[.025,.975]).tolist(),
                           'scope':'Fixed-prediction group resampling only; repeated-development OOF, not retraining or model-selection uncertainty.'})
        for index, row in truth_frame.iterrows():
            item = {'split_seed':split,'fold':int(row.fold),'image_id':row.image_id,'group':row.group}
            for name, pred in predictions.items():
                difference = pred[index]-truth[index]
                item[f'{name}_euclidean_error_px'] = float(np.linalg.norm(difference))
                item[f'{name}_squared_error_px2'] = float((difference**2).sum())
            item['wide_full_minus_single_error_px'] = item['WIDE_FULL_euclidean_error_px']-item['SINGLE_euclidean_error_px']
            item['wide_full_minus_dev_error_px'] = item['WIDE_FULL_euclidean_error_px']-item['WIDE_DEV_euclidean_error_px']
            item['tail_sort_error_px'] = max(item['SINGLE_euclidean_error_px'],item['WIDE_FULL_euclidean_error_px'])
            tail_rows.append(item)
        accepted[str(split)] = []
        actual_decisions = []
        for fold in range(5):
            ff = folder/f'fold_{fold}'
            decision = js(ff/'selection.json')
            actual_decisions.append(decision)
            single, wide = decision['single_metrics'], decision['wide_dev_metrics']
            gain = single['rmse']-wide['rmse']
            gate = bool(gain >= .5 and gain >= .02*single['rmse'] and wide['p90_distance_px'] <= single['p90_distance_px']+1e-9)
            require(decision['accepted'] == gate and abs(decision['gain_px']-gain)<1e-8, 'Fold gate changed')
            require(decision['weight_if_accepted'] == .25 and decision['p90_must_not_increase'] is True, 'Gate protocol differs')
            stages = {stage: js(ff/stage/'training_summary.json') for stage in ('wide_select','wide_dev','wide_full')}
            require(stages['wide_dev']['fixed_epochs'] and stages['wide_full']['fixed_epochs'], 'Final refits used a validation score')
            require(stages['wide_dev']['best_epoch'] == stages['wide_full']['best_epoch'] == decision['selected_epoch'], 'DEV/FULL epochs differ')
            require(stages['wide_full']['training_images'] == 64, 'Expected all 64 outer-training images for FULL')
            require(stages['wide_dev']['training_images'] in (47,48), 'Unexpected DEV size')
            if gate:
                accepted[str(split)].append(fold)
            decisions.append({'split_seed':split,'fold_zero_based':fold,'fold':fold+1,'accepted':gate,
                              'selected_epoch':decision['selected_epoch'], 'single_selection_rmse':single['rmse'],
                              'wide_dev_selection_rmse':wide['rmse'],'rmse_gain_px':gain,
                              'single_selection_p90':single['p90_distance_px'], 'wide_dev_selection_p90':wide['p90_distance_px'],
                              'development_images':stages['wide_dev']['training_images'],'full_images':64,
                              'dev_successful_updates':stages['wide_dev']['successful_optimizer_steps'],
                              'full_successful_updates':stages['wide_full']['successful_optimizer_steps'],
                              'submission_weight':.25 if gate and split==2026 else 0.})
        require(actual_decisions == execution['decisions'][str(split)], 'Execution decisions do not match fold records')
    require(accepted == validation['accepted_folds_by_split'], 'Independent validator gate decisions differ')
    pd.DataFrame(decisions).to_csv(OUT/'fold_decisions.csv',index=False)
    pd.DataFrame(tail_rows).sort_values(['split_seed','tail_sort_error_px'],ascending=[True,False]).to_csv(OUT/'oof_tail_errors.csv',index=False)
    pd.DataFrame(summaries).to_csv(OUT/'recomputed_metrics.csv',index=False)

    # Compare submitted predictions only; no test reference values are loaded.
    old_csv, new_csv = read(OUT/'previous_submission.csv'), read(OUT/'submission.csv')
    require(list(old_csv.columns) == list(new_csv.columns) == ['ImageID','value'], 'Invalid submission schema')
    require(len(old_csv) == len(new_csv) == 40 and old_csv.ImageID.is_unique and new_csv.ImageID.is_unique, 'Invalid submission row count')
    require(set(old_csv.ImageID) == set(new_csv.ImageID), 'Submission ID sets differ')
    new_csv = new_csv.set_index('ImageID').loc[old_csv.ImageID].reset_index()
    changes = old_csv.rename(columns={'value':'previous_value'}).copy()
    changes['candidate_value'] = new_csv.value.to_numpy(float)
    changes['delta_px'] = changes.candidate_value-changes.previous_value
    require(np.isfinite(changes[['previous_value','candidate_value','delta_px']].to_numpy(float)).all(), 'Invalid coordinate changes')
    changes.to_csv(OUT/'submission_changes.csv',index=False)
    delta_by_id = {}
    for _, row in changes.iterrows():
        iid, axis = row.ImageID.split('_Fovea_')
        delta_by_id.setdefault(iid,{})[axis] = float(row.delta_px)
    movement = np.array([np.hypot(item['X'],item['Y']) for item in delta_by_id.values()])
    require(len(movement) == 20, 'Expected 20 test prediction displacements')
    changed_rows = int((changes.delta_px != 0).sum())
    if not accepted['2026']:
        require(previous_sha == candidate_sha, 'No accepted primary folds but CSV changed')
    primary_comparison = next(item for item in paired if item['baseline']=='PREVIOUS_SUBMISSION')
    comparison = {'status':'COMPLETE_AND_INDEPENDENTLY_VERIFIED', 'primary_candidate_vs_previous':primary_comparison,
                  'paired_training_oof_comparisons':paired,'accepted_folds_by_split_zero_based':accepted,
                  'previous_submission_sha256':previous_sha,'candidate_submission_sha256':candidate_sha,
                  'submission_bytes_changed':previous_sha!=candidate_sha,'changed_coordinate_rows':changed_rows,
                  'maximum_absolute_coordinate_change_px':float(changes.delta_px.abs().max()),
                  'mean_test_prediction_displacement_px':float(movement.mean()),
                  'maximum_test_prediction_displacement_px':float(movement.max()),
                  'new_training_stages':30,'reloaded_final_networks':30,
                  'protected_previous_verified_files':validation['protected_previous_verified_files'],
                  'new_competition_score':None,'reference_scores_used_for_training_or_selection':False,
                  'external_reference_scoring_scope':'Any later reference audit is separate, after candidate freeze; this finalizer never reads it.',
                  'separate_user_requested_manual_diagnostic_branch':'Exists outside this experiment; not read or used as a model baseline/input.',
                  'manual_postprocessing_used_as_baseline_or_model_output':False}
    write_json(OUT/'comparison.json',comparison)
    for name in ('geometry_cpu_validation.json','gpu_probe.json','source_manifest.json'):
        shutil.copy2(WORK/name,OUT/name)

    def summary_row(seed,method):
        return next(item for item in summaries if item['split_seed']==seed and item['method']==method)
    labels = {'COARSE':'粗定位', 'SINGLE':'旧 SINGLE：0.15 / 256 / 64',
              'WIDE_DEV':'WIDE_DEV：原开发组47–48图', 'WIDE_FULL':'WIDE_FULL：全64图',
              'MATCHED_DEPLOY':'固定门槛的 SINGLE/WIDE_FULL 对照',
              'PREVIOUS_SUBMISSION':'上一轮真实模型提交方案', 'SUBMISSION_CANDIDATE':'本轮固定规则提交方案'}
    tables = []
    for seed in SPLITS:
        rows = [f"| {labels[item['method']]} | {item['rmse']:.6f} | {item['p90_distance_px']:.6f} | {item['max_distance_px']:.6f} | {item['worse_than_coarse_fraction']:.2%} |"
                for item in summaries if item['split_seed']==seed]
        tables.append(f"### 分组种子 {seed}\n\n| 方法 | 坐标 RMSE | P90 欧氏误差 | 最大欧氏误差 | 比粗定位更差比例 |\n|---|---:|---:|---:|---:|\n"+'\n'.join(rows))
    gate_table = '\n'.join(f"| {item['split_seed']} | {item['fold']} | {item['selected_epoch']} | {item['single_selection_rmse']:.4f} | {item['wide_dev_selection_rmse']:.4f} | {'通过' if item['accepted'] else '不通过'} | {item['dev_successful_updates']} / {item['full_successful_updates']} |"
                           for item in decisions)
    difference = primary_comparison['rmse_delta_px']
    metric_statement = (f'主划分提交方案的 OOF 坐标 RMSE 降低 {-difference:.6f}px。' if difference < -1e-9 else
                        f'主划分提交方案的 OOF 坐标 RMSE 升高 {difference:.6f}px，本地证据不支持宣称这次提交优于上一轮。' if difference > 1e-9 else
                        '主划分提交方案的 OOF 坐标 RMSE 与上一轮相同。')
    identity_statement = ('最终 CSV 与上一轮逐字节相同，不能称作新增改进预测。' if previous_sha==candidate_sha else
                          f'最终 CSV 与上一轮不同，40 个坐标中有 {changed_rows} 个改变。')
    role_count = len(protected['files'])
    physical = geometry['physical_scale_example_for_shortside_2000']
    coverage_lines = '\n'.join(f"- {seed}：旧 SINGLE 窗口覆盖 {summary_row(seed,'SINGLE')['old_single_roi_coverage']:.2%}；新 0.25 窗口覆盖 {summary_row(seed,'WIDE_FULL')['proposed_roi_coverage']:.2%}。" for seed in SPLITS)
    ci = primary_comparison['fixed_prediction_group_bootstrap_95pct_delta_interval']
    replay_note = ''
    if (OUT/'verification_attempt1.json').is_file() and (WORK/'roi_replay_diagnosis.json').is_file():
        attempt = js(OUT/'verification_attempt1.json')
        diagnosis = js(WORK/'roi_replay_diagnosis.json')
        require(attempt['status'] == 'FAIL', 'Unexpected first verification attempt status')
        require(diagnosis['current_submission_sha256'] == candidate_sha, 'Replay diagnosis refers to another submission')
        operand = diagnosis['differing_operand']
        copied_delta = diagnosis['outputs']['copied_default_parse']['maximum_difference_px']
        original_delta = diagnosis['outputs']['original_round_trip']['maximum_difference_px']
        require(diagnosis['tolerance_px'] == validation['independent_reload_tolerance_px'] == .001,
                'Replay tolerance changed')
        require(copied_delta >= diagnosis['tolerance_px'] and original_delta == 0.0,
                'Replay diagnosis no longer supports the reported correction')
        replay_note = (f"验证过程保留了一次失败记录：首次旧 SINGLE OOF 重载差异为 {copied_delta:.12f}px，超过未放宽的 "
                       f"{diagnosis['tolerance_px']:.3f}px 阈值。原因是复制旧 ROI CSV 时默认浮点解析把窗口边长 "
                       f"`{operand['original_csv_literal']}` 写成 `{operand['copied_csv_literal']}`；相同模型、完整帧及批次下，"
                       f"改用原生产者 ROI 文件并以 round-trip 解析重放，差异恢复为 {original_delta:.1f}px。"
                       "仅修正验证器对继承旧 OOF/test 预测的输入重放；模型、训练引擎、提交 CSV、接入门槛和误差容差均未改变。"
                       "首次失败原样保留；本报告的 PASS 来自修正后完整重跑，相关证据随 ZIP 保存。")
    report = f'''# 宽窗口单分支精修：两套五折与全训练组重训

{metric_statement} {identity_statement} 真实线上成绩在训练与选型时未知。本模型训练与选择流程不读取外部测试参考标签；同会话另有用户指定人工诊断支路，独立保存，不进入本实验。本报告生成器也不读取该诊断支路或外部参考标签。

## 实际改动与数据边界

保留原三通道 ImageNet ResNet18，将 fine 改为图像短边 **0.25** 的窗口、384×384 输入、96×96 热图、sigma=1.5。沿用两套分组种子2026/2027及已核验的粗定位模型与图像留出裁剪中心。早停只在内部 stopping 组、真实部署尺度下，以原图坐标 RMSE 和同一 TTA 方式选轮。没有重新训练粗网络，也没有添加新回退分类器。

每折包含 `wide_select → wide_dev → 冻结接入决策 → wide_full`。WIDE_DEV 只使用原开发组47–48张；WIDE_FULL在固定轮数下加入原校准组，合计64张非外层图。两套五折共 **30 个新训练阶段**。DEV/FULL同初始化种子、同epoch数，但FULL每轮样本和实际优化步数更多；这是训练数据量与更新预算的联合对照，不能把收益全部归因于新增图像。

原校准组先比较冻结旧 SINGLE 与 WIDE_DEV，要求 RMSE 至少降低 **0.5px 且2%**，并且 P90 不升高；通过与否在 FULL 重训前保存。此组随后成为 FULL 训练数据，因此 **FULL没有独立校准组**。旧半径、风险回退及覆盖率校准不会移植到新FULL模型。OOF外层图始终不参与上述训练或选择。

本次对照同时改变窗口和输入/热图尺寸，不能称为单因素“增加分辨率”。短边2000px的例子中，旧300px窗口变500px；原图每输入像素由{physical['baseline_original_px_per_input_px']:.6f}变{physical['wide_original_px_per_input_px']:.6f}，sigma物理宽度由{physical['baseline_sigma_original_px']:.6f}px变{physical['wide_sigma_original_px']:.6f}px，训练中心抖动标准差由{physical['baseline_jitter_sigma_px']:.1f}px变{physical['wide_jitter_sigma_px']:.1f}px。训练时保留并记录目标出框扩大机制；验证和推理不使用标签修复裁剪。

## 训练结果

GPU：{execution['gpu']}；新增训练与折内推理耗时 {execution['elapsed_seconds']/60:.2f} 分钟，记录的峰值CUDA分配内存 {execution['peak_cuda_mb']:.2f}MB。独立验证重载30个最终网络（20个新DEV/FULL和10个旧SINGLE），状态PASS；测试图进入训练或选型的数量为0。

{replay_note}

{chr(10).join(tables)}

窗口覆盖率按实际坐标几何单独检查：

{coverage_lines}

这些覆盖率描述旧/新裁剪提议，并不是融合模型拥有统一窗口或获得覆盖保证。融合预测可能来自多个分支，不能将表中的共享几何诊断解释为每种方法自己的可校准置信区间。

## 逐折选择与固定提交规则

| 分组种子 | 折 | 选定epoch | 旧SINGLE选择组RMSE | WIDE_DEV选择组RMSE | 接入 | DEV/FULL实际优化步数 |
|---|---:|---:|---:|---:|---|---:|
{gate_table}

主划分2026通过 {len(accepted['2026'])}/5 折，副划分2027通过 {len(accepted['2027'])}/5 折。主划分通过的折固定为 `0.75×上一轮该折提交预测 + 0.25×WIDE_FULL`；不通过的折保留上一轮预测，最终平均全部五折。副划分只做训练稳定性审计，不生成测试预测，也不改变主划分提交权重。

主划分提交相对上一轮：{primary_comparison['improved_images']}张OOF图改善，{primary_comparison['worsened_images']}张变差，{primary_comparison['unchanged_images']}张不变。固定预测的组bootstrap差值95%区间为 [{ci[0]:.6f}, {ci[1]:.6f}]px；其他配对结果在 `comparison.json`。这是2000次固定预测重采样，不涵盖重新训练随机性、反复查看OOF的选择偏差或患者相关性，不能当作新的独立检验。

## 文件与版本保护

- `submission.csv`：真实模型按事先固定规则生成的候选；没有逐坐标手工后处理。
- `previous_submission.csv`：上一轮真实双尺度模型提交的原样副本，SHA256 `{previous_sha}`。
- `baseline_8p2.csv`：更早的8.2版本原样副本，SHA256 `{BASELINE_SHA}`。
- 新提交SHA256：`{candidate_sha}`。
- 新旧实际CSV坐标平均移动 {movement.mean():.6f}px、最大移动 {movement.max():.6f}px；这是预测位移，不是测试误差。
- `fold_decisions.csv`、`oof_tail_errors.csv`、`recomputed_metrics.csv`：逐折选择、所有训练图尾部误差与重算指标。
- `validation_report.json`：独立重载、隔离、来源与旧版本保护核验；{role_count}个旧文件的完整性检查通过。
- `wide_refine_documents.zip`：本报告、源代码、小型记录及三份CSV；不含训练权重，也不包含Kaggle Notebook。模型仍保留在本轮实验目录。

本次没有覆盖之前的版本。独立参考分数审计如果随后进行，必须放在训练项目外并明确晚于候选冻结；它不进入本报告的算法对照、门槛或模型基线。同会话的用户指定人工诊断支路独立保存，不属于这里的模型结果，也不成为任何训练或选择输入。

## 如何理解结论

单分支WIDE_DEV/FULL与旧SINGLE的两个划分结果须一起看，不能只报告较好的划分。两套划分均来自同样80张已被多轮开发查看的图像，所以只能说明开发集稳定性；没有患者ID，精确重复组隔离也不等于患者独立。实际线上效果需要另行验证，本报告不保证提分，也不会通过修改门槛追求事后更好的OOF。
'''
    (OUT/'report.md').write_text(report,encoding='utf-8')
    (OUT/'README.md').write_text(f'''# 宽窗口精修交付

{metric_statement} {identity_statement}

请先读 report.md。submission.csv 是固定规则的模型输出，未进行手工坐标后处理；previous_submission.csv 和 baseline_8p2.csv 为历史副本。本模型训练与选择流程不读取外部测试参考标签；同会话另有用户指定人工诊断支路，独立保存，不进入本实验。

本轮新训练30阶段，两个完整五折；验证PASS并独立重载30个最终网络。旧文件保护核验数量为{role_count}。文档ZIP只含源代码、记录和CSV，不含模型权重或Notebook；源代码复现实验依赖原工程中已核验的历史粗模型和数据。
''',encoding='utf-8')

    # Only explicitly listed new experiment records and source files are packed.
    names = ['README.md','report.md','submission.csv','previous_submission.csv','baseline_8p2.csv',
             'comparison.json','fold_decisions.csv','oof_tail_errors.csv','recomputed_metrics.csv',
             'submission_changes.csv','ablation_summary.csv','execution_report.json','validation_report.json',
             'run_protocol.json','planned_experiment.json','prepared_config.json','source_manifest.json',
             'protected_previous.json','geometry_cpu_validation.json','gpu_probe.json','test_manifest.csv']
    files = {name:OUT/name for name in names}
    for name, path in (
        ('verification_history/verification_attempt1.json', OUT/'verification_attempt1.json'),
        ('verification_history/roi_replay_diagnosis.json', WORK/'roi_replay_diagnosis.json'),
        ('verification_history/verifier_preflight.md', WORK/'verifier_preflight.md'),
    ):
        if path.is_file():
            files[name] = path
    for name in ('engine.py','pipeline.py','prepare.py','run.py','test_geometry_cpu.py','probe.py','verify_results.py','finalize.py'):
        path = WORK/name
        if path.is_file():
            files[f'reproduction_source/{name}'] = path
    files['reproduction_source/previous_dual_engine.py'] = ROOT/'work/dual_context_20261027/engine.py'
    for seed in SPLITS:
        folder = OUT/f'split_{seed}'
        for path in folder.rglob('*'):
            if path.is_file() and path.suffix in ('.json','.csv'):
                require(path.stat().st_size < 8*1024*1024, f'Unexpectedly large record: {path}')
                files[path.relative_to(OUT).as_posix()] = path
    require(all(path.is_file() for path in files.values()), 'Missing deliverable record')
    archive_manifest = {name:{'bytes':path.stat().st_size,'sha256':sha(path)} for name,path in sorted(files.items())}
    archive = OUT/'wide_refine_documents.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for name,path in sorted(files.items()):
            z.write(path,name)
        z.writestr('manifest.json',json.dumps(archive_manifest,ensure_ascii=False,indent=2))
    with zipfile.ZipFile(archive) as z:
        require(z.testzip() is None,'Document archive CRC failure')
        require(set(z.namelist()) == set(archive_manifest)|{'manifest.json'},'Unexpected archive entries')
        for name,record in archive_manifest.items():
            data = z.read(name)
            require(len(data)==record['bytes'] and hashlib.sha256(data).hexdigest()==record['sha256'],f'Archive data changed: {name}')
    write_json(OUT/'package_verification.json',{'status':'PASS','archive':str(archive),'sha256':sha(archive),
               'files':len(archive_manifest),'all_archived_bytes_verified':True,
               'trained_weights_included':False,'pretrained_weights_included':False,
               'notebook_included':False,'external_reference_files_read':False,'training_launched':False})
    print(json.dumps(comparison,ensure_ascii=False,indent=2),flush=True)
    print('PACKAGED:',archive,flush=True)


if __name__ == '__main__':
    main()
