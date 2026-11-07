"""Report and package completed precision experiments; never train or alter gates.

Run only after run.py and verify_results.py have finished successfully. This
module reads the genuine execution/validation artifacts and does not invent
missing metrics. Importing it does not create reports or launch computation.
"""
from pathlib import Path
import hashlib
import json
import shutil
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[2]
WORK = Path(__file__).resolve().parent
OUT = ROOT / "outputs/research82_precision_20261027"
PHASE1 = ROOT / "outputs/research82_20261027"
PAPERS = ROOT / "work/research82"
METHODS = ("BASELINE8P2", "resnet18_narrow", "GUARDED20")
NOTEBOOK = "fovea_research82_precision_kaggle.ipynb"


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def js(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")


def main():
    sys.path.insert(0, str(ROOT / "work/gpu_test_deps"))
    import numpy as np
    import pandas as pd

    def read(path):
        return pd.read_csv(path, dtype={"image_id": str, "group": str,
                                        "ImageID": str})

    def align(reference, candidate):
        require(reference.image_id.is_unique and candidate.image_id.is_unique,
                "Duplicate image identifiers in comparison")
        require(set(reference.image_id) == set(candidate.image_id),
                "Comparison identifier sets differ")
        return candidate.set_index("image_id").loc[reference.image_id].reset_index()

    def metric(frame):
        difference = frame[["x", "y"]].to_numpy() - frame[["x_true", "y_true"]].to_numpy()
        distance = np.linalg.norm(difference, axis=1)
        return {"rmse": float(np.sqrt(np.mean(difference ** 2))),
                "median_distance_px": float(np.median(distance)),
                "p90_distance_px": float(np.quantile(distance, .9)),
                "max_distance_px": float(distance.max())}

    # A partial run must not produce a completed report or a deliverable ZIP.
    execution = js(OUT / "execution_report.json")
    validation = js(OUT / "validation_report.json")
    protocol = js(OUT / "run_protocol.json")
    notebook_validation = js(OUT / "notebook_validation.json")
    source_manifest = js(WORK / "source_manifest.json")
    backup = js(OUT / "backup_verification.json")
    require(execution["status"] == "COMPLETE" and execution["folds"] == 5,
            "Precision training has not completed five folds")
    require(validation["status"] == "PASS", "Independent validation did not pass")
    require(execution["new_training_stages"] == validation["new_training_stages"] == 15,
            "Expected exactly 15 new training stages")
    require(validation["reloaded_final_networks"] == 25 and
            validation["baseline_final_networks_unchanged"] == 20,
            "Expected independent reload of 20 baseline and 5 new final networks")
    require(protocol["variants"] == ["resnet18_narrow"], "Unexpected variant set")
    require(protocol["blend_weight"] == .20 and
            protocol["minimum_absolute_gain_px"] == .5 and
            protocol["minimum_relative_gain"] == .02,
            "The predeclared gate or blend differs from this report protocol")
    require(validation["test_images_in_training_or_selection"] == 0,
            "Validation detected test data in training or selection")
    require(validation["source_engine_sha256"] == source_manifest["engine_sha256"] ==
            sha(WORK / "engine.py"), "Engine identity changed after validation")
    require(notebook_validation["status"] == "PASS" and
            notebook_validation["total_training_stages"] == 95,
            "The 95-stage Kaggle Notebook has not passed static validation")
    require(notebook_validation["notebook_sha256"] == sha(OUT / NOTEBOOK),
            "Notebook changed after static validation")
    require(notebook_validation["engine_sha256"] == source_manifest["engine_sha256"],
            "Notebook and the verified local engine differ")
    require(backup["status"] == "PASS", "Baseline backup reference is not verified")

    baseline_path = OUT / "baseline_8p2.csv"
    submission_path = OUT / "submission.csv"
    baseline_sha, submission_sha = sha(baseline_path), sha(submission_path)
    require(baseline_sha == validation["baseline_sha256"] == backup["submission_sha256"],
            "Original 8.2 submission changed")
    require(submission_sha == validation["submission_sha256"] == execution["submission_sha256"],
            "Candidate CSV changed after execution or independent validation")

    summary = read(OUT / "ablation_summary.csv").set_index("method")
    require(summary.index.is_unique and set(summary.index) == set(METHODS),
            "Unexpected or incomplete ablation summary")
    summary = summary.loc[list(METHODS)]
    base = read(OUT / "oof_BASELINE8P2.csv").sort_values("image_id").reset_index(drop=True)
    pred = align(base, read(OUT / "oof_GUARDED20.csv"))
    require(len(base) == len(pred) == 80, "Expected 80 OOF rows")
    require(base.group.notna().all(), "Missing group identifiers")
    require(np.array_equal(base[["x_true", "y_true", "fold"]].to_numpy(),
                           pred[["x_true", "y_true", "fold"]].to_numpy()),
            "OOF labels or fold assignments differ")
    for name in METHODS:
        frame = align(base, read(OUT / f"oof_{name}.csv"))
        for key, value in metric(frame).items():
            require(np.isclose(value, float(summary.loc[name, key]), rtol=1e-10, atol=1e-8),
                    f"Summary no longer matches OOF: {name}/{key}")

    details = base[["image_id", "fold", "group", "width", "height"]].copy()
    details["baseline_error_px"] = np.hypot(base.x - base.x_true, base.y - base.y_true)
    details["candidate_error_px"] = np.hypot(pred.x - pred.x_true, pred.y - pred.y_true)
    details["change_px"] = details.candidate_error_px - details.baseline_error_px
    details.sort_values("change_px", ascending=False).to_csv(OUT / "oof_error_changes.csv", index=False)

    baseline_test = read(OUT / "test_ensemble_BASELINE8P2.csv").sort_values("image_id").reset_index(drop=True)
    candidate_test = align(baseline_test, read(OUT / "test_ensemble_GUARDED20.csv"))
    require(len(baseline_test) == len(candidate_test) == 20, "Expected 20 test images")
    test_changes = baseline_test[["image_id", "x", "y"]].rename(
        columns={"x": "baseline_x", "y": "baseline_y"})
    test_changes[["candidate_x", "candidate_y"]] = candidate_test[["x", "y"]].to_numpy()
    test_changes["displacement_px"] = np.hypot(candidate_test.x - baseline_test.x,
                                                candidate_test.y - baseline_test.y)
    test_changes.to_csv(OUT / "test_prediction_changes.csv", index=False)

    baseline_csv, candidate_csv = read(baseline_path), read(submission_path)
    require(list(baseline_csv.columns) == list(candidate_csv.columns) == ["ImageID", "value"],
            "Submission columns differ from the required format")
    require(baseline_csv.ImageID.is_unique and candidate_csv.ImageID.is_unique and
            baseline_csv.ImageID.tolist() == candidate_csv.ImageID.tolist() and len(candidate_csv) == 40,
            "Submission row identity or order differs")
    csv_changes = pd.DataFrame({"ImageID": baseline_csv.ImageID,
                                "baseline_value": baseline_csv.value,
                                "candidate_value": candidate_csv.value})
    csv_changes["delta_px"] = csv_changes.candidate_value - csv_changes.baseline_value
    require(np.isfinite(csv_changes[["baseline_value", "candidate_value", "delta_px"]]).all().all(),
            "Non-finite submitted coordinates")
    csv_changes.to_csv(OUT / "submission_value_changes.csv", index=False)

    folds = []
    selections = []
    for fold in range(5):
        metrics = read(OUT / f"fold_{fold}/metrics.csv").set_index("method")
        selection = js(OUT / f"fold_{fold}/selection.json")
        selections.append(selection)
        require(selection["selected"] == "resnet18_narrow" and
                len(selection["candidates"]) == 1 and
                selection["candidates"][0]["variant"] == "resnet18_narrow",
                "Unexpected selected variant")
        candidate = selection["candidates"][0]
        proxy = float(selection["proxy_resnet18_inner_rmse"])
        gain = proxy - float(candidate["best_rmse"])
        accepted = gain >= .5 and gain >= .02 * proxy
        require(selection["accepted"] == accepted and
                selection["blend_weight"] == (.20 if accepted else 0.),
                "Recorded decision differs from predeclared gate")
        require(selection["proxy_is_8p2_ensemble"] is False,
                "Internal proxy was mislabeled as ensemble performance")
        final_config = js(OUT / f"fold_{fold}/resnet18_narrow/final_config.json")["config"]
        require(final_config["HEATMAP_SIGMA"] == 1.5 and
                final_config["FINE_HEATMAP_SIZE"] == 64 and
                final_config["FINE_INPUT_SIZE"] == 256 and
                final_config["FINE_VARIANT"] == "resnet18",
                "The actual trained fine configuration is not the precision candidate")
        folds.append({"fold": fold, "selected_variant": selection["selected"],
                      "selected_policy": candidate["policy"], "selected_epoch": int(candidate["best_epoch"]),
                      "proxy_resnet18_inner_rmse": proxy, "candidate_inner_rmse": float(candidate["best_rmse"]),
                      "inner_gain_px": gain, "accepted": accepted, "new_weight": float(selection["blend_weight"]),
                      "baseline_rmse": float(metrics.loc["BASELINE8P2", "rmse"]),
                      "branch_rmse": float(metrics.loc["resnet18_narrow", "rmse"]),
                      "candidate_rmse": float(metrics.loc["GUARDED20", "rmse"])})
    require(execution["selected_folds"] == selections, "Execution decisions do not match fold records")
    accepted_ids = [row["fold"] for row in folds if row["accepted"]]
    require(validation["accepted_folds"] == accepted_ids, "Validator accepted-fold list differs")
    pd.DataFrame(folds).to_csv(OUT / "fold_comparison.csv", index=False)
    if not accepted_ids:
        require(baseline_sha == submission_sha,
                "No fold passed, but the exported CSV is not byte-identical to 8.2")

    # Descriptive uncertainty for these fixed predictions only, not a claim
    # about retraining, patient independence, or repeated model selection.
    groups = base.group.unique()
    indexes_by_group = {group: np.flatnonzero(base.group.to_numpy() == group) for group in groups}
    baseline_squared = ((base[["x", "y"]].to_numpy() - base[["x_true", "y_true"]].to_numpy()) ** 2).mean(axis=1)
    candidate_squared = ((pred[["x", "y"]].to_numpy() - pred[["x_true", "y_true"]].to_numpy()) ** 2).mean(axis=1)
    rng = np.random.default_rng(827)
    deltas = []
    for _ in range(2000):
        indexes = np.concatenate([indexes_by_group[group] for group in rng.choice(groups, len(groups), replace=True)])
        deltas.append(float(np.sqrt(candidate_squared[indexes].mean()) - np.sqrt(baseline_squared[indexes].mean())))
    comparison = {
        "baseline_rmse": float(np.sqrt(baseline_squared.mean())),
        "branch_rmse": float(summary.loc["resnet18_narrow", "rmse"]),
        "candidate_rmse": float(np.sqrt(candidate_squared.mean())),
        "delta_rmse": float(np.sqrt(candidate_squared.mean()) - np.sqrt(baseline_squared.mean())),
        "fixed_prediction_group_bootstrap_95pct_delta_interval": np.quantile(deltas, [.025, .975]).tolist(),
        "images_improved": int((details.change_px < -1e-9).sum()),
        "images_worsened": int((details.change_px > 1e-9).sum()),
        "images_unchanged": int((details.change_px.abs() <= 1e-9).sum()),
        "mean_test_displacement_px": float(test_changes.displacement_px.mean()),
        "max_test_displacement_px": float(test_changes.displacement_px.max()),
        "submitted_coordinate_rows_changed": int((csv_changes.delta_px != 0).sum()),
        "max_submitted_coordinate_change_px": float(csv_changes.delta_px.abs().max()),
        "submission_bytes_changed": baseline_sha != submission_sha,
        "submitted_prediction_values_changed": bool((csv_changes.delta_px != 0).any()),
        "baseline_submission_sha256": baseline_sha, "candidate_submission_sha256": submission_sha,
        "accepted_folds": len(accepted_ids), "accepted_fold_ids_zero_based": accepted_ids,
        "new_training_stages": 15, "reloaded_final_networks": 25,
        "baseline_online_score": 8.2, "candidate_online_score": None,
        "online_score_source": "Baseline 8.2 is user feedback; no new Kaggle result was read or submitted by this script.",
        "scope": "Repeated development OOF; fixed-prediction group bootstrap does not include retraining randomness or selection bias.",
    }
    write_json(OUT / "comparison.json", comparison)

    # Phase 1 is negative evidence and is never mixed into this phase's output.
    prior_execution = js(PHASE1 / "execution_report.json")
    prior_validation = js(PHASE1 / "validation_report.json")
    prior_summary = read(PHASE1 / "ablation_summary.csv").set_index("method")
    require(prior_execution["status"] == "COMPLETE" and prior_validation["status"] == "PASS",
            "Phase 1 background is not a completed, verified experiment")
    prior_accepted = sum(bool(row["accepted"]) for row in prior_execution["selected_folds"])
    prior_same = sha(PHASE1 / "submission.csv") == baseline_sha
    require(prior_accepted == 0 and prior_same, "Phase 1 no longer matches the recorded negative background")
    phase1_background = {"status": "COMPLETE_AND_VALIDATED", "accepted_folds": prior_accepted,
        "submission_identical_to_8p2": prior_same,
        "oof_rmse": {str(name): float(row.rmse) for name, row in prior_summary.iterrows()},
        "source_sha256": {name: sha(PHASE1 / name) for name in
                          ("execution_report.json", "validation_report.json", "ablation_summary.csv", "submission.csv")},
        "interpretation": "Previous architecture candidates were not accepted; these are development results, not new Kaggle scores."}
    write_json(OUT / "phase1_background.json", phase1_background)

    # Optional evidence belongs to this precision phase, never the prior probe.
    optional_names = []
    optional_descriptions = []
    optional_sources = {
        "error_diagnosis.md": WORK / "error_diagnosis.md",
        "gpu_probe.json": WORK / "gpu_probe.json",
        "notebook_terminal_audit.json": OUT / "notebook_terminal_audit.json",
    }
    for name, source in optional_sources.items():
        if not source.is_file():
            continue
        if name.endswith(".json"):
            evidence = js(source)
            require(evidence.get("status") == "PASS", f"Optional evidence did not pass: {name}")
            if name == "notebook_terminal_audit.json":
                require(evidence.get("notebook_sha256") == notebook_validation["notebook_sha256"],
                        "Optional terminal audit refers to a different Notebook")
            if name == "gpu_probe.json":
                require(evidence.get("fine_sigma_heatmap_cells") == 1.5,
                        "Optional GPU probe is not the precision sigma=1.5 experiment")
        destination = OUT / name
        if source.resolve() != destination.resolve():
            shutil.copy2(source, destination)
        optional_names.append(name)
    if "error_diagnosis.md" in optional_names:
        optional_descriptions.append("- [error_diagnosis.md](error_diagnosis.md)：补充 OOF 大误差与 ROI 诊断；诊断本身不改变既定训练和接入规则。")
    if "gpu_probe.json" in optional_names:
        optional_descriptions.append("- [gpu_probe.json](gpu_probe.json)：本轮 sigma=1.5 的实际 GPU 小样本功能探针，状态 PASS；探针权重不用于提交，其结果不替代正式五折指标。")
    if "notebook_terminal_audit.json" in optional_names:
        optional_descriptions.append("- [notebook_terminal_audit.json](notebook_terminal_audit.json)：独立 CPU 终检 PASS，核对离线权重、数据读取和编排路径；编排使用训练替身与历史预测，不是 Kaggle 上新完成的训练或新成绩。")
    optional_evidence_report = ("\n附加证据随本包提供：\n\n" + "\n".join(optional_descriptions) + "\n"
                                if optional_descriptions else "")

    labels = {"BASELINE8P2": "8.2 基线", "resnet18_narrow": "ResNet18 fine sigma=1.5 单分支",
              "GUARDED20": "按预设门槛融合的最终提交"}
    table = "\n".join(f"| {labels[name]} | {row.rmse:.6f} | {row.median_distance_px:.6f} | {row.p90_distance_px:.6f} | {row.max_distance_px:.6f} |"
                      for name, row in summary.iterrows())
    fold_table = "\n".join(f"| {row['fold'] + 1} | {row['selected_policy']} / {row['selected_epoch']} | {row['proxy_resnet18_inner_rmse']:.4f} | {row['candidate_inner_rmse']:.4f} | {row['inner_gain_px']:+.4f} | {'通过' if row['accepted'] else '未通过'} | {row['candidate_rmse']:.4f} |"
                           for row in folds)
    if not accepted_ids:
        decision = "五折均未通过预设内部门槛，因此最终 submission.csv 保留原 8.2 基线；两份 CSV 逐字节一致。本轮训练没有产生可报告的提交改进，不能将重新打包称为更高成绩的新模型。"
    elif not comparison["submitted_prediction_values_changed"]:
        decision = (f"{len(accepted_ids)}/5 折通过预设内部门槛，但实际导出的 40 个坐标值与 8.2 文件相同。"
                    "是否字节相同见下表，不能把通过门槛直接当作提交预测发生改变。")
    else:
        decision = (f"{len(accepted_ids)}/5 折通过预设内部门槛，已按固定 20% 权重接入新分支；"
                    f"实际 submission.csv 有 {comparison['submitted_coordinate_rows_changed']}/40 个坐标值改变。新文件的线上成绩尚未知。")
    if comparison["delta_rmse"] < -1e-9:
        result_sentence = f"最终提交对应的开发 OOF RMSE 比基线降低 {-comparison['delta_rmse']:.6f}px，这不等于 Kaggle 分数已改善。"
    elif comparison["delta_rmse"] > 1e-9:
        result_sentence = f"最终提交对应的开发 OOF RMSE 比基线增加 {comparison['delta_rmse']:.6f}px；本地证据没有显示融合改善，8.2 原文件仍保留。"
    else:
        result_sentence = "最终提交对应的开发 OOF RMSE 与基线相同，没有观察到融合后的本地收益。"
    byte_description = "是，逐字节一致" if not comparison["submission_bytes_changed"] else "否，文件字节不同"
    bootstrap = comparison["fixed_prediction_group_bootstrap_95pct_delta_interval"]
    report = f"""# 8.2 基线之后：精修热图监督宽度的受控实验

{decision}

8.2 是用户反馈的历史 Kaggle 分数，越低越好；本报告由真实运行产物生成，没有查询或伪造新的线上成绩。训练完成且独立验证 PASS 后才允许生成本报告。

## 为什么进行这项实验

前一轮已真实完成两个架构对照：8.2 基线开发 OOF RMSE 为 {prior_summary.loc['BASELINE8P2', 'rmse']:.6f}，EfficientNet-B0 为 {prior_summary.loc['efficientnet_b0', 'rmse']:.6f}，ResNet18 真实 128×128 热图解码为 {prior_summary.loc['resnet18_highres', 'rmse']:.6f}。五折均未过门槛，提交与原 8.2 CSV 相同。这是阴性结果，说明在当前设置下更换编码器或增加高分辨率解码没有产生可接入收益，不能据此宣称这些架构普遍无效。

本轮唯一新增处理是 **ResNet18 fine 监督 Gaussian 的 sigma 从 2.5 降为 1.5 个热图格点**。输入仍为 256×256、输出仍为 64×64；保留原编码器、KL + DSNT 坐标损失、粗定位、随机种子、数据划分和既定接入规则。粗定位 sigma 仍为 2.5。不是继续放大网络，也不是多个 sigma 的网格搜索。

[Thaler 等，Modeling Annotation Uncertainty with Gaussian Heatmaps in Landmark Localization，MELBA 2021](https://www.melba-journal.org/papers/2021:014.html) 说明监督 Gaussian 的宽度与标注不确定性有关。但论文采用的热图目标与本项目归一化 KL 不同；**sigma=1.5 是事先固定的待检验假设，不是论文已经证明的本任务最优值**。[DSNT 原文](https://arxiv.org/abs/1801.07372) 支持用分布期望得到连续坐标，不能把 64 格热图直接理解为只能输出整数格点。

[GAMMA](https://doi.org/10.1016/j.media.2023.102938)、[ADAM](https://doi.org/10.1109/TMI.2022.3172773) 和 [VascX](https://doi.org/10.1167/tvst.14.7.19) 支持 CNN、热图与集成这一算法家族，但不证明更窄 sigma 或某个骨干必胜。JOINEDTrans 与 2026 年 MNv4Fovea 的额外解剖标注、数据及算力条件也不同，详见附带文献笔记。不能将其他数据集的 AED/IoU 与本项目坐标 RMSE 或 8.2 直接排名。

## 数据、选轮与接入协议

本轮沿用 80 张训练图、{len(groups)} 个精确重复组与原外层五折；没有患者 ID，因此不能声称已经实现患者级独立。训练/选择与测试图重叠数由独立验证器核对为 {validation['test_images_in_training_or_selection']}。仅使用官方 ResNet18 ImageNet 初始化，没有引入外部中心凹定位权重或标签。

每折仅用内部 stopping 集选择 fixed/multiscale 策略和轮数，再以所选轮数重训 fine；独立 calibration 组用于重算回退规则。沿用经身份校验的粗模型与图像级交叉预测裁剪中心。单候选每折 3 个训练阶段，五折共 **15 个新训练阶段**；独立重载验证 **25 个最终网络：20 个旧基线网络 + 5 个新增 fine 网络**。阶段数与最终模型数量含义不同。

内部门槛始终为：相对原 seed10000 单 ResNet18 的内部 stop 最佳 RMSE，至少降低 **0.5px 且 2%**。代理分数不是 8.2 集成的内部验证分数；小规模内部集合还用于选择策略与轮数，存在选择噪声。

逐折基线 B=0.75×旧 seed0 + 0.25×seed10000。通过门槛则 P=0.8×B+0.2×新精修分支，否则 P=B；测试预测对五折 P 等权平均。没有根据本轮外层表现调低门槛或提高融合权重。报告中的单分支五折结果只作对照，不据此另选全局“最优”提交。

## 真实结果

GPU：{execution['gpu']}。本轮训练与折内推理耗时 {execution['elapsed_seconds'] / 60:.2f} 分钟；记录峰值 CUDA 分配内存 {execution['peak_cuda_mb']:.2f}MB。独立验证器状态：{validation['status']}。

| 方法 | OOF 坐标 RMSE | 中位欧氏误差 | P90 欧氏误差 | 最大欧氏误差 |
|---|---:|---:|---:|---:|
{table}

| 折 | 内部所选策略 / 轮数 | 旧单模型内部 RMSE | 新分支内部 RMSE | 内部降低 px | 门槛 | 最终提交 OOF RMSE |
|---|---|---:|---:|---:|---|---:|
{fold_table}

{result_sentence} 按逐图欧氏距离统计，{comparison['images_improved']} 张改善、{comparison['images_worsened']} 张变差、{comparison['images_unchanged']} 张不变。

固定 OOF 预测的组 bootstrap 差值区间为 [{bootstrap[0]:.6f}, {bootstrap[1]:.6f}]px（2000 次，95% 分位区间）。这仅描述现有预测的重采样差异，不包含重新训练随机性、患者相关性或反复选方案的偏差，不应当作新的独立测试置信结论。

## CSV 是否真的发生变化

| 核对项 | 真实结果 |
|---|---|
| 通过接入门槛 | {len(accepted_ids)}/5 折 |
| 与原 8.2 CSV 是否相同 | {byte_description} |
| 实际导出坐标发生改变 | {comparison['submitted_coordinate_rows_changed']}/40 行 |
| 实际导出坐标最大绝对变化 | {comparison['max_submitted_coordinate_change_px']:.6f}px |
| 测试预测平均 / 最大移动 | {comparison['mean_test_displacement_px']:.6f} / {comparison['max_test_displacement_px']:.6f}px |
| 新的线上评估 | 尚无新反馈，本脚本未自动提交 |

预测移动不等于测试误差，因为没有测试标签。基线 SHA256：`{baseline_sha}`。最终提交 SHA256：`{submission_sha}`。`submission_value_changes.csv` 比较实际导出值，`test_prediction_changes.csv` 比较保存的浮点集成预测，避免把 CSV 舍入误差当作模型改变。

## 交付和复现

- `submission.csv`：严格按预先声明规则产生的结果；是否改变由上表明确说明。
- `baseline_8p2.csv`：原 8.2 提交的原样副本。
- `{NOTEBOOK}`：原生 Kaggle Notebook。附比赛数据和已有 `3pretrained` Dataset 中的 `resnet18-f37072fd.pth`，可关闭 Internet；无需 EfficientNet 权重。
- Notebook 从头重训两套基线共 80 阶段，再训练本轮 15 阶段，合计 **95 阶段**。本机复用已经核验的基线，只新增 15 阶段。Notebook 已静态核验，但本轮没有另行执行完整 95 阶段的 Kaggle 训练。
- `comparison.json`、逐折/逐图 CSV、`execution_report.json` 与 `validation_report.json` 提供真实结果和核验依据。
- `research82_precision_documents.zip` 包含上述文档、Notebook、两份 CSV 和复现源码，附逐文件 SHA256 清单；不含已训练模型或预训练权重。模型仍在各实验目录，历史 8.2 模型仍保存在原备份 ZIP。

历史备份引用：`{backup['backup']}`，其中 {backup['trained_final_models']} 个最终基线模型。precision 准备阶段引用既有备份并检查路径、大小和 CSV 摘要；不能把引用旧核验记录写成又创建或完整重验了一次新备份。

附带 `research_plan.md` 是前一轮架构方案的历史说明，`fovea_literature.md` 和 `keypoint_literature.md` 是研究笔记；**本报告与本轮 run_protocol.json 才对应当前 sigma=1.5 实验**。前一轮记录单独放在 `phase1_background.json`，不与本轮训练结果合并。
{optional_evidence_report}

## 能得出的结论与限制

80 张图与现有外层划分已用于多轮开发，本轮是重复开发 OOF，不能称为新独立验证。即使本地更好，也不能推断新的 Kaggle 分数；若文件相同，更不能声称线上提升。没有患者 ID 与近重复完整标注，精确重复组隔离不能保证患者级独立。最终融合坐标不自动继承单模型的不确定性覆盖率。不同 GPU、库版本与随机执行细节可能导致 Notebook 重训后不逐位复现本地 CSV。
"""
    (OUT / "report.md").write_text(report, encoding="utf-8")
    readme = f"""# Precision 实验交付

{decision}

先阅读 `report.md`；结构化结果见 `comparison.json`。`submission.csv` 为预设规则的输出，`baseline_8p2.csv` 为原 8.2 文件。8.2 来自用户反馈，本次没有新 Kaggle 分数。

导入 `{NOTEBOOK}`，添加比赛数据和已有 `3pretrained` Dataset（官方 `resnet18-f37072fd.pth`），开启 GPU 后运行。Internet 可关闭，无需 EfficientNet。Notebook 从头共 95 个训练阶段，本地本轮新增 15 阶段；25 个最终网络已独立重载核验。

文档 ZIP 不含训练权重，也不含预训练权重。基线模型备份继续保留于原实验备份。源码中 `finalize.py` 只整理已完成且独立验证 PASS 的结果，不执行训练、不改变接入门槛。
"""
    (OUT / "README.md").write_text(readme, encoding="utf-8")
    for name in ("research_plan.md", "fovea_literature.md", "keypoint_literature.md"):
        shutil.copy2(PAPERS / name, OUT / name)
    shutil.copy2(WORK / "source_manifest.json", OUT / "source_manifest.json")

    names = ["README.md", "report.md", "submission.csv", "baseline_8p2.csv", NOTEBOOK,
             "research_plan.md", "fovea_literature.md", "keypoint_literature.md",
             "comparison.json", "phase1_background.json", "ablation_summary.csv",
             "fold_comparison.csv", "oof_error_changes.csv", "test_prediction_changes.csv",
             "submission_value_changes.csv", "execution_report.json", "validation_report.json",
             "notebook_validation.json", "run_protocol.json", "source_manifest.json",
             "backup_verification.json", "baseline_identity.json", "planned_experiment.json",
             "prepared_config.json", "assembly_validation.json", "folds.csv", "test_manifest.csv"]
    names.extend(optional_names)
    archive_files = {name: OUT / name for name in names}
    for name in ("engine.py", "pipeline.py", "run.py", "assemble.py", "build_notebook.py",
                 "verify_results.py", "finalize.py"):
        archive_files[f"reproduction_source/{name}"] = WORK / name
    for fold in range(5):
        for name in ("selection.json", "metrics.csv", "split_roles.csv", "coarse_lineage.json"):
            relative = f"fold_{fold}/{name}"
            archive_files[relative] = OUT / relative
        for name in ("policy_selection.json", "final_config.json", "fallback_rule.json"):
            relative = f"fold_{fold}/resnet18_narrow/{name}"
            archive_files[relative] = OUT / relative
    for name in METHODS:
        for prefix in ("oof_", "test_ensemble_"):
            relative = f"{prefix}{name}.csv"
            archive_files[relative] = OUT / relative
    require(all(path.is_file() for path in archive_files.values()),
            "A required document, source file, or genuine experiment artifact is missing")
    manifest = {name: {"sha256": sha(path), "bytes": path.stat().st_size}
                for name, path in sorted(archive_files.items())}
    archive_path = OUT / "research82_precision_documents.zip"
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, path in sorted(archive_files.items()):
            archive.write(path, name)
        archive.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
    with zipfile.ZipFile(archive_path) as archive:
        require(archive.testzip() is None, "Document ZIP integrity test failed")
        require(set(archive.namelist()) == set(manifest) | {"manifest.json"},
                "Document ZIP entries differ from manifest")
        for name, entry in manifest.items():
            data = archive.read(name)
            require(len(data) == entry["bytes"] and hashlib.sha256(data).hexdigest() == entry["sha256"],
                    f"Archived bytes do not match source: {name}")
    write_json(OUT / "package_verification.json", {
        "status": "PASS", "archive": str(archive_path), "sha256": sha(archive_path),
        "files": len(manifest), "all_archived_bytes_verified": True,
        "trained_weights_included": False, "pretrained_weights_included": False,
        "new_training_launched_by_finalize": False,
    })
    print(json.dumps(comparison, ensure_ascii=False, indent=2), flush=True)
    print("PACKAGED:", archive_path, flush=True)


if __name__ == "__main__":
    main()
