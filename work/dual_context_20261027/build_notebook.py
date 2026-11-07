"""Build and statically verify the portable dual-context Kaggle Notebook.

This builder writes the notebook/report only; it never imports the training
engine or launches training. Run it after root freezes the assembled engine.
"""
from pathlib import Path
import ast
import hashlib
import json
import nbformat


ROOT = Path(__file__).resolve().parents[2]
WORK = Path(__file__).resolve().parent
OUT = ROOT / 'outputs/dual_context_20261027'


def definitions(source):
    result = {}
    for node in ast.parse(source).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            result[node.name] = ast.dump(node, include_attributes=False)
    return result


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    engine_path = WORK / 'engine.py'
    source = engine_path.read_text(encoding='utf-8')
    engine_digest = sha(engine_path)
    manifest = json.loads((WORK / 'source_manifest.json').read_text(encoding='utf-8'))
    assert manifest['engine_sha256'] == engine_digest
    config = json.loads((ROOT / 'outputs/restored_20261027/seed_10000/run_config.json').read_text(encoding='utf-8'))['config']
    for key in ('DATA_DIR', 'LABEL_CSV', 'SAMPLE_CSV', 'PRETRAINED_WEIGHTS_PATH'):
        config[key] = None
    config.update({'INPUT_ROOT': '/kaggle/input',
                   'OUTPUT_DIR': '/kaggle/working/baseline_seed0',
                   'SUBMISSION_PATH': '/kaggle/working/submission_seed0.csv',
                   'MODEL_SEED_OFFSET': 0, 'FINE_VARIANT': 'single', 'CONTEXT_RATIO': 2.0})
    assert config['HEATMAP_SIGMA'] == 2.5
    assert config['INPUT_SIZE'] == config['FINE_INPUT_SIZE'] == 256
    assert config['HEATMAP_SIZE'] == config['FINE_HEATMAP_SIZE'] == 64
    assert config['SEED'] == 2026 and config['N_FOLDS'] == 5
    assert config['FINE_MULTISCALE_TRAIN'] and config['FINE_MULTISCALE_EARLY_STOP']
    assert config['SMOKE_TEST'] is False and config['ALLOW_WEIGHT_DOWNLOAD'] is False

    intro = '''# 2026task3：双尺度精修、风险回退与两套五折对照

完整、可离线运行的 Kaggle Notebook。只需比赛数据和既有 `3pretrained` Dataset 中的官方 `resnet18-f37072fd.pth`，也支持原名 `fovea_pretrained_weights.zip`。所有权重先进行完整 SHA256 校验，缺失或损坏会明确报错。

## 运行步骤

1. 导入本 `.ipynb`，添加比赛数据和 ResNet18 预训练权重 Dataset。
2. 启用 GPU；Internet 可以关闭。代码不安装额外依赖，不下载其他竞赛模型或标签。
3. Run All，等待两个完整五折协议结束。最终提交路径为 `/kaggle/working/submission.csv`。

首次从头运行共 **145 个训练阶段**：重训两套旧方法基线各 40 阶段，共 80；本轮新实验 65 阶段。新实验中，原划分 `2026` 复用已核验粗模型和训练裁剪中心，单/双尺度两臂各选轮及最终重训，共 20 阶段；第二套划分 `2027` 重新训练粗模型与交叉预测中心，再做同样两臂，共 45 阶段。第二套划分不推理测试图，也不影响提交权重。

原 8.2 是用户反馈的历史成绩。Notebook 中先从两套真实重训结果产生 75/25 基线参考；不同 GPU、软件版本以及 CUDA 非确定性算子可能导致差异，不保证重现历史文件或线上成绩。本次重训参考明确命名为 `submission_retrained_8p2_method.csv`。

## 固定实验设计

- 单尺度与双尺度 fine 都使用相同的 256 输入、64 热图、sigma=1.5、训练样本、随机种子及增强。双尺度共享一个 ImageNet ResNet18，输入局部窗口与两倍物理范围的上下文，在四个编码层按真实像素中心位置对齐后融合。不是简单把低分辨率特征直接叠加。
- 单尺度只使用六通道输入的局部 RGB；两臂使用相同数据流程。局部沿用原光度增强；上下文复用完整物理几何变换，不再额外进行亮度增强。
- 本轮 coarse sigma 仍为 2.5；其余 fine 配置均写入各折文件。fine 的训练基础窗口与内部选轮窗口统一为图像短边的 15%，上下文为局部的两倍；训练集中的增强/训练目标出框修复仍单独记录，验证及推理不使用真值修复窗口。
- 独立 calibration 组用于风险规则：跨尺度位移、TTA 位移、第二峰比例；用预设的小型规则集合比较 fine、回退 coarse 或融合，记录留一组校准检验，避免把校准样本上的最小误差当作独立泛化成绩。
- 每折只有当双尺度的内部选轮 RMSE 相比匹配单尺度至少降低 0.5px 且 2%，才以 20% 权重接入该折原 8.2 方法预测；否则保留原预测。条件在外层与测试评估前固定，不按外层结果或测试参考答案改门槛。
- 两套五折都报告 RMSE、P90/P95、最大误差、窗口覆盖率和精修恶化比例。`2027` 是相同训练样本的另一套分组划分，属于开发稳定性审计，不是新患者或新数据集的独立验证。
- 外部参考答案、私榜划分和已知私榜分数不进入代码，不用于选模型、调参或改坐标。本 Notebook 不会自动向 Kaggle 提交。

全部代码来自冻结的本地实验引擎，单元没有伪造输出。此前文件和模型无需覆盖。Kaggle 从头运行与本机复用旧模型的耗时不同；145 阶段的总时长取决于设备、早停和会话限制。
'''

    setup = '''import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("MPLBACKEND", "Agg")
'''
    configure = 'BASE_CFG0 = DualConfig(**' + repr(config) + ''')
torch.set_num_threads(2)
assert torch.cuda.is_available(), "请在 Kaggle Notebook 设置中启用 GPU"
print("GPU:", torch.cuda.get_device_name(0))
print(research_geometry_selfcheck())
assert np.allclose(context_output_transform(256, 2.0) @ np.array([127.5,127.5,1.0]), [127.5,127.5,1.0])
Path(BASE_CFG0.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
BASE_CFG0.PRETRAINED_WEIGHTS_PATH = resolve_research_weights(BASE_CFG0)
assert research_file_digest(BASE_CFG0.PRETRAINED_WEIGHTS_PATH) == RESEARCH_WEIGHT_SHA256
print("Official ResNet18 SHA256 verified")
print(research_preflight(BASE_CFG0))
'''
    baselines = '''# Each baseline has five folds x eight stages = 40 stages.
# FINE_VARIANT="single" uses the original local architecture and original settings.
BASE0 = run_research(BASE_CFG0)
BASE_CFG1 = replace(BASE_CFG0, MODEL_SEED_OFFSET=10000,
    OUTPUT_DIR="/kaggle/working/baseline_seed10000",
    SUBMISSION_PATH="/kaggle/working/submission_seed10000.csv")
BASE1 = run_research(BASE_CFG1)
'''
    reference = '''# Use this run's genuine predictions as the reference; no historical values embedded.
BASE0_TEST = dc_read(Path(BASE0["config"].OUTPUT_DIR) / "test_ensemble_B4.csv")
BASE1_TEST = dc_read(Path(BASE1["config"].OUTPUT_DIR) / "test_ensemble_B4.csv")
BASE_REFERENCE_PREDICTION = dc_blend(BASE0_TEST, BASE1_TEST, .25)
BASE_REFERENCE_CSV = Path("/kaggle/working/submission_retrained_8p2_method.csv")
BASE_REFERENCE_SUBMISSION = dc_export(BASE0["config"].SUBMISSION_PATH,
    BASE_REFERENCE_PREDICTION, BASE0["test"], BASE_REFERENCE_CSV)
print("Current-run baseline reference:", BASE_REFERENCE_CSV)
display(BASE_REFERENCE_SUBMISSION.head(6))
'''
    experiment = '''# 20 new stages on split2026 plus 45 on split2027 = 65 new stages.
DUAL_CFG = replace(BASE_CFG1,
    OUTPUT_DIR="/kaggle/working/dual_context_results",
    SUBMISSION_PATH="/kaggle/working/dual_context_results/submission.csv",
    FINE_VARIANT="single", CONTEXT_RATIO=2.0,
    HEATMAP_SIGMA=2.5, MODEL_SEED_OFFSET=10000,
    FINE_MULTISCALE_TRAIN=False, FINE_MULTISCALE_EARLY_STOP=False,
    FINE_VALIDATION_ROI_FRACTIONS=(.15,),
    SELECT_FINE_CROP_POLICY=False, FIXED_ROI_FRACTION=.15,
    ROI_MARGIN_PX=0.0)
assert DUAL_SPLIT_SEEDS == (2026, 2027)
assert DUAL_CFG.HEATMAP_SIGMA == 2.5
RESULT = run_dual_context(DUAL_CFG,
    old_run=BASE0["config"].OUTPUT_DIR,
    source_run=BASE1["config"].OUTPUT_DIR,
    baseline_csv=BASE_REFERENCE_CSV)
'''
    export = '''# The function returns only after both complete fivefold protocols finish.
import shutil
SOURCE_SUBMISSION = Path(DUAL_CFG.OUTPUT_DIR) / "submission.csv"
EXECUTION_REPORT = dc_json(Path(DUAL_CFG.OUTPUT_DIR) / "execution_report.json")
assert EXECUTION_REPORT["status"] == "COMPLETE"
assert EXECUTION_REPORT["completed_folds"] == 10
assert EXECUTION_REPORT["new_training_stages"] == 65
FINAL_SUBMISSION = Path("/kaggle/working/submission.csv")
shutil.copyfile(SOURCE_SUBMISSION, FINAL_SUBMISSION)
assert research_file_digest(FINAL_SUBMISSION) == research_file_digest(SOURCE_SUBMISSION)
display(RESULT["summary"])
display(pd.read_csv(FINAL_SUBMISSION).head(10))
print("Submission:", FINAL_SUBMISSION)
print("Its Kaggle score is unknown until evaluated by the competition.")
'''
    cells = [nbformat.v4.new_markdown_cell(intro), nbformat.v4.new_code_cell(setup),
             nbformat.v4.new_markdown_cell('## 1. 完整冻结代码\n\n包含数据读取、训练、模型、风险规则、两套五折和提交生成。'),
             nbformat.v4.new_code_cell(source),
             nbformat.v4.new_markdown_cell('## 2. 配置与离线权重预检'), nbformat.v4.new_code_cell(configure),
             nbformat.v4.new_markdown_cell('## 3. 重训两套基线：80 阶段'), nbformat.v4.new_code_cell(baselines),
             nbformat.v4.new_markdown_cell('## 4. 当前重训结果的 75/25 参考提交'), nbformat.v4.new_code_cell(reference),
             nbformat.v4.new_markdown_cell('## 5. 双尺度与风险规则对照：两套五折、65 阶段'), nbformat.v4.new_code_cell(experiment),
             nbformat.v4.new_markdown_cell('## 6. 完成检查与 submission.csv 导出'), nbformat.v4.new_code_cell(export)]
    notebook = nbformat.v4.new_notebook(cells=cells, metadata={
        'kernelspec': {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'},
        'language_info': {'name': 'python'},
        'kaggle': {'isInternetEnabled': False, 'isGpuEnabled': True, 'language': 'python'}})
    expected, actual, code_cells = definitions(source), {}, []
    for cell in cells:
        if cell.cell_type != 'code':
            continue
        compile(cell.source, 'kaggle_cell', 'exec')
        actual.update(definitions(cell.source))
        code_cells.append(cell.source)
        assert cell.outputs == [] and cell.execution_count is None
        assert 'C:/Users/' not in cell.source and 'C:\\Users\\' not in cell.source
        assert 'D:/xwechat_files' not in cell.source and 'score_audit' not in cell.source
    assert expected == actual
    assert "'HEATMAP_SIGMA': 2.5" in configure
    assert "'FINE_VARIANT': 'single'" in configure
    assert "'MODEL_SEED_OFFSET': 0" in configure
    assert baselines.count('run_research(') == 2
    assert 'dc_blend(BASE0_TEST, BASE1_TEST, .25)' in reference
    assert 'run_dual_context(' in experiment
    assert 'Path(DUAL_CFG.OUTPUT_DIR)' in export
    assert 'shutil.copyfile(SOURCE_SUBMISSION, FINAL_SUBMISSION)' in export
    assert sha(engine_path) == engine_digest
    nbformat.validate(notebook)
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / 'fovea_dual_context_kaggle.ipynb'
    nbformat.write(notebook, target)
    reopened = nbformat.read(target, as_version=4)
    nbformat.validate(reopened)
    assert reopened.cells[3].source == source
    validation = {'status': 'PASS', 'notebook': str(target), 'notebook_sha256': sha(target),
                  'engine_sha256': engine_digest, 'matched_final_definitions': len(expected),
                  'all_function_and_class_final_definitions_match_engine': True,
                  'all_code_cells_compile': True, 'nbformat_schema_roundtrip': 'PASS',
                  'cells': len(cells), 'code_cells': len(code_cells), 'no_fabricated_cell_outputs': True,
                  'no_local_machine_paths_or_external_score_reference_in_code': True,
                  'baseline_training_stages': 80, 'new_training_stages': 65,
                  'total_training_stages': 145, 'split_seeds': [2026, 2027],
                  'coarse_sigma': 2.5, 'fine_arm_sigma': 1.5, 'context_ratio': 2.0,
                  'only_resnet18_initialization_required': True,
                  'reference_generated_from_current_retrained_baselines': True,
                  'export_requires_both_fivefold_protocols_complete': True,
                  'execution_scope': 'Static notebook validation; builder does not import engine or launch training.'}
    (OUT / 'notebook_validation.json').write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(validation, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
