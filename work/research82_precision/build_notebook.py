"""Build a portable, offline Kaggle Notebook without touching training sources."""
from pathlib import Path
import ast
import hashlib
import json
import nbformat


ROOT = Path(__file__).resolve().parents[2]
WORK = Path(__file__).resolve().parent
OUT = ROOT / "outputs/research82_precision_20261027"
ENGINE = WORK / "engine.py"
SOURCE_CONFIG = ROOT / "outputs/restored_20261027/seed_10000/run_config.json"
source = ENGINE.read_text(encoding="utf-8")
engine_sha256 = hashlib.sha256(ENGINE.read_bytes()).hexdigest()
config = json.loads(SOURCE_CONFIG.read_text(encoding="utf-8"))["config"]
for key in ("DATA_DIR", "LABEL_CSV", "SAMPLE_CSV", "PRETRAINED_WEIGHTS_PATH"):
    config[key] = None
config.update({
    "INPUT_ROOT": "/kaggle/input",
    "OUTPUT_DIR": "/kaggle/working/baseline_seed0",
    "SUBMISSION_PATH": "/kaggle/working/submission_seed0.csv",
    "MODEL_SEED_OFFSET": 0,
    "FINE_VARIANT": "resnet18",
    "EFFNET_WEIGHTS_PATH": None,
})
assert config["SEED"] == 2026 and config["N_FOLDS"] == 5
assert config["INPUT_SIZE"] == config["FINE_INPUT_SIZE"] == 256
assert config["HEATMAP_SIZE"] == config["FINE_HEATMAP_SIZE"] == 64
assert config["SMOKE_TEST"] is False and config["ALLOW_WEIGHT_DOWNLOAD"] is False

intro = '''# 2026task3：保留 8.2 基线，单变量精细热图对照

本 Notebook 可在 Kaggle 离线从头训练。用户反馈既有 75/25 随机种子集成的线上成绩为 **8.2**（越低越好）；新候选的线上成绩未知，论文和本地验证都不能保证提分。

## 直接运行

1. 添加本比赛数据。
2. 添加已有的 `3pretrained` Dataset，包含官方 `resnet18-f37072fd.pth`；也支持保留 ZIP 的数据布局。
3. 启用 GPU，Internet 可以关闭。先完整核验 ResNet18 权重 SHA256，再开始训练。此实验只需这份权重。
4. Run All。最终文件为 `/kaggle/working/submission.csv`。

**首次完整运行共 95 个训练阶段**：两组五折 ResNet18 基线共 80 阶段；单个 fine 对照每折两种裁剪策略选轮及最终重训，共 15 个新阶段。本地实验可复用已有基线；本 Notebook 不依赖本机路径或已训练的中心凹权重，所以会重训全部阶段。运行时间取决于 GPU、早停轮数和 Kaggle 会话时长。

原样得到 8.2 的预测另行保存为 `baseline_8p2.csv`。这里先从真实重训结果生成 `submission_retrained_8p2_method.csv`，不同硬件和库版本可能导致差异，不能承诺逐位复现或重现相同线上分数。

## 固定的单变量方案

- 基线保留 ResNet18、256/256 输入、64×64 热图、连续 DSNT 解码、五折分组、内部选轮和独立校准。分组种子始终为 2026；两组模型随机种子偏移分别为 0 和 10000，预测按 75% / 25% 集成。
- 新 fine 候选仍使用同一 ResNet18，只将监督 Gaussian 的 `HEATMAP_SIGMA` 从 2.5 改为预先提出的 **1.5 个热图格点**。粗模型和其他超参数保持原配置，粗定位、校准 ROI 和图像留出的训练裁剪中心沿用 seed10000 基线。
- 每折按相同规则比较固定/多尺度两种训练裁剪策略，在内部选轮后重训。对照不搜索其他 sigma、学习率或网络结构。
- 仅当内部早停误差相比 seed10000 单模型的内部代理分数至少改善 **0.5px 且 2%**，才加入该折 20% 新候选预测；否则保留该折 8.2 方法的预测。此代理分数不等于 75/25 集成的误差，不能保证混合后必然提升。
- 校准图不用于训练或选轮；测试图不用于训练、调参或模型选择。外层 OOF 用于报告，且数据已参与多轮开发，不能当作全新独立验证。新集成不沿用单模型的覆盖率承诺。

[Modeling Annotation Uncertainty with Gaussian Heatmaps in Landmark Localization](https://www.melba-journal.org/papers/2021:014.html) 指出 Gaussian 宽度影响定位精度与稳健性，为检查热图监督尺度提供依据。这里采用固定单变量对照，并非复现论文的可学习宽度损失；**1.5 是预先给出的实验假设，不是论文证明的最优数值**。坐标期望解码参考 [DSNT](https://arxiv.org/abs/1801.07372)。

代码单元只包含定义、配置和调用，无伪造运行输出。所有函数和类的最终定义与本地 precision engine 一致。
'''

setup_code = '''import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("MPLBACKEND", "Agg")
'''

config_code = 'CFG0 = ImprovedConfig(**' + repr(config) + ''')
torch.set_num_threads(2)
assert torch.cuda.is_available(), "请在 Kaggle Notebook 设置中启用 GPU"
print("GPU:", torch.cuda.get_device_name(0))
print(research_geometry_selfcheck())

# Resolve and verify the sole required official initialization before training.
Path(CFG0.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
CFG0.PRETRAINED_WEIGHTS_PATH = resolve_research_weights(CFG0)
assert research_file_digest(CFG0.PRETRAINED_WEIGHTS_PATH) == RESEARCH_WEIGHT_SHA256
print("ResNet18 SHA256: verified")
print(research_preflight(CFG0))
'''

baseline_code = '''# 40 stages: seed0. Internal split seed remains 2026.
BASE0 = run_research(CFG0)

# 40 stages: a different model/augmentation seed, identical data partitions.
CFG1 = replace(CFG0, MODEL_SEED_OFFSET=10000,
    OUTPUT_DIR="/kaggle/working/baseline_seed10000",
    SUBMISSION_PATH="/kaggle/working/submission_seed10000.csv")
BASE1 = run_research(CFG1)
'''

reference_code = '''# Form the actual reference from these freshly trained predictions.
BASE0_TEST = refinement_read(Path(BASE0["config"].OUTPUT_DIR) / "test_ensemble_B4.csv")
BASE1_TEST = refinement_read(Path(BASE1["config"].OUTPUT_DIR) / "test_ensemble_B4.csv")
BASELINE_PREDICTIONS = refinement_blend(BASE0_TEST, BASE1_TEST, .25)
REFERENCE_CSV = Path("/kaggle/working/submission_retrained_8p2_method.csv")
REFERENCE_SUBMISSION = refinement_submit(
    BASE0["config"].SUBMISSION_PATH, BASELINE_PREDICTIONS, BASE0["test"], REFERENCE_CSV)
print("Re-trained 8.2-method reference:", REFERENCE_CSV)
display(REFERENCE_SUBMISSION.head(6))
'''

refinement_code = '''# 15 new fine stages: one preplanned sigma candidate, identical local protocol.
CFG_REFINEMENT = replace(CFG1, MODEL_SEED_OFFSET=10000,
    OUTPUT_DIR="/kaggle/working/research82_precision_results",
    SUBMISSION_PATH="/kaggle/working/research82_precision_results/submission.csv")
REFINEMENT = run_refinement(
    CFG_REFINEMENT,
    old_run=BASE0["config"].OUTPUT_DIR,
    source_run=BASE1["config"].OUTPUT_DIR,
    reference_csv=REFERENCE_CSV)
'''

export_code = '''# run_refinement writes OUTPUT_DIR/submission.csv; publish that exact file.
import shutil
SOURCE_SUBMISSION = Path(REFINEMENT["config"].OUTPUT_DIR) / "submission.csv"
FINAL_SUBMISSION = Path("/kaggle/working/submission.csv")
shutil.copyfile(SOURCE_SUBMISSION, FINAL_SUBMISSION)
assert research_file_digest(FINAL_SUBMISSION) == research_file_digest(SOURCE_SUBMISSION)
display(REFINEMENT["summary"])
display(pd.read_csv(FINAL_SUBMISSION).head(10))
print("Candidate submission:", FINAL_SUBMISSION)
print("Its Kaggle score is unknown until submitted.")
'''

cells = [
    nbformat.v4.new_markdown_cell(intro),
    nbformat.v4.new_code_cell(setup_code),
    nbformat.v4.new_markdown_cell("## 1. 完整训练与推理定义\n\n下方代码直接来自已固定的本地 engine。"),
    nbformat.v4.new_code_cell(source),
    nbformat.v4.new_markdown_cell("## 2. 可移植配置、GPU 与 ResNet18 权重预检"),
    nbformat.v4.new_code_cell(config_code),
    nbformat.v4.new_markdown_cell("## 3. 从头训练两组五折基线（80 个阶段）"),
    nbformat.v4.new_code_cell(baseline_code),
    nbformat.v4.new_markdown_cell("## 4. 从真实重训结果生成 75/25 参考提交"),
    nbformat.v4.new_code_cell(reference_code),
    nbformat.v4.new_markdown_cell("## 5. 单变量 fine 对照与内部选择（15 个阶段）"),
    nbformat.v4.new_code_cell(refinement_code),
    nbformat.v4.new_markdown_cell("## 6. 导出并查看 submission.csv"),
    nbformat.v4.new_code_cell(export_code),
]
notebook = nbformat.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
    "language_info": {"name": "python"},
    "kaggle": {"isInternetEnabled": False, "isGpuEnabled": True, "language": "python"},
})

def final_definitions(text):
    # Same-name functions/classes intentionally take their LAST definition,
    # as Python does for the appended fine model factory override.
    result = {}
    for node in ast.parse(text).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            result[node.name] = ast.dump(node, include_attributes=False)
    return result

expected = final_definitions(source)
actual = {}
code_cells = []
for cell in cells:
    if cell.cell_type == "code":
        compile(cell.source, "kaggle_cell", "exec")
        actual.update(final_definitions(cell.source))
        code_cells.append(cell.source)
        assert cell.execution_count is None and cell.outputs == []
        assert "C:/Users/" not in cell.source and "C:\\Users\\" not in cell.source
assert actual == expected
assert "MODEL_SEED_OFFSET': 0" in config_code
assert "refinement_effnet_weights" not in config_code
assert "resolve_research_weights(CFG0)" in config_code
assert "RESEARCH_WEIGHT_SHA256" in config_code
assert "run_research(" not in config_code
assert baseline_code.count("run_research(") == 2
assert "refinement_blend(BASE0_TEST, BASE1_TEST, .25)" in reference_code
assert "BASE0[\"test\"]" in reference_code
assert "run_refinement(" in refinement_code
assert "SOURCE_SUBMISSION" in export_code and "shutil.copyfile" in export_code
assert hashlib.sha256(ENGINE.read_bytes()).hexdigest() == engine_sha256
nbformat.validate(notebook)
OUT.mkdir(parents=True, exist_ok=True)
path = OUT / "fovea_research82_precision_kaggle.ipynb"
nbformat.write(notebook, path)
roundtrip = nbformat.read(path, as_version=4)
nbformat.validate(roundtrip)
assert roundtrip.cells[3].source == source
validation = {
    "status": "PASS",
    "notebook": str(path),
    "notebook_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    "engine_sha256": engine_sha256,
    "function_and_class_final_definitions_match_engine": True,
    "matched_definitions": len(expected),
    "same_name_definition_rule": "Last definition wins, exactly as Python execution",
    "cells": len(cells),
    "code_cells": len(code_cells),
    "all_code_cells_compile": True,
    "nbformat_schema_and_roundtrip": "PASS",
    "no_fabricated_cell_outputs": True,
    "no_local_machine_paths_in_code": True,
    "baseline_training_stages": 80,
    "new_fine_training_stages": 15,
    "total_training_stages": 95,
    "only_resnet18_weights_required_and_verified_before_training": True,
    "fixed_single_variant": "resnet18_narrow",
    "fixed_fine_heatmap_sigma": 1.5,
    "reference_csv_built_from_retrained_baseline_test_predictions": True,
    "candidate_copied_from_refinement_output": True,
    "execution_scope": "Static Notebook validation; no extra training was launched by this builder",
}
(OUT / "notebook_validation.json").write_text(
    json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(validation, ensure_ascii=False, indent=2))
