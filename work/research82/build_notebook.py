"""Build a portable, offline Kaggle Notebook without touching training sources."""
from pathlib import Path
import ast
import hashlib
import json
import nbformat


ROOT = Path(__file__).resolve().parents[2]
WORK = Path(__file__).resolve().parent
OUT = ROOT / "outputs/research82_20261027"
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

intro = '''# 2026task3：保留 8.2 方法，比较两种 fine 定位升级

本 Notebook 是完整、可离线运行的 Kaggle 训练流程。用户反馈上一份 75/25 随机种子集成的线上成绩为 **8.2**（越低越好）；该成绩未由本文件从 Kaggle 读取，当前改进候选的线上分数未知。

## 直接运行

1. 添加本比赛数据。
2. 把交付的 `research82_pretrained_weights.zip` 上传成 Kaggle Dataset，并通过 Add Input 添加。包中应同时包含官方 ResNet18 和 EfficientNet-B0 的 ImageNet 权重。
3. 启用 GPU。Internet 可以关闭。代码兼容 Kaggle 自动解压的 `.pth` 文件和保留 ZIP 的布局，先对两份权重进行完整 SHA256 校验，找不到或损坏会报错。
4. Run All。最终候选输出为 `/kaggle/working/submission.csv`。

**首次完整运行共 110 个训练阶段**：两组 ResNet18 基线各五折，共 80 个阶段；两种 fine 改进各五折，每折两种裁剪策略选轮与最终重训，共 30 个新阶段。此 Notebook 不依赖本机目录或已训练的中心凹权重。训练耗时取决于 GPU 和早停轮数。

本地本轮实验复用了已完成的两组基线，只新训练 30 个阶段；Notebook 为可从零运行而重新训练全部 110 个阶段。不同 GPU/库版本可能改变数值。要使用**原样达到 8.2 的预测**，请使用另行交付的 `submission_baseline_8p2.csv`，不要把这里重训的 CSV 当作逐位复现承诺。

## 方案与选择规则

- 基线：ResNet18、256/256 输入、连续热图 + DSNT、五折分组、独立校准、内部选轮，模型/增强随机种子偏移分别为 0 和 10000；数据分组种子始终为 2026。
- 重建的基线预测为 seed0 的 75% + seed10000 的 25%。
- 新候选只强化 fine：EfficientNet-B0 编码器，或者带 ResNet stem skip 的真实 128×128 学习式高分辨率解码头；粗定位与图像留出的训练裁剪中心沿用 seed10000 基线。
- 高分辨率 fine 的 Gaussian sigma 按 127/63 放大，保持监督在归一化坐标中的宽度；这不是完整 HRNet 的复现。
- 每折仅根据内部早停分数选择 fine 结构与裁剪策略。相对 **seed10000 单个 ResNet18 的内部代理分数**至少改善 0.5px 且 2% 时，才加入 20% 新 fine 预测；否则保留该折的基线预测。代理分数不等于 8.2 集成的内部误差。
- 校准图不进入模型训练或选轮，测试图不用于训练。外层 OOF 仅作报告；同一批数据已经用于多轮开发，不能视为全新独立验证。新集成不沿用单模型覆盖率承诺。

连续坐标解码参考 [DSNT](https://arxiv.org/abs/1801.07372)；细节路径的研究动机参考 [U-Net](https://arxiv.org/abs/1505.04597) 与 [HRNet](https://openaccess.thecvf.com/content_CVPR_2019/html/Sun_Deep_High-Resolution_Representation_Learning_for_Human_Pose_Estimation_CVPR_2019_paper.html)。这些论文不证明某个方法在当前 80 张训练图上必然最优。

代码单元不含伪造执行输出。所有模型、函数、类定义与本地训练 engine 的最终定义一致。
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

# Resolve and verify BOTH official initializations before any baseline training.
Path(CFG0.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
CFG0.PRETRAINED_WEIGHTS_PATH = resolve_research_weights(CFG0)
CFG0.EFFNET_WEIGHTS_PATH = refinement_effnet_weights(CFG0)
assert research_file_digest(CFG0.PRETRAINED_WEIGHTS_PATH) == RESEARCH_WEIGHT_SHA256
assert research_file_digest(CFG0.EFFNET_WEIGHTS_PATH) == EFFNET_SHA256
print("ResNet18 SHA256: verified")
print("EfficientNet-B0 SHA256: verified")
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

refinement_code = '''# 30 new fine stages, with the same configuration used for the local experiment.
CFG_REFINEMENT = replace(CFG1, MODEL_SEED_OFFSET=10000,
    OUTPUT_DIR="/kaggle/working/research82_results",
    SUBMISSION_PATH="/kaggle/working/research82_results/submission.csv")
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
    nbformat.v4.new_markdown_cell("## 2. 可移植配置、GPU 与两份权重预检"),
    nbformat.v4.new_code_cell(config_code),
    nbformat.v4.new_markdown_cell("## 3. 从头训练两组五折基线（80 个阶段）"),
    nbformat.v4.new_code_cell(baseline_code),
    nbformat.v4.new_markdown_cell("## 4. 从真实重训结果生成 75/25 参考提交"),
    nbformat.v4.new_code_cell(reference_code),
    nbformat.v4.new_markdown_cell("## 5. 两种 fine 改进与内部选择（30 个阶段）"),
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
assert config_code.index("refinement_effnet_weights") < len(config_code)
assert "run_research(" not in config_code
assert baseline_code.count("run_research(") == 2
assert "refinement_blend(BASE0_TEST, BASE1_TEST, .25)" in reference_code
assert "BASE0[\"test\"]" in reference_code
assert "run_refinement(" in refinement_code
assert "SOURCE_SUBMISSION" in export_code and "shutil.copyfile" in export_code
assert hashlib.sha256(ENGINE.read_bytes()).hexdigest() == engine_sha256
nbformat.validate(notebook)
OUT.mkdir(parents=True, exist_ok=True)
path = OUT / "fovea_research82_kaggle.ipynb"
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
    "new_fine_training_stages": 30,
    "total_training_stages": 110,
    "both_weights_verified_before_training": True,
    "reference_csv_built_from_retrained_baseline_test_predictions": True,
    "candidate_copied_from_refinement_output": True,
    "execution_scope": "Static Notebook validation; no extra training was launched by this builder",
}
(OUT / "notebook_validation.json").write_text(
    json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(validation, ensure_ascii=False, indent=2))
