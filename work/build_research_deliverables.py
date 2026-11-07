"""Build a self-contained Kaggle notebook and identical Python entry point."""
from pathlib import Path
import ast
import hashlib
import json
import zipfile

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)
read = lambda name: (ROOT / "work" / name).read_text(encoding="utf-8")
data = read("data_part.py")
base_config, data_functions = data.split("IMAGE_EXTENSIONS =", 1)
data_functions = "IMAGE_EXTENSIONS =" + data_functions
config = read("research_config_part.py")
weights = read("research_weights_part.py")
model = read("model_training_part.py")
research_model = read("research_model_part.py")
calibration = read("research_calibration_part.py")
audit = read("pipeline_part.py").split("def run_pipeline(config):", 1)[0]
pipeline = read("research_pipeline_part.py")
parts = [base_config, config, weights, data_functions, model, research_model, calibration, audit, pipeline]
source_hash = hashlib.sha256("\n".join(parts).encode()).hexdigest()
stamp = f'RESEARCH_SOURCE_HASH = "{source_hash}"\n'
script = "# -*- coding: utf-8 -*-\n" + "\n\n".join(parts) + "\n\n" + stamp
script += '\nif __name__ == "__main__":\n    print(research_geometry_selfcheck())\n    RESULTS = run_research(CFG)\n'
ast.parse(script)
(OUT / "fovea_research_kaggle.py").write_text(script, encoding="utf-8")

def markdown(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}

def code(text):
    ast.parse(text)
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": text.splitlines(keepends=True)}

intro = """# 2026task3：粗到细中心凹定位与不确定性校准

完整、可独立运行的 Kaggle 科研 Notebook。包括真实文件名映射、重复图分组、五折外层交叉验证、折内早停与独立校准、粗到细模型、6组消融、TTA、误差与覆盖率报告、submission.csv。

## 直接运行
1. 导入本 `.ipynb`，添加比赛 `2026task3` 数据。
2. 把附带的 **fovea_pretrained_weights.zip** 上传成自己的 Kaggle Dataset，再通过 **Add Input** 挂到 Notebook。代码同时支持 Kaggle 自动解压后的 `.pth` 和保留原 ZIP 的布局，无需改写其目录名。
3. 设置 GPU 加速器。权重已打包，**Internet 可以关闭**。代码不安装依赖，也不访问其他模型网站。
4. 保持 `SMOKE_TEST=False`，运行全部单元格。
5. 正式提交文件为 `/kaggle/working/submission.csv`；默认选择预先指定的 B4，不根据外层验证分数自动挑方法。

权重是 torchvision 官方 **ResNet18 ImageNet-1K** 初始化，不是已经训练好的中心凹模型。文件经过完整 SHA256 校验；找不到或损坏时明确报错，不悄悄改用随机初始化。

本地测试仅用于验证程序。Notebook 内没有伪造训练结果、比赛分数或预填输出。
"""
configuration_notes = """## 1. 配置与依赖

通常只需修改下方 `ResearchConfig`：
- 正式运行默认外层5折，粗模型最多50轮、细模型最多40轮，内部验证8轮不改善则停止。
- 默认每折7次模型训练：粗模型轮数选择、3个训练裁剪的交叉拟合模型、细模型轮数选择，以及最终粗/细模型。五折共35次；所有消融共享模型，无需6倍重复训练。
- `BATCH_SIZE=8`；显存不足时改4或2。单张GPU即可；不要求双卡。
- `SMOKE_TEST=True` 仅做低分辨率、每阶段1轮的流程检查，输出 `submission_SMOKE_TEST.csv`。正式训练必须改回False。
- `REUSE_CHECKPOINTS=True` 只复用数据/代码/配置一致的**已完成外层折**。中断的折会从该折开头重跑；不是优化器断点续训。新会话须保留旧输出目录才可复用。
- `PRETRAINED_WEIGHTS_PATH=None` 自动查找。也可填 `/kaggle/input/实际权重数据集/resnet18-f37072fd.pth` 或ZIP路径。
- `ALLOW_WEIGHT_DOWNLOAD=False` 默认离线；已有离线包无需开启下载。
- `FINE_TRAIN_CROPS="oof"` 使用未训练过目标图像的粗模型预测中心。`"jitter"`是明确不同的廉价训练设置，不可冒充OOF方案。
- 外层验证只评价。校准组不参与两个最终模型的训练或早停，不能在校准后把它们加入模型重训而沿用旧校准参数。

依赖：Kaggle通常预装的torch、torchvision、numpy、pandas、opencv、Pillow、scikit-learn、matplotlib。建议先运行最后的“预检”单元格确认数据与权重，再开始完整实验。
"""
protocol = """## 4. 校准与消融的含义

每个外层训练折按组保留25%作校准集；其余开发集再分拟合/早停子集。选择训练轮数后，最终网络只在开发集重训。外层验证集完全不参与训练、早停、半径校准或回退阈值选择。

细模型训练中心：拟合子集内部3折粗预测 + 粗模型对早停子集的预测。每个中心由没有训练过该图像组的网络生成；这些模型训练集大小不同。用于产生训练裁剪的轮数在开发集内选择，训练裁剪预测不作为独立性能估计。训练中漏裁会扩窗并记录，验证/测试绝不借真值修复裁剪。

不确定性定义为原图坐标下热图协方差最大特征值的平方根。它是空间分散程度，不天然等于可信概率，也不自动区分数据噪声与模型不确定性。校准使用独立校准组的 `max(|dx|,|dy|)/uncertainty` 残差分位数，并按重复/患者组取最大值。这里实现的是**经验误差半径校准**，不是温度缩放，也不声称conformal理论覆盖保证。

| 方法 | 内容 |
|---|---|
| B0 | 单阶段粗定位 |
| B1 | 固定ROI粗到细 |
| B2 | 未校准不确定性ROI |
| B3 | 校准不确定性ROI |
| B4 | B3加回退到粗预测 |
| C1 | 按校准集平均裁剪面积匹配的固定ROI对照 |

B1/B2/B3/B4/C1共享同一个精修网络，主要检验推理裁剪与回退策略；不代表分别针对每种策略训练至最优的6个独立模型。C1匹配平均原图裁剪面积，不声称精确匹配所有硬件耗时。回退阈值由校准集拟合，可根据图像裁剪边缘距离和粗细预测分歧触发；所有样本均保留在最终评价和提交中。

此版本未实现自适应高斯新监督、视盘/血管多任务或额外数据半监督；没有这些额外标注，不能把它们写成已完成实验。
"""
outputs_notes = """## 6. 输出与评价

`/kaggle/working/fovea_research/` 保存：
- `folds.csv`、每折 `split_roles.csv`/`split_audit.json`：外层划分和拟合/早停/校准角色。
- `fold_*/crop_crossfit/crossfit_audit.json`：训练裁剪的逐组交叉拟合记录。
- `fold_*/coarse_final/model.pt`、`fine_final/model.pt`：最终两阶段模型；其余stage目录保存轮数选择/交叉拟合模型和日志。
- `radius_calibration.json`、`fallback_rule.json`、`area_matched_control.json`：每折独立校准参数。
- `oof_B0.csv`…`oof_C1.csv`、`ablation_summary.csv`：每张训练图只出现一次的外层预测和对比。
- `paired_comparisons.csv`：按已知图像组的配对bootstrap区间；不是患者独立性证明。
- 不确定性诊断CSV/JSON/PNG：名义50/80/90/95%覆盖与实际外层覆盖、风险—覆盖率。
- `submission_B0.csv`…`submission_C1.csv`：各方法五折平均预测；顶层 `submission.csv`复制预先配置的方法。
- `run_config.json`、`environment.json`、`run_summary.json`：可追踪配置、环境与完成状态。

比赛RMSE把原始像素坐标X/Y共2N个数一起计算；另报欧氏误差中位数/P90及ROI覆盖率。预测CSV中的 `inference_seconds` 仅计当前网络阶段（含读图）的每图平均时间，精修阶段的值不等于完整两阶段端到端耗时。校准集拟合分数不能当作外层成绩。跨域评价和小样本学习曲线需要额外独立实验，本Notebook不伪造这些结果。

真实数据有80训练图/20测试图，训练39与77为重复图但原标签相差约4.073像素，保留原标签且强制同组。全部训练图可见，本科研版本不训练不可见分类器。XML仅用于标签审计。没有患者ID时只能保证已知重复图不跨折。
"""
preflight = """print(research_geometry_selfcheck())
if not CFG.SMOKE_TEST and CFG.REQUIRE_GPU and not torch.cuda.is_available():
    raise RuntimeError("请在Kaggle设置中启用GPU；快速检查可设SMOKE_TEST=True。")
print("Device:", "cuda" if torch.cuda.is_available() else "cpu")
print("Verified weights:", resolve_research_weights(CFG))
print("正式外层折数:", CFG.N_FOLDS, "；提交方法:", CFG.SUBMISSION_EXPERIMENT)
"""
cells = [markdown(intro), markdown(configuration_notes), code(base_config + "\n\n" + config),
         markdown("## 2. 离线权重查找与完整校验"), code(weights),
         markdown("## 3. 数据读取与可追溯的坐标变换"), code(data_functions),
         code(model), code(research_model), markdown(protocol), code(calibration),
         markdown("## 5. 外层五折实验执行器"), code(audit), code(pipeline + "\n\n" + stamp),
         markdown(outputs_notes), markdown("## 7. 运行前自检"), code(preflight),
         markdown("## 8. 开始训练\n\n此单元格执行全部折和消融。低分辨率冒烟检查只能验证流程，不能代表正式定位精度。"),
         code("RESULTS = run_research(CFG)"),
         markdown("## 9. 查看结果"),
         code('print("Submission:", RESULTS["config"].SUBMISSION_PATH)\ndisplay(RESULTS["summary"])\ndisplay(RESULTS["submission"].head(10))'),
         markdown("""## 方法与资源来源
- [官方ResNet18文档](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html)
- [GAMMA挑战原文](https://arxiv.org/html/2202.06511v3)
- [DSNT](https://arxiv.org/abs/1801.07372)
- [热图定位不确定性评估](https://arxiv.org/html/2203.02351v2)
- [Contour-Hugging Heatmaps](https://openaccess.thecvf.com/content/CVPR2022/html/McCouat_Contour-Hugging_Heatmaps_for_Landmark_Detection_CVPR_2022_paper.html)

本Notebook是针对该小样本任务的可复现实验实现，不是上述论文的逐项官方复现，也不声称组合模块本身具有原创性。
""")]
for i, cell in enumerate(cells):
    cell["id"] = f"fovea-research-{i:02d}"
nb = {"nbformat": 4, "nbformat_minor": 5, "metadata": {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.10"}}, "cells": cells}
(OUT / "fovea_research_kaggle.ipynb").write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")

readme = """2026task3 Kaggle科研实验包 v2.0

1. 导入 fovea_research_kaggle.ipynb 到 Kaggle Notebook。
2. Add Input 添加2026task3比赛数据。
3. 将 fovea_pretrained_weights.zip 上传为一个Kaggle Dataset，挂载到Notebook。
   ZIP自动解压或保留ZIP均可，程序会自动查找并校验。
4. 启用GPU，Internet可关闭，保持SMOKE_TEST=False，Run All。
5. /kaggle/working/submission.csv 为正式运行输出，默认B4。

文件：
  fovea_research_kaggle.ipynb  完整Notebook，代码全部内嵌
  fovea_research_kaggle.py     同等功能Python脚本
  fovea_pretrained_weights.zip  官方ResNet18 ImageNet-1K权重及来源清单
  research_validation.json   本地实际验收范围（若包含），不是正式比赛成绩

不需要同时把.py上传到Kaggle；Notebook可独立运行。
权重是ImageNet初始化，不是训练完成的中心凹预测模型。
默认5个外层折，每折7个训练任务，共35次；消融共享粗/细模型。
每折校准数据始终不参与最终网络训练；不能使用测试标签调整坐标。
SMOKE_TEST只做低分辨率1轮流程检查，输出submission_SMOKE_TEST.csv。
完整正式五折训练需由Kaggle GPU执行；本地验收不代表精度。
仅支持当前80张均可见的训练数据；不存在可靠的不可见类学习证据。
复用只覆盖已完成的外层折，不是中断epoch/优化器恢复。
详细方法、假设、消融和输出清单见Notebook说明。
"""
(OUT / "README_research.txt").write_text(readme, encoding="utf-8")
bundle_files = ["fovea_research_kaggle.ipynb", "fovea_research_kaggle.py", "README_research.txt", "fovea_pretrained_weights.zip"]
if (OUT / "research_validation.json").exists():
    bundle_files.append("research_validation.json")
with zipfile.ZipFile(OUT / "fovea_research_complete.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
    for name in bundle_files:
        archive.write(OUT / name, name)
print(json.dumps({"cells": len(cells), "script_bytes": len(script.encode()), "source_sha256": source_hash,
                  "bundle": str(OUT / "fovea_research_complete.zip")}, indent=2))
