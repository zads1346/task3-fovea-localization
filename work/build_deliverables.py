from pathlib import Path
import ast
import nbformat as nbf

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)
data = (ROOT / "work/data_part.py").read_text(encoding="utf-8")
model = (ROOT / "work/model_training_part.py").read_text(encoding="utf-8")
pipeline = (ROOT / "work/pipeline_part.py").read_text(encoding="utf-8")
checks = '''
def quick_sanity_checks():
    for width, height in [(1001, 777), (603, 1403), (512, 512)]:
        *_, forward = letterbox_geometry(width, height, 512)
        inverse = np.linalg.inv(forward)
        points = np.array([[0, 0, 1], [width - 1, height - 1, 1],
                           [(width - 1) / 2, (height - 1) / 2, 1]], dtype=float)
        restored = (inverse @ (forward @ points.T)).T
        assert np.allclose(restored, points, atol=1e-8)
    assert parse_submission_id("eye__001.jpg_Fovea_X") == ("eye__001", "x")
    assert parse_submission_id("patient left.PNG_Fovea_Y") == ("patient left", "y")
    assert parse_submission_id("0007_Fovea_X") == ("0007", "x")
    target = gaussian_heatmap([0.4, 0.6], 128, 2.5)
    assert np.isfinite(target).all() and np.isclose(target.sum(), 1.0)
    dummy = pd.DataFrame({"x": [20.0, 30.0], "y": [40.0, 50.0],
                          "visibility_prob": [0.8, 0.2]})
    assert np.array_equal(apply_visibility(dummy, 0.5), [[20, 40], [0, 0]])
    print("Geometry, heatmap, ID parsing and invisible-output checks passed.")
'''
script = (
    '# -*- coding: utf-8 -*-\n'
    '# Kaggle 2026task3: complete grouped CV heatmap localization pipeline.\n'
    '# Source and instructions are also available in fovea_kaggle.ipynb.\n'
    + data + "\n\n" + model + "\n\n" + pipeline + "\n\n" + checks
    + '\n\nif __name__ == "__main__":\n'
      '    quick_sanity_checks()\n'
      '    RESULTS = run_pipeline(CFG)\n'
)
ast.parse(script)
(OUT / "fovea_kaggle.py").write_text(script, encoding="utf-8")

configuration, data_functions = data.split("IMAGE_EXTENSIONS =", 1)
data_functions = "IMAGE_EXTENSIONS =" + data_functions
notebook = nbf.v4.new_notebook()
notebook.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.10"},
}
notebook.cells = [
    nbf.v4.new_markdown_cell("""# 中心凹检测定位：Kaggle 完整训练 Notebook

**任务：** `2026task3`，眼底图像预测中心凹坐标；不可见时输出 `(0,0)`。

**修订 1.2：已核对用户提供的真实比赛 ZIP。** 训练标注 ID 为 `1`—`80`，对应 `train/0001.jpg`—`0080.jpg`；测试图片为 `0081.jpg`—`0100.jpg`，提交模板为 `81_Fovea_X/Y` 至 `100_Fovea_X/Y`，共 40 行。本版本支持这种前导零映射。

真实数据中，`0039.jpg` 和 `0077.jpg` 的像素完全相同，标注相距约 **4.073 像素**。代码保留原始两条标注，强制同折，记录警告；允许的重复图标注差异默认上限为 5 像素，更大差异或可见性矛盾仍会报错。全部 80 张训练图均标为可见，因此这份数据不能训练可靠的不可见分类器，程序会统一关闭该分支。

本 Notebook 实现可独立运行的主方案：**ResNet18 + U-Net 类热图定位 + 可见性分支 + 分组交叉验证 + 折模型/TTA 融合**。XML 检测框用于审计，CSV 中心点用于训练。局部精修、YOLO 对照不是该主流程的依赖；应在取得真实交叉验证证据后另做消融。

## 运行方法

1. 在比赛页面新建 Kaggle Notebook，导入此 `.ipynb`。
2. **Add Input / Add Data** 添加比赛 `2026task3`。接受参赛规则、登录等操作请在 Kaggle 页面自行完成。
3. 在 Notebook 设置中启用 GPU（如 T4 或 P100）。本代码使用一张 GPU。
4. 检查下方 `Config`，通常无需修改数据路径；然后 **Run All**。
5. 下载或提交 `/kaggle/working/submission.csv`。本 Notebook 只生成文件，不自动上传。

预训练：默认尝试官方 ResNet18 权重，失败会警告并回退随机初始化。推荐在课程允许的前提下开启 Internet，或添加官方权重文件并设置 `PRETRAINED_WEIGHTS_PATH`；离线时把 `ALLOW_WEIGHT_DOWNLOAD=False`。随机初始化可以跑通，但 80 张图的效果可能更差。

依赖均使用 Kaggle 镜像通常预装的 PyTorch、torchvision、OpenCV、NumPy、pandas、scikit-learn、Pillow、matplotlib，不运行 pip 安装。

**真实数据流程检查已通过：** 使用全部 80 张训练图、20 张测试图，在 CPU 上完成 2 折、每折 2 epoch、128 输入的缩小配置运行，验证了权重保存／重载、TTA、重复图同折、80 条 OOF 预测与 40 行提交格式。此次检查使用随机初始化，不能代表正式成绩。

**本文件不附带正式训练分数或权重。** 正式结果需要在 Kaggle GPU 上完成默认配置训练，保留 `SMOKE_TEST=False`。"""),
    nbf.v4.new_markdown_cell("""## 1. 配置

- 默认 5 折，每折最多 80 epoch，12 epoch 不提升则早停；每折的可训练类别/独立组不足时会调整或报错。
- `BATCH_SIZE=8`；GPU 显存不足时改成 4 或 2，不要首先降低原图坐标精度。
- 先试运行可设 `SMOKE_TEST=True`：2 折、2 epoch、128 输入，输出 `submission_SMOKE_TEST.csv`，**不能当作正式训练结果**。正式运行务必设回 False。
- 自动找不到数据或找到多个副本时，设置 `DATA_DIR="/kaggle/input/实际数据集目录"`，必要时指定 `LABEL_CSV`、`SAMPLE_CSV`。
- `COORDINATE_ORIGIN=0` 是待数据审计核实的默认像素约定；仅在标注文档确认 1-based 时改为 1。
- 若有真实患者分组列，可填写 `GROUP_COLUMN`。没有患者信息时仅能保证精确重复图不跨折。
- `DUPLICATE_LABEL_TOLERANCE_PX=5.0` 允许真实数据中约 4.073 像素的重复图标注差异，不修改或平均原标签。
- `REUSE_CHECKPOINTS=True` 可复用已完成的折；校验数据和配置签名。它不是从中断 epoch 恢复优化器。
- 默认不使用 OOF 标签搜索可见性阈值。如开启 `TUNE_VISIBILITY_THRESHOLD`，调参后分数会明确标为乐观估计。"""),
    nbf.v4.new_code_cell(configuration.rstrip()),
    nbf.v4.new_markdown_cell("""## 2. 数据读取、审计、划分与增强

训练图仅从标注 CSV 的 `data` 解析；测试图仅从 sample 的 `ImageID` 解析。保留原始提交 ID。图像匹配先排除相反 split，再按精确 stem、同前后缀的数字编号、纯数字标签对应的唯一单数字块文件名依次查找。禁止按排序位置、任意子串或多数字文件名的尾数猜测。

程序会先打印实际图片路径，再打印训练／测试 ID 对应文件名。多路径同名、缺图、多个编号候选、多个 ID 映射同一图片、重复坐标项、非有限标签、越界坐标均显式报错。

中心点 `(0,0)` 是不可见状态；只有 X 和 Y **同时**为零才判定不可见。图像等比例缩放与填充使用实际像素中心变换，增强与点使用同一个仿射矩阵。训练增强不会把出界目标伪造为不可见标签。"""),
    nbf.v4.new_code_cell(data_functions.rstrip()),
    nbf.v4.new_markdown_cell("""## 3. 模型、损失、训练与推理

- 编码器：ResNet18；解码器：带跳接的轻量 U-Net 类结构；小批量训练冻结编码器 BatchNorm 的运行统计。
- 点定位：归一化高斯目标 + KL 热图损失 + 原图像素误差。空间 softmax 的期望产生连续坐标。
- 可见性：独立 BCE 分支；不可见图的热图/坐标损失被屏蔽。如果没有足够不可见样本支持每个训练折，统一禁用 gate 并报告限制。
- 优化：AdamW、warmup/cosine、CUDA AMP、梯度裁剪、早停，保存并重载最优 checkpoint。
- RMSE：按原始提交坐标，把 X/Y 共 `2N` 个标量一起计算，而非分别计算两个 RMSE 再平均。
- TTA：水平翻转后先反变换到原图，再融合。折模型融合后才应用可见性阈值，避免将零点与正常坐标混合。

热图多峰可能让期望落在两峰之间，应通过 OOF 误差图检查。本 Notebook 不把热图峰值冒充可见性概率。"""),
    nbf.v4.new_code_cell(model.rstrip()),
    nbf.v4.new_markdown_cell("""## 4. 完整交叉验证与提交

每张训练图只由没有见过该图所属组的模型生成 OOF 预测。最佳 epoch 根据验证折选择，因此 OOF 是开发期交叉验证估计，不等于完全独立的最终测试结果。公开榜数据很少，避免据其反复调参。

输出包括：

| 文件 | 内容 |
|---|---|
| `/kaggle/working/submission.csv` | 可上传的正式预测，沿用 sample 的原始顺序 |
| `fovea_run/folds.csv` | 标签、图像路径、重复图分组和折分配 |
| `fovea_run/filename_mapping.csv` | 训练／测试表中的 ID 与实际图像路径映射 |
| `fovea_run/duplicate_label_audit.csv` | 重复图片、标注距离及保留原标签／同折策略 |
| `fovea_run/labels_*.png` | 全部训练标签叠加图；绿色点、黄色 XML 框 |
| `fovea_run/xml_box_audit.csv` | XML 框与 CSV 点的关系；不替代点标签 |
| `fovea_run/fold_*_best.pt` | 各折最优模型参数与元数据 |
| `fovea_run/fold_*_history.csv` | 各折训练记录 |
| `fovea_run/oof_predictions.csv` | 每张训练图的折外预测 |
| `fovea_run/oof_worst_cases.png` | 最差 OOF 样本，绿色真值、红色预测 |
| `fovea_run/scores.json` | 总体/分折 RMSE、可见性错误及阈值 |
| `fovea_run/test_ensemble_raw.csv` | 应用可见性判断前的测试预测 |
| `fovea_run/run_config.json`、`environment.json` | 配置与运行环境 |

不手动修正测试坐标，不使用人工测试标注。比赛规则和预训练/外部数据的课程要求请以主办方当前说明为准。"""),
    nbf.v4.new_code_cell(pipeline.rstrip()),
    nbf.v4.new_markdown_cell("""## 5. 运行前自检"""),
    nbf.v4.new_code_cell(checks.strip() + "\n\nquick_sanity_checks()"),
    nbf.v4.new_markdown_cell("""## 6. 开始训练并生成 submission.csv

这一格执行完整流程。开启 GPU 后运行；总耗时取决于硬件、早停和图像读取速度。输出中的 `*` 表示该 epoch 刷新了当前折最佳验证分数。"""),
    nbf.v4.new_code_cell("RESULTS = run_pipeline(CFG)"),
    nbf.v4.new_markdown_cell("""## 7. 查看结果

先检查 OOF 误差和标签叠加图，再决定下一轮实验。阈值、TTA 或超参数调优应如实记录；不能把本地合成自检当作真实比赛精度。"""),
    nbf.v4.new_code_cell('print("Submission:", RESULTS["config"].SUBMISSION_PATH)\n'
                         'print("OOF RMSE:", RESULTS["scores"]["oof_rmse_fixed_threshold"])\n'
                         'RESULTS["submission"].head(10)'),
    nbf.v4.new_markdown_cell("""资料：

- [比赛数据](https://www.kaggle.com/competitions/2026task3/data)
- [任务和评分](https://www.kaggle.com/competitions/2026task3/overview)
- [比赛规则](https://www.kaggle.com/competitions/2026task3/rules)
- [U-Net](https://arxiv.org/abs/1505.04597)
- [DSNT](https://arxiv.org/abs/1801.07372)
"""),
]
nbf.validate(notebook)
for cell in notebook.cells:
    if cell.cell_type == "code":
        ast.parse(cell.source)
nbf.write(notebook, OUT / "fovea_kaggle.ipynb")
print(f"Built {len(notebook.cells)} notebook cells and matching Python script.")
