# 8.2 基线之后：关键点定位文献与可实施改进

检索日期：2026-10-27。此文是研究与代码审查笔记，不含新训练结果。没有论文证明某一种算法在当前 80 张眼底训练图上必然最优。

检索遵循 agent-reach 路由；本机 PATH 未发现 mcporter，Jina 对 CVF 的读取在 15 秒后超时，随后使用官方 CVF、arXiv、NeurIPS 和期刊原文页面。没有采用博客结论替代原论文。

## 论文核实与任务适配

| 论文 | 核心证据 | 对本任务的判断 |
|---|---|---|
| [Nibali 等，Numerical Coordinate Regression with Convolutional Neural Networks，2018](https://arxiv.org/abs/1801.07372) | DSNT 以可微期望输出连续坐标，也能使用低分辨率热图。 | 现有方法已采用这条路线；低热图分辨率本身不等于坐标被量化。 |
| [Sun 等，Deep High-Resolution Representation Learning for Human Pose Estimation，CVPR 2019](https://openaccess.thecvf.com/content_CVPR_2019/html/Sun_Deep_High-Resolution_Representation_Learning_for_Human_Pose_Estimation_CVPR_2019_paper.html) | HRNet 在网络中保留高分辨率分支并反复融合多尺度信息；实验证据是人体关键点。 | 支持增加可学习的高分辨率细节路径，但不证明整套 HRNet 在 80 张眼底图上胜出。 |
| [Ronneberger 等，U-Net，MICCAI 2015](https://arxiv.org/abs/1505.04597) | 收缩路径提取上下文，扩张路径与细节结合，实现医学图像定位；论文重视少量图像与增强。 | 可复用现有 ResNet18 stem 特征，给 fine 增加一级学习式解码。这里是自定义改进，不是 HRNet 复现。 |
| [Wang 等，Adaptive Wing Loss，ICCV 2019](https://openaccess.thecvf.com/content_ICCV_2019/html/Wang_Adaptive_Wing_Loss_for_Robust_Face_Alignment_via_Heatmap_Regression_ICCV_2019_paper.html) | 按目标热图值调整损失，并提高前景/困难背景权重；主要证据来自人脸关键点。 | 原损失针对热图幅值，不可不改尺度地替换当前概率 KL；还涉及训练、解码和不确定性一致性。 |
| [Zhang 等，DARK，CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/html/Zhang_Distribution-Aware_Coordinate_Representation_for_Human_Pose_Estimation_CVPR_2020_paper.html) | 改进关键点热图编码与分布感知解码；人体关键点实验支持其作用。 | 局部峰值解码与当前概率均值不同；对多峰热图、低置信度和边缘峰值须单独评估，不能直接沿用旧协方差校准。 |
| [Huang 等，UDP，CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/html/Huang_The_Devil_Is_in_the_Details_Delving_Into_Unbiased_Data_CVPR_2020_paper.html) | 指出翻转、变换和编码解码中的离散坐标偏差，强调基于像素间距的连续空间处理。 | 当前代码已用 extent=size−1、连续标签和逆变换处理，核心问题已部分解决；无须把“增加 UDP”视作全新必然增益。 |
| [Payer 等，SpatialConfiguration-Net，2019](https://arxiv.org/abs/1908.00748) | 将局部外观与多个解剖关键点的空间关系分开建模，适用于有限标注。 | 本任务只有一个点，没有视盘等额外标注，不能照搬多关键点空间约束收益。 |
| [Thaler 等，Modeling Annotation Uncertainty with Gaussian Heatmaps in Landmark Localization，MELBA 2021](https://www.melba-journal.org/papers/2021:014.html) | 研究 Gaussian 宽度/方向与标注不确定性；DOI：10.59275/j.melba.2021-77a7。 | 支持把 sigma 当作关键参数。论文的未归一化热图 MSE 与宽度正则不等同于本项目的概率 KL，不能直接照抄。 |
| [Gu 等，Bias-Compensated Integral Regression，2023](https://arxiv.org/abs/2301.10431) | 分析 softmax 与期望组合的偏差，并以偏差补偿与 Gaussian 先验改进人体/手部定位。 | 提醒均值解码存在条件性偏差；当前已经有 Gaussian KL，不宜只按论文标题添加“去偏”公式。 |
| [Hu 等，NerPE，NeurIPS 2024](https://papers.nips.cc/paper_files/paper/2024/hash/b90cb10d4dae058dd167388e76168c1b-Abstract-Conference.html) | 连续隐式热图与渐进坐标解码，可在推理时查询任意分辨率。 | 比简单上采样更完整，但增加训练/解码体系改造；人体基准证据不能直接推断本任务收益。 |
| [Yang、Yeh，Heatmap Regression without Soft-Argmax，ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Yang_Heatmap_Regression_without_Soft-Argmax_for_Facial_Landmark_Detection_ICCV_2025_paper.html) | 采用结构化预测目标和图像相关标签平滑；在人脸基准报告更快收敛。 | 是值得后续研究的损失路线，但会同时改变目标、平滑和解码，暂不与本轮架构升级混合。 |

## 当前代码审查：哪些问题真实存在

这些是对本地 `work/restore_20261027/engine.py` 的分析，不是论文实验结果。

1. `gaussian_heatmap` 使用浮点坐标乘 `(size−1)`，没有先取整；`heatmap_to_coords` 对 spatial softmax 求期望。当前已经支持亚像素输出。
2. fine 的 256 输入经 ResNet stem 得到 128×128×64，但现有解码只恢复到 stride-4 的 64×64；把 `FINE_HEATMAP_SIZE` 单独设成 128 仅会对 logits 做双线性插值，不产生新的学习式细节。
3. `HEATMAP_SIGMA=2.5` 的单位是热图格点。忽略小的像素中心差异，64 热图下 300px crop 对应约 11.9px 原图 sigma，600px crop 对应约 23.8px。这与裁剪尺度有关，并非固定的标注误差。
4. 若使用归一化 Gaussian，sigma=2.5 时中心峰值约为 `1/(2πσ²)=0.0255`。把强调幅值前景/背景的 AWing 常规参数直接施加在这些概率值上，会改变原设计的损失作用；这是数学尺度分析。

## 推荐的两个受控候选

### A：只增加 fine 的真实高分辨率解码

- 保留现有 256 输入、ResNet18 ImageNet 编码器、粗定位及数据划分。
- 新增 `decoder0 = DecoderBlock(64, 64, 32)`，将 decoder1 的 64² 特征上采样后与 ResNet stem 的 128² 特征连接，再用 1×1 卷积输出 128² logits。
- 沿用 KL + DSNT 坐标损失、连续几何变换与 TTA。
- 首次仅检验学习式高分辨率路径：fine sigma 使用 `2.5×127/63≈5.04`，使归一化坐标中的监督宽度保持不变。不要同时改小 sigma 并归因于分辨率。
- 这是受 U-Net/HRNet 原则启发的轻量方案，不能写成完整 HRNet。额外参数和输入分辨率都受控，但需要实际测显存、梯度和五折结果。

### B：独立检验 fine 的监督宽度

- 保留 64² fine 头、粗定位 sigma=2.5；fine-only 比较原 2.5 与一个预先固定的较窄值，例如 1.5。
- 或将原图 10–12px 的 sigma 通过每个样本的实际仿射矩阵换算到热图空间；这比固定格点 sigma 改动更大，应另列实验。
- sigma 不能无限缩小；低于约 0.5 格点时，离散采样后的 Gaussian 会趋向单格点，使其离散期望偏向最近网格。原图尺度方案应设合理下限并检查实际目标均值偏差。
- 学习 sigma 也要防退化，并须重建适合概率 KL 的目标；不能照搬未归一化 MSE 的正则。

架构候选与 sigma 候选应分开验证，避免一次改多个因素。已有粗网络和交叉预测中心只有在训练图、分组、预处理、粗网络参数全一致时才可复用；fine 改变后，回退规则要在原独立校准集重算。融合比例应在实验前固定或仅在内部校准数据确定，不能从测试答案/逐张预测手工调整。

## 结论

更适合当前约束的首轮实验，是保留已成功的粗到细、ImageNet CNN 与集成框架，仅提高 fine 的可学习空间细节；监督宽度作为另一项单独对照。没有足够证据宣称 AWing、HRNet、DARK 或 2025 新方法必然优于 8.2，更不能把人体/人脸基准排名当作本竞赛排名。
