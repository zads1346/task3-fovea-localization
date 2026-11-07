# 8.2 基线之后：论文驱动的细定位对照实验

用户报告上一份候选在 Kaggle 达到 8.2，越低越好。本轮没有读取新的线上成绩，也没有自动上传提交。
已经备份该版本的 20 个最终模型、CSV、源代码与核验资料：`backups/baseline_8p2_before_research_20261027.zip`。202 个文件逐项 SHA256 核验通过；旧版未被改写。

## 研究结论与实际方法

原始论文最支持的适配路线是预训练 CNN、空间热图、粗到细精修和模型集成，不能据此宣布一种算法对本数据“普适最好”。GAMMA、ADAM 的 EfficientNet 路线还结合了不同监督与外部数据，不能把冠军全部收益归因于编码器。[GAMMA](https://doi.org/10.1016/j.media.2023.102938)、[ADAM](https://doi.org/10.1109/TMI.2022.3172773)。

本轮只升级 fine，分别检验 ImageNet EfficientNet-B0 + U-Net 式解码，以及保留 ResNet18、增加与 stem 相连的可学习 stride-2 解码层。后者输出真实 128×128 热图，输入仍为 256×256；它是高分辨率表示原则启发的轻量改动，不是完整 HRNet 复现。[EfficientNet](https://arxiv.org/abs/1905.11946)、[HRNet](https://openaccess.thecvf.com/content_CVPR_2019/html/Sun_Deep_High-Resolution_Representation_Learning_for_Human_Pose_Estimation_CVPR_2019_paper.html)。

两个候选保留旧版 KL + DSNT 和坐标损失、TTA、分组、内部选轮、校准隔离与ROI参数。高分辨率分支 sigma=2.5×127/63≈5.04，保持目标在原图中的物理宽度；没有同时收窄监督分布。官方权重仅来自 ImageNet，没有引入外部眼底标签或其他竞赛已训练的定位权重。

详细论文核查、最新研究限制及可复用设计见 `research_plan.md`、`fovea_literature.md`、`keypoint_literature.md`。DARK、UDP、Adaptive Wing 与当前归一化概率热图/DSNT 的适用差异在笔记中单独说明，未直接硬套。

## 实验与选择协议

保留 80 个训练样本、79 个精确重复组，外层五折；同组不跨折。继承核验过的 seed10000 粗模型与图像级交叉预测裁剪中心。两个 fine 候选各自选择 fixed/multiscale 策略、选择轮数并重训，共 30 个真实新训练阶段。

每折只用内部 stopping 集选择候选；最低门槛在训练前固定为相对现存 seed10000 单 ResNet18 内部分数至少降低 2% 且 0.5px。注意它是单分支代理分数，**不是8.2集成的内部验证分数**。内部样本仅9–10张，这个门槛是启发式，仍存在选择噪声。

逐折计算 B_k=0.75×旧种子预测+0.25×seed10000预测。门槛通过则 P_k=0.8×B_k+0.2×新分支；不通过则 P_k=B_k。测试集对五个 P_k 等权平均，不只平均通过门槛的折。门槛和结构在外层/测试评估前写入记录；没有按外层分数改门槛或融合权重。

## 真实运行结果

GPU：NVIDIA GeForce RTX 4060 Laptop GPU，正式新增训练与折内推理耗时 15.29 分钟，峰值 CUDA 内存 738.60MB。

| 方法 | OOF 坐标 RMSE | 中位欧氏误差 | P90 欧氏误差 |
|---|---:|---:|---:|
| 8.2 基线 | 28.7624 | 13.1059 | 52.2468 |
| EfficientNet-B0 局部精修分支 | 29.6650 | 9.8885 | 43.7966 |
| ResNet18 学习式 128 热图分支 | 31.8895 | 16.3465 | 59.4200 |
| 按内部门槛融合的提交候选 | 28.7624 | 13.1059 | 52.2468 |

两条单独分支行是全部五折的新架构模型表现，提交使用的是事先约定的门槛融合，不按这张外层表挑全局最小者。

| 折 | 内部选择候选 | 门槛 | 8.2 基线 RMSE | 提交候选 RMSE |
|---|---|---|---:|---:|
| 1 | resnet18_highres | 不通过，保留基线 | 25.6311 | 25.6311 |
| 2 | efficientnet_b0 | 不通过，保留基线 | 12.2378 | 12.2378 |
| 3 | efficientnet_b0 | 不通过，保留基线 | 16.0370 | 16.0370 |
| 4 | efficientnet_b0 | 不通过，保留基线 | 41.8216 | 41.8216 |
| 5 | resnet18_highres | 不通过，保留基线 | 36.3789 | 36.3789 |

提交候选的 OOF RMSE 未降低（差值 +0.0000px），本地证据不支持直接替换已获 8.2 的基线。 共 0 张 OOF 图改善、0 张变差、80 张不变。
固定 OOF 组 bootstrap 的RMSE差值95%区间：[0.0, 0.0]。这些图与划分已用于多轮开发，区间不覆盖重训随机性、未知患者相关性或反复选方案的偏差，不能当作新的独立测试结论。

测试预测相对 8.2 CSV 平均移动 0.0000px、最大 0.0000px。这是预测之间的差异，不是测试误差。新候选真实 Kaggle 分数未知；已获8.2的文件继续保留。

## 交付与复现

- `submission.csv`：本次预设门槛融合的真实候选。
- `submission_baseline_8p2.csv`：原样的8.2提交文件。
- `fovea_research82_kaggle.ipynb`：完整原生Notebook，先重训两个基线共80阶段，再做本轮30阶段，总计110阶段。只需附比赛数据和新的预训练权重Dataset，GPU运行，Internet可关闭。
- `research82_pretrained_weights.zip`：官方ResNet18和EfficientNet-B0 ImageNet初始化，不能当作已经训练好的中心凹模型。
- `validation_report.json`：真实输入隔离、源文件身份、模型重载与CSV检查。
- `research82_documents.zip`：笔记本、两份CSV、研究说明、结果和核验。模型权重保留在本地实验目录，文档包不包含这些训练权重。

Notebook从头重训会受GPU/库版本影响，不保证逐位复现已交付CSV。融合后的坐标不继承单模型不确定性覆盖率承诺。没有病人ID，因此精确重复组隔离不代表患者级独立。
