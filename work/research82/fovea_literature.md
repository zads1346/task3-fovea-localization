# 8.2 基线之后的中心凹定位文献核查

检索日期：2026-10-27。研究对象为 80 张训练图、20 张测试图、点坐标监督、单 RTX 4060 Laptop 8GB 的 Kaggle 2026task3。8.2 是用户反馈的线上分数，本文没有从 Kaggle 读取分数，也没有训练或修改现有模型。

## 判断

目前证据最支持的算法家族是：**ImageNet 预训练卷积编码器 + U-Net/FPN 式稠密定位 + 局部精修 + 有控制的模型集成**。不存在一篇论文能证明某个算法在这 80 张图上必然最好。尤其不能将不同数据集上的平均欧氏距离（AED）、归一化误差、分割 IoU 与本项目坐标 RMSE 排成一个“最优算法”榜单。

最值得做的单项架构对照是 **EfficientNet-B0 编码器的 U-Net 式热图分支**：与现有 ResNet18 产生结构差异，保留已经得到线上支持的 256 分辨率、粗到细裁剪、DSNT 解码和五折流程。这是结合任务论文和硬件约束提出的适配假设；文献并没有直接证明 B0 比本项目 ResNet18 更优。分支若没有验证优势，应继续保留 8.2 文件。

## 五篇核实文献

### 1. GAMMA challenge: Glaucoma grAding from Multi-Modality imAges

Wu 等，Medical Image Analysis 90, 102938，2023。DOI：[10.1016/j.media.2023.102938](https://doi.org/10.1016/j.media.2023.102938)。[作者公开全文](https://arxiv.org/html/2202.06511v4)，[作者网页所附正式论文 PDF](https://ignaciorlando.github.io/publication/2023-media/2023-media.pdf)。

- 原始挑战共 300 个眼底/OCT 配对样本，训练、预赛、终赛各 100。中心凹任务第一名 DIAGNOS-ETS 使用双堆叠 W-Net、中心圆形伪掩膜分割和四折温度集成；第二名 IBME 使用 EfficientNet-B5 U-Net 热图；第三名 SmartDSP 使用 EfficientNet-B4 两阶段坐标回归。依据是中心凹附表 7/9，不能误把青光眼分类总榜当作定位榜。
- 直接启发：密集空间监督和不同模型的互补比单纯扩大模型更有依据。当前点标签可以生成热图或圆形训练目标，不需要把圆形伪标签说成真实分割标注。
- 指标限制：论文中的归一化 ED/挑战分数不能直接与本项目原图像素坐标 RMSE 比较。
- 数据来源注意：该文的两种眼底尺寸恰为 2992×2000、1956×1934，与本项目吻合。这只是可能同源的线索，不是同源结论；在排除交集前，不应直接复用 GAMMA 已训练定位权重或引入其额外坐标标签。

### 2. ADAM Challenge: Detecting Age-Related Macular Degeneration From Fundus Images

Fang 等，IEEE Transactions on Medical Imaging 41(10), 2828–2847，2022。DOI：[10.1109/TMI.2022.3172773](https://doi.org/10.1109/TMI.2022.3172773)。[作者全文](https://arxiv.org/html/2202.07983v3)，[PubMed 记录](https://pubmed.ncbi.nlm.nih.gov/35507621/)。

- 1200 张图，训练、在线评估、现场评估各 400。中心凹第一名 VUNO EYE TEAM 使用 EfficientNet 编码器 U-Net、血管先验、外部数据和两模型平均；其他强方法采用分割与回归融合。原文表 6 和附录 6.3 可核对。
- 对本项目的支持：预训练 CNN 编码器、空间输出及集成值得优先比较。论文胜负不能单独归因于 EfficientNet，因为额外数据和血管信息也是变量。
- 限制：该挑战还含中心凹不可见样本，其处理规则不适合全部 80 张均可见的当前数据。这里也采用平均欧氏距离，并非本项目坐标 RMSE。

### 3. JOINEDTrans: Prior guided multi-task transformer for joint optic disc/cup segmentation and fovea detection

He 等，Computers in Biology and Medicine 177, 108613，2024。DOI：[10.1016/j.compbiomed.2024.108613](https://doi.org/10.1016/j.compbiomed.2024.108613)。[作者预印本全文](https://arxiv.org/html/2305.11504v1)，[PubMed 记录](https://pubmed.ncbi.nlm.nih.gov/38781644/)。

- 联合视盘/视杯分割与中心凹检测，使用血管预训练先验和粗到细结构。在 GAMMA 实验中使用 100 张训练图；血管编码器另用六个数据集共 338 张标注图训练。作者预印本在 GAMMA 报告精定位 AED 14.97±28.15，但该指标不能作为本竞赛预计成绩。
- 可借鉴的是解剖上下文与局部精修。现有数据缺少视盘/血管掩膜，直接照搬整套多任务监督并不成立。
- 作者跨域讨论也指出 100 图条件下 Transformer 的数据量敏感性。预印本中的精定位模块最终使用 128×128，说明分辨率不是越高越好。
- 作者论文给出的 [GitHub 地址](https://github.com/HuaqingHe/JOINEDTrans) 在当前连接下读取仓库和 README 均返回 404；本次无法确认其可直接运行性，不能声称已复现。

### 4. VascX Models: Deep Ensembles for Retinal Vascular Analysis From Color Fundus Images

Vargas Quiros 等，Translational Vision Science & Technology 14(7), 19，2025。DOI：[10.1167/tvst.14.7.19](https://doi.org/10.1167/tvst.14.7.19)。[正式论文](https://tvst.arvojournals.org/article.aspx?articleid=2810436)，[作者预印本](https://arxiv.org/abs/2409.16016)，[作者中心凹模型卡](https://github.com/Eyened/rtnls_vascx_models/blob/main/fovea/fovea_july24.md)。

- 中心凹部分为五模型 U-Net 热图回归集成，使用 IDRiD、ADAM、PALM、REFUGE2 和 Rotterdam 数据；不是仅凭某个巨大基础模型取胜。正式论文采用 Gaussian 目标和 MSE，并强调预处理、增强及多来源数据。
- 正式论文的中心凹表报告 REFUGE2 欧氏距离约 10.68px、IDRiD 14.68px，属于其五折方案；输入预处理、数据和坐标尺度不同，不能与我们的 8.2 直接排序。
- 作者旧仓库已经迁移至 [retinalysis-vascx](https://github.com/eyened/retinalysis-vascx)，旧 Python 包已移除。模型卡列出外部训练集，使用现成权重之前必须核验训练数据覆盖，不能把未知数据覆盖的权重当作无泄漏初始化。

### 5. Contextual anatomy-guided deep learning for accurate fovea segmentation in diabetic retinopathy fundus images

Chankhachon 等，Scientific Reports，2026，正式页面发布日期 2026-02-24。DOI：[10.1038/s41598-026-40287-y](https://doi.org/10.1038/s41598-026-40287-y)。[期刊页面](https://www.nature.com/articles/s41598-026-40287-y)，[期刊提供的接收稿 PDF](https://www.nature.com/articles/s41598-026-40287-y_reference.pdf)，[作者代码](https://github.com/acidrainpsu1980-byte/MN4Fovea)。

- MNv4Fovea 使用 ImageNet MobileNetV4 + DeepLabV3+，联合中心凹、视盘、血管、视网膜、背景五类监督，并引入 GEV 颜色分布增强。原始训练仅 54 图，独立测试 27 图，但有完整像素级标注；小样本不等于与本项目监督条件相同。
- 文中输入 1024×1024、训练环境四张 RTX A4000（每张 16GB）。研究支持增强与解剖上下文的价值，不能由此直接推荐在单 8GB 上照搬训练。
- 作者代码可读，README 的首层卷积说明为 7×7，而论文写 11×11；复现前需要核对实现差异。论文 GEV 还借助外部血管数据集的颜色分布，不能直接改成从本竞赛测试集拟合颜色统计。

## 建议的受控实验

1. 固定并备份 8.2 CSV、旧模型和 OOF；把本次新增算法作为独立版本。
2. 新增 ImageNet EfficientNet-B0 编码器的 U-Net 式热图定位模型；保留当前输入尺寸、损失、增强、裁剪、早停、分组和校准规则，先只改变编码器家族。不要同时改变分辨率、引入更多训练图和取消校准，这样才能解释结果。
3. 先进行单批前向/反向和坐标还原检查，再完整五折训练。报告该分支单独 OOF、与 8.2 基线的固定融合、逐折以及逐图误差。融合权重需在看到新外层结果前确定；保留 8.2 原提交。
4. 不把温度校准后的定位分布直接当作融合后的覆盖率保证。单模型校准和集成后的不确定性需要区分。
5. 若该分支值得继续，再单独对照圆形伪掩膜 BCE/Dice、颜色增强或局部回归头。不能同时添加所有论文模块后把任何收益归因于某一模块。

这里“最适配”指证据、标注条件、计算预算和可审计验证共同支持的优先候选，而不是已经证明的任务最优算法。8.2 之前出现过本地 OOF 改善但线上退步，因此本轮仍须由真实 Kaggle 提交确认泛化。

## 检索及代码可用性记录

- 已读取 agent-reach 技能及 search/web 路由。PATH 中无 mcporter/agent-reach；按 Jina Reader 路由读取 arXiv 超时（12 秒），随后改用可用网页检索核对原始论文。
- 论文元数据以期刊、PubMed、作者 arXiv 和作者网站为依据；没有用二手综述的“最高分”替代原始结果。
- 作者 GitHub 文件通过已连接 GitHub 工具只读核验。VascX 和 MN4Fovea 可读，JOINEDTrans 当前读取 404。
- `conda run -n dl agent-reach check-update` 失败：本机不存在 anaconda3/envs/dl 环境。未安装或改动环境，也没有因此暂停研究。
- 本文是交付研究笔记，不是原始网页抓取缓存；没有下载额外眼底标签或训练数据。
