# 双尺度精修与风险回退：训练前协议审查

2026-10-27。本审查只读取既有训练代码、训练划分与模型记录，没有读取外部测试参考坐标或私榜审计目录，也没有运行 GPU、改动旧代码或旧提交。

## 结论与已核实事实

1. **原 crop cross-fit 并未把早停图加入训练。** `work/restore_20261027/engine.py` 的 `run_research_fold` 实际调用 `research_crossfit_crops(fitting, ...)`，而非对整个 development 调用。逐折核对 `outputs/restored_20261027/seed_10000/fold_k/crop_crossfit/crossfit_audit.json`：三个子折 held 集并集等于 model_fit，fit 集并集也等于 model_fit，与 early_stop 的交集全部为空。因此可以复用经 SHA 校验的 primary 裁剪中心，没有证据支持此前担心的“早停图经过粗网络进入 fine 训练中心”的直接泄漏。
2. 原 coarse_select 在 model_fit 训练、early_stop 选轮；stopping 裁剪来自该模型。原 cross-fit 的粗模型轮数沿用同一个 early_stop 选出的 coarse 轮数。这是内层验证的多次选型使用，存在选择噪声，不能称为完全独立的内层检验；但不同于把 early_stop 图像用于模型梯度训练。新方案保留此层级，外层仍只评估。
3. **早停尺度与部署尺度应统一。** 新 single 与 dual 都在实际推理的 nominal 0.15 短边窗口选轮，不再用旧 0.20/0.30/0.40 的平均分作为这一轮门槛。
4. 新 single control 同时受到统一窗口、统一选轮、sigma=1.5 等既定处理，因而它与历史 8.2 不构成只改变 dual 结构的因果对照。架构效果必须比较这轮匹配的 SINGLE 与 DUAL_RAW；历史 8.2 单独作为保留的提交基线。

已核对的 primary 子折数量：

| 外折，0-based | model_fit 图数 | early_stop 图数 | crop cross-fit held 并集 | 与 early_stop 交集 |
|---:|---:|---:|---:|---:|
| 0 | 38 | 10 | 恰为 model_fit | 0 |
| 1 | 38 | 9 | 恰为 model_fit | 0 |
| 2 | 38 | 9 | 恰为 model_fit | 0 |
| 3 | 39 | 9 | 恰为 model_fit | 0 |
| 4 | 39 | 9 | 恰为 model_fit | 0 |

## 已确认冻结的实验范围

- Primary：原分组种子 2026，复用 seed10000 经文件摘要核实的 coarse_select、coarse_final、fitting 交叉预测裁剪中心与 stopping 预测中心。
- Secondary：预先固定分组种子 2027，重新执行完整五折所需的粗模型训练与交叉预测；仅作稳定性审计，不进入最终 submission，也不据其结果改变模型、sigma、门槛或融合比例。
- 两套实验均保持原 80 张训练图和精确重复组。校准组与外层验证组均不得进入任何模型梯度训练；同组不得跨 model_fit/early_stop/calibration/outer 角色。
- 两个 fine arms：SINGLE 与 DUAL。共同使用 ImageNet ResNet18、256 输入、64 热图、fine sigma=1.5、原 KL + DSNT + 坐标损失、相同优化超参数、相同种子策略、相同模型输入行序与训练中心。
- 局部推理窗口精确定义为 `0.15 * min(width,height)`，不叠加旧 ROI_MARGIN_PX；dual context 是同中心、相同原图几何变换下的 2 倍窗口。不得写成“0.15 加上 32px”或其他隐含尺度。
- `FINE_MULTISCALE_TRAIN=False`、`FINE_MULTISCALE_EARLY_STOP=False`。每个 fine arm 只做一次选轮和一次固定轮数 refit，不同时搜索其他输入尺度、sigma、编码器或损失权重。
- 继续保留原训练 jitter 及 **仅训练时**目标落窗外的 enlargement。因此“固定 0.15”指 nominal 训练尺度和实际验证/推理尺度，并不代表每个增强后训练窗口都严格为 0.15。必须记录训练 enlargement 比例；验证和推理不得用标签修复窗口。

新增训练阶段预算：primary 20 个 fine 阶段；secondary 20 个 fine 阶段 + 25 个 coarse 阶段，共 **65 个新阶段**。Secondary 每折的 coarse_select + 3 个 fitting cross-fit + coarse_final 共 5 阶段。Primary 旧 coarse 的复用不计为新增训练。

## 分工清楚的消融与选择规则

每套 split 均报告 COARSE、SINGLE、DUAL_RAW、DUAL_RISK、MATCHED_DEPLOY；primary 另报告旧 8.2 与最终提交。

| 问题 | 配对对照 |
|---|---|
| 双尺度特征是否有用 | SINGLE vs DUAL_RAW |
| 回退是否改善精修 | DUAL_RAW vs DUAL_RISK |
| 内部门槛融合是否稳健 | SINGLE vs MATCHED_DEPLOY |
| 最终候选相对历史基线如何 | primary 历史 8.2 vs FINAL |

每折只比较同一 stopping 集、同一 nominal ROI 与 TTA 口径下的 matched single/dual 最佳早停 RMSE。DUAL 至少改善 **0.5px 且 2%** 才接入。该决定应先写入 selection 记录，再产生该折外层和测试新分支预测。此处使用的是选轮内层分数，存在选型噪声，不是新的独立检验。

若通过：

- `MATCHED_DEPLOY_k = 0.8 * SINGLE_k + 0.2 * DUAL_RISK_k`
- primary `FINAL_k = 0.8 * BASELINE_8P2_k + 0.2 * DUAL_RISK_k`

若未通过：

- `MATCHED_DEPLOY_k = SINGLE_k`
- primary `FINAL_k = BASELINE_8P2_k`

测试坐标对全部五折 `FINAL_k` 等权平均，不只平均通过门槛的折。旧 `BASELINE_8P2_k = 0.75 * 原 seed0_k + 0.25 * seed10000_k`。全部拒绝时必须逐字节复现原 8.2 CSV；不能无条件混入新 single control 后宣称“拒绝保留原版”。

## 风险回退的边界

只使用未参与两个最终网络训练的 calibration 组拟合离散回退策略。风险特征只能来自图像、模型输出、预测中心和给定 ROI，包括尺度扰动稳定性、TTA 分歧、抑制主峰邻域后的次峰比例。特征生成函数对输入采用 image_id/path/width/height 白名单；不得根据标签决定预测裁剪、特征值或“困难图”规则。

少量预定策略在 calibration 组内按组留一验证。最终完整 calibration 上选择的规则只有在留一汇总达到预定 RMSE 和 P90 条件时才启用。需要记录每个留一组、其余拟合组、逐候选分数、触发比例、最终选择以及拒绝原因。重复使用同一小型 calibration 进行规则选择和启用判断仍有噪声，不能把该校准分数视为独立证据；外层配对结果才用于报告实际效果。

如果 gate 规则改变了坐标或组合，原 coarse 椭圆覆盖率不能直接移植到最终集成。回退/融合方法的覆盖率或置信声明需独立报告，不能从单模型的不确定性继承。

## 必须保留的报告指标

- 坐标 RMSE：`sqrt(mean(dx^2,dy^2))`；P90 和最大误差用单图欧氏距离，明确区分单位与定义。
- fine 相比 coarse 变差的比例，SINGLE/DUAL 配对改善图数。
- 局部 ROI 和双倍 context ROI 的真实目标覆盖率，仅作外层评价，不用于改推理窗口。
- 风险触发率与实际启用/回退率，避免把“实现了回退”当成“回退起作用”。
- 每折数值与两套固定 split 分开列出，不能把同一 80 张图的两套 OOF 当成 160 个独立样本。
- 若进行 group bootstrap，解释这是固定预测上的重采样，不包含重训随机性、患者相关性或重复开发选择偏差。
- Secondary 无论正负都完整报告。外层已用于多轮开发，本次是开发稳健性检查，不能宣传为新的最终独立测试。

## 独立核验清单

1. 旧 submission、旧 source 及所有复用模型/预测/crop lineage 在运行前后摘要不变；新产物仅写新目录。
2. train 与 test 的 ID、路径、像素摘要无交集；运行程序不访问附件测试参考、私榜分数或其他外部标签。
3. 65 个新阶段均有真实 optimizer 更新、有限损失、history 与训练/选轮输入记录。每个阶段输入符合其角色；primary inherited cross-fit lineage 仍满足上述隔离。
4. single/dual 共享 nominal ROI、几何增强规则和选轮指标；context 对齐几何检查与模型梯度检查通过。
5. 独立重算每折架构门槛、risk calibration 决策、所有融合算术、OOF 指标及提交 40 行格式。
6. 重载实际 final checkpoints 复现候选预测；必须验证新 single、dual 以及参与混合的原基线。Checkpoint 推理不得再初始化下载权重或隐式更换模型。
7. 只在完整五折与独立验证通过后生成最终文件和文档包；失败或中断留下真实状态，不伪造“完成”。

本文件是训练前协议审查，不是已经完成的模型验证报告。后续 verifier 应按实际输出 schema 实现并运行。
