# 独立 verifier 准备与 CPU 记录检查

`verify_results.py` 已实现并通过 Python 3.12 编译检查。本记录不代表完整实验已经验证通过。

在 primary split 2026 的 fold 0 完成后，仅用 CPU 读取实际产物核对了 single/dual 的四个 fine 阶段：训练输入行顺序、组/像素身份、裁剪、阶段 signature、有限 history、optimizer 更新、最佳轮次与最终 refit 轮次均通过。没有构建模型或执行 GPU 运算。

同折 calibration 风险规则按独立实现重新计算：四项固定策略、每组留一拟合/留出划分、选取策略、汇总 RMSE/P90、启用门槛和最终策略均通过。此次示例 effective policy 为 `global_quarter`，calibration 有 16 个组；这只是已有记录的核对，不是据此改实验规则。

CSV 浮点复现细节：本轮内存中的 `.15*1956` 为 `293.40000000000003`。普通 pandas 解析可能将 CSV 中该值转为 `293.4`，造成严格训练 signature 误报。Verifier 对本轮新 CSV 使用 `float_precision='round_trip'`，对继承的旧 CSV 仍采用生产代码原来的默认 parser。这能精确保留原训练 operands；没有放宽 signature 检查或最终重载的 `1e-3px` 容差。

正式 verifier 将在 65 阶段与两套五折全部 COMPLETE 后执行：

- 原始训练数据与像素身份复核，审计文件仅写自己的 `verification_input_audit/` 子目录。
- 所有旧版本保护清单核验、source SHA、65 阶段记录与角色隔离。
- 独立风险 leave-group-out、架构门槛、所有融合与 OOF 指标计算。
- 实际重载 30 个最终网络，逐 calibration/outer/test 原调用顺序与原 batch 划分推理，避免合批改变末批形状带来的数值差异。
- Secondary 无测试预测且不进入最终提交；测试 ROI 只核对几何，无虚构标签或覆盖率。
- 最终 40 行 submission 与五折融合重算一致，旧 8.2 文件保持原 SHA。

只有全部真实通过后才写 `validation_report.json` 的 PASS；任何异常都会保存 FAIL 和 traceback。

## 正式重载时发现的额外检查设计问题

正式 verifier 的 65 阶段记录核验已通过，但一次额外的 `Diagnostic decoder agreement` 检查报告 `0.01318359375px >= 0.001px`，因此该次总体状态确实为 FAIL，不能当作正式验证完成。

只读检查代码确认，该额外检查用 `both.head(2)` 做一次 batch=2 的普通推理，再与完整 calibration 按 batch=8 推理所产生的对应行比较。虽然图像相同，这并未保持原调用的批形状，在 AMP/CuDNN 下会把批形状相关浮点差异也纳入“解码器差异”。这属于验证测试设计的不匹配；不能只凭本次错误就判定已保存模型或提交被改动，也不能未重跑便断言批形状是全部差异的唯一原因。

合理修复是保留整个 calibration frame、同一行序、同一 ROI 与 BATCH_SIZE，再比较普通解码路径和诊断解码路径的 calibration 预测。这样测试隔离的是两条解码路径，且与原生产调用保持相同批次边界。Root 负责此项 verifier 修复；本审查没有修改 verifier、模型、训练代码或 CSV，也没有运行 GPU。

修复不能放宽 `1e-3px` 坐标容差或风险特征容差，先前 FAIL 记录应保留。最终是否所有 30 个模型、风险规则和 submission 都能复现，仍由修复后的完整真实重跑决定。
