from pathlib import Path

root = Path(__file__).resolve().parent
text = (root / 'build_research_deliverables.py').read_text(encoding='utf-8')
text = text.replace('read = lambda name: (ROOT / "work" / name).read_text(encoding="utf-8")',
'''def read(name):
    path = ROOT / "work/fovea_v3" / name
    if not path.exists():
        path = ROOT / "work" / name
    return path.read_text(encoding="utf-8")''')
for old, new in [('fovea_research_kaggle', 'fovea_optimized_5rounds'),
                 ('fovea_research_complete.zip', 'fovea_optimized_complete.zip'),
                 ('README_research.txt', 'README_optimized.txt'),
                 ('research_validation.json', 'five_round_validation.json'),
                 ('/kaggle/working/fovea_research/', '/kaggle/working/fovea_optimized/'),
                 ('科研实验包 v2.0', '五轮优化实验包 v3.0')]:
    text = text.replace(old, new)
text = text.replace('# 2026task3：粗到细中心凹定位与不确定性校准',
                    '# 2026task3：五轮优化 · 粗到细定位与不确定性校准')
text = text.replace('本地测试仅用于验证程序。Notebook 内没有伪造训练结果、比赛分数或预填输出。',
'''本地测试包含真实数据的小预算训练和功能验收；不代表正式比赛精度。Notebook 没有预填训练结果。

### 本版五轮改进
1. 修复早停选择与最终重训的学习率计划不一致；早停与推理共用 TTA 解码。
2. OOF 预测中心配合多尺度裁剪；多尺度早停，记录裁剪扩大率。
3. 编码器分组学习率、可选动态 EMA、损失分量和学习率日志；这些候选仍需足够训练预算评价。
4. 小校准集的回退采用实质性改善门槛；报告裁剪钳制、尺度退化和误差关联。
5. 完成训练阶段可复用；模型、CSV、校准文件做 SHA256 完整性验证，五折独立验收。

已有的 `3pretrained` 数据集可继续使用，无需重新上传权重。''')
text = text.replace('`REUSE_CHECKPOINTS=True` 只复用数据/代码/配置一致的**已完成外层折**。中断的折会从该折开头重跑；不是优化器断点续训。新会话须保留旧输出目录才可复用。',
'''`REUSE_CHECKPOINTS=True` 可复用数据/代码/配置一致的**已完成训练阶段和外层折**。所有依赖文件经 SHA256 校验；损坏时明确报错。中断阶段从该阶段开头重训，不是 epoch/优化器断点续训。必须保留旧输出目录；新 Kaggle 会话可先把保存的输出下载/挂载并复制到新的工作目录，路径配置保持一致。''')
text = text.replace('复用只覆盖已完成的外层折，不是中断epoch/优化器恢复。',
'''复用覆盖已完成的训练阶段和外层折，SHA256检查；中断阶段从头重训。
所有代码内嵌，无需额外Python模块。旧3pretrained数据集可直接使用。
五轮是代码改进与验证轮次，不是五次已证实的比赛提分。''')
text = text.replace('- `BATCH_SIZE=8`；显存不足时改4或2。单张GPU即可；不要求双卡。',
'''- `BATCH_SIZE=8`；显存不足时改4或2。单张GPU即可；不要求双卡。
- `OOF_MULTISCALE=True` 在预测中心采样训练尺度；`FINE_VALIDATION_FRACTIONS` 指定细模型早停尺度。验证窗口不会用真值扩窗。
- `BACKBONE_LR_MULTIPLIER` 控制预训练编码器学习率，`USE_EMA` 控制指数滑动平均。默认及候选详情见下方配置，不能把短预算开发结果等同于正式收益。
- 每轮日志包含热图/坐标损失、各参数组学习率、裁剪尺度及扩大率。校准退化提示会保存在每折 `calibration_health.json`。
- 建议预留至少约4GB输出空间；所有35个阶段权重会保存以支持复用。''')
text = text.replace('"## 8. 开始训练\\n\\n此单元格执行全部折和消融。低分辨率冒烟检查只能验证流程，不能代表正式定位精度。"',
                    '"## 8. 开始训练\\n\\n此单元格执行全部折和消融。低分辨率冒烟检查只能验证流程，不能代表正式定位精度。五轮优化已体现在代码中；这里是一次完整的五折实验。"')
text = text.replace('readme = """', 'readme = """')
text = text.replace('for i, cell in enumerate(cells):',
'''report_path = OUT / "five_round_report.md"
if report_path.exists():
    cells.insert(1, markdown(report_path.read_text(encoding="utf-8")))
for i, cell in enumerate(cells):''')
text = text.replace('cells = [markdown(intro)', 'preflight = "PREFLIGHT = research_preflight(CFG)\\n"\ncells = [markdown(intro)')
text = text.replace('`FINE_VALIDATION_FRACTIONS`', '`FINE_VALIDATION_ROI_FRACTIONS`')
text = text.replace('`OOF_MULTISCALE=True`', '`FINE_MULTISCALE_TRAIN=True`')
text = text.replace('默认每折7次模型训练：粗模型轮数选择、3个训练裁剪的交叉拟合模型、细模型轮数选择，以及最终粗/细模型。五折共35次',
                    '默认每折8次模型训练：粗模型轮数选择、3个训练裁剪交叉拟合模型、2种细模型裁剪策略比较，以及最终粗/细模型。五折共40次')
text = text.replace('默认5个外层折，每折7个训练任务，共35次', '默认5个外层折，每折8个训练任务，共40次')
text = text.replace('所有35个阶段权重', '所有40个阶段权重')
text = text.replace('OOF 预测中心配合多尺度裁剪；多尺度早停，记录裁剪扩大率。',
                    'OOF 预测中心配合候选多尺度裁剪；每折内比较固定/多尺度训练，多尺度早停，记录裁剪扩大率。')
text = text.replace('- `FINE_MULTISCALE_TRAIN=True` 在预测中心采样训练尺度；',
                    '- `SELECT_FINE_CROP_POLICY=True` 默认在每折内比较固定/多尺度，外层验证与校准标签不参与选择。关闭后由 `FINE_MULTISCALE_TRAIN` 固定策略；')
text = text.replace('`fold_*/coarse_final/model.pt`、`fine_final/model.pt`',
                    '`fold_*/fine_policy_selection.json` 记录折内裁剪策略比较；`fold_*/coarse_final/model.pt`、`fine_final/model.pt`')
text = text.replace('with zipfile.ZipFile(OUT / "fovea_optimized_complete.zip",',
'''if (OUT / "five_round_report.md").exists():
    bundle_files.append("five_round_report.md")
with zipfile.ZipFile(OUT / "fovea_optimized_complete.zip",''')
(root / 'build_optimized_deliverables.py').write_text(text, encoding='utf-8')
print('Created optimized builder; old artifacts preserved.')
