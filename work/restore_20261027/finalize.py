from pathlib import Path
import json,hashlib,zipfile,csv
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/restored_20261027'
js=lambda p:json.loads(p.read_text(encoding='utf-8'))
comparison=js(OUT/'comparison.json');validation=js(OUT/'validation_report.json');execution=js(OUT/'seed_10000/execution_report.json')
assert validation['status']=='PASS'
a,b,c=comparison['scores'];delta=comparison['rmse_delta_candidate_minus_baseline']
direction='下降' if delta<0 else '上升'
text=f'''# 恢复 8.9 旧版，并检验保守随机种子集成

用户反馈的 Kaggle 分数：上一版 8.9，上一轮升级约 12.4，越低越好。线上成绩不是本轮工具读取的结果。
已恢复上一版为当前基线。`submission_restored_8p9.csv` 与原 `outputs/submission.csv` 的 SHA256 完全一致：
`{validation['baseline_sha256']}`。
原版代码、权重和上一轮 384 版本均保留；恢复前的代码及两版提交备份在 `backups/before_restore_20261027.zip`，逐文件验证通过。

## 本次改动

保留旧版 ResNet18、256/256 输入、热图 + DSNT、小范围细定位、独立校准、水平翻转 TTA、内部早停、内部裁剪策略选择和 B4 推理流程。
保留全部外层五折、内部早停、校准和交叉拟合分组。仅将各训练阶段模型初始化、采样和增强的种子加 10000。
新增五折共 40 个真实训练阶段；使用 {execution['gpu']}，耗时 {execution['elapsed_seconds']/60:.2f} 分钟。
没有将退步的 384 版本混入候选，没有改变标签，也没有针对测试图片手工调整坐标。

训练前固定候选为：旧版 75% + 新随机种子 25%。此配比是保守设计选择，并不是经验证的最佳配比。
两组各五折预测分别平均，然后加权。旧版原有五折权重被原样复用。
无法仅凭这两次线上分数，证明分辨率、ROI、独立校准或融合中的某一项是胜负原因；本轮通过控制变量检查随机种子集成。

## 实际内部结果

| 方法 | OOF 坐标 RMSE | 中位欧氏误差 | P90 欧氏误差 |
|---|---:|---:|---:|
| 8.9 原版 | {a['coordinate_rmse_px']:.4f} | {a['median_distance_px']:.4f} | {a['p90_distance_px']:.4f} |
| 新种子单独 | {b['coordinate_rmse_px']:.4f} | {b['median_distance_px']:.4f} | {b['p90_distance_px']:.4f} |
| 固定 75/25 候选 | {c['coordinate_rmse_px']:.4f} | {c['median_distance_px']:.4f} | {c['p90_distance_px']:.4f} |

候选 OOF RMSE 相比旧版{direction} {abs(delta):.4f}px。固定 OOF 预测的组 bootstrap 差值区间为 {comparison['paired_group_bootstrap_95pct_delta_interval']}。
这批 80 张训练图与划分已用于多轮开发；上述不是全新独立验证，也不是 Kaggle 成绩。区间没有覆盖重训随机性或未知患者关联。
旧版在线上更好而本地更差，已经说明当前 OOF 不能单独决定发布。**不自动晋升候选，8.9 文件仍为基线。**
候选的 Kaggle 成绩未知，不能承诺提升。

按逐图欧氏误差计，{comparison['oof_images_improved']} 张训练图的 OOF 预测改善，{comparison['oof_images_worsened']} 张变差。逐图变化保存在 `oof_error_changes.csv`，不对其中个别图片单独修改预测规则。

候选对 20 张测试图的预测平均移动 {comparison['mean_test_displacement_px']:.4f}px，最大移动 {comparison['max_test_displacement_px']:.4f}px。
这是两套预测之间的差异，不是测试误差。详见 `test_prediction_changes.csv`。

## 核验

首次独立复算发现核验脚本未调用训练流程中的 `seed_everything`，缺少相同的确定性后端设置，第一折旧模型出现最大约 0.686px 差异。补齐该调用后重新核验全部 20 个最终网络通过；没有修改已训练权重、预测文件或融合比例。首次失败日志保留于 `work/restore_20261027/verification_before_determinism_fix.log`。

- 所有 40 个新训练阶段输入记录检查通过；训练/早停没有测试图或外层验证图，独立校准图不进入模型训练。
- 所有数据角色和外层分组与旧版一致。重复图 39/77 保持同折；训练测试像素哈希无交集。
- 重新加载两组共 20 个最终网络，对全部测试图重新推理，候选最大差值 {validation['reloaded_candidate_max_difference_px']:.10f}px，仅来自 CSV 数值舍入。
- 40 行，`ImageID,value`，ID 顺序一致，坐标有限且在图片范围内。旧版文件未改动。
- 两组各自保留校准隔离，但融合后不能直接套用单模型的不确定性覆盖率；本文件没有作此承诺。
- 未知患者身份和近似重复不在精确像素去重保证内。

## 交付与使用

- `submission_restored_8p9.csv`：恢复的原版，可直接提交。
- `submission_candidate.csv`：完成训练与核验的 75/25 候选，可单独提交以获得实际 Kaggle 分数。
- `fovea_restored_ensemble_20261027.ipynb`：Kaggle 原生完整 Notebook，添加比赛数据与 `3pretrained` Dataset 后启用 GPU 运行。
- Notebook 会从 ImageNet 权重真实训练两组共 80 个阶段，生成 `/kaggle/working/submission.csv`（候选）及 `submission_restored.csv`（重训旧方法）。GPU 和库差异可能改变重训结果；需要原样恢复 8.9 时使用交付的 CSV。
- `comparison.csv`、`fold_comparison.csv`、`validation_report.json`：对比和核验。
- `delivery_20261027.zip`：上述小型交付文件与说明，不含训练权重。新权重保留于本地 `seed_10000/`，旧权重保留在原目录。

后续以实际 Kaggle 候选成绩决定是否采用。若候选没有优于 8.9，就继续保留原版，不通过反复试排行榜权重宣称验证改进。
'''
(OUT/'report.md').write_text(text,encoding='utf-8')
(OUT/'active_version.json').write_text(json.dumps({'active_submission':str(OUT/'submission_restored_8p9.csv'),
 'candidate_submission':str(OUT/'submission_candidate.csv'),'candidate_promoted':False,
 'reason':'User reports baseline 8.9 and rejected version about 12.4; candidate not yet scored.'},indent=2),encoding='utf-8')
names=['submission_restored_8p9.csv','submission_candidate.csv','fovea_restored_ensemble_20261027.ipynb','report.md',
 'comparison.csv','comparison.json','fold_comparison.csv','test_prediction_changes.csv','oof_error_changes.csv','validation_report.json',
 'protocol.json','implementation_check.json','notebook_validation.json','active_version.json']
manifest={name:hashlib.sha256((OUT/name).read_bytes()).hexdigest() for name in names}
with zipfile.ZipFile(OUT/'delivery_20261027.zip','w',zipfile.ZIP_DEFLATED) as z:
 for name in names:z.write(OUT/name,name)
 z.writestr('manifest.json',json.dumps(manifest,indent=2))
with zipfile.ZipFile(OUT/'delivery_20261027.zip') as z:
 assert z.testzip() is None
 for name,digest in manifest.items():assert hashlib.sha256(z.read(name)).hexdigest()==digest
print(text)
