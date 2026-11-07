from pathlib import Path
import csv,json,math,hashlib,zipfile,sys
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/optimized_20261026'
sys.path.insert(0,str(ROOT/'work/kaggle_test_deps'))
import numpy as np
def read(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def js(p):return json.loads(Path(p).read_text(encoding='utf-8'))
validation=js(OUT/'validation_report.json');assert validation['status']=='PASS'
execution=js(OUT/'execution_report.json');backup=js(ROOT/'backups/backup_20261026_verification.json')
oldrows={r['image_id']:r for r in read(ROOT/'outputs/trained_submission_20261025_fixed/oof_B4.csv')}
newrows=read(OUT/'oof_BLEND.csv')
errors=[];olderrors=[];group_ids=[]
for r in newrows:
    o=oldrows[r['image_id']]
    assert r['fold']==o['fold'] and float(r['x_true'])==float(o['x_true']) and float(r['y_true'])==float(o['y_true'])
    errors.append((float(r['x'])-float(r['x_true']))**2+(float(r['y'])-float(r['y_true']))**2)
    olderrors.append((float(o['x'])-float(o['x_true']))**2+(float(o['y'])-float(o['y_true']))**2)
    group_ids.append(r['group'])
errors=np.array(errors);olderrors=np.array(olderrors);names=sorted(set(group_ids))
counts=np.array([group_ids.count(g) for g in names])
group_new=np.array([errors[np.array(group_ids)==g].sum()/2 for g in names])
group_old=np.array([olderrors[np.array(group_ids)==g].sum()/2 for g in names])
rng=np.random.default_rng(2026);weights=rng.multinomial(len(names),np.full(len(names),1/len(names)),size=2000)
deltas=np.sqrt((weights@group_new)/(weights@counts))-np.sqrt((weights@group_old)/(weights@counts))
newscore=float(np.sqrt(errors.mean()/2));oldscore=float(np.sqrt(olderrors.mean()/2))
comparison={'old_rmse':oldscore,'new_rmse':newscore,'difference_new_minus_old':newscore-oldscore,
 'relative_reduction':(oldscore-newscore)/oldscore,'delta_ci_low':float(np.quantile(deltas,.025)),
 'delta_ci_high':float(np.quantile(deltas,.975)),'old_median_distance':float(np.median(np.sqrt(olderrors))),
 'new_median_distance':float(np.median(np.sqrt(errors))),'old_p90_distance':float(np.quantile(np.sqrt(olderrors),.9)),
 'new_p90_distance':float(np.quantile(np.sqrt(errors),.9)),
 'scope':'Paired group bootstrap of fixed OOF predictions on reused development data; not an independent test or a causal ablation of individual changes.'}
(OUT/'comparison_with_previous.json').write_text(json.dumps(comparison,indent=2),encoding='utf-8')
summary=read(OUT/'ablation_summary.csv')
table='| 方法 | 坐标 RMSE（像素） | 中位欧氏误差 | P90 欧氏误差 |\n|---|---:|---:|---:|\n'
table+=f"| 旧版 B4 | {oldscore:.4f} | {comparison['old_median_distance']:.4f} | {comparison['old_p90_distance']:.4f} |\n"
for r in summary:table+=f"| 新版 {r['method']} | {float(r['rmse']):.4f} | {float(r['median_distance_px']):.4f} | {float(r['p90_distance_px']):.4f} |\n"
report=f'''# 2026-10-26 优化与实际 GPU 训练记录

已先完成备份，再创建独立新版并训练。旧版 `outputs/submission.csv` 未被修改。

## 备份

备份文件：`{Path(backup['backup_path']).name}`。共 {backup['file_count']} 个现存文件，{backup['zip_bytes']/1e9:.2f} GB，逐文件 SHA256 校验通过。
包括基础版、科研版、五轮优化版、10月25日训练版、原始用户附件，以及现存模型和训练记录。
仅排除可重建的依赖目录和 Python 缓存；此前已删除的临时测试权重无法补回。

## 本轮具体改动

1. 粗/细输入从 256 提升到 384，热图从 64 提升到 96。
2. 取消原来退化为固定最小裁剪的不确定性校准和未触发的回退分支。本版本不提供校准覆盖率结论。
3. 各折内部选择训练轮数，再用该外层训练折全部 64 张图拟合最终网络。旧版最终网络只用了约 47–48 张。
4. 内部三折粗预测生成训练裁剪中心；细训练与推理都覆盖 0.20、0.30、0.40 的视野比例，并按原方法加边距。推理时不读取目标坐标、不修复裁剪。
5. 预先固定最终输出为 50% 粗定位 + 50% 三尺度细定位均值。保留水平翻转 TTA，最后对五折平均。未根据最终 OOF 结果选择融合权重。
6. 粗模型从第 10 轮、细模型从第 8 轮起选择 checkpoint；最大轮数 35/30，patience 10。降低预训练编码器学习率到解码器的 0.25 倍（5e-5 对 2e-4）。
7. 将 AMP 初始 loss scale 设为 1024，试跑确认粗/细模型均有实际优化器更新；早停计算的融合方法与部署推理一致。
8. 缓存粗定位等比例缩放后的图像，返回副本供增强使用，减少重复 JPEG 解码。没有使用测试图像拟合归一化统计量。

这些是组合改动。本次组件输出比较不能单独证明某一个改动带来了多少收益。

## 实际运行与验证

- GPU：{execution['device']}；真实训练耗时 {execution['elapsed_seconds']/60:.2f} 分钟。
- 80 张训练图、20 张测试图；五折 × 七个阶段 = 35 个训练阶段。
- 与旧版完全相同的外层折划分；图像 39/77 保持同组。没有患者 ID，不能证明患者级独立性。
- 每阶段均保存实际传入训练和早停的图像 ID、路径、分组；审计没有发现测试 ID 或外层验证图像进入这些阶段。
- 全部 10 个最终网络已重新加载，并重新读取测试图进行推理；与交付 CSV 最大差异 {validation['reloaded_submission_max_difference_px']:.9f} 像素。
- `submission.csv` 的 40 行、列名、ID 顺序、有限数值和图像范围均已检查。

## 同划分内部验证结果

{table}

新版减旧版的 RMSE 差：{comparison['difference_new_minus_old']:.4f} 像素；配对图像组 bootstrap 95% 区间 [{comparison['delta_ci_low']:.4f}, {comparison['delta_ci_high']:.4f}]。
相对 RMSE 降幅为 {comparison['relative_reduction']:.2%}，负值表示变差。

这些是内部 OOF 指标，不是 Kaggle 线上成绩。由于本轮设计参考了此前同一数据的验证结果，不能将上述比较当作全新独立测试；重采样区间也不包含重训随机性和未知患者相关性。

## 交付

- 本目录 `submission.csv`：新版本的测试集预测。
- `fovea_optimized_20261026.ipynb`：自包含 Kaggle Notebook。添加比赛数据和现有 `3pretrained` Dataset，启用 GPU 后 Run All。
- `validation_report.json`：实际训练输入、模型文件和重载预测检查。
- `comparison_with_previous.json`、`ablation_summary.csv`：新旧比较和组件结果。
- `fold_0/` 至 `fold_4/`：现存实际模型、训练日志、数据清单与各折预测。

旧版提交和历史代码仍保留原路径，同时存在于独立备份压缩包中。
'''
(OUT/'optimization_report.md').write_text(report,encoding='utf-8')
files=['submission.csv','fovea_optimized_20261026.ipynb','optimization_report.md','validation_report.json',
       'comparison_with_previous.json','ablation_summary.csv','run_summary.json','notebook_static_validation.json']
package=OUT/'delivery_20261026.zip'
with zipfile.ZipFile(package,'w',zipfile.ZIP_DEFLATED) as z:
    for name in files:z.write(OUT/name,name)
with zipfile.ZipFile(package) as z:
    assert z.testzip() is None
    for name in files:assert z.read(name)==(OUT/name).read_bytes()
print(json.dumps(comparison,indent=2));print(table)
