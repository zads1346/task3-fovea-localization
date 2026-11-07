from pathlib import Path
import csv
import json
import math
import shutil

ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'outputs/trained_submission_20261025_fixed'
def rows(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def js(p):return json.loads(Path(p).read_text(encoding='utf-8'))
metrics=rows(RUN/'ablation_summary.csv')
execution=js(RUN/'execution_report.json')
reload_report=js(ROOT/'outputs/submission_reload_verification.json')
assert reload_report['status']=='PASS'
table='| 方法 | OOF 坐标 RMSE（像素） | 中位欧氏误差 | P90 欧氏误差 |\n|---|---:|---:|---:|\n'
names={'B0':'B0 粗定位','B1':'B1 固定裁剪细定位','B2':'B2 未校准半径','B3':'B3 校准半径','B4':'B4 校准半径 + 回退（提交）','C1':'C1 面积匹配固定裁剪'}
for r in metrics:
    table+=f"| {names[r['experiment']]} | {float(r['rmse']):.4f} | {float(r['median_distance_px']):.4f} | {float(r['p90_distance_px']):.4f} |\n"
fold_table='| 折 | 最终网络训练图数 | 校准图数 | 外层验证图数 | 粗轮数 | 细轮数 | 细训练尺度策略 |\n|---|---:|---:|---:|---:|---:|---|\n'
min_clamps=[];fallbacks=[]
for f in range(5):
    p=RUN/f'fold_{f}'
    info=js(p/'complete.json')
    policy=js(p/'fine_policy_selection.json')['selected']['policy']
    fold_table+=f"| {f+1} | {info['n_development']} | {info['n_calibration']} | {info['n_validation']} | {info['coarse_epochs']} | {info['fine_epochs']} | {policy} |\n"
    fm={r['experiment']:r for r in rows(p/'metrics.csv')}
    min_clamps.append(float(fm['B3']['roi_min_clamp_rate'])*int(fm['B3']['n_images']))
    fallbacks.append(float(fm['B4']['fallback_rate'])*int(fm['B4']['n_images']))
b3={r['image_id']:r for r in rows(RUN/'oof_B3.csv')}
c1={r['image_id']:r for r in rows(RUN/'oof_C1.csv')}
max_difference=max(abs(float(b3[k][a])-float(c1[k][a])) for k in b3 for a in ['x','y'])
comparisons=rows(RUN/'paired_group_bootstrap.csv')
b4=next(r for r in comparisons if r['method']=='B4')
oof=rows(RUN/'oof_B4.csv')
worst=sorted(oof,key=lambda r:(float(r['x'])-float(r['x_true']))**2+(float(r['y'])-float(r['y_true']))**2,reverse=True)[:5]
worst_text=', '.join(f"{r['image_id']}（{math.hypot(float(r['x'])-float(r['x_true']),float(r['y'])-float(r['y_true'])):.2f}px）" for r in worst)
result=f'''# 2026task3：实际训练结果与优化建议

已生成 `submission.csv`：20 张测试图、40 行、列名 `ImageID,value`。数值来自 2026-10-25 新训练的五折模型，每个测试坐标取五折等权平均。

## 本次实际运行

- 本机 RTX 4060 Laptop GPU；PyTorch 2.5.1+cu121；官方 ImageNet ResNet18 初始化。
- 80 张有标签图，79 个完全重复图像组；五折，每折 16 张外层验证图。
- 粗/细输入均为 256×256，热图均为 64×64；batch 8，AMP，随机种子 2026。
- 粗/细上限 40/30 轮，7 轮早停；内部比较固定与多尺度训练；共 40 个训练阶段。
- 真实执行耗时 {execution['elapsed_seconds']/60:.2f} 分钟，不含此前定位兼容性问题的失败运行、后续权重重载和结果检查。
- 重新加载全部 10 个最终网络，重新读取 20 张测试图，重算五折预测；与交付 CSV 最大坐标差 {reload_report['submission_max_abs_difference_px']:.9f}px（CSV 保留六位小数）。
- B4 在训练前指定。无测试标签、无排行榜反馈，不根据外层 OOF 结果临时改选提交方法。

## 六组消融结果

{table}

坐标 RMSE = sqrt(所有 80 张图的 dx²+dy² 之和 / 160)。欧氏距离的中位数、P90 与坐标 RMSE 是不同指标，不应混用。

B4 相对 B0 的配对组 bootstrap RMSE 差为 {float(b4['delta_rmse']):.4f}px，95% 区间 [{float(b4['delta_ci_low']):.4f}, {float(b4['delta_ci_high']):.4f}]。区间仅反映这批固定 OOF 预测的图像组重采样波动，不包含重训随机性和未知患者相关性。

这是内部验证，不是 Kaggle 线上成绩。附件原版未做同预算重新训练，因此不宣称本次相对附件原版有确定的分数提升。

## 各折实际训练选择

{fold_table}

策略和轮数仅由各折内部早停集决定。校准图像不进入两个最终网络训练。重复图像组会使开发/校准集大小略有变化。

## 从本次结果看到的后续重点

- 校准裁剪在外层验证上的最小尺寸限制触发率为 {sum(min_clamps)/80:.1%}；B3 与 C1 最大坐标差为 {max_difference:.6f}px。这可以判断“自适应”是否实际提供了不同于固定裁剪的行为。
- B4 实际回退比例为 {sum(fallbacks)/80:.1%}。回退未启用或未触发时，与 B3 相同是符合代码定义的结果。
- B4 最大的五个 OOF 欧氏误差样本：{worst_text}。这些是下一轮开发诊断对象，当前测试坐标没有据此手工修改。
- 优先在内部开发集比较 ROI 下限、上下文范围、细定位分辨率和训练数据利用率；再考虑更大编码器与多种子。任何改动都需要重新计算 OOF，而不是沿用本次分数。

## 文件使用

- `submission.csv`：可直接按本次比赛模板上传。
- `fovea_submission_20261025.ipynb`：Kaggle 原生 Notebook，包含兼容性修复和本次训练配置。添加比赛数据与已有 `3pretrained` Dataset、启用 GPU 后 Run All。
- `submission_validation.json`：格式、坐标范围、折划分、训练阶段及文件 SHA256 检查。
- `submission_reload_verification.json`：十个最终网络独立重载复算记录。
- `ablation_results_20261025.csv`：完整消融指标。
- `trained_submission_20261025_fixed/`：实际模型、训练日志、校准参数、每折 OOF 和测试预测，保留在本地输出目录。

下面是对附件代码的具体检查和进一步实验思路。

---

'''
notes=(ROOT/'work/current_notebook_review/review_notes.md').read_text(encoding='utf-8')
(ROOT/'outputs/optimization_report_20261025.md').write_text(result+notes,encoding='utf-8')
shutil.copyfile(RUN/'ablation_summary.csv',ROOT/'outputs/ablation_results_20261025.csv')
nbpath=ROOT/'outputs/fovea_submission_20261025.ipynb'
nb=js(nbpath)
marker='## 已完成的本地运行记录'
nb['cells']=[c for c in nb['cells'] if not ''.join(c.get('source',[])).startswith(marker)]
nb['cells'].append({'cell_type':'markdown','id':'actual-local-result-20261025','metadata':{},'source':(marker+'\n\n2026-10-25，RTX 4060，实际五折训练；下表为内部 OOF 指标，不是排行榜分数。\n\n'+table+'\n本 Notebook 的代码定义已与实际训练代码核对；重新执行时会重新训练并生成提交。\n').splitlines(keepends=True)})
nbpath.write_text(json.dumps(nb,ensure_ascii=False,indent=1),encoding='utf-8')
print(table)
print(fold_table)
print('ROI min clamp',sum(min_clamps)/80,'B3 vs C1 max difference',max_difference)
