from pathlib import Path
import json, hashlib
ROOT=Path(__file__).resolve().parent.parent
TEST=ROOT/'work/five_round_tests';OUT=ROOT/'outputs'
def read(name):return json.loads((TEST/name).read_text(encoding='utf-8'))
rounds=[read(f'round_{i:02d}/evidence.json') for i in range(1,4)]
cal=read('round4_calibration_report.json')
stage=read('stage_recovery_report.json')
final=read('final_notebook_report.json')
if 'epochs_per_training_job' in final:
    final['max_epochs_per_training_job']=final.pop('epochs_per_training_job')
full=read('final_fullsize_report.json')
dev=read('development_results_summary.json')
assert all(item['status']=='PASS' for item in rounds+[stage,final,full])
assert cal['failures']==cal['errors']==0
assert stage['source_sha256']==final['source_sha256']
summary={
    'version':'3.0.0','status':'PASS','completed_optimization_rounds':5,
    'rounds':[
        {'round':1,'change':'Fixed schedule horizon; common TTA validation/inference decoder','status':'PASS','evidence':rounds[0]},
        {'round':2,'change':'Prediction-centred multiscale crops and inner multiscale stopping; final pipeline compares fixed/multiscale on inner splits','status':'PASS','evidence':rounds[1]},
        {'round':3,'change':'Differential LR and optional warm-start EMA; diagnostics; conservative defaults retained','status':'PASS','evidence':rounds[2]},
        {'round':4,'change':'Minimum calibration group/gain guards; ROI degeneration diagnostics','status':'PASS','evidence':cal},
        {'round':5,'change':'Completed-stage recovery, integrity manifests, self-contained notebook, full five-fold execution','status':'PASS','evidence':stage}],
    'development_comparisons':dev,
    'final_notebook_execution':final,'full_resolution_functional_check':full,
    'release_defaults':{'outer_folds':5,'coarse_input':512,'fine_input':384,'coarse_max_epochs':50,
                        'fine_max_epochs':40,'backbone_lr_multiplier':1.0,'ema':False,
                        'select_fine_crop_policy_on_inner_only':True,'training_jobs':40,
                        'submission_experiment':'B4','smoke_test':False},
    'source_sha256':final['source_sha256'],
    'limitations':['No full-resolution full-budget five-fold training or Kaggle leaderboard evaluation was completed.',
                   'Development comparison: one seed, 38 fitting and 10 reused stopping images, 128px, six epochs.',
                   'Final five-fold execution: CPU, 128px, at most two epochs per training job; execution acceptance only.',
                   'Protocol was developed with this dataset; results are internal validation, not independent external validation.',
                   'No patient IDs; exact-image duplicate grouping only.',
                   'Windows GPU was detected but full-model training failed memory-allocation checks; no GPU speed/accuracy claim.']}
(OUT/'five_round_validation.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
report='''## 五轮优化与实际验证记录

已完成五轮代码改进、问题检查和真实数据验证。没有完成正式分辨率、完整预算的 Kaggle 五折训练，也没有排行榜成绩；不能称为“五轮均提分”。

| 轮次 | 发现与处理 | 验证 |
|---|---|---|
| 1 | 重训压缩余弦学习率周期；早停没有使用最终TTA | 重训保留原日程前缀；早停与推理RMSE一致 |
| 2 | OOF细模型训练尺度单一 | 多尺度变换与标签隔离通过；发现短预算负结果，最终改为每折内部比较固定/多尺度 |
| 3 | 增加小样本优化候选与可诊断训练 | 分组学习率、EMA数值/跳步/BN/重载检查；记录热图及坐标损失 |
| 4 | 少量校准图上易因微小改善选择回退 | 至少12组、至少1px及1%改善、P90约束；26项测试通过，报告ROI钳制与尺度退化 |
| 5 | 中断后重训整折、损坏CSV可被误用 | 阶段/整折SHA256、必需文件清单、代码/环境签名；原生Notebook五折执行、篡改与恢复测试通过 |

### 真实开发对照：包括负结果

固定38张拟合图、10张早停图、128输入、6轮、同一个种子；没有评价对应校准集和外层留出集。下列分数是重复使用的早停集结果，只能作开发诊断。

| 对照 | 早停RMSE（原图像素） |
|---|---:|
| 粗模型原版／第一轮修复后 | 64.836／64.836 |
| 粗模型编码器LR倍率1.0，无EMA | 64.836 |
| 粗模型编码器LR倍率0.25，无EMA | 61.419 |
| 粗模型编码器LR倍率0.25，加EMA | 63.473 |
| 细模型固定尺度训练 | 42.267 |
| 细模型多尺度训练 | 45.335 |

粗模型采用TTA RMSE；细模型采用三个预设裁剪尺度的TTA RMSE均值。两类指标不能直接当作同一项模型排名。细模型训练中心来自拟合集内部三折粗模型预测。

**据此采取的处理：**没有直接强开EMA或宣称多尺度必然更好。默认编码器LR倍率仍为1.0、EMA关闭；0.25作为可配置候选。固定/多尺度训练在每个外层折内重新比较，只有该折早停集决定策略及轮数。外层验证和校准组不参与选择。默认每折8个训练任务，五折40个，六组定位消融共享网络。

### 最终验收范围

- 从原生 `.ipynb` 的独立命名空间执行全部代码单元，不导入项目Python文件；离线权重ZIP完整校验。
- 真实80张训练图、20张测试图；五折、40个真实训练任务、六组消融全部完成。验收采用CPU、128输入、每阶段最多2轮。
- 每种方法80条唯一OOF预测、40行提交格式；39/77重复图保持同折并保留原标签。
- 校准组、内部早停组、外层验证组分离；OOF训练中心不由训练过该目标组的模型产生。
- 完整折重载、缺失清单条目、CSV/模型篡改、参数/代码修改检测通过；模拟折中断后复用全部已完成训练阶段，模型文件未重训。
- 另完成正式512/384尺寸前向、反向、TTA和150种边界几何检查；这不是正式训练。

交付默认 `SMOKE_TEST=False`，粗/细输入512/384、最多50/40轮、五折、默认B4输出。正式运行在 `/kaggle/working/submission.csv` 生成提交文件。已有 `3pretrained` 可继续使用。

### 结果边界

本地GPU能识别，但整网训练未通过内存分配检查，因此验收使用CPU。没有把测试输出冒充正式submission，也没有承诺排行榜提升。实验方案已经结合本数据开发，五折仍属于内部验证；无独立外部数据、无患者ID，不能声称患者层面或跨域泛化。完整机器可读记录见随包 `five_round_validation.json`。
'''
(OUT/'five_round_report.md').write_text(report,encoding='utf-8')
print('Five completed rounds documented from actual evidence.')
