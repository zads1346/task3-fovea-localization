from pathlib import Path
import json,ast,hashlib,nbformat
ROOT=Path(__file__).resolve().parents[2];WORK=Path(__file__).resolve().parent;OUT=ROOT/'outputs/restored_20261027'
source=(WORK/'engine.py').read_text(encoding='utf-8')
source=source.replace('import os\n','import os\nos.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")\n',1)
combine=(WORK/'combine.py').read_text(encoding='utf-8').split("if __name__=='__main__':")[0]
intro='''# 2026task3：恢复旧版，小范围细定位 + 双随机种子集成

用户反馈：旧版 Kaggle 分数 8.9，新版约 12.4，越低越好。本文件恢复旧版 256/256、B4、独立校准、内部选轮与裁剪策略。

唯一主要实验变量是模型随机种子；外层五折、内部早停、独立校准及交叉拟合分组均保持不变。第二组模型的初始化、采样和增强种子统一加 10000。
候选坐标固定为旧版 75% + 第二组 25%，不根据外层验证或测试结果挑融合权重。此候选尚无 Kaggle 成绩，不能保证优于 8.9。

添加比赛数据和 `3pretrained` 权重 Dataset，启用 GPU，然后 Run All。代码使用官方 ImageNet ResNet18 权重，不需要 Internet。
会真实训练两组各五折，共 80 个阶段；本地旧模型没有嵌入 Notebook。运行时间随 GPU 而变，库版本也可能影响重新训练结果。

- `/kaggle/working/submission_restored.csv`：重新训练的旧方法。
- `/kaggle/working/submission.csv`：固定 75/25 融合候选。
- 原来实际取得 8.9 的 CSV 单独交付为 `submission_restored_8p9.csv`；重训不保证逐位复现。

每组内部校准保留隔离；融合后的坐标不沿用单模型覆盖率承诺。外层 OOF 已多次用于开发，不能当作全新独立测试。
'''
cfg=json.loads((ROOT/'outputs/trained_submission_20261025_fixed/run_config.json').read_text())['config']
for key in ['DATA_DIR','LABEL_CSV','SAMPLE_CSV','PRETRAINED_WEIGHTS_PATH']:cfg[key]=None
cfg['INPUT_ROOT']='/kaggle/input';cfg['OUTPUT_DIR']='/kaggle/working/restored_seed0';cfg['SUBMISSION_PATH']='/kaggle/working/submission_restored.csv'
cfg['MODEL_SEED_OFFSET']=0
config_code='torch.set_num_threads(2)\nCFG = ResearchConfig(**'+repr(cfg)+')\nprint(research_geometry_selfcheck())\nprint(research_preflight(CFG))\n'
train_code='''BASELINE = run_research(CFG)
CFG_NEW = replace(CFG, MODEL_SEED_OFFSET=10000,
    OUTPUT_DIR="/kaggle/working/restored_seed10000",
    SUBMISSION_PATH="/kaggle/working/submission_seed10000.csv")
ADDITIONAL = run_research(CFG_NEW)
'''
end='''COMPARISON = combine_runs(CFG.OUTPUT_DIR, CFG_NEW.OUTPUT_DIR,
    "/kaggle/working/restored_comparison", CFG.SUBMISSION_PATH,
    "/kaggle/working/submission.csv")
display(COMPARISON)
display(pd.read_csv("/kaggle/working/submission.csv").head(10))
print("Candidate (not yet scored): /kaggle/working/submission.csv")
'''
cells=[nbformat.v4.new_markdown_cell(intro),nbformat.v4.new_code_cell(source),
       nbformat.v4.new_code_cell(config_code),nbformat.v4.new_code_cell(train_code),
       nbformat.v4.new_code_cell(combine),nbformat.v4.new_code_cell(end)]
nb=nbformat.v4.new_notebook(cells=cells,metadata={'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'},
    'language_info':{'name':'python'},'kaggle':{'isInternetEnabled':False,'isGpuEnabled':True,'language':'python'}})
for cell in cells:
 if cell.cell_type=='code':compile(cell.source,'cell','exec')
nbformat.validate(nb)
path=OUT/'fovea_restored_ensemble_20261027.ipynb';nbformat.write(nb,path)
definitions=lambda t:{n.name:ast.dump(n,include_attributes=False) for n in ast.parse(t).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
assert definitions(source)==definitions((WORK/'engine.py').read_text(encoding='utf-8'))
(OUT/'notebook_validation.json').write_text(json.dumps({'status':'PASS','cells':len(cells),'engine_definitions':len(definitions(source)),
    'notebook_sha256':hashlib.sha256(path.read_bytes()).hexdigest()},indent=2),encoding='utf-8')
print(path)
