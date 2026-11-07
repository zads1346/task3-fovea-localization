from pathlib import Path
import ast,json,hashlib
import nbformat
ROOT=Path(__file__).resolve().parents[2]
source=Path(__file__).with_name('engine.py').read_text(encoding='utf-8')
source=source.replace('import os\n','import os\nos.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")\n',1)
split=source.index('@dataclass\nclass OptimizedConfig')
base=source[:split];new=source[split:]
intro='''# 2026task3：384 分辨率、多尺度细定位与五折融合

本版本对应 2026-10-26 的本地 GPU 实验。旧文件已独立备份，新结果使用新目录。

- ResNet18 ImageNet 初始化，粗/细输入 384×384，热图 96×96。
- 五折，完全重复图像按组放在同一折。每折内部留出早停集，再使用该外层训练折全部 64 张图重新拟合。
- 粗模型最多 35 轮、从第 10 轮起选择 checkpoint；细模型最多 30 轮、从第 8 轮起选择 checkpoint；早停 patience=10。
- 细训练中心由内部交叉预测产生；训练与推理均覆盖 0.20/0.30/0.40 三种视野尺度，另加统一边距。
- 使用水平翻转 TTA。最终坐标 = 0.5 × 粗预测 + 0.5 × 三尺度细预测均值，再对五折取平均。
- 编码器学习率为解码器的 0.25 倍，AMP 初始 loss scale 为 1024。
- 本版不做独立不确定性校准或阈值回退，不作覆盖率保证；因此可以将原来的校准样本用于网络拟合。

## Kaggle 使用

添加比赛数据和原有 `3pretrained` Dataset，启用 GPU。运行全部单元，输出 `/kaggle/working/submission.csv`。
Internet 可以关闭；官方 ResNet18 权重会自动查找并验证 SHA256。

使用了既往验证结果指导本轮设计，因此本轮 OOF 是开发过程中的内部比较，不能当作全新独立测试。
代码单元没有伪造训练输出；本地完成的结果另附报告。GPU/库版本差异可能导致重新训练的数值不同。
'''
cells=[nbformat.v4.new_markdown_cell(intro),nbformat.v4.new_markdown_cell('## 1. 数据、网络与训练函数'),nbformat.v4.new_code_cell(base),
 nbformat.v4.new_markdown_cell('## 2. 本次配置与五折执行流程\n\n实际配置由下方 `OptimizedConfig` 和 `CFG` 决定，覆盖上方历史通用默认值。'),
 nbformat.v4.new_code_cell(new),nbformat.v4.new_markdown_cell('## 3. GPU 检查与正式训练'),
 nbformat.v4.new_code_cell('torch.set_num_threads(2)\nprint(research_geometry_selfcheck())\nassert torch.cuda.is_available(), "请在 Kaggle 设置中启用 GPU"\nprint(torch.cuda.get_device_name(0))\nRESULTS = run_optimized(CFG)\n'),
 nbformat.v4.new_markdown_cell('## 4. 查看输出'),
 nbformat.v4.new_code_cell('display(RESULTS["summary"])\nprint("Submission:", RESULTS["config"].SUBMISSION_PATH)\ndisplay(pd.read_csv(RESULTS["config"].SUBMISSION_PATH).head(10))\n')]
nb=nbformat.v4.new_notebook(cells=cells,metadata={'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'},
 'language_info':{'name':'python'},'kaggle':{'isInternetEnabled':False,'isGpuEnabled':True,'language':'python'}})
nbformat.validate(nb)
for c in cells:
 if c.cell_type=='code':compile(c.source,'cell','exec')
def defs(text):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(text).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
expected=defs(Path(__file__).with_name('engine.py').read_text(encoding='utf-8'));actual={}
for c in cells:
 if c.cell_type=='code':actual.update(defs(c.source))
assert expected==actual
out=ROOT/'outputs/optimized_20261026';out.mkdir(exist_ok=True)
path=out/'fovea_optimized_20261026.ipynb';nbformat.write(nb,path)
(out/'notebook_static_validation.json').write_text(json.dumps({'status':'PASS','matched_definitions':len(expected),'cells':len(cells),
 'notebook_sha256':hashlib.sha256(path.read_bytes()).hexdigest()},indent=2),encoding='utf-8')
print(path,len(expected),'matched definitions')
