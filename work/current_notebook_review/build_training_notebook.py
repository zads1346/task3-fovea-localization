from pathlib import Path
import ast
import hashlib
import json

ROOT = Path(__file__).resolve().parents[2]
source = ROOT / 'outputs/fovea_optimized_5rounds.ipynb'
nb = json.loads(source.read_text(encoding='utf-8'))
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        cell['execution_count'] = None
        cell['outputs'] = []
        text = ''.join(cell['source'])
        for flag in ('crop_from_prediction', 'crop_was_enlarged', 'initial_crop_miss'):
            text = text.replace(f'torch.tensor({flag})', f'torch.tensor(bool({flag}), dtype=torch.bool)')
        if 'from __future__ import annotations' in text:
            text = text.replace('import os\n', 'import os\nos.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")\n', 1)
        if 'class ResearchConfig(Config):' in text:
            text += '''
# Configuration used for the newly trained submission dated 2026-10-25.
# These are real training settings, not SMOKE_TEST settings.
CFG = ResearchConfig(
    OUTPUT_DIR="/kaggle/working/fovea_submission_20261025",
    SUBMISSION_PATH="/kaggle/working/submission.csv",
    INPUT_SIZE=256, HEATMAP_SIZE=64,
    FINE_INPUT_SIZE=256, FINE_HEATMAP_SIZE=64,
    COARSE_EPOCHS=40, FINE_EPOCHS=30, PATIENCE=7,
    BATCH_SIZE=8, NUM_WORKERS=0, AMP=True,
    N_FOLDS=5, SMOKE_TEST=False, REQUIRE_GPU=True,
    SELECT_FINE_CROP_POLICY=True,
    BACKBONE_LR_MULTIPLIER=1.0, USE_EMA=False,
    SUBMISSION_EXPERIMENT="B4", BOOTSTRAP_REPEATS=1000,
)
torch.set_num_threads(2)
'''
        ast.parse(text)
        cell['source'] = text.splitlines(keepends=True)

intro = '''# 2026task3：五折训练与 submission.csv

本文件对应 2026-10-25 实际训练的计算预算：粗定位 256×256、细定位 256×256，最多 40/30 轮，7 轮早停。
从用户提供的 2.0 版升级到已审查的 3.0 训练流程，保留 ResNet18、热图 + DSNT、独立校准和六组消融。

1. 导入本 Notebook，添加 2026task3 比赛数据和已有 `3pretrained` 权重 Dataset。
2. 在 Kaggle 中启用 GPU；Internet 可以关闭。运行全部代码单元。
3. 最终文件：`/kaggle/working/submission.csv`。结果、每折模型和校准参数位于 `fovea_submission_20261025/`。

五折中，重复图像按组划分；每折只用内部早停集选轮数和固定/多尺度训练策略。校准图像不参与网络训练和早停。
最终 B4 方法提前固定，不按外层 OOF 分数临时挑选。保留六组结果用于诊断。

这是可重新训练的 Notebook，代码单元没有伪造运行输出。随交付包提供的 CSV 来自本地 RTX 4060 实际训练；不同 GPU、PyTorch 版本和数值库可能产生不同结果。
本次没有同预算重训旧版，因此不能把新模型验证分数表述为相对旧版的已证实提升。
'''
nb['cells'][0]['source'] = intro.splitlines(keepends=True)
for cell in nb['cells']:
    if cell['cell_type'] == 'markdown' and '配置与依赖' in ''.join(cell['source']):
        cell['source'] = '''## 配置与依赖

下方最后一次 `CFG = ResearchConfig(...)` 是本次实际配置，会覆盖类中的通用默认值。
默认五折，每折八个训练阶段（粗模型选轮、三个粗模型交叉拟合、两种细模型选轮、最终粗/细模型），共四十个阶段。
`SMOKE_TEST=False`，输入 256/256，训练上限 40/30，早停耐心 7，batch 8。
已修正重训学习率计划、TTA 早停与推理一致性，并增加尺度选择、保守回退和阶段完整性校验。
可修改分辨率和预算继续实验；修改后使用新输出目录，不能把原配置的阶段缓存混用。
权重自动查找已有 `3pretrained` Dataset 中的 `resnet18-f37072fd.pth`，并验证 SHA256。
这份 ImageNet 权重仅用于初始化；中心凹模型需要本文件实际训练后得到。
'''.splitlines(keepends=True)

nb['metadata']['kaggle'] = {'isInternetEnabled': False, 'isGpuEnabled': True, 'language': 'python'}
dest = ROOT / 'outputs/fovea_submission_20261025.ipynb'
dest.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding='utf-8')
import nbformat
nbformat.validate(nbformat.read(dest, as_version=4))
print(dest)
print('cells', len(nb['cells']), 'sha256', hashlib.sha256(dest.read_bytes()).hexdigest())
