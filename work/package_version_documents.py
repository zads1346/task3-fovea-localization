"""Build two independent, byte-verified documentation bundles; never modify sources."""
from pathlib import Path
import csv, hashlib, io, json, zipfile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs'
NEW = OUT / 'restored_20261027'

def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()

def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))

old_check = read_json(OUT / 'submission_validation.json')
new_check = read_json(NEW / 'validation_report.json')
nb_check = read_json(NEW / 'notebook_validation.json')
assert sha_bytes((OUT / 'submission.csv').read_bytes()) == old_check['submission_sha256']
assert sha_bytes((OUT / 'fovea_submission_20261025.ipynb').read_bytes()) == old_check['notebook_sha256']
assert sha_bytes((NEW / 'submission_candidate.csv').read_bytes()) == new_check['new_submission_sha256']
assert sha_bytes((NEW / 'fovea_restored_ensemble_20261027.ipynb').read_bytes()) == nb_check['notebook_sha256']
old_rows = list(csv.DictReader(io.StringIO((OUT / 'submission.csv').read_text())))
new_rows = list(csv.DictReader(io.StringIO((NEW / 'submission_candidate.csv').read_text())))
assert len(old_rows) == len(new_rows) == 40
assert [r['ImageID'] for r in old_rows] == [r['ImageID'] for r in new_rows]
assert old_check['submission_sha256'] != new_check['new_submission_sha256']

old_files = {
    'submission.csv': OUT / 'submission.csv',
    'fovea_submission_20261025.ipynb': OUT / 'fovea_submission_20261025.ipynb',
    'docs/optimization_report_20261025.md': OUT / 'optimization_report_20261025.md',
    'results/ablation_results_20261025.csv': OUT / 'ablation_results_20261025.csv',
    'results/oof_B4.csv': OUT / 'trained_submission_20261025_fixed/oof_B4.csv',
    'results/test_ensemble_B4.csv': OUT / 'trained_submission_20261025_fixed/test_ensemble_B4.csv',
    'audits/submission_validation.json': OUT / 'submission_validation.json',
    'audits/submission_reload_verification.json': OUT / 'submission_reload_verification.json',
    'audits/test_leakage_audit.json': OUT / 'test_leakage_audit.json',
    'run_records/run_config.json': OUT / 'trained_submission_20261025_fixed/run_config.json',
    'run_records/execution_report.json': OUT / 'trained_submission_20261025_fixed/execution_report.json',
    'run_records/environment.json': OUT / 'trained_submission_20261025_fixed/environment.json',
}
new_files = {
    'submission_candidate.csv': NEW / 'submission_candidate.csv',
    'fovea_restored_ensemble_20261027.ipynb': NEW / 'fovea_restored_ensemble_20261027.ipynb',
    'docs/report.md': NEW / 'report.md',
    'docs/protocol.json': NEW / 'protocol.json',
    'results/comparison.csv': NEW / 'comparison.csv',
    'results/comparison.json': NEW / 'comparison.json',
    'results/fold_comparison.csv': NEW / 'fold_comparison.csv',
    'results/oof_candidate.csv': NEW / 'oof_candidate.csv',
    'results/oof_error_changes.csv': NEW / 'oof_error_changes.csv',
    'results/test_ensemble_candidate.csv': NEW / 'test_ensemble_candidate.csv',
    'results/test_prediction_changes.csv': NEW / 'test_prediction_changes.csv',
    'results/additional_seed_ablation_summary.csv': NEW / 'seed_10000/ablation_summary.csv',
    'audits/validation_report.json': NEW / 'validation_report.json',
    'audits/notebook_validation.json': NEW / 'notebook_validation.json',
    'audits/implementation_check.json': NEW / 'implementation_check.json',
    'run_records/baseline_run_config.json': OUT / 'trained_submission_20261025_fixed/run_config.json',
    'run_records/additional_seed_run_config.json': NEW / 'seed_10000/run_config.json',
    'run_records/additional_seed_execution_report.json': NEW / 'seed_10000/execution_report.json',
    'run_records/additional_seed_environment.json': NEW / 'seed_10000/environment.json',
}

old_readme = '''# 2026task3：8.9 原版文档包

本包只对应你反馈 Kaggle 成绩为 8.9 的原版；该成绩未由本次工具从 Kaggle 读取。

## 使用哪个文件

- 直接提交原来的预测：[submission.csv](submission.csv)。文件内容与原版完全一致，40 行，ImageID/value 两列。
- 在 Kaggle 重新训练：[fovea_submission_20261025.ipynb](fovea_submission_20261025.ipynb)。添加比赛数据和已有的 3pretrained Dataset，启用 GPU 后运行全部单元。
- 实验说明：[优化报告](docs/optimization_report_20261025.md)。

原版采用 ResNet18、256/256 输入、五折、粗到细定位、内部早停、独立校准和 B4 推理。内部 OOF 坐标 RMSE 为 32.0267，这不是线上 8.9 分数。

## 内容与范围

results/ 保存消融结果、80 张训练图的 OOF 预测和测试集五折平均预测。
audits/ 保存格式、权重重载复算和测试集泄漏审计。
run_records/ 保存当时的配置、运行耗时和环境记录。记录中的本机绝对路径仅用于追溯；在 Kaggle 上运行根目录的 Notebook。

不包含比赛原始图片、ImageNet 预训练权重或训练后的模型权重。Notebook 需要比赛数据和 3pretrained 权重 Dataset；不同 GPU/软件版本重新训练可能产生数值差异。要使用原样的 8.9 预测，直接提交本包 CSV。

历史报告中提到的原工作目录和其他交付 ZIP 是历史记录，不代表本包还嵌套包含这些文件。manifest.json 列出了本包实际文件及 SHA256。
'''

new_readme = '''# 2026task3：75/25 改进候选文档包

本包只对应 2026-10-27 完成的改进候选。它的 Kaggle 成绩尚未获得；不能把 8.9 当成本候选的成绩。

## 使用哪个文件

- 提交本次候选：[submission_candidate.csv](submission_candidate.csv)。这是本包唯一的主提交 CSV，40 行，ImageID/value 两列。
- Kaggle 原生完整代码：[fovea_restored_ensemble_20261027.ipynb](fovea_restored_ensemble_20261027.ipynb)。添加比赛数据和已有的 3pretrained Dataset，启用 GPU 后运行全部单元。
- 完整说明与核验：[实验报告](docs/report.md)。

候选预测 = 8.9 原版预测 × 75% + 新随机种子五折预测 × 25%。配比在训练前固定。
网络、256/256 分辨率、裁剪设置、外层五折和全部内部数据分组保持旧版流程；新增训练阶段的模型、采样和增强种子偏移 10000。
本地实际复用了原版五折模型，只新增训练 40 个阶段。Notebook 从 ImageNet 初始化重新训练两组，共 80 个阶段，输出 /kaggle/working/submission.csv（融合候选）。

本地 OOF RMSE 为 28.7624，原版为 32.0267，五折均改善；这些是多轮开发中的内部验证结果，并不保证线上优于 8.9。
候选与旧版 40 个坐标都不同，测试预测平均移动约 1.89 像素。全部 20 个最终网络已重载复算通过。

## 内容与范围

results/ 保存总体、逐折、逐图对照，候选 OOF 与测试预测，以及新增种子的消融结果。
audits/ 保存输入隔离、权重重载、Notebook 和实现核验。
run_records/ 同时保存原版与新增种子的实际配置；本机绝对路径仅用于追溯，不是 Kaggle 运行路径。

不包含比赛原始图片、ImageNet 预训练权重或训练后的模型权重。运行 Notebook 需要比赛数据和 3pretrained 权重 Dataset；重新训练受环境差异影响，不保证逐位复现本包 CSV。

8.9 原版 CSV 在独立的 2026task3_8p9_documents.zip 中。本包不放另一份原版主提交，避免误交文件。报告中的 submission_restored_8p9.csv、备份路径和旧 delivery_20261027.zip 为历史追溯信息；以本包 README 和 manifest.json 为实际目录说明。
'''

def make_package(filename, folder, files, readme, version):
    destination = OUT / filename
    entries = {'README.md': readme.encode('utf-8')}
    manifest = {'version': version, 'package_date': '2026-10-27', 'files': [],
                'trained_weights_included': False, 'raw_images_included': False}
    for name, path in files.items():
        assert path.is_file(), path
        entries[name] = path.read_bytes()
    for name, data in entries.items():
        manifest['files'].append({'file': name, 'bytes': len(data), 'sha256': sha_bytes(data),
            'source': files[name].relative_to(ROOT).as_posix() if name in files else 'Generated package README'})
    entries['manifest.json'] = json.dumps(manifest, ensure_ascii=False, indent=2).encode('utf-8')
    with zipfile.ZipFile(destination, 'x', zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, data in entries.items():
            archive.writestr(folder + '/' + name, data)
    with zipfile.ZipFile(destination) as archive:
        assert archive.testzip() is None
        assert len(archive.namelist()) == len(set(archive.namelist())) == len(entries)
        for name, data in entries.items():
            assert archive.read(folder + '/' + name) == data, name
        for entry in manifest['files']:
            assert sha_bytes(archive.read(folder + '/' + entry['file'])) == entry['sha256']
    return {'path': str(destination), 'version': version, 'files_including_manifest': len(entries),
            'bytes': destination.stat().st_size, 'sha256': sha_bytes(destination.read_bytes()),
            'crc_check': 'PASS', 'all_member_bytes_and_sha256': 'PASS'}

results = [
    make_package('2026task3_8p9_documents.zip', '2026task3_8p9', old_files, old_readme, 'baseline_8p9'),
    make_package('2026task3_candidate_75_25_documents.zip', '2026task3_candidate_75_25', new_files, new_readme, 'candidate_75_25'),
]
(OUT / 'version_documents_packaging_verification.json').write_text(
    json.dumps({'status': 'PASS', 'packages': results}, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(results, ensure_ascii=False, indent=2))
