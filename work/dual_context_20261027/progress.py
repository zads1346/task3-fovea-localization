"""Read-only progress of the current isolated experiment."""
from pathlib import Path
import csv,json,math,sys
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/dual_context_20261027'
def read(p):
    with Path(p).open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))
stages=list(OUT.rglob('stage_complete.json'));folds=[]
for marker in stages:
    folder=marker.parent
    info=json.loads((folder/'training_summary.json').read_text())
    assert info['successful_optimizer_steps']>0
    assert all(math.isfinite(float(row['train_loss'])) for row in read(folder/'history.csv'))
    assert len(read(folder/'actual_training_inputs.csv'))==info['training_images']
for seed in (2026,2027):
    for k in range(5):
        folder=OUT/f'split_{seed}'/f'fold_{k}'
        if not (folder/'complete.json').exists():continue
        selection=json.loads((folder/'selection.json').read_text())
        rule=json.loads((folder/'risk_rule.json').read_text())
        scores={r['method']:float(r['rmse']) for r in read(folder/'metrics.csv')}
        folds.append({'split_seed':seed,'fold':k+1,'accepted':selection['accepted'],
                     'risk_enabled':rule['enabled'],'scores':scores})
status={'completed_stages':len(stages),'total_new_stages':65,'completed_stage_checks':'PASS','completed_folds':folds}
(OUT/'live_progress.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
shown = ({'completed_stages':len(stages),'total_new_stages':65,
          'completed_stage_checks':'PASS','completed_folds':len(folds),
          'latest_fold':folds[-1] if folds else None} if '--brief' in sys.argv else status)
print(json.dumps(shown,indent=2))
