from pathlib import Path
import csv,json,math
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/research82_precision_20261027'
def read(p):
 with Path(p).open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))
stages=list(OUT.rglob('stage_complete.json'));folds=[]
for marker in stages:
 folder=marker.parent
 info=json.loads((folder/'training_summary.json').read_text())
 assert info['successful_optimizer_steps']>0
 assert all(math.isfinite(float(row['train_loss'])) for row in read(folder/'history.csv'))
 fold_folder=OUT/marker.relative_to(OUT).parts[0]
 roles=read(fold_folder/'split_roles.csv')
 allowed={r['image_id'] for r in roles if r['role'] in ('model_fit','early_stop')}
 actual=read(folder/'actual_training_inputs.csv')
 assert {r['image_id'] for r in actual}<=allowed
 assert len(actual)==info['training_images']
 stopfile=folder/'actual_early_stop_inputs.csv'
 if stopfile.exists():
  stop=read(stopfile)
  assert {r['image_id'] for r in stop}<=allowed
  assert not {r['group'] for r in actual}&{r['group'] for r in stop}
for k in range(5):
 folder=OUT/f'fold_{k}'
 if not (folder/'complete.json').exists():continue
 sel=json.loads((folder/'selection.json').read_text())
 metrics={r['method']:float(r['rmse']) for r in read(folder/'metrics.csv')}
 folds.append({'fold':k+1,'selected':sel['selected'],'accepted':sel['accepted'],
               'baseline_rmse':metrics['BASELINE8P2'],'candidate_rmse':metrics['GUARDED20']})
status={'completed_stages':len(stages),'total_stages':15,'completed_stage_input_loss_checks':'PASS','folds':folds}
(OUT/'live_progress.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
print(json.dumps(status,indent=2))
