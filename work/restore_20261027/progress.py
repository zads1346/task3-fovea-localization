from pathlib import Path
import csv,json,math
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/restored_20261027/seed_10000'
def read(p):
 with p.open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
rows=[]
for f in range(5):
 p=OUT/f'fold_{f}'
 if not (p/'complete.json').exists():continue
 old=read(ROOT/f'outputs/trained_submission_20261025_fixed/fold_{f}/B4_oof.csv')
 new={r['image_id']:r for r in read(p/'B4_oof.csv')}
 errors={k:[] for k in ('baseline','additional','candidate')}
 for a in old:
  b=new[a['image_id']]
  for col in ('x','y'):
   truth=float(a[col+'_true']);x=float(a[col]);y=float(b[col])
   for k,v in [('baseline',x),('additional',y),('candidate',.75*x+.25*y)]:errors[k].append((v-truth)**2)
 rows.append({'fold':f,**{k:round(math.sqrt(sum(e)/len(e)),4) for k,e in errors.items()}})
stages=list(OUT.rglob('stage_complete.json'))
for marker in stages:
 folder=marker.parent
 info=json.loads((folder/'training_summary.json').read_text())
 assert info['successful_optimizer_steps']>0
 assert all(math.isfinite(float(r['train_loss'])) for r in read(folder/'history.csv'))
 actual=read(folder/'actual_training_inputs.csv')
 assert len(actual)==info['training_images']
 fold_dir=OUT/marker.relative_to(OUT).parts[0]
 roles=read(fold_dir/'split_roles.csv')
 allowed={r['image_id'] for r in roles if r['role'] in ('model_fit','early_stop')}
 assert {r['image_id'] for r in actual}<=allowed
 stop_file=folder/'actual_early_stop_inputs.csv'
 if stop_file.exists():
  stop=read(stop_file)
  assert {r['image_id'] for r in stop}<=allowed
  assert not {r['group'] for r in actual}&{r['group'] for r in stop}
print(json.dumps({'completed_folds':len(rows),'completed_stages':len(stages),
 'completed_stage_input_and_finite_loss_checks':'PASS','partial_fold_rmse':rows},indent=2))
