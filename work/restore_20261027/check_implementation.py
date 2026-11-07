"""Check the controlled change and submission alignment without training another model."""
from pathlib import Path
import ast,json,tempfile,importlib.util,sys
ROOT=Path(__file__).resolve().parents[2];WORK=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'work/gpu_test_deps'))
import pandas as pd,numpy as np
def definitions(path):
 return {n.name:n for n in ast.parse(path.read_text(encoding='utf-8')).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
old=definitions(ROOT/'work/current_notebook_review/submission_engine.py');new=definitions(WORK/'engine.py')
assert old.keys()==new.keys()
changed=[k for k in old if ast.dump(old[k],include_attributes=False)!=ast.dump(new[k],include_attributes=False)]
assert set(changed)=={'ResearchConfig','_research_stage'},changed
spec=importlib.util.spec_from_file_location('combine_check',WORK/'combine.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
OLD=ROOT/'outputs/trained_submission_20261025_fixed'
with tempfile.TemporaryDirectory(dir=WORK,prefix='alignment_check_') as temp:
 temp=Path(temp);second=temp/'second';second.mkdir()
 for name in ['oof_B4.csv','test_ensemble_B4.csv']:
  frame=pd.read_csv(OLD/name,dtype={'image_id':str,'group':str})
  frame['x']+=4;frame['y']-=8
  frame.iloc[::-1].to_csv(second/name,index=False)
 m.combine_runs(OLD,second,temp/'result',ROOT/'outputs/submission.csv',temp/'candidate.csv')
 before=pd.read_csv(ROOT/'outputs/submission.csv');after=pd.read_csv(temp/'candidate.csv')
 expected=np.where(before.ImageID.str.endswith('_X'),1.,-2.)
 assert np.max(np.abs(after.value-before.value-expected))<2e-6
 # A mismatched OOF fold must be rejected instead of silently mixing in-sample predictions.
 frame=pd.read_csv(second/'oof_B4.csv');frame.loc[0,'fold']=(frame.loc[0,'fold']+1)%5
 frame.to_csv(second/'oof_B4.csv',index=False)
 try:m.combine_runs(OLD,second,temp/'bad',ROOT/'outputs/submission.csv',temp/'bad.csv')
 except AssertionError:pass
 else:raise AssertionError('Fold mismatch was accepted')
report={'status':'PASS','changed_definitions':changed,'all_model_data_split_loss_inference_functions_unchanged':True,
 'coordinate_ID_alignment_with_shuffled_rows':'PASS','mismatched_OOF_fold_rejected':'PASS',
 'synthetic_test_outputs':'Temporary directory automatically removed; never used as submissions.'}
(ROOT/'outputs/restored_20261027/implementation_check.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
