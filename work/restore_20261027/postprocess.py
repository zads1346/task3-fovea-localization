from pathlib import Path
import runpy,sys
WORK=Path(__file__).resolve().parent;ROOT=WORK.parents[1]
sys.path.insert(0,str(ROOT/'work/gpu_test_deps'))
assert (ROOT/'outputs/restored_20261027/seed_10000/execution_report.json').exists(), 'Training must finish successfully first.'
for name in ['combine.py','verify.py','finalize.py']:
    print('POSTPROCESS:',name,flush=True)
    runpy.run_path(str(WORK/name),run_name='__main__')
