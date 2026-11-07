from pathlib import Path
import json,ast,hashlib
ROOT=Path(__file__).resolve().parents[2]
nb_path=ROOT/'outputs/fovea_optimized_5rounds.ipynb'
py_path=ROOT/'outputs/fovea_optimized_5rounds.py'
nb=json.loads(nb_path.read_text(encoding='utf-8'))
script=py_path.read_text(encoding='utf-8')
def definitions(source):
    tree=ast.parse(source)
    return {node.name:ast.dump(node,include_attributes=False) for node in tree.body if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef))}
nb_defs={}
for cell in nb['cells']:
    if cell['cell_type']=='code':
        assert cell['outputs']==[] and cell['execution_count'] is None
        nb_defs.update(definitions(''.join(cell['source'])))
py_defs=definitions(script)
assert nb_defs==py_defs
source_defs={}
for part in (ROOT/'work/fovea_v3').glob('research_*_part.py'):
    source_defs.update(definitions(part.read_text(encoding='utf-8')))
assert all(py_defs.get(name)==node for name,node in source_defs.items())
schema='not_run'
try:
    import nbformat
    nbformat.validate(nbformat.read(nb_path,as_version=4));schema='PASS'
except ImportError:pass
report={'status':'PASS','matching_function_and_class_definitions':len(py_defs),'source_definitions_matched':len(source_defs),
        'cells':len(nb['cells']),'no_preexecuted_outputs':True,'nbformat_schema':schema,
        'notebook_sha256':hashlib.sha256(nb_path.read_bytes()).hexdigest(),
        'python_sha256':hashlib.sha256(py_path.read_bytes()).hexdigest()}
(ROOT/'work/five_round_tests/final_artifact_static_report.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
