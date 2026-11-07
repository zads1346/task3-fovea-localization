from pathlib import Path
import ast,hashlib,json
ROOT=Path(__file__).resolve().parents[2];WORK=Path(__file__).resolve().parent
parts=[ROOT/'work/restore_20261027/engine.py',WORK/'fine_variants.py',WORK/'pipeline.py']
source='\n\n'.join(p.read_text(encoding='utf-8') for p in parts)
compile(source,'research82_engine','exec');ast.parse(source)
(WORK/'engine.py').write_text(source,encoding='utf-8')
report={'engine_sha256':hashlib.sha256((WORK/'engine.py').read_bytes()).hexdigest(),
 'source_parts':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in parts}}
(WORK/'source_manifest.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
