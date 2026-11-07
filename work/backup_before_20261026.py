from pathlib import Path
import hashlib,json,zipfile,time
ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'backups'
DEST.mkdir(exist_ok=True)
stamp=time.strftime('%H%M%S')
target=DEST/f'before_optimization_20261026_{stamp}.zip'
skip={'gpu_test_deps','kaggle_test_deps','__pycache__','.pytest_cache'}
files=[]
for name in ('outputs','work'):
    files.extend((p,p.relative_to(ROOT).as_posix()) for p in (ROOT/name).rglob('*') if p.is_file() and not skip.intersection(p.parts))
downloads=Path(r'C:/Users/lenovo/Downloads')
for name in ['2026task3.zip','notebook3 (1).ipynb','notebook3.ipynb','notebook3.log','notebook3 (1).log','notebook860c18bd89.log','fovea_optimized_5rounds.ipynb','fovea_kaggle.ipynb']:
    p=downloads/name
    if p.is_file():files.append((p,'user_attachments/'+name))
manifest=[]
with zipfile.ZipFile(target,'x',zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as z:
    for i,(p,name) in enumerate(files):
        h=hashlib.sha256();size=0
        with p.open('rb') as src,z.open(name,'w',force_zip64=True) as dst:
            for chunk in iter(lambda:src.read(1024*1024),b''):
                h.update(chunk);size+=len(chunk);dst.write(chunk)
        manifest.append({'path':name,'bytes':size,'sha256':h.hexdigest()})
        if i%200==0:print('BACKUP',i+1,'/',len(files),flush=True)
    z.writestr('backup_manifest.json',json.dumps(manifest,ensure_ascii=False,indent=2))
with zipfile.ZipFile(target) as z:
    for i,entry in enumerate(manifest):
        h=hashlib.sha256()
        with z.open(entry['path']) as src:
            for chunk in iter(lambda:src.read(1024*1024),b''):h.update(chunk)
        assert h.hexdigest()==entry['sha256'],entry['path']
        if i%300==0:print('VERIFY',i+1,'/',len(manifest),flush=True)
report={'status':'PASS','backup_path':str(target),'file_count':len(manifest),'original_bytes':sum(e['bytes'] for e in manifest),
 'zip_bytes':target.stat().st_size,'every_file_sha256_verified':True,'excluded':'Regenerable dependency folders and Python caches only.',
 'scope':'All currently present outputs/work plus listed original user attachments. Previously deleted temporary checkpoints cannot be recovered.'}
(DEST/'backup_20261026_verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(report,ensure_ascii=True,indent=2),flush=True)
