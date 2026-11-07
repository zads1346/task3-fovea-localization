from pathlib import Path
import urllib.request,hashlib,json,shutil,zipfile
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/research82_20261027/pretrained_weights'
OUT.mkdir(parents=True,exist_ok=True)
name='efficientnet_b0_rwightman-7f5810bc.pth';url='https://download.pytorch.org/models/'+name
dest=OUT/name
if not dest.exists():
 with urllib.request.urlopen(url,timeout=60) as response, dest.with_suffix('.partial').open('wb') as f:
  shutil.copyfileobj(response,f,1024**2)
 partial=dest.with_suffix('.partial');digest=hashlib.sha256(partial.read_bytes()).hexdigest()
 assert digest.startswith('7f5810bc'),digest
 partial.replace(dest)
digest=hashlib.sha256(dest.read_bytes()).hexdigest();assert digest.startswith('7f5810bc')
resnet=ROOT/'outputs/fovea_pretrained_weights/resnet18-f37072fd.pth'
shutil.copy2(resnet,OUT/resnet.name)
shutil.copy2(ROOT/'outputs/fovea_pretrained_weights/LICENSE-torchvision.txt',OUT/'LICENSE-torchvision.txt')
manifest={'origin':'Official Torchvision ImageNet-1K weights, not retinal/fovea supervised weights.',
 'weights':[{'filename':name,'url':url,'sha256':digest,'bytes':dest.stat().st_size},
 {'filename':resnet.name,'url':'https://download.pytorch.org/models/'+resnet.name,
 'sha256':hashlib.sha256(resnet.read_bytes()).hexdigest(),'bytes':resnet.stat().st_size}]}
(OUT/'weights_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
(OUT/'README.txt').write_text('Official ImageNet initialization only. Upload this folder or the ZIP as a Kaggle Dataset. Attach alongside competition data. These are not trained fovea models.\n',encoding='utf-8')
target=OUT.parent/'research82_pretrained_weights.zip'
with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as z:
 for p in OUT.iterdir():z.write(p,p.name)
with zipfile.ZipFile(target) as z:
 assert z.testzip() is None
 for row in manifest['weights']:assert hashlib.sha256(z.read(row['filename'])).hexdigest()==row['sha256']
print(json.dumps(manifest,indent=2),flush=True)
