"""Download and validate the official torchvision ResNet18 ImageNet-1K weights."""
from pathlib import Path
import datetime
import hashlib
import json
import shutil
import sys
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parent.parent
BUNDLE_DIR = ROOT / "outputs/fovea_pretrained_weights"
ZIP_PATH = ROOT / "outputs/fovea_pretrained_weights.zip"
URL = "https://download.pytorch.org/models/resnet18-f37072fd.pth"
DOCS = "https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html"
SOURCE = "https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py"
LICENSE_URL = "https://raw.githubusercontent.com/pytorch/vision/main/LICENSE"

def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def download(url, target):
    temporary = target.with_suffix(target.suffix + ".partial")
    with urllib.request.urlopen(url, timeout=60) as response, temporary.open("wb") as out:
        shutil.copyfileobj(response, out, length=1024 * 1024)
    temporary.replace(target)

def main():
    BUNDLE_DIR.mkdir(parents=True, exist_ok=True)
    weights = BUNDLE_DIR / "resnet18-f37072fd.pth"
    if not weights.exists():
        print("Downloading official ImageNet-1K weights...", flush=True)
        download(URL, weights)
    sha256 = digest(weights)
    if not sha256.startswith("f37072fd"):
        raise RuntimeError(f"Official torchvision filename hash mismatch: {sha256}")
    sys.path.insert(0, str(ROOT / "work/kaggle_test_deps"))
    import torch
    import torchvision
    from torchvision.models import resnet18, ResNet18_Weights
    assert ResNet18_Weights.IMAGENET1K_V1.url == URL
    state = torch.load(weights, map_location="cpu", weights_only=True)
    model = resnet18(weights=None)
    result = model.load_state_dict(state, strict=True)
    assert not result.missing_keys and not result.unexpected_keys
    torch.set_num_threads(2)
    model.eval()
    with torch.inference_mode():
        prediction = model(torch.zeros(1, 3, 224, 224))
    assert tuple(prediction.shape) == (1, 1000)
    assert torch.isfinite(prediction).all()
    license_path = BUNDLE_DIR / "LICENSE-torchvision.txt"
    download(LICENSE_URL, license_path)
    manifest = {
        "bundle_name": "fovea_pretrained_weights",
        "model": "torchvision.models.resnet18",
        "weights_enum": "ResNet18_Weights.IMAGENET1K_V1",
        "training_dataset": "ImageNet-1K",
        "purpose": "Encoder initialization only; not a fovea-trained or competition-trained model.",
        "filename": weights.name,
        "size_bytes": weights.stat().st_size,
        "sha256": sha256,
        "source_url": URL,
        "official_documentation": DOCS,
        "official_source_code": SOURCE,
        "source_license_url": LICENSE_URL,
        "license_note": "The included BSD-3-Clause license is torchvision's software license. Pretrained models may be subject to terms for the datasets used to train them; this bundle does not relicense the weights or ImageNet data.",
        "downloaded_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "validation": {
            "torch": torch.__version__, "torchvision": torchvision.__version__,
            "weights_only_load": True, "strict_state_dict_load": True,
            "finite_forward_shape": [1, 1000],
        },
    }
    (BUNDLE_DIR / "weights_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (BUNDLE_DIR / "README.txt").write_text(
        "Fovea research notebook: offline ResNet18 pretrained encoder weights\n"
        "================================================================\n\n"
        "These are the original, unmodified torchvision ResNet18 IMAGENET1K_V1\n"
        "weights, trained on ImageNet-1K. They are not trained for fovea localization.\n"
        "The notebook trains its coarse and refinement models on your competition data.\n\n"
        "Kaggle setup:\n"
        "1. Create a Kaggle Dataset by uploading fovea_pretrained_weights.zip.\n"
        "2. Add that Dataset to your Kaggle Notebook using Add Input.\n"
        "3. Add the 2026task3 competition data as a separate input.\n"
        "4. Leave PRETRAINED=True and ALLOW_WEIGHT_DOWNLOAD=False.\n"
        "   The notebook finds either the extracted .pth or the original ZIP under\n"
        "   /kaggle/input and checks the complete SHA256 before loading.\n"
        "5. If needed, set PRETRAINED_WEIGHTS_PATH to the .pth, ZIP, or dataset folder.\n\n"
        f"Weight SHA256: {sha256}\n"
        f"Weight bytes: {weights.stat().st_size}\n"
        f"Official download: {URL}\n"
        f"Documentation: {DOCS}\n"
        f"Source: {SOURCE}\n\n"
        "License/source notes:\n"
        "LICENSE-torchvision.txt reproduces the torchvision repository's BSD-3-Clause\n"
        "software license. The pretrained weights may also be subject to terms of\n"
        "their training data. This bundle does not relicense weights or ImageNet data.\n"
        "See https://github.com/pytorch/vision#models for the upstream model notice.\n",
        encoding="utf-8",
    )
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(BUNDLE_DIR.iterdir()):
            archive.write(path, arcname=path.name)
    report = {"zip_path": str(ZIP_PATH), "zip_size_bytes": ZIP_PATH.stat().st_size,
              "zip_sha256": digest(ZIP_PATH), "weight_manifest": manifest,
              "zip_members": [p.name for p in sorted(BUNDLE_DIR.iterdir())]}
    (ROOT / "work/weights_bundle_validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
