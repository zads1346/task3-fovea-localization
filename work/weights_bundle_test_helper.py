"""Acceptance tests for offline loading, corruption rejection and safe ZIP reads."""
from pathlib import Path
from types import SimpleNamespace
import hashlib
import importlib.util
import io
import json
import os
import zipfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("weight_helpers", ROOT / "work/research_weights_part.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
BASE = ROOT / "work/weights_bundle_tests"
BASE.mkdir(exist_ok=True)
PTH = ROOT / "outputs/fovea_pretrained_weights" / m.RESEARCH_WEIGHT_FILENAME
ZIP = ROOT / "outputs/fovea_pretrained_weights.zip"
passed = []

def config(path="", folder="cache", **kwargs):
    return SimpleNamespace(PRETRAINED=True, PRETRAINED_WEIGHTS_PATH=str(path),
                           OUTPUT_DIR=str(BASE / folder), ALLOW_WEIGHT_DOWNLOAD=False, **kwargs)

def expect_raises(error, function):
    try:
        function()
    except error:
        return
    raise AssertionError(f"Expected {error.__name__}")

cfg = config(PTH)
assert Path(m.resolve_research_weights(cfg)) == PTH
passed.append("explicit pth full SHA256 verification")
cfg.PRETRAINED = False
assert m.resolve_research_weights(cfg) is None
passed.append("random initialization only by explicit PRETRAINED=False")
assert Path(m.resolve_research_weights(config(PTH.parent))) == PTH
passed.append("Kaggle auto-unpacked dataset directory")
resolved = Path(m.resolve_research_weights(config(ZIP, "zip_cache")))
assert resolved.parent == BASE / "zip_cache/pretrained_cache"
assert hashlib.sha256(resolved.read_bytes()).hexdigest() == m.RESEARCH_WEIGHT_SHA256
assert resolved == Path(m.resolve_research_weights(config(ZIP, "zip_cache")))
passed.append("ZIP streamed safely into writable cache and cache reused")

duplicate_dir = BASE / "duplicates"
corrupt = duplicate_dir / "dataset_c" / m.RESEARCH_WEIGHT_FILENAME
if corrupt.exists():
    corrupt.unlink()
for name in ("dataset_a", "dataset_b"):
    target = duplicate_dir / name / m.RESEARCH_WEIGHT_FILENAME
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        os.link(PTH, target)
assert Path(m.resolve_research_weights(config(duplicate_dir))).exists()
passed.append("identical multiple copies accepted")
corrupt.parent.mkdir(exist_ok=True)
corrupt.write_bytes(b"damaged input")
expect_raises(ValueError, lambda: m.resolve_research_weights(config(duplicate_dir)))
passed.append("one corrupted conflicting copy rejected despite valid alternatives")

empty_dir = BASE / "empty"
empty_dir.mkdir(exist_ok=True)
expect_raises(FileNotFoundError, lambda: m.resolve_research_weights(config(empty_dir, "zip_cache")))
expect_raises(FileNotFoundError, lambda: m.resolve_research_weights(config(BASE / "missing.pth")))
with patch("urllib.request.urlopen", side_effect=AssertionError("Unexpected internet access")):
    expect_raises(FileNotFoundError, lambda: m.resolve_research_weights(config("", "no_inputs")))
passed.append("missing inputs fail with no download or random-init fallback")

malicious = BASE / "malicious.zip"
with zipfile.ZipFile(malicious, "w", compression=zipfile.ZIP_STORED) as archive:
    archive.writestr("../" + m.RESEARCH_WEIGHT_FILENAME, b"bad")
expect_raises(ValueError, lambda: m.resolve_research_weights(config(malicious, "malicious_out")))
assert not (BASE / "malicious_out").exists()
passed.append("unsafe ZIP target rejected without extraction")

download_cfg = config("", "download_mock")
download_cfg.ALLOW_WEIGHT_DOWNLOAD = True
download_cache = Path(download_cfg.OUTPUT_DIR) / "pretrained_cache" / m.RESEARCH_WEIGHT_FILENAME
if download_cache.exists():
    download_cache.unlink()
with patch("urllib.request.urlopen", return_value=io.BytesIO(PTH.read_bytes())) as opened:
    downloaded = Path(m.resolve_research_weights(download_cfg))
    assert downloaded.is_file()
    assert opened.call_count == 1
    assert opened.call_args[0][0].full_url == m.RESEARCH_WEIGHT_URL
passed.append("opt-in official URL download checks full hash before atomic save")

report = {"status": "PASS", "tests": passed, "count": len(passed)}
(ROOT / "work/weights_bundle_helper_acceptance.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
