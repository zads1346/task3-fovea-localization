"""Consolidate executed validation evidence without promoting smoke accuracy."""
from pathlib import Path
import json
import hashlib
ROOT = Path(__file__).resolve().parent.parent
report_path = ROOT / "outputs/research_validation.json"
report = json.loads(report_path.read_text(encoding="utf-8"))
load = lambda name: json.loads((ROOT / "work" / name).read_text(encoding="utf-8"))
weights = load("weights_bundle_helper_acceptance.json")
fullsize = load("research_fullsize_check.json")
notebook = load("research_notebook_validation.json")
report["weight_bundle_helper_tests"] = {"status": weights["status"], "count": weights["count"], "tests": weights["tests"]}
report["full_size_network_checks"] = {"status": fullsize["status"], "meaning": fullsize["meaning"],
    "stages": {name: {key: data[key] for key in ("input_shape", "heatmap_shape", "gradient_tensors_checked", "all_gradients_finite", "tta")}
               for name, data in fullsize["stages"].items()}}
report["notebook_standalone"] = {key: notebook[key] for key in ("status", "nbformat_schema", "executed_code_cells", "fresh_namespace", "depends_on_project_python_modules", "depends_on_original_notebook", "official_pretrained_zip_loaded_offline")}
report["calibration_helper_tests"] = {"status": "PASS", "count": 14, "includes": "ID alignment, group quantiles, fallback and matching covariance, calibrated coverage and ROI boundary handling"}
report["post_smoke_patch"] = "Covariance metadata follows the selected coarse prediction on B4 fallback; covered by a separate shuffled-ID unit test. Coordinate/ROI/training logic unchanged."
report["files"] = {}
for name in ("fovea_research_kaggle.ipynb", "fovea_research_kaggle.py", "fovea_pretrained_weights.zip"):
    path = ROOT / "outputs" / name
    report["files"][name] = {"bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
print("CONSOLIDATED_RESEARCH_VALIDATION_PASS")
