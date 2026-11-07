"""Execute the delivered notebook definition/preflight cells in a fresh module."""
from pathlib import Path
import ast
import hashlib
import json
import os
import subprocess
import sys
import time
import types

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK = ROOT / "outputs/fovea_research_kaggle.ipynb"
REPORT = ROOT / "work/research_notebook_validation.json"
PREFLIGHT = ROOT / "work/notebook_preflight"
PREFLIGHT.mkdir(parents=True, exist_ok=True)
schema_python = "C:/Users/lenovo/AppData/Local/Programs/Python/Python312/python.exe"
schema = subprocess.run(
    [schema_python, "-c", "import nbformat,sys; nbformat.validate(nbformat.read(sys.argv[1],as_version=4)); print('PASS')", str(NOTEBOOK)],
    capture_output=True, text=True, check=True,
)
assert schema.stdout.strip() == "PASS"
nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
sys.dont_write_bytecode = True
# Test-only dependencies supply third-party packages, not project source modules.
sys.path = [str(ROOT / "work/kaggle_test_deps")] + [p for p in sys.path if p and Path(p).resolve() not in {ROOT, ROOT / "work", ROOT / "outputs"}]
os.environ["MPLBACKEND"] = "Agg"
os.chdir(PREFLIGHT)
module = types.ModuleType("fovea_notebook_standalone")
sys.modules[module.__name__] = module
namespace = module.__dict__
forbidden_imports = {"data_part", "model_training_part", "pipeline_part", "research_config_part",
                     "research_model_part", "research_calibration_part", "research_pipeline_part",
                     "research_weights_part", "fovea_kaggle", "fovea_research_kaggle", "work", "outputs"}
executed, skipped, imports = [], [], []
started = time.perf_counter()
for index, cell in enumerate(nb["cells"]):
    if cell["cell_type"] != "code":
        continue
    source = "".join(cell["source"])
    if "RESULTS = run_research" in source or 'RESULTS["summary"]' in source:
        skipped.append({"index": index, "reason": "training/results display explicitly excluded"})
        continue
    tree = ast.parse(source, filename=f"{NOTEBOOK.name}:cell_{index}")
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        else:
            continue
        assert all(name.split(".")[0] not in forbidden_imports for name in names), names
        imports.extend(names)
    if "print(research_geometry_selfcheck())" in source:
        namespace["CFG"].SMOKE_TEST = True
        namespace["CFG"].PRETRAINED_WEIGHTS_PATH = str(ROOT / "outputs/fovea_pretrained_weights.zip")
        namespace["CFG"].OUTPUT_DIR = str(PREFLIGHT)
        namespace["torch"].set_num_threads(2)
    exec(compile(source, f"{NOTEBOOK.name}:cell_{index}", "exec"), namespace, namespace)
    executed.append(index)

assert "RESULTS" not in namespace
assert not hasattr(module, "__file__")  # Notebook does not rely on a .py script location.
apis = ["ResearchConfig", "prepare_tables", "assign_folds", "resolve_research_weights",
        "research_geometry_selfcheck", "research_group_holdout", "fit_research_model",
        "predict_research", "fit_radius_calibration", "make_roi_spec", "fit_fallback_rule",
        "apply_fallback", "run_research", "write_submission", "research_uncertainty_evaluation",
        "paired_group_bootstrap"]
assert all(callable(namespace.get(name)) for name in apis)
cfg = namespace["CFG"]
assert cfg.N_FOLDS == 5 and cfg.SMOKE_TEST and cfg.PRETRAINED and not cfg.ALLOW_WEIGHT_DOWNLOAD
assert cfg.EXPERIMENTS == ("B0", "B1", "B2", "B3", "B4", "C1")
cache = PREFLIGHT / "pretrained_cache/resnet18-f37072fd.pth"
assert cache.is_file()
sha256 = hashlib.sha256(cache.read_bytes()).hexdigest()
assert sha256 == namespace["RESEARCH_WEIGHT_SHA256"]
assert all(name not in sys.modules for name in forbidden_imports)
report = {
    "status": "PASS", "notebook": str(NOTEBOOK),
    "notebook_sha256": hashlib.sha256(NOTEBOOK.read_bytes()).hexdigest(),
    "nbformat_schema": "PASS", "total_cells": len(nb["cells"]),
    "executed_code_cell_indices": executed, "executed_code_cells": len(executed),
    "skipped_code_cells": skipped, "fresh_namespace": True,
    "executed_in_separate_working_directory": str(PREFLIGHT),
    "depends_on_project_python_modules": False,
    "depends_on_original_notebook": False,
    "test_only_third_party_dependency_directory": str(ROOT / "work/kaggle_test_deps"),
    "official_pretrained_zip_loaded_offline": True,
    "weight_sha256": sha256, "preflight_geometry_checks": "PASS",
    "required_apis_present": apis, "default_outer_folds": cfg.N_FOLDS,
    "default_experiments": list(cfg.EXPERIMENTS),
    "training_executed": False,
    "scope": "Notebook code definitions and live preflight execute sequentially without project imports. Full training is covered by the separate real-data five-fold acceptance run.",
    "python": sys.version, "torch": str(namespace["torch"].__version__),
    "elapsed_seconds": time.perf_counter() - started,
}
REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
