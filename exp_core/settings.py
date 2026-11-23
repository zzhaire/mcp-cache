"""Repository-relative inputs and explicitly configured external runtime paths."""
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CONFIGS = ROOT / "configs"
RESULTS = ROOT / "results"
ARCHIVE = ROOT / "archive" / "mcp-smoke"
CONTRACTS = ROOT / "exp_core" / "contracts"


def load_environment():
    path = Path(os.environ.get("MCP_EXPERIMENT_ENV",
                               CONFIGS / "environment.local.json"))
    value = json.loads(path.read_text()) if path.is_file() else {}
    if not isinstance(value, dict):
        raise ValueError(f"{path}: environment must be a JSON object")
    return value


ENVIRONMENT = load_environment()
SCRATCH = Path(ENVIRONMENT.get(
    "work_dir", Path(tempfile.gettempdir()) / "mcp-experiments"))
# traex (original runs), codex or claude; MCP_MODEL_BACKEND overrides the file
# so one environment can calibrate several backends without editing it.
MODEL_BACKEND = os.environ.get("MCP_MODEL_BACKEND") or ENVIRONMENT.get("model_backend", "traex")
_CLI_KEY = "model_cli" if MODEL_BACKEND == "traex" else f"{MODEL_BACKEND}_cli"
MODEL_CLI = (ENVIRONMENT.get(_CLI_KEY) or shutil.which(MODEL_BACKEND) or MODEL_BACKEND)
MODEL_EFFORT = os.environ.get("MCP_MODEL_EFFORT") or ENVIRONMENT.get("model_effort")
MODEL_CONCURRENCY = int(ENVIRONMENT.get("model_concurrency", 8))
APPWORLD_PYTHON = ENVIRONMENT.get("appworld_python", sys.executable)
TAU2_PYTHON = ENVIRONMENT.get("tau2_python", sys.executable)
APPWORLD_ROOT = Path(ENVIRONMENT.get("appworld_root", SCRATCH / "appworld"))
TAU2_ROOT = Path(ENVIRONMENT.get("tau2_root", SCRATCH / "tau2-bench"))
