"""Verify frozen data bytes before selecting or executing tasks."""
import hashlib
import json
from pathlib import Path
from exp_core.settings import DATA


def validate_files(data_dir=DATA):
    root = Path(data_dir)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    verified = {}
    for dataset, definition in manifest["datasets"].items():
        path = root / f"{dataset}-tasks.json"
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != definition["task_file_sha256"]:
            raise ValueError(f"{path.name}: sha256 does not match frozen manifest")
        tasks = json.loads(path.read_text())
        ids = [task["id"] for task in tasks]
        if len(ids) != len(set(ids)):
            raise ValueError(f"{dataset}: duplicate task ids")
        dev, heldout = set(definition["dev"]), set(definition["heldout"])
        if dev & heldout or not (dev | heldout) <= set(ids):
            raise ValueError(f"{dataset}: invalid development/heldout split")
        verified[dataset] = {"tasks": len(tasks), "sha256": actual}
    return {"datasets": verified,
            "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest()}
