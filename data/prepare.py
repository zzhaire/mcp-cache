"""Freeze small task sets before collecting any model scores."""
import hashlib
import json
from pathlib import Path
import requests
import bfcl_eval

from exp_core.settings import ROOT, DATA, SCRATCH, APPWORLD_ROOT
TEMP = SCRATCH
REV = "60e37c7a19672769a6034c45a5c8b36e7cd3768b"


def ordered(values):
    return sorted(values, key=lambda x: hashlib.sha256(
        ("20261116:" + str(x)).encode()).hexdigest())


def write(name, value):
    (DATA / name).write_text(json.dumps(value, indent=2))


def main():
    DATA.mkdir(exist_ok=True)
    manifest = {"seed": 20261116, "models": ["GPT-6.1-Sol", "DeepSeek-V4.1-Flash"],
                "selection": "sha256(seed:id), without performance filtering",
                "datasets": {}}
    base = Path(bfcl_eval.__file__).parent / "data"
    entries = [json.loads(x) for x in (base / "BFCL_v4_multi_turn_base.json").read_text().splitlines()]
    answers = {x["id"]: x for x in map(json.loads, (base / "possible_answer" /
               "BFCL_v4_multi_turn_base.json").read_text().splitlines())}
    selected = sorted(entries, key=lambda x: ordered_key(x["id"]))[:20]
    write("bfcl-tasks.json", selected)
    write("bfcl-oracle.json", {x["id"]: answers[x["id"]] for x in selected})
    manifest["datasets"]["bfcl"] = {
        "version": "bfcl-eval==2026.3.23 / BFCL_v4_multi_turn_base",
        "dev": [x["id"] for x in selected[:10]],
        "heldout": [x["id"] for x in selected[10:]],
        "status": "data_and_official_oracle_available"}
    appbase = APPWORLD_ROOT / "data"
    ids = (appbase / "datasets/train.txt").read_text().splitlines()
    groups = {}
    for task in ordered(ids):
        groups.setdefault(task.split("_")[0], task)
    chosen = ordered(list(groups.values()))[:20]
    write("appworld-tasks.json", [
        {"id": x, "spec": json.loads((appbase / "tasks" / x / "specs.json").read_text())}
        for x in chosen])
    manifest["datasets"]["appworld"] = {
        "version": "appworld==0.1.3.post1 / data-0.1.0 / train",
        "selection_group": "one task per generator; generators do not cross splits",
        "dev": chosen[:10], "heldout": chosen[10:], "status": "environment_preflight_passed"}
    url = f"https://huggingface.co/datasets/HuggingFaceH4/tau2-bench-data/resolve/{REV}/domains/telecom/tasks.json"
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    entries = response.json()
    selected = sorted(entries, key=lambda x: ordered_key(x["id"]))[:20]
    write("tau2-tasks.json", selected)
    manifest["datasets"]["tau2"] = {
        "source": url, "revision": REV, "domain": "telecom",
        "dev": [x["id"] for x in selected[:10]],
        "heldout": [x["id"] for x in selected[10:]],
        "status": "data_only_official_executor_blocked_by_github_auth"}
    for dataset in manifest["datasets"]:
        path = DATA / f"{dataset}-tasks.json"
        manifest["datasets"][dataset]["task_file_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    write("manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


def ordered_key(value):
    return hashlib.sha256(("20261116:" + str(value)).encode()).hexdigest()


if __name__ == "__main__":
    main()
