"""Commands for the shared experiment bed; model work starts only on execution."""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys

from exp_core.methods import ARM_NAMES, ARM_ALIASES, describe_methods, resolve_arm
from exp_core.settings import ROOT, CONFIGS, DATA, RESULTS

DELEGATES = {
    "task": "exp_core.runner", "report": "exp_core.report",
    "check": "exp_core.checks", "preflight": "exp_core.preflight",
    "rsi": "ours.rsi", "prepare": "data.prepare", "select": "data.select_cases",
    "lab-check": "exp_core.lab.selfcheck",
    "adapt": "ours.adapt",
    "lab-rejudge": "exp_core.lab.rejudge",
    "lab-report": "exp_core.lab.report",
}
# Commands whose module exposes a function of that name instead of ``main``.
FUNCTIONS = {"recon-build": ("data.recon.check", "build"),
             "recon-check": ("data.recon.check", "check"),
             "recon-report": ("data.recon.check", "report"),
             "long-build": ("data.lab_long", "main"),
             "gain-build": ("data.lab_gain", "main")}


def profile_parser(profile):
    parser = argparse.ArgumentParser(prog=f"run {profile}")
    parser.add_argument("--config", type=Path, default=CONFIGS / f"{profile}.json")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--methods", nargs="+", choices=[*ARM_NAMES, *ARM_ALIASES])
    parser.add_argument("--arms", action="append", choices=[*ARM_NAMES, "fixed"])
    parser.add_argument("--cases", action="append")
    parser.add_argument("--split", choices=["dev", "heldout", "all"])
    parser.add_argument("--model")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--delay-ms", type=float)
    parser.add_argument("--jitter-ms", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--ours-policy-file", type=Path)
    parser.add_argument("--scheduler",
                        help="M1–M4 scheduler preset or JSON file for the ours arm")
    parser.add_argument("--shadow-verify", action="store_true",
                        help="evaluation only: report the stale-hit rate of cache hits")
    parser.add_argument("--lab-listing", choices=["per_turn", "per_connection"])
    parser.add_argument("--lab-contract", choices=["full", "annotations", "none", "wrong"])
    parser.add_argument("--repeats", type=int, default=1,
                        help="run everything this many times, each with its own seed, "
                             "into <results-root>/rep<N>")
    parser.add_argument("--results-root", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def run_profile(profile, argv):
    args = profile_parser(profile).parse_args(argv)
    config = json.loads(args.config.read_text())
    manifest_path = args.manifest or ROOT / config["manifest"]
    document = json.loads(manifest_path.read_text())
    if document.get("status") == "unregistered":
        raise ValueError(f"{profile} manifest is unregistered: {document.get('reason', '')}")
    from data.integrity import validate_files
    integrity = validate_files()
    if document.get("manifest_sha256") not in (None, integrity["manifest_sha256"]):
        raise ValueError("suite manifest_sha256 does not match frozen data manifest")
    methods = args.methods or args.arms or config["methods"]
    if profile == "full":
        pending = [item["name"] for item in describe_methods()
                   if item["method_id"] in {resolve_arm(name)["arm"] for name in methods}
                   and item["reproduction_status"] == "partial_adaptation"]
        if pending:
            raise ValueError(f"formal baseline reproduction pending: {', '.join(pending)}")
    execution = {
        key: getattr(args, key) if getattr(args, key) is not None else config[key]
        for key in ("model", "max_steps", "delay_ms", "jitter_ms", "seed", "workers")}
    # Non-default lab factors are part of the run configuration (and its hash).
    for key in ("lab_listing", "lab_contract"):
        if getattr(args, key):
            execution[key] = getattr(args, key)
    results_root = args.results_root or RESULTS / profile
    if args.repeats > 1:
        # Independent repetitions: same cases and methods, seed + n, own folder.
        status = 0
        for repeat in range(args.repeats):
            again = [a for a in argv]
            for flag in ("--repeats", "--seed", "--results-root"):
                while flag in again:
                    at = again.index(flag)
                    del again[at:at + 2]
            status = run_profile(profile, again + [
                "--seed", str(execution["seed"] + repeat),
                "--results-root", str(Path(results_root) / f"rep{repeat}")]) or status
        return status
    forwarded = []
    for key, value in execution.items():
        forwarded.extend(["--" + key.replace("_", "-"), str(value)])
    forwarded.extend(["--split", args.split or config["split"],
                      "--results-root", str(results_root)])
    for method in methods:
        forwarded.extend(["--arms", resolve_arm(method)["arm"]])
    for case in args.cases or []:
        forwarded.extend(["--cases", case])
    if args.limit is not None:
        forwarded.extend(["--limit", str(args.limit)])
    if args.ours_policy_file:
        forwarded.extend(["--ours-policy-file", str(args.ours_policy_file)])
    if args.scheduler:
        forwarded.extend(["--scheduler", args.scheduler])
    if args.shadow_verify:
        forwarded.append("--shadow-verify")
    if args.dry_run:
        forwarded.append("--dry-run")
    from exp_core import matrix
    return matrix.main(
        forwarded, document=document, execution_settings=execution,
        suite_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        profile=profile)


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        prog="run", description="MCP 实验：统一入口、独立方法、固定数据与可恢复执行")
    parser.add_argument("command", nargs="?", choices=[
        "list", "validate", "smoke", "full", "test", *DELEGATES, *FUNCTIONS])
    if not arguments or arguments[0] in ("-h", "--help"):
        parser.print_help()
        return 0
    command, rest = arguments[0], arguments[1:]
    try:
        if command == "list":
            print(json.dumps({"methods": describe_methods()}, ensure_ascii=False, indent=2))
            return 0
        if command == "validate":
            check = argparse.ArgumentParser(prog="run validate")
            check.add_argument("--data-dir", type=Path, default=DATA)
            args = check.parse_args(rest)
            from data.integrity import validate_files
            result = validate_files(args.data_dir)
            from exp_core.matrix import validate_cases
            document = json.loads((args.data_dir / "scenarios-36.json").read_text())
            validate_cases(document, data_dir=args.data_dir)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0
        if command in ("smoke", "full"):
            return run_profile(command, rest)
        if command == "test":
            return subprocess.call(
                [sys.executable, "-m", "unittest", "discover",
                 "-s", str(ROOT / "tests"), *rest], cwd=ROOT)
        if command in FUNCTIONS:
            module, function = FUNCTIONS[command]
            return getattr(importlib.import_module(module), function)(rest) or 0
        if command in DELEGATES:
            module = importlib.import_module(DELEGATES[command])
            # Legacy helpers parse sys.argv internally.
            sys.argv = [f"run {command}", *rest]
            return module.main() or 0
        parser.error(f"unknown command: {command}")
    except (OSError, ValueError, KeyError) as error:
        print(f"experiment validation failed: {error}", file=sys.stderr)
        return 2
