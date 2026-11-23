"""Resumable 36-case x 7-arm smoke matrix executor.

A task identity is ``(case_id, arm, policy_sha256, code_sha256)``. Any terminal
result (completed or error) under ``results/matrix`` is skipped and never
overwritten, so interrupted matrices resume without rerunning finished runs.
"""
import argparse
import fcntl
from concurrent.futures import ThreadPoolExecutor
import json
import sys
import threading
import uuid
from pathlib import Path

from exp_core import runner
from exp_core.methods import ARM_NAMES, EXTRA_ARMS, policy_sha256, resolve_arm
from exp_core.scheduler_policy import apply_to_arm
from exp_core.lab.factors import CONTRACT_LEVELS, LISTINGS
from exp_core.settings import ROOT, DATA, RESULTS as RESULTS_ROOT

SCENARIOS = DATA / "scenarios-36.json"
RESULTS = RESULTS_ROOT / "smoke"
IDENTITY_FIELDS = ("case_id", "arm", "policy_sha256", "code_sha256",
                   "run_config_sha256")
SUMMARY_NAME = "matrix-summary.jsonl"
TASK_FILES = {
    "appworld": "appworld-tasks.json",
    "bfcl": "bfcl-tasks.json",
    "tau2": "tau2-tasks.json",
    "lab": "lab-tasks.json",
}
TERMINAL = {"completed", "error"}
SPLITS = ("dev", "heldout")
TOTAL_CASES = 36
TOTAL_CELLS = 9
CELL_CASES = 4


def load_document(path=SCENARIOS):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_lab_cases(cases, data_dir=DATA):
    """Lab suite: any number of cases, each naming a task of the lab server."""
    if not isinstance(cases, list) or not cases:
        raise ValueError("lab suite has no cases")
    known = {task["id"] for task in json.loads(
        (Path(data_dir) / TASK_FILES["lab"]).read_text(encoding="utf-8"))}
    heldout = Path(data_dir) / "lab-heldout-tasks.json"
    if heldout.is_file():
        known |= {task["id"] for task in json.loads(heldout.read_text(encoding="utf-8"))}
    for extra in sorted([*Path(data_dir).glob("recon-*-tasks.json"),  # rebuilt benchmarks
                         *Path(data_dir).glob("lab-long-tasks.json")]):
        known |= {task["id"] for task in json.loads(extra.read_text(encoding="utf-8"))}
    seen = set()
    for case in cases:
        for field in ("case_id", "dataset", "task_id", "scenario", "split", "augmented"):
            if field not in case:
                raise ValueError(f"{case.get('case_id')}: missing field {field}")
        if case["dataset"] != "lab" or case["task_id"] not in known:
            raise ValueError(f"{case['case_id']}: not a task of the lab server")
        if case["split"] not in SPLITS:
            raise ValueError(f"{case['case_id']}: unknown split {case['split']}")
        if case["case_id"] in seen:
            raise ValueError(f"duplicate case_id {case['case_id']}")
        seen.add(case["case_id"])
    return cases


def validate_cases(document, data_dir=DATA):
    """Reject the matrix before any run if the pre-registration is incomplete."""
    cases = document.get("cases")
    if document.get("suite") == "lab":
        return validate_lab_cases(cases, data_dir)
    if not isinstance(cases, list) or len(cases) != TOTAL_CASES:
        raise ValueError(f"expected {TOTAL_CASES} cases, found "
                         f"{len(cases) if isinstance(cases, list) else 'none'}")
    case_ids = [case.get("case_id") for case in cases]
    if any(not isinstance(case_id, str) or not case_id for case_id in case_ids):
        raise ValueError("every case needs a non-empty case_id")
    if len(set(case_ids)) != len(case_ids):
        raise ValueError("duplicate case_id in scenarios")

    cells = document.get("cells")
    if not isinstance(cells, list) or len(cells) != TOTAL_CELLS:
        raise ValueError(f"expected {TOTAL_CELLS} cells, found "
                         f"{len(cells) if isinstance(cells, list) else 'none'}")

    by_cell = {}
    for case in cases:
        for field in ("case_id", "dataset", "task_id", "scenario", "split", "augmented"):
            if field not in case:
                raise ValueError(f"{case.get('case_id')}: missing field {field}")
        if case["split"] not in SPLITS:
            raise ValueError(f"{case['case_id']}: unknown split {case['split']}")
        if not isinstance(case["augmented"], bool):
            raise ValueError(f"{case['case_id']}: augmented must be boolean")
        if case["augmented"] and not (isinstance(case.get("augmentation"), str)
                                      and case["augmentation"].strip()):
            raise ValueError(f"{case['case_id']}: augmented case needs augmentation text")
        if not case["augmented"] and case.get("augmentation") is not None:
            raise ValueError(f"{case['case_id']}: non-augmented case has augmentation text")
        by_cell.setdefault((case["dataset"], case["scenario"]), []).append(case)
    if len(by_cell) != TOTAL_CELLS:
        raise ValueError(f"expected {TOTAL_CELLS} dataset x scenario cells, "
                         f"found {len(by_cell)}")
    for key, cell_cases in by_cell.items():
        if len(cell_cases) != CELL_CASES:
            raise ValueError(f"cell {key[0]}/{key[1]} has {len(cell_cases)} cases, "
                             f"expected {CELL_CASES}")
    declared = {(cell.get("dataset"), cell.get("scenario")) for cell in cells}
    if declared != set(by_cell):
        raise ValueError("cells do not match the case dataset x scenario pairs")

    tasks = {}
    for dataset in sorted({case["dataset"] for case in cases}):
        if dataset not in TASK_FILES:
            raise ValueError(f"unknown dataset {dataset}")
        payload = json.loads((Path(data_dir) / TASK_FILES[dataset]).read_text(
            encoding="utf-8"))
        tasks[dataset] = {task["id"] for task in payload}
    for case in cases:
        if case["task_id"] not in tasks[case["dataset"]]:
            raise ValueError(f"{case['case_id']}: task_id {case['task_id']} is not "
                             f"in {TASK_FILES[case['dataset']]}")
    return cases


def build_jobs(cases, arms=None, *, split="all", ours_policy=None,
               results_root=RESULTS, code_sha=None, execution_settings=None,
               suite_sha256=None):
    """Build de-duplicated case x arm jobs with their resume identity."""
    if split not in SPLITS + ("all",):
        raise ValueError(f"unknown split {split}")
    requested = list(arms) if arms else list(ARM_NAMES)
    code_sha = code_sha or runner.code_sha256(ROOT)
    execution = execution_settings or {
        "model": "DeepSeek-V4.1-Flash", "max_steps": 16, "delay_ms": 20,
        "jitter_ms": 5, "seed": 20261116, "workers": 1}
    run_config_sha256 = policy_sha256({
        "execution": execution,
        "suite_sha256": suite_sha256 or policy_sha256(cases)})
    results_root = Path(results_root)
    jobs, seen = [], set()
    for case in cases:
        if split != "all" and case["split"] != split:
            continue
        for name in requested:
            policy = resolve_arm(name)
            if policy["arm"] == "ours" and ours_policy is not None:
                # The policy file may only override Ours; never its arm label.
                policy.update({key: value for key, value in ours_policy.items()
                               if key != "arm"})
            if policy.get("scheduler"):
                # Hash the resolved scheduler content, never just a preset name.
                apply_to_arm(policy)
            digest = policy_sha256(policy)
            identity = (case["case_id"], policy["arm"], digest, code_sha,
                        run_config_sha256)
            if identity in seen:
                continue
            seen.add(identity)
            jobs.append({
                "case_id": case["case_id"],
                "dataset": case["dataset"],
                "task_id": case["task_id"],
                "scenario": case["scenario"],
                "split": case["split"],
                "augmented": case["augmented"],
                "augmentation": case.get("augmentation"),
                "method": policy["arm"],
                "arm": policy["arm"],
                "policy": policy,
                "policy_sha256": digest,
                "code_sha256": code_sha,
                "run_config_sha256": run_config_sha256,
                "dataset_sha256": case.get("task_file_sha256"),
                "suite_sha256": suite_sha256,
                "output_dir": results_root / case["case_id"] / policy["arm"],
                "identity": identity,
            })
    return jobs


def completed_identities(results_root=RESULTS):
    """Terminal identities already persisted under a results root."""
    root = Path(results_root)
    done = set()
    if not root.is_dir():
        return done
    for path in root.rglob("*.result.json"):
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if row.get("status") not in TERMINAL:
            continue
        identity = tuple(row.get(field) for field in IDENTITY_FIELDS)
        if all(isinstance(value, str) and value for value in identity):
            done.add(identity)
    return done


def pending_jobs(jobs, completed):
    return [job for job in jobs if job["identity"] not in completed]


def find_result(job, results_root=None):
    """Return the persisted row matching a job identity, if one exists."""
    root = Path(results_root) if results_root is not None else job["output_dir"]
    if not root.is_dir():
        return None
    for path in root.rglob("*.result.json"):
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        identity = tuple(row.get(field) for field in IDENTITY_FIELDS)
        if identity == job["identity"]:
            return row
    return None


def summary_row(job, row):
    cache = row.get("server_cache")
    hits = cache.get("hits") if isinstance(cache, dict) else None
    return {
        "case_id": job["case_id"],
        "dataset": job["dataset"],
        "task_id": job["task_id"],
        "scenario": job["scenario"],
        "split": row.get("split") or job["split"],
        "augmented": job["augmented"],
        "arm": job["arm"],
        "policy_sha256": job["policy_sha256"],
        "code_sha256": job["code_sha256"],
        "run_config_sha256": job["run_config_sha256"],
        "suite_sha256": job.get("suite_sha256"),
        "dataset_sha256": job.get("dataset_sha256"),
        "method_info": row.get("method_info"),
        "run_id": row.get("run_id"),
        "status": row.get("status"),
        "task_success": row.get("task_success"),
        "active_time_s": row.get("active_time_s"),
        "input_tokens": row.get("input_tokens"),
        "output_tokens": row.get("output_tokens"),
        "mcp_requests": row.get("mcp_requests"),
        "server_cache_hits": hits,
        "error": row.get("error"),
    }


def _error_result(job, error):
    """Persist a terminal infrastructure error without touching older results."""
    existing = find_result(job)
    if existing is not None:
        return existing
    job["output_dir"].mkdir(parents=True, exist_ok=True)
    run_id = uuid.uuid4().hex
    row = {
        "run_id": run_id,
        "case_id": job["case_id"],
        "dataset": job["dataset"],
        "task_id": job["task_id"],
        "scenario": job["scenario"],
        "augmented": job["augmented"],
        "arm": job["arm"],
        "method": job["method"],
        "split": job["split"],
        "policy": job["policy"],
        "policy_sha256": job["policy_sha256"],
        "code_sha256": job["code_sha256"],
        "run_config_sha256": job["run_config_sha256"],
        "suite_sha256": job.get("suite_sha256"),
        "dataset_sha256": job.get("dataset_sha256"),
        "status": "error",
        "error": str(error),
        "error_type": type(error).__name__,
        "task_success": None,
        "active_time_s": None,
        "input_tokens": 0,
        "output_tokens": 0,
        "mcp_requests": 0,
        "server_cache": {"hits": None, "error": "run did not start"},
    }
    (job["output_dir"] / f"{run_id}.result.json").write_text(
        json.dumps(row, indent=2, default=str), encoding="utf-8")
    return row


def run_matrix(jobs, args, results_root=RESULTS, summary_path=None):
    results_root = Path(results_root)
    summary_path = Path(summary_path or results_root / SUMMARY_NAME)
    results_root.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    total = len(jobs)

    def execute(index, job):
        try:
            row = runner.run_task(
                job["dataset"], job["task_id"], args.model, job["method"],
                job["output_dir"], max_steps=args.max_steps, policy=job["policy"],
                delay_ms=args.delay_ms, jitter_ms=args.jitter_ms, seed=args.seed,
                case_id=job["case_id"], scenario=job["scenario"],
                augmented=job["augmented"], case_split=job["split"],
                augmentation=job.get("augmentation"),
                run_config_sha256=job["run_config_sha256"],
                dataset_sha256=job.get("dataset_sha256"),
                suite_sha256=job.get("suite_sha256"),
                lab_options={"listing": args.lab_listing, "contract": args.lab_contract,
                             "audit": args.shadow_verify})
        except Exception as error:  # one broken run must not stop the matrix
            row = _error_result(job, error)
        line = summary_row(job, row)
        with lock:
            with summary_path.open("a", encoding="utf-8") as output:
                output.write(json.dumps(line, ensure_ascii=False) + "\n")
        print(f"[{index}/{total}] {job['case_id']} {job['arm']} {line['status']}",
              flush=True)
        return line

    if args.workers <= 1:
        for index, job in enumerate(jobs, 1):
            execute(index, job)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(lambda pair: execute(*pair), enumerate(jobs, 1)))


def load_ours_policy(path):
    if path is None:
        return None
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("--ours-policy-file must contain a JSON object")
    return value


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", action="append", choices=[*ARM_NAMES, *EXTRA_ARMS, "fixed"],
                        help="arm to run; repeatable (default: all 7 arms)")
    parser.add_argument("--cases", action="append",
                        help="case_id to run; repeatable (default: all cases)")
    parser.add_argument("--split", choices=["dev", "heldout", "all"], default="all")
    parser.add_argument("--limit", type=int, default=None,
                        help="cap the selected job count")
    parser.add_argument("--dry-run", action="store_true",
                        help="print planned identities and exit; no runs")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=16)
    parser.add_argument("--delay-ms", type=float, default=20)
    parser.add_argument("--jitter-ms", type=float, default=5)
    parser.add_argument("--seed", type=int, default=20261116)
    parser.add_argument("--model", default="DeepSeek-V4.1-Flash")
    parser.add_argument("--ours-policy-file", type=Path,
                        help="JSON overrides for the ours arm only")
    parser.add_argument("--scheduler",
                        help="M1–M4 scheduler for the ours arm: preset name in "
                             "configs/scheduler-presets.json or a JSON file")
    parser.add_argument("--shadow-verify", action="store_true",
                        help="evaluation only: compare every scheduler cache hit with "
                             "the backend (unmetered) and report the stale-hit rate")
    parser.add_argument("--lab-listing", choices=LISTINGS, default="per_turn",
                        help="lab: how often the agent framework asks for the tool list")
    parser.add_argument("--lab-contract", choices=CONTRACT_LEVELS, default="full",
                        help="lab: how much of the tool contract every host is given")
    parser.add_argument("--results-root", type=Path, default=RESULTS,
                        help="root for result files and the summary journal")
    return parser


def main(argv=None, *, document=None, execution_settings=None, suite_sha256=None,
         profile="smoke"):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.limit is not None and args.limit < 0:
        parser.error("--limit must not be negative")
    try:
        cases = validate_cases(document if document is not None else load_document())
        ours_policy = load_ours_policy(args.ours_policy_file)
        if args.scheduler:
            ours_policy = {**(ours_policy or {}), "scheduler": args.scheduler}
        if args.shadow_verify:
            ours_policy = {**(ours_policy or {}), "eval_shadow_verify": True}
    except (OSError, ValueError, KeyError) as error:
        print(f"matrix validation failed: {error}", file=sys.stderr)
        return 2
    if args.cases:
        known = {case["case_id"] for case in cases}
        unknown = sorted(set(args.cases) - known)
        if unknown:
            print(f"matrix validation failed: unknown case_id {', '.join(unknown)}",
                  file=sys.stderr)
            return 2
        selected = set(args.cases)
        cases = [case for case in cases if case["case_id"] in selected]
    try:
        execution_settings = execution_settings or {
            key: getattr(args, key) for key in
            ("model", "max_steps", "delay_ms", "jitter_ms", "seed", "workers")}
        jobs = build_jobs(
            cases, args.arms, split=args.split, ours_policy=ours_policy,
            results_root=args.results_root, execution_settings=execution_settings,
            suite_sha256=suite_sha256)
    except (KeyError, ValueError) as error:
        print(f"matrix validation failed: {error}", file=sys.stderr)
        return 2
    if args.limit is not None:
        jobs = jobs[:args.limit]
    if args.dry_run:
        print(json.dumps({
            "dry_run": True,
            "profile": profile,
            "run_config_sha256": jobs[0]["run_config_sha256"] if jobs else None,
            "jobs": len(jobs),
            "identities": [{"case_id": job["case_id"], "arm": job["arm"],
                            "policy_sha256": job["policy_sha256"],
                            "code_sha256": job["code_sha256"],
                            "run_config_sha256": job["run_config_sha256"]}
                           for job in jobs],
        }, ensure_ascii=False))
        return 0
    config_sha = jobs[0]["run_config_sha256"] if jobs else policy_sha256(execution_settings)
    output = args.results_root / config_sha
    for job in jobs:
        job["output_dir"] = output / job["case_id"] / job["arm"]
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".run.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(f"experiment already running: {output}", file=sys.stderr)
            return 2
        manifest_path = output / "run.json"
        if not manifest_path.exists():
            manifest_path.write_text(json.dumps({
                "profile": profile, "run_config_sha256": config_sha,
                "execution": execution_settings, "suite_sha256": suite_sha256,
                "code_sha256": jobs[0]["code_sha256"] if jobs else runner.code_sha256(),
                "planned_jobs": len(jobs),
            }, indent=2), encoding="utf-8")
        completed = completed_identities(output)
        pending = pending_jobs(jobs, completed)
        pending.sort(key=lambda job: (ARM_NAMES.index(job["arm"]), job["case_id"]))
        print(f"matrix: {len(jobs)} jobs, {len(jobs) - len(pending)} terminal, "
              f"{len(pending)} to run -> {output}", flush=True)
        if pending:
            run_matrix(pending, args, results_root=output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
