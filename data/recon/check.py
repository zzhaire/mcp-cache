"""Build and check the rebuilt benchmarks.

``./run recon-build [bfcl tau2 appworld]`` regenerates the task files.
``./run recon-check [bfcl tau2 appworld]`` replays every rebuilt task's
scripted solution under each method (no model) and prints, per scenario
family, how many pass and how many requests reached the server.
``./run recon-report <results root>`` summarises real-model runs of rebuilt
tasks: the added part of each task (``judge_overlay``), whether the original
task came out as the reference has it, and the requests each method sent.
"""
import argparse
import collections
import importlib
import json
from pathlib import Path

from exp_core.lab.judge import rescore_overlay
from exp_core.settings import DATA

DATASETS = ("bfcl", "tau2", "appworld")
ARMS = (("bare", "bare", None), ("openai", "rw_openai", None),
        ("fastmcp", "rw_fastmcp", None), ("gateway", "rw_gateway", None),
        ("mehan", "rw_mehan", None), ("merchant", "rw_merchant", None),
        ("ours-old", "ours", {"context_dedup": False}), ("strong", "strong", None),
        ("ours-new", "ours", {"scheduler": "default"}))
SEED = 20261116


def paths(dataset):
    return DATA / f"recon-{dataset}-tasks.json", DATA / f"recon-{dataset}-cases.json"


def build(argv=None):
    parser = argparse.ArgumentParser(prog="run recon-build")
    parser.add_argument("datasets", nargs="*", default=list(DATASETS))
    args = parser.parse_args(argv)
    for dataset in args.datasets:
        tasks, cases = importlib.import_module(f"data.recon.{dataset}").build()
        tasks_path, cases_path = paths(dataset)
        tasks_path.write_text(json.dumps(tasks, indent=1, ensure_ascii=False))
        cases_path.write_text(json.dumps({
            "version": f"recon-{dataset}-v0", "suite": "lab",
            "note": f"{dataset} 原任务接到实验服务器层之后叠加场景得到的 case；"
                    "split 沿用 data/manifest.json 对原任务的划分。",
            "cases": cases}, indent=1, ensure_ascii=False))
        families = collections.Counter(task["family"] for task in tasks)
        print(f"{dataset}: {len(tasks)} tasks", dict(families))
    return 0


def check(argv=None):
    parser = argparse.ArgumentParser(prog="run recon-check")
    parser.add_argument("datasets", nargs="*", default=list(DATASETS))
    parser.add_argument("--split", choices=["dev", "heldout", "all"], default="all")
    parser.add_argument("--arms", nargs="*", help="arm labels (default: all)")
    parser.add_argument("--listing", default="per_turn")
    parser.add_argument("--contract", default="full")
    parser.add_argument("--out", help="write every run as JSON lines")
    parser.add_argument("--why", action="store_true")
    args = parser.parse_args(argv)
    from exp_core import runner
    from exp_core.lab.harness import run_gold
    arms = [arm for arm in ARMS if not args.arms or arm[0] in args.arms]
    rows = []
    for dataset in args.datasets:
        tasks_path, cases_path = paths(dataset)
        if not tasks_path.is_file():
            print(f"{dataset}: not built")
            continue
        tasks = {t["id"]: t for t in json.loads(tasks_path.read_text())}
        cases = [c for c in json.loads(cases_path.read_text())["cases"]
                 if args.split in ("all", c["split"])]
        for case in cases:
            task = tasks[case["task_id"]]
            for label, method, override in arms:
                run = run_gold(task, runner.resolve_policy(method, override), seed=SEED,
                               shadow_verify=True, listing=args.listing,
                               contract_level=args.contract)
                delivery = run["delivery"]
                rows.append({
                    "dataset": dataset, "task": task["id"], "family": task["family"],
                    "split": case["split"], "arm": label,
                    "passed": bool(run["oracle"]["success"]),
                    "safe": bool(run["oracle"].get("safe", True)),
                    "overlay": run["oracle"]["overlay"]["verdict"],
                    "requests": run["requests"], "catalog": run["catalog_requests"],
                    "logical_calls": run["logical_calls"],
                    "stale": delivery["stale"] + delivery["stale_references"],
                    "after_revocation": delivery["after_revocation"],
                    "failed": [f"{c['check']}: {str(c['got'])[:160]}"
                               for c in run["oracle"]["checks"] if not c["ok"]]})
        table(dataset, [r for r in rows if r["dataset"] == dataset], arms)
    if args.why:
        for row in rows:
            if not row["passed"]:
                print(f"- {row['task']} / {row['arm']}: " + "; ".join(row["failed"])[:400])
    if args.out:
        with open(args.out, "w") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return 0


def report(argv=None):
    parser = argparse.ArgumentParser(prog="run recon-report")
    parser.add_argument("root", type=Path)
    parser.add_argument("--why", action="store_true", help="list the failed checks")
    args = parser.parse_args(argv)
    runs = collections.defaultdict(list)
    for path in sorted(args.root.rglob("*.result.json")):
        row = json.loads(path.read_text())
        overlay = (row.get("oracle") or {}).get("overlay")
        if row.get("status") != "completed" or overlay is None:
            print(f"skipped {row.get('task_id')} / {row.get('method')}: "
                  f"{row.get('error') or 'no judgement of the added part'}")
            continue
        overlay = rescore_overlay(overlay, row.get("turn_answers") or [])
        runs[row["method"]].append((row, overlay))
    methods = sorted(runs)
    width = max((len(row["task_id"]) for rows in runs.values() for row, _ in rows), default=4)
    print(f"{'case':{width}}  " + "  ".join(f"{m + ': added / original / requests':>36}"
                                           for m in methods))
    for task in sorted({row["task_id"] for rows in runs.values() for row, _ in rows}):
        line = f"{task:{width}}  "
        for method in methods:
            cells = [f"{o['verdict']} ({o['answers_ok']}/{o['answers']})"
                     f" / {'same' if o['original_matches_reference'] else 'differs'}"
                     f" / {row['mcp_requests']}"
                     for row, o in runs[method] if row["task_id"] == task]
            line += f"{'; '.join(cells) or '-':>36}  "
        print(line)
    print()
    for method in methods:
        rows = runs[method]
        total = lambda key: sum(row.get(key) or 0 for row, _ in rows)  # noqa: E731
        print(f"{method}: {len(rows)} runs; added part "
              + ", ".join(f"{sum(o['verdict'] == v for _, o in rows)} {v}"
                          for v in ("pass", "fail", "untested"))
              + f"; answers {sum(o['answers_ok'] for _, o in rows)}"
                f"/{sum(o['answers'] for _, o in rows)}"
              + f"; repeated write in {sum(not o['safe'] for _, o in rows)}"
              + f"; original as reference in "
                f"{sum(o['original_matches_reference'] for _, o in rows)}"
              + f"; requests {total('mcp_requests')} (catalog {total('mcp_requests_catalog')})"
              + f"; model calls {total('model_calls')}; input tokens {total('input_tokens')}"
              + f"; published results used {total('published_results_used')}")
    if args.why:
        for method in methods:
            for row, overlay in runs[method]:
                for check in overlay["checks"]:
                    if not check["ok"]:
                        print(f"- {row['task_id']} / {method}: {check['check']}: want "
                              f"{str(check['want'])[:80]!r}, got {str(check['got'])[:160]!r}")
    return 0


def table(dataset, rows, arms):
    labels = [arm[0] for arm in arms]
    families = sorted({r["family"] for r in rows})
    count = len({r["task"] for r in rows})
    print(f"\n== {dataset}: {count} cases ==")
    print(f"{'family (cases)':24}" + "".join(f"{label:>11}" for label in labels))
    for family in [*families, None]:
        picked = [r for r in rows if family is None or r["family"] == family]
        cases = len({r["task"] for r in picked})
        line = f"{(family or 'TOTAL') + f' ({cases})':24}"
        for label in labels:
            mine = [r for r in picked if r["arm"] == label]
            line += f"{str(sum(r['passed'] for r in mine)) + '/' + str(sum(r['requests'] for r in mine)):>11}"
        print(line)
    for title, key in (("stale values delivered", "stale"),
                       ("delivered after revocation", "after_revocation")):
        print(f"{title:24}" + "".join(
            f"{sum(r[key] for r in rows if r['arm'] == label):>11}" for label in labels))
    print(f"{'runs with unsafe write':24}" + "".join(
        f"{sum(not r['safe'] for r in rows if r['arm'] == label):>11}" for label in labels))
    print(f"{'added part passes':24}" + "".join(
        f"{sum(r['overlay'] == 'pass' for r in rows if r['arm'] == label):>11}"
        for label in labels))
    print("(each cell: cases passed / requests that reached the server)")


def main(argv=None):
    raise SystemExit("use ./run recon-build or ./run recon-check")
