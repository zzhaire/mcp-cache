"""Judge recorded lab runs again with the current oracle, without a model.

``./run lab-rejudge --root results/<run> [--root ...]``

The lab world is deterministic, so a run's trace is enough to rebuild it: every
request that reached the server is replayed, in order, into a fresh world, and
each reply is compared with the recorded one. The rebuilt world, the recorded
answers and the results the agent was handed then go through the oracle.
A run whose replies no longer match (the task or the server changed since) is
reported as "diverged" and not judged.
"""
import argparse
import glob
import json
from collections import Counter
from pathlib import Path

from exp_core.lab.harness import load_tasks
from exp_core.lab.judge import served_entry
from exp_core.lab.domains import make_world

REDACTED = "[redacted]"


def _body(record):
    content = (record.get("response", {}).get("result") or {}).get("content") or [{}]
    try:
        return json.loads(content[0].get("text", ""))
    except ValueError:
        return None


def rebuild(trace_path, task):
    """``(world, served, diverged)`` from one trace."""
    world, served, diverged = make_world(task), [], 0
    users = task["initial"].get("users", {})
    for line in open(trace_path, encoding="utf-8"):
        try:
            record = json.loads(line)
        except ValueError:
            continue
        kind = record.get("kind")
        if kind == "action":
            while len(served) <= record["turn"]:
                served.append([])
            served[record["turn"]].append(
                served_entry(record["name"], record["arguments"], record["result"]))
        if kind != "mcp" or record.get("method") != "tools/call":
            continue
        params = record["request"].get("params", {})
        arguments = params.get("arguments") or {}
        credentials = arguments.get("credentials")
        if params.get("name") == "lab_advance":
            world.advance(arguments["step"], arguments["turn"], credentials)
        elif params.get("name") == "lab_note":
            world.note(arguments["name"], credentials)
        elif params.get("name") == "lab_list_tools":
            world.catalog(credentials)
        elif params.get("name") == "invoke":
            inner = arguments.get("arguments")
            if isinstance(inner, dict) and inner.get("password") == REDACTED:
                inner = {**inner, "password": users.get(inner.get("username"))}
            payload = world.invoke(arguments.get("name"), inner, key=arguments.get("key"),
                                   shadow=bool(arguments.get("shadow")),
                                   credentials=credentials)
            recorded = _body(record)
            if recorded is not None and json.dumps(payload, sort_keys=True, default=str) != \
                    json.dumps(recorded, sort_keys=True, default=str):
                diverged += 1
    return world, served, diverged


def rejudge(root, tasks=None):
    tasks = tasks or load_tasks()
    rows = []
    for path in sorted(glob.glob(f"{root}/**/*.result.json", recursive=True)):
        row = json.loads(Path(path).read_text(encoding="utf-8"))
        if row.get("dataset") != "lab" or row.get("status") != "completed":
            continue
        world, served, diverged = rebuild(path.replace(".result.json", ".jsonl"),
                                          tasks[row["task_id"]])
        answers = row.get("turn_answers") or []
        while len(served) < len(answers):
            served.append([])
        verdict = world.oracle(answers, served)
        rows.append({"task": row["task_id"], "method": row["method"],
                     "was": bool(row.get("task_success")),
                     "verdict": "diverged" if diverged else verdict["verdict"],
                     "diverged": diverged,
                     "problems": [c for c in verdict["checks"] if not c["ok"]],
                     "result_file": path})
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(prog="run lab-rejudge", description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, help="write every row as JSON")
    parser.add_argument("--why", action="store_true", help="print the failed or unclear checks")
    args = parser.parse_args(argv)
    everything = []
    for root in args.root:
        rows = rejudge(root)
        everything.extend({"root": str(root), **row} for row in rows)
        for method in sorted({row["method"] for row in rows}):
            mine = [row for row in rows if row["method"] == method]
            tally = Counter(row["verdict"] for row in mine)
            print(f"{root.name} / {method}: {len(mine)} runs, was {sum(r['was'] for r in mine)} "
                  f"passed -> pass {tally['pass']}, review {tally['review']}, "
                  f"fail {tally['fail']}, diverged {tally['diverged']}")
        if args.why:
            for row in rows:
                if row["verdict"] in ("pass",):
                    continue
                print(f"  - {row['task']} / {row['method']}: {row['verdict']}"
                      + (f" ({row['diverged']} replies differ)" if row["diverged"] else ""))
                for check in row["problems"]:
                    if row["diverged"]:
                        break
                    note = check.get("note") or json.dumps(check.get("got"), ensure_ascii=False,
                                                           default=str)[:220]
                    print(f"      {check['status']}: {check['check']}: {note}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(everything, ensure_ascii=False, indent=1, default=str)
                            + "\n", encoding="utf-8")
    return 0
