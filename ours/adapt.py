"""Fit the scheduler to a scenario from one recorded run.

``./run adapt --trace-root results/<run> --out-dir configs/adapted``

1. Profile: read the traces of one no-cache run (every call reached the server,
   so the record is ground truth) and measure, per tool, how often the same
   read comes back, how far apart, how large it is, and how often its value
   changed without the agent having written anything that affects it.
2. Derive: turn those measurements into parameters by rule (freshness class
   per tool, expiry per class).
3. Tune: for the parameters no rule covers, replay the same record under
   one-change neighbours and keep a change only if it serves no stale value and
   sends fewer requests (``ours.replay``; no model call).

The result is a complete scheduler policy file, usable as ``--scheduler FILE``,
plus a report of what was measured and why each parameter was set.

Limits: a replay assumes the agent makes the same calls under every policy, and
one run only shows the changes that happened in it. Parameters that act on the
model's context (the ``publish`` section) cannot be judged by replay and are
left alone. Confirm the result with a real run.
"""
import argparse
import copy
import glob
import json
from collections import defaultdict
from pathlib import Path

from exp_core.access import _expand
from exp_core.lab.harness import load_tasks
from exp_core.lab.domains import make_world
from exp_core.scheduler_policy import resolve, validate
from exp_core.settings import CONFIGS, RESULTS
from ours.replay import load_run, replay
from ours.rsi_space import get_value, neighbours, tunable_space, with_value

CLASS_ORDER = {"static": 0, "slow": 1, "volatile": 2, "strict": 3}
# Parameters a replay can judge: they change what the host sends or serves,
# not what the model is shown.
REPLAY_AXES = ("ttl.slow", "ttl.volatile", "invalidation.granularity",
               "forget.idle_steps", "forget.on_turn_end", "recovery.failure_index")
ARM_BASE = {"arm": "ours", "disabled_modules": (), "m4_failure_only": False}


def _key(name, arguments):
    return name + " " + json.dumps(arguments, sort_keys=True, default=str)


def _ok(result):
    return not (isinstance(result, dict) and "error" in result)


def profile_run(run, contract, into=None):
    """Add one run's observations to ``into`` (created when None)."""
    prof = into or {"runs": 0, "tools": {}, "auth": defaultdict(int),
                    "catalog": defaultdict(int), "errors": defaultdict(int)}
    prof["runs"] += 1
    last, writes, step, turn = {}, [], 0, 0
    signed_in, listing, version = set(), None, None
    users = {}  # app -> user currently signed in; a read belongs to that user
    for event in run["events"]:
        if event[0] == "advance":
            step, turn = event[1], event[2]
            continue
        if event[0] == "catalog":
            cat = prof["catalog"]
            cat["listings"] += 1
            cat["bytes"] = max(cat["bytes"], len(json.dumps(event[1])))
            content = json.dumps(event[1].get("tools"), sort_keys=True)
            if listing is not None:
                same_version = event[1].get("version") == version
                if content == listing:
                    cat["unchanged_relists"] += 1
                    cat["version_bumps_without_change"] += not same_version
                else:
                    cat["content_changes"] += 1
                    cat["silent_changes"] += same_version
            listing, version = content, event[1].get("version")
            continue
        _, _, _, name, arguments, _, result = event
        spec = contract.get(name) or {"kind": "write", "app": "lab"}
        kind = spec["kind"]
        if not _ok(result):
            prof["errors"][str(result.get("error"))] += 1
            if result.get("error") == "unauthorized":
                signed_in.discard(spec.get("principal"))
            continue
        if kind == "login":
            who = (arguments.get("app"), arguments.get("username"))
            prof["auth"]["logins"] += 1
            prof["auth"]["repeated_logins"] += who in signed_in
            signed_in.add(who)
            signed_in.add(arguments.get("app"))
            users[arguments.get("app")] = arguments.get("username")
            continue
        if kind == "logout":
            prof["auth"]["logouts"] += 1
            signed_in = {s for s in signed_in if s != arguments.get("app")
                         and not (isinstance(s, tuple) and s[0] == arguments.get("app"))}
            continue
        if kind == "write":
            affects = set(_expand(spec.get("affects", []), arguments))
            writes.append((step, affects, spec.get("app")))
            continue
        if kind != "read":
            continue
        tool = prof["tools"].setdefault(name, {
            "contract_class": spec.get("fresh_class") or "slow", "calls": 0, "repeats": 0,
            "unchanged": 0, "changed_after_own_write": 0, "changed_externally": 0,
            "max_unchanged_gap_steps": 0, "min_external_gap_steps": None, "max_bytes": 0})
        tool["calls"] += 1
        tool["max_bytes"] = max(tool["max_bytes"], len(json.dumps(result, default=str)))
        identity = _key(name, arguments) + " as " + str(users.get(spec.get("principal")))
        reads = set(_expand(spec.get("reads", []), arguments))
        if identity in last:
            before_step, before = last[identity]
            gap = step - before_step
            wrote = any(at > before_step and (affected & reads or (not affected and app == spec.get("app")))
                        for at, affected, app in writes)
            tool["repeats"] += 1
            if result == before:
                tool["unchanged"] += 1
                tool["max_unchanged_gap_steps"] = max(tool["max_unchanged_gap_steps"], gap)
            elif wrote:
                tool["changed_after_own_write"] += 1
            else:
                tool["changed_externally"] += 1
                low = tool["min_external_gap_steps"]
                tool["min_external_gap_steps"] = gap if low is None else min(low, gap)
        last[identity] = (step, result)
    return prof


def derive(prof, base):
    """Rule-based parameters from a profile. Returns ``(policy, notes)``."""
    policy, notes = copy.deepcopy(base), []
    space = tunable_space()
    overrides = dict(policy["ttl"].get("fresh_class_overrides") or {})
    policy["ttl"]["adaptive"]["enabled"] = False
    notes.append("自适应有效期关闭：数据变化的快慢已在画像运行里量过，正常运行不再抽样回源。")
    volatile_gaps, slow_gaps = [], []
    for name, tool in sorted(prof["tools"].items()):
        cls = max(tool["contract_class"], overrides.get(name, tool["contract_class"]),
                  key=CLASS_ORDER.get)
        external, repeats = tool["changed_externally"], tool["repeats"]
        if external:
            gap = tool["min_external_gap_steps"]
            if cls in ("static", "slow"):
                # The contract says only the agent's own writes change it; the record disagrees.
                cls = "strict" if (external * 2 >= repeats or gap <= 1) else "volatile"
                overrides[name] = cls
                notes.append(f"{name}：契约是 {tool['contract_class']}，但 {repeats} 次重复读取里有 "
                             f"{external} 次在没有相关写操作的情况下变了，改按 {cls} 处理。")
            if cls == "volatile":
                volatile_gaps.append((gap, name))
        elif cls == "slow" and tool["unchanged"]:
            slow_gaps.append(tool["max_unchanged_gap_steps"])
    if volatile_gaps:
        gap, name = min(volatile_gaps)
        spec = space["ttl.volatile"]
        value = max(spec.get("min", 0), min(gap - 1, policy["ttl"]["volatile"]))
        if value != policy["ttl"]["volatile"]:
            notes.append(f"易变类有效期由 {policy['ttl']['volatile']} 改为 {value}："
                         f"{name} 最快隔 {gap} 步就被外部改动过。")
            policy["ttl"]["volatile"] = value
    if slow_gaps:
        spec = space["ttl.slow"]
        value = min(spec.get("max", 3600), max(policy["ttl"]["slow"], max(slow_gaps) + 1))
        if value != policy["ttl"]["slow"]:
            notes.append(f"慢变类有效期由 {policy['ttl']['slow']} 改为 {value}：这类读取最长隔 "
                         f"{max(slow_gaps)} 步重复且未见外部变化。")
            policy["ttl"]["slow"] = value
    policy["ttl"]["fresh_class_overrides"] = overrides
    return validate(policy), notes


def score(runs, scheduler, tasks, seed=0):
    """(values served that differ from the record, requests) over all runs."""
    arm = dict(ARM_BASE, scheduler=scheduler)
    stale = requests = revoked = 0
    for run in runs:
        out = replay(run, arm, tasks, seed=seed)
        stale += out["stale_hits"] + out["stale_catalogs"]
        requests += out["requests"]
        revoked += out["served_after_revocation"]
    return {"stale": stale, "requests": requests, "served_after_revocation": revoked}


def tune(runs, policy, tasks, axes=REPLAY_AXES, rounds=3):
    """Coordinate descent by replay: keep a one-change neighbour only when it
    serves no more stale values and sends fewer requests."""
    space, notes = tunable_space(), []
    best = score(runs, policy, tasks)
    for _ in range(rounds):
        improved = False
        for path in axes:
            for label, candidate in neighbours(policy, path, space):
                got = score(runs, candidate, tasks)
                if (got["stale"], got["requests"]) < (best["stale"], best["requests"]):
                    notes.append(f"{path}：{get_value(policy, path)} 改为 {get_value(candidate, path)}，"
                                 f"回放的过时返回 {best['stale']}→{got['stale']}，"
                                 f"请求数 {best['requests']}→{got['requests']}。")
                    policy, best, improved = candidate, got, True
        if not improved:
            break
    return policy, best, notes


def adapt(runs, tasks, base="default"):
    """Profile, derive and tune for one group of runs."""
    base_policy = resolve(base)
    prof = None
    for run in runs:
        prof = profile_run(run, make_world(tasks[run["task_id"]]).contract(), prof)
    derived, notes = derive(prof, base_policy)
    tuned, predicted, tune_notes = tune(runs, derived, tasks)
    for key in ("auth", "catalog", "errors"):
        prof[key] = dict(prof[key])
    return {"policy": tuned, "profile": prof, "notes": notes + tune_notes,
            "replay": {"no_cache_requests": sum(1 for run in runs for event in run["events"]
                                                if event[0] == "catalog")
                       + sum(len(event[5]) for run in runs for event in run["events"]
                             if event[0] == "call"),
                       "base": score(runs, base_policy, tasks),
                       "derived": score(runs, derived, tasks), "adapted": predicted}}


def load_runs(trace_root, method="bare"):
    """One completed trace of ``method`` per task under the root. With several
    runs of a task, the one whose path sorts last is used (not the most recent)."""
    runs = {}
    for result in sorted(glob.glob(f"{trace_root}/**/{method}/*.result.json", recursive=True)):
        row = json.loads(Path(result).read_text(encoding="utf-8"))
        if row.get("status") == "completed" and row.get("dataset") == "lab":
            runs[row["task_id"]] = load_run(result.replace(".result.json", ".jsonl"))
    return runs


def main(argv=None):
    parser = argparse.ArgumentParser(prog="run adapt", description=__doc__.split("\n\n")[0])
    parser.add_argument("--trace-root", type=Path, required=True,
                        help="results of a run that includes the no-cache method")
    parser.add_argument("--method", default="bare", help="the recorded method to learn from")
    parser.add_argument("--group", choices=("all", "family"), default="family",
                        help="one policy for everything, or one per scenario family")
    parser.add_argument("--base", default="default", help="starting scheduler preset")
    parser.add_argument("--name", default=None, help="prefix of the output files")
    parser.add_argument("--out-dir", type=Path, default=CONFIGS / "adapted")
    parser.add_argument("--report-dir", type=Path, default=RESULTS / "adapt")
    args = parser.parse_args(argv)
    tasks = load_tasks()
    runs = load_runs(args.trace_root, args.method)
    if not runs:
        raise ValueError(f"no completed {args.method} lab traces under {args.trace_root}")
    groups = defaultdict(list)
    for task_id, run in runs.items():
        groups["all" if args.group == "all" else tasks[task_id].get("family", "basic")].append(run)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    name = args.name or args.trace_root.name
    index = {}
    for group, members in sorted(groups.items()):
        result = adapt(members, tasks, args.base)
        policy_path = args.out_dir / f"{name}.{group}.json"
        policy_path.write_text(json.dumps(result["policy"], ensure_ascii=False, indent=2) + "\n",
                               encoding="utf-8")
        report = {"group": group, "tasks": sorted(run["task_id"] for run in members),
                  "base": args.base, "policy_file": str(policy_path),
                  "replay": result["replay"], "notes": result["notes"],
                  "profile": result["profile"]}
        (args.report_dir / f"{name}.{group}.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        index[group] = {"policy_file": str(policy_path), "tasks": report["tasks"]}
        rep = result["replay"]
        print(f"{group}: {len(members)} runs, no cache {rep['no_cache_requests']} requests; "
              f"replay base {rep['base']['requests']} ({rep['base']['stale']} stale) -> "
              f"adapted {rep['adapted']['requests']} ({rep['adapted']['stale']} stale)")
        for note in result["notes"]:
            print("  - " + note)
    (args.report_dir / f"{name}.index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0
