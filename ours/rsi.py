"""Two-round RSI over the pre-registered Ours policy space, dev cases only.

The search space is frozen in ``POLICY_SPACE``: every candidate is the Ours
base policy with only those five keys overridden. Round 1 evaluates six
pre-registered candidates; round 2 evaluates up to three neighbours of the
round-1 winner that were not evaluated before. Selection is correctness-first,
then the normalized weighted mean of ``active_time_s``, ``input_tokens`` and
``mcp_requests``.

The evaluation set is exactly one dev case per ``(dataset, scenario)`` cell,
the lexicographically first dev ``case_id``. Heldout cases are never read and
the journal records ``heldout_access=false``.

``--space scheduler`` searches the M1–M4 scheduler policy instead
(``ours/rsi_space.py``): round 1 compares pre-registered presets, and every
later round evaluates one-parameter neighbours of the best policy so far along
the axes listed in ``configs/rsi-scheduler-space.json``. Only schema parameters
tagged L1/L2 can change and every candidate passes the full policy validation.
"""
import argparse
import copy
from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path
import re
import sys
import threading

from exp_core import runner
from exp_core.methods import policy_sha256, resolve_arm
from exp_core.scheduler_policy import apply_to_arm, resolve as resolve_scheduler
from exp_core.settings import CONFIGS, ROOT, DATA, RESULTS as RESULTS_ROOT
from ours import rsi_space

SCENARIOS = DATA / "scenarios-36.json"
RESULTS = RESULTS_ROOT / "rsi"
JOURNAL = RESULTS / "journal.json"
POLICY_OUT = RESULTS / "selected-policy.json"
SPACE_PLAN = CONFIGS / "rsi-scheduler-space.json"
MAX_SCHEDULER_ROUNDS = 8
TERMINAL = {"completed", "error"}
TOTAL_CELLS = 9
ROUND2_LIMIT = 3
PRINT_LOCK = threading.Lock()
POLICY_SPACE = {
    "ttl_seconds": (30, 60, 120),
    "publish_read_results": (False, True),
    "publish_budget_bytes": (2048, 8192, 16384),
    "publish_operations": (False, True),
    "context_dedup": (False, True),
}
OVERRIDE_KEYS = tuple(POLICY_SPACE)
# Equal weights: the score is the mean of the three normalized metrics.
METRIC_WEIGHTS = {"active_time_s": 1 / 3, "input_tokens": 1 / 3,
                  "mcp_requests": 1 / 3}
METRICS = tuple(METRIC_WEIGHTS)
ROW_FIELDS = ("case_id", "status", "task_success", "active_time_s",
              "input_tokens", "output_tokens", "mcp_requests")
ROUND1_CANDIDATES = (
    ("base", {}),
    ("ttl30", {"ttl_seconds": 30}),
    ("ttl120", {"ttl_seconds": 120}),
    ("budget2048", {"publish_budget_bytes": 2048}),
    ("no_ops", {"publish_operations": False}),
    ("no_dedup", {"context_dedup": False}),
)


def _in_space(key, value):
    """Strict membership: ``1`` is not accepted for a boolean slot."""
    return any(type(value) is type(allowed) and value == allowed
               for allowed in POLICY_SPACE[key])


def resolve_candidate(overrides, base=None):
    """Return the Ours base policy with only pre-registered keys overridden."""
    if not isinstance(overrides, dict):
        raise ValueError("candidate overrides must be a mapping")
    unknown = sorted(set(overrides) - set(OVERRIDE_KEYS))
    if unknown:
        raise ValueError(f"unexpected candidate override keys: {', '.join(unknown)}")
    policy = resolve_arm("ours") if base is None else copy.deepcopy(base)
    for key, value in overrides.items():
        if not _in_space(key, value):
            raise ValueError(f"{key}={value!r} is outside the pre-registered "
                             "policy space")
        policy[key] = value
    return policy


def candidate_id(policy):
    """Deterministic identifier derived from the five searchable fields."""
    return (f"ttl{policy['ttl_seconds']}"
            f"-budget{policy['publish_budget_bytes']}"
            f"-read{int(bool(policy['publish_read_results']))}"
            f"-ops{int(bool(policy['publish_operations']))}"
            f"-dedup{int(bool(policy['context_dedup']))}")


def round1_candidates():
    """The six pre-registered round-1 candidates with stable ids."""
    return [{"id": name, "overrides": dict(overrides),
             "policy": resolve_candidate(overrides)}
            for name, overrides in ROUND1_CANDIDATES]


def round2_candidates(winner_policy, evaluated_sha256, limit=ROUND2_LIMIT):
    """Up to ``limit`` one-field neighbours of the winner never evaluated before.

    Deterministic order: TTL at both other values (ascending), then the largest
    publish budget, then disabling read-result publication. Duplicates by
    ``policy_sha256`` are skipped, so a neighbour identical to the winner or to
    an earlier candidate is not re-run.
    """
    overrides_list = [{"ttl_seconds": value} for value in POLICY_SPACE["ttl_seconds"]
                      if value != winner_policy["ttl_seconds"]]
    top_budget = max(POLICY_SPACE["publish_budget_bytes"])
    if winner_policy["publish_budget_bytes"] != top_budget:
        overrides_list.append({"publish_budget_bytes": top_budget})
    if winner_policy["publish_read_results"]:
        overrides_list.append({"publish_read_results": False})
    evaluated = set(evaluated_sha256)
    candidates = []
    for overrides in overrides_list:
        policy = resolve_candidate(overrides, base=winner_policy)
        digest = policy_sha256(policy)
        if digest in evaluated:
            continue
        evaluated.add(digest)
        candidates.append({"id": candidate_id(policy), "overrides": overrides,
                           "policy": policy})
        if len(candidates) >= limit:
            break
    return candidates


def load_search_plan(path=SPACE_PLAN, space=None):
    """Read and check the pre-registered scheduler search plan."""
    plan = json.loads(Path(path).read_text(encoding="utf-8"))
    for field in ("round1_presets", "axes", "factors", "round_limit"):
        if field not in plan:
            raise ValueError(f"search plan is missing {field}")
    if not plan["round1_presets"] or not plan["axes"]:
        raise ValueError("search plan needs at least one preset and one axis")
    if int(plan["round_limit"]) < 1:
        raise ValueError("search plan round_limit must be at least 1")
    if len(set(plan["axes"])) != len(plan["axes"]):
        raise ValueError("search plan lists an axis twice")
    rsi_space.check_axes(plan["axes"], space or rsi_space.tunable_space())
    for preset in plan["round1_presets"]:
        resolve_scheduler(preset)
    return plan


def scheduler_candidate(identifier, scheduler_policy, change=None, shadow_verify=False):
    """Wrap a scheduler policy as an executable Ours candidate."""
    policy = resolve_arm("ours")
    policy["scheduler"] = copy.deepcopy(scheduler_policy)
    if shadow_verify:
        policy["eval_shadow_verify"] = True
    apply_to_arm(policy)
    return {"id": identifier, "overrides": change or {}, "policy": policy}


def scheduler_round1(plan, shadow_verify=False):
    """Round 1: the pre-registered presets, one candidate each."""
    return [scheduler_candidate(name, resolve_scheduler(name), {"preset": name},
                                shadow_verify) for name in plan["round1_presets"]]


def scheduler_neighbours(base_candidate, evaluated_sha256, plan, space, start=0,
                         shadow_verify=False):
    """Up to ``round_limit`` unevaluated one-change neighbours of the base.

    Axes are visited in plan order starting at ``start`` and wrapping once, so
    successive rounds continue where the previous round stopped. Returns
    ``(candidates, next_start)``.
    """
    base = base_candidate["policy"]["scheduler"]
    axes, limit = plan["axes"], int(plan["round_limit"])
    evaluated = set(evaluated_sha256)
    candidates, position = [], start
    for offset in range(len(axes)):
        position = (start + offset) % len(axes)
        path = axes[position]
        for label, neighbour in rsi_space.neighbours(
                base, path, space, factors=tuple(plan["factors"]),
                share_slice=plan.get("share_slice", 0.1)):
            neighbour["preset"] = str(base.get("preset", "custom")).rstrip("*") + "*"
            candidate = scheduler_candidate("", neighbour, {path: label}, shadow_verify)
            digest = policy_sha256(candidate["policy"])
            if digest in evaluated:
                continue
            evaluated.add(digest)
            # The id is also a directory name: keep it filesystem-safe.
            safe = re.sub(r"[^A-Za-z0-9._+-]", "_", f"{path}={label}")
            candidate["id"] = f"{safe}@{digest[:8]}"
            candidates.append(candidate)
            if len(candidates) >= limit:
                return candidates, (position + 1) % len(axes)
    return candidates, (position + 1) % len(axes)


def stale_hit_rate(rows):
    """Stale hits / shadow checks over the rows that carry shadow measurements."""
    checks = sum((row.get("stale_hits") or {}).get("shadow_checks") or 0 for row in rows)
    stale = sum((row.get("stale_hits") or {}).get("shadow_stale") or 0 for row in rows)
    return (stale / checks) if checks else None


def select_dev_cases(document, requested=None):
    """One lexicographically first dev case per cell, or explicit dev cases."""
    cases = document.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("scenarios document has no cases")
    by_id = {}
    for case in cases:
        for field in ("case_id", "dataset", "scenario", "split"):
            if not case.get(field):
                raise ValueError(f"case is missing field {field}")
        if case["case_id"] in by_id:
            raise ValueError(f"duplicate case_id {case['case_id']}")
        by_id[case["case_id"]] = case
    if requested:
        unknown = sorted(set(requested) - set(by_id))
        if unknown:
            raise ValueError(f"unknown case_id: {', '.join(unknown)}")
        selected, seen = [], set()
        for case_id in requested:
            case = by_id[case_id]
            if case["split"] != "dev":
                raise ValueError(f"refusing non-dev case {case_id} "
                                 f"(split={case['split']})")
            if case_id not in seen:
                seen.add(case_id)
                selected.append(case)
        return selected
    cells = {}
    for case in cases:
        cells.setdefault((case["dataset"], case["scenario"]), []).append(case)
    if len(cells) != TOTAL_CELLS:
        raise ValueError(f"expected {TOTAL_CELLS} dataset x scenario cells, "
                         f"found {len(cells)}")
    selected = []
    for key in sorted(cells):
        dev = sorted((case for case in cells[key] if case["split"] == "dev"),
                     key=lambda case: case["case_id"])
        if not dev:
            raise ValueError(f"cell {key[0]}/{key[1]} has no dev case")
        selected.append(dev[0])
    for case in selected:
        if case["split"] != "dev":
            raise ValueError(f"selected non-dev case {case['case_id']}")
    return selected


def find_terminal_result(output_dir):
    """First terminal ``*.result.json`` in an output dir, or ``None``."""
    output_dir = Path(output_dir)
    if not output_dir.is_dir():
        return None
    for path in sorted(output_dir.glob("*.result.json")):
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if row.get("status") in TERMINAL:
            return row
    return None


def _numeric(value):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(float(value)))


def aggregate_rows(rows):
    """Correctness counts plus metric means over rows that carry a number.

    Error rows have no numeric metrics; they are counted as failures and are
    not imputed. Missing values are excluded from the mean.
    """
    status_counts = {"completed": 0, "error": 0}
    for row in rows:
        status = str(row.get("status"))
        status_counts[status] = status_counts.get(status, 0) + 1
    means = {}
    for metric in METRICS:
        values = [float(row[metric]) for row in rows if _numeric(row.get(metric))]
        means[metric] = sum(values) / len(values) if values else None
    return {
        "cases": len(rows),
        "success_count": sum(1 for row in rows if row.get("task_success") is True),
        "status_counts": status_counts,
        "error_count": status_counts.get("error", 0),
        "means": means,
        "eligible": None,
        "score": None,
    }


def _normalize(value, values):
    if value is None:
        return None
    low, high = min(values), max(values)
    if high == low:
        return 0.0
    return (value - low) / (high - low)


def _tie_value(means, metric):
    value = means.get(metric)
    return float("inf") if value is None else float(value)


def rank_candidates(aggregates):
    """Return ``(winner, ranked, eligible_ids, max_success)``.

    Only candidates with the maximum success count are eligible. A candidate
    with an error result is ineligible unless every candidate has an error.
    Eligible candidates are min-max normalized over the candidate set; a
    candidate with no numeric value for a metric gets the worst value (1.0),
    so missing measurements are never rewarded. Lowest weighted score wins,
    ties break on lower input-token mean, lower MCP-request mean, then id.
    """
    if not aggregates:
        raise RuntimeError("no candidates to select from")
    max_success = max(item["success_count"] for item in aggregates)
    eligible = [item for item in aggregates if item["success_count"] == max_success]
    if not all(item.get("error_count", 0) > 0 for item in aggregates):
        eligible = [item for item in eligible if item.get("error_count", 0) == 0]
    if not eligible:
        raise RuntimeError(
            "no eligible candidate: the maximum success_count is "
            f"{max_success} but every candidate reaching it has error results")
    ranges = {}
    for metric in METRICS:
        ranges[metric] = [item["means"][metric] for item in eligible
                          if item["means"].get(metric) is not None]
    scored = []
    for item in eligible:
        score, total_weight = 0.0, 0.0
        for metric, weight in METRIC_WEIGHTS.items():
            normalized = _normalize(item["means"].get(metric), ranges[metric])
            score += weight * (1.0 if normalized is None else normalized)
            total_weight += weight
        scored.append({"id": item["id"], "score": score / total_weight,
                       "means": item["means"],
                       "success_count": item["success_count"],
                       "error_count": item.get("error_count", 0)})
    scored.sort(key=lambda item: (
        item["score"],
        _tie_value(item["means"], "input_tokens"),
        _tie_value(item["means"], "mcp_requests"),
        item["id"]))
    return scored[0], scored, [item["id"] for item in eligible], max_success


def _format_mean(value):
    return "n/a" if value is None else f"{value:.4f}"


def _selection_reason(round_number, winner, ranked, eligible_ids, max_success):
    scores = ",".join(f"{item['id']}:{item['score']:.4f}" for item in ranked)
    means = ",".join(f"{metric}={_format_mean(winner['means'].get(metric))}"
                     for metric in METRICS)
    return (f"round {round_number}: max success_count={max_success}; "
            f"eligible={','.join(eligible_ids)}; selected={winner['id']} "
            f"score={winner['score']:.4f} means={means}; scores={scores}")


def evaluate_candidate(candidate, cases, args, round_number,
                       results_root=RESULTS, printer=print):
    """Run one candidate on the dev cases, reusing terminal results per case.

    Cases run in parallel up to ``args.workers``; candidates stay sequential so
    every candidate sees the same level of machine load. Row order follows the
    case list, not completion order, so the journal is deterministic.
    """
    def evaluate(case):
        output_dir = (Path(results_root) / f"round-{round_number}"
                      / candidate["id"] / case["case_id"])
        row = find_terminal_result(output_dir)
        reused = row is not None
        if not reused:
            row = runner.run_task(
                case["dataset"], case["task_id"], args.model, "ours", output_dir,
                max_steps=args.max_steps, policy=candidate["policy"],
                delay_ms=args.delay_ms, jitter_ms=args.jitter_ms, seed=args.seed,
                case_id=case["case_id"], scenario=case["scenario"],
                augmented=case.get("augmented", False), case_split="dev",
                augmentation=case.get("augmentation"))
        record = {field: row.get(field) for field in ROW_FIELDS}
        record["reused"] = reused
        if row.get("stale_hits"):
            record["stale_hits"] = row["stale_hits"]
        with PRINT_LOCK:
            printer(f"[rsi r{round_number} {candidate['id']}] {case['case_id']} "
                    f"status={record['status']} task_success={record['task_success']} "
                    f"active_time_s={record['active_time_s']} "
                    f"input_tokens={record['input_tokens']} "
                    f"output_tokens={record['output_tokens']} "
                    f"mcp_requests={record['mcp_requests']}"
                    + (" reused" if reused else ""))
        return record

    workers = max(1, int(getattr(args, "workers", 1) or 1))
    if workers == 1 or len(cases) == 1:
        return [evaluate(case) for case in cases]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(evaluate, cases))


def new_journal(dev_cases, run_config=None, policy_space=None):
    """Fresh JSON-serializable journal skeleton with no heldout access."""
    return {
        "policy_space": policy_space if policy_space is not None else {
            key: list(values) for key, values in POLICY_SPACE.items()},
        "heldout_access": False,
        "dev_cases": [{
            "case_id": case["case_id"], "dataset": case["dataset"],
            "task_id": case["task_id"], "scenario": case["scenario"],
            "split": case["split"], "augmented": case.get("augmented", False),
        } for case in dev_cases],
        "rounds": [],
        "selected": None,
        "reason": None,
        "selected_policy": None,
        "policy_sha256": None,
        "rounds_run": 0,
        "run_config": dict(run_config or {}),
    }


def write_journal(path, journal):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(journal, indent=2, ensure_ascii=False),
                    encoding="utf-8")


def overall_candidates(journal):
    """Every distinct candidate evaluated in any round, first occurrence wins."""
    seen, entries = set(), []
    for round_entry in journal["rounds"]:
        for entry in round_entry["candidates"]:
            digest = entry.get("policy_sha256") or policy_sha256(entry["policy"])
            if digest in seen:
                continue
            seen.add(digest)
            entries.append(entry)
    return entries


def select_overall(journal):
    """Freeze the best candidate across all rounds, not just the last round.

    Round 2 only evaluates neighbours of the round-1 winner; a neighbour can be
    worse on the nine dev cases by chance. Spec R6 selects by correctness first
    and then the weighted metric sum, so the frozen policy must be the winner
    of the union of all evaluated candidates.
    """
    entries = [entry for entry in overall_candidates(journal)
               if entry["aggregate"].get("stale_gate") != "failed"]
    if not entries:
        raise RuntimeError("no evaluated candidate in the journal passes the gates")
    aggregates = [{"id": entry["id"],
                   "success_count": entry["aggregate"]["success_count"],
                   "error_count": entry["aggregate"]["error_count"],
                   "means": entry["aggregate"]["means"]} for entry in entries]
    winner, ranked, eligible_ids, max_success = rank_candidates(aggregates)
    scores = {item["id"]: item["score"] for item in ranked}
    for entry in entries:
        entry["aggregate"]["eligible"] = entry["id"] in eligible_ids
        entry["aggregate"]["score"] = scores.get(entry["id"])
    entry = next(entry for entry in entries if entry["id"] == winner["id"])
    rounds = [round_entry["round"] for round_entry in journal["rounds"]
              if any(candidate["id"] == winner["id"]
                     for candidate in round_entry["candidates"])]
    journal["reason"] = (
        f"overall: max success_count={max_success}; "
        f"eligible={','.join(eligible_ids)}; selected={winner['id']} "
        f"(first evaluated in round {rounds[0]}); "
        f"score={winner['score']:.4f}")
    return {"id": winner["id"], "policy": entry["policy"],
            "policy_sha256": entry["policy_sha256"] or
            policy_sha256(entry["policy"]), "reason": journal["reason"]}


def finalize_journal(journal, selected):
    journal["selected"] = selected["id"]
    journal["reason"] = selected.get("reason") or journal["rounds"][-1]["reason"]
    journal["selected_policy"] = selected["policy"]
    journal["policy_sha256"] = policy_sha256(selected["policy"])
    journal["rounds_run"] = len(journal["rounds"])
    return journal


def run_round(round_number, candidates, cases, args, journal, journal_path,
              results_root=RESULTS, printer=print, max_stale_rate=None):
    """Evaluate, persist after every candidate, then select this round's winner."""
    round_entry = {"round": round_number, "candidates": [], "selected": None,
                   "reason": None}
    journal["rounds"].append(round_entry)
    entries = []
    for candidate in candidates:
        rows = evaluate_candidate(candidate, cases, args, round_number,
                                  results_root=results_root, printer=printer)
        aggregate = aggregate_rows(rows)
        rate = stale_hit_rate(rows)
        if rate is not None:
            aggregate["stale_hit_rate"] = rate
        if max_stale_rate is not None and rate is not None and rate > max_stale_rate:
            # Correctness gate: a cheaper policy that serves stale reads never wins.
            aggregate["stale_gate"] = "failed"
        entry = {"id": candidate["id"], "policy": candidate["policy"],
                 "policy_sha256": policy_sha256(candidate["policy"]),
                 "cases": rows, "aggregate": aggregate,
                 "status_counts": aggregate["status_counts"]}
        round_entry["candidates"].append(entry)
        entries.append(entry)
        write_journal(journal_path, journal)
    aggregates = [{"id": entry["id"],
                   "success_count": entry["aggregate"]["success_count"],
                   "error_count": entry["aggregate"]["error_count"],
                   "means": entry["aggregate"]["means"]} for entry in entries
                  if entry["aggregate"].get("stale_gate") != "failed"]
    if not aggregates:
        raise RuntimeError(f"round {round_number}: every candidate exceeds the "
                           f"stale-hit limit {max_stale_rate}")
    winner, ranked, eligible_ids, max_success = rank_candidates(aggregates)
    scores = {item["id"]: item["score"] for item in ranked}
    for entry in entries:
        entry["aggregate"]["eligible"] = entry["id"] in eligible_ids
        entry["aggregate"]["score"] = scores.get(entry["id"])
    round_entry["selected"] = winner["id"]
    round_entry["reason"] = _selection_reason(round_number, winner, ranked,
                                              eligible_ids, max_success)
    write_journal(journal_path, journal)
    policy = next(entry["policy"] for entry in entries
                  if entry["id"] == winner["id"])
    printer(f"[rsi r{round_number}] selected {winner['id']} "
            f"score={scores[winner['id']]:.4f}")
    return {"id": winner["id"], "policy": policy,
            "policy_sha256": policy_sha256(policy)}


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=2,
                        help="RSI rounds to run (default: 2; legacy space allows "
                             f"1-2, scheduler space 1-{MAX_SCHEDULER_ROUNDS})")
    parser.add_argument("--space", choices=("legacy", "scheduler"), default="legacy",
                        help="legacy: the five original Ours fields; scheduler: "
                             "the M1–M4 scheduler policy")
    parser.add_argument("--space-file", type=Path, default=SPACE_PLAN,
                        help="pre-registered scheduler search plan")
    parser.add_argument("--shadow-verify", action="store_true",
                        help="scheduler space: measure the stale-hit rate of every "
                             "candidate (unmetered)")
    parser.add_argument("--max-stale-rate", type=float, default=None,
                        help="scheduler space with --shadow-verify: candidates whose "
                             "stale-hit rate exceeds this are never selected")
    parser.add_argument("--model", default="DeepSeek-V4.1-Flash")
    parser.add_argument("--max-steps", type=int, default=16)
    parser.add_argument("--delay-ms", type=float, default=20)
    parser.add_argument("--jitter-ms", type=float, default=5)
    parser.add_argument("--seed", type=int, default=20261116)
    parser.add_argument("--cases", action="append",
                        help="dev case_id to evaluate; repeatable (debugging)")
    parser.add_argument("--dry-run", action="store_true",
                        help="print planned candidates and cases; run nothing")
    parser.add_argument("--manifest", type=Path, default=SCENARIOS,
                        help="case manifest (default: the 36-case smoke suite; "
                             "data/lab-cases.json for the lab server)")
    parser.add_argument("--journal", type=Path, default=JOURNAL)
    parser.add_argument("--policy-out", type=Path, default=POLICY_OUT,
                        help="frozen selected policy JSON written after a real run")
    parser.add_argument("--workers", type=int, default=1,
                        help="parallel dev cases per candidate (default: 1)")
    return parser


def _dry_run(candidates, cases, args):
    print(f"rsi dry-run: rounds={args.rounds} cases={len(cases)} "
          f"candidates={len(candidates)} model={args.model} "
          f"max_steps={args.max_steps} delay_ms={args.delay_ms} "
          f"jitter_ms={args.jitter_ms} seed={args.seed}", flush=True)
    for candidate in candidates:
        print(f"[dry-run] round 1 candidate {candidate['id']} "
              f"policy_sha256={policy_sha256(candidate['policy'])}", flush=True)
        for case in cases:
            print(f"[dry-run] round 1 {candidate['id']} {case['case_id']} "
                  "would run", flush=True)
    if args.rounds >= 2:
        print("[dry-run] round 2 candidates are derived from the round-1 "
              "winner after a real round-1 run", flush=True)
    print(json.dumps({
        "dry_run": True,
        "rounds_requested": args.rounds,
        "cases": [case["case_id"] for case in cases],
        "candidates": [{"id": candidate["id"],
                        "policy_sha256": policy_sha256(candidate["policy"])}
                       for candidate in candidates],
        "planned_runs": len(candidates) * len(cases),
        "executions": 0,
        "model": args.model,
        "max_steps": args.max_steps,
        "delay_ms": args.delay_ms,
        "jitter_ms": args.jitter_ms,
        "seed": args.seed,
    }, ensure_ascii=False), flush=True)


def main(argv=None, results_root=RESULTS):
    parser = build_parser()
    args = parser.parse_args(argv)
    scheduler_space = args.space == "scheduler"
    limit = MAX_SCHEDULER_ROUNDS if scheduler_space else 2
    if not 1 <= args.rounds <= limit:
        parser.error(f"--rounds must be between 1 and {limit} for the {args.space} space")
    if args.max_stale_rate is not None and not (scheduler_space and args.shadow_verify):
        parser.error("--max-stale-rate needs --space scheduler and --shadow-verify")
    plan = space = None
    try:
        document = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
        if document.get("suite") == "lab" and not args.cases:
            # The lab suite has no dataset x scenario grid: use every dev case.
            args.cases = [case["case_id"] for case in document["cases"]
                          if case["split"] == "dev"]
        cases = select_dev_cases(document, args.cases)
        if scheduler_space:
            space = rsi_space.tunable_space()
            plan = load_search_plan(args.space_file, space)
            candidates = scheduler_round1(plan, args.shadow_verify)
        else:
            candidates = round1_candidates()
    except (OSError, ValueError) as error:
        print(f"rsi validation failed: {error}", file=sys.stderr)
        return 2
    if args.dry_run:
        _dry_run(candidates, cases, args)
        return 0
    if scheduler_space:
        # Keep the scheduler search apart from the legacy five-field search.
        if Path(results_root) == RESULTS:
            results_root = RESULTS_ROOT / "rsi-scheduler"
        if args.journal == JOURNAL:
            args.journal = Path(results_root) / "journal.json"
        if args.policy_out == POLICY_OUT:
            args.policy_out = Path(results_root) / "selected-policy.json"
    results_root = Path(results_root)
    journal_path = Path(args.journal)
    journal = new_journal(cases, {
        "model": args.model, "max_steps": args.max_steps,
        "delay_ms": args.delay_ms, "jitter_ms": args.jitter_ms,
        "seed": args.seed, "rounds_requested": args.rounds,
        "results_root": str(results_root),
        **({"space": "scheduler", "shadow_verify": args.shadow_verify,
            "max_stale_rate": args.max_stale_rate} if scheduler_space else {}),
    }, policy_space={"plan": plan, "tunable": space} if scheduler_space else None)
    selected, evaluated, axis_start = None, set(), 0
    try:
        for round_number in range(1, args.rounds + 1):
            if round_number == 1:
                round_candidates = candidates
            elif scheduler_space:
                # Coordinate search continues from the best policy seen so far.
                best = select_overall(journal)
                round_candidates, axis_start = scheduler_neighbours(
                    best, evaluated, plan, space, axis_start, args.shadow_verify)
                if not round_candidates:
                    print("rsi: no unevaluated neighbour left; stopping", flush=True)
                    break
            else:
                round_candidates = round2_candidates(selected["policy"], evaluated)
                if not round_candidates:
                    print("rsi: no new round-2 neighbours; keeping the round-1 "
                          "winner", flush=True)
                    break
            evaluated.update(policy_sha256(candidate["policy"])
                             for candidate in round_candidates)
            selected = run_round(round_number, round_candidates, cases, args,
                                 journal, journal_path, results_root=results_root,
                                 max_stale_rate=args.max_stale_rate)
    except RuntimeError as error:
        print(f"rsi selection failed: {error}", file=sys.stderr)
        return 3
    try:
        selected = select_overall(journal)
    except RuntimeError as error:
        print(f"rsi selection failed: {error}", file=sys.stderr)
        return 3
    finalize_journal(journal, selected)
    write_journal(journal_path, journal)
    policy_out = Path(args.policy_out)
    policy_out.parent.mkdir(parents=True, exist_ok=True)
    policy_out.write_text(json.dumps(selected["policy"], indent=2,
                                     ensure_ascii=False), encoding="utf-8")
    print(f"selected: {selected['id']} "
          f"policy_sha256={journal['policy_sha256']} "
          f"rounds={journal['rounds_run']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
