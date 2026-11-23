"""Summarise lab runs, with repetition and uncertainty.

``./run lab-report --root results/<run> [--baseline bare]``

Reads every completed lab result under the root (including ``rep<N>``
folders written by ``--repeats``). For each method it reports how many runs
passed, needed review, failed or applied an unsafe write, and the mean number
of requests, split into listing the tools and everything else. Against the
baseline method it reports the paired difference per case (averaged over the
repetitions of that case) with a 95% bootstrap interval over cases.

With one run per case and method the interval only reflects which cases were
chosen, not how much a model varies from run to run; the report says so.
"""
import argparse
import glob
import json
import random
import re
import statistics
from collections import defaultdict
from pathlib import Path


def label(row):
    name = row["method"]
    if name == "ours" and row.get("scheduler_policy_sha256"):
        # Two parameter sets are two methods, even under the same preset name.
        return f"ours[{row.get('scheduler_preset')}:{row['scheduler_policy_sha256'][:6]}]"
    return name


def load(root):
    """{label: {case: [run, ...]}} of the completed lab runs under ``root``."""
    runs = defaultdict(lambda: defaultdict(list))
    for path in sorted(glob.glob(f"{root}/**/*.result.json", recursive=True)):
        row = json.loads(Path(path).read_text(encoding="utf-8"))
        if row.get("dataset") != "lab" or row.get("status") != "completed":
            continue
        oracle = row.get("oracle") or {}
        delivery = row.get("delivery") or {}
        repeat = re.search(r"/rep(\d+)/", path)
        runs[label(row)][row["task_id"]].append({
            "repeat": int(repeat.group(1)) if repeat else 0,
            "verdict": oracle.get("verdict", "pass" if row.get("task_success") else "fail"),
            "safe": oracle.get("safe", True),
            "requests": row.get("mcp_requests", 0),
            "catalog": row.get("mcp_requests_catalog", 0),
            "model_calls": row.get("model_calls", 0),
            "input_tokens": row.get("input_tokens", 0),
            "stale": delivery.get("stale", 0) if isinstance(delivery.get("stale"), int)
            else len([d for d in delivery.get("stale", []) if d.get("kind") == "stale"]),
            "audited": bool(delivery)})
    return runs


def _mean(values):
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def bootstrap(differences, rounds=4000, seed=20261120):
    """Mean and 95% interval of per-case differences, resampling cases."""
    if not differences:
        return None
    rng = random.Random(seed)
    means = sorted(_mean(rng.choices(differences, k=len(differences))) for _ in range(rounds))
    return (_mean(differences), means[int(0.025 * rounds)], means[int(0.975 * rounds) - 1])


def summarise(runs, baseline="bare"):
    out = {}
    for name, cases in runs.items():
        flat = [run for group in cases.values() for run in group]
        repeats = max(len(group) for group in cases.values())
        per_repeat = defaultdict(int)
        for group in cases.values():
            for run in group:
                per_repeat[run["repeat"]] += run["requests"]
        out[name] = {
            "cases": len(cases), "runs": len(flat), "repeats": repeats,
            "pass": sum(r["verdict"] == "pass" for r in flat),
            "review": sum(r["verdict"] == "review" for r in flat),
            "fail": sum(r["verdict"] == "fail" for r in flat),
            "unsafe": sum(not r["safe"] for r in flat),
            "requests": _mean(r["requests"] for r in flat),
            "catalog": _mean(r["catalog"] for r in flat),
            "other": _mean(r["requests"] - r["catalog"] for r in flat),
            "model_calls": _mean(r["model_calls"] for r in flat),
            "input_tokens": _mean(r["input_tokens"] for r in flat),
            "stale_deliveries": sum(r["stale"] for r in flat),
            "audited_runs": sum(r["audited"] for r in flat),
            "requests_total_sd_over_repeats": (statistics.stdev(per_repeat.values())
                                               if len(per_repeat) > 1 else None)}
        if baseline in runs and name != baseline:
            shared = sorted(set(cases) & set(runs[baseline]))
            for metric in ("requests", "other"):
                value = (lambda r: r["requests"] - r["catalog"]) if metric == "other" else \
                    (lambda r: r["requests"])
                out[name][f"{metric}_vs_{baseline}"] = bootstrap(
                    [_mean(map(value, cases[c])) - _mean(map(value, runs[baseline][c]))
                     for c in shared])
            out[name][f"pass_rate_vs_{baseline}"] = bootstrap(
                [_mean(r["verdict"] == "pass" for r in cases[c])
                 - _mean(r["verdict"] == "pass" for r in runs[baseline][c]) for c in shared])
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(prog="run lab-report", description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", type=Path, action="append", required=True)
    parser.add_argument("--baseline", default="bare")
    parser.add_argument("--json", type=Path, help="also write the summary as JSON")
    args = parser.parse_args(argv)
    runs = defaultdict(lambda: defaultdict(list))
    for root in args.root:
        for name, cases in load(root).items():
            for case, group in cases.items():
                runs[name][case].extend(group)
    if not runs:
        raise ValueError("no completed lab runs under the given roots")
    summary = summarise(runs, args.baseline)
    print(f"{'method':22}{'cases':>6}{'runs':>6}{'pass':>6}{'review':>7}{'fail':>6}"
          f"{'unsafe':>7}{'requests':>10}{'catalog':>9}{'other':>8}{'stale':>7}")
    for name, s in summary.items():
        print(f"{name:22}{s['cases']:>6}{s['runs']:>6}{s['pass']:>6}{s['review']:>7}"
              f"{s['fail']:>6}{s['unsafe']:>7}{s['requests']:>10.2f}{s['catalog']:>9.2f}"
              f"{s['other']:>8.2f}{s['stale_deliveries']:>7}")
    print(f"\npaired difference to {args.baseline}, per case: mean [95% bootstrap interval]")
    for name, s in summary.items():
        for key in (f"requests_vs_{args.baseline}", f"other_vs_{args.baseline}",
                    f"pass_rate_vs_{args.baseline}"):
            if s.get(key):
                mean, low, high = s[key]
                print(f"  {name:22}{key:24}{mean:>9.3f}  [{low:.3f}, {high:.3f}]")
    single = [name for name, s in summary.items() if s["repeats"] < 2]
    if single:
        print("\nOne run per case for: " + ", ".join(single) + ". The intervals above only "
              "reflect the choice of cases, not run-to-run variation of the model.")
    unaudited = [name for name, s in summary.items() if s["audited_runs"] < s["runs"]]
    if unaudited:
        print("Stale deliveries were not audited in every run of: " + ", ".join(unaudited)
              + " (run with --shadow-verify).")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(summary, ensure_ascii=False, indent=1) + "\n",
                             encoding="utf-8")
    return 0
