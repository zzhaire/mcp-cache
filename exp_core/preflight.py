"""Run M1–M4 coverage, then two scenario-selected tasks on one stable model."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from exp_core.runner import run_task
from exp_core.settings import ROOT, DATA, RESULTS



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="DeepSeek-V4.1-Flash")
    parser.add_argument("--max-steps", type=int, default=16)
    parser.add_argument("--output", type=Path, default=RESULTS / "preflight")
    args = parser.parse_args()
    subprocess.run([sys.executable, "-m", "exp_core.checks"], check=True, cwd=ROOT)
    scenarios = json.loads((DATA / "smoke-scenarios.json").read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for scenario in scenarios["agent_tasks"]:
        row = run_task(scenario["dataset"], scenario["task"], args.model, "fixed",
                       args.output, args.max_steps, **scenarios["network"])
        rows.append(row)
        summary = {"model": args.model, "method": "fixed",
                   "selection_basis": scenarios["selection_basis"],
                   "requested_tasks": scenarios["agent_tasks"],
                   "rows": rows,
                   "status": "running" if len(rows) < len(scenarios["agent_tasks"]) else
                       ("passed" if all(r["task_success"] is True for r in rows) else
                        "agent_task_failure"),
                   "scope": "module coverage and bounded task correctness, not a gain experiment",
                   "scientist": "v1 selected; multi-round search waits for stable smoke"}
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
        if row["status"] == "error":
            summary["status"] = "infrastructure_error"
            summary["not_run_tasks"] = scenarios["agent_tasks"][len(rows):]
            (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
            return 1
    return 0 if all(r["task_success"] is True for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
