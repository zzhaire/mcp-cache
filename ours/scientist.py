"""Bounded AI Scientist backend proposal/evaluation adapter, not full BFTS."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import jsonschema
from exp_core.model import generate
from ours.selection import accept_candidate
from exp_core.settings import ROOT, DATA, ARCHIVE, RESULTS

sys.path.insert(0, str(ROOT.parent / "AI-Scientist-v2"))
from ai_scientist.treesearch.backend import query
from ai_scientist.treesearch.backend.utils import FunctionSpec
from ai_scientist import traex_backend


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=1, choices=range(1, 6))
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--gate", action="store_true",
                        help="Evaluate only BFCL dev[1] and AppWorld dev[0], not the full pilot")
    args = parser.parse_args()
    output = RESULTS / "scientist"
    output.mkdir(parents=True, exist_ok=True)
    schema = json.loads((ROOT / "ours/policy.schema.json").read_text())
    manifest = json.loads((DATA / "manifest.json").read_text())
    chosen = ([("bfcl", manifest["datasets"]["bfcl"]["dev"][1]),
               ("appworld", manifest["datasets"]["appworld"]["dev"][0])] if args.gate else
              [(dataset, task) for dataset in ("bfcl", "appworld")
               for task in manifest["datasets"][dataset]["dev"][:2]])
    audit = {r["run_id"]: r for r in json.loads((ARCHIVE / "oracle-audit.json").read_text())}
    def load_row(path):
        row = json.loads(path.read_text())
        if row["run_id"] in audit:
            row["task_success"] = audit[row["run_id"]]["corrected_success"]
            row["oracle"] = audit[row["run_id"]]["oracle"]
            row["oracle_audit_applied"] = True
        return row
    sources = [ARCHIVE / "results", ARCHIVE / "paired-preflight"]
    # The serial gate has observed repeated GPT queue errors. Use the other
    # frozen model for proposal/evaluation and report that choice explicitly.
    model = manifest["models"][1 if args.gate else 0]
    feedback = []
    for path in [p for folder in sources for p in folder.glob("*.result.json")]:
        row = load_row(path)
        if row["split"] == "dev":
            feedback.append({k: row.get(k) for k in [
                "run_id", "dataset", "task_id", "model", "method", "task_success",
                "task_time_s", "input_tokens", "output_tokens", "runtime", "status",
                "runtime_source_sha256", "usage_unknown_attempts", "budget_exhausted"]})
    prior_path = output / "journal.json"
    prior = json.loads(prior_path.read_text()) if prior_path.exists() else None
    journal = {"integration": "AI Scientist-v2 treesearch.backend.query",
               "search_mode": "bounded schema proposals; full BFTS/paper pipeline not launched",
               "heldout_access": False, "requested_evaluation_tasks": chosen,
               "proposal_model": model, "evaluation_model": model,
               "gate_rule": "Only evaluate a task with a same-source completed fixed DEV reference; "
                            "blocked tasks remain in frozen pilot and are reported as skipped.",
               "previous_attempts": ([prior] if prior else []),
               "rounds": []}
    for round_index in range(args.rounds):
        trace = output / f"round-{round_index + 1}.model.jsonl"
        usages = []
        def tracked(system_message, messages, output_schema, model=None, **kwargs):
            response, usage = generate(model or journal["proposal_model"],
                str(system_message) + "\n" + json.dumps(messages), trace, output_schema)
            usages.append(usage)
            return response
        # Replace only transport for real usage, retaining AI Scientist query/spec routing.
        traex_backend.traex_chat_json = tracked
        start = time.monotonic()
        try:
            candidate = query(
                "Propose a constrained MCP Runtime policy. Never change task data, "
                "tools, oracles, authorization or validity checks. Use only provided "
                "development feedback. Reads are invalidated on all writes and auth "
                "changes. Publishing means adding already valid read results before "
                "the model's next decision; all added tokens count. Do not weaken "
                "the baseline to manufacture improvements. Return the schema.",
                json.dumps({"initial_policy": {"ttl_seconds": 60, "publish_read_results": False,
                                    "publish_budget_bytes": 0},
                 "feedback": feedback,
                 "limitations": "M3 has a tested contract-gated store but no benchmark "
                    "validity contract. M4 conditional failure recovery is not implemented "
                    "in this first smoke; do not claim their gains."}),
                model=model,
                func_spec=FunctionSpec("runtime_policy", schema, "Constrained Runtime policy"))
            jsonschema.validate(candidate, schema)
            record = {"round": round_index + 1, "candidate": candidate,
                      "proposal_time_s": time.monotonic() - start, "actual_usage": usages,
                      "accepted": False, "status": "proposed_not_evaluated"}
            if args.evaluate:
                from exp_core.runner import run_task
                fixed, eligible, skipped = [], [], []
                for dataset, task in chosen:
                    matches = [load_row(p) for folder in sources for p in folder.glob("*.result.json")]
                    matches = [r for r in matches if r["dataset"] == dataset and r["task_id"] == task
                               and r["model"] == model and r["method"] == "fixed"
                               and r["status"] == "completed"
                               and r.get("runtime_source_sha256") == hashlib.sha256(
                                   (ROOT / "exp_core/runtime.py").read_bytes()).hexdigest()
                               and not r.get("budget_exhausted") and not r.get("usage_unknown_attempts")]
                    if args.gate and not matches:
                        skipped.append({"dataset": dataset, "task_id": task,
                                        "reason": "no valid completed paired fixed reference for evaluation model"})
                        continue
                    fixed.append(matches[0] if matches else run_task(dataset, task,
                        model, "fixed", output / "evaluations", 16))
                    eligible.append((dataset, task))
                records = [run_task(dataset, task, model, "candidate",
                                    output / "evaluations", 16, candidate) for dataset, task in eligible]
                record["evaluated_tasks"] = eligible
                record["skipped_tasks"] = skipped
                record["evaluation_run_ids"] = [r["run_id"] for r in records]
                record["reference_run_ids"] = [r["run_id"] for r in fixed]
                record["accepted"] = accept_candidate(records, fixed)
                record["selection_rule"] = (
                    "Effective policy must change. All candidate tasks correct; "
                    "both arms completed, budget/usage valid, "
                    "same task/model/source/cap. Maximize correctness then minimize sum time. "
                    "Single-run DEV selection is exploratory; heldout not evaluated.")
                record["status"] = "evaluated"
                feedback.extend({k: r.get(k) for k in ["dataset", "task_id", "task_success",
                                                     "task_time_s", "runtime"]} for r in records)
            journal["rounds"].append(record)
        except Exception as error:
            journal["rounds"].append({"round": round_index + 1, "status": "error",
                                      "error": str(error), "actual_usage": usages})
        (output / "journal.json").write_text(json.dumps(journal, indent=2))
        print(json.dumps(journal["rounds"][-1]), flush=True)


if __name__ == "__main__":
    main()
