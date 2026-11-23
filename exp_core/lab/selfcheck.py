"""Model-free checks of the lab server: ``./run lab-check [--server]``.

Without flags it replays every task's scripted solution in process under each
method and prints which pass and how many requests they needed. With
``--server`` it drives the same scripted agent through the real MCP server and
the normal runner, and verifies both paths agree. No model is called.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from unittest import mock

from exp_core.settings import DATA
from exp_core.lab.factors import CONTRACT_LEVELS, LISTINGS
from exp_core.lab.harness import TurnScript, load_tasks, run_gold
from exp_core.lab.domains import make_world

ARMS = (("bare", "bare", None), ("openai", "rw_openai", None),
        ("fastmcp", "rw_fastmcp", None), ("gateway", "rw_gateway", None),
        ("mehan", "rw_mehan", None), ("merchant", "rw_merchant", None),
        ("ours-old", "ours", {"context_dedup": False}),
        ("ours-new", "ours", {"scheduler": "default"}))
# Not part of the design document's comparison; shown with --strong.
STRONG = ("strong", "strong", None)


SEED = 20261116


def scripted_agent(task):
    """A fake model that plays ``task["gold"]`` from the conversation it is shown."""
    contract = make_world(task).contract()
    state = {"turn": 0, "script": None, "pending": [], "logins": {}}

    def generate(model, context, trace, *args, **kwargs):
        start = context.index("Conversation:\n") + len("Conversation:\n")
        messages, _ = json.JSONDecoder().raw_decode(context[start:])
        shown, full = [], {}
        for message in messages:
            if message["role"] != "tool":
                continue
            content = json.loads(message["content"])
            if "result" in content:
                canonical = json.dumps(content, sort_keys=True, separators=(",", ":"))
                full[hashlib.sha256(canonical.encode()).hexdigest()[:16]] = content["result"]
                shown.append(content["result"])
            else:
                # A large repeated result arrives as a pointer; read it where
                # it was first shown, as a model would.
                shown.append(full[content["result_ref"]])
        if state["script"] is None:
            state["script"] = TurnScript(task["gold"][state["turn"]], contract, state["logins"])
        elif state["pending"] is not None:
            calls = state["pending"]
            state["script"].record(calls, shown[len(shown) - len(calls):] if calls else [])
        calls = state["script"].next_calls()
        if calls is None:
            answer = state["script"].answer()
            state.update(turn=state["turn"] + 1, script=None, pending=[])
            return {"calls": [], "done": True, "answer": answer}, {}
        state["pending"] = calls
        return {"calls": [{"name": call["name"], "arguments": call["arguments"]}
                          for call in calls], "done": False, "answer": ""}, {}
    return generate


SUITES = {"basic": ["lab-cases.json"], "stress": ["lab-stress-cases.json"],
          "long": ["lab-long-cases.json"],
          "gain": ["lab-gain-cases.json"],
          "heldout": ["lab-heldout-cases.json"],
          "all": ["lab-cases.json", "lab-stress-cases.json"]}


def main(argv=None):
    parser = argparse.ArgumentParser(prog="run lab-check", description=__doc__)
    parser.add_argument("--server", action="store_true",
                        help="also run through the real MCP server and compare")
    parser.add_argument("--tasks", nargs="*", help="task ids (default: the whole suite)")
    parser.add_argument("--suite", choices=tuple(SUITES), default="basic",
                        help="basic: data/lab-cases.json; stress: the extreme cases; "
                             "heldout: the frozen held-out cases; all: basic + stress")
    parser.add_argument("--listing", choices=LISTINGS, default="per_turn",
                        help="how often the agent framework asks for the tool list")
    parser.add_argument("--contract", choices=CONTRACT_LEVELS, default="full",
                        help="how much of the tool contract every host is given")
    parser.add_argument("--why", action="store_true",
                        help="print the failed oracle checks of every failing run")
    parser.add_argument("--strong", action="store_true",
                        help="also show the capability-equal fixed-rule baseline")
    args = parser.parse_args(argv)
    from exp_core import runner
    arms = ARMS[:-1] + ((STRONG,) if args.strong else ()) + ARMS[-1:]
    wanted = set()
    for name in SUITES[args.suite]:
        wanted |= {case["task_id"] for case in json.loads(
            (DATA / name).read_text(encoding="utf-8"))["cases"]}
    tasks = {tid: task for tid, task in load_tasks().items()
             if (tid in args.tasks if args.tasks else tid in wanted)}
    reasons = []
    print(f"{'task':34}" + "".join(f"{arm[0]:>10}" for arm in arms))
    totals = {arm[0]: {"passed": 0, "requests": 0, "catalog": 0, "stale": 0, "revoked": 0,
                       "unsafe": 0} for arm in arms}
    mismatches = 0
    with tempfile.TemporaryDirectory() as temp:
        for tid, task in tasks.items():
            line = f"{tid:34}"
            for label, method, override in arms:
                # Same seed as the runner, so sampled verification draws the same hits.
                gold = run_gold(task, runner.resolve_policy(method, override), seed=SEED,
                                shadow_verify=True, listing=args.listing,
                                contract_level=args.contract)
                passed, requests = gold["oracle"]["success"], gold["requests"]
                delivery = gold["delivery"]
                note = ""
                if not passed:
                    reasons.append((tid, label, [
                        f"{c['check']}: want {c['want']}, got {c['got']}"
                        for c in gold["oracle"]["checks"] if not c["ok"]]))
                if args.server:
                    with mock.patch.object(runner, "generate", scripted_agent(task)), \
                            mock.patch("builtins.print"):
                        row = runner.run_task(
                            "lab", tid, "scripted", method, Path(temp) / tid / label,
                            max_steps=16, policy={**(override or {}), "eval_shadow_verify": True},
                            case_id=tid, case_split="dev", seed=SEED,
                            lab_options={"listing": args.listing, "contract": args.contract,
                                         "audit": True})
                    # The in-process replay has no "still valid" reference
                    # channel, so only results handed back for a call are compared.
                    results = [d for d in (row.get("delivery") or {}).get("stale", [])
                               if d["where"] == "result"]
                    seen = (row.get("task_success"), row.get("mcp_requests"),
                            row.get("mcp_requests_catalog"), results)
                    if seen != (passed, requests, gold["catalog_requests"],
                                gold["stale_deliveries"]):
                        mismatches += 1
                        note = "!"
                total = totals[label]
                total["passed"] += passed
                total["requests"] += requests
                total["catalog"] += gold["catalog_requests"]
                total["stale"] += delivery["stale"] + delivery["stale_references"]
                total["revoked"] += delivery["after_revocation"]
                total["unsafe"] += not gold["oracle"].get("safe", True)
                line += f"{('ok' if passed else 'FAIL') + ' ' + str(requests) + note:>10}"
            print(line, flush=True)
    for title, key in (("passed", "passed"), ("requests", "requests"),
                       ("  of which catalog", "catalog"),
                       ("stale values delivered", "stale"),
                       ("delivered after revocation", "revoked"),
                       ("runs with an unsafe write", "unsafe")):
        print(f"{title:34}" + "".join(f"{v[key]:>10}" for v in totals.values()))
    if args.why:
        for tid, label, failed in reasons:
            print(f"- {tid} / {label}: " + "; ".join(failed))
    if args.server:
        print(f"server runs disagreeing with the in-process replay: {mismatches}")
    return 1 if mismatches else 0
