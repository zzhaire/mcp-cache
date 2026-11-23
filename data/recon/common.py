"""Shared pieces of the rebuilt benchmark tasks."""
import copy

from data.lab_expect import BARE
from exp_core.lab.harness import run_gold

USER, PASSWORD = "user", "pw-user"
# The sentence every rebuilt first turn carries; overlays split the turn there.
SIGN_IN = "Sign in to an app before using it. "


def login_step(apps):
    return [{"name": "auth_login", "arguments": {"app": app, "username": USER,
                                                 "password": PASSWORD}} for app in sorted(apps)]


def reference(task):
    """The task's scripted solution with no cache and no injected fault."""
    return run_gold({**copy.deepcopy(task), "faults": [], "expected": {"writes": {}}}, BARE)


def finish(task, checks=()):
    """Fill ``expected`` from the reference run.

    ``checks`` lists ``(turn, alias, field, label)``: the value the reference
    run read there must be stated in that turn's answer. The final state and
    the committed writes are taken from the reference run as a whole.
    """
    task = copy.deepcopy(task)
    run = reference(task)
    if run["errors"] and not task.get("reference_errors_ok"):
        raise ValueError(f"{task['id']}: reference run has errors: {run['errors'][:3]}")
    answers = []
    for turn, token, label in checks:
        answers.append({"turn": turn, "contains": [token], "labels": {str(token): label}})
    task["expected"] = {
        "final_state": run["final_state"],
        "write_log": [{"tool": w["tool"], "arguments": w["arguments"]} for w in run["writes"]],
        "answers": answers}
    return task, run
