"""Derive the full oracle of a lab task from its reference solution.

The hand-written part of ``expected`` names a few facts. This module runs the
task's scripted solution once, with no cache and with the injected faults
removed (a fault must not change what the right outcome is), and records:

- ``final_state``: the whole world state the run must end in;
- ``write_log``: every write that must be committed, with its arguments;
- ``labels`` per answer rule: the entity word each required value belongs to,
  read from the reference answer ("checking 500" -> 500 belongs to "checking");
- ``refused_write`` per answer rule: the reference shows a write refused in
  that turn, so an answer claiming success there needs a second look.

The derived oracle must agree with the hand-written one; ``enrich`` raises if
the reference solution does not pass its own oracle.
"""
import copy
import re

from exp_core.lab.harness import run_gold
from exp_core.lab.judge import occurrences
from exp_core.lab.domains import make_world

BARE = {"arm": "bare", "disabled_modules": ("M1", "M2", "M3", "M4"), "m4_failure_only": False,
        "ttl_seconds": 0, "server_cache_ttl": 0}


def _label(answer, token):
    """The entity word in front of ``token`` in the reference answer."""
    for place in occurrences(answer, token):
        segment = re.split(r"[,;]", answer[:place])[-1]
        words = re.sub(r"[^A-Za-z0-9\- ]", " ", segment).split()
        # "checking balance:" names the account; generic attribute words follow it.
        words = [w for w in words if w.lower() not in {"balance", "is", "the", "of"}]
        if words and any(ch.isalpha() for ch in " ".join(words)):
            return words[0]  # the entity; attribute words after it vary in real answers
    return None


def enrich(task):
    """Return ``task`` with the derived oracle fields filled in."""
    task = copy.deepcopy(task)
    expected = task["expected"]
    for key in ("final_state", "write_log"):
        expected.pop(key, None)
    for rule in expected.get("answers", []):
        rule.pop("labels", None)
        rule.pop("refused_write", None)
    reference = run_gold({**copy.deepcopy(task), "faults": []}, BARE)
    if not reference["oracle"]["success"]:
        failed = [c for c in reference["oracle"]["checks"] if not c["ok"]]
        raise ValueError(f"{task['id']}: the reference solution fails its own oracle: {failed}")
    contract = make_world(task).contract()
    refused_turns = {error["turn"] for error in reference["errors"]
                     if contract.get(error["tool"], {}).get("kind") == "write"
                     and error["error"] == "business_error"}
    expected["final_state"] = reference["final_state"]
    expected["write_log"] = [{"tool": w["tool"], "arguments": w["arguments"]}
                             for w in reference["writes"]]
    for rule in expected.get("answers", []):
        answer = reference["answers"][rule["turn"]]
        labels = {}
        for token in rule.get("contains", []):
            label = _label(answer, token)
            if label:
                labels[str(token)] = label
        if labels:
            rule["labels"] = labels
        if rule["turn"] in refused_turns:
            rule["refused_write"] = True
    return task
