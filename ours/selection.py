"""Feasibility gate for measured development candidates."""
import math


def effective_policy(row):
    policy = row.get("policy") or {}
    return (policy.get("ttl_seconds", 60), policy.get("publish_read_results", False),
            policy.get("publish_budget_bytes", 0) if policy.get("publish_read_results") else 0)


def accept_candidate(candidate, reference):
    if not candidate or len(candidate) != len(reference):
        return False
    if all(effective_policy(a) == effective_policy(b) for a, b in zip(candidate, reference)):
        return False
    for a, b in zip(candidate, reference):
        if any(a.get(k) != b.get(k) for k in ("dataset", "task_id", "model", "split",
                                            "max_steps_per_turn", "runtime_source_sha256")):
            return False
        for row in (a, b):
            if (row.get("status") != "completed" or row.get("budget_exhausted")
                    or row.get("usage_unknown_attempts")
                    or not math.isfinite(row.get("task_time_s", float("inf")))):
                return False
        if a.get("task_success") is not True:
            return False
    # Correct completion takes priority over time; faster failure never wins.
    score = lambda rows: (sum(r.get("task_success") is True for r in rows),
                          -sum(r["task_time_s"] for r in rows))
    return score(candidate) > score(reference)
