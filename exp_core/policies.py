"""Shared policy contract and canonical hashing; no method implementation."""
import hashlib
import json


def make_policy(arm, label, description, disabled_modules, *,
                m4_failure_only=False, server_cache_ttl=0, ttl_seconds=60,
                publish_read_results=False, publish_budget_bytes=0,
                publish_operations=False, context_dedup=False):
    return {
        "arm": arm, "label": label, "description": description,
        "disabled_modules": tuple(disabled_modules),
        "m4_failure_only": m4_failure_only,
        "server_cache_ttl": server_cache_ttl, "ttl_seconds": ttl_seconds,
        "publish_read_results": publish_read_results,
        "publish_budget_bytes": publish_budget_bytes,
        "publish_operations": publish_operations,
        "context_dedup": context_dedup,
    }


def policy_sha256(policy):
    canonical = json.dumps(policy, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()
