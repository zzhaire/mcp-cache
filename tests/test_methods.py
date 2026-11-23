"""Method registry and Runtime adapter checks; no model or network calls."""
import json
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from exp_core.methods import ARM_NAMES, LABELS, policy_sha256, resolve_arm
from exp_core.runner import resolve_policy
from exp_core.runtime import AccessFailure, Runtime

REQUIRED_KEYS = {
    "arm", "label", "description", "disabled_modules", "m4_failure_only",
    "server_cache_ttl", "ttl_seconds", "publish_read_results",
    "publish_budget_bytes", "publish_operations", "context_dedup"}

EXPECTED = {
    "bare": dict(label="baseline", disabled_modules=("M1", "M2", "M3", "M4"),
                 m4_failure_only=False, server_cache_ttl=0, ttl_seconds=0,
                 publish_read_results=False, publish_budget_bytes=0,
                 publish_operations=False, context_dedup=False),
    "rw_openai": dict(label="Host 侧适配", disabled_modules=("M2", "M3", "M4"),
                      m4_failure_only=False, server_cache_ttl=0, ttl_seconds=0,
                      publish_read_results=False, publish_budget_bytes=0,
                      publish_operations=False, context_dedup=False),
    "rw_fastmcp": dict(label="Host 侧适配", disabled_modules=("M1", "M2", "M3", "M4"),
                       m4_failure_only=False, server_cache_ttl=3600, ttl_seconds=0,
                       publish_read_results=False, publish_budget_bytes=0,
                       publish_operations=False, context_dedup=False),
    "rw_gateway": dict(label="Host 侧适配", disabled_modules=("M4",),
                       m4_failure_only=False, server_cache_ttl=0, ttl_seconds=60,
                       publish_read_results=False, publish_budget_bytes=0,
                       publish_operations=False, context_dedup=False),
    "rw_mehan": dict(label="Host 侧适配", disabled_modules=("M1", "M2", "M3"),
                     m4_failure_only=True, server_cache_ttl=0, ttl_seconds=60,
                     publish_read_results=False, publish_budget_bytes=0,
                     publish_operations=False, context_dedup=False),
    "rw_merchant": dict(label="Host 侧适配", disabled_modules=(),
                        m4_failure_only=False, server_cache_ttl=0, ttl_seconds=60,
                        publish_read_results=False, publish_budget_bytes=0,
                        publish_operations=False, context_dedup=False),
    "ours": dict(label="本文方法", disabled_modules=(),
                 m4_failure_only=False, server_cache_ttl=0, ttl_seconds=60,
                 publish_read_results=True, publish_budget_bytes=8192,
                 publish_operations=True, context_dedup=True),
}


class Registry(unittest.TestCase):
    def test_seven_named_arms_exist(self):
        self.assertEqual(len(ARM_NAMES), 7)
        self.assertEqual(set(ARM_NAMES), set(EXPECTED))
        for name in ARM_NAMES:
            self.assertEqual(resolve_arm(name)["arm"], name)

    def test_arm_semantics_match_the_matrix(self):
        for name, expected in EXPECTED.items():
            policy = resolve_arm(name)
            self.assertLessEqual(REQUIRED_KEYS, set(policy), name)
            self.assertIn(policy["label"], LABELS, name)
            for key, value in expected.items():
                self.assertEqual(policy[key], value, f"{name}.{key}")

    def test_fixed_is_the_rw_merchant_alias(self):
        fixed, merchant = resolve_arm("fixed"), resolve_arm("rw_merchant")
        self.assertEqual(fixed, merchant)
        self.assertEqual(fixed["arm"], "rw_merchant")
        self.assertEqual(policy_sha256(fixed), policy_sha256(merchant))

    def test_unknown_arm_is_rejected(self):
        with self.assertRaises(KeyError):
            resolve_arm("candidate")

    def test_policy_hash_is_stable_and_arm_specific(self):
        hashes = {}
        for name in ARM_NAMES:
            policy = resolve_arm(name)
            digest = policy_sha256(policy)
            self.assertEqual(len(digest), 64)
            self.assertEqual(digest, policy_sha256(resolve_arm(name)), name)
            # Policies round-trip through result JSON before reruns compare hashes.
            self.assertEqual(digest, policy_sha256(json.loads(json.dumps(policy))), name)
            self.assertEqual(digest, policy_sha256(dict(reversed(list(policy.items())))), name)
            hashes[name] = digest
        self.assertEqual(len(set(hashes.values())), len(ARM_NAMES))

    def test_resolved_policy_is_an_independent_copy(self):
        policy = resolve_arm("ours")
        policy["publish_budget_bytes"] = 1
        self.assertEqual(resolve_arm("ours")["publish_budget_bytes"], 8192)

    def test_runner_keeps_fixed_and_legacy_candidate_compatibility(self):
        fixed = resolve_policy("fixed")
        self.assertEqual(fixed["arm"], "rw_merchant")
        self.assertEqual(fixed, resolve_arm("rw_merchant"))
        candidate = resolve_policy("candidate", {
            "ttl_seconds": 30, "publish_read_results": True,
            "publish_budget_bytes": 2048})
        self.assertEqual(candidate["arm"], "candidate")
        self.assertEqual(candidate["ttl_seconds"], 30)
        self.assertTrue(candidate["publish_read_results"])
        self.assertEqual(candidate["publish_budget_bytes"], 2048)
        self.assertFalse(candidate["publish_operations"])
        self.assertFalse(candidate["context_dedup"])


class CommandLine(unittest.TestCase):
    def test_help_accepts_every_arm_and_the_fixed_alias(self):
        result = subprocess.run([sys.executable, str(ROOT / "run"), "task", "--help"],
                                capture_output=True, text=True, cwd=ROOT)
        self.assertEqual(result.returncode, 0, result.stderr)
        for name in (*ARM_NAMES, "fixed"):
            self.assertIn(name, result.stdout)

    def test_unknown_method_is_rejected_before_any_task_runs(self):
        result = subprocess.run([sys.executable, str(ROOT / "run"), "task", "--method", "nope"],
                                capture_output=True, text=True, cwd=ROOT)
        self.assertEqual(result.returncode, 2)
        self.assertIn("invalid choice", result.stderr)


class M4FailureOnly(unittest.TestCase):
    def test_successful_result_is_not_reused(self):
        now = [0.0]
        runtime = Runtime(clock=lambda: now[0], m4_failure_only=True)
        calls = []

        def fetch():
            calls.append(1)
            return {"completed": len(calls)}

        self.assertEqual(runtime.call("write", {}, fetch, operation_id="op"),
                         {"completed": 1})
        self.assertEqual(runtime.call("write", {}, fetch, operation_id="op"),
                         {"completed": 2})
        self.assertEqual(len(calls), 2)
        self.assertEqual(runtime.stats["M4_hits"], 0)

    def test_failed_result_is_reused_without_a_backend_call(self):
        now = [0.0]
        runtime = Runtime(clock=lambda: now[0], m4_failure_only=True)
        calls = []

        def fetch():
            calls.append(1)
            raise AccessFailure("denied", retryable=False, outcome="not_executed")

        with self.assertRaises(AccessFailure):
            runtime.call("write", {}, fetch, operation_id="op")
        with self.assertRaises(AccessFailure) as caught:
            runtime.call("write", {}, fetch, operation_id="op")
        self.assertEqual(caught.exception.code, "denied")
        self.assertEqual(len(calls), 1)
        self.assertEqual(runtime.stats["M4_failure_hits"], 1)

    def test_default_runtime_still_reuses_successful_results(self):
        runtime = Runtime()
        calls = []

        def fetch():
            calls.append(1)
            return {"completed": len(calls)}

        runtime.call("write", {}, fetch, operation_id="op")
        self.assertEqual(runtime.call("write", {}, fetch, operation_id="op"),
                         {"completed": 1})
        self.assertEqual(len(calls), 1)
        self.assertEqual(runtime.stats["M4_hits"], 1)
        self.assertFalse(runtime.m4_failure_only)


if __name__ == "__main__":
    unittest.main()
