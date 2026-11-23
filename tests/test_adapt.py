"""Replay a recorded no-cache run under another policy and fit a policy from it."""
import glob
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from exp_core import runner  # noqa: E402
from exp_core.lab.harness import load_tasks, run_gold  # noqa: E402
from exp_core.lab.selfcheck import SEED, scripted_agent  # noqa: E402
from ours.adapt import adapt, profile_run  # noqa: E402
from ours.replay import load_run, replay  # noqa: E402
from exp_core.lab.world import LabWorld  # noqa: E402

TASKS = load_tasks()
NEW = {"scheduler": "default"}
RUNS = {}


def recorded(task_id):
    """The no-cache run of the scripted agent through the real server."""
    if task_id not in RUNS:
        with tempfile.TemporaryDirectory() as temp:
            with mock.patch.object(runner, "generate", scripted_agent(TASKS[task_id])), \
                    mock.patch("builtins.print"):
                runner.run_task("lab", task_id, "scripted", "bare", Path(temp), max_steps=8,
                                case_id=task_id, case_split="dev", seed=SEED)
            RUNS[task_id] = load_run(glob.glob(f"{temp}/*.jsonl")[0])
    return RUNS[task_id]


def policy(override=NEW):
    return runner.resolve_policy("ours", override)


class ReplayFidelity(unittest.TestCase):
    def test_no_cache_replay_reproduces_the_recorded_request_count(self):
        for tid in ("lab-repeat-read", "lab-fault-rate-limit", "m3-session-churn"):
            bare = runner.resolve_policy("bare")
            self.assertEqual(replay(recorded(tid), bare, TASKS)["requests"],
                             run_gold(TASKS[tid], runner.resolve_policy("bare"))["requests"], tid)

    def test_replay_predicts_the_scheduler_request_count(self):
        # Exact on plain cases; recovery paths are approximate, so the suite
        # total only has to stay within a few percent.
        for tid in ("lab-repeat-read", "lab-write-disjoint", "m2-read-storm",
                    "m2-write-interleave", "m3-session-churn", "m3-relogin-storm"):
            self.assertEqual(replay(recorded(tid), policy(), TASKS)["requests"],
                             run_gold(TASKS[tid], policy())["requests"], tid)

    def test_replay_flags_exactly_the_cases_that_serve_stale_values(self):
        for tid, stale in (("lab-contract-drift", True), ("m2-contract-drift-repeated", True),
                           ("lab-external-volatile", False), ("m3-user-switch", False),
                           ("m3-logout-must-not-serve", False)):
            out = replay(recorded(tid), policy(), TASKS)
            self.assertEqual(bool(out["stale_hits"]), stale, tid)

    def test_cached_value_after_a_dropped_session_is_reported_separately(self):
        out = replay(recorded("lab-auth-expiry"), policy(), TASKS)
        self.assertEqual((out["stale_hits"], out["served_after_revocation"]), (0, 1))


class Profile(unittest.TestCase):
    def test_external_change_is_told_apart_from_the_agents_own_write(self):
        def tools(tid):
            return profile_run(recorded(tid), LabWorld(TASKS[tid]).contract())["tools"]
        drift = tools("m2-contract-drift-repeated")["bank_get_balance"]
        self.assertEqual((drift["changed_externally"], drift["changed_after_own_write"]), (3, 0))
        own = tools("m2-write-interleave")["bank_get_balance"]
        self.assertEqual(own["changed_externally"], 0)
        self.assertGreater(own["changed_after_own_write"], 0)

    def test_another_users_view_is_not_an_external_change(self):
        tid = "m3-user-switch"
        tools = profile_run(recorded(tid), LabWorld(TASKS[tid]).contract())["tools"]
        self.assertEqual(sum(tool["changed_externally"] for tool in tools.values()), 0)


class Adapt(unittest.TestCase):
    def test_drifting_tool_is_tightened_and_the_stale_values_disappear(self):
        tid = "m2-contract-drift-repeated"
        result = adapt([recorded(tid)], TASKS)
        self.assertIn(result["policy"]["ttl"]["fresh_class_overrides"]["bank_get_balance"],
                      {"volatile", "strict"})
        self.assertGreater(result["replay"]["base"]["stale"], 0)
        self.assertEqual(result["replay"]["adapted"]["stale"], 0)
        # The fitted policy passes the case the default policy fails.
        self.assertFalse(run_gold(TASKS[tid], policy())["oracle"]["success"])
        self.assertTrue(run_gold(TASKS[tid], policy({"scheduler": result["policy"]}))
                        ["oracle"]["success"])

    def test_stable_scenario_keeps_its_savings(self):
        tid = "m2-read-storm"
        result = adapt([recorded(tid)], TASKS)
        self.assertEqual(result["policy"]["ttl"]["fresh_class_overrides"], {})
        self.assertEqual(result["replay"]["adapted"]["stale"], 0)
        self.assertLessEqual(result["replay"]["adapted"]["requests"],
                             result["replay"]["base"]["requests"])
        self.assertLess(result["replay"]["adapted"]["requests"],
                        result["replay"]["no_cache_requests"])


if __name__ == "__main__":
    unittest.main()
