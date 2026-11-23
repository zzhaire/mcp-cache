"""RSI over the M1–M4 scheduler policy space; no model, no server."""
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from exp_core.methods import policy_sha256  # noqa: E402
from exp_core.scheduler_policy import resolve, validate  # noqa: E402
from ours import rsi, rsi_space  # noqa: E402

CASES = ["bfcl-read-dev-1", "tau2-read-dev-1"]


def fake_run_task(dataset, task_id, model, method, output_dir, **kwargs):
    """Cost falls as ttl.slow grows and as the publish budget shrinks."""
    scheduler = kwargs["policy"]["scheduler"]
    # Only step-based TTLs count in this fake world (the ablation preset uses seconds).
    slow = scheduler["ttl"]["slow"] if scheduler["ttl"]["unit"] == "steps" else 24
    row = {"case_id": kwargs["case_id"], "status": "completed", "task_success": True,
           "active_time_s": 100.0 - min(slow, 96) / 2,
           "input_tokens": 1000 + scheduler["publish"]["budget_bytes"] // 16,
           "output_tokens": 20,
           "mcp_requests": 40 - min(slow, 96) / 4}
    if kwargs["policy"].get("eval_shadow_verify"):
        # Long TTLs are cheaper but serve stale reads in this fake world.
        stale = 5 if slow > 24 else 0
        row["stale_hits"] = {"shadow_checks": 10, "shadow_stale": stale,
                             "stale_hit_rate": stale / 10}
    return row


def run_main(argv, temp):
    journal, policy_out = Path(temp) / "journal.json", Path(temp) / "policy.json"
    out, err = StringIO(), StringIO()
    with mock.patch.object(rsi.runner, "run_task", side_effect=fake_run_task) as run_task, \
            redirect_stdout(out), redirect_stderr(err):
        code = rsi.main(["--space", "scheduler", "--journal", str(journal),
                         "--policy-out", str(policy_out),
                         *[arg for case in CASES for arg in ("--cases", case)], *argv],
                        results_root=Path(temp) / "rsi")
    loaded = json.loads(journal.read_text()) if journal.exists() else None
    frozen = json.loads(policy_out.read_text()) if policy_out.exists() else None
    return code, run_task, loaded, frozen, out.getvalue() + err.getvalue()


class TunableSpace(unittest.TestCase):
    def setUp(self):
        self.space = rsi_space.tunable_space()
        self.base = resolve("default")

    def test_locked_parameters_are_not_tunable(self):
        self.assertFalse([path for path in self.space if path.startswith("locked")])
        self.assertNotIn("tiers.M2", self.space)  # const in the schema
        self.assertEqual(self.space["ttl.slow"]["tier"], "L2")
        self.assertEqual(self.space["invalidation.granularity"]["tier"], "L1")
        self.assertEqual(self.space["ttl.unit"]["tier"], "L1")  # leaf overrides parent

    def test_numeric_neighbours_scale_and_clip_to_the_schema_range(self):
        labels = lambda path, policy=None: [label for label, _ in rsi_space.neighbours(
            policy or self.base, path, self.space)]
        self.assertEqual(labels("ttl.slow"), ["12", "48"])
        top = rsi_space.with_value(self.base, "ttl.volatile", 600)
        self.assertEqual(labels("ttl.volatile", top), ["300"])  # 1200 clips to the max
        zero = rsi_space.with_value(self.base, "forget.idle_steps", 0)
        self.assertEqual(labels("forget.idle_steps", zero), ["1"])
        self.assertEqual(labels("ttl.static"), [])  # null = no bound, left alone

    def test_switch_neighbours(self):
        labels = lambda path: [label for label, _ in rsi_space.neighbours(
            self.base, path, self.space)]
        self.assertEqual(labels("publish.ops_first"), ["true"])
        self.assertEqual(labels("invalidation.granularity"), ["global", "server", "app"])
        self.assertEqual(len(labels("coupling.enabled_rules")), 4)  # toggle each rule
        self.assertEqual(labels("recovery.allow_revalidate"), [])  # fails validation

    def test_share_neighbours_keep_summing_to_one(self):
        result = rsi_space.neighbours(self.base, "capacity.share", self.space)
        self.assertEqual([label for label, _ in result], ["+M1", "+M2", "+M3", "+M4"])
        for _, policy in result:
            self.assertAlmostEqual(sum(policy["capacity"]["share"].values()), 1.0)

    def test_every_neighbour_of_every_axis_is_a_valid_policy(self):
        for path in list(self.space) + ["capacity.share"]:
            if path.startswith("capacity.share."):
                continue
            for _, policy in rsi_space.neighbours(self.base, path, self.space):
                validate(policy)
                self.assertEqual(policy["locked"], self.base["locked"])

    def test_axes_outside_the_space_are_rejected(self):
        for path in ("locked.retry_max_attempts", "capacity.share.M1", "no.such.key"):
            with self.assertRaises(ValueError):
                rsi_space.check_axes([path], self.space)
            with self.assertRaises(ValueError):
                rsi_space.neighbours(self.base, path, self.space)


class SearchPlan(unittest.TestCase):
    def test_shipped_plan_is_valid(self):
        plan = rsi.load_search_plan()
        self.assertIn("default", plan["round1_presets"])
        self.assertIn("ablation_current", plan["round1_presets"])

    def test_plan_with_a_locked_axis_is_refused(self):
        plan = json.loads(rsi.SPACE_PLAN.read_text())
        plan["axes"].append("locked.m2_cross_task")
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "plan.json"
            path.write_text(json.dumps(plan))
            with self.assertRaises(ValueError):
                rsi.load_search_plan(path)

    def test_round_one_candidates_are_complete_ours_policies(self):
        plan = rsi.load_search_plan()
        candidates = rsi.scheduler_round1(plan)
        self.assertEqual([c["id"] for c in candidates], plan["round1_presets"])
        for candidate in candidates:
            policy = candidate["policy"]
            self.assertEqual(policy["arm"], "ours")
            self.assertEqual(policy["scheduler"]["preset"], candidate["id"])
            self.assertEqual(policy["publish_budget_bytes"],
                             policy["scheduler"]["publish"]["budget_bytes"])
        self.assertEqual(len({policy_sha256(c["policy"]) for c in candidates}), len(candidates))

    def test_neighbours_respect_limit_skip_evaluated_and_resume_axis_order(self):
        plan, space = rsi.load_search_plan(), rsi_space.tunable_space()
        base = rsi.scheduler_round1(plan)[0]
        first, position = rsi.scheduler_neighbours(base, set(), plan, space)
        self.assertEqual(len(first), plan["round_limit"])
        self.assertTrue(first[0]["id"].startswith("ttl.slow_12@"))
        seen = {policy_sha256(c["policy"]) for c in first}
        second, _ = rsi.scheduler_neighbours(base, seen, plan, space, position)
        self.assertFalse(seen & {policy_sha256(c["policy"]) for c in second})
        self.assertNotEqual(list(second[0]["overrides"]), list(first[0]["overrides"]))
        for candidate in first + second:
            self.assertRegex(candidate["id"], r"^[A-Za-z0-9._+@-]+$")
            self.assertEqual(len(candidate["overrides"]), 1)
            self.assertTrue(candidate["policy"]["scheduler"]["preset"].endswith("*"))
            self.assertFalse(candidate["policy"]["scheduler"]["preset"].endswith("**"))


class SchedulerSearchRun(unittest.TestCase):
    def test_search_moves_toward_lower_cost_and_freezes_a_usable_policy(self):
        with tempfile.TemporaryDirectory() as temp:
            code, run_task, journal, frozen, output = run_main(["--rounds", "3"], temp)
        self.assertEqual(code, 0, output)
        plan = rsi.load_search_plan()
        sizes = [len(r["candidates"]) for r in journal["rounds"]]
        self.assertEqual(sizes, [len(plan["round1_presets"]), plan["round_limit"],
                                 plan["round_limit"]])
        self.assertEqual(run_task.call_count, sum(sizes) * len(CASES))
        self.assertEqual(journal["rounds"][0]["selected"], "read_heavy")  # slow=48 is cheapest
        self.assertGreater(frozen["scheduler"]["ttl"]["slow"], 48)        # then it grows further
        self.assertEqual(frozen, journal["selected_policy"])
        self.assertEqual(journal["policy_sha256"], policy_sha256(frozen))
        self.assertEqual(journal["run_config"]["space"], "scheduler")
        self.assertEqual(journal["heldout_access"], False)
        self.assertIn("ttl.slow", journal["policy_space"]["tunable"])
        validate(copy.deepcopy(frozen["scheduler"]))
        self.assertEqual(frozen["scheduler"]["locked"], resolve("default")["locked"])

    def test_stale_gate_blocks_cheap_but_stale_policies(self):
        with tempfile.TemporaryDirectory() as temp:
            code, _, journal, frozen, output = run_main(
                ["--rounds", "2", "--shadow-verify", "--max-stale-rate", "0.1"], temp)
        self.assertEqual(code, 0, output)
        ttl = frozen["scheduler"]["ttl"]
        self.assertTrue(ttl["unit"] != "steps" or ttl["slow"] <= 24)  # not a stale one
        self.assertTrue(frozen["eval_shadow_verify"])
        gated = [entry["id"] for r in journal["rounds"] for entry in r["candidates"]
                 if entry["aggregate"].get("stale_gate") == "failed"]
        self.assertIn("read_heavy", gated)
        self.assertNotIn(journal["selected"], gated)

    def test_without_the_gate_the_stale_policy_would_win(self):
        with tempfile.TemporaryDirectory() as temp:
            _, _, journal, frozen, _ = run_main(["--rounds", "1", "--shadow-verify"], temp)
        self.assertEqual(journal["selected"], "read_heavy")
        self.assertEqual(journal["rounds"][0]["candidates"][1]["aggregate"]["stale_hit_rate"],
                         0.5)

    def test_argument_checks(self):
        with tempfile.TemporaryDirectory() as temp:
            for argv in (["--rounds", "9"], ["--max-stale-rate", "0.1"]):
                with self.assertRaises(SystemExit):
                    run_main(argv, temp)

    def test_dry_run_executes_nothing(self):
        with tempfile.TemporaryDirectory() as temp:
            code, run_task, journal, frozen, output = run_main(["--dry-run"], temp)
        self.assertEqual((code, run_task.call_count, journal, frozen), (0, 0, None, None))
        self.assertIn("ablation_current", output)

    def test_legacy_space_still_limited_to_two_rounds(self):
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            rsi.main(["--rounds", "3", "--dry-run"])


if __name__ == "__main__":
    unittest.main()
