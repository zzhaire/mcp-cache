"""RSI policy space, dev-only selection and resume checks; no model or server."""
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import itertools
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import time
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ours import rsi  # noqa: E402
from exp_core.methods import policy_sha256, resolve_arm  # noqa: E402

ARGS = SimpleNamespace(model="fake-model", max_steps=16, delay_ms=20,
                       jitter_ms=5, seed=20261116)


def load_document():
    return json.loads(rsi.SCENARIOS.read_text(encoding="utf-8"))


def dev_cases():
    return rsi.select_dev_cases(load_document())


def fake_row(policy, case_id):
    return {
        "case_id": case_id,
        "status": "completed",
        "task_success": True,
        "active_time_s": 10.0 + policy["ttl_seconds"] / 100.0,
        "input_tokens": policy["publish_budget_bytes"] // 8,
        "output_tokens": 20,
        "mcp_requests": 3 if policy["context_dedup"] else 5,
    }


def fake_rows(candidate, cases):
    """Deterministic completed rows derived only from a candidate policy."""
    return [fake_row(candidate["policy"], case["case_id"]) for case in cases]


def fake_run_task(dataset, task_id, model, method, output_dir, **kwargs):
    return fake_row(kwargs["policy"], kwargs["case_id"])


def aggregate(candidate_id, *, success=1, errors=0, active=None, tokens=None,
              mcp=None):
    return {"id": candidate_id, "success_count": success, "error_count": errors,
            "means": {"active_time_s": active, "input_tokens": tokens,
                      "mcp_requests": mcp}}


class CandidateSpace(unittest.TestCase):
    def test_full_grid_is_accepted(self):
        keys = list(rsi.POLICY_SPACE)
        ids = set()
        for values in itertools.product(*(rsi.POLICY_SPACE[key] for key in keys)):
            overrides = dict(zip(keys, values))
            policy = rsi.resolve_candidate(overrides)
            for key, value in overrides.items():
                self.assertEqual(policy[key], value)
            base = resolve_arm("ours")
            self.assertEqual(policy["arm"], "ours")
            self.assertEqual(policy["label"], base["label"])
            self.assertEqual(policy["disabled_modules"], base["disabled_modules"])
            self.assertEqual(policy["m4_failure_only"], base["m4_failure_only"])
            self.assertEqual(policy["server_cache_ttl"], base["server_cache_ttl"])
            ids.add(rsi.candidate_id(policy))
        self.assertEqual(len(ids), 3 * 2 * 3 * 2 * 2)

    def test_out_of_space_values_are_rejected(self):
        for overrides in ({"ttl_seconds": 45}, {"ttl_seconds": 300},
                          {"publish_budget_bytes": 4096},
                          {"publish_budget_bytes": 1024}):
            with self.assertRaises(ValueError, msg=str(overrides)):
                rsi.resolve_candidate(overrides)

    def test_non_boolean_switch_values_are_rejected(self):
        for overrides in ({"context_dedup": 1},
                          {"publish_operations": "true"},
                          {"publish_read_results": 0}):
            with self.assertRaises(ValueError, msg=str(overrides)):
                rsi.resolve_candidate(overrides)

    def test_unknown_override_key_is_rejected(self):
        with self.assertRaises(ValueError):
            rsi.resolve_candidate({"publish_cache": True})


class RoundOne(unittest.TestCase):
    def test_six_unique_stable_candidates(self):
        candidates = rsi.round1_candidates()
        self.assertEqual([candidate["id"] for candidate in candidates],
                         ["base", "ttl30", "ttl120", "budget2048", "no_ops",
                          "no_dedup"])
        self.assertEqual(len({candidate["id"] for candidate in candidates}), 6)
        self.assertEqual(len({policy_sha256(candidate["policy"])
                              for candidate in candidates}), 6)
        self.assertEqual(candidates[0]["policy"], resolve_arm("ours"))
        self.assertEqual(candidates[0]["policy"]["ttl_seconds"], 60)
        self.assertEqual(candidates[0]["policy"]["publish_budget_bytes"], 8192)
        self.assertTrue(candidates[0]["policy"]["publish_read_results"])
        self.assertEqual(candidates[1]["policy"]["ttl_seconds"], 30)
        self.assertEqual(candidates[2]["policy"]["ttl_seconds"], 120)
        self.assertEqual(candidates[3]["policy"]["publish_budget_bytes"], 2048)
        self.assertFalse(candidates[4]["policy"]["publish_operations"])
        self.assertFalse(candidates[5]["policy"]["context_dedup"])

    def test_round_two_neighbours_skip_evaluated_and_cap_at_three(self):
        unevaluated = rsi.round2_candidates(resolve_arm("ours"), set())
        self.assertEqual(len(unevaluated), 3)
        self.assertEqual(len({policy_sha256(item["policy"])
                              for item in unevaluated}), 3)
        self.assertEqual(len({item["id"] for item in unevaluated}), 3)
        self.assertEqual(unevaluated[0]["policy"]["ttl_seconds"], 30)
        self.assertEqual(unevaluated[1]["policy"]["ttl_seconds"], 120)
        self.assertEqual(unevaluated[2]["policy"]["publish_budget_bytes"], 16384)
        evaluated = {policy_sha256(candidate["policy"])
                     for candidate in rsi.round1_candidates()}
        neighbours = rsi.round2_candidates(resolve_arm("ours"), evaluated)
        self.assertEqual(len(neighbours), 2)
        for item in neighbours:
            self.assertNotIn(policy_sha256(item["policy"]), evaluated)
            self.assertEqual(policy_sha256(item["policy"]),
                             policy_sha256(rsi.resolve_candidate(
                                 {key: item["policy"][key]
                                  for key in rsi.OVERRIDE_KEYS})))
        again = rsi.round2_candidates(resolve_arm("ours"), evaluated)
        self.assertEqual([item["id"] for item in again],
                         [item["id"] for item in neighbours])


class DevCaseSelection(unittest.TestCase):
    def test_one_lexicographically_first_dev_case_per_cell(self):
        document = load_document()
        cases = rsi.select_dev_cases(document)
        self.assertEqual(len(cases), 9)
        self.assertEqual({case["split"] for case in cases}, {"dev"})
        cells = {}
        for case in document["cases"]:
            cells.setdefault((case["dataset"], case["scenario"]), []).append(case)
        self.assertEqual(len(cells), 9)
        selected = {(case["dataset"], case["scenario"]): case["case_id"]
                    for case in cases}
        for key, cell_cases in cells.items():
            expected = min(case["case_id"] for case in cell_cases
                           if case["split"] == "dev")
            self.assertEqual(selected[key], expected)

    def test_heldout_cases_are_refused(self):
        with self.assertRaises(ValueError):
            rsi.select_dev_cases(load_document(),
                                 ["appworld-discovery-heldout-1"])
        with self.assertRaises(ValueError):
            rsi.select_dev_cases(load_document(), ["unknown-case"])

    def test_cell_without_dev_case_is_refused(self):
        document = {"cases": []}
        for dataset in ("a", "b", "c"):
            for scenario in ("s1", "s2", "s3"):
                split = "heldout" if (dataset, scenario) == ("a", "s1") else "dev"
                document["cases"].append({
                    "case_id": f"{dataset}-{scenario}-{split}-1",
                    "dataset": dataset, "scenario": scenario,
                    "task_id": "task-1", "split": split, "augmented": False})
        with self.assertRaises(ValueError):
            rsi.select_dev_cases(document)


class Resume(unittest.TestCase):
    def test_terminal_result_is_reused_without_rerun(self):
        cases = dev_cases()
        candidate = rsi.round1_candidates()[0]
        with tempfile.TemporaryDirectory() as temp:
            results = Path(temp) / "rsi"
            first = results / "round-1" / candidate["id"] / cases[0]["case_id"]
            first.mkdir(parents=True)
            existing = dict(fake_rows(candidate, cases[:1])[0], reused_hint=True)
            (first / "terminal.result.json").write_text(json.dumps(existing),
                                                        encoding="utf-8")
            fresh = fake_rows(candidate, cases[1:])
            with mock.patch.object(rsi.runner, "run_task",
                                   side_effect=fresh) as run_task:
                rows = rsi.evaluate_candidate(candidate, cases, ARGS, 1,
                                              results_root=results,
                                              printer=lambda *_: None)
            self.assertEqual(run_task.call_count, len(cases) - 1)
            self.assertEqual(rows[0]["task_success"], True)
            self.assertTrue(rows[0]["reused"])
            self.assertFalse(rows[1]["reused"])
            call = run_task.call_args_list[0]
            self.assertEqual(call.args[3], "ours")
            self.assertEqual(call.kwargs["policy"], candidate["policy"])
            self.assertEqual(call.kwargs["case_split"], "dev")
            self.assertEqual(call.kwargs["case_id"], cases[1]["case_id"])
            self.assertEqual(call.args[4],
                             results / "round-1" / candidate["id"]
                             / cases[1]["case_id"])

    def test_non_terminal_result_is_rerun(self):
        cases = dev_cases()[:1]
        candidate = rsi.round1_candidates()[0]
        with tempfile.TemporaryDirectory() as temp:
            results = Path(temp) / "rsi"
            output = results / "round-1" / candidate["id"] / cases[0]["case_id"]
            output.mkdir(parents=True)
            (output / "running.result.json").write_text(
                json.dumps({"status": "running"}), encoding="utf-8")
            with mock.patch.object(rsi.runner, "run_task",
                                   side_effect=fake_rows(candidate, cases)) as run_task:
                rows = rsi.evaluate_candidate(candidate, cases, ARGS, 1,
                                              results_root=results,
                                              printer=lambda *_: None)
            self.assertEqual(run_task.call_count, 1)
            self.assertFalse(rows[0]["reused"])
            self.assertEqual(rows[0]["status"], "completed")


    def test_parallel_cases_keep_the_case_order(self):
        cases = dev_cases()
        candidate = rsi.round1_candidates()[0]
        args = SimpleNamespace(**vars(ARGS), workers=4)

        def slow_run(dataset, task_id, model, method, output_dir, **kwargs):
            time.sleep(0.01)
            return fake_row(kwargs["policy"], kwargs["case_id"])

        with tempfile.TemporaryDirectory() as temp:
            with mock.patch.object(rsi.runner, "run_task",
                                   side_effect=slow_run) as run_task:
                rows = rsi.evaluate_candidate(candidate, cases, args, 1,
                                              results_root=Path(temp),
                                              printer=lambda *_: None)
            self.assertEqual(run_task.call_count, len(cases))
            self.assertEqual([row["case_id"] for row in rows],
                             [case["case_id"] for case in cases])


class Aggregation(unittest.TestCase):
    def test_missing_active_time_is_excluded_from_the_mean(self):
        rows = [
            {"case_id": "a", "status": "completed", "task_success": True,
             "active_time_s": 10.0, "input_tokens": 1000,
             "output_tokens": 100, "mcp_requests": 5},
            {"case_id": "b", "status": "completed", "task_success": False,
             "active_time_s": None, "input_tokens": 2000,
             "output_tokens": 50, "mcp_requests": 7},
        ]
        result = rsi.aggregate_rows(rows)
        self.assertEqual(result["success_count"], 1)
        self.assertEqual(result["status_counts"], {"completed": 2, "error": 0})
        self.assertEqual(result["error_count"], 0)
        self.assertAlmostEqual(result["means"]["active_time_s"], 10.0)
        self.assertAlmostEqual(result["means"]["input_tokens"], 1500.0)
        self.assertAlmostEqual(result["means"]["mcp_requests"], 6.0)

    def test_error_rows_are_failures_without_imputed_metrics(self):
        rows = [
            {"case_id": "a", "status": "error", "task_success": None,
             "active_time_s": None, "input_tokens": None,
             "output_tokens": None, "mcp_requests": None},
            {"case_id": "b", "status": "completed", "task_success": True,
             "active_time_s": 4.0, "input_tokens": 800,
             "output_tokens": 10, "mcp_requests": 2},
        ]
        result = rsi.aggregate_rows(rows)
        self.assertEqual(result["success_count"], 1)
        self.assertEqual(result["status_counts"], {"completed": 1, "error": 1})
        self.assertAlmostEqual(result["means"]["active_time_s"], 4.0)
        self.assertAlmostEqual(result["means"]["input_tokens"], 800.0)
        self.assertAlmostEqual(result["means"]["mcp_requests"], 2.0)


class Selection(unittest.TestCase):
    def test_correctness_beats_metrics(self):
        winner, ranked, eligible, max_success = rsi.rank_candidates([
            aggregate("fast-but-wrong", success=1, active=1.0, tokens=10, mcp=1),
            aggregate("slow-but-right", success=2, active=90.0, tokens=9000,
                      mcp=30),
        ])
        self.assertEqual(winner["id"], "slow-but-right")
        self.assertEqual(eligible, ["slow-but-right"])
        self.assertEqual(max_success, 2)
        self.assertEqual([item["id"] for item in ranked], ["slow-but-right"])

    def test_error_candidate_is_ineligible_unless_all_candidates_error(self):
        winner, _, eligible, _ = rsi.rank_candidates([
            aggregate("cheap-with-error", success=2, errors=1, active=1.0,
                      tokens=10, mcp=1),
            aggregate("clean", success=2, errors=0, active=90.0, tokens=9000,
                      mcp=30),
        ])
        self.assertEqual(winner["id"], "clean")
        self.assertEqual(eligible, ["clean"])
        winner, ranked, eligible, _ = rsi.rank_candidates([
            aggregate("a", success=0, errors=1, active=None, tokens=None,
                      mcp=None),
            aggregate("b", success=0, errors=2, active=None, tokens=None,
                      mcp=None),
        ])
        self.assertEqual(winner["id"], "a")
        self.assertEqual(sorted(eligible), ["a", "b"])
        self.assertEqual(len(ranked), 2)

    def test_no_eligible_candidate_raises(self):
        with self.assertRaises(RuntimeError):
            rsi.rank_candidates([
                aggregate("max-but-error", success=2, errors=1, active=1.0,
                          tokens=10, mcp=1),
                aggregate("lower-but-clean", success=1, errors=0, active=2.0,
                          tokens=20, mcp=2),
            ])
        with self.assertRaises(RuntimeError):
            rsi.rank_candidates([])

    def test_normalized_weighted_selection(self):
        winner, ranked, eligible, _ = rsi.rank_candidates([
            aggregate("a", success=2, active=10.0, tokens=1000, mcp=10),
            aggregate("b", success=2, active=20.0, tokens=500, mcp=5),
            aggregate("c", success=2, active=15.0, tokens=750, mcp=7.5),
        ])
        self.assertEqual(winner["id"], "b")
        self.assertEqual(sorted(eligible), ["a", "b", "c"])
        scores = {item["id"]: item["score"] for item in ranked}
        self.assertAlmostEqual(scores["a"], 2 / 3)
        self.assertAlmostEqual(scores["b"], 1 / 3)
        self.assertAlmostEqual(scores["c"], 0.5)

    def test_tie_break_on_input_tokens_then_mcp_then_id(self):
        winner, _, _, _ = rsi.rank_candidates([
            aggregate("a", success=1, active=10.0, tokens=1000, mcp=5),
            aggregate("b", success=1, active=20.0, tokens=500, mcp=5),
        ])
        self.assertEqual(winner["id"], "b")
        winner, _, _, _ = rsi.rank_candidates([
            aggregate("a", success=1, active=10.0, tokens=1000, mcp=10),
            aggregate("b", success=1, active=20.0, tokens=1000, mcp=5),
        ])
        self.assertEqual(winner["id"], "b")
        winner, ranked, _, _ = rsi.rank_candidates([
            aggregate("beta", success=1, active=10.0, tokens=1000, mcp=5),
            aggregate("alpha", success=1, active=10.0, tokens=1000, mcp=5),
        ])
        self.assertEqual(winner["id"], "alpha")
        self.assertEqual([item["score"] for item in ranked], [0.0, 0.0])


class Journal(unittest.TestCase):
    def test_round_journal_round_trip(self):
        cases = dev_cases()
        candidates = rsi.round1_candidates()
        with tempfile.TemporaryDirectory() as temp:
            journal_path = Path(temp) / "rsi" / "journal.json"
            results = Path(temp) / "rsi"
            journal = rsi.new_journal(cases, {"model": ARGS.model})
            with mock.patch.object(rsi.runner, "run_task",
                                   side_effect=fake_run_task) as run_task:
                selected = rsi.run_round(1, candidates, cases, ARGS, journal,
                                         journal_path, results_root=results,
                                         printer=lambda *_: None)
            self.assertEqual(run_task.call_count, len(candidates) * len(cases))
            rsi.finalize_journal(journal, selected)
            rsi.write_journal(journal_path, journal)
            loaded = json.loads(journal_path.read_text(encoding="utf-8"))
        self.assertIs(loaded["heldout_access"], False)
        self.assertEqual(len(loaded["dev_cases"]), 9)
        self.assertEqual(set(loaded["policy_space"]), set(rsi.POLICY_SPACE))
        self.assertEqual(loaded["rounds_run"], 1)
        self.assertEqual(len(loaded["rounds"]), 1)
        round_entry = loaded["rounds"][0]
        self.assertEqual(round_entry["round"], 1)
        self.assertEqual(len(round_entry["candidates"]), 6)
        self.assertEqual(round_entry["selected"], "budget2048")
        self.assertIn("max success_count=9", round_entry["reason"])
        for entry in round_entry["candidates"]:
            self.assertEqual(set(entry), {"id", "policy", "policy_sha256",
                                          "cases", "aggregate", "status_counts"})
            self.assertEqual(len(entry["policy_sha256"]), 64)
            self.assertIsInstance(entry["policy"]["disabled_modules"], list)
            self.assertEqual(len(entry["cases"]), 9)
            for row in entry["cases"]:
                self.assertTrue(set(rsi.ROW_FIELDS) <= set(row))
                self.assertEqual(row["status"], "completed")
            self.assertEqual(entry["status_counts"],
                             {"completed": 9, "error": 0})
            self.assertEqual(entry["aggregate"]["success_count"], 9)
            self.assertIsNotNone(entry["aggregate"]["score"])
            self.assertTrue(entry["aggregate"]["eligible"])
        self.assertEqual(loaded["selected"], "budget2048")
        self.assertEqual(loaded["selected_policy"]["publish_budget_bytes"], 2048)
        self.assertEqual(loaded["policy_sha256"],
                         policy_sha256(loaded["selected_policy"]))
        json.dumps(loaded)

    def test_two_round_run_writes_the_frozen_policy(self):
        with tempfile.TemporaryDirectory() as temp:
            journal_path = Path(temp) / "journal.json"
            policy_out = Path(temp) / "policy.json"
            results = Path(temp) / "rsi"
            with mock.patch.object(rsi.runner, "run_task",
                                   side_effect=fake_run_task) as run_task, \
                    redirect_stdout(StringIO()):
                code = rsi.main(["--rounds", "2", "--journal", str(journal_path),
                                 "--policy-out", str(policy_out)],
                                results_root=results)
            self.assertEqual(code, 0)
            self.assertEqual(run_task.call_count, (6 + 3) * 9)
            journal = json.loads(journal_path.read_text(encoding="utf-8"))
            frozen = json.loads(policy_out.read_text(encoding="utf-8"))
        self.assertEqual(journal["rounds_run"], 2)
        self.assertEqual([len(item["candidates"]) for item in journal["rounds"]],
                         [6, 3])
        self.assertEqual(journal["rounds"][0]["selected"], "budget2048")
        self.assertEqual(journal["selected"],
                         "ttl30-budget2048-read1-ops1-dedup1")
        self.assertEqual(journal["rounds"][1]["selected"], journal["selected"])
        self.assertEqual(frozen, journal["selected_policy"])
        self.assertEqual(journal["policy_sha256"], policy_sha256(frozen))
        self.assertEqual(frozen["ttl_seconds"], 30)
        self.assertEqual(frozen["publish_budget_bytes"], 2048)
        self.assertEqual(journal["heldout_access"], False)


class OverallSelection(unittest.TestCase):
    def _entry(self, candidate, success, active):
        return {
            "id": candidate["id"], "policy": candidate["policy"],
            "policy_sha256": policy_sha256(candidate["policy"]),
            "cases": [],
            "aggregate": {"success_count": success, "error_count": 0,
                          "means": {"active_time_s": active,
                                    "input_tokens": 100.0,
                                    "mcp_requests": 2.0}},
            "status_counts": {"completed": 9, "error": 0},
        }

    def test_worse_round_two_neighbour_does_not_replace_round_one_winner(self):
        cases = dev_cases()
        journal = rsi.new_journal(cases, {})
        good, bad = rsi.round1_candidates()[0], rsi.round1_candidates()[-1]
        journal["rounds"] = [
            {"round": 1, "candidates": [self._entry(good, 9, 10.0)],
             "selected": good["id"], "reason": "round 1"},
            {"round": 2, "candidates": [self._entry(bad, 4, 5.0)],
             "selected": bad["id"], "reason": "round 2"},
        ]
        selected = rsi.select_overall(journal)
        self.assertEqual(selected["id"], good["id"])
        self.assertIn("first evaluated in round 1", selected["reason"])
        rsi.finalize_journal(journal, selected)
        self.assertEqual(journal["selected"], good["id"])
        self.assertEqual(journal["selected_policy"]["context_dedup"], True)
        self.assertEqual(journal["policy_sha256"],
                         policy_sha256(journal["selected_policy"]))


class DryRun(unittest.TestCase):
    def test_dry_run_plans_zero_executions_and_writes_nothing(self):
        with tempfile.TemporaryDirectory() as temp:
            journal = Path(temp) / "journal.json"
            policy_out = Path(temp) / "policy.json"
            stdout, stderr = StringIO(), StringIO()
            with mock.patch.object(rsi.runner, "run_task",
                                   side_effect=AssertionError("must not run")) as run_task, \
                    redirect_stdout(stdout), redirect_stderr(stderr):
                code = rsi.main(["--dry-run", "--journal", str(journal),
                                 "--policy-out", str(policy_out)])
            self.assertEqual(code, 0, stderr.getvalue())
            self.assertEqual(run_task.call_count, 0)
            self.assertFalse(journal.exists())
            self.assertFalse(policy_out.exists())
            self.assertEqual(list(Path(temp).iterdir()), [])
            lines = [line for line in stdout.getvalue().splitlines()
                     if line.strip()]
            payload = json.loads(lines[-1])
            self.assertTrue(payload["dry_run"])
            self.assertEqual(payload["executions"], 0)
            self.assertEqual(payload["planned_runs"], 6 * 9)
            self.assertEqual(len(payload["cases"]), 9)
            self.assertEqual(len(payload["candidates"]), 6)
            self.assertEqual(payload["rounds_requested"], 2)
            self.assertEqual(len([line for line in lines
                                  if line.endswith("would run")]), 6 * 9)

    def test_dry_run_refuses_heldout_cases(self):
        with tempfile.TemporaryDirectory() as temp:
            stderr = StringIO()
            with mock.patch.object(rsi.runner, "run_task",
                                   side_effect=AssertionError("must not run")), \
                    redirect_stdout(StringIO()), redirect_stderr(stderr):
                code = rsi.main(["--dry-run", "--cases",
                                 "appworld-discovery-heldout-1",
                                 "--journal", str(Path(temp) / "journal.json"),
                                 "--policy-out", str(Path(temp) / "policy.json")])
            self.assertEqual(code, 2)
            self.assertIn("non-dev", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
