"""Report generator checks on synthetic result files; no model or network."""
import csv
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from exp_core import report as report_matrix  # noqa: E402

CURRENT = "a" * 64
STALE = "b" * 64
CASES = [
    {"case_id": "appworld-discovery-dev-1", "dataset": "appworld",
     "task_id": "task-1", "scenario": "discovery", "split": "dev",
     "augmented": False},
    {"case_id": "bfcl-read-dev-2", "dataset": "bfcl", "task_id": "task-2",
     "scenario": "read", "split": "dev", "augmented": False},
    {"case_id": "tau2-auth_operation-dev-3", "dataset": "tau2",
     "task_id": "task-3", "scenario": "auth_operation", "split": "heldout",
     "augmented": True},
]


class Fixture:
    def __init__(self, temp):
        self.root = Path(temp)
        self.results = self.root / "matrix"
        self.scenarios = self.root / "scenarios.json"
        self.scenarios.write_text(json.dumps({"cases": CASES}), encoding="utf-8")
        self.counter = 0

    def write(self, case_id, arm, *, status="completed", success=True,
              active=10.0, input_tokens=100, output_tokens=10, mcp=2,
              policy="policy", code_sha=CURRENT, run_id=None, mtime=None,
              error=None, budget_exhausted=None, runtime=None,
              server_cache=None, model="DeepSeek-V4.1-Flash"):
        case = next(item for item in CASES if item["case_id"] == case_id)
        run_id = run_id or f"run-{self.counter:03d}"
        row = {
            "run_id": run_id, "case_id": case_id, "dataset": case["dataset"],
            "task_id": case["task_id"], "scenario": case["scenario"],
            "augmented": case["augmented"], "model": model, "arm": arm,
            "method": arm, "status": status, "task_success": success,
            "active_time_s": active,
            "task_time_s": None if active is None else active + 1.0,
            "harness_queue_wait_s": None if active is None else 1.0,
            "input_tokens": input_tokens, "output_tokens": output_tokens,
            "cached_input_tokens": 0, "logical_tool_calls": 1,
            "mcp_requests": mcp, "request_body_bytes": 10,
            "response_body_bytes": 20, "publish_events": 0, "publish_bytes": 0,
            "published_results_used": 0, "context_dedup_saved_bytes": 0,
            "model_calls": 1, "model_attempts": 1,
            "model_process_failures": 0, "model_retries": 0,
            "policy_sha256": policy, "code_sha256": code_sha,
            "runtime": runtime if runtime is not None else {
                "M1_hits": 1, "M1_lookups": 2, "M2_hits": 0, "M2_eligible": 1,
                "M3_hits": 0, "M3_lookups": 0, "M4_hits": 0,
                "M4_failure_hits": 0, "retries": 0, "logical_calls": 1},
            "server_cache": server_cache if server_cache is not None
            else {"hits": 0, "misses": 1},
        }
        if error is not None:
            row["error"] = error
        if budget_exhausted is not None:
            row["budget_exhausted"] = budget_exhausted
        path = self.results / case_id / arm / f"{run_id}.result.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(row), encoding="utf-8")
        if mtime is not None:
            os.utime(path, (mtime, mtime))
        self.counter += 1
        return path

    def report(self, **kwargs):
        kwargs.setdefault("results_root", self.results)
        kwargs.setdefault("scenarios", self.scenarios)
        kwargs.setdefault("code_sha256", CURRENT)
        return report_matrix.build_report(**kwargs)


class CodeHashFiltering(unittest.TestCase):
    def test_current_and_stale_rows_are_separated(self):
        with tempfile.TemporaryDirectory() as temp:
            fixture = Fixture(temp)
            fixture.write("appworld-discovery-dev-1", "ours", active=10.0,
                          run_id="current-run")
            fixture.write("appworld-discovery-dev-1", "ours", active=99.0,
                          run_id="stale-run", code_sha=STALE, policy="old")
            report = fixture.report(cases=["appworld-discovery-dev-1"],
                                    arms=["ours"])
            self.assertEqual([row["run_id"] for row in report.rows],
                             ["current-run"])
            self.assertEqual([row["run_id"] for row in report.stale],
                             ["stale-run"])
            markdown = report.markdown()
            self.assertIn("10.00", markdown)
            self.assertNotIn("99.00", markdown)
            self.assertIn("stale-run", markdown)
            records = list(csv.DictReader(io.StringIO(report.csv_text())))
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["run_id"], "current-run")


class CsvOutput(unittest.TestCase):
    def test_columns_and_deterministic_order(self):
        with tempfile.TemporaryDirectory() as temp:
            fixture = Fixture(temp)
            fixture.write("bfcl-read-dev-2", "bare", active=None, success=None)
            fixture.write("appworld-discovery-dev-1", "ours")
            fixture.write("appworld-discovery-dev-1", "bare")
            report = fixture.report()
            text = report.csv_text()
            self.assertEqual(text, report.csv_text())
            records = list(csv.DictReader(io.StringIO(text)))
            self.assertEqual(list(records[0]), list(report_matrix.CSV_COLUMNS))
            self.assertEqual([(record["case_id"], record["arm"])
                              for record in records],
                             [("appworld-discovery-dev-1", "bare"),
                              ("appworld-discovery-dev-1", "ours"),
                              ("bfcl-read-dev-2", "bare")])
            self.assertEqual(records[0]["active_time_s"], "10.0")
            self.assertEqual(records[2]["active_time_s"], "")
            self.assertEqual(records[2]["task_success"], "")
            self.assertEqual(records[1]["dataset"], "appworld")
            self.assertEqual(records[1]["task_id"], "task-1")
            self.assertEqual(records[1]["augmented"], "False")
            self.assertEqual(records[1]["M1_hits"], "1")


class IntegrityReporting(unittest.TestCase):
    def test_missing_identities_are_listed(self):
        with tempfile.TemporaryDirectory() as temp:
            fixture = Fixture(temp)
            fixture.write("appworld-discovery-dev-1", "ours")
            report = fixture.report(cases=["appworld-discovery-dev-1"],
                                    arms=["bare", "ours"])
            self.assertEqual(report.missing,
                             [("appworld-discovery-dev-1", "bare")])
            markdown = report.markdown()
            self.assertIn("缺失身份", markdown)
            self.assertIn("`appworld-discovery-dev-1` / `bare`", markdown)
            self.assertIn("缺失：1", markdown)

    def test_duplicate_identity_is_reported_and_oldest_file_used(self):
        with tempfile.TemporaryDirectory() as temp:
            fixture = Fixture(temp)
            fixture.write("appworld-discovery-dev-1", "bare", run_id="later",
                          active=20.0, mtime=2000)
            fixture.write("appworld-discovery-dev-1", "bare", run_id="earlier",
                          active=10.0, mtime=1000)
            report = fixture.report(cases=["appworld-discovery-dev-1"],
                                    arms=["bare"])
            self.assertEqual(report.duplicates, 1)
            self.assertEqual([row["run_id"] for row in report.rows],
                             ["earlier"])
            self.assertTrue(any("重复身份" in error
                                for error in report.integrity_errors))
            markdown = report.markdown()
            self.assertIn("earlier", markdown)
            self.assertNotIn("20.00", markdown)
            self.assertIn("重复身份", markdown.split("## 附录：完整性错误")[1])


class PairedComparison(unittest.TestCase):
    def test_only_cases_completed_by_both_arms_are_paired(self):
        with tempfile.TemporaryDirectory() as temp:
            fixture = Fixture(temp)
            fixture.write("appworld-discovery-dev-1", "ours", active=10.0,
                          input_tokens=100, mcp=2)
            fixture.write("appworld-discovery-dev-1", "rw_merchant",
                          active=20.0, input_tokens=200, mcp=4)
            fixture.write("bfcl-read-dev-2", "ours", active=30.0,
                          input_tokens=300, mcp=6)
            fixture.write("bfcl-read-dev-2", "rw_merchant", status="error",
                          success=None, active=None, input_tokens=0, mcp=0,
                          error="boom")
            report = fixture.report()
            pair = report_matrix.paired(report.rows, "ours", "rw_merchant")
            self.assertEqual(pair["n"], 1)
            self.assertEqual(pair["case_ids"], ["appworld-discovery-dev-1"])
            self.assertAlmostEqual(pair["ours_mean_active"], 10.0)
            self.assertAlmostEqual(pair["other_mean_active"], 20.0)
            self.assertAlmostEqual(pair["delta_active"], -10.0)
            self.assertEqual(pair["ours_input"], 100)
            self.assertEqual(pair["other_input"], 200)
            self.assertAlmostEqual(pair["delta_input_pct"], -50.0)
            self.assertEqual(pair["ours_mcp"], 2)
            self.assertEqual(pair["other_mcp"], 4)
            self.assertEqual(pair["ours_success"], 1)
            self.assertEqual(pair["other_success"], 1)


class Verdict(unittest.TestCase):
    def test_all_items_met_when_ours_wins_on_every_metric(self):
        with tempfile.TemporaryDirectory() as temp:
            fixture = Fixture(temp)
            fixture.write("appworld-discovery-dev-1", "ours", active=10.0,
                          input_tokens=100, mcp=2)
            fixture.write("appworld-discovery-dev-1", "rw_merchant",
                          active=20.0, input_tokens=200, mcp=4)
            verdict = fixture.report().verdict()
            self.assertEqual(verdict["strongest"], "rw_merchant")
            self.assertTrue(verdict["met"])
            self.assertEqual([item["met"] for item in verdict["items"]],
                             [True, True, True, True])

    def test_verdict_fails_when_mcp_requests_are_higher(self):
        with tempfile.TemporaryDirectory() as temp:
            fixture = Fixture(temp)
            fixture.write("appworld-discovery-dev-1", "ours", active=10.0,
                          input_tokens=100, mcp=9)
            fixture.write("appworld-discovery-dev-1", "rw_merchant",
                          active=20.0, input_tokens=200, mcp=3)
            report = fixture.report()
            verdict = report.verdict()
            items = {item["key"]: item for item in verdict["items"]}
            self.assertTrue(items["a"]["met"])
            self.assertTrue(items["b"]["met"])
            self.assertTrue(items["c"]["met"])
            self.assertFalse(items["d"]["met"])
            self.assertFalse(verdict["met"])
            line = next(line for line in report.markdown().splitlines()
                        if line.startswith("- (d)"))
            self.assertIn("未达成", line)

    def test_verdict_fails_when_accuracy_is_lower(self):
        with tempfile.TemporaryDirectory() as temp:
            fixture = Fixture(temp)
            fixture.write("appworld-discovery-dev-1", "ours", success=False,
                          active=10.0, input_tokens=100, mcp=1)
            fixture.write("appworld-discovery-dev-1", "bare", success=True,
                          active=20.0, input_tokens=200, mcp=5)
            report = fixture.report()
            verdict = report.verdict()
            self.assertEqual(verdict["strongest"], "bare")
            items = {item["key"]: item for item in verdict["items"]}
            self.assertFalse(items["a"]["met"])
            self.assertTrue(items["b"]["met"])
            self.assertTrue(items["c"]["met"])
            self.assertTrue(items["d"]["met"])
            self.assertFalse(verdict["met"])
            line = next(line for line in report.markdown().splitlines()
                        if line.startswith("- (a)"))
            self.assertIn("未达成", line)

    def test_no_completed_ours_rows_is_no_data(self):
        with tempfile.TemporaryDirectory() as temp:
            fixture = Fixture(temp)
            fixture.write("appworld-discovery-dev-1", "ours", status="error",
                          success=None, active=None, error="boom")
            fixture.write("appworld-discovery-dev-1", "bare", success=True)
            verdict = fixture.report().verdict()
            self.assertTrue(verdict["no_data"])
            self.assertFalse(verdict["met"])
            self.assertEqual([item["met"] for item in verdict["items"]],
                             [False, False, False, False])


class SplitsAndFailures(unittest.TestCase):
    def test_original_and_augmented_are_separated(self):
        with tempfile.TemporaryDirectory() as temp:
            fixture = Fixture(temp)
            fixture.write("appworld-discovery-dev-1", "bare", active=10.0)
            fixture.write("tau2-auth_operation-dev-3", "bare", active=999.0)
            markdown = fixture.report().markdown()
            original = markdown.split("### 原始 case（augmented=false）")[1].split(
                "### 增广 case（augmented=true）")[0]
            self.assertIn("10.00", original)
            self.assertNotIn("999.00", original)
            augmented = markdown.split("### 增广 case（augmented=true）")[1].split(
                "## 失败 / 超时 / 预算耗尽")[0]
            self.assertIn("999.00", augmented)
            self.assertNotIn("10.00", augmented)

    def test_error_and_budget_rows_are_listed_and_unsuccessful(self):
        with tempfile.TemporaryDirectory() as temp:
            fixture = Fixture(temp)
            fixture.write("appworld-discovery-dev-1", "bare", status="error",
                          success=None, active=None, error="boom",
                          run_id="error-run")
            fixture.write("appworld-discovery-dev-1", "ours",
                          budget_exhausted=True, run_id="budget-run")
            report = fixture.report(cases=["appworld-discovery-dev-1"],
                                    arms=["bare", "ours"])
            bare = report_matrix.aggregate(report.rows_for("bare"))
            self.assertEqual(bare["success_rate"], 0.0)
            self.assertEqual(bare["error"], 1)
            self.assertIsNone(bare["mean_active"])
            failures = report.markdown().split("## 失败 / 超时 / 预算耗尽")[1].split(
                "## 逐 case 明细")[0]
            self.assertIn("error-run", failures)
            self.assertIn("budget-run", failures)
            self.assertIn("boom", failures)
            self.assertIn("error 1 行", failures)

    def test_mean_and_median_ignore_missing_active_time(self):
        with tempfile.TemporaryDirectory() as temp:
            fixture = Fixture(temp)
            fixture.write("appworld-discovery-dev-1", "bare", active=10.0)
            fixture.write("bfcl-read-dev-2", "bare", active=30.0)
            fixture.write("tau2-auth_operation-dev-3", "bare", active=None,
                          success=None)
            report = fixture.report(arms=["bare"])
            summary = report_matrix.aggregate(report.rows_for("bare"))
            self.assertEqual(summary["n"], 3)
            self.assertEqual(summary["n_active"], 2)
            self.assertAlmostEqual(summary["mean_active"], 20.0)
            self.assertAlmostEqual(summary["median_active"], 20.0)
            self.assertIn("20.00 (n=2)", report.markdown())

    def test_errors_count_as_unsuccessful_in_success_rate(self):
        with tempfile.TemporaryDirectory() as temp:
            fixture = Fixture(temp)
            fixture.write("appworld-discovery-dev-1", "bare", success=True)
            fixture.write("bfcl-read-dev-2", "bare", status="error",
                          success=None, active=None, run_id="failed")
            report = fixture.report(arms=["bare"])
            summary = report_matrix.aggregate(report.rows_for("bare"))
            self.assertEqual(summary["n"], 2)
            self.assertEqual(summary["success"], 1)
            self.assertAlmostEqual(summary["success_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()
