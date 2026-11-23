"""Decision-time publication checks; no model, network or benchmark calls."""
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ours.publication import (collect_operations, collect_references, collect_results,
                         dedupe_result)
from exp_core.runner import active_time
from exp_core.runtime import AccessFailure, Runtime

SCOPE = "bfcl:t1:run"


def read(runtime, name, arguments, value, readonly=True, operation_id=None):
    return runtime.call(name, arguments, lambda: value, readonly=readonly,
                        operation_id=operation_id, scope=SCOPE)


def fail(code, outcome):
    def fetch():
        raise AccessFailure(code, retryable=False, outcome=outcome)
    return fetch


class ValidResults(unittest.TestCase):
    def test_matching_scope_epoch_and_ttl_is_published(self):
        runtime = Runtime(clock=lambda: 0.0, ttl=10)
        read(runtime, "ls", {"path": "/"}, {"files": ["a"]})
        items, used, identities = collect_results(runtime, SCOPE, 1024)
        self.assertEqual(items, [{"name": "ls", "arguments": {"path": "/"},
                                  "result": {"files": ["a"]}}])
        self.assertEqual(len(identities), 1)
        self.assertGreater(used, 0)
        self.assertLessEqual(used, 1024)
        self.assertEqual(used, sum(len(json.dumps(item).encode()) for item in items))
        expected = json.dumps([SCOPE, "ls", {"path": "/"}], sort_keys=True)
        self.assertEqual(identities[0][2], expected)

    def test_published_values_are_deep_copies(self):
        runtime = Runtime(clock=lambda: 0.0, ttl=10)
        read(runtime, "ls", {"path": "/"}, {"files": ["a"]})
        items, _, _ = collect_results(runtime, SCOPE, 1024)
        items[0]["result"]["files"].append("mutated")
        again, _, _ = collect_results(runtime, SCOPE, 1024)
        self.assertEqual(again[0]["result"], {"files": ["a"]})

    def test_publication_order_is_deterministic(self):
        runtime = Runtime(clock=lambda: 0.0, ttl=10)
        read(runtime, "ls", {"path": "/b"}, ["b"])
        read(runtime, "ls", {"path": "/a"}, ["a"])
        first, _, _ = collect_results(runtime, SCOPE, 1024)
        second, _, _ = collect_results(runtime, SCOPE, 1024)
        self.assertEqual(first, second)
        self.assertEqual([item["arguments"]["path"] for item in first], ["/a", "/b"])

    def test_expired_result_is_not_published(self):
        now = [0.0]
        runtime = Runtime(clock=lambda: now[0], ttl=5)
        read(runtime, "ls", {"path": "/"}, ["a"])
        now[0] = 5.0
        self.assertEqual(collect_results(runtime, SCOPE, 1024), ([], 0, []))

    def test_other_scope_is_not_published(self):
        runtime = Runtime(clock=lambda: 0.0, ttl=10)
        read(runtime, "ls", {"path": "/"}, ["a"])
        self.assertEqual(collect_results(runtime, "bfcl:other:run", 1024), ([], 0, []))

    def test_changed_epoch_is_not_published(self):
        runtime = Runtime(clock=lambda: 0.0, ttl=10)
        read(runtime, "ls", {"path": "/"}, ["a"])
        runtime.data_epoch += 1
        self.assertEqual(collect_results(runtime, SCOPE, 1024), ([], 0, []))
        runtime.data_epoch -= 1
        runtime.auth_epoch += 1
        self.assertEqual(collect_results(runtime, SCOPE, 1024), ([], 0, []))

    def test_write_invalidates_previously_published_results(self):
        runtime = Runtime(clock=lambda: 0.0, ttl=10)
        read(runtime, "ls", {"path": "/"}, ["a"])
        read(runtime, "write", {"value": 1}, "ok", readonly=False)
        self.assertEqual(collect_results(runtime, SCOPE, 1024), ([], 0, []))

    def test_budget_truncates_instead_of_overspending(self):
        runtime = Runtime(clock=lambda: 0.0, ttl=10)
        for index in range(5):
            read(runtime, "ls", {"path": f"/{index}"}, {"data": "x" * 40})
        items, used, identities = collect_results(runtime, SCOPE, 200)
        self.assertGreater(len(items), 0)
        self.assertLess(len(items), 5)
        self.assertEqual(len(items), len(identities))
        self.assertLessEqual(used, 200)
        self.assertEqual(used, sum(len(json.dumps(item).encode()) for item in items))

    def test_zero_budget_publishes_nothing(self):
        runtime = Runtime(clock=lambda: 0.0, ttl=10)
        read(runtime, "ls", {"path": "/"}, ["a"])
        self.assertEqual(collect_results(runtime, SCOPE, 0), ([], 0, []))


class ReferencePublication(unittest.TestCase):
    def identity(self, name, arguments):
        return json.dumps([SCOPE, name, arguments], sort_keys=True)

    def test_visible_read_publishes_a_compact_reference(self):
        runtime = Runtime(clock=lambda: 0.0, ttl=10)
        read(runtime, "ls", {"path": "/"}, {"files": ["a"]})
        seen = {self.identity("ls", {"path": "/"})}
        items, used, identities = collect_references(runtime, SCOPE, 1024, seen)
        self.assertEqual(len(items), 1)
        item = items[0]
        self.assertEqual(item["name"], "ls")
        self.assertEqual(item["arguments"], {"path": "/"})
        self.assertEqual(item["state"], "valid")
        self.assertNotIn("result", item)
        self.assertEqual(len(item["result_ref"]), 16)
        self.assertEqual(used, len(json.dumps(item).encode()))
        self.assertEqual(identities[0], self.identity("ls", {"path": "/"}))

    def test_unseen_read_is_not_published(self):
        runtime = Runtime(clock=lambda: 0.0, ttl=10)
        read(runtime, "ls", {"path": "/"}, ["a"])
        self.assertEqual(collect_references(runtime, SCOPE, 1024, set()), ([], 0, []))

    def test_reference_respects_ttl_scope_and_epoch(self):
        now = [0.0]
        runtime = Runtime(clock=lambda: now[0], ttl=5)
        read(runtime, "ls", {"path": "/"}, ["a"])
        seen = {self.identity("ls", {"path": "/"})}
        now[0] = 5.0
        self.assertEqual(collect_references(runtime, SCOPE, 1024, seen), ([], 0, []))
        now[0] = 0.0
        self.assertEqual(collect_references(runtime, "other", 1024, seen), ([], 0, []))
        runtime.data_epoch += 1
        self.assertEqual(collect_references(runtime, SCOPE, 1024, seen), ([], 0, []))

    def test_reference_limit_keeps_newest_reads(self):
        runtime = Runtime(clock=lambda: 0.0, ttl=10)
        for index in range(4):
            read(runtime, "ls", {"path": f"/{index}"}, [index])
        seen = {self.identity("ls", {"path": f"/{index}"}) for index in range(4)}
        items, _, _ = collect_references(runtime, SCOPE, 1024, seen, limit=2)
        self.assertEqual([item["arguments"]["path"] for item in items], ["/3", "/2"])

    def test_catalog_read_is_published_as_a_reference(self):
        runtime = Runtime(clock=lambda: 0.0, ttl=10)
        arguments = {"app": "api_docs", "api": "show_api_doc",
                     "arguments": {"app_name": "spotify", "api_name": "show_song"}}
        catalog_identity = json.dumps([SCOPE, arguments], sort_keys=True)
        runtime.catalog(catalog_identity, "v1", lambda: {"doc": "show_song"})
        items, _, identities = collect_references(
            runtime, SCOPE, 1024, set(), latest=("catalog", catalog_identity))
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["name"], "api_call")
        self.assertEqual(items[0]["arguments"], arguments)
        self.assertNotIn("result", items[0])
        # The published identity must match the api_call the model would repeat.
        self.assertEqual(identities, [self.identity("api_call", arguments)])

    def test_latest_result_reference_respects_ttl(self):
        now = [0.0]
        runtime = Runtime(clock=lambda: now[0], ttl=5)
        read(runtime, "ls", {"path": "/"}, ["a"])
        identity = self.identity("ls", {"path": "/"})
        items, _, _ = collect_references(
            runtime, SCOPE, 1024, {identity}, latest=("result", identity))
        self.assertEqual(len(items), 1)
        now[0] = 5.0
        self.assertEqual(collect_references(
            runtime, SCOPE, 1024, {identity}, latest=("result", identity)), ([], 0, []))

    def test_recent_read_list_is_published_in_given_order(self):
        runtime = Runtime(clock=lambda: 0.0, ttl=10)
        for index in range(4):
            read(runtime, "ls", {"path": f"/{index}"}, [index])
        order = [("result", self.identity("ls", {"path": f"/{index}"}))
                 for index in (3, 1, 0)]
        items, _, keys = collect_references(
            runtime, SCOPE, 1024, set(), limit=3, latest=order)
        self.assertEqual([item["arguments"]["path"] for item in items],
                         ["/3", "/1", "/0"])
        self.assertEqual(keys, [self.identity("ls", {"path": f"/{index}"})
                                for index in (3, 1, 0)])

    def test_recent_read_list_skips_invalid_entries(self):
        now = [0.0]
        runtime = Runtime(clock=lambda: now[0], ttl=5)
        read(runtime, "ls", {"path": "/old"}, ["a"])
        now[0] = 6.0
        read(runtime, "ls", {"path": "/new"}, ["b"])
        order = [("result", self.identity("ls", {"path": "/old"})),
                 ("result", self.identity("ls", {"path": "/new"}))]
        items, _, _ = collect_references(
            runtime, SCOPE, 1024, set(), limit=3, latest=order)
        self.assertEqual([item["arguments"]["path"] for item in items], ["/new"])

    def test_reference_budget_never_overspends(self):
        runtime = Runtime(clock=lambda: 0.0, ttl=10)
        read(runtime, "ls", {"path": "/"}, ["a"])
        seen = {self.identity("ls", {"path": "/"})}
        self.assertEqual(collect_references(runtime, SCOPE, 1, seen), ([], 0, []))


class OperationSummaries(unittest.TestCase):
    def test_failed_operation_is_published_without_result_objects(self):
        runtime = Runtime(clock=lambda: 0.0, ttl=10)
        runtime.call("write", {"x": 1}, lambda: "ok", operation_id="done")
        with self.assertRaises(AccessFailure):
            runtime.call("read", {"x": 2}, fail("denied", "not_executed"),
                         readonly=True, operation_id="failed")
        items = collect_operations(runtime)
        self.assertEqual(len(items), 1)
        item = items[0]
        self.assertEqual(set(item), {"name", "arguments", "state", "error_code",
                                     "outcome", "attempts", "retry_at"})
        self.assertEqual((item["name"], item["arguments"]), ("read", {"x": 2}))
        self.assertEqual((item["state"], item["error_code"], item["outcome"]),
                         ("failed", "denied", "not_executed"))
        self.assertEqual(item["attempts"], 1)
        for excluded in ("result", "failure", "operation_id", "request"):
            self.assertNotIn(excluded, item)

    def test_completed_operation_is_not_published(self):
        runtime = Runtime(clock=lambda: 0.0, ttl=10)
        runtime.call("write", {"x": 1}, lambda: "ok", operation_id="done")
        self.assertEqual(collect_operations(runtime), [])

    def test_uncertain_operation_requires_reconciliation(self):
        runtime = Runtime(clock=lambda: 0.0, ttl=10)
        with self.assertRaises(AccessFailure):
            runtime.call("write", {"x": 1}, fail("lost_response", "unknown"),
                         operation_id="uncertain")
        items = collect_operations(runtime)
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["state"], "uncertain")
        self.assertEqual(items[0]["note"], "reconcile before retry")
        self.assertEqual(items[0]["error_code"], "lost_response")

    def test_limit_is_enforced(self):
        runtime = Runtime(clock=lambda: 0.0, ttl=10)
        for index in range(25):
            with self.assertRaises(AccessFailure):
                runtime.call("read", {"index": index}, fail("denied", "not_executed"),
                             readonly=True, operation_id=f"op-{index}")
        self.assertEqual(len(collect_operations(runtime)), 20)
        self.assertEqual(len(collect_operations(runtime, limit=3)), 3)

    def test_operations_budget_truncates_instead_of_overspending(self):
        runtime = Runtime(clock=lambda: 0.0, ttl=10)
        for index in range(5):
            with self.assertRaises(AccessFailure):
                runtime.call("read", {"index": index}, fail("denied", "not_executed"),
                             readonly=True, operation_id=f"op-{index}")
        full = collect_operations(runtime)
        self.assertEqual(len(full), 5)
        one = len(json.dumps(full[0], default=str).encode())
        capped = collect_operations(runtime, budget_bytes=one * 2 + 5)
        self.assertEqual(len(capped), 2)


class Dedupe(unittest.TestCase):
    def test_second_identical_result_becomes_a_reference(self):
        seen = set()
        payload = {"name": "ls", "result": {"files": ["file-%02d" % i for i in range(20)]}}
        first, saved, duplicated = dedupe_result(payload, seen)
        self.assertIs(first, payload)
        self.assertEqual((saved, duplicated), (0, False))
        second, saved, duplicated = dedupe_result(
            {"name": "ls", "result": {"files": ["file-%02d" % i for i in range(20)]}}, seen)
        self.assertTrue(duplicated)
        self.assertEqual(second["name"], "ls")
        self.assertEqual(len(second["result_ref"]), 16)
        self.assertEqual(second["note"], "identical to earlier result")
        self.assertNotIn("result", second)
        self.assertEqual(saved, len(json.dumps(payload).encode())
                         - len(json.dumps(second).encode()))
        self.assertGreater(saved, 0)

    def test_key_order_does_not_defeat_the_fingerprint(self):
        seen = set()
        dedupe_result({"name": "ls", "result": {"a": 1, "b": 2}}, seen)
        _, _, duplicated = dedupe_result({"result": {"b": 2, "a": 1}, "name": "ls"}, seen)
        self.assertTrue(duplicated)

    def test_different_results_are_not_deduplicated(self):
        seen = set()
        dedupe_result({"name": "ls", "result": {"files": ["a"]}}, seen)
        _, saved, duplicated = dedupe_result(
            {"name": "ls", "result": {"files": ["a", "b"]}}, seen)
        self.assertEqual((saved, duplicated), (0, False))
        _, saved, duplicated = dedupe_result(
            {"name": "cat", "result": {"files": ["a"]}}, seen)
        self.assertEqual((saved, duplicated), (0, False))


class OperationAudit(unittest.TestCase):
    def test_request_is_kept_for_publication_but_excluded_from_audit(self):
        runtime = Runtime()
        arguments = {"amount": 5, "password": "secret"}
        with self.assertRaises(AccessFailure):
            runtime.call("transfer", arguments, fail("denied", "not_executed"),
                         operation_id="op")
        arguments["amount"] = 99
        self.assertEqual(runtime.operations["op"]["request"],
                         {"name": "transfer",
                          "arguments": {"amount": 5, "password": "secret"}})
        audit = runtime.operation_audit()
        self.assertEqual(len(audit), 1)
        self.assertEqual(audit[0]["state"], "failed")
        for excluded in ("request", "result", "failure"):
            self.assertNotIn(excluded, audit[0])


class ActiveTime(unittest.TestCase):
    def test_queue_wait_is_removed(self):
        self.assertEqual(active_time(10.0, 3.0), 7.0)

    def test_missing_or_larger_queue_wait_never_goes_negative(self):
        self.assertEqual(active_time(2.0, None), 2.0)
        self.assertEqual(active_time(2.0, 5.0), 0.0)


if __name__ == "__main__":
    unittest.main()


class LargeRepeatTest(unittest.TestCase):
    def test_large_identical_result_is_handed_over_once(self):
        from ours.publication import reference_large_repeat
        rows = [{"id": f"t-{i}", "amount": 1} for i in range(300)]
        seen = {}
        first, saved = reference_large_repeat("list", {"account": "a"}, rows, seen, 4096)
        self.assertEqual((first["result"], saved), (rows, 0))
        again, saved = reference_large_repeat("list", {"account": "a"}, rows, seen, 4096)
        self.assertNotIn("result", again)
        self.assertEqual(again["same_result_as"], {"name": "list", "arguments": {"account": "a"}})
        self.assertGreater(saved, 4000)
        # A changed result is handed over in full again.
        changed, saved = reference_large_repeat("list", {"account": "a"}, rows[1:], seen, 4096)
        self.assertEqual((changed["result"], saved), (rows[1:], 0))

    def test_small_results_and_a_zero_threshold_are_left_alone(self):
        from ours.publication import reference_large_repeat
        seen = {}
        for _ in range(2):
            small, saved = reference_large_repeat("balance", {}, {"balance": 500}, seen, 4096)
            self.assertEqual((small["result"], saved), ({"balance": 500}, 0))
        rows = list(range(5000))
        for _ in range(2):
            off, saved = reference_large_repeat("list", {}, rows, seen, 0)
            self.assertEqual((off["result"], saved), (rows, 0))
        self.assertEqual(seen, {})


class KeyValues(unittest.TestCase):
    """Design 4.3: the model is given key values, not only a reference."""

    def test_record_keeps_scalars_and_names_what_it_left_out(self):
        from ours.publication import key_values
        self.assertEqual(key_values({"account": "checking", "balance": 500}, 1024),
                         {"account": "checking", "balance": 500})
        self.assertEqual(key_values({"sku": "a", "history": [1, 2]}, 1024),
                         {"sku": "a", "_omitted": ["history"]})

    def test_list_is_given_whole_only_when_it_fits(self):
        from ours.publication import key_values
        rows = [{"id": f"tr-{i}", "amount": i} for i in range(3)]
        self.assertEqual(key_values(rows, 1024), rows)
        self.assertEqual(key_values(rows * 100, 1024), {"count": 300, "_omitted": ["rows"]})

    def test_reference_in_key_value_mode_carries_values_and_version(self):
        from ours.publication import _reference
        item = _reference("bank_get_balance", {"account": "checking"},
                          {"account": "checking", "balance": 500}, "key_values", 1024)
        self.assertEqual(item["values"], {"account": "checking", "balance": 500})
        self.assertEqual(len(item["version"]), 16)
        self.assertNotIn("result_ref", item)
