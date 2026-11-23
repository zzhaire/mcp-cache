"""The controllable lab world, its contract and the scripted mechanism checks."""
from contextlib import redirect_stdout
from io import StringIO
import copy
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from exp_core import matrix, runner  # noqa: E402
from exp_core.lab.harness import load_tasks, run_gold  # noqa: E402
from exp_core.lab.world import LATER_TOOLS, TOOLS, LabWorld  # noqa: E402
from ours import rsi  # noqa: E402

ALL_TASKS = load_tasks()
# This file covers the basic suite; the stress suite has its own tests.
BASIC = [case["task_id"] for case in json.loads(
    (ROOT / "data" / "lab-cases.json").read_text(encoding="utf-8"))["cases"]]
TASKS = {task_id: ALL_TASKS[task_id] for task_id in BASIC}


def world(task_id="lab-repeat-read", **changes):
    task = copy.deepcopy(TASKS[task_id])
    task.update(changes)
    return LabWorld(task)


def signed_in(task_id="lab-repeat-read", **changes):
    w = world(task_id, **changes)
    for app in ("bank", "shop"):
        w.invoke("auth_login", {"app": app, "username": "alice", "password": "pw-alice"})
    return w


class World(unittest.TestCase):
    def test_login_is_required_and_checked(self):
        w = world()
        self.assertEqual(w.invoke("bank_get_balance", {"account": "checking"})["error_code"],
                         "unauthorized")
        bad = w.invoke("auth_login", {"app": "bank", "username": "alice", "password": "x"})
        self.assertEqual(bad["error_code"], "business_error")
        ok = w.invoke("auth_login", {"app": "bank", "username": "alice", "password": "pw-alice"})
        self.assertEqual(ok["credential_contract"]["subject"], "bank+alice")
        self.assertTrue(w.invoke("bank_get_balance", {"account": "checking"})["ok"])
        self.assertTrue(w.invoke("ref_get_rate", {"currency": "EUR"})["ok"])  # public

    def test_wrong_arguments_and_unknown_tools_change_nothing(self):
        w = signed_in()
        before = w.snapshot()
        for name, arguments in (("bank_transfer", {"source": "checking"}),
                                ("no_such_tool", {}),
                                ("bank_transfer", {"source": "checking", "target": "rent",
                                                   "amount": 99999})):
            payload = w.invoke(name, arguments)
            self.assertFalse(payload["ok"])
            self.assertEqual(payload["outcome"], "not_executed")
        self.assertEqual((w.snapshot(), w.writes), (before, []))

    def test_idempotency_key_deduplicates_only_where_declared(self):
        w = signed_in()
        transfer = {"source": "checking", "target": "rent", "amount": 100}
        first = w.invoke("bank_transfer", transfer, key="k1")
        again = w.invoke("bank_transfer", transfer, key="k1")
        self.assertEqual((again["value"], again.get("replayed")), (first["value"], True))
        self.assertEqual(w.bank["accounts"]["rent"]["balance"], 100)
        order = {"sku": "sku-pen", "quantity": 1}
        w.invoke("shop_place_order", order, key="k2")
        w.invoke("shop_place_order", order, key="k2")  # this tool declares no key
        self.assertEqual(len(w.shop["orders"]), 2)
        self.assertEqual(w.invoke("op_status", {"key": "k2"})["value"]["state"], "executed")
        self.assertEqual(w.operations["k2"]["executions"], 2)

    def test_operation_status_only_states_what_the_server_saw(self):
        w = signed_in(faults=[{"tool": "bank_transfer", "nth": 1, "kind": "lost_ack"}])
        transfer = {"source": "checking", "target": "rent", "amount": 100}
        # Sent without a key and applied, reply lost: the server cannot match
        # any key to it, and must not claim it was "not executed".
        self.assertEqual(w.invoke("bank_transfer", transfer)["outcome"], "unknown")
        self.assertEqual(w.invoke("op_status", {"key": "any"})["value"]["state"], "unknown")
        # A keyed request the server refused is known not to have been applied.
        refused = w.invoke("bank_transfer", {**transfer, "amount": 10 ** 6}, key="k-big")
        self.assertEqual(refused["error_code"], "business_error")
        self.assertEqual(w.invoke("op_status", {"key": "k-big"})["value"]["state"],
                         "not_executed")

    def test_operation_key_is_bound_to_its_request_and_its_sender(self):
        w = signed_in()
        transfer = {"source": "checking", "target": "rent", "amount": 100}
        w.invoke("bank_transfer", transfer, key="k1")
        other = w.invoke("bank_transfer", {**transfer, "amount": 5}, key="k1")
        self.assertEqual(other["error_code"], "business_error")
        self.assertEqual(w.bank["accounts"]["rent"]["balance"], 100)
        w.invoke("auth_logout", {"app": "bank"})
        self.assertEqual(w.invoke("op_status", {"key": "k1"})["error_code"], "unauthorized")

    def test_refusal_at_the_door_does_not_use_up_a_planned_fault(self):
        w = world(faults=[{"tool": "bank_transfer", "nth": 1, "kind": "lost_ack"}])
        transfer = {"source": "checking", "target": "rent", "amount": 100}
        self.assertEqual(w.invoke("bank_transfer", transfer)["error_code"], "unauthorized")
        w.invoke("auth_login", {"app": "bank", "username": "alice", "password": "pw-alice"})
        self.assertEqual(w.invoke("bank_transfer", transfer)["error_code"], "response_lost")
        self.assertEqual([hit["kind"] for hit in w.fault_hits], ["lost_ack"])
        self.assertEqual([a["outcome"] for a in w.write_attempts], ["committed_reply_lost"])

    def test_malformed_requests_are_refused_and_change_nothing(self):
        w = signed_in()
        for amount in (float("nan"), float("inf"), "100", True, None, [1]):
            reply = w.invoke("bank_transfer", {"source": "checking", "target": "rent",
                                               "amount": amount})
            self.assertEqual((reply["error_code"], reply["outcome"]),
                             ("business_error", "not_executed"), amount)
        self.assertEqual(w.invoke("bank_get_balance", {"account": ["x"]})["error_code"],
                         "business_error")
        self.assertEqual(w.bank["accounts"]["checking"]["balance"], 500)
        self.assertFalse(w.invoke("auth_login", {"app": "bank", "username": "ghost",
                                                 "password": "x"})["ok"])

    def test_task_file_with_a_fault_that_cannot_fire_is_rejected(self):
        for fault in ({"tool": "bank_get_balance", "nth": 1, "kind": "lost_ack"},
                      {"tool": "no_such_tool", "nth": 1, "kind": "rate_limit"},
                      {"tool": "bank_transfer", "nth": 1, "kind": "explode"},
                      {"tool": "bank_transfer", "kind": "lost_ack"}):
            with self.assertRaises(ValueError):
                world(faults=[fault])

    def test_faults_fire_on_the_configured_call(self):
        w = signed_in(faults=[{"tool": "shop_get_stock", "nth": 1, "kind": "rate_limit",
                               "retry_after": 2},
                              {"tool": "bank_transfer", "nth": 1, "kind": "lost_ack"}])
        limited = w.invoke("shop_get_stock", {"sku": "sku-pen"})
        self.assertEqual((limited["error_code"], limited["retryable"], limited["retry_after"]),
                         ("rate_limited", True, 2))
        self.assertTrue(w.invoke("shop_get_stock", {"sku": "sku-pen"})["ok"])
        lost = w.invoke("bank_transfer", {"source": "checking", "target": "rent",
                                          "amount": 100}, key="k")
        self.assertEqual((lost["ok"], lost["outcome"]), (False, "unknown"))
        self.assertEqual(w.bank["accounts"]["rent"]["balance"], 100)  # it did apply

    def test_events_fire_at_turn_starts_for_every_agent(self):
        w = signed_in("lab-external-volatile")
        self.assertEqual(w.advance(5, 0)["applied"], [])  # many steps, same turn
        self.assertEqual(w.invoke("shop_get_price", {"sku": "sku-lamp"})["value"]["price"], 40)
        self.assertEqual(w.advance(6, 1)["applied"], ["set_price"])
        self.assertEqual(w.invoke("shop_get_price", {"sku": "sku-lamp"})["value"]["price"], 55)

    def test_shadow_reads_leave_no_trace_and_see_the_truth(self):
        w = signed_in(faults=[{"tool": "bank_get_balance", "nth": 1, "kind": "unavailable"}])
        shadow = w.invoke("bank_get_balance", {"account": "checking"}, shadow=True)
        self.assertEqual(shadow["value"]["balance"], 500)  # no fault, no counter
        self.assertNotIn("bank_get_balance", w.calls)
        # A measurement read is still subject to authorization.
        w.invoke("auth_logout", {"app": "bank"})
        self.assertEqual(w.invoke("bank_get_balance", {"account": "checking"},
                                  shadow=True)["error_code"], "unauthorized")
        self.assertFalse(w.invoke("bank_transfer", {"source": "checking", "target": "rent",
                                                    "amount": 1}, shadow=True)["ok"])

    def test_catalog_versions_and_protocol_fields(self):
        w = world("lab-catalog-change")
        listing = w.catalog()
        self.assertEqual((listing["version"], listing["cacheScope"], listing["ttlMs"]),
                         ("v1", "public", 600000))
        names = {tool["name"] for tool in listing["tools"]}
        self.assertNotIn("shop_get_bulk_price", names)
        self.assertEqual(w.advance(1, 1)["catalog_version"], "v2")
        self.assertIn("shop_get_bulk_price", {t["name"] for t in w.catalog()["tools"]})
        hints = {t["name"]: t["annotations"] for t in listing["tools"]}
        self.assertTrue(hints["bank_get_balance"]["readOnlyHint"])
        self.assertFalse(hints["shop_place_order"]["readOnlyHint"])
        self.assertTrue(hints["bank_transfer"]["idempotentHint"])

    def test_oracle_checks_state_writes_and_answers(self):
        w = signed_in("lab-write-disjoint")
        w.invoke("bank_transfer", {"source": "checking", "target": "rent", "amount": 100})
        good = ["savings 1200, checking 500, desk stock 3", "",
                "savings 1200, checking 400, sku-desk has 3 in stock"]
        self.assertEqual(w.oracle(good)["verdict"], "pass")
        self.assertEqual(w.oracle(["1200 500 3", "", "savings 1200, checking 1400, 3"])
                         ["verdict"], "fail")
        w.invoke("bank_transfer", {"source": "checking", "target": "rent", "amount": 100})
        self.assertFalse(w.oracle(good)["success"])  # a duplicate write is a failure

    def test_answer_values_are_read_as_numbers_next_to_their_entity(self):
        from exp_core.lab.judge import judge_answer
        rule = {"turn": 0, "contains": [1200, 500],
                "labels": {"1200": "savings", "500": "checking"}}

        def status(text):
            return judge_answer(text, rule)[0]
        self.assertEqual(status("Savings balance: 1,200.00. Checking balance: 500."), "ok")
        self.assertEqual(status("checking is 500 (it was 700); savings 1200"), "ok")
        # The two balances swapped: both numbers are there, next to the wrong account.
        self.assertEqual(status("savings 500, checking 1200"), "review")
        # Other numbers are not the required ones.
        self.assertEqual(status("savings 1200, checking 500.99"), "fail")
        self.assertEqual(status("savings 1200, checking -500"), "fail")
        self.assertEqual(status("savings 1200, checking 5000"), "fail")
        # Digits inside identifiers are not values.
        self.assertEqual(status("savings 1200; see transfer tr-500 on checking"), "fail")

    def test_claimed_success_after_a_refused_write_needs_review(self):
        from exp_core.lab.judge import judge_answer
        rule = {"turn": 0, "contains": [500], "labels": {"500": "checking"},
                "refused_write": True}
        self.assertEqual(judge_answer("The transfer failed; checking is still 500.", rule)[0], "ok")
        self.assertEqual(judge_answer("The transfer succeeded. Checking: 500.", rule)[0],
                         "review")

    def test_final_state_and_write_arguments_are_compared_in_full(self):
        w = signed_in("lab-write-overlap")
        w.invoke("auth_login", {"app": "shop", "username": "alice", "password": "pw-alice"})
        w.invoke("shop_place_order", {"sku": "sku-desk", "quantity": 2})  # reference: 1
        checks = {c["check"]: c["ok"] for c in w.oracle([])["checks"]}
        self.assertFalse(checks["final state"])
        self.assertFalse(checks["committed writes"])


class Contract(unittest.TestCase):
    """The published contract must be true, or every cache result is meaningless."""

    def setUp(self):
        self.contract = world().contract()

    def test_every_tool_has_a_complete_contract(self):
        for name, spec in {**TOOLS, **LATER_TOOLS}.items():
            entry = self.contract[name]
            if spec["kind"] == "read":
                self.assertIn(entry["fresh_class"], {"static", "slow", "volatile", "strict"})
            if spec["kind"] == "write":
                self.assertTrue(entry["affects"], name)

    def test_declared_write_effects_cover_every_read_they_change(self):
        """Run each write and confirm every read whose result changed depends on a
        resource the write declares (the contract may over-declare, never under)."""
        reads = [("bank_get_balance", {"account": a}) for a in ("checking", "savings", "rent")]
        reads += [("bank_list_transfers", {"account": a}) for a in ("checking", "savings", "rent")]
        reads += [("bank_list_accounts", {}), ("shop_list_items", {}), ("shop_list_orders", {})]
        reads += [(tool, {"sku": sku}) for tool in ("shop_get_item", "shop_get_price",
                                                    "shop_get_stock")
                  for sku in ("sku-pen", "sku-lamp", "sku-desk")]
        reads += [("ref_get_rate", {"currency": "EUR"}), ("ref_list_currencies", {})]
        writes = [("bank_transfer", {"source": "checking", "target": "rent", "amount": 10}),
                  ("bank_transfer", {"source": "savings", "target": "checking", "amount": 5}),
                  ("shop_place_order", {"sku": "sku-lamp", "quantity": 1}),
                  ("shop_cancel_order", {"order_id": "ord-1"})]
        w = signed_in()
        w.invoke("shop_place_order", {"sku": "sku-pen", "quantity": 1})  # so ord-1 exists
        for name, arguments in writes:
            before = [w.invoke(r, a, shadow=True) for r, a in reads]
            self.assertTrue(w.invoke(name, arguments)["ok"], name)
            affected = {t.format(**arguments) for t in self.contract[name]["affects"]}
            for (read, read_args), old in zip(reads, before):
                if w.invoke(read, read_args, shadow=True) == old:
                    continue
                depends = {t.format(**read_args) for t in self.contract[read]["reads"]}
                self.assertTrue(depends & affected,
                                f"{name}{arguments} changed {read}{read_args} undeclared")


class ScriptedMechanismChecks(unittest.TestCase):
    def passed(self, method, override=None):
        return {tid for tid, task in TASKS.items()
                if run_gold(task, runner.resolve_policy(method, override))["oracle"]["success"]}

    def test_every_task_is_solvable_without_any_cache(self):
        # The scripted agent blindly retries a failed call, so the lost-ack task
        # is the one a cache-free host cannot get right.
        self.assertEqual(set(TASKS) - self.passed("bare"), {"lab-fault-lost-ack"})

    def test_scheduler_passes_everything_except_the_contract_drift_task(self):
        self.assertEqual(set(TASKS) - self.passed("ours", {"scheduler": "default"}),
                         {"lab-contract-drift"})

    def test_server_cache_without_invalidation_fails_after_writes(self):
        failed = set(TASKS) - self.passed("rw_fastmcp")
        self.assertLessEqual({"lab-write-disjoint", "lab-write-overlap"}, failed)

    def test_scheduler_needs_fewer_requests_than_no_cache(self):
        def total(method, override=None):
            return sum(run_gold(task, runner.resolve_policy(method, override))["requests"]
                       for task in TASKS.values())
        self.assertLess(total("ours", {"scheduler": "default"}), total("bare"))

    def test_lost_ack_is_recovered_with_the_idempotency_key(self):
        task = TASKS["lab-fault-lost-ack"]
        new = run_gold(task, runner.resolve_policy("ours", {"scheduler": "default"}))
        self.assertEqual(len(new["writes"]), 1)
        self.assertEqual(new["errors"], [])  # recovered inside the host
        off = run_gold(task, runner.resolve_policy("ours", {"scheduler": {
            "base": "default", "overrides": {"recovery": {"unknown_write": "surface_only"}}}}))
        self.assertEqual(len(off["writes"]), 2)  # surfaced, and the naive agent re-sent it

    def test_doomed_call_is_not_sent_twice(self):
        task = TASKS["lab-failure-repeat"]
        new = run_gold(task, runner.resolve_policy("ours", {"scheduler": "default"}))
        bare = run_gold(task, runner.resolve_policy("bare"))
        self.assertEqual((new["hits"]["M4_failure_hits"], bare["requests"] - new["requests"]),
                         (1, 1))

    def auth_task(self):
        """Turn 0 reads the shop; in turn 1 the bank session is gone and a bank
        read that is not cached surfaces the authorization error."""
        login = lambda app: {"name": "auth_login", "arguments": {
            "app": app, "username": "alice", "password": "pw-alice"}}
        task = copy.deepcopy(TASKS["lab-auth-expiry"])
        task["gold"] = [
            {"steps": [[login("bank"), login("shop")],
                       [{"name": "shop_list_orders", "arguments": {}}]], "answer": ""},
            {"steps": [[{"name": "bank_get_balance", "arguments": {"account": "savings"}}],
                       [{"name": "shop_list_orders", "arguments": {}}]], "answer": ""}]
        return task

    def test_partitioned_auth_keeps_the_other_app_cached(self):
        base = {"ttl": {"adaptive": {"enabled": False}}, "recovery": {"auth": "surface_only"}}
        for partition, shop_hits in ((True, 1), (False, 0)):
            result = run_gold(self.auth_task(), runner.resolve_policy("ours", {"scheduler": {
                "base": "default", "overrides": {**base, "coupling": {
                    "partition_by_principal": partition}}}}))
            self.assertEqual([e["error"] for e in result["errors"]], ["unauthorized"])
            self.assertEqual(result["hits"]["M2_hits"], shop_hits, partition)

    def test_host_signs_in_again_by_itself_when_allowed(self):
        result = run_gold(self.auth_task(), runner.resolve_policy("ours", {"scheduler": {
            "base": "default", "overrides": {"ttl": {"adaptive": {"enabled": False}}}}}))
        self.assertEqual(result["errors"], [])      # refreshed and retried in the host
        self.assertEqual(result["retries"], 1)

    def test_revoked_session_is_unknown_to_the_host_until_a_request_fails(self):
        """A read cached before the server dropped the session is still served:
        there is no revocation signal until some request reaches the server."""
        result = run_gold(TASKS["lab-auth-expiry"], runner.resolve_policy("ours", {
            "scheduler": {"base": "default", "overrides": {
                "ttl": {"adaptive": {"enabled": False}}}}}))
        self.assertTrue(result["oracle"]["success"])
        self.assertEqual(result["errors"], [])
        self.assertGreaterEqual(result["hits"]["M2_hits"], 1)

    def test_stale_hit_rate_exposes_the_contract_drift(self):
        policy = runner.resolve_policy("ours", {"scheduler": {
            "base": "default", "overrides": {"ttl": {"adaptive": {"enabled": False}}}}})
        drift = run_gold(TASKS["lab-contract-drift"], policy, shadow_verify=True)
        clean = run_gold(TASKS["lab-repeat-read"], policy, shadow_verify=True)
        self.assertEqual((drift["stale_hit_rate"], clean["stale_hit_rate"]), (1.0, 0.0))

    def test_sampled_verification_catches_the_drift_when_it_samples(self):
        policy = runner.resolve_policy("ours", {"scheduler": {
            "base": "default", "overrides": {"ttl": {"adaptive": {"enabled": True, "verify_sample_rate": 1.0}}}}})
        result = run_gold(TASKS["lab-contract-drift"], policy)
        self.assertTrue(result["oracle"]["success"])


class SuiteWiring(unittest.TestCase):
    def test_lab_manifest_validates_and_builds_jobs(self):
        document = json.loads((ROOT / "data" / "lab-cases.json").read_text(encoding="utf-8"))
        cases = matrix.validate_cases(document)
        self.assertEqual({case["task_id"] for case in cases}, set(TASKS))
        jobs = matrix.build_jobs(cases, ["bare", "ours"], code_sha="x" * 64)
        self.assertEqual(len(jobs), 2 * len(TASKS))
        broken = copy.deepcopy(document)
        broken["cases"][0]["task_id"] = "missing"
        with self.assertRaises(ValueError):
            matrix.validate_cases(broken)

    def test_rsi_accepts_the_lab_manifest(self):
        out = StringIO()
        with redirect_stdout(out):
            code = rsi.main(["--space", "scheduler", "--dry-run", "--manifest",
                             str(ROOT / "data" / "lab-cases.json")])
        self.assertEqual(code, 0)
        self.assertIn("lab-fault-lost-ack", out.getvalue())


if __name__ == "__main__":
    unittest.main()


class ErrorDetail(unittest.TestCase):
    def test_agent_is_told_why_the_server_refused_a_malformed_call(self):
        from exp_core.access import Access
        from exp_core.runtime import AccessFailure, failure_result
        from exp_core.lab.harness import build_runtime
        task = TASKS["lab-write-disjoint"]
        for override in (None, {"scheduler": "default"}):
            world = LabWorld(task)
            method = "ours" if override else "bare"
            access = Access(build_runtime(runner.resolve_policy(method, override)), "lab", "s",
                            lambda name, arguments: world.invoke(name, arguments),
                            invoke_keyed=lambda name, arguments, key: world.invoke(
                                name, arguments, key=key),
                            lab_contract=world.contract())
            access.call("auth_login", {"app": "bank", "username": "alice",
                                       "password": "pw-alice"}, "op-1")
            with self.assertRaises(AccessFailure) as caught:
                access.call("bank_transfer", {"source": "checking", "target": "rent",
                                              "amount": "100"}, "op-2")
            shown = failure_result(caught.exception, "op-2")
            self.assertEqual(shown["error"], "business_error")
            self.assertIn("finite number", shown["detail"])
            self.assertEqual(shown["operation_id"], "op-2")

    def test_failure_without_an_explanation_has_no_detail(self):
        from exp_core.runtime import AccessFailure, failure_result
        self.assertEqual(failure_result(AccessFailure("write_outcome_unknown")),
                         {"error": "write_outcome_unknown"})
