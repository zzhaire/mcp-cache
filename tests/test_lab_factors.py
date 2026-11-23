"""Virtual time, sign-in tokens, event triggers, experiment factors, the
delivery audit and the run report of the lab server."""
import copy
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from exp_core import runner  # noqa: E402
from exp_core.lab.factors import DeliveryAudit, degrade_contract  # noqa: E402
from exp_core.lab.harness import load_tasks, run_gold  # noqa: E402
from exp_core.lab.report import bootstrap, summarise  # noqa: E402
from exp_core.lab.world import LabWorld  # noqa: E402

TASKS = load_tasks()
NEW = {"scheduler": "default"}
ALICE = {"app": "bank", "username": "alice", "password": "pw-alice"}


def world(task_id="lab-repeat-read", **changes):
    task = copy.deepcopy(TASKS[task_id])
    task.update(changes)
    return LabWorld(task)


def gold(task_id, method="ours", override=NEW, **kw):
    return run_gold(TASKS[task_id], runner.resolve_policy(method, override), **kw)


class TimeAndTokens(unittest.TestCase):
    def test_time_is_virtual_and_follows_steps_and_turn_gaps(self):
        w = world(step_seconds=7, turn_gap_s=100)
        self.assertEqual(w.advance(3, 0)["now"], 21)
        self.assertEqual(w.advance(4, 2)["now"], 228)

    def test_a_sign_in_is_a_token_that_expires_on_the_virtual_clock(self):
        w = world(session_ttl_s=60)
        token = w.invoke("auth_login", ALICE, credentials={})["value"]["session"]
        read = lambda held: w.invoke("bank_get_balance", {"account": "checking"},  # noqa: E731
                                     credentials=held)
        self.assertTrue(read({"bank": token})["ok"])
        self.assertEqual(read({})["error_code"], "unauthorized")           # nothing presented
        self.assertEqual(read({"bank": "made-up"})["error_code"], "unauthorized")
        w.advance(6, 0)                                                     # 60 virtual seconds
        self.assertEqual(read({"bank": token})["error_code"], "unauthorized")

    def test_logout_and_server_side_revocation_kill_the_token(self):
        w = world()
        first = w.invoke("auth_login", ALICE, credentials={})["value"]["session"]
        w.invoke("auth_logout", {"app": "bank"}, credentials={"bank": first})
        self.assertFalse(w.invoke("bank_list_accounts", {}, credentials={"bank": first})["ok"])
        w = world(events=[{"at_turn": 1, "kind": "expire_session", "app": "bank"}])
        token = w.invoke("auth_login", ALICE, credentials={})["value"]["session"]
        w.advance(1, 1)
        self.assertFalse(w.invoke("bank_list_accounts", {}, credentials={"bank": token})["ok"])

    def test_host_credential_cache_does_not_outlive_the_token(self):
        result = gold("time-token-expires-mid-task")
        self.assertTrue(result["oracle"]["success"])
        self.assertEqual(result["delivery"]["after_revocation"], 0)
        self.assertLess(result["requests"], gold("time-token-expires-mid-task", "bare", None)
                        ["requests"])


class Events(unittest.TestCase):
    def test_after_call_counts_the_agents_calls_not_the_servers(self):
        w = world(events=[{"after_call": {"tool": "shop_get_price", "nth": 2},
                           "kind": "set_price", "sku": "sku-lamp", "price": 99}])
        w.note("shop_get_price")          # answered from a host cache: the server saw nothing
        self.assertEqual(w.shop["items"]["sku-lamp"]["price"], 40)
        w.note("shop_get_price")
        self.assertEqual(w.shop["items"]["sku-lamp"]["price"], 99)

    def test_at_time_event_fires_when_the_clock_reaches_it(self):
        w = world(events=[{"at_time": 25, "kind": "set_stock", "sku": "sku-pen", "stock": 1}])
        w.advance(2, 0)
        self.assertEqual(w.shop["items"]["sku-pen"]["stock"], 100)
        w.advance(3, 0)
        self.assertEqual(w.shop["items"]["sku-pen"]["stock"], 1)

    def test_event_needs_exactly_one_trigger(self):
        for event in ({"kind": "set_price", "sku": "sku-pen", "price": 1},
                      {"kind": "set_price", "sku": "sku-pen", "price": 1, "at_turn": 1,
                       "at_time": 5}):
            with self.assertRaises(ValueError):
                world(events=[event])

    def test_lost_request_looks_like_lost_reply_but_changes_nothing(self):
        transfer = {"source": "checking", "target": "rent", "amount": 100}
        replies = {}
        for kind in ("lost_request", "lost_ack"):
            w = world(faults=[{"tool": "bank_transfer", "kind": kind, "on_calls": [1]}])
            w.invoke("auth_login", ALICE)
            replies[kind] = (w.invoke("bank_transfer", transfer, key="k"),
                             w.bank["accounts"]["checking"]["balance"],
                             w.invoke("op_status", {"key": "k"})["value"]["state"])
        self.assertEqual(replies["lost_request"][0], replies["lost_ack"][0])
        self.assertEqual((replies["lost_request"][1:], replies["lost_ack"][1:]),
                         ((500, "unknown"), (400, "executed")))


class Factors(unittest.TestCase):
    def test_contract_levels(self):
        full = world().contract()
        notes = degrade_contract(full, "annotations")
        self.assertEqual((notes["bank_get_balance"]["kind"], notes["bank_get_balance"]["reads"],
                          notes["bank_get_balance"]["fresh_class"]), ("read", [], None))
        self.assertEqual(degrade_contract(full, "none")["bank_get_balance"]["kind"], "write")
        wrong = degrade_contract(full, "wrong")
        self.assertNotIn("bank/balance/{target}", wrong["bank_transfer"]["affects"])
        self.assertEqual(full["shop_get_price"]["fresh_class"], "volatile")   # untouched

    def test_less_contract_means_fewer_savings_never_wrong_answers(self):
        requests = {}
        for level in ("full", "annotations", "none"):
            result = gold("m2-write-interleave", contract_level=level)
            self.assertTrue(result["oracle"]["success"], level)
            requests[level] = result["requests"]
        self.assertLess(requests["full"], requests["annotations"])
        self.assertLessEqual(requests["annotations"], requests["none"])

    def test_wrong_contract_is_caught_by_the_delivery_audit(self):
        result = gold("m2-write-interleave", contract_level="wrong", shadow_verify=True)
        self.assertGreater(result["delivery"]["stale"], 0)
        self.assertFalse(result["oracle"]["success"])

    def test_listing_once_per_connection_removes_the_repeated_lookups(self):
        for method, override in (("bare", None), ("ours", NEW)):
            turn = gold("m1-relist-large-stable", method, override)
            once = gold("m1-relist-large-stable", method, override, listing="per_connection")
            self.assertEqual((turn["catalog_requests"], once["catalog_requests"]),
                             ((8, 1) if method == "bare" else (1, 1)))
            self.assertEqual(turn["other_requests"], once["other_requests"])

    def test_audit_counts_every_method_the_same_way(self):
        for method, override in (("bare", None), ("strong", None), ("rw_gateway", None)):
            delivery = gold("lab-contract-drift", method, override)["delivery"]
            self.assertEqual(delivery["stale"] > 0, method != "bare", method)

    def test_audit_tells_revocation_from_staleness(self):
        truth = lambda name, arguments: {"ok": False, "error_code": "unauthorized"}  # noqa: E731
        audit = DeliveryAudit(truth, world().contract())
        audit.delivered("bank_get_balance", {"account": "checking"}, {"balance": 1}, False)
        self.assertEqual((audit.stats["stale"], audit.stats["after_revocation"]), (0, 1))


class StrongBaseline(unittest.TestCase):
    def test_same_abilities_fixed_rules(self):
        policy = runner.resolve_policy("strong")["scheduler"]
        ours = runner.resolve_policy("ours", NEW)["scheduler"]
        self.assertEqual(policy["recovery"], ours["recovery"])
        self.assertEqual((policy["invalidation"]["granularity"], policy["ttl"]["volatile"],
                          policy["publish"]["read_results"]), ("app", 0, False))

    def test_it_recovers_unclear_writes_but_invalidates_coarsely(self):
        self.assertTrue(gold("lab-fault-lost-ack", "strong", None)["oracle"]["success"])
        self.assertGreater(gold("m2-write-interleave", "strong", None)["requests"],
                           gold("m2-write-interleave")["requests"])


class Report(unittest.TestCase):
    def test_paired_difference_and_repeat_spread(self):
        run = lambda rep, requests, verdict="pass": {  # noqa: E731
            "repeat": rep, "verdict": verdict, "safe": True, "requests": requests, "catalog": 1,
            "model_calls": 1, "input_tokens": 1, "stale": 0, "audited": True}
        runs = {"bare": {"a": [run(0, 10), run(1, 12)], "b": [run(0, 20), run(1, 20)]},
                "ours": {"a": [run(0, 6), run(1, 6)], "b": [run(0, 15), run(1, 17, "fail")]}}
        summary = summarise(runs)
        self.assertEqual((summary["ours"]["pass"], summary["ours"]["fail"]), (3, 1))
        mean, low, high = summary["ours"]["requests_vs_bare"]
        self.assertAlmostEqual(mean, -4.5)            # ((6-11) + (16-20)) / 2
        self.assertLessEqual(low, mean)
        self.assertGreaterEqual(high, mean)
        self.assertEqual(summary["bare"]["requests_total_sd_over_repeats"] > 0, True)
        self.assertIsNone(bootstrap([]))


if __name__ == "__main__":
    unittest.main()
