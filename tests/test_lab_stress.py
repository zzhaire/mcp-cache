"""The M1 / M2 / M3 / mixed stress cases: well-formed, solvable, and each one
exercises the mechanism it was written for."""
import collections
import copy
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data import lab_stress  # noqa: E402
from exp_core import matrix, runner  # noqa: E402
from exp_core.lab.harness import TurnScript, load_tasks, run_gold  # noqa: E402
from exp_core.lab.world import LabWorld  # noqa: E402

DOCUMENT = json.loads((ROOT / "data" / "lab-stress-cases.json").read_text(encoding="utf-8"))
TASKS = {case["task_id"]: load_tasks()[case["task_id"]] for case in DOCUMENT["cases"]}
ADVERSARIAL = {tid for tid, task in TASKS.items() if task.get("adversarial")}
NEW = {"scheduler": "default"}
MID_TURN = "time-midturn-price-change"
RESULTS = {}


def run(task_id, method="ours", override=None, **kw):
    key = (task_id, method, json.dumps(override, sort_keys=True), tuple(sorted(kw.items())))
    if key not in RESULTS:
        RESULTS[key] = run_gold(TASKS[task_id], runner.resolve_policy(method, override), **kw)
    return RESULTS[key]


def passed(task_id, method="ours", override=None):
    return run(task_id, method, override)["oracle"]["success"]


class Suite(unittest.TestCase):
    def test_six_scenarios_of_five_cases(self):
        cases = matrix.validate_cases(DOCUMENT)
        self.assertEqual(collections.Counter(case["scenario"] for case in cases),
                         {"m1_catalog": 5, "m2_reads": 5, "m3_credentials": 5, "mixed": 5,
                          "timing": 5, "m4_operations": 5})

    def test_generated_files_match_the_builder(self):
        from data.lab_expect import enrich
        built = {task["id"]: enrich(task) for task in lab_stress.TASKS}
        self.assertEqual(built, TASKS)

    def test_every_case_is_fully_specified(self):
        for tid, task in TASKS.items():
            with self.subTest(task=tid):
                self.assertGreaterEqual(len(task["turns"]), 3)
                self.assertEqual(len(task["turns"]), len(task["gold"]))
                self.assertTrue(task["extreme"] and task["note"])
                rules = task["expected"]["answers"]
                self.assertEqual(sorted(rule["turn"] for rule in rules),
                                 list(range(len(task["turns"]))))
                served = {rule["turn"] for rule in task["expected"].get("served", [])}
                for rule in rules:
                    # Every turn is judged by its answer or by what was served.
                    self.assertTrue(rule["contains"] or rule["turn"] in served)
                    # A truthful answer may mention an old value; leaks are
                    # judged on the served results instead.
                    self.assertNotIn("excludes", rule)

    def test_adversarial_cases_are_the_two_contract_drifts(self):
        self.assertEqual(ADVERSARIAL, {"m2-contract-drift-repeated",
                                       "time-midturn-contract-drift"})


class Correctness(unittest.TestCase):
    def test_every_case_is_solvable_without_any_cache(self):
        self.assertEqual([tid for tid in TASKS if not passed(tid, "bare")], [])

    def test_scheduler_fails_only_the_adversarial_cases(self):
        self.assertEqual({tid for tid in TASKS if not passed(tid, "ours", NEW)}, ADVERSARIAL)

    def test_reusing_volatile_reads_misses_a_change_inside_a_turn(self):
        # Why the default does not keep volatile reads: with a lifetime of a few
        # steps, a price that changes between two reads of one turn is missed
        # and an order is placed that should not have been.
        kept = run(MID_TURN, "ours", {"scheduler": {
            "base": "default", "overrides": {"ttl": {"volatile": 4}}}}, shadow_verify=True)
        self.assertFalse(kept["oracle"]["success"])
        self.assertFalse(kept["oracle"]["safe"])
        self.assertGreater(kept["delivery"]["stale"], 0)
        self.assertTrue(run(MID_TURN, "ours", NEW)["oracle"]["safe"])

    def test_scheduler_delivers_stale_values_only_in_the_adversarial_cases(self):
        for tid in TASKS:
            delivery = run(tid, "ours", NEW, shadow_verify=True)["delivery"]
            with self.subTest(task=tid):
                if tid in ADVERSARIAL:
                    self.assertGreater(delivery["stale"], 0)
                else:
                    self.assertEqual(delivery["stale"], 0)

    def test_no_method_with_the_scheduler_applies_a_write_twice(self):
        for tid in TASKS:
            if TASKS[tid]["family"] == "m4_operations":
                for method, override in (("ours", NEW), ("strong", None)):
                    with self.subTest(task=tid, method=method):
                        result = run(tid, method, override)
                        self.assertTrue(result["oracle"]["safe"])
                        self.assertTrue(result["oracle"]["success"])

    def test_planned_faults_really_fire(self):
        for tid, task in TASKS.items():
            planned = sum(len(f.get("on_calls", [f.get("nth")])) for f in task["faults"]
                          if f["tool"] != "op_status")
            if planned:
                with self.subTest(task=tid):
                    self.assertEqual(len([h for h in run(tid, "bare")["fault_hits"]
                                          if h["tool"] != "op_status"]), planned)

    def test_full_verification_rescues_the_adversarial_case(self):
        always = {"scheduler": {"base": "default", "overrides": {
            "ttl": {"adaptive": {"enabled": True, "verify_sample_rate": 1.0}}}}}
        for tid in ADVERSARIAL:
            self.assertTrue(passed(tid, "ours", always))

    def test_scheduler_never_needs_more_requests_than_no_cache(self):
        for tid in TASKS:
            with self.subTest(task=tid):
                self.assertLessEqual(run(tid, "ours", NEW)["requests"],
                                     run(tid, "bare")["requests"])


class CatalogExtremes(unittest.TestCase):
    def test_stable_large_catalog_is_listed_once(self):
        new, bare = run("m1-relist-large-stable", "ours", NEW), run("m1-relist-large-stable", "bare")
        self.assertEqual((new["hits"]["M1_hits"], bare["hits"]["M1_hits"]), (7, 0))
        self.assertEqual(len(LabWorld(TASKS["m1-relist-large-stable"]).catalog()["tools"]), 77)

    def test_changing_catalog_is_never_served_from_cache(self):
        self.assertEqual(run("m1-version-churn", "ours", NEW)["hits"]["M1_hits"], 0)

    def test_silent_changes_need_the_server_ttl_and_the_content_hash(self):
        for tid in ("m1-silent-semantic-change", "m1-tool-retired"):
            with self.subTest(task=tid):
                self.assertTrue(passed(tid, "ours", NEW))
                self.assertFalse(passed(tid, "rw_merchant"))  # caches the catalog by version only
        stats = run("m1-silent-semantic-change", "ours", NEW)["stats"]
        self.assertEqual(stats["catalog_silent_changes"], 1)

    def test_notification_noise_does_not_drop_unrelated_reads(self):
        scoped = run("m1-notification-noise", "ours", NEW)
        blunt = run("m1-notification-noise", "ours", {"scheduler": {
            "base": "default", "overrides": {"coupling": {"enabled_rules": [
                "auth_upgrade_keeps_m2", "credential_scope_superset"]}}}})
        self.assertTrue(scoped["oracle"]["success"] and blunt["oracle"]["success"])
        self.assertEqual(scoped["hits"]["M2_hits"], 16)   # 4 reads x 4 repeat turns
        self.assertLess(blunt["hits"]["M2_hits"], scoped["hits"]["M2_hits"])


class ReadExtremes(unittest.TestCase):
    def test_read_storm_hits_every_repeat(self):
        new = run("m2-read-storm", "ours", NEW)
        self.assertEqual(new["hits"]["M2_hits"], 30)  # 6 reads x 5 repeat turns
        self.assertEqual(new["requests"], 9)

    def test_resource_level_invalidation_beats_coarser_levels(self):
        def requests(granularity):
            result = run("m2-write-interleave", "ours", {"scheduler": {
                "base": "default", "overrides": {
                    "invalidation": {"granularity": granularity},
                    "ttl": {"adaptive": {"enabled": False}}}}})
            self.assertTrue(result["oracle"]["success"], granularity)
            return result["requests"]
        self.assertLess(requests("resource"), requests("app"))
        self.assertEqual(requests("app"), requests("global"))  # one app does all the writing

    def test_stale_volatile_data_places_the_wrong_order(self):
        self.assertTrue(passed("m2-volatile-churn", "ours", NEW))
        self.assertFalse(passed("m2-volatile-churn", "ours", {"context_dedup": False}))

    def test_oversized_result_is_refused_but_still_answered(self):
        # Both histories fit the default entry limit and are cached.
        new = run("m2-large-payloads", "ours", NEW)
        self.assertTrue(new["oracle"]["success"])
        self.assertEqual(new["stats"]["rejected_admissions"], 0)
        self.assertEqual(new["hits"]["M2_hits"], 6)
        # With a 20 KB entry limit the 600-row history (about 40 KB) is refused
        # twice per turn: as a cached read (M2) and as the stored result of its
        # operation record (M4). The 150-row history is still cached.
        small = run("m2-large-payloads", "ours", {"scheduler": {
            "overrides": {"capacity": {"max_entry_bytes": 20000}}}})
        self.assertTrue(small["oracle"]["success"])
        self.assertEqual(small["stats"]["rejected_admissions"], 8)
        self.assertEqual(small["hits"]["M2_hits"], 3)


class CredentialExtremes(unittest.TestCase):
    def test_repeated_sign_ins_are_answered_from_the_credential_cache(self):
        new = run("m3-relogin-storm", "ours", NEW)
        self.assertEqual((new["hits"]["M3_hits"], new["requests"]), (10, 5))

    def test_host_recovers_expired_sessions_without_surfacing_them(self):
        new, bare = run("m3-session-churn", "ours", NEW), run("m3-session-churn", "bare")
        self.assertEqual(new["errors"], [])
        self.assertEqual(len(bare["errors"]), 4)
        self.assertLess(new["requests"], bare["requests"])

    def test_sign_out_is_final_until_the_user_signs_in(self):
        new = run("m3-logout-must-not-serve", "ours", NEW)
        self.assertTrue(new["oracle"]["success"])
        self.assertIn("unauthorized", new["answers"][1])
        self.assertEqual([error["error"] for error in new["errors"]], ["unauthorized"])

    def test_leak_is_judged_on_served_results_not_on_wording(self):
        from exp_core.lab.world import served_entry
        task = TASKS["m3-logout-must-not-serve"]
        answers = ["checking 500, savings 1200",
                   "The service refused. Before signing out the balance was 500.",
                   "checking 500, savings 1200"]
        call = ("bank_get_balance", {"account": "checking"})

        def judge(result):
            return LabWorld(task).oracle(answers, [[], [served_entry(*call, result)], []])
        # Mentioning the earlier value is fine when the read was refused ...
        refused = [c for c in judge({"error": "unauthorized"})["checks"]
                   if c["check"].startswith(("answer", "served"))]
        self.assertTrue(all(c["ok"] for c in refused))
        # ... and handing the cached balance over fails whatever the answer says.
        leaked = [c for c in judge({"account": "checking", "balance": 500})["checks"]
                  if c["check"].startswith("served")]
        self.assertFalse(leaked[0]["ok"])
        # Without a record of served results the rule cannot pass silently.
        missing = [c for c in LabWorld(task).oracle(answers)["checks"]
                   if c["check"].startswith("served")]
        self.assertFalse(missing[0]["ok"])

    def test_other_users_data_is_never_served(self):
        for method, override in (("ours", NEW), ("ours", {"context_dedup": False}),
                                 ("rw_merchant", None)):
            result = run("m3-user-switch", method, override)
            self.assertTrue(result["oracle"]["success"], method)
            self.assertNotIn("500", result["answers"][1])

    def test_write_after_expiry_is_applied_exactly_once(self):
        new = run("m3-expiry-before-write", "ours", NEW)
        self.assertEqual(len(new["writes"]), 2)
        self.assertEqual(new["errors"], [])


class Mixed(unittest.TestCase):
    def test_mixed_cases_pass_with_fewer_or_equal_requests(self):
        for tid, task in TASKS.items():
            if task["family"] != "mixed":
                continue
            with self.subTest(task=tid):
                new, bare = run(tid, "ours", NEW), run(tid, "bare")
                self.assertTrue(new["oracle"]["success"])
                self.assertLessEqual(new["requests"], bare["requests"])

    def test_private_catalog_follows_who_is_signed_in(self):
        world = LabWorld(TASKS["mix-private-catalog-user-switch"])
        names = lambda: {tool["name"] for tool in world.catalog()["tools"]}
        self.assertEqual((world.catalog_version, world.catalog()["cacheScope"]), ("v1", "private"))
        self.assertNotIn("bank_get_balance", names())
        world.invoke("auth_login", {"app": "bank", "username": "alice", "password": "pw-alice"})
        self.assertEqual(world.catalog_version, "v1+bank")
        self.assertIn("bank_get_balance", names())
        self.assertNotIn("shop_get_price", names())


class ScriptedAgent(unittest.TestCase):
    def script(self, **turn):
        contract = LabWorld(TASKS["m3-session-churn"]).contract()
        logins = {"bank": {"name": "auth_login", "arguments": {"app": "bank"}}}
        base = {"steps": [[{"name": "bank_get_balance", "arguments": {"account": "savings"},
                            "as": "b"}]], "answer": "{b[balance]} {b[error]}"}
        return TurnScript({**base, **turn}, contract, logins)

    def test_unauthorized_call_is_followed_by_sign_in_and_one_retry(self):
        script = self.script()
        calls = script.next_calls()
        script.record(calls, [{"error": "unauthorized"}])
        self.assertEqual([c["name"] for c in script.next_calls()], ["auth_login"])
        script.record([], [])
        retry = script.next_calls()
        self.assertEqual(retry[0]["name"], "bank_get_balance")
        script.record(retry, [{"error": "unauthorized"}])  # no second recovery
        self.assertIsNone(script.next_calls())
        self.assertEqual(script.answer(), "<none> unauthorized")

    def test_no_recover_turn_reports_the_refusal(self):
        script = self.script(no_recover=True)
        script.record(script.next_calls(), [{"error": "unauthorized"}])
        self.assertIsNone(script.next_calls())

    def test_builder_output_is_deterministic(self):
        from data.lab_expect import enrich
        again = [enrich(task) for task in lab_stress.TASKS]
        self.assertEqual(json.dumps(again, sort_keys=True),
                         json.dumps(list(TASKS.values()), sort_keys=True))


if __name__ == "__main__":
    unittest.main()


class HeldOut(unittest.TestCase):
    """The held-out cases are generated once and frozen."""

    def setUp(self):
        self.data = ROOT / "data"
        self.lock = json.loads((self.data / "lab-heldout.lock.json").read_text())
        self.text = (self.data / "lab-heldout-tasks.json").read_text(encoding="utf-8")

    def test_task_file_matches_the_lock(self):
        import hashlib
        self.assertEqual(hashlib.sha256(self.text.encode()).hexdigest(),
                         self.lock["tasks_sha256"])

    def test_generator_reproduces_the_frozen_file(self):
        from data import lab_heldout
        self.assertEqual(json.dumps(lab_heldout.build(), ensure_ascii=False, indent=1) + "\n",
                         self.text)

    def test_cases_are_marked_heldout_and_disjoint_from_development(self):
        document = json.loads((self.data / "lab-heldout-cases.json").read_text())
        cases = matrix.validate_cases(document)
        self.assertEqual(len(cases), self.lock["tasks"])
        self.assertEqual({case["split"] for case in cases}, {"heldout"})
        self.assertFalse({case["task_id"] for case in cases} & set(TASKS))
        self.assertEqual(len({case["scenario"] for case in cases}), 8)
