"""Public benchmarks rebuilt on the lab server layer."""
import copy
import json
import unittest

from data.lab_expect import BARE
from data.recon import bfcl as recon_bfcl
from data.recon.overlays import _retry, probe_turn
from exp_core import runner
from exp_core.lab.domains import DomainWorld, make_world
from exp_core.lab.harness import load_tasks, run_gold
from exp_core.lab.world import LabWorld
from exp_core.settings import APPWORLD_ROOT, DATA

ENTRIES = json.loads((DATA / "bfcl-tasks.json").read_text())
ORACLE = json.loads((DATA / "bfcl-oracle.json").read_text())
OURS = runner.resolve_policy("ours", {"scheduler": "default"})


def base(index=0, **extra):
    entry = ENTRIES[index]
    task = recon_bfcl.base_task(entry, ORACLE[entry["id"]]["ground_truth"])
    return {**task, "family": "base", "events": [], "faults": [], "expected": {}, **extra}


def sign_in(world, apps):
    tokens = {}
    for app in apps:
        reply = world.invoke("auth_login", {"app": app, "username": "user",
                                            "password": "pw-user"}, credentials=tokens)
        tokens[app] = reply["value"]["session"]
    return tokens


class Worlds(unittest.TestCase):
    def test_a_task_without_a_domain_runs_in_the_built_in_world(self):
        task = next(iter(load_tasks().values()))
        world = make_world(task)
        self.assertIs(type(world), LabWorld)
        self.assertIn("bank_transfer", world.tools)

    def test_a_bfcl_task_exposes_the_original_tools_behind_the_server_layer(self):
        world = make_world(base())
        self.assertIsInstance(world, DomainWorld)
        self.assertEqual(world.login_apps, {"ticket", "travel"})
        for name in ("auth_login", "op_status", "book_flight", "create_ticket"):
            self.assertIn(name, world.tools)
        self.assertNotIn("bank_transfer", world.tools)
        contract = world.contract()
        self.assertTrue(contract["book_flight"]["idempotency_key"])
        self.assertEqual(contract["get_credit_card_balance"]["fresh_class"], "slow")

    def test_a_bfcl_tool_needs_a_sign_in_and_optional_arguments_stay_optional(self):
        world = make_world(base())
        refused = world.invoke("get_booking_history", {"access_token": "ABCD1234"},
                               credentials={})
        self.assertEqual(refused["error_code"], "unauthorized")
        tokens = sign_in(world, ["ticket", "travel"])
        made = world.invoke("create_ticket", {"title": "t"}, credentials=tokens)
        self.assertTrue(made["ok"], made)
        wrong = world.invoke("create_ticket", {"title": "t", "colour": "red"},
                             credentials=tokens)
        self.assertEqual(wrong["error_code"], "business_error")

    def test_a_refusal_of_the_backend_is_a_business_error(self):
        world = make_world(base())
        tokens = sign_in(world, ["ticket", "travel"])
        reply = world.invoke("get_ticket", {"ticket_id": 987654}, credentials=tokens)
        self.assertEqual(reply["error_code"], "business_error")

    def test_reads_of_the_reference_solutions_leave_the_backend_unchanged(self):
        for index in range(len(ENTRIES)):
            task = base(index)
            world = make_world(task)
            tokens = sign_in(world, sorted(world.login_apps))
            kinds = world.contract()
            for turn in task["gold"]:
                for step in turn["steps"]:
                    for call in step:
                        if kinds[call["name"]]["kind"] != "read" or \
                                kinds[call["name"]]["fresh_class"] == "strict":
                            world.invoke(call["name"], call["arguments"], credentials=tokens)
                            continue
                        before = world.final_state()
                        world.invoke(call["name"], call["arguments"], credentials=tokens)
                        self.assertEqual(world.final_state(), before,
                                         f"{task['id']}: {call['name']} changed state")

    def test_a_measurement_read_does_not_advance_the_backend_generator(self):
        task = base(next(i for i, e in enumerate(ENTRIES)
                         if "VehicleControlAPI" in e["involved_classes"]))
        first, second = make_world(task), make_world(task)
        tokens = sign_in(first, ["vehicle"])
        sign_in(second, ["vehicle"])
        for _ in range(3):
            first.invoke("get_current_speed", {}, shadow=True, credentials=tokens)
        self.assertEqual(first.invoke("get_current_speed", {}, credentials=tokens)["value"],
                         second.invoke("get_current_speed", {}, credentials=tokens)["value"])

    def test_an_external_change_moves_the_value_a_probe_reads(self):
        task = base(next(i for i, e in enumerate(ENTRIES)
                         if "TradingBot" in e["involved_classes"]))
        probe = next(p for p in recon_bfcl.probes(task) if p["fresh"] == "volatile")
        world = make_world({**task, "events": [{"at_turn": 1, **probe["event"]}]})
        tokens = sign_in(world, sorted(world.login_apps))
        before = world.invoke(**probe["call"], credentials=tokens)["value"]["price"]
        world.advance(1, 1, credentials=tokens)
        after = world.invoke(**probe["call"], credentials=tokens)["value"]["price"]
        self.assertAlmostEqual(after - before, 7.5)


class Overlays(unittest.TestCase):
    def test_a_probe_turn_asks_for_every_value_and_names_it_in_the_answer(self):
        probes = [{"call": {"name": "a", "arguments": {}}, "field": ["x"], "label": "one",
                   "ask": "the first"},
                  {"call": {"name": "b", "arguments": {}}, "field": [], "label": "two",
                   "ask": "the second"}]
        text, gold = probe_turn(probes)
        self.assertEqual(text, "Tell me: the first, the second.")
        self.assertEqual(gold["answer"], "one: {p0[x]}; two: {p1}")
        self.assertEqual([c["as"] for c in gold["steps"][0]], ["p0", "p1"])

    def test_a_keyed_write_is_resent_and_an_unkeyed_one_is_checked_first(self):
        gold = [{"steps": [[{"name": "login", "arguments": {}}],
                           [{"name": "pay", "arguments": {"n": 1}}]], "answer": "done"}]
        keyed = _retry(gold, 0, 1)[0]["steps"]
        self.assertEqual([s[0]["name"] for s in keyed], ["login", "pay", "pay"])
        self.assertEqual(keyed[2][0]["if_error"], "w")
        look = {"name": "look", "arguments": {}}
        unkeyed = _retry(gold, 0, 1, verify=look)[0]["steps"]
        self.assertEqual([s[0]["name"] for s in unkeyed],
                         ["login", "look", "pay", "look", "pay"])
        self.assertEqual(unkeyed[4][0]["if"], "chk == pre")
        self.assertEqual(gold[0]["steps"][1][0], {"name": "pay", "arguments": {"n": 1}})


class RebuiltBfcl(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = DATA / "recon-bfcl-tasks.json"
        if not path.is_file():
            raise unittest.SkipTest("run ./run recon-build bfcl first")
        cls.tasks = json.loads(path.read_text())
        cls.cases = json.loads((DATA / "recon-bfcl-cases.json").read_text())["cases"]

    def test_every_case_names_a_task_and_keeps_the_original_split(self):
        manifest = json.loads((DATA / "manifest.json").read_text())["datasets"]["bfcl"]
        ids = {task["id"] for task in self.tasks}
        for case in self.cases:
            self.assertIn(case["task_id"], ids)
            self.assertIn(case["source_task_id"], manifest[case["split"]])

    def test_the_reference_solution_passes_its_own_oracle_without_a_cache(self):
        for task in self.tasks:
            run = run_gold({**copy.deepcopy(task), "faults": []}, BARE)
            self.assertTrue(run["oracle"]["success"], task["id"])

    def test_the_scheduler_never_applies_a_write_twice(self):
        for task in self.tasks:
            if task["family"].startswith("lost_"):
                run = run_gold(task, OURS, seed=1)
                self.assertTrue(run["oracle"]["safe"], task["id"])
                self.assertTrue(run["oracle"]["success"], task["id"])

    def test_a_changed_volatile_value_is_never_served_from_the_cache(self):
        for task in self.tasks:
            if task["family"] == "external_volatile":
                run = run_gold(task, OURS, seed=1)
                self.assertTrue(run["oracle"]["success"], task["id"])
                self.assertEqual(run["delivery"]["stale"], 0, task["id"])


@unittest.skipUnless((APPWORLD_ROOT / "data" / "api_docs" / "standard").is_dir(),
                     "AppWorld data is not installed")
class AppWorldRoutes(unittest.TestCase):
    def test_a_recorded_call_maps_to_its_api_and_path_arguments(self):
        from exp_core.lab.domains.appworld import route
        self.assertEqual(route("get", "/spotify/songs/12"), ("show_song", {"song_id": 12}))
        self.assertEqual(route("post", "/venmo/auth/token"), ("login", {}))
        self.assertEqual(route("post", "/venmo/transactions/7/comments"),
                         ("create_transaction_comment", {"transaction_id": 7}))
        self.assertEqual(route("get", "/spotify/songs"), ("search_songs", {}))

    def test_an_api_without_a_token_is_public(self):
        from exp_core.lab.domains.appworld import api_docs, spec_of
        docs = api_docs("spotify")
        self.assertFalse(spec_of("spotify", docs["show_song"])[1]["auth"])
        self.assertTrue(spec_of("spotify", docs["show_volume"])[1]["auth"])
        self.assertNotIn("access_token", spec_of("spotify", docs["show_volume"])[1]["parameters"])


class OverlayJudge(unittest.TestCase):
    """A real model's run is judged on the added part, against its own state."""
    TASK = {"overlay": {"probes": [{"turn": 1, "call": {}, "field": [], "label": "balance"}]},
            "expected": {"write_log": [{"tool": "pay", "arguments": {"to": "a", "amount": 5}}]},
            "faults": [{"tool": "pay", "kind": "lost_ack", "on_calls": [1]}]}
    PAY = {"tool": "pay", "arguments": {"to": "a", "amount": 5}}
    HIT = [{"tool": "pay", "kind": "lost_ack", "call": 1, "turn": 0}]

    def judge(self, truth, attempts, hits, answers, task=None):
        from exp_core.lab.judge import judge_overlay
        return judge_overlay(task or self.TASK, truth, attempts, hits, {"pay": len(attempts)},
                             answers)

    def test_the_answer_is_compared_with_what_this_run_left_on_the_server(self):
        once = [{**self.PAY, "outcome": "committed_reply_lost"}, {**self.PAY, "outcome": "replayed"}]
        # The reference would say 95; this run paid another way and holds 80.5.
        good = self.judge({0: 80.50000000000001}, once, self.HIT, ["ok", "Your balance is $80.50."])
        self.assertEqual(good["verdict"], "pass")
        stale = self.judge({0: 80.5}, once, self.HIT, ["ok", "Your balance is 95."])
        self.assertEqual(stale["verdict"], "fail")
        self.assertTrue(stale["safe"])

    def test_a_lost_reply_followed_by_a_second_application_is_unsafe(self):
        twice = [{**self.PAY, "outcome": "committed_reply_lost"}, {**self.PAY, "outcome": "committed"}]
        verdict = self.judge({0: 1}, twice, self.HIT, ["ok", "balance 1"])
        self.assertEqual(verdict["verdict"], "fail")
        self.assertFalse(verdict["safe"])

    def test_a_write_done_differently_from_the_reference_is_not_a_failure(self):
        other = {"tool": "pay", "arguments": {"to": "a", "amount": 5, "note": "thanks"}}
        attempts = [{**other, "outcome": "committed_reply_lost"}, {**other, "outcome": "replayed"}]
        self.assertEqual(self.judge({0: 1}, attempts, self.HIT, ["ok", "balance 1"])["verdict"],
                         "pass")

    def test_a_display_string_may_be_restated_without_its_symbols(self):
        from exp_core.lab.judge import states
        bar = "Status Bar: 📶⁴ Excellent | 5G | 📱 Data Enabled | 🔋 80%"
        self.assertTrue(states("The status bar shows excellent signal on 5G, data enabled, "
                               "battery at 80%.", bar))
        self.assertFalse(states("The status bar shows no signal, data enabled, 80%.", bar))
        self.assertFalse(states("battery 180%, Excellent, 5G, data enabled, status bar", bar))

    def test_a_fault_the_run_never_reached_tests_nothing(self):
        self.assertEqual(self.judge({0: 1}, [], [], ["ok", "balance 1"])["verdict"], "untested")

    def test_a_faulted_call_the_service_refused_tests_nothing(self):
        refused = [{**self.PAY, "outcome": "refused:business_error"}]
        self.assertEqual(self.judge({0: 1}, refused, self.HIT, ["ok", "balance 1"])["verdict"],
                         "untested")

    def test_the_server_notes_a_probe_value_before_the_next_turn_changes_it(self):
        task = load_tasks()["recon-bfcl-133-external_volatile"]
        run = run_gold(task, BARE)
        overlay = run["oracle"]["overlay"]
        self.assertEqual(overlay["verdict"], "pass")
        first, second = [c["want"] for c in overlay["checks"] if c["check"].startswith("answer")]
        self.assertNotEqual(first, second)  # someone else moved the price in between
        old = run_gold(task, runner.resolve_policy("ours", {"context_dedup": False}))
        self.assertEqual(old["oracle"]["overlay"]["verdict"], "fail")


class LongCases(unittest.TestCase):
    """One agent with a compacted history, and two agents on one host."""

    def test_both_cases_pass_with_and_without_the_cache(self):
        tasks = load_tasks()
        for task_id in ("lab-long-compacted", "lab-two-agents"):
            bare = run_gold(tasks[task_id], BARE)
            ours = run_gold(tasks[task_id], OURS)
            for run in (bare, ours):
                self.assertTrue(run["oracle"]["success"], task_id)
                self.assertEqual(run["oracle"]["overlay"]["verdict"], "pass", task_id)
            self.assertLess(ours["requests"], 0.6 * bare["requests"], task_id)

    def test_the_task_says_how_the_conversation_is_kept(self):
        tasks = load_tasks()
        self.assertEqual(tasks["lab-long-compacted"]["context"], "answers_only")
        self.assertEqual(len(tasks["lab-two-agents"]["agents"]),
                         len(tasks["lab-two-agents"]["turns"]))


class PublishWhatReplacesACall(unittest.TestCase):
    def runtime(self, reads):
        class Fake:
            supports_meta = True

            def iter_valid_reads(self):
                return iter(reads)
        return Fake()

    def identity(self, account):
        return json.dumps(["s", "bank_get_balance", {"account": account}], sort_keys=True)

    def test_a_read_made_in_this_turn_is_not_published_again(self):
        from ours.publication import collect_replacements
        now, earlier, unseen = (self.identity(a) for a in ("checking", "savings", "travel"))
        reads = [(now, {"balance": 1}), (earlier, {"balance": 2}), (unseen, {"balance": 3}),
                 (json.dumps(["other", "bank_get_balance", {"account": "x"}]), {"balance": 9})]
        items, used, identities = collect_replacements(
            self.runtime(reads), "s", 8192, shown_this_turn={now}, visible={now, earlier})
        self.assertEqual(identities, [earlier, unseen])
        # Still in the conversation: a pointer. Not in the conversation: the values.
        self.assertEqual(items[0], {"call": "bank_get_balance",
                                    "arguments": {"account": "savings"}, "shown_earlier": True})
        self.assertEqual(items[1]["values"], {"balance": 3})
        self.assertGreater(used, 0)

    def test_the_budget_and_the_item_limit_hold(self):
        from ours.publication import collect_replacements
        reads = [(self.identity(f"a{i}"), {"balance": i}) for i in range(30)]
        items, used, _ = collect_replacements(self.runtime(reads), "s", 8192, set(), set(),
                                              limit=5)
        self.assertEqual(len(items), 5)
        items, used, _ = collect_replacements(self.runtime(reads), "s", 300, set(), set())
        self.assertLessEqual(used, 300)


class ReadAhead(unittest.TestCase):
    def history(self):
        def read(tool, key, name, asked, fetched=-9):
            identity = json.dumps(["s", tool, {key: name}], sort_keys=True)
            return identity, {"name": tool, "arguments": {key: name}, "asked": asked,
                              "fetched": fetched}
        return dict([read("shop_get_price", "sku", "sku-lamp", 2),
                     read("shop_get_price", "sku", "sku-desk", 2),
                     read("bank_get_balance", "account", "checking", 6),
                     read("bank_get_balance", "account", "travel", 5, fetched=6),
                     read("bank_get_balance", "account", "bills", 3, fetched=6)])

    def names(self, text, turn=7, valid=(), limit=6):
        from ours.publication import choose_prefetch
        history = self.history()
        chosen = choose_prefetch(history, text, turn, lambda key: history[key]["arguments"]
                                 in [{"account": v} for v in valid], limit)
        return [next(iter(history[key]["arguments"].values())) for key in chosen]

    def test_a_read_the_message_names_or_the_agent_is_following_is_made_ahead(self):
        # Named in the question first; then the one asked last turn, and the one
        # fetched ahead last turn after being asked the turn before. Not the one
        # the agent stopped asking about, and not a sku the message only contains.
        self.assertEqual(self.names("What does sku-lamp cost now? (not sku-desk-2)"),
                         ["sku-lamp", "checking", "travel"])

    def test_a_read_the_host_still_holds_is_not_repeated_and_the_limit_holds(self):
        self.assertEqual(self.names("sku-lamp?", valid=("checking",)), ["sku-lamp", "travel"])
        self.assertEqual(self.names("sku-lamp?", limit=1), ["sku-lamp"])

    def test_a_pointer_is_published_only_for_the_version_the_conversation_shows(self):
        from ours.publication import collect_replacements, version_of
        identity = json.dumps(["s", "bank_get_balance", {"account": "checking"}], sort_keys=True)

        class Fake:
            supports_meta = True

            def iter_valid_reads(self):
                return iter([(identity, {"balance": 400})])
        same, _, _ = collect_replacements(Fake(), "s", 8192, set(),
                                          {identity: version_of({"balance": 400})})
        old, _, _ = collect_replacements(Fake(), "s", 8192, set(),
                                         {identity: version_of({"balance": 500})})
        self.assertTrue(same[0]["shown_earlier"])
        self.assertEqual(old[0]["values"], {"balance": 400})


class ForeignAddresses(unittest.TestCase):
    def test_an_address_the_prompt_never_contained_is_masked(self):
        from exp_core.model import mask_foreign_emails
        value, count = mask_foreign_emails(
            {"answer": "sent to bob@shop.example and me@host.example",
             "calls": [{"arguments": {"to": "ME@host.example"}}]}, "write to bob@shop.example")
        self.assertEqual(count, 2)
        self.assertEqual(value["answer"], "sent to bob@shop.example and [address-not-in-task]")
        self.assertEqual(value["calls"][0]["arguments"]["to"], "[address-not-in-task]")


if __name__ == "__main__":
    unittest.main()


class GainCases(unittest.TestCase):
    """The cases of data/lab_gain.py: how many turns need no call at all is a
    property of the task, and the control has none."""

    WANT = {"lab-gain-revisit": 5, "lab-gain-chain": 3, "lab-gain-handoff": 4,
            "lab-gain-control": 0}

    def test_turns_that_need_no_call(self):
        from exp_core import runner
        from exp_core.lab import harness
        tasks = harness.load_tasks()
        for task_id, want in self.WANT.items():
            reached = []
            invoke = harness.Backend.invoke

            def counted(backend, *args, _invoke=invoke, **kwargs):
                reached[-1] += 1
                return _invoke(backend, *args, **kwargs)

            answer = harness.TurnScript.answer

            def next_turn(script, *args, _answer=answer, **kwargs):
                reached.append(0)
                return _answer(script, *args, **kwargs)

            for policy, expected in ((runner.resolve_policy("bare", None), 0),
                                     (runner.resolve_policy("ours", {"scheduler": "default"}),
                                      want)):
                reached[:] = [0]
                harness.Backend.invoke, harness.TurnScript.answer = counted, next_turn
                try:
                    run = harness.run_gold(tasks[task_id], policy, seed=0)
                finally:
                    harness.Backend.invoke, harness.TurnScript.answer = invoke, answer
                self.assertEqual(run["oracle"]["verdict"], "pass", task_id)
                self.assertEqual(sum(count == 0 for count in reached[:-1]), expected, task_id)
