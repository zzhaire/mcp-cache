"""Mechanism checks for the M1–M4 scheduler (no model, no network)."""
import copy
import json
import math
import random
import unittest
from unittest import mock

from exp_core.access import Access
from exp_core.runtime import AccessFailure, Runtime
from exp_core.scheduler import Scheduler
from exp_core.scheduler_policy import deep_merge, load_presets, resolve
from ours.publication import collect_operations, collect_references


def policy(base="default", **overrides):
    # Sampled verification is random; keep it off unless a test asks for it.
    merged = deep_merge({"ttl": {"adaptive": {"enabled": False}}}, overrides)
    return resolve({"base": base, "overrides": merged})


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def counter(log, value=None):
    def fetch():
        log.append(1)
        return {"n": len(log)} if value is None else copy.deepcopy(value)
    return fetch


def meta(app="a", tool=None, principal=None, fresh="slow", reads=(), affects=()):
    return {"server": "srv", "app": app, "tool": tool, "principal": principal,
            "fresh_class": fresh, "reads": list(reads), "affects": list(affects)}


class PolicyLoading(unittest.TestCase):
    def test_every_preset_is_valid(self):
        for name in load_presets():
            self.assertEqual(resolve(name)["preset"], name)

    def test_locked_values_cannot_be_overridden(self):
        with self.assertRaises(ValueError):
            policy(locked={"retry_max_attempts": 3})
        with self.assertRaises(ValueError):
            policy(tiers={"M2": "session"})

    def test_loader_rules(self):
        with self.assertRaises(ValueError):
            policy(capacity={"share": {"M1": 0.5, "M2": 0.5, "M3": 0.5, "M4": 0}})
        with self.assertRaises(ValueError):
            policy(recovery={"allow_revalidate": True})
        with self.assertRaises(ValueError):
            resolve({"base": "missing"})
        with self.assertRaises(ValueError):
            resolve("no-such-preset-or-file")
        self.assertIsNone(resolve(None))

    def test_full_policy_round_trips(self):
        full = resolve("write_heavy")
        self.assertEqual(resolve(copy.deepcopy(full)), full)


class AblationParity(unittest.TestCase):
    """ablation_current must reproduce the legacy Runtime step by step."""

    def run_trace(self, runtime, clock, ops):
        out = []
        for op in ops:
            kind = op[0]
            if kind == "tick":
                clock.now += op[1]
                continue
            if kind == "auth":
                runtime.invalidate_authorization()
                continue
            if kind == "catalog":
                log = []
                value = runtime.catalog(op[1], "v1", counter(log, {"tools": op[1]}))
                out.append(("catalog", value, len(log)))
                continue
            if kind == "cred_save":
                runtime.save_credential("svc", "alice", ["account"], "tok", clock.now + op[1])
                continue
            if kind == "cred":
                out.append(("cred", runtime.credential("svc", "alice", ["account"])))
                continue
            _, name, readonly, op_id, behaviour = op
            log = []

            def fetch():
                log.append(1)
                if behaviour == "business":
                    raise AccessFailure("business_error", outcome="not_executed")
                if behaviour == "unknown":
                    raise AccessFailure("timeout", outcome="unknown")
                if behaviour == "flaky" and len(log) == 1:
                    raise AccessFailure("rate_limited", retryable=True,
                                        outcome="not_executed")
                if behaviour == "down":
                    raise AccessFailure("rate_limited", retryable=True,
                                        outcome="not_executed", retry_after=1)
                return {"name": name, "clock": clock.now}
            try:
                value = runtime.execute(name, {"x": name}, fetch, readonly=readonly,
                                        operation_id=op_id, scope="s")
                out.append(("ok", value, len(log)))
            except (AccessFailure, ValueError) as error:
                out.append(("err", str(error), len(log)))
        keys = ("M1_hits", "M2_hits", "M3_hits", "M4_hits", "M4_failure_hits",
                "backend_calls", "retries", "logical_calls")
        return out, {key: runtime.stats[key] for key in keys}

    def random_ops(self, seed):
        rng = random.Random(seed)
        ops, op_ids = [], []
        for _ in range(120):
            roll = rng.random()
            if roll < 0.08:
                ops.append(("tick", rng.choice([1, 30, 130])))
            elif roll < 0.12:
                ops.append(("auth",))
            elif roll < 0.18:
                ops.append(("catalog", rng.choice(["c1", "c2"])))
            elif roll < 0.21:
                ops.append(("cred_save", rng.choice([5, 500])))
            elif roll < 0.25:
                ops.append(("cred",))
            else:
                readonly = rng.random() < 0.6
                if op_ids and rng.random() < 0.3:
                    op_id = rng.choice(op_ids)
                    name, readonly = op_id.split("#")[0], op_id.endswith("r")
                else:
                    name = rng.choice(["r1", "r2", "r3", "w1", "w2"])
                    op_id = f"{name}#{len(op_ids)}{'r' if readonly else 'w'}"
                    op_ids.append(op_id)
                behaviour = rng.choice(["ok"] * 6 + ["business", "unknown", "flaky", "down"])
                if readonly and behaviour == "unknown":
                    behaviour = "ok"
                ops.append(("call", name, readonly, op_id if rng.random() < 0.8 else None,
                            behaviour))
        return ops

    def test_random_traces_match_legacy_runtime(self):
        for seed in range(40):
            ops = self.random_ops(seed)
            old_clock, new_clock = Clock(), Clock()
            legacy = Runtime(clock=old_clock, ttl=120)
            old_sleeps, new_sleeps = [], []
            scheduler = Scheduler(resolve("ablation_current"), clock=new_clock,
                                  sleep=new_sleeps.append)
            with self.subTest(seed=seed):
                with mock.patch("exp_core.runtime.time.sleep", old_sleeps.append):
                    expected = self.run_trace(legacy, old_clock, ops)
                self.assertEqual(expected, self.run_trace(scheduler, new_clock, ops))
                self.assertEqual(old_sleeps, new_sleeps)


class Invalidation(unittest.TestCase):
    def read(self, runtime, name, log, **kw):
        return runtime.call(name, {}, counter(log), readonly=True, scope="s", meta=meta(**kw))

    def write(self, runtime, **kw):
        runtime.call("w", {}, lambda: "ok", readonly=False, scope="s", meta=meta(**kw))

    def test_app_granularity_keeps_other_apps(self):
        runtime = Scheduler(policy(invalidation={"granularity": "app"}))
        a, b = [], []
        self.read(runtime, "ra", a, app="A")
        self.read(runtime, "rb", b, app="B")
        self.write(runtime, app="A")
        self.read(runtime, "ra", a, app="A")
        self.read(runtime, "rb", b, app="B")
        self.assertEqual((len(a), len(b)), (2, 1))

    def test_global_granularity_drops_everything(self):
        runtime = Scheduler(policy(invalidation={"granularity": "global"}))
        b = []
        self.read(runtime, "rb", b, app="B")
        self.write(runtime, app="A")
        self.read(runtime, "rb", b, app="B")
        self.assertEqual(len(b), 2)

    def test_resource_granularity_uses_declared_effects(self):
        runtime = Scheduler(policy())
        one, two = [], []
        self.read(runtime, "p1", one, reads=["pl/1"])
        self.read(runtime, "p2", two, reads=["pl/2"])
        self.write(runtime, affects=["pl/1"])
        self.read(runtime, "p1", one, reads=["pl/1"])
        self.read(runtime, "p2", two, reads=["pl/2"])
        self.assertEqual((len(one), len(two)), (2, 1))

    def test_undeclared_write_falls_back_to_app(self):
        runtime = Scheduler(policy())
        log = []
        self.read(runtime, "p2", log, reads=["pl/2"])
        self.write(runtime)  # resource granularity, no affects declared
        self.read(runtime, "p2", log, reads=["pl/2"])
        self.assertEqual(len(log), 2)

    def test_unknown_write_quarantines_app(self):
        runtime = Scheduler(policy())
        log = []
        self.read(runtime, "p2", log, reads=["pl/2"])

        def lost():
            raise AccessFailure("timeout", outcome="unknown")
        with self.assertRaises(AccessFailure):
            runtime.call("w", {}, lost, scope="s", operation_id="op",
                         meta=meta(affects=["pl/1"]))
        self.read(runtime, "p2", log, reads=["pl/2"])
        self.assertEqual(len(log), 2)


class Freshness(unittest.TestCase):
    def test_strict_is_never_cached(self):
        runtime = Scheduler(policy())
        log = []
        for _ in range(2):
            runtime.call("now", {}, counter(log), readonly=True, meta=meta(fresh="strict"))
        self.assertEqual(len(log), 2)
        self.assertEqual(runtime.stats["M2_strict_bypass"], 2)

    def test_overrides_can_only_tighten(self):
        # The contract says volatile; here volatile reads are kept for a few steps.
        loose = Scheduler(policy(ttl={"volatile": 4, "fresh_class_overrides": {"bal": "static"}}))
        tight = Scheduler(policy(ttl={"volatile": 4, "fresh_class_overrides": {"bal": "strict"}}))
        for runtime, expected in ((loose, 1), (tight, 2)):
            log = []
            for _ in range(2):
                runtime.call("bal", {}, counter(log), readonly=True,
                             meta=meta(tool="bal", fresh="volatile"))
            self.assertEqual(len(log), expected)

    def test_step_ttl_and_wall_clock_cap(self):
        clock = Clock()
        runtime = Scheduler(policy(ttl={"slow": 2, "wall_clock_cap_s": 50}), clock=clock)
        log = []
        read = lambda: runtime.call("r", {}, counter(log), readonly=True, meta=meta())
        read()
        runtime.advance_step()
        read()
        self.assertEqual(len(log), 1)
        runtime.advance_step()
        read()
        self.assertEqual(len(log), 2)
        clock.now = 51
        read()
        self.assertEqual(len(log), 3)


class CapacityAndForgetting(unittest.TestCase):
    def test_entry_limit_evicts_lru(self):
        runtime = Scheduler(policy(eviction={"policy": "lru"},
                                   capacity={"max_entries": {"M1": 0, "M2": 2, "M3": 0, "M4": 0}}))
        logs = {name: [] for name in "abc"}
        read = lambda name: runtime.call(name, {}, counter(logs[name]), readonly=True,
                                         meta=meta())
        read("a"); read("b"); read("a"); read("c")
        read("a"); read("b")
        self.assertEqual({k: len(v) for k, v in logs.items()}, {"a": 1, "b": 2, "c": 1})
        self.assertEqual(runtime.stats["evictions"], 2)

    def test_gdsf_keeps_expensive_entry(self):
        clock = Clock()
        runtime = Scheduler(policy(capacity={"max_entries": {"M1": 0, "M2": 1, "M3": 0, "M4": 0}}),
                            clock=clock)

        def slow_fetch(log):
            def fetch():
                clock.now += 2  # 2000 ms refetch cost
                log.append(1)
                return {"v": 1}
            return fetch
        expensive, cheap = [], []
        runtime.call("e", {}, slow_fetch(expensive), readonly=True, meta=meta())
        runtime.call("c", {}, counter(cheap), readonly=True, meta=meta())
        runtime.call("e", {}, slow_fetch(expensive), readonly=True, meta=meta())
        self.assertEqual(len(expensive), 1)

    def test_oversized_entry_is_not_admitted(self):
        runtime = Scheduler(policy(capacity={"max_entry_bytes": 10}))
        log = []
        for _ in range(2):
            runtime.call("big", {}, counter(log, {"x": "y" * 50}), readonly=True, meta=meta())
        self.assertEqual(len(log), 2)
        self.assertEqual(runtime.stats["rejected_admissions"], 2)

    def test_byte_budget_never_evicts_pinned_uncertain_write(self):
        runtime = Scheduler(policy(capacity={"total_bytes": 400, "borrow": True}))

        def lost():
            raise AccessFailure("timeout", outcome="unknown")
        with self.assertRaises(AccessFailure):
            runtime.call("w", {"k": 1}, lost, operation_id="op-u", meta=meta())
        for index in range(20):
            runtime.call(f"r{index}", {}, counter([], {"pad": "z" * 40}), readonly=True,
                         meta=meta())
        self.assertEqual(runtime.operations["op-u"]["state"], "uncertain")
        self.assertLessEqual(runtime.usage()["M2"]["bytes"], 400)

    def test_idle_forgetting_and_turn_end(self):
        runtime = Scheduler(policy(forget={"idle_steps": 2, "on_turn_end": "drop_volatile"}))
        slow, volatile = [], []
        runtime.call("s", {}, counter(slow), readonly=True, meta=meta())
        runtime.call("v", {}, counter(volatile), readonly=True, meta=meta(fresh="volatile"))
        runtime.end_turn()
        runtime.call("s", {}, counter(slow), readonly=True, meta=meta())
        runtime.call("v", {}, counter(volatile), readonly=True, meta=meta(fresh="volatile"))
        self.assertEqual((len(slow), len(volatile)), (1, 2))
        runtime.advance_step(); runtime.advance_step()
        runtime.call("s", {}, counter(slow), readonly=True, meta=meta())
        self.assertEqual(len(slow), 2)


class Authorization(unittest.TestCase):
    def test_partitioned_revocation_keeps_other_principals(self):
        for partition, expected in ((True, 1), (False, 2)):
            runtime = Scheduler(policy(coupling={"partition_by_principal": partition}))
            a, b = [], []
            read = lambda name, log, p: runtime.call(name, {}, counter(log), readonly=True,
                                                    meta=meta(app=p, principal=p))
            read("ra", a, "A"); read("rb", b, "B")
            runtime.invalidate_authorization(principal="A")
            read("ra", a, "A"); read("rb", b, "B")
            self.assertEqual((len(a), len(b)), (2, expected))

    def test_public_catalog_survives_auth_change_only_when_allowed(self):
        for scoped, expected in ((False, 1), (True, 2)):
            runtime = Scheduler(policy(coupling={"catalog_auth_scoped": scoped}))
            log = []
            runtime.catalog("docs", "v1", counter(log), public=True)
            runtime.invalidate_authorization()
            runtime.catalog("docs", "v1", counter(log), public=True)
            self.assertEqual(len(log), expected)

    def test_credential_superset_and_refresh_ahead(self):
        clock = Clock()
        refreshed = []
        runtime = Scheduler(
            policy(base="auth_churn", coupling={"refresh_margin_s": 10}), clock=clock,
            refreshers={"A": lambda: refreshed.append(1) or ("new", clock.now + 100)})
        runtime.save_credential("svc", "u", ["read", "write"], "old", 20, principal="A")
        self.assertEqual(runtime.credential("svc", "u", ["read"], principal="A"), "old")
        clock.now = 15
        self.assertEqual(runtime.credential("svc", "u", ["read", "write"], principal="A"), "new")
        self.assertEqual(len(refreshed), 1)

    def test_login_invalidates_principal_reads_without_keep_rule(self):
        runtime = Scheduler(policy(coupling={"enabled_rules": []}))
        log = []
        read = lambda: runtime.call("r", {}, counter(log), readonly=True,
                                    meta=meta(principal="A"))
        read()
        runtime.save_credential("svc", "u", ["x"], "tok", 100, principal="A")
        read()
        self.assertEqual(len(log), 2)


class Recovery(unittest.TestCase):
    def failing(self, log):
        def fetch():
            log.append(1)
            raise AccessFailure("business_error", outcome="not_executed")
        return fetch

    def test_identity_failure_index_replays_across_operation_ids(self):
        for index, expected in (("identity", 1), ("operation_id", 2)):
            runtime = Scheduler(policy(recovery={"failure_index": index}))
            log = []
            for op_id in ("op1", "op2"):
                with self.assertRaises(AccessFailure):
                    runtime.call("w", {"a": 1}, self.failing(log), operation_id=op_id,
                                 scope="s", meta=meta())
            self.assertEqual(len(log), expected, index)

    def test_failure_replay_stops_after_state_change(self):
        runtime = Scheduler(policy(invalidation={"granularity": "app"}))
        log = []
        with self.assertRaises(AccessFailure):
            runtime.call("w", {"a": 1}, self.failing(log), operation_id="op1", meta=meta())
        runtime.call("other", {}, lambda: "ok", operation_id="op2", meta=meta())
        with self.assertRaises(AccessFailure):
            runtime.call("w", {"a": 1}, self.failing(log), operation_id="op3", meta=meta())
        self.assertEqual(len(log), 2)

    def test_reconcile_resolves_uncertain_write(self):
        verdicts = {"executed": ("executed", {"id": 7}), "not": ("not_executed", None)}
        for case, verdict in verdicts.items():
            log = []
            runtime = Scheduler(policy(), reconciler=lambda request, v=verdict: v)

            def lost():
                log.append(1)
                raise AccessFailure("timeout", outcome="unknown")
            with self.assertRaises(AccessFailure):
                runtime.call("pay", {"amt": 1}, lost, operation_id="op", meta=meta())
            result = runtime.call("pay", {"amt": 1}, lambda: log.append(1) or {"id": 8},
                                  operation_id="op", meta=meta())
            self.assertEqual(result, {"id": 7} if case == "executed" else {"id": 8})
            self.assertEqual(len(log), 1 if case == "executed" else 2)

    def test_uncertain_write_without_reconciler_is_blocked(self):
        runtime = Scheduler(policy(recovery={"unknown_write": "surface_only"}))

        def lost():
            raise AccessFailure("timeout", outcome="unknown")
        with self.assertRaises(AccessFailure):
            runtime.call("pay", {}, lost, operation_id="op", meta=meta())
        with self.assertRaisesRegex(AccessFailure, "write_outcome_unknown"):
            runtime.call("pay", {}, lambda: "ok", operation_id="op", meta=meta())

    def test_backoff_and_shared_retry_cap(self):
        sleeps, log = [], []
        runtime = Scheduler(policy(retry={"jitter": "none"}), sleep=sleeps.append)

        def flaky():
            log.append(1)
            if len(log) == 1:
                raise AccessFailure("rate_limited", retryable=True, outcome="not_executed")
            return "ok"
        self.assertEqual(runtime.execute("r", {}, flaky, max_attempts=5, meta=meta()), "ok")
        self.assertEqual(sleeps, [0.2])
        log.clear()

        def always():
            log.append(1)
            raise AccessFailure("rate_limited", retryable=True, outcome="not_executed")
        with self.assertRaises(AccessFailure):
            runtime.execute("r", {}, always, max_attempts=5, meta=meta())
        self.assertEqual(len(log), 2)  # locked retry_max_attempts

    def test_auth_refresh_then_retry_once(self):
        tokens = {"t": "old"}
        runtime = Scheduler(policy(), sleep=lambda s: None,
                            refreshers={"A": lambda: tokens.update(t="new") or ("new", 99)})

        def call():
            if tokens["t"] == "old":
                raise AccessFailure("token_expired", outcome="not_executed")
            return "ok"
        self.assertEqual(runtime.execute("r", {}, call, meta=meta(principal="A")), "ok")
        self.assertEqual(runtime.stats["auth_refreshes"], 1)


class CatalogChange(unittest.TestCase):
    def test_scoped_catalog_change_only_hits_changed_tools(self):
        for scoped, expected in ((True, (1, 2)), (False, (2, 2))):
            rules = ["catalog_change_scoped"] if scoped else []
            runtime = Scheduler(policy(coupling={"enabled_rules": rules}))
            v1 = {"tools": [{"name": "keep", "x": 1}, {"name": "change", "x": 1}]}
            v2 = {"tools": [{"name": "keep", "x": 1}, {"name": "change", "x": 2}]}
            runtime.catalog("srv", "v1", lambda: v1)
            keep, change = [], []
            read = lambda tool, log: runtime.call(tool, {}, counter(log), readonly=True,
                                                  meta=meta(tool=tool))
            read("keep", keep); read("change", change)
            runtime.catalog("srv", "v2", lambda: v2)
            read("keep", keep); read("change", change)
            self.assertEqual((len(keep), len(change)), expected)


class SharedTier(unittest.TestCase):
    def test_session_tier_shares_public_catalogs_only(self):
        shared = {}
        first = Scheduler(policy(), shared=shared)
        log = []
        first.catalog("docs", "v1", counter(log), public=True)
        first.catalog("private", "v1", counter(log), principal="A")
        second = Scheduler(policy(), shared=shared)
        second.catalog("docs", "v1", counter(log), public=True)
        second.catalog("private", "v1", counter(log), principal="A")
        self.assertEqual(len(log), 3)
        self.assertEqual(second.stats["M1_shared_hits"], 1)


class PublicationIntegration(unittest.TestCase):
    def test_references_and_operations_use_scheduler_validity(self):
        runtime = Scheduler(policy(invalidation={"granularity": "app"}))
        runtime.call("r", {"q": 1}, lambda: {"v": 1}, readonly=True, scope="s",
                     meta=meta(app="A"))
        identity = json.dumps(["s", "r", {"q": 1}], sort_keys=True)
        items, _, ids = collect_references(runtime, "s", 4096, {identity}, limit=3)
        self.assertEqual(ids, [identity])
        self.assertNotIn("result", items[0])
        items, _, _ = collect_references(runtime, "s", 4096, {identity}, limit=3,
                                         latest=[("result", identity)], mode="full")
        self.assertEqual(items[0]["result"], {"v": 1})

        def fail():
            raise AccessFailure("business_error", outcome="not_executed")
        with self.assertRaises(AccessFailure):
            runtime.call("w", {}, fail, operation_id="op", scope="s", meta=meta(app="A"))
        self.assertEqual(len(collect_operations(runtime)), 1)
        runtime.call("w2", {}, lambda: "ok", scope="s", meta=meta(app="A"))
        self.assertEqual(collect_operations(runtime), [])
        self.assertEqual(collect_references(runtime, "s", 4096, {identity})[0], [])


class AccessIntegration(unittest.TestCase):
    def test_appworld_logout_only_affects_that_app(self):
        runtime = Scheduler(policy())
        calls = []

        def invoke(name, arguments):
            calls.append((arguments["app"], arguments["api"]))
            return {"ok": True, "value": {"app": arguments["app"], "n": len(calls)}}
        access = Access(runtime, "appworld", "scope", invoke)
        show = lambda app: access.call("api_call", {"app": app, "api": "show_profile",
                                                    "arguments": {}}, f"op-{len(calls)}-{app}")
        show("spotify"); show("venmo")
        access.call("api_call", {"app": "spotify", "api": "logout", "arguments": {}}, "op-x")
        show("spotify"); show("venmo")
        self.assertEqual(calls.count(("spotify", "show_profile")), 2)
        self.assertEqual(calls.count(("venmo", "show_profile")), 1)

    def test_appworld_clock_read_is_strict(self):
        runtime = Scheduler(policy())
        calls = []
        access = Access(runtime, "appworld", "scope",
                        lambda n, a: calls.append(1) or {"ok": True, "value": len(calls)})
        for index in range(2):
            access.call("api_call", {"app": "phone", "api": "get_current_date_and_time",
                                     "arguments": {}}, f"op{index}")
        self.assertEqual(len(calls), 2)


class AdaptiveFreshness(unittest.TestCase):
    """A1/A2: sampled verification tunes the TTL; shadow checks only measure."""

    def adaptive(self, **kw):
        base = {"enabled": True, "verify_sample_rate": 1.0, "target_stale_rate": 0.1,
                "step": 1.0, "min_scale": 0.05, "max_scale": 8}
        base.update(kw)
        return base

    def test_verification_shrinks_ttl_when_stale_and_grows_when_fresh(self):
        runtime = Scheduler(policy(ttl={"adaptive": self.adaptive()}))
        state = {"v": 1}
        read = lambda tool: runtime.call(tool, {}, lambda: {"v": state["v"]}, readonly=True,
                                         meta=meta(tool=tool))
        read("stable"); read("stable")
        self.assertGreater(runtime.ttl_scale["stable"], 1.0)
        read("drifting")
        state["v"] = 2  # changed behind the cache's back
        self.assertEqual(read("drifting"), {"v": 2})  # verification returns the truth
        self.assertLess(runtime.ttl_scale["drifting"], 1.0)
        self.assertEqual(runtime.stats["M2_verifications"], 2)
        self.assertEqual(runtime.stats["M2_verify_stale"], 1)
        self.assertEqual(runtime.stats["M2_hits"], 0)       # verified hits save nothing
        self.assertEqual(runtime.stats["backend_calls"], 4)  # and are metered

    def test_learned_scale_shortens_the_next_expiry(self):
        runtime = Scheduler(policy(ttl={"slow": 10, "adaptive": self.adaptive(
            verify_sample_rate=0.0)}))
        runtime.ttl_scale["t"] = 0.2  # as if verification had found it stale
        log = []
        read = lambda: runtime.call("t", {}, counter(log), readonly=True, meta=meta(tool="t"))
        read()
        runtime.advance_step(); runtime.advance_step()
        read()
        self.assertEqual(len(log), 2)  # 10 steps * 0.2 = 2 steps

    def test_scale_converges_toward_target_stale_rate(self):
        runtime = Scheduler(policy(ttl={"adaptive": self.adaptive(
            target_stale_rate=0.25, step=0.2)}), seed=7)
        rng = random.Random(3)
        state = {"v": 0}
        for _ in range(400):
            if rng.random() < 0.25:  # true staleness equals the target
                state["v"] += 1
            runtime.call("t", {}, lambda: {"v": state["v"]}, readonly=True, meta=meta(tool="t"))
        self.assertLess(abs(math.log(runtime.ttl_scale["t"])), 2.5)  # no runaway

    def test_disabled_never_verifies(self):
        runtime = Scheduler(policy(ttl={"adaptive": {"enabled": False}}))
        log = []
        for _ in range(5):
            runtime.call("t", {}, counter(log), readonly=True, meta=meta())
        self.assertEqual((len(log), runtime.stats["M2_verifications"]), (1, 0))

    def test_shadow_check_measures_without_changing_behaviour(self):
        runtime = Scheduler(policy(ttl={"adaptive": {"enabled": False}}))
        truth = {"v": 1}
        m = meta()
        m["shadow_fetch"] = lambda: dict(truth)
        log = []
        read = lambda: runtime.call("t", {}, counter(log, {"v": 1}), readonly=True, meta=dict(m))
        read(); read()
        truth["v"] = 2
        self.assertEqual(read(), {"v": 1})  # still serves the cached value
        self.assertEqual(len(log), 1)
        self.assertEqual((runtime.stats["shadow_checks"], runtime.stats["shadow_stale"]), (2, 1))
        self.assertEqual(runtime.stats["M2_hits"], 2)


class TierOneRecovery(unittest.TestCase):
    def lost_then_ok(self, log):
        def fetch():
            log.append(1)
            if len(log) == 1:
                raise AccessFailure("timeout", outcome="unknown")
            return {"id": 1}
        return fetch

    def test_unknown_write_is_retried_only_with_an_idempotency_key(self):
        keyed, plain = [], []
        runtime = Scheduler(policy(), sleep=lambda s: None)
        m = meta(); m["idempotency_key"] = True
        self.assertEqual(runtime.execute("pay", {}, self.lost_then_ok(keyed),
                                         operation_id="k", meta=m), {"id": 1})
        self.assertEqual((len(keyed), runtime.stats["idempotent_retries"]), (2, 1))
        with self.assertRaises(AccessFailure):
            runtime.execute("pay", {}, self.lost_then_ok(plain), operation_id="p", meta=meta())
        self.assertEqual(len(plain), 1)
        with self.assertRaisesRegex(AccessFailure, "write_outcome_unknown"):
            runtime.execute("pay", {}, self.lost_then_ok(plain), operation_id="p", meta=meta())
        self.assertEqual(len(plain), 1)

    def test_idempotent_retry_can_be_switched_off(self):
        log = []
        runtime = Scheduler(policy(recovery={"unknown_write": "surface_only"}),
                            sleep=lambda s: None)
        m = meta(); m["idempotency_key"] = True
        with self.assertRaises(AccessFailure):
            runtime.execute("pay", {}, self.lost_then_ok(log), operation_id="k", meta=m)
        self.assertEqual(len(log), 1)

    def test_full_jitter_stays_within_backoff_and_honours_retry_after(self):
        sleeps = []
        runtime = Scheduler(policy(), sleep=sleeps.append, seed=1)
        for retry_after in (0, 3):
            log = []

            def flaky():
                log.append(1)
                if len(log) == 1:
                    raise AccessFailure("rate_limited", retryable=True,
                                        outcome="not_executed", retry_after=retry_after)
                return "ok"
            runtime.execute(f"r{retry_after}", {}, flaky, meta=meta())
        self.assertTrue(0 < sleeps[0] <= 0.2)
        self.assertEqual(sleeps[1], 3)

    def test_breaker_opens_after_transient_failures_and_recovers(self):
        log = []
        runtime = Scheduler(policy(recovery={"breaker": {"threshold": 2, "cooldown_steps": 3}}),
                            sleep=lambda s: None)
        state = {"down": True}

        def fetch():
            log.append(1)
            if state["down"]:
                raise AccessFailure("unavailable", retryable=True, outcome="not_executed")
            return "ok"
        call = lambda: runtime.execute("r", {}, fetch, readonly=True, meta=meta(tool="r"))
        for _ in range(2):
            with self.assertRaises(AccessFailure):
                call()
        self.assertEqual((len(log), runtime.stats["breaker_trips"]), (4, 1))
        with self.assertRaisesRegex(AccessFailure, "circuit_open"):
            call()
        self.assertEqual(len(log), 4)  # no backend call while open
        other = []
        runtime.execute("s", {}, counter(other), readonly=True, meta=meta(tool="s"))
        self.assertEqual(len(other), 1)  # breakers are per tool
        for _ in range(3):
            runtime.advance_step()
        state["down"] = False
        self.assertEqual(call(), "ok")
        self.assertEqual(runtime.breakers["srv/r"]["fails"], 0)

    def test_business_errors_do_not_trip_the_breaker(self):
        runtime = Scheduler(policy(recovery={"breaker": {"threshold": 1, "cooldown_steps": 5},
                                             "reuse_same_condition_failure": False}))
        log = []

        def fetch():
            log.append(1)
            raise AccessFailure("business_error", outcome="not_executed")
        for index in range(3):
            with self.assertRaisesRegex(AccessFailure, "business_error"):
                runtime.execute("w", {"i": index}, fetch, meta=meta(tool="w"))
        self.assertEqual((len(log), runtime.stats["breaker_trips"]), (3, 0))

    def test_described_failures_do_not_repeat_the_arguments(self):
        runtime = Scheduler(policy())
        runtime.advance_step()

        def fail():
            raise AccessFailure("business_error", outcome="not_executed")
        with self.assertRaises(AccessFailure):
            runtime.call("transfer", {"amount": 98765}, fail, operation_id="op", meta=meta())
        verbatim = collect_operations(runtime)
        described = collect_operations(runtime, failure_form="described")
        self.assertIn("98765", json.dumps(verbatim))
        text = json.dumps(described)
        self.assertNotIn("98765", text)
        self.assertIn("transfer", text)
        self.assertIn("decision step 1", text)
        self.assertEqual(set(described[0]), {"state", "description"})


class ProtocolCatalog(unittest.TestCase):
    """A6/A7: SEP-2549 freshness inputs, list_changed and content hashing."""

    def test_server_ttl_and_cache_scope_come_from_the_list_result(self):
        clock = Clock()
        runtime = Scheduler(policy(), clock=clock)
        log = []
        listing = {"tools": [{"name": "a"}], "ttlMs": 5000, "cacheScope": "public"}
        fetch = lambda: log.append(1) or dict(listing)
        runtime.catalog("srv", "v1", fetch)
        runtime.invalidate_authorization()  # public scope survives an auth change
        runtime.catalog("srv", "v1", fetch)
        self.assertEqual(len(log), 1)
        clock.now = 5.1
        runtime.catalog("srv", "v1", fetch)
        self.assertEqual(len(log), 2)

    def test_private_scope_and_default_ttl(self):
        clock = Clock()
        runtime = Scheduler(policy(ttl={"catalog_default": 60}), clock=clock)
        log = []
        fetch = lambda: log.append(1) or {"tools": [], "cacheScope": "private"}
        runtime.catalog("srv", "v1", fetch, public=True)
        runtime.invalidate_authorization()
        runtime.catalog("srv", "v1", fetch, public=True)
        self.assertEqual(len(log), 2)  # server said private: follows authorization
        clock.now = 61
        runtime.catalog("srv", "v1", fetch, public=True)
        self.assertEqual(len(log), 3)

    def test_silent_catalog_change_is_detected_by_content_hash(self):
        clock = Clock()
        runtime = Scheduler(policy(), clock=clock)
        tools = {"tools": [{"name": "t", "description": "safe"}], "ttlMs": 1000}
        runtime.catalog("srv", "v1", lambda: copy.deepcopy(tools))
        reads = []
        read = lambda: runtime.call("t", {}, counter(reads), readonly=True, meta=meta(tool="t"))
        read()
        tools["tools"][0]["description"] = "now exfiltrates data"  # same version label
        clock.now = 2
        runtime.catalog("srv", "v1", lambda: copy.deepcopy(tools))
        read()
        self.assertEqual(runtime.stats["catalog_silent_changes"], 1)
        self.assertEqual(len(reads), 2)

    def test_list_changed_forces_refetch_but_keeps_reads_if_nothing_changed(self):
        runtime = Scheduler(policy())
        log, reads = [], []
        fetch = lambda: log.append(1) or {"tools": [{"name": "t"}]}
        runtime.catalog("srv", "v1", fetch)
        runtime.call("t", {}, counter(reads), readonly=True, meta=meta(tool="t"))
        runtime.notify_list_changed("srv")
        runtime.catalog("srv", "v1", fetch)
        runtime.call("t", {}, counter(reads), readonly=True, meta=meta(tool="t"))
        self.assertEqual((len(log), len(reads)), (2, 1))
        self.assertEqual(runtime.stats["catalog_silent_changes"], 0)


class AccessTierOne(unittest.TestCase):
    def test_idempotency_key_and_shadow_reads_reach_the_adapter(self):
        from exp_core import access as access_module
        runtime = Scheduler(policy(ttl={"adaptive": {"enabled": False}}), sleep=lambda s: None)
        plain, keyed, shadow = [], [], []

        def invoke_keyed(name, arguments, key):
            keyed.append(key)
            if len(keyed) == 1:
                return {"ok": False, "error_code": "timeout", "outcome": "unknown"}
            return {"ok": True, "value": "paid"}
        contract = copy.deepcopy(access_module.SCHEDULER_CONTRACT)
        contract["idempotency_key"]["bfcl"] = ["pay"]
        access = Access(runtime, "bfcl", "scope",
                        lambda n, a: plain.append(n) or {"ok": True, "value": ["x"]},
                        invoke_keyed=invoke_keyed,
                        shadow_invoke=lambda n, a: shadow.append(n) or {"ok": True, "value": ["y"]})
        with mock.patch.object(access_module, "SCHEDULER_CONTRACT", contract):
            self.assertEqual(access.call("pay", {"amt": 1}, "op-1"), "paid")
            self.assertEqual(keyed, ["op-1", "op-1"])  # same key on the retry
            access.call("ls", {}, "op-2"); access.call("ls", {}, "op-3")
        self.assertEqual((plain, shadow), (["ls"], ["ls"]))
        self.assertEqual(runtime.stats["shadow_stale"], 1)


class RunnerWiring(unittest.TestCase):
    def test_policy_resolution_and_job_hash(self):
        from exp_core import matrix, runner
        resolved = runner.resolve_policy("ours", {"scheduler": "write_heavy"})
        self.assertEqual(resolved["scheduler"]["preset"], "write_heavy")
        self.assertEqual(resolved["publish_budget_bytes"],
                         resolved["scheduler"]["publish"]["budget_bytes"])
        case = {"case_id": "c", "split": "dev", "dataset": "bfcl", "task_id": "t",
                "scenario": "read", "augmented": False, "augmentation": None}
        digests = set()
        for preset in ("default", "write_heavy"):
            jobs = matrix.build_jobs([case], ["ours"], ours_policy={"scheduler": preset},
                                     code_sha="x" * 64)
            digests.add(jobs[0]["policy_sha256"])
        self.assertEqual(len(digests), 2)


if __name__ == "__main__":
    unittest.main()
