"""M1–M4 scheduler (docs/m1-m4-scheduler.md, scheduler-v0).

Drop-in replacement for ``exp_core.runtime.Runtime``: the same public calls
(``catalog``, ``save_credential``, ``credential``, ``invalidate_authorization``,
``execute``, ``call``, ``operation_audit``, ``stats``), plus an optional
``meta`` keyword carrying contract facts (server, app, principal, tool,
freshness class, resource tags). Without ``meta`` every call falls back to the
most conservative setting.

Validity is tracked with versioned tags instead of global epochs. Every entry
records the version of each tag it depends on; an invalidation bumps a tag, and
an entry is valid only while all of its tag versions are unchanged. Tags:

    "*"            every data read (global invalidation)
    "auth*"        every authorization-dependent entry (global auth change)
    "auth:<p>"     entries bound to principal p
    "srv:<s>"      reads served by server s
    "app:<a>"      reads of app a
    "tool:<s>/<t>" reads of tool t (catalog schema changes)
    "res:<r>"      declared resource r
    "cat:<s>"      catalog of server s

Safety rules (policy ``locked``) are enforced by construction here, not only by
the schema: strict reads are never cached, uncertain writes are never
replayed, and undeclared write effects fall back to at least app granularity.
"""
import copy
import hashlib
import json
import math
import random
import time
from collections import defaultdict

from exp_core.runtime import AccessFailure

MODULES = ("M1", "M2", "M3", "M4")
CLASS_ORDER = {"static": 0, "slow": 1, "volatile": 2, "strict": 3}
ALL, AUTH_ALL = "*", "auth*"
AUTH_CODES = frozenset({"auth_error", "unauthorized", "token_expired",
                        "invalid_token", "authorization_required"})
GRANULARITY_ORDER = {"resource": 0, "app": 1, "server": 2, "global": 3}


def _canonical(value):
    return json.dumps(value, default=str, sort_keys=True)


def _size(value):
    return len(_canonical(value).encode())


class Entry:
    __slots__ = ("module", "key", "value", "size", "tags", "fresh_class",
                 "expires_at", "expires_step", "expires_turn", "last_step",
                 "hits", "cost", "priority", "seq", "extra")

    def __init__(self, module, key, value, size, tags, fresh_class=None,
                 expiry=(None, None, None), cost=0.0, extra=None):
        self.module, self.key, self.value, self.size = module, key, value, size
        self.tags, self.fresh_class = tags, fresh_class
        self.expires_at, self.expires_step, self.expires_turn = expiry
        self.last_step = 0
        self.hits, self.cost, self.priority, self.seq = 0, cost, 0.0, 0
        self.extra = extra or {}


class Scheduler:
    supports_meta = True

    def __init__(self, policy, clock=time.monotonic, sleep=time.sleep, disabled=(),
                 m4_failure_only=False, refreshers=None, reconciler=None, shared=None,
                 seed=0):
        self.policy = policy
        self.clock, self.sleep = clock, sleep
        self.disabled = set(disabled)
        self.m4_failure_only = m4_failure_only
        self.rules = set(policy["coupling"]["enabled_rules"])
        self._authorized_until = {}  # principal -> when its sign-in runs out
        # principal -> callable() returning (token, valid_until) or None.
        self.refreshers = dict(refreshers or {})
        # callable(request) returning ("executed", value) / ("not_executed", None) / None.
        self.reconciler = reconciler
        # Session-tier store shared by schedulers of one Host session: {"M1": {}, "M3": {}}.
        self.shared = shared
        self.tag_versions = defaultdict(int)
        self.step = self.turn = 0
        # Kept for trace compatibility; validity uses tag versions.
        self.auth_epoch = self.data_epoch = 0
        self._entries = {module: {} for module in MODULES}
        self._bytes = dict.fromkeys(MODULES, 0)
        self._seq = 0
        self._aging = 0.0
        self.failure_index = {}
        self.last_catalog = {}
        # Servers whose catalog was announced as changed (list_changed) and not re-read yet.
        self.catalog_dirty = set()
        # Deterministic randomness for sampled verification and retry jitter.
        self.rng = random.Random(seed)
        # Per-tool TTL multiplier learned from sampled verification (ttl.adaptive).
        self.ttl_scale = {}
        # Per-tool circuit breaker: {"fails": consecutive transient failures, "open_until": step}.
        self.breakers = {}
        self.stats = {"logical_calls": 0, "M2_eligible": 0, "M2_hits": 0,
                      "M4_hits": 0, "backend_calls": 0, "M1_lookups": 0,
                      "M1_hits": 0, "M3_lookups": 0, "M3_hits": 0,
                      "M4_failure_hits": 0, "retries": 0,
                      "M1_shared_hits": 0, "M3_shared_hits": 0,
                      "M2_strict_bypass": 0, "evictions": 0, "rejected_admissions": 0,
                      "invalidation_events": 0, "invalidated_entries": 0,
                      "forgotten_entries": 0, "catalog_changes": 0,
                      "reconciliations": 0, "auth_refreshes": 0, "backoff_ms": 0.0,
                      "capacity_overflow": 0, "bytes_peak": 0,
                      "M2_verifications": 0, "M2_verify_stale": 0, "M2_verify_errors": 0,
                      "shadow_checks": 0, "shadow_stale": 0, "shadow_errors": 0,
                      "shadow_time_s": 0.0, "breaker_trips": 0,
                      "breaker_short_circuits": 0, "idempotent_retries": 0,
                      "catalog_silent_changes": 0}

    # ------------------------------------------------------------------ tags
    def _snapshot(self, tags):
        return {tag: self.tag_versions[tag] for tag in tags}

    def _tags_valid(self, snapshot):
        return all(self.tag_versions[tag] == version for tag, version in snapshot.items())

    def _bump(self, tags):
        self.stats["invalidation_events"] += 1
        for tag in tags:
            self.tag_versions[tag] += 1
        self._purge(("M1", "M2", "M3"))

    # ---------------------------------------------------------------- expiry
    def _expiry(self, ttl):
        """(expires_at, expires_step, expires_turn) for a ttl in the policy unit."""
        cfg = self.policy["ttl"]
        now = self.clock()
        if cfg["unit"] == "seconds":
            return (None if ttl is None else now + ttl, None, None)
        cap = now + cfg["wall_clock_cap_s"]
        if cfg["unit"] == "steps":
            return (cap, None if ttl is None else self.step + ttl, None)
        return (cap, None, None if ttl is None else self.turn + ttl)

    def _alive(self, entry):
        return ((entry.expires_at is None or self.clock() < entry.expires_at)
                and (entry.expires_step is None or self.step < entry.expires_step)
                and (entry.expires_turn is None or self.turn < entry.expires_turn))

    def _valid(self, entry):
        return self._alive(entry) and self._tags_valid(entry.tags)

    # ------------------------------------------------------------- the store
    def _pinned(self, entry):
        if entry.module == "M3":
            return self._valid(entry)
        if entry.module == "M4":
            record = entry.value
            return (record["state"] in {"running", "uncertain"}
                    or (not record["readonly"] and record["state"] == "completed"))
        return False

    def _priority(self, entry):
        weights = self.policy["eviction"]
        freq = (entry.hits + 1) ** weights["freq_exponent"]
        return self._aging + freq * entry.cost / max(entry.size, 1)

    def _touch(self, entry):
        entry.hits += 1
        entry.last_step = self.step
        self._seq += 1
        entry.seq = self._seq
        entry.priority = self._priority(entry)

    def _cost(self, latency_ms, size):
        weights = self.policy["eviction"]
        return (weights["w_lat"] * latency_ms + weights["w_bytes"] * size
                + weights["w_tok"] * size / 4)

    def _put(self, entry, latency_ms=0.0):
        capacity = self.policy["capacity"]
        pinned = self._pinned(entry)
        if (capacity["max_entry_bytes"] and entry.size > capacity["max_entry_bytes"]
                and not pinned):
            self.stats["rejected_admissions"] += 1
            return None
        if (entry.module == "M2"
                and latency_ms < self.policy["admission"]["min_refetch_cost_ms"]):
            self.stats["rejected_admissions"] += 1
            return None
        self._remove(entry.module, entry.key)
        entry.cost = self._cost(latency_ms, entry.size)
        entry.last_step = self.step
        self._seq += 1
        entry.seq = self._seq
        entry.priority = self._priority(entry)
        self._entries[entry.module][entry.key] = entry
        self._bytes[entry.module] += entry.size
        self._enforce(entry.module)
        self.stats["bytes_peak"] = max(self.stats["bytes_peak"], sum(self._bytes.values()))
        return entry if entry.key in self._entries[entry.module] else None

    def _remove(self, module, key):
        entry = self._entries[module].pop(key, None)
        if entry is None:
            return None
        self._bytes[module] -= entry.size
        if module == "M4":
            self.failure_index = {fp: oid for fp, oid in self.failure_index.items()
                                  if oid != key}
        return entry

    def _victim(self, modules):
        candidates = [entry for module in modules for entry in self._entries[module].values()
                      if not self._pinned(entry)]
        if not candidates:
            return None
        policy = self.policy["eviction"]["policy"]
        if policy == "lru":
            return min(candidates, key=lambda e: e.seq)
        if policy == "lfu":
            return min(candidates, key=lambda e: (e.hits, e.seq))
        return min(candidates, key=lambda e: (e.priority, e.seq))

    def _enforce(self, module):
        capacity = self.policy["capacity"]
        total = capacity["total_bytes"]
        while True:
            limit = capacity["max_entries"][module]
            if limit and len(self._entries[module]) > limit:
                scope = (module,)
            elif (total and not capacity["borrow"]
                  and self._bytes[module] > total * capacity["share"][module]):
                scope = (module,)
            elif total and sum(self._bytes.values()) > total:
                over = tuple(m for m in MODULES
                             if self._bytes[m] > total * capacity["share"][m])
                scope = over or MODULES
            else:
                return
            victim = self._victim(scope)
            if victim is None and scope != MODULES and total and \
                    sum(self._bytes.values()) > total:
                victim = self._victim(MODULES)
            if victim is None:
                self.stats["capacity_overflow"] += 1
                return
            if self.policy["eviction"]["policy"] == "gdsf":
                self._aging = victim.priority
            self._remove(victim.module, victim.key)
            self.stats["evictions"] += 1

    def _purge(self, modules):
        for module in modules:
            for key, entry in list(self._entries[module].items()):
                if not self._valid(entry):
                    self._remove(module, key)
                    self.stats["invalidated_entries"] += 1

    # ------------------------------------------------------------ step hooks
    def advance_step(self):
        """Call once per agent decision step, before the model is asked."""
        self.step += 1
        idle = self.policy["forget"]["idle_steps"]
        for key, entry in list(self._entries["M2"].items()):
            if not self._alive(entry) or (idle and self.step - entry.last_step >= idle):
                self._remove("M2", key)
                self.stats["forgotten_entries"] += 1

    def end_turn(self):
        """Call at each user-turn boundary."""
        self.turn += 1
        mode = self.policy["forget"]["on_turn_end"]
        for key, entry in list(self._entries["M2"].items()):
            if mode == "drop_all_M2" or (mode == "drop_volatile"
                                          and entry.fresh_class == "volatile"):
                self._remove("M2", key)
                self.stats["forgotten_entries"] += 1

    # -------------------------------------------------------------------- M1
    def _catalog_scoped(self, public):
        return self.policy["coupling"]["catalog_auth_scoped"] or not public

    def catalog(self, server, version, fetch, principal=None, public=False,
                ttl_ms=None, cache_scope=None):
        """Catalog read. ``ttl_ms`` / ``cache_scope`` follow MCP SEP-2549; when the
        fetched list result carries ``ttlMs`` / ``cacheScope`` those take effect."""
        self.stats["M1_lookups"] += 1
        key = json.dumps([server, version, principal])
        enabled = "M1" not in self.disabled
        entry = self._entries["M1"].get(key)
        if enabled and entry is not None and self._valid(entry):
            self._touch(entry)
            self.stats["M1_hits"] += 1
            return copy.deepcopy(entry.value)
        if cache_scope is not None:
            public = cache_scope == "public"
        shared = self._shared("M1") if not self._catalog_scoped(public) else None
        if (enabled and shared is not None and key in shared
                and server not in self.catalog_dirty):
            self.stats["M1_hits"] += 1
            self.stats["M1_shared_hits"] += 1
            value = shared[key]
            self._store_catalog(key, server, version, value, False, principal, 0.0, ttl_ms)
            return copy.deepcopy(value)
        start = self.clock()
        value = fetch()
        latency_ms = (self.clock() - start) * 1000
        if isinstance(value, dict):
            ttl_ms = value.get("ttlMs", ttl_ms)
            if value.get("cacheScope") in {"public", "private"}:
                public = value["cacheScope"] == "public"
        scoped = self._catalog_scoped(public)
        digest = hashlib.sha256(_canonical(value).encode()).hexdigest()
        previous = self.last_catalog.get(server)
        if previous is not None and previous[1] != digest:
            if previous[0] == version and server not in self.catalog_dirty:
                # Same announced version, different content: a silent change
                # (e.g. a tool description edited after approval).
                self.stats["catalog_silent_changes"] += 1
            self._catalog_changed(server, previous[2], value)
        self.catalog_dirty.discard(server)
        self.last_catalog[server] = (version, digest, copy.deepcopy(value))
        if enabled:
            self._store_catalog(key, server, version, value, scoped, principal, latency_ms,
                                ttl_ms)
            if not scoped and self._shared("M1") is not None:
                self._shared("M1")[key] = copy.deepcopy(value)
        return value

    def notify_list_changed(self, server):
        """MCP ``notifications/tools/list_changed``: drop the catalog now; dependent
        reads are invalidated when the new catalog is read and compared."""
        self.catalog_dirty.add(server)
        self._bump({"cat:" + server})
        if self.shared is not None:
            store = self.shared.get("M1", {})
            for key in [k for k in store if json.loads(k)[0] == server]:
                store.pop(key)

    def _store_catalog(self, key, server, version, value, scoped, principal, latency_ms,
                       ttl_ms=None):
        tags = {"cat:" + server}
        if scoped:
            tags.add(AUTH_ALL)
            if principal:
                tags.add("auth:" + principal)
        # Freshness: the server's ttlMs, else ttl.catalog_default (seconds), else
        # version + content hash + authorization only.
        default = self.policy["ttl"]["catalog_default"]
        if ttl_ms is not None:
            expiry = (self.clock() + ttl_ms / 1000, None, None)
        elif default is not None:
            expiry = (self.clock() + default, None, None)
        else:
            expiry = (None, None, None)
        self._put(Entry("M1", key, copy.deepcopy(value), _size(value), self._snapshot(tags),
                        expiry=expiry, extra={"server": server, "version": version}),
                  latency_ms)

    def _shared(self, module):
        tier = self.policy["tiers"][module]
        if tier == "task" or self.shared is None:
            return None
        return self.shared.setdefault(module, {})

    @staticmethod
    def _tool_map(value):
        tools = value.get("tools") if isinstance(value, dict) else value
        if not isinstance(tools, list) or not all(
                isinstance(tool, dict) and "name" in tool for tool in tools):
            return None
        return {tool["name"]: json.dumps(tool, sort_keys=True, default=str) for tool in tools}

    def _catalog_changed(self, server, old, new):
        self.stats["catalog_changes"] += 1
        before, after = self._tool_map(old), self._tool_map(new)
        if "catalog_change_scoped" in self.rules and before is not None and after is not None:
            changed = {name for name in set(before) | set(after)
                       if before.get(name) != after.get(name)}
            tags = {f"tool:{server}/{name}" for name in changed}
        else:
            tags = {"srv:" + server}
        tags.add("cat:" + server)
        self._bump(tags)

    # -------------------------------------------------------------------- M3
    def _credential_key(self, service, subject, scopes, principal):
        return json.dumps([service, subject, sorted(scopes), principal])

    def save_credential(self, service, subject, scopes, token, valid_until, principal=None):
        # valid_until comes from the adapter's explicit contract; opaque tokens
        # never receive an invented lifetime.
        if principal and "auth_upgrade_keeps_m2" not in self.rules:
            self._bump({"auth:" + principal})
        tags = {AUTH_ALL} | ({"auth:" + principal} if principal else set())
        if principal:
            # The latest sign-in decides until when this principal is authorized.
            self._authorized_until[principal] = valid_until
        key = self._credential_key(service, subject, scopes, principal)
        entry = Entry("M3", key, token, _size(token), self._snapshot(tags),
                      expiry=(valid_until, None, None),
                      extra={"service": service, "subject": subject,
                             "scopes": sorted(scopes), "principal": principal})
        self._put(entry)
        shared = self._shared("M3")
        if shared is not None:
            shared[key] = (token, valid_until, entry.extra)

    def credential(self, service, subject, scopes, principal=None):
        self.stats["M3_lookups"] += 1
        if "M3" in self.disabled:
            return None
        key = self._credential_key(service, subject, scopes, principal)
        entry = self._entries["M3"].get(key)
        if entry is None and "credential_scope_superset" in self.rules:
            wanted = set(scopes)
            for candidate in self._entries["M3"].values():
                extra = candidate.extra
                if (extra["service"] == service and extra["subject"] == subject
                        and extra["principal"] == principal
                        and wanted <= set(extra["scopes"]) and self._valid(candidate)):
                    entry = candidate
                    break
        if entry is not None and self._valid(entry):
            refreshed = self._maybe_refresh(entry, principal)
            if refreshed is not None:
                return refreshed
            self._touch(entry)
            self.stats["M3_hits"] += 1
            return entry.value
        shared = self._shared("M3")
        if shared is not None and key in shared:
            token, valid_until, extra = shared[key]
            if self.clock() < valid_until:
                self.save_credential(service, subject, scopes, token, valid_until, principal)
                self.stats["M3_hits"] += 1
                self.stats["M3_shared_hits"] += 1
                return token
            shared.pop(key, None)
        return None

    def _maybe_refresh(self, entry, principal):
        if ("credential_refresh_ahead" not in self.rules or principal not in self.refreshers
                or entry.expires_at is None):
            return None
        if entry.expires_at - self.clock() >= self.policy["coupling"]["refresh_margin_s"]:
            return None
        fresh = self.refreshers[principal]()
        if not fresh:
            return None
        token, valid_until = fresh
        extra = entry.extra
        self.save_credential(extra["service"], extra["subject"], extra["scopes"], token,
                             valid_until, principal)
        self.stats["auth_refreshes"] += 1
        self.stats["M3_hits"] += 1
        return token

    def _lapse_authorizations(self):
        """A sign-in the host knows has run out no longer covers what was read
        under it: such results must not be served as if the user were still
        signed in. Checked before every call."""
        now = self.clock()
        for principal in [p for p, until in self._authorized_until.items() if now >= until]:
            self._authorized_until.pop(principal)
            self.stats["authorization_lapses"] = self.stats.get("authorization_lapses", 0) + 1
            self.invalidate_authorization(principal)

    def invalidate_authorization(self, principal=None):
        """Logout, revocation or password change. ``None`` means every principal."""
        for name in ([principal] if principal else list(self._authorized_until)):
            self._authorized_until.pop(name, None)
        self.auth_epoch += 1
        partition = self.policy["coupling"]["partition_by_principal"]
        if principal is None or not partition:
            self._bump({AUTH_ALL})
            if self.shared is not None:
                self.shared.get("M3", {}).clear()
        else:
            self._bump({"auth:" + principal})
            if self.shared is not None:
                store = self.shared.get("M3", {})
                for key in [k for k, v in store.items() if v[2].get("principal") == principal]:
                    store.pop(key)

    # -------------------------------------------------------------------- M4
    @property
    def operations(self):
        return {key: entry.value for key, entry in self._entries["M4"].items()}

    def operation_audit(self):
        """JSON-safe operation state, without arguments, credentials or results."""
        return [{k: copy.deepcopy(v) for k, v in entry.value.items()
                 if k not in {"result", "failure", "request"} and not k.startswith("_")}
                for entry in self._entries["M4"].values()]

    def operation_current(self, record):
        """Whether a failed/uncertain record still describes the current conditions."""
        if not self._tags_valid(record.get("_auth", {})):
            return False
        if record.get("state") == "uncertain":
            return True
        return self._tags_valid(record.get("_cond", {}))

    def _store_operation(self, operation_id, record):
        entry = Entry("M4", operation_id, record, _size({k: v for k, v in record.items()
                                                         if k != "failure"}), {})
        self._put(entry)

    # ------------------------------------------------------------ meta/tags
    def _normalize(self, name, meta):
        meta = dict(meta or {})
        server = meta.get("server") or "default"
        app = meta.get("app") or server
        tool = meta.get("tool") or name
        principal = meta.get("principal")
        contract_class = meta.get("fresh_class") or "slow"
        override = self.policy["ttl"].get("fresh_class_overrides", {}).get(tool)
        fresh = max((contract_class, override or contract_class), key=CLASS_ORDER.get)
        auth_tags = {AUTH_ALL} | ({"auth:" + principal} if principal else set())
        read_tags = ({ALL, "srv:" + server, "app:" + app, f"tool:{server}/{tool}"}
                     | {"res:" + r for r in meta.get("reads", [])} | auth_tags)
        return {"server": server, "app": app, "tool": tool, "principal": principal,
                "fresh_class": fresh, "read_tags": read_tags, "auth_tags": auth_tags,
                "affects": list(meta.get("affects", [])),
                "idempotent": bool(meta.get("idempotency_key")),
                "shadow_fetch": meta.get("shadow_fetch")}

    def _write_tags(self, meta, unknown):
        granularity = self.policy["invalidation"]["granularity"]
        if granularity == "resource" and not meta["affects"]:
            granularity = self.policy["locked"]["untagged_write_min_granularity"]
        if (unknown and self.policy["invalidation"]["quarantine_on_unknown_write"]
                and GRANULARITY_ORDER[granularity] < GRANULARITY_ORDER["app"]):
            granularity = "app"
        if granularity == "global":
            return {ALL}
        if granularity == "server":
            return {"srv:" + meta["server"]}
        if granularity == "app":
            return {"app:" + meta["app"]}
        return {"res:" + r for r in meta["affects"]}

    # ------------------------------------------------------------- execution
    def _backoff_s(self, attempt, error):
        cfg = self.policy["retry"]
        delay_ms = min(cfg["backoff_cap_ms"],
                       cfg["backoff_base_ms"] * cfg["backoff_factor"] ** attempt)
        if cfg["jitter"] == "full" and delay_ms:
            delay_ms = self.rng.uniform(0, delay_ms)
        if cfg["honor_retry_after"] and error.retry_after:
            delay_ms = max(delay_ms, error.retry_after * 1000)
        return delay_ms / 1000

    def _breaker_key(self, name, meta):
        meta = meta or {}
        return f"{meta.get('server') or 'default'}/{meta.get('tool') or name}"

    def _idempotent_write(self, readonly, meta):
        return (not readonly and bool((meta or {}).get("idempotency_key"))
                and self.policy["recovery"]["unknown_write"] == "idempotent_retry")

    def execute(self, name, arguments, fetch, *, readonly=False, operation_id=None,
                scope="", max_attempts=2, invalidate=True, meta=None):
        # The shared retry cap is locked; M4 adds state, not a larger budget.
        max_attempts = min(max_attempts, self.policy["locked"]["retry_max_attempts"])
        principal = (meta or {}).get("principal")
        breaker_cfg = self.policy["recovery"]["breaker"]
        breaker = self.breakers.setdefault(self._breaker_key(name, meta),
                                           {"fails": 0, "open_until": 0})
        if breaker_cfg["threshold"] and self.step < breaker["open_until"]:
            self.stats["logical_calls"] += 1
            self.stats["breaker_short_circuits"] += 1
            raise AccessFailure("circuit_open", retryable=True, outcome="not_executed")
        refreshed = False
        unknown_seen = False  # an attempt of this write got no confirmed outcome
        for attempt in range(max_attempts):
            try:
                value = self.call(name, arguments, fetch, readonly, operation_id, scope,
                                  invalidate, meta=meta)
                breaker["fails"] = 0
                return value
            except AccessFailure as error:
                last = attempt + 1 >= max_attempts
                if (not last and not refreshed and error.code in AUTH_CODES
                        and error.outcome == "not_executed"
                        and self.policy["recovery"]["auth"] == "refresh_then_retry_once"
                        and principal in self.refreshers):
                    fresh = self.refreshers[principal]()
                    if fresh:
                        refreshed = True
                        self.stats["auth_refreshes"] += 1
                        self.stats["retries"] += 1
                        continue
                # A write whose outcome is unknown may be re-sent only under the same
                # idempotency key, where the server deduplicates.
                keyed_retry = (error.outcome == "unknown" and operation_id is not None
                               and self._idempotent_write(readonly, meta))
                transient = error.retryable and error.outcome == "not_executed"
                unknown_seen = unknown_seen or error.outcome == "unknown"
                if last or not (transient or keyed_retry):
                    if transient and breaker_cfg["threshold"]:
                        breaker["fails"] += 1
                        if breaker["fails"] >= breaker_cfg["threshold"]:
                            breaker["open_until"] = self.step + breaker_cfg["cooldown_steps"]
                            self.stats["breaker_trips"] += 1
                    if unknown_seen and error.outcome != "unknown":
                        # An earlier attempt may have been applied. That a later
                        # attempt was turned away (signed out, busy) says nothing
                        # about it, so the agent must not be told "not executed".
                        if error.code in AUTH_CODES and principal:
                            self.invalidate_authorization(principal)
                        raise AccessFailure(
                            "write_outcome_unknown", outcome="unknown",
                            detail="an earlier attempt of this write got no reply; a "
                                   f"later attempt was refused ({error.code})") from error
                    raise
                delay = self._backoff_s(attempt, error)
                if delay:
                    self.stats["backoff_ms"] += delay * 1000
                    self.sleep(delay)
                self.stats["retries"] += 1
                if keyed_retry:
                    self.stats["idempotent_retries"] += 1

    def _failure_replay(self, record):
        error = record.get("failure")
        if error is None:
            return None
        self.stats["M4_failure_hits"] += 1
        return AccessFailure(error.code, retryable=error.retryable, outcome=error.outcome,
                             retry_after=max(0, record["retry_at"] - self.clock()),
                             detail=error.detail)

    def call(self, name, arguments, fetch, readonly=False, operation_id=None, scope="",
             invalidate=True, meta=None):
        self.stats["logical_calls"] += 1
        self._lapse_authorizations()
        info = self._normalize(name, meta)
        identity = json.dumps([scope, name, arguments], sort_keys=True)
        fingerprint = hashlib.sha256(identity.encode()).hexdigest()
        recovery = self.policy["recovery"]
        m4_on = "M4" not in self.disabled
        replay_failures = m4_on and recovery["reuse_same_condition_failure"]
        now = self.clock()
        old_entry = self._entries["M4"].get(operation_id) if operation_id else None
        old = old_entry.value if old_entry else None
        if old:
            if old["identity_sha256"] != fingerprint:
                raise ValueError("operation_id was reused for a different request")
            if old["state"] == "uncertain" and not (
                    info["idempotent"] and not readonly
                    and recovery["unknown_write"] == "idempotent_retry"):
                verdict = self._reconcile(old)
                if verdict is None:
                    raise AccessFailure("write_outcome_unknown", outcome="unknown")
                if verdict[0] == "executed":
                    return copy.deepcopy(verdict[1])
                old = self._entries["M4"][operation_id].value
            auth_same = self._tags_valid(old["_auth"])
            if not readonly and not auth_same:
                raise AccessFailure("operation_authorization_changed")
            valid = auth_same and (not readonly or (self._tags_valid(old["_cond"])
                                                    and self._record_alive(old)))
            if (m4_on and not self.m4_failure_only and valid
                    and old["state"] == "completed"):
                self.stats["M4_hits"] += 1
                return copy.deepcopy(old["result"])
            # An authorization failure says the session is bad, not that the
            # request is doomed: it must stay retryable after a sign-in.
            if (replay_failures and old.get("failure")
                    and old["failure"].code not in AUTH_CODES
                    and self._tags_valid(old["_cond"]) and self._failure_fresh(old)
                    and (not old["failure"].retryable or now < old["retry_at"])):
                raise self._failure_replay(old)
        if replay_failures and recovery["failure_index"] == "identity":
            prior_id = self.failure_index.get(fingerprint)
            prior = self._entries["M4"].get(prior_id) if prior_id else None
            if (prior is not None and prior.value["state"] == "failed"
                    and prior.value.get("failure") is not None
                    and not prior.value["failure"].retryable
                    and self._tags_valid(prior.value["_cond"])
                    and self._failure_fresh(prior.value)):
                raise self._failure_replay(prior.value)
        cacheable = readonly and info["fresh_class"] != "strict"
        if readonly:
            self.stats["M2_eligible"] += 1
            if not cacheable:
                self.stats["M2_strict_bypass"] += 1
            entry = self._entries["M2"].get(identity)
            if "M2" not in self.disabled and cacheable and entry and self._valid(entry):
                self._shadow_check(entry, info)
                adaptive = self.policy["ttl"]["adaptive"]
                if adaptive["enabled"] and self.rng.random() < adaptive["verify_sample_rate"]:
                    return self._verified_hit(entry, info, fetch, identity, scope, name)
                self._touch(entry)
                self.stats["M2_hits"] += 1
                return copy.deepcopy(entry.value)
        self.stats["backend_calls"] += 1
        record = {
            "operation_id": operation_id, "identity_sha256": fingerprint,
            "auth_epoch": self.auth_epoch, "data_epoch": self.data_epoch,
            "readonly": readonly, "state": "running",
            "attempts": (old or {}).get("attempts", 0) + 1,
            "created_at": (old or {}).get("created_at", now),
            "updated_at": now, "valid_until": now, "retry_at": now,
            "error_code": None, "outcome": None,
            "history": copy.deepcopy((old or {}).get("history", [])),
            "request": {"name": name, "arguments": copy.deepcopy(arguments)},
            "_cond": self._snapshot(info["read_tags"]),
            "_auth": self._snapshot(info["auth_tags"]),
            "_server": info["server"], "_tool": info["tool"], "_step": self.step}
        start = self.clock()
        try:
            value = fetch()
        except Exception as error:
            outcome = getattr(error, "outcome", "unknown")
            # Once uncertain, a write stays uncertain until the server says what
            # became of it: a refused re-send is not such a statement.
            was_uncertain = bool(old and old.get("state") == "uncertain")
            uncertain = not readonly and (outcome == "unknown" or was_uncertain)
            if was_uncertain:
                outcome = "unknown"
            if uncertain and invalidate:
                self.data_epoch += 1
                self._bump(self._write_tags(info, unknown=True))
            elif (not readonly and invalidate and outcome == "not_executed"
                  and isinstance(error, AccessFailure) and not error.retryable
                  and error.code not in AUTH_CODES
                  and self.policy["invalidation"].get("on_refused_write")):
                # The server refused the write for a business reason. Whatever the
                # host believed about the state it touches is suspect.
                self._bump(self._write_tags(info, unknown=False))
                # The refusal itself is the newest thing known about that state.
                record["_cond"] = self._snapshot(info["read_tags"])
            record.update(state="uncertain" if uncertain else "failed",
                          outcome=outcome, updated_at=self.clock(),
                          error_code=(error.code if isinstance(error, AccessFailure)
                                      else "unknown_error"),
                          retry_at=self.clock() + getattr(error, "retry_after", 0),
                          # A remembered refusal is a cached answer about server
                          # state that others can change: it ages like a volatile read.
                          _fail_expiry=list(self._expiry(
                              self.policy["recovery"].get("failure_ttl", 0))))
            record["history"].append({"attempt": record["attempts"], "state": record["state"],
                                      "error_code": record["error_code"]})
            if isinstance(error, AccessFailure):
                record["failure"] = error
            if operation_id:
                self._store_operation(operation_id, record)
                if (isinstance(error, AccessFailure) and not error.retryable
                        and outcome == "not_executed" and m4_on
                        and error.code not in AUTH_CODES):
                    self.failure_index[fingerprint] = operation_id
            raise
        latency_ms = (self.clock() - start) * 1000
        if not readonly and invalidate:
            self.data_epoch += 1
            self._bump(self._write_tags(info, unknown=False))
        elif readonly and cacheable and "M2" not in self.disabled:
            self._store_read(identity, value, info, scope, name, latency_ms)
        if operation_id:
            expiry = self._expiry(self.policy["ttl"]["m4_completed"])
            record.update(state="completed", result=copy.deepcopy(value),
                          updated_at=self.clock(), outcome="completed",
                          data_epoch=self.data_epoch,
                          valid_until=expiry[0] if expiry[0] is not None else self.clock(),
                          _expiry=list(expiry),
                          _cond=self._snapshot(info["read_tags"]))
            record["history"].append({"attempt": record["attempts"], "state": "completed",
                                      "error_code": None})
            self.failure_index.pop(fingerprint, None)
            self._store_operation(operation_id, record)
        return value

    # ------------------------------------------------- M2 freshness learning
    def _class_ttl(self, info):
        """TTL of the read's class, scaled by what sampled verification learned."""
        ttl = self.policy["ttl"][info["fresh_class"]]
        scale = self.ttl_scale.get(info["tool"], 1.0)
        if scale == 1.0:
            return ttl
        # A static read observed stale is demoted to the slow class before scaling.
        return (ttl if ttl is not None else self.policy["ttl"]["slow"]) * scale

    def _store_read(self, identity, value, info, scope, name, latency_ms):
        self._put(Entry("M2", identity, copy.deepcopy(value), _size(value),
                        self._snapshot(info["read_tags"]), info["fresh_class"],
                        self._expiry(self._class_ttl(info)),
                        extra={"scope": scope, "name": name, "tool": info["tool"]}),
                  latency_ms)

    def _shadow_check(self, entry, info):
        """Evaluation only: compare a hit with the backend without changing the
        returned value. The caller supplies an unmetered fetch; its time is
        reported separately so it can be excluded from method time."""
        shadow = info["shadow_fetch"]
        if shadow is None:
            return
        start = self.clock()
        try:
            truth = shadow()
        except Exception:
            self.stats["shadow_errors"] += 1
        else:
            self.stats["shadow_checks"] += 1
            if _canonical(truth) != _canonical(entry.value):
                self.stats["shadow_stale"] += 1
        self.stats["shadow_time_s"] += self.clock() - start

    def _verified_hit(self, entry, info, fetch, identity, scope, name):
        """Sampled verification of a hit; its backend call is a real, metered cost."""
        adaptive = self.policy["ttl"]["adaptive"]
        self.stats["M2_verifications"] += 1
        self.stats["backend_calls"] += 1
        start = self.clock()
        try:
            value = fetch()
        except Exception:
            # The check itself failed; the entry is still valid by its own rules.
            self.stats["M2_verify_errors"] += 1
            self._touch(entry)
            return copy.deepcopy(entry.value)
        latency_ms = (self.clock() - start) * 1000
        stale = _canonical(value) != _canonical(entry.value)
        scale = self.ttl_scale.get(info["tool"], 1.0) * math.exp(
            adaptive["step"] * (adaptive["target_stale_rate"] - (1.0 if stale else 0.0)))
        self.ttl_scale[info["tool"]] = min(adaptive["max_scale"],
                                           max(adaptive["min_scale"], scale))
        if stale:
            self.stats["M2_verify_stale"] += 1
        # Either way the backend value is the newest; store it with a fresh expiry.
        self._store_read(identity, value, info, scope, name, latency_ms)
        return copy.deepcopy(value)

    def _record_alive(self, record):
        expires_at, expires_step, expires_turn = record.get("_expiry", (None, None, None))
        return ((expires_at is None or self.clock() < expires_at)
                and (expires_step is None or self.step < expires_step)
                and (expires_turn is None or self.turn < expires_turn))

    def _failure_fresh(self, record):
        """Whether a remembered refusal may still be answered without asking.
        With ``recovery.failure_ages`` it lives for ``recovery.failure_ttl``."""
        if not self.policy["recovery"].get("failure_ages"):
            return True
        expires_at, expires_step, expires_turn = record.get("_fail_expiry", (None, None, None))
        return ((expires_at is None or self.clock() < expires_at)
                and (expires_step is None or self.step < expires_step)
                and (expires_turn is None or self.turn < expires_turn))

    def _reconcile(self, record):
        """Resolve an uncertain write through the declared reconciliation read."""
        # idempotent_retry falls back to reconciliation for tools without a key.
        if (self.policy["recovery"]["unknown_write"] == "surface_only"
                or self.reconciler is None):
            return None
        verdict = self.reconciler({**copy.deepcopy(record["request"]),
                                   "operation_id": record["operation_id"]})
        if not verdict or verdict[0] not in {"executed", "not_executed"}:
            return None
        self.stats["reconciliations"] += 1
        state, value = verdict
        record = copy.deepcopy(record)
        record["history"].append({"attempt": record["attempts"],
                                  "state": "completed" if state == "executed" else "failed",
                                  "error_code": None if state == "executed"
                                  else "reconciled_not_executed"})
        if state == "executed":
            record.update(state="completed", outcome="completed", result=copy.deepcopy(value),
                          updated_at=self.clock())
        else:
            record.update(state="failed", outcome="not_executed",
                          error_code="reconciled_not_executed", updated_at=self.clock())
            record.pop("failure", None)
        self._store_operation(record["operation_id"], record)
        return verdict

    # ------------------------------------------------------- publication API
    def iter_valid_reads(self):
        """(identity, value) of currently valid M2 entries, newest first."""
        for entry in sorted(self._entries["M2"].values(), key=lambda e: -e.seq):
            if self._valid(entry):
                yield entry.key, entry.value

    def valid_read(self, identity):
        entry = self._entries["M2"].get(identity)
        if entry is not None and self._valid(entry):
            return True, entry.value
        return False, None

    def valid_catalog(self, server):
        for entry in self._entries["M1"].values():
            if entry.extra.get("server") == server and self._valid(entry):
                return True, entry.value
        return False, None

    def usage(self):
        """Bytes and entry counts per module, for traces and reports."""
        usage = {module: {"bytes": self._bytes[module],
                          "entries": len(self._entries[module])} for module in MODULES}
        usage["ttl_scale"] = dict(self.ttl_scale)
        return usage
