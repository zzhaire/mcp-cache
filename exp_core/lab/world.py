"""A small, fully controllable service world for MCP access experiments.

Three apps sit behind one MCP server: ``bank`` and ``shop`` need a login,
``ref`` and ``clock`` are public. Everything a cache or recovery policy can get
wrong is a controlled input of a task:

- external changes (a price moves, a deposit arrives) fire at the start of a
  user turn, so every method and every agent sees the same world timeline;
- faults (rate limit, unavailable, lost acknowledgement) fire on the n-th call
  of a tool;
- sessions can be expired by the server at a given turn;
- the tool catalog has a version and can change mid-task.

The world also publishes the contract a host needs: which tools are read-only,
how fresh their results stay, which resources a read depends on and a write
affects, and which writes accept an idempotency key. The module is pure Python
(no MCP, no network) so the same world runs in the server and in tests.
"""
import copy
import json
import math
import re

from exp_core.lab.judge import judge, judge_overlay, served_entry  # noqa: F401

OK_USERS = "users"


class BusinessError(Exception):
    """A request the service understood and refused; nothing was changed."""


class Unauthorized(Exception):
    """The caller is not signed in as someone who may see this."""


# lost_ack: applied, reply lost. lost_request: dropped before the server applied
# it, reply lost. The host sees the same thing in both cases.
FAULT_KINDS = {"rate_limit", "unavailable", "lost_ack", "lost_request"}
TRIGGERS = ("at_turn", "after_call", "at_time")
STEP_SECONDS = 10       # virtual seconds one agent decision step takes
SESSION_TTL_S = 3600    # lifetime of a sign-in unless the task says otherwise
EVENT_KINDS = {"set_price", "set_stock", "deposit", "expire_session", "catalog_update"}


def _type_error(name, value, schema):
    """Why ``value`` does not fit a parameter schema, or None."""
    if schema.get("type") == "string":
        return None if isinstance(value, str) else f"{name} must be a string"
    if schema.get("type") == "number":
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value)):
            return f"{name} must be a finite number"
    return None


def _tool(app, kind, description, parameters, *, fresh=None, reads=(), affects=(),
          auth=True, key=False, idempotent=False):
    return {"app": app, "kind": kind, "description": description,
            "parameters": parameters, "fresh_class": fresh, "reads": list(reads),
            "affects": list(affects), "auth": auth, "idempotency_key": key,
            "idempotent": idempotent}


S, N = {"type": "string"}, {"type": "number"}

TOOLS = {
    "ref_list_currencies": _tool(
        "ref", "read", "List the currency codes with a published exchange rate.", {},
        fresh="static", reads=["ref/currencies"], auth=False),
    "ref_get_rate": _tool(
        "ref", "read", "Exchange rate of one currency against USD.", {"currency": S},
        fresh="static", reads=["ref/rate/{currency}"], auth=False),
    "clock_now": _tool(
        "clock", "read", "Current service time.", {}, fresh="strict", auth=False),
    "auth_login": _tool(
        "auth", "login", "Sign in to an app (bank or shop). Needed before using it.",
        {"app": S, "username": S, "password": S}, auth=False),
    "auth_logout": _tool(
        "auth", "logout", "Sign out of an app.", {"app": S}, auth=False),
    "bank_list_accounts": _tool(
        "bank", "read", "List the user's bank accounts (id and owner, no balances).", {},
        fresh="slow", reads=["bank/accounts"]),
    "bank_get_balance": _tool(
        "bank", "read", "Current balance of one account.", {"account": S},
        fresh="slow", reads=["bank/balance/{account}"]),
    "bank_list_transfers": _tool(
        "bank", "read", "Transfers into or out of one account.", {"account": S},
        fresh="slow", reads=["bank/transfers/{account}"]),
    "bank_transfer": _tool(
        "bank", "write", "Move money between two accounts. Returns the transfer id.",
        {"source": S, "target": S, "amount": N},
        affects=["bank/balance/{source}", "bank/balance/{target}",
                 "bank/transfers/{source}", "bank/transfers/{target}"], key=True),
    "shop_list_items": _tool(
        "shop", "read", "List the shop's items (sku and name).", {},
        fresh="static", reads=["shop/items"]),
    "shop_get_item": _tool(
        "shop", "read", "Name and category of one item.", {"sku": S},
        fresh="static", reads=["shop/item/{sku}"]),
    "shop_get_price": _tool(
        "shop", "read", "Current unit price of one item. Prices change over time.",
        {"sku": S}, fresh="volatile", reads=["shop/price/{sku}"]),
    "shop_get_stock": _tool(
        "shop", "read", "Units of one item in stock. Other customers buy too.",
        {"sku": S}, fresh="volatile", reads=["shop/stock/{sku}"]),
    "shop_place_order": _tool(
        "shop", "write", "Order a quantity of one item at the current price.",
        {"sku": S, "quantity": N}, affects=["shop/stock/{sku}", "shop/orders"]),
    "shop_list_orders": _tool(
        "shop", "read", "The user's orders with their status.", {},
        fresh="slow", reads=["shop/orders"]),
    "shop_cancel_order": _tool(
        "shop", "write", "Cancel one order. Cancelling twice is an error.",
        {"order_id": S}, affects=["shop/orders"], idempotent=False),
    "op_status": _tool(
        "ops", "status", "Whether a write submitted under an operation key was applied.",
        {"key": S}, fresh="strict", auth=False),
}
# Tools that only exist after a catalog update (see the ``catalog_update`` event).
LATER_TOOLS = {
    "shop_get_bulk_price": _tool(
        "shop", "read", "Unit price of one item when ordering at least 10 units.",
        {"sku": S}, fresh="volatile", reads=["shop/price/{sku}"]),
    "shop_get_quote": _tool(
        "shop", "read", "Total price for a quantity of one item (10% off from 10 units).",
        {"sku": S, "quantity": N}, fresh="volatile", reads=["shop/price/{sku}"]),
}
PAD_PREFIX = "archive_lookup_"


def _pad_tool(index):
    """A public, static filler tool: it only makes the catalog bigger."""
    return _tool("archive", "read",
                 f"Look up one record in archive volume {index:02d}. Rarely needed: use it "
                 "only when a request names this archive volume explicitly.",
                 {"record": S}, fresh="static", reads=[f"archive/{index:02d}/{{record}}"],
                 auth=False)


class LabWorld:
    def __init__(self, task):
        self.task = copy.deepcopy(task)
        state = self.task["initial"]
        self.bank = state.get("bank", {"accounts": {}})
        self.bank.setdefault("transfers", [])
        # Bulk history for large-payload tasks, expanded here to keep task files small.
        self._seeded = tuple(seed["prefix"] + "-" for seed in self.bank.get("seed_transfers", []))
        for seed in self.bank.pop("seed_transfers", []):
            self.bank["transfers"].extend(
                {"id": f"{seed['prefix']}-{i}", "source": seed["source"],
                 "target": seed["target"], "amount": seed.get("amount", 1)}
                for i in range(1, seed["count"] + 1))
        self.shop = state.get("shop", {"items": {}})
        self.shop.setdefault("orders", {})
        self._order_owner = {}
        self.ref = state.get("ref", {"rates": {}})
        self.users = state.get(OK_USERS, {})
        # Sign-ins are tokens the caller presents with every request. The server
        # checks who the token belongs to, whether it was revoked and whether
        # it has expired on the virtual clock. ``sessions`` is the view of the
        # tokens presented with the request being handled.
        self.tokens = {}
        self.sessions = {}
        self._jar = {}  # used when a caller presents nothing: the last token per app
        self.session_ttl_s = self.task.get("session_ttl_s", SESSION_TTL_S)
        self.step = 0
        self.turn = 0
        # Virtual time: the only clock in an experiment. Hosts read it instead
        # of the wall clock, so no result depends on how fast a model answers.
        self.step_seconds = self.task.get("step_seconds", STEP_SECONDS)
        self.turn_gap_s = self.task.get("turn_gap_s", 0)
        self.now = 0
        self.events = list(self.task.get("events", []))
        self.applied_events = []
        self.faults = self.task.get("faults", [])
        self.calls = {}
        # Calls the agent made, whether or not they reached the server (a host
        # may have answered from its cache). ``after_call`` events count these,
        # so they fire at the same point of the task for every method.
        self.logical = {}
        # Idempotency / operation keys the server has seen: key -> who sent what,
        # and whether it was applied. A key it has never seen is "unknown".
        self.operations = {}
        self.writes = []
        self.truth = {}  # probe index -> the value the server held that turn
        # Every write request that passed validation and authorization, with
        # its outcome: committed, refused (and why) or answered from a key.
        self.write_attempts = []
        self.fault_hits = []
        self.logins = 0
        self.base_tools, self.later_tools = self._tool_tables()
        self.tools = dict(self.base_tools)
        for name in self.task.get("tools_add", []):
            self.tools[name] = self.later_tools[name]
        # Filler tools: the task's initial padding plus whatever events add later.
        self.padding = 0
        self._pad_total = self.task.get("catalog_padding", 0) + sum(
            event.get("pad", 0) for event in self.task.get("events", []))
        self._pad(self.task.get("catalog_padding", 0))
        self._version = "v1"
        # Switches a catalog update can flip to change what a tool returns.
        self.behavior = {}
        # Task flags: accounts are private to their owner; the catalog only shows
        # the tools of apps the caller is signed in to.
        self.strict_ownership = bool(self.task.get("strict_ownership"))
        self.catalog_private = bool(self.task.get("catalog_private"))
        self._check_task()

    # What a domain other than the built-in bank/shop world replaces
    # (see ``exp_core.lab.domains``): the tools, the apps that need a sign-in,
    # the kinds of external change, and how arguments are checked.
    login_apps = frozenset({"bank", "shop"})
    event_kinds = frozenset(EVENT_KINDS)

    def _tool_tables(self):
        """``(tools at the start, tools a catalog update may add)``."""
        return TOOLS, LATER_TOOLS

    def _schema(self, spec):
        return {"type": "object", "properties": spec["parameters"],
                "required": list(spec["parameters"])}

    def _invalid(self, name, spec, arguments):
        """Why ``arguments`` do not fit the tool, or None."""
        if not isinstance(arguments, dict) or set(arguments) != set(spec["parameters"]):
            return f"{name} takes exactly: {sorted(spec['parameters'])}"
        for parameter, schema in spec["parameters"].items():
            problem = _type_error(parameter, arguments[parameter], schema)
            if problem:
                return problem
        return None

    def _apply_domain(self, event):
        raise ValueError(f"unknown event kind {event['kind']}")

    def _check_task(self):
        """Refuse a task file whose faults or events could never take effect."""
        known = {**self.base_tools, **self.later_tools}
        for fault in self.faults:
            tool = fault.get("tool")
            if fault.get("kind") not in FAULT_KINDS:
                raise ValueError(f"{self.task.get('id')}: unknown fault kind {fault.get('kind')}")
            if tool not in known and not str(tool).startswith(PAD_PREFIX):
                raise ValueError(f"{self.task.get('id')}: fault on unknown tool {tool}")
            if (fault["kind"] in {"lost_ack", "lost_request"}
                    and known.get(tool, {}).get("kind") != "write"):
                raise ValueError(f"{self.task.get('id')}: {fault['kind']} needs a write tool, "
                                 f"got {tool}")
            if not fault.get("on_calls") and fault.get("nth") is None:
                raise ValueError(f"{self.task.get('id')}: fault on {tool} names no call")
        for event in self.task.get("events", []):
            if event.get("kind") not in self.event_kinds:
                raise ValueError(f"{self.task.get('id')}: unknown event kind {event.get('kind')}")
            if sum(trigger in event for trigger in TRIGGERS) != 1:
                raise ValueError(f"{self.task.get('id')}: an event needs exactly one of {TRIGGERS}")
            after = event.get("after_call")
            if after is not None and (after.get("tool") not in known or not after.get("nth")):
                raise ValueError(f"{self.task.get('id')}: bad after_call trigger {after}")

    def _pad(self, count):
        for _ in range(count):
            self.tools[f"{PAD_PREFIX}{self.padding:02d}"] = _pad_tool(self.padding)
            self.padding += 1

    @property
    def catalog_version(self):
        """The announced version. A private catalog depends on who is signed in."""
        if not self.catalog_private:
            return self._version
        return "+".join([self._version, *sorted(self.sessions)])

    def _visible(self, spec):
        return not self.catalog_private or not spec["auth"] or spec["app"] in self.sessions

    # -------------------------------------------------------------- catalog
    def catalog(self, credentials=None):
        self._present(credentials)
        self._apply_due()
        tools = [{"name": name, "description": spec["description"],
                  "parameters": self._schema(spec),
                  "annotations": {"readOnlyHint": spec["kind"] in {"read", "status"},
                                  "idempotentHint": spec["kind"] in {"read", "status"}
                                  or spec["idempotent"] or spec["idempotency_key"]}}
                 for name, spec in sorted(self.tools.items()) if self._visible(spec)]
        listing = {"version": self.catalog_version, "tools": tools,
                   "cacheScope": "private" if self.catalog_private else "public"}
        if self.task.get("catalog_ttl_ms") is not None:
            listing["ttlMs"] = self.task["catalog_ttl_ms"]
        return listing

    def contract(self):
        """What a host may rely on, per tool (the experiment's ground truth)."""
        return {name: {"app": spec["app"], "kind": spec["kind"],
                       "fresh_class": spec["fresh_class"], "reads": spec["reads"],
                       "affects": spec["affects"],
                       "principal": spec["app"] if spec["auth"] else None,
                       "idempotency_key": spec["idempotency_key"]}
                for name, spec in {
                    **self.base_tools, **self.later_tools,
                    **{f"{PAD_PREFIX}{i:02d}": _pad_tool(i) for i in range(self._pad_total)},
                }.items()}

    # ---------------------------------------------------------------- time
    def advance(self, step, turn=0, credentials=None):
        """Move the world to an agent decision step of a user turn. Every step
        takes ``step_seconds`` of virtual time; events that are due are applied."""
        self.step = max(self.step, int(step))
        if int(turn) > self.turn:
            self._note_truth()  # before the next turn's events change anything
        self.turn = max(self.turn, int(turn))
        # A user takes ``turn_gap_s`` to come back between turns (think time).
        self.now = self.step * self.step_seconds + self.turn * self.turn_gap_s
        self._present(credentials)
        applied = self._apply_due()
        return {"step": self.step, "applied": applied, "now": self.now,
                "catalog_version": self.catalog_version}

    def note(self, name, credentials=None):
        """The harness reports that the agent's call to ``name`` has completed.
        Events waiting for that call take effect now, after its result was
        delivered and before anything else happens."""
        self.logical[name] = self.logical.get(name, 0) + 1
        self._present(credentials)
        return {"applied": self._apply_due(), "catalog_version": self.catalog_version}

    def _due(self, event):
        if "at_turn" in event:
            return event["at_turn"] <= self.turn
        if "at_time" in event:
            return event["at_time"] <= self.now
        after = event["after_call"]
        return self.logical.get(after["tool"], 0) >= after["nth"]

    def _apply_due(self):
        """Apply the events whose trigger has fired."""
        applied = []
        for event in [e for e in self.events if self._due(e)]:
            self.events.remove(event)
            self._apply(event)
            applied.append(event["kind"])
            self.applied_events.append({**event, "applied_at": {
                "turn": self.turn, "step": self.step, "now": self.now}})
        return applied

    # ---------------------------------------------------------------- auth
    def _valid(self, token):
        record = self.tokens.get(token)
        return record if (record and not record["revoked"]
                          and self.now < record["expires_at"]) else None

    def _present(self, credentials):
        """Set ``sessions`` to what the presented tokens prove. ``credentials``
        maps an app to a token; None means "whatever this connection holds"."""
        presented = self._jar if credentials is None else credentials
        self._presented = presented
        self.sessions = {}
        for app, token in (presented or {}).items():
            record = self._valid(token)
            if record and record["app"] == app:
                self.sessions[app] = {"user": record["user"], "session": token}

    def _apply(self, event):
        kind = event["kind"]
        if kind == "set_price":
            self.shop["items"][event["sku"]]["price"] = event["price"]
        elif kind == "set_stock":
            self.shop["items"][event["sku"]]["stock"] = event["stock"]
        elif kind == "deposit":
            self.bank["accounts"][event["account"]]["balance"] += event["amount"]
            self.bank["transfers"].append({"id": f"ext-{len(self.bank['transfers']) + 1}",
                                           "source": "external", "target": event["account"],
                                           "amount": event["amount"]})
        elif kind == "expire_session":
            for record in self.tokens.values():  # the server drops every sign-in to the app
                if record["app"] == event["app"]:
                    record["revoked"] = True
            self.sessions.pop(event["app"], None)
        elif kind == "catalog_update":
            # Without "version" the change is silent: same label, new content.
            self._version = event.get("version", self._version)
            for name in event.get("add", []):
                self.tools[name] = self.later_tools[name]
            for name in event.get("remove", []):
                self.tools.pop(name, None)
            for name, text in event.get("describe", {}).items():
                self.tools[name] = {**self.tools[name], "description": text}
            self._pad(event.get("pad", 0))
            self.behavior.update(event.get("behavior", {}))
        else:
            self._apply_domain(event)

    # -------------------------------------------------------------- invoke
    @staticmethod
    def _fail(code, *, retryable=False, outcome="not_executed", **extra):
        return {"ok": False, "error_code": code, "retryable": retryable,
                "outcome": outcome, **extra}

    def _fault(self, name):
        count = self.calls[name]
        for fault in self.faults:
            if fault["tool"] == name and count in fault.get("on_calls", [fault.get("nth")]):
                return fault
        return None

    def invoke(self, name, arguments, key=None, shadow=False, credentials=None):
        """Run one tool. Returns the adapter payload ``{"ok": ..., ...}``.

        ``shadow`` is a measurement read: it sees the true current state and
        leaves no trace (no counters, faults, auth checks or events).
        """
        spec = self.tools.get(name)
        if spec is None:
            return self._fail("unknown_tool", error=f"no tool named {name}")
        problem = self._invalid(name, spec, arguments)
        if problem:
            # The request itself is malformed: this says nothing about the
            # service's state, unlike a refusal for a business reason.
            return self._fail("invalid_arguments", error=problem)
        self._shadow = shadow
        self._present(credentials)
        if not shadow:
            self._apply_due()
        signed_out = spec["auth"] and spec["app"] not in self.sessions
        if shadow:
            # A measurement read: the true current answer, no counters or faults.
            # It is still subject to authorization, so a host that serves a
            # cached value after its session was dropped is seen to do so.
            if spec["kind"] not in {"read", "status"}:
                return self._fail("shadow_only_reads")
            if signed_out:
                return self._fail("unauthorized", error=f"sign in to {spec['app']} first")
            return self._run(name, arguments)
        if signed_out:
            # Refused at the door: it does not count as a call to the tool, so
            # it cannot use up a fault planned for "the n-th call".
            return self._fail("unauthorized", error=f"sign in to {spec['app']} first")
        self.calls[name] = self.calls.get(name, 0) + 1
        fault = self._fault(name)
        if fault:
            self.fault_hits.append({"tool": name, "kind": fault["kind"],
                                    "call": self.calls[name], "turn": self.turn})
        user = self.sessions.get(spec["app"], {}).get("user")
        is_write = spec["kind"] == "write"
        attempt = {"tool": name, "arguments": copy.deepcopy(arguments), "turn": self.turn}
        if fault and fault["kind"] in {"rate_limit", "unavailable"}:
            if is_write:
                self._attempt(attempt, "refused:" + fault["kind"], key, user)
            if fault["kind"] == "rate_limit":
                return self._fail("rate_limited", retryable=True,
                                  retry_after=fault.get("retry_after", 0))
            return self._fail("unavailable", retryable=True)
        if fault and fault["kind"] == "lost_request":
            # Dropped on the way in: nothing was applied and the server never
            # saw the request or its key. The caller only sees a missing reply.
            self.write_attempts.append({**attempt, "outcome": "dropped_before_commit"})
            return self._fail("response_lost", outcome="unknown")
        if key and is_write and key in self.operations:
            known = self.operations[key]
            if (known["tool"], known["arguments"], known["user"]) != (name, arguments, user):
                # The same key for a different request is a client bug; applying
                # either request under it would be wrong.
                self._attempt(attempt, "refused:key_conflict", None, user)
                return self._fail("business_error",
                                  error="this operation key was used for a different request")
            if spec["idempotency_key"] and known["state"] == "executed":
                # Same key, same write: answer with the recorded result, apply nothing.
                self._attempt(attempt, "replayed", None, user)
                return {"ok": True, "value": copy.deepcopy(known["result"]), "replayed": True}
        payload = self._run(name, arguments)
        if not is_write:
            if payload["ok"] and spec["kind"] == "login":
                payload["credential_contract"] = {
                    "subject": f"{arguments['app']}+{arguments['username']}",
                    "scopes": ["account"], "valid_for_seconds": self.session_ttl_s}
            return payload
        if not payload["ok"]:
            self._attempt(attempt, "refused:" + payload["error_code"], key, user)
            return payload
        self.writes.append({"tool": name, "arguments": copy.deepcopy(arguments),
                            "step": self.step})
        self._attempt(attempt, "committed", key, user, result=payload["value"])
        if fault and fault["kind"] == "lost_ack":
            self.write_attempts[-1]["outcome"] = "committed_reply_lost"
            return self._fail("response_lost", outcome="unknown")
        return payload

    def _run(self, name, arguments):
        try:
            return {"ok": True, "value": self._handler(name)(**arguments)}
        except BusinessError as error:
            return self._fail("business_error", error=str(error))
        except Unauthorized as error:
            return self._fail("unauthorized", error=str(error))

    def _attempt(self, attempt, outcome, key, user, result=None):
        self.write_attempts.append({**attempt, "outcome": outcome})
        if not key:
            return
        record = self.operations.get(key)
        if record is None:
            record = self.operations[key] = {
                "tool": attempt["tool"], "arguments": attempt["arguments"], "user": user,
                "app": self.tools[attempt["tool"]]["app"], "state": "not_executed",
                "result": None, "executions": 0}
        if outcome == "committed":
            # A tool that does not deduplicate on the key may apply it again;
            # the record then says so instead of forgetting the first time.
            record.update(state="executed", result=copy.deepcopy(result),
                          executions=record["executions"] + 1)

    # --------------------------------------------------------------- tools
    def _handler(self, name):
        if name.startswith(PAD_PREFIX):
            volume = name[len(PAD_PREFIX):]
            return lambda record: {"archive": volume, "record": record,
                                   "summary": f"record {record} of volume {volume}"}
        return getattr(self, "_" + name)

    def _ref_list_currencies(self):
        return sorted(self.ref["rates"])

    def _ref_get_rate(self, currency):
        if currency not in self.ref["rates"]:
            raise BusinessError(f"no rate for {currency}")
        return {"currency": currency, "rate": self.ref["rates"][currency]}

    def _clock_now(self):
        return {"seconds_since_start": self.now}

    def _auth_login(self, app, username, password):
        if app not in self.login_apps:
            raise BusinessError(f"unknown app {app}")
        self._check_credentials(app, username, password)
        self.logins += 1
        token = f"{app}-session-{self.logins}"
        self.tokens[token] = {"app": app, "user": username, "revoked": False,
                              "expires_at": self.now + self.session_ttl_s}
        self._jar[app] = token
        self.sessions[app] = {"user": username, "session": token}
        return {"app": app, "session": token}

    def _check_credentials(self, app, username, password):
        if username not in self.users or self.users[username] != password:
            raise BusinessError("wrong username or password")

    def _auth_logout(self, app):
        token = (self._presented or {}).get(app) or self._jar.get(app)
        if token in self.tokens:
            self.tokens[token]["revoked"] = True
        self._jar.pop(app, None)
        self.sessions.pop(app, None)
        return {"app": app, "signed_out": True}

    def _account(self, account):
        if account not in self.bank["accounts"]:
            raise BusinessError(f"no account {account}")
        return self.bank["accounts"][account]

    def _own(self, account):
        """The account, if the signed-in bank user may look into it."""
        data = self._account(account)
        user = self.sessions.get("bank", {}).get("user")
        if self.strict_ownership and user is not None and data["owner"] != user:
            raise BusinessError(f"no access to account {account}")
        return data

    def _bank_list_accounts(self):
        user = self.sessions.get("bank", {}).get("user")
        return [{"account": name, "owner": data["owner"]}
                for name, data in sorted(self.bank["accounts"].items())
                if not self.strict_ownership or user is None or data["owner"] == user]

    def _bank_get_balance(self, account):
        return {"account": account, "balance": self._own(account)["balance"]}

    def _bank_list_transfers(self, account):
        self._own(account)
        return [copy.deepcopy(t) for t in self.bank["transfers"]
                if account in (t["source"], t["target"])]

    def _bank_transfer(self, source, target, amount):
        src, dst = self._own(source), self._account(target)
        if amount <= 0:
            raise BusinessError("amount must be a positive number")
        if source == target:
            raise BusinessError("source and target must differ")
        if src["balance"] < amount:
            raise BusinessError(f"insufficient funds in {source}")
        src["balance"] -= amount
        dst["balance"] += amount
        transfer = {"id": f"tr-{len(self.bank['transfers']) + 1}", "source": source,
                    "target": target, "amount": amount}
        self.bank["transfers"].append(transfer)
        return copy.deepcopy(transfer)

    def _item(self, sku):
        if sku not in self.shop["items"]:
            raise BusinessError(f"no item {sku}")
        return self.shop["items"][sku]

    def _shop_list_items(self):
        return [{"sku": sku, "name": data["name"]}
                for sku, data in sorted(self.shop["items"].items())]

    CATEGORY_CODES = {"home": "H-12", "office": "O-07", "general": "G-01"}

    def _shop_get_item(self, sku):
        item = self._item(sku)
        category = item.get("category", "general")
        if self.behavior.get("category_codes"):
            category = self.CATEGORY_CODES[category]
        return {"sku": sku, "name": item["name"], "category": category}

    def _shop_get_price(self, sku):
        price = self._item(sku)["price"]
        if self.behavior.get("price_unit") == "cents":
            return {"sku": sku, "price": int(round(price * 100)), "unit": "cents"}
        return {"sku": sku, "price": price}

    def _shop_get_quote(self, sku, quantity):
        if not isinstance(quantity, int) or isinstance(quantity, bool) or quantity <= 0:
            raise BusinessError("quantity must be a positive integer")
        unit = self._item(sku)["price"] * (0.9 if quantity >= 10 else 1)
        return {"sku": sku, "quantity": quantity, "total": round(unit * quantity, 2)}

    def _shop_get_bulk_price(self, sku):
        return {"sku": sku, "bulk_price": round(self._item(sku)["price"] * 0.9, 2)}

    def _shop_get_stock(self, sku):
        return {"sku": sku, "stock": self._item(sku)["stock"]}

    def _shop_place_order(self, sku, quantity):
        item = self._item(sku)
        if not isinstance(quantity, int) or isinstance(quantity, bool) or quantity <= 0:
            raise BusinessError("quantity must be a positive integer")
        if item["stock"] < quantity:
            raise BusinessError(f"only {item['stock']} of {sku} in stock")
        item["stock"] -= quantity
        order_id = f"ord-{len(self.shop['orders']) + 1}"
        self.shop["orders"][order_id] = {"sku": sku, "quantity": quantity,
                                         "unit_price": item["price"], "status": "placed"}
        self._order_owner[order_id] = self.sessions.get("shop", {}).get("user")
        return {"order_id": order_id, "total": item["price"] * quantity}

    def _my_order(self, order_id):
        """Whether the signed-in shop user may see this order."""
        user = self.sessions.get("shop", {}).get("user")
        return (not self.strict_ownership or user is None
                or self._order_owner.get(order_id, user) == user)

    def _shop_list_orders(self):
        return [{"order_id": oid, **copy.deepcopy(o)}
                for oid, o in sorted(self.shop["orders"].items()) if self._my_order(oid)]

    def _shop_cancel_order(self, order_id):
        order = self.shop["orders"].get(order_id)
        if order is None or not self._my_order(order_id):
            raise BusinessError(f"no order {order_id}")
        if order["status"] == "cancelled":
            raise BusinessError(f"order {order_id} is already cancelled")
        order["status"] = "cancelled"
        return {"order_id": order_id, "status": "cancelled"}

    def _op_status(self, key):
        """What became of the write sent under ``key``.

        ``executed`` and ``not_executed`` are statements about a request the
        server received. A key it never saw is ``unknown``: the request may
        have been sent without a key, or never arrived, and the server cannot
        tell which, so "not executed" would be a guess.
        """
        record = self.operations.get(key)
        if record is None:
            return {"key": key, "state": "unknown"}
        if self.sessions.get(record["app"], {}).get("user") != record["user"]:
            raise Unauthorized(f"sign in to {record['app']} as the sender of this request")
        if record["state"] != "executed":
            return {"key": key, "state": "not_executed"}
        return {"key": key, "state": "executed", "result": copy.deepcopy(record["result"])}

    # -------------------------------------------------------------- oracle
    def snapshot(self):
        return {"bank": copy.deepcopy(self.bank), "shop": copy.deepcopy(self.shop)}

    def _lookup(self, path):
        node = self.snapshot()
        for part in path.split("."):
            node = node[int(part)] if isinstance(node, list) else node[part]
        return node

    def final_state(self):
        """Everything a run can change, in a form that does not depend on the
        order of independent actions: balances, prices and stock, orders, and
        the transfers made during the run (seeded history is left out)."""
        seeded = self._seeded
        moved = sorted(json.dumps({k: t[k] for k in ("source", "target", "amount")},
                                  sort_keys=True)
                       for t in self.bank["transfers"] if not t["id"].startswith(seeded or ("\0",)))
        orders = sorted(json.dumps({k: v for k, v in order.items()}, sort_keys=True)
                        for order in self.shop["orders"].values())
        return {"balances": {name: data["balance"]
                             for name, data in sorted(self.bank["accounts"].items())},
                "items": {sku: {"price": item["price"], "stock": item["stock"]}
                          for sku, item in sorted(self.shop["items"].items())},
                "orders": [json.loads(order) for order in orders],
                "transfers": [json.loads(transfer) for transfer in moved]}

    def _note_truth(self):
        """Record what the current turn's probes read right now, as the server
        itself sees it (no sign-in check, no counters, no faults)."""
        for index, probe in enumerate((self.task.get("overlay") or {}).get("probes", [])):
            if probe["turn"] != self.turn or index in self.truth:
                continue
            self._shadow = True
            try:
                payload = self._run(probe["call"]["name"],
                                    copy.deepcopy(probe["call"]["arguments"]))
                value = payload["value"] if payload["ok"] else None
                for part in probe["field"]:
                    value = value[part]
            except Exception:  # the run left nothing to read (no sign-in, say)
                value = None
            finally:
                self._shadow = False
            self.truth[index] = value

    def oracle(self, answers=None, served=None):
        """Judge the run (see ``exp_core.lab.judge``): final state, committed
        writes, what the agent was handed, and what each answer states. A
        rebuilt benchmark task is also judged on its added part alone."""
        self._note_truth()
        verdict = judge(self.task.get("expected", {}), self.final_state(), self.writes,
                        answers, served, lookup=self._lookup)
        if self.task.get("overlay"):
            verdict["overlay"] = judge_overlay(
                self.task, self.truth, self.write_attempts, self.fault_hits,
                self.calls, answers)
            verdict["overlay"]["original_matches_reference"] = verdict["success"]
        return verdict
