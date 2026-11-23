"""Replay one recorded run under another scheduler policy, without a model.

A run made with no cache is a ground-truth timeline: every call reached the
server, so its recorded result is what the server said at that moment. Feeding
the same calls through a scheduler built from a candidate policy answers, in
seconds, how many requests that policy would have sent and whether any cached
value it returned differs from what the server said.

The replay assumes the agent makes the same calls whatever the policy. That is
an approximation (a model may read more when reads are cheap, and a host that
signs in again by itself removes the agent's own sign-in), so a replay ranks
candidate policies; the chosen one still needs a real confirming run.

Only lab traces are supported: the per-tool contract comes from the task.
"""
import copy
import json
import uuid

from exp_core.access import Access
from exp_core.lab.harness import SERVER, build_runtime, load_tasks
from exp_core.lab.domains import make_world

HIT_KEYS = ("M1_hits", "M2_hits", "M3_hits", "M4_hits", "M4_failure_hits")


def _tool_text(record):
    content = (record.get("response", {}).get("result") or {}).get("content") or [{}]
    return content[0].get("text", "")


def load_run(trace_path):
    """Events of one recorded run, in order.

    ``("advance", step, turn, catalog_version)``; ``("catalog", listing)`` when
    the host listed the tools; ``("call", turn, step, name, arguments, payload,
    result)`` for every tool call the agent made, where ``payload`` is the list
    of the server's raw replies for it, one per attempt (empty when the call
    never left the host).
    """
    events, payloads, task_id, method = [], [], None, None
    for line in open(trace_path, encoding="utf-8"):
        try:
            record = json.loads(line)
        except ValueError:
            continue
        kind = record.get("kind")
        if kind == "mcp" and record.get("method") == "tools/call":
            params = record["request"].get("params", {})
            name, arguments = params.get("name"), params.get("arguments") or {}
            try:
                body = json.loads(_tool_text(record))
            except ValueError:
                continue
            if name == "benchmark_init":
                task_id = arguments.get("task_id")
            elif name == "lab_advance":
                events.append(("advance", arguments["step"], arguments["turn"],
                               body["catalog_version"]))
            elif name == "lab_list_tools":
                events.append(("catalog", body))
            elif name == "invoke":
                payloads.append(body)
        elif kind == "action":
            events.append(("call", record["turn"], record["step"], record["name"],
                           record["arguments"], payloads,
                           record["result"]))
            payloads = []
    return {"task_id": task_id, "events": events}


def _comparable(value):
    """A result without the per-call operation id, as canonical text."""
    if isinstance(value, dict) and "error" in value:
        value = {"error": value["error"]}
    return json.dumps(value, sort_keys=True, default=str)


class _Backend:
    """Answers from the record: the pending call gets its recorded replies."""

    def __init__(self):
        self.requests = 0
        self.unreplayable = 0
        self.pending = None           # (name, arguments, [payload per attempt])
        self.attempt = 0
        self.after_sign_in = None     # () -> payload of the agent's own later retry
        self.host_signed_in = False
        self.listing = None
        self.logins = {}              # (app, username) -> last successful payload

    def invoke(self, name, arguments, key=None):
        self.requests += 1
        if self.pending and (name, arguments) == self.pending[:2]:
            if self.host_signed_in:
                # The host signed in again by itself and is retrying: the server
                # now answers what it answered the agent's own retry in the record.
                later = self.after_sign_in()
                if later is not None:
                    return copy.deepcopy(later)
            payloads = self.pending[2]
            payload = payloads[min(self.attempt, len(payloads) - 1)]
            self.attempt += 1
            return copy.deepcopy(payload)
        if name == "auth_login":
            known = self.logins.get((arguments.get("app"), arguments.get("username")))
            if known is not None:  # the host signing in again on its own
                self.host_signed_in = self.pending is not None
                return copy.deepcopy(known)
        # The record cannot say what the server would have answered here
        # (for example whether a write with a lost reply had been applied).
        self.unreplayable += 1
        return {"ok": False, "error_code": "replay_unknown", "retryable": False,
                "outcome": "unknown", "error": "not in the recorded run"}

    def list_tools(self):
        self.requests += 1
        return copy.deepcopy(self.listing)


def replay(run, policy, tasks=None, seed=0):
    """Replay ``run`` (from ``load_run``) under an arm ``policy``."""
    tasks = tasks or load_tasks()
    contract = make_world(tasks[run["task_id"]]).contract()
    backend = _Backend()
    runtime = build_runtime(policy, seed=seed)
    scheduled = bool(policy.get("scheduler"))
    access = Access(runtime, "lab", f"lab:{run['task_id']}",
                    lambda name, arguments: backend.invoke(name, arguments),
                    invoke_keyed=lambda name, arguments, key: backend.invoke(name, arguments, key),
                    lab_contract=contract)
    out = {"task": run["task_id"], "calls": 0, "stale_hits": 0, "stale_catalogs": 0,
           "served_after_revocation": 0, "divergent": 0, "stale": []}
    turn, version, hits = None, None, lambda: sum(runtime.stats.get(k, 0) for k in HIT_KEYS)
    events, absorbed = run["events"], set()

    def later_retry(index, name, arguments):
        """The agent's own recovery in the record: its sign-in(s) and the retry
        of this call. Once the host has recovered by itself the agent never
        sees the refusal, so those recorded calls would not have been made."""
        skipped, turn_of_call = [], events[index][1]
        for later in range(index + 1, len(events)):
            other = events[later]
            if other[0] != "call":
                continue
            if other[1] != turn_of_call:
                return None  # the agent did not recover within this turn
            if other[3] == "auth_login":
                skipped.append(later)
            elif (other[3], other[4]) == (name, arguments):
                if other[5] and other[5][-1].get("ok"):
                    absorbed.update(skipped + [later])
                    return other[5][-1]
                return None
        return None

    for index, event in enumerate(events):
        if index in absorbed:
            continue
        if event[0] == "advance":
            _, _, new_turn, new_version = event
            if turn is not None and new_turn != turn and scheduled:
                runtime.end_turn()
            turn = new_turn
            if scheduled:
                runtime.advance_step()
                if version is not None and new_version != version:
                    runtime.notify_list_changed(SERVER)
            version = new_version
        elif event[0] == "catalog":
            backend.listing = event[1]
            served = runtime.catalog(SERVER, event[1].get("version", version), backend.list_tools)
            names = lambda listing: sorted(  # noqa: E731
                (tool["name"], tool.get("description")) for tool in listing["tools"])
            if names(served) != names(event[1]):
                out["stale_catalogs"] += 1
        else:
            _, call_turn, _, name, arguments, payload, truth = event
            out["calls"] += 1
            if not payload:
                continue  # answered by the host itself (unknown tool name)
            signing = contract.get(name, {}).get("kind") in {"login", "logout"}
            if name == "auth_login" and payload[-1].get("ok"):
                backend.logins[(arguments.get("app"), arguments.get("username"))] = payload[-1]
            backend.pending, backend.attempt = (name, arguments, payload), 0
            backend.host_signed_in = False
            backend.after_sign_in = lambda i=index, n=name, a=arguments: later_retry(i, n, a)
            before = hits()
            try:
                result = access.call(name, copy.deepcopy(arguments), str(uuid.uuid4()))
            except Exception as error:  # noqa: BLE001 - the agent sees it as a result
                result = {"error": str(error)}
            backend.pending = None
            if signing or backend.host_signed_in:
                continue  # session ids differ by nature; a recovered call is not the record's
            if _comparable(result) != _comparable(truth):
                refused = isinstance(truth, dict) and truth.get("error") == "unauthorized"
                if hits() > before and refused:
                    # The server had dropped the session; the host cannot know
                    # until a request fails, so it still served its cached value.
                    out["served_after_revocation"] += 1
                elif hits() > before:
                    out["stale_hits"] += 1
                    out["stale"].append({"turn": call_turn, "tool": name, "arguments": arguments})
                else:
                    out["divergent"] += 1
    if scheduled and turn is not None:
        runtime.end_turn()
    out.update(requests=backend.requests, unreplayable=backend.unreplayable,
               hits={key: runtime.stats.get(key, 0) for key in HIT_KEYS})
    return out
