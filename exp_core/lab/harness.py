"""Deterministic mechanism check: replay a task's scripted solution through a
method's access stack, in process, with no model and no network.

The script plays a fixed, slightly naive agent: it repeats reads whenever the
user asks again, signs in again and retries once when a call comes back
"unauthorized", and (where the script says so) retries a call that returned
another error. That makes cache and recovery behaviour visible: a stale cache yields a
wrong answer, a blind retry of a lost write yields a duplicate, and the task's
oracle fails. Results say whether a mechanism is correct, not how a real model
would behave.
"""
import copy
import json
import re
import string
import time
import uuid

from exp_core.access import Access
from exp_core.adapters.server import ResponseCache, is_cacheable
from exp_core.lab.factors import DeliveryAudit, degrade_contract
from exp_core.lab.domains import make_world
from exp_core.lab.world import served_entry
from exp_core.runtime import Runtime
from exp_core.scheduler import Scheduler
from exp_core.settings import DATA

TASKS_PATH = DATA / "lab-tasks.json"
SERVER = "lab"


def load_tasks(path=TASKS_PATH):
    """Every lab task by id: the development tasks and, if present, the frozen
    held-out tasks kept in their own file."""
    tasks = {task["id"]: task for task in json.loads(path.read_text(encoding="utf-8"))}
    heldout = path.with_name("lab-heldout-tasks.json")
    if path == TASKS_PATH and heldout.is_file():
        tasks.update({task["id"]: task
                      for task in json.loads(heldout.read_text(encoding="utf-8"))})
    if path == TASKS_PATH:
        # Public benchmarks rebuilt on this server layer (data/recon).
        for extra in sorted([*path.parent.glob("recon-*-tasks.json"),
                             *path.parent.glob("lab-long-tasks.json"),
                             *path.parent.glob("lab-gain-tasks.json")]):
            tasks.update({task["id"]: task
                          for task in json.loads(extra.read_text(encoding="utf-8"))})
    return tasks


def _is_error(value):
    return isinstance(value, dict) and "error" in value


def _condition(expression, results):
    """Evaluate ``alias[key] <op> literal`` style conditions from the task file."""
    python = re.sub(r"\[([A-Za-z_]\w*)\]", r"['\1']", expression)
    try:
        return bool(eval(python, {"__builtins__": {}}, dict(results)))  # noqa: S307
    except Exception:
        return False


class _Lenient(string.Formatter):
    """Renders a field that is missing from the results as ``<none>``."""

    def get_field(self, field_name, args, kwargs):
        try:
            return super().get_field(field_name, args, kwargs)
        except (KeyError, IndexError, TypeError, AttributeError):
            return "<none>", field_name


def render_answer(template, results):
    """The scripted agent's answer for a turn. A value the script could not
    obtain shows up as ``<none>``, so the oracle sees exactly what is missing."""
    return _Lenient().vformat(template, (), results)


class TurnScript:
    """One user turn of the scripted agent, shared by the in-process replay and
    the scripted model that drives the real server.

    Besides the task's own steps it applies one general rule, the way any agent
    would: a call answered "unauthorized" is followed by a sign-in to that app
    (with the credentials the agent used before) and one retry of the call. A
    turn marked ``no_recover`` skips the rule, for tasks that ask what happens
    without signing in.
    """

    def __init__(self, turn, contract, logins):
        self.turn, self.contract, self.logins = turn, contract, logins
        self.queue = [(calls, False) for calls in turn["steps"]]
        self.results = {}

    def next_calls(self):
        """Calls of the next step after applying conditions, or None when done."""
        if not self.queue:
            return None
        calls, self._recovering = self.queue.pop(0)
        return [call for call in calls
                if ("if" not in call or _condition(call["if"], self.results))
                and ("if_error" not in call or _is_error(self.results.get(call["if_error"])))]

    def record(self, calls, results):
        """Store a step's results; schedule a sign-in and retry if a call was refused."""
        refused = []
        for call, result in zip(calls, results):
            if call.get("as"):
                self.results[call["as"]] = result
            if call["name"] == "auth_login" and not _is_error(result):
                self.logins[call["arguments"]["app"]] = {
                    "name": call["name"], "arguments": copy.deepcopy(call["arguments"])}
            if _is_error(result) and result["error"] == "unauthorized":
                refused.append(call)
        if not refused or self._recovering or self.turn.get("no_recover"):
            return
        apps = []
        for call in refused:
            app = (self.contract.get(call["name"]) or {}).get("principal")
            if app in self.logins and app not in apps:
                apps.append(app)
        if not apps:
            return
        retries = [{key: copy.deepcopy(value) for key, value in call.items()
                    if key in {"name", "arguments", "as"}} for call in refused]
        self.queue[0:0] = [([self.logins[app] for app in apps], True), (retries, True)]

    def answer(self):
        return render_answer(self.turn["answer"], self.results)


class Backend:
    """The server side of one run: the world, the optional response cache and
    counters for what actually reached it. All time is the world's virtual time."""

    def __init__(self, task, server_cache_ttl=0):
        self.world = make_world(task)
        self.cache = ResponseCache(server_cache_ttl)
        self.requests = 0          # metered host -> server requests
        self.catalog_requests = 0  # of which: listing the tools
        self.executions = 0        # requests that reached the world

    def invoke(self, name, arguments, key=None, credentials=None):
        self.requests += 1
        cached = self.cache.get(name, arguments, self.world.now)
        if cached is not None:
            return cached
        self.executions += 1
        payload = self.world.invoke(name, arguments, key=key, credentials=credentials)
        if is_cacheable(payload):
            self.cache.put(name, arguments, payload, self.world.now)
        return payload

    def shadow(self, name, arguments, credentials=None):
        return self.world.invoke(name, arguments, shadow=True, credentials=credentials)

    def list_tools(self, credentials=None):
        self.requests += 1
        self.catalog_requests += 1
        return self.world.catalog(credentials)


def build_runtime(policy, seed=0, sleep=lambda seconds: None, clock=None):
    """The host runtime of an arm. ``clock`` is the experiment's virtual clock;
    every expiry on the host is measured on it."""
    clock = {"clock": clock} if clock else {}
    if policy.get("scheduler"):
        return Scheduler(policy["scheduler"], disabled=policy["disabled_modules"],
                         m4_failure_only=policy["m4_failure_only"], seed=seed, sleep=sleep,
                         **clock)
    return Runtime(disabled=policy["disabled_modules"], ttl=policy["ttl_seconds"],
                   m4_failure_only=policy["m4_failure_only"], **clock)


def run_gold(task, policy, shadow_verify=False, seed=0, listing="per_turn",
             contract_level="full"):
    """Replay ``task["gold"]`` under an arm policy; return oracle and counters.

    ``listing``: the agent framework asks the host for the tool list at the
    start of every user turn ("per_turn") or once ("per_connection"); in both
    cases again whenever the server announces a change.
    ``contract_level``: how much of the tool contract the host is given.
    """
    backend = Backend(task, server_cache_ttl=policy.get("server_cache_ttl", 0))
    world = backend.world
    runtime = build_runtime(policy, seed=seed, clock=lambda: world.now)
    scheduled = bool(policy.get("scheduler"))
    scope = f"lab:{task['id']}"
    truth = world.contract()
    tokens = lambda: dict(access.tokens)  # noqa: E731 - the host's current sign-ins
    access = Access(runtime, "lab", scope,
                    lambda name, arguments: backend.invoke(name, arguments,
                                                           credentials=tokens()),
                    invoke_keyed=lambda name, arguments, key: backend.invoke(
                        name, arguments, key, credentials=tokens()),
                    shadow_invoke=(lambda name, arguments: backend.shadow(
                        name, arguments, credentials=tokens()))
                    if (shadow_verify and scheduled) else None,
                    lab_contract=degrade_contract(truth, contract_level))
    audit = DeliveryAudit(lambda name, arguments: backend.shadow(
        name, arguments, credentials=tokens()), truth)
    answers, errors, logical_calls, step = [], [], 0, 0
    served = []
    catalog_version, logins = None, {}
    for turn_index, turn in enumerate(task["gold"]):
        script = TurnScript(turn, truth, logins)
        first = True
        served.append([])
        while True:
            # One decision step of the agent, including the last one of the
            # turn in which it only answers: time passes for that one too.
            step += 1
            state = world.advance(step, turn_index, credentials=tokens())
            if scheduled:
                runtime.advance_step()
            changed = state["catalog_version"] != catalog_version
            if changed and scheduled and catalog_version is not None:
                runtime.notify_list_changed(SERVER)  # MCP list_changed
            wanted = first if listing == "per_turn" else catalog_version is None
            if changed or wanted:
                catalog_version = state["catalog_version"]
                tools = {tool["name"] for tool in runtime.catalog(
                    SERVER, catalog_version,
                    lambda: backend.list_tools(credentials=tokens()))["tools"]}
            first = False
            calls = script.next_calls()
            if calls is None:
                break
            results = []
            for call in calls:
                logical_calls += 1
                before = backend.requests
                if call["name"] not in tools:
                    result = {"error": f"unknown tool {call['name']}"}
                else:
                    try:
                        result = access.call(call["name"], copy.deepcopy(call["arguments"]),
                                             str(uuid.uuid4()))
                    except Exception as error:
                        result = {"error": str(error)}
                if _is_error(result):
                    errors.append({"turn": turn_index, "tool": call["name"],
                                   "error": result["error"]})
                audit.delivered(call["name"], call["arguments"], result,
                                reached_server=backend.requests > before)
                world.note(call["name"], credentials=tokens())
                results.append(result)
                served[-1].append(served_entry(call["name"], call["arguments"], result))
            script.record(calls, results)
        answers.append(script.answer())
        if scheduled:
            runtime.end_turn()
    stats = runtime.stats
    checks = stats.get("shadow_checks", 0)
    return {
        "task": task["id"], "oracle": world.oracle(answers, served), "answers": answers,
        "errors": errors, "logical_calls": logical_calls,
        "requests": backend.requests, "executions": backend.executions,
        "catalog_requests": backend.catalog_requests,
        "other_requests": backend.requests - backend.catalog_requests,
        "delivery": dict(audit.stats), "stale_deliveries": list(audit.stale),
        "hits": {key: stats[key] for key in ("M1_hits", "M2_hits", "M3_hits", "M4_hits",
                                             "M4_failure_hits")},
        "retries": stats["retries"], "stats": dict(stats),
        "stale_hit_rate": (stats["shadow_stale"] / checks) if checks else None,
        "writes": copy.deepcopy(world.writes),
        "write_attempts": copy.deepcopy(world.write_attempts),
        "fault_hits": copy.deepcopy(world.fault_hits),
        "final_state": world.final_state(),
    }
