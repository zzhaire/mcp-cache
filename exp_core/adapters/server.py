"""Real Streamable HTTP MCP server exposing unmodified benchmark actions."""
import argparse
import copy
import importlib
import inspect
import json
import os
from pathlib import Path
import random
import subprocess
import time
from mcp.server.fastmcp import FastMCP

from exp_core.settings import (
    ROOT, DATA, APPWORLD_PYTHON, APPWORLD_ROOT, TAU2_PYTHON, TAU2_ROOT)


class ResponseCache:
    """Server-side tools/call cache mirroring FastMCP ResponseCachingMiddleware.

    Entries are keyed by name plus arguments for the lifetime of one server
    process. There is no auth or data-epoch invalidation: that asymmetry is
    the point of the rw_fastmcp arm.
    """

    def __init__(self, ttl_seconds=0):
        self.ttl_seconds = ttl_seconds
        self.hits = 0
        self.misses = 0
        self._entries = {}

    @staticmethod
    def key(name, arguments):
        return json.dumps([name, arguments], sort_keys=True, separators=(",", ":"))

    def get(self, name, arguments, now):
        identity = self.key(name, arguments)
        entry = self._entries.get(identity)
        if entry is None:
            self.misses += 1
            return None
        expiry, value = entry
        if now >= expiry:
            del self._entries[identity]
            self.misses += 1
            return None
        self.hits += 1
        return copy.deepcopy(value)

    def put(self, name, arguments, value, now):
        if self.ttl_seconds <= 0:
            return
        self._entries[self.key(name, arguments)] = (
            now + self.ttl_seconds, copy.deepcopy(value))

    @property
    def stats(self):
        return {"hits": self.hits, "misses": self.misses,
                "entries": len(self._entries), "ttl_seconds": self.ttl_seconds}


def is_cacheable(payload):
    """Only successful backend results enter the response cache."""
    return isinstance(payload, dict) and payload.get("ok") is True


class Benchmark:
    def __init__(self):
        self.dataset = None
        self.instances, self.methods = {}, {}
        self.worker = None

    def worker_call(self, command):
        self.worker.stdin.write(json.dumps(command) + "\n")
        self.worker.stdin.flush()
        response = json.loads(self.worker.stdout.readline())
        if not response["ok"]:
            raise RuntimeError(response["error"])
        self.credential_contract = response.get("credential_contract")
        return response["value"]

    def initialize(self, dataset, task_id, run_id):
        self.dataset = dataset
        if dataset == "lab":
            from exp_core.lab.domains import make_world
            from exp_core.lab.harness import load_tasks
            task = load_tasks()[task_id]
            self.world = make_world(task)
            self.latency_ms = task.get("latency_ms", {})
            info = {"turns": task["turns"], "contract": self.world.contract(),
                    "family": task["family"]}
            # A rebuilt benchmark may come with standing instructions (a policy).
            instructions = getattr(getattr(self.world, "domain", None), "instructions", None)
            if instructions:
                info["instructions"] = instructions
            # How the host keeps the conversation (see the runner): compacted
            # history, and which agent of several takes each turn.
            for key in ("context", "agents"):
                if task.get(key):
                    info[key] = task[key]
            return info
        if dataset == "tau2":
            env = dict(os.environ, TAU2_DATA_DIR=str(TAU2_ROOT / "data"))
            self.worker = subprocess.Popen(
                [TAU2_PYTHON, "-m", "exp_core.adapters.tau2_worker"], stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, env=env,
                cwd=ROOT)
            self.info = self.worker_call({"action": "init", "id": task_id})
            return self.info
        if dataset == "appworld":
            env = dict(os.environ, APPWORLD_ROOT=str(APPWORLD_ROOT))
            self.worker = subprocess.Popen(
                [APPWORLD_PYTHON, "-m", "exp_core.adapters.appworld_worker"], stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, env=env,
                cwd=ROOT)
            self.info = self.worker_call({"action": "init", "id": task_id, "run_id": run_id})
            return self.info
        from bfcl_eval.constants.executable_backend_config import (
            CLASS_FILE_PATH_MAPPING, MULTI_TURN_FUNC_DOC_FILE_MAPPING, STATELESS_CLASSES)
        import bfcl_eval
        self.entry = next(x for x in json.loads((ROOT / "data/bfcl-tasks.json").read_text())
                          if x["id"] == task_id)
        docs = []
        for name in self.entry["involved_classes"]:
            instance = getattr(importlib.import_module(CLASS_FILE_PATH_MAPPING[name]), name)()
            if name not in STATELESS_CLASSES:
                instance._load_scenario(copy.deepcopy(self.entry["initial_config"].get(name, {})))
            self.instances[name] = instance
            for method, fn in inspect.getmembers(instance, inspect.ismethod):
                if not method.startswith("_"):
                    self.methods[method] = fn
            path = Path(bfcl_eval.__file__).parent / "data/multi_turn_func_doc" / MULTI_TURN_FUNC_DOC_FILE_MAPPING[name]
            docs.extend(json.loads(line) for line in path.read_text().splitlines())
        self.docs = [x for x in docs if x["name"] not in self.entry.get("excluded_function", [])]
        self.methods = {k: v for k, v in self.methods.items() if k in {x["name"] for x in self.docs}}
        return {"question": self.entry["question"], "tools": self.docs}

    def invoke(self, name, arguments):
        if self.dataset == "tau2":
            return self.worker_call({"action": "call", "name": name, "arguments": arguments})
        if self.dataset == "appworld":
            return self.worker_call({"action": "call", "app": arguments["app"],
                                     "api": arguments["api"], "arguments": arguments["arguments"]})
        if name not in self.methods:
            raise ValueError("Tool is not in this task's exposed catalog")
        return self.methods[name](**arguments)

    def close(self):
        if self.dataset == "lab":
            return
        if self.worker:
            self.worker_call({"action": "close"})
            self.worker.terminate()


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--delay-ms", type=float, default=0)
    parser.add_argument("--jitter-ms", type=float, default=0)
    parser.add_argument("--seed", type=int, default=20261116)
    parser.add_argument("--response-cache-ttl", type=float, default=0,
                        help="Server-side name+args response cache TTL in seconds; "
                             "0 disables the cache")
    parser.add_argument("--inject-transient-once", action="store_true",
                        help="Controlled coverage check only; fail before executing the first invoke")
    parser.add_argument("--inject-uncertain-write", default="",
                        help="Controlled coverage check only; lose the first matching write result")
    return parser


def main():
    args = build_parser().parse_args()
    rng = random.Random(args.seed)
    injected = set()
    benchmark = Benchmark()
    cache = ResponseCache(args.response_cache_ttl)
    server = FastMCP("mcp-benchmark-smoke", host="127.0.0.1", port=args.port,
                     stateless_http=True, json_response=True)

    @server.tool()
    def benchmark_init(dataset: str, task_id: str, run_id: str) -> dict:
        """Harness control only: initialize the benchmark. Not exposed to task model."""
        return benchmark.initialize(dataset, task_id, run_id)

    @server.tool()
    def invoke(name: str, arguments: dict, key: str = "", shadow: bool = False,
               credentials: dict = None) -> dict:
        """Invoke exactly one original benchmark tool, preserving parameters and results.

        Lab server only: ``key`` is the host's operation/idempotency key and
        ``shadow`` marks an unmetered measurement read that leaves no trace.
        """
        if benchmark.dataset == "lab" and shadow:
            return benchmark.world.invoke(name, arguments, shadow=True, credentials=credentials)
        # The lab world has its own virtual clock; nothing there runs on wall time.
        now = benchmark.world.now if benchmark.dataset == "lab" else time.monotonic()
        cached = cache.get(name, arguments, now)
        if cached is not None:
            return cached
        try:
            if args.inject_transient_once and "transient" not in injected:
                injected.add("transient")
                return {"ok": False, "error_code": "injected_transient",
                        "retryable": True, "outcome": "not_executed"}
            delay = max(0, args.delay_ms + rng.uniform(-args.jitter_ms, args.jitter_ms))
            if benchmark.dataset == "lab":
                delay += benchmark.latency_ms.get(name, 0)
            if delay:
                time.sleep(delay / 1000)
            if benchmark.dataset == "lab":
                # The lab world returns the adapter payload itself, faults included.
                result = benchmark.world.invoke(name, arguments, key=key or None,
                                                credentials=credentials)
                if is_cacheable(result):
                    cache.put(name, arguments, result, benchmark.world.now)
                return result
            result = {"ok": True, "value": benchmark.invoke(name, arguments)}
            if name == args.inject_uncertain_write and "uncertain" not in injected:
                injected.add("uncertain")
                return {"ok": False, "error_code": "injected_lost_write_response",
                        "retryable": False, "outcome": "unknown"}
            if getattr(benchmark, "credential_contract", None):
                result["credential_contract"] = benchmark.credential_contract
            if is_cacheable(result):
                cache.put(name, arguments, result, time.monotonic())
            return result
        except Exception as error:
            return {"ok": False, "error": str(error),
                    "error_code": "benchmark_api_error", "retryable": False,
                    "outcome": "unknown"}

    @server.tool()
    def lab_list_tools(credentials: dict = None) -> dict:
        """Lab server: the tool catalog (version, tools, ttlMs, cacheScope)."""
        delay = max(0, args.delay_ms + rng.uniform(-args.jitter_ms, args.jitter_ms))
        if delay:
            time.sleep(delay / 1000)
        return benchmark.world.catalog(credentials)

    @server.tool()
    def lab_advance(step: int, turn: int, credentials: dict = None) -> dict:
        """Harness control only: move the lab world to a decision step of a turn."""
        return benchmark.world.advance(step, turn, credentials)

    @server.tool()
    def lab_note(name: str, credentials: dict = None) -> dict:
        """Harness control only: the agent's call to ``name`` has completed."""
        return benchmark.world.note(name, credentials)

    @server.tool()
    def lab_oracle(answers: list, served: list = None) -> dict:
        """Harness control only: check final state, writes, stated answers and
        the tool results the agent was handed."""
        verdict = benchmark.world.oracle(answers, served)
        world = benchmark.world
        verdict["world"] = {"write_attempts": world.write_attempts,
                            "fault_hits": world.fault_hits,
                            "applied_events": world.applied_events,
                            "unapplied_events": len(world.events), "now": world.now}
        return verdict

    @server.tool()
    def server_cache_stats() -> dict:
        """Harness control only: server-side response cache counters."""
        return cache.stats

    @server.tool()
    def appworld_oracle() -> dict:
        """Harness control only: evaluate final business state."""
        return benchmark.worker_call({"action": "oracle"})

    @server.tool()
    def tau2_oracle() -> dict:
        """Harness control only: official strict telecom state evaluation."""
        return benchmark.worker_call({"action": "oracle"})

    @server.tool()
    def benchmark_close() -> dict:
        """Harness control only: close an initialized environment."""
        benchmark.close()
        return {}

    server.run(transport="streamable-http")


if __name__ == "__main__":
    main()
