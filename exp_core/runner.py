"""Bounded task agent using a real MCP HTTP connection and benchmark oracles."""
import argparse
import hashlib
import json
import keyword
import os
import re
from pathlib import Path
import socket
import subprocess
import sys
import time
import traceback
import uuid
import requests
from exp_core.model import generate
from exp_core.settings import MODEL_BACKEND
from exp_core.runtime import Runtime, failure_result
from exp_core.access import Access
from exp_core.redaction import redact
from exp_core.methods import ARM_NAMES, METHOD_INFO, policy_sha256, resolve_arm
from exp_core.scheduler import Scheduler
from exp_core.lab.factors import DeliveryAudit, degrade_contract
from exp_core.lab.world import served_entry
from exp_core.scheduler_policy import apply_to_arm
from exp_core.settings import CONTRACTS, DATA, ROOT, RESULTS, SCRATCH
from ours.publication import (collect_operations, collect_references,
                              ahead_item, choose_prefetch, collect_replacements,
                              dedupe_result, version_of,
                              reference_large_repeat)

READS = {"cat", "diff", "du", "find", "grep", "ls", "pwd", "tail", "wc"}


def code_sha256(root=ROOT):
    """Hash all executable modules and contracts, including method definitions."""
    root = Path(root)
    paths = [root / "run", root / "pyproject.toml"]
    for package in ("exp_core", "related_work", "ours", "data"):
        paths.extend((root / package).rglob("*.py"))
    paths.extend((root / "exp_core" / "contracts").glob("*.json"))
    paths.extend((root / "ours").glob("*.schema.json"))
    digest = hashlib.sha256()
    for path in sorted(set(paths)):
        digest.update(str(path.relative_to(root)).encode() + b"\0")
        digest.update(path.read_bytes() + b"\0")
    return digest.hexdigest()


APP_TOOLS = [{
    "name": "api_call",
    "description": "Call one original AppWorld API. First explore api_docs. Example: "
        '{"app":"api_docs","api":"show_api_descriptions","arguments":{"app_name":"spotify"}}. '
        "Use show_app_descriptions with {} and show_api_doc with app_name, api_name. "
        "Get supervisor profile/passwords via supervisor APIs. "
        "Complete task via supervisor.complete_task(answer=...).",
    "parameters": {"type": "object", "properties": {
        "app": {"type": "string"}, "api": {"type": "string"},
        "arguments": {"type": "object"}}, "required": ["app", "api", "arguments"]}}]


class MCP:
    def __init__(self, port, trace):
        self.url = f"http://127.0.0.1:{port}/mcp"
        self.trace = trace
        self.session = requests.Session()
        self.session.headers.update({"Accept": "application/json, text/event-stream"})
        self.metrics = {"mcp_requests": 0, "mcp_requests_catalog": 0, "request_body_bytes": 0,
                        "response_body_bytes": 0, "mcp_wait_s": 0}
        self.request_id = 0

    def rpc(self, method, params=None, control=False):
        self.request_id += 1
        body = {"jsonrpc": "2.0", "id": self.request_id, "method": method, "params": params or {}}
        data = json.dumps(body).encode()
        start = time.monotonic()
        response = self.session.post(self.url, data=data,
                                     headers={"Content-Type": "application/json"}, timeout=120)
        elapsed = time.monotonic() - start
        response.raise_for_status()
        value = response.json()
        with self.trace.open("a") as output:
            output.write(json.dumps({"kind": "mcp", "method": method, "control": control,
                                    "elapsed_s": elapsed, "request_body_bytes": len(data),
                                    "response_body_bytes": len(response.content),
                                    "request": redact(body), "response": redact(value)}) + "\n")
        if not control:
            self.metrics["mcp_requests"] += 1
            if method == "tools/call" and (params or {}).get("name") == "lab_list_tools":
                self.metrics["mcp_requests_catalog"] += 1
            self.metrics["request_body_bytes"] += len(data)
            self.metrics["response_body_bytes"] += len(response.content)
            self.metrics["mcp_wait_s"] += elapsed
        if "error" in value:
            raise RuntimeError(str(value["error"]))
        return value["result"]

    def initialize(self):
        result = self.rpc("initialize", {
            "protocolVersion": "2025-11-25", "capabilities": {},
            "clientInfo": {"name": "mcp-smoke", "version": "0.1"}}, control=True)
        self.session.headers["MCP-Protocol-Version"] = result["protocolVersion"]
        response = self.session.post(self.url, json={
            "jsonrpc": "2.0", "method": "notifications/initialized"},
            headers={"Content-Type": "application/json"}, timeout=30)
        response.raise_for_status()

    def tool(self, name, arguments, control=False):
        result = self.rpc("tools/call", {"name": name, "arguments": arguments}, control)
        if result.get("isError"):
            raise RuntimeError(str(result.get("content")))
        if "structuredContent" in result:
            return result["structuredContent"]
        return json.loads(next(x["text"] for x in result["content"] if x["type"] == "text"))


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def wait_server(process, port):
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("MCP server failed to start")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=.1):
                return
        except OSError:
            time.sleep(.1)
    raise RuntimeError("MCP server startup timed out")


def readonly(dataset, name, arguments):
    if dataset == "bfcl":
        return name in READS
    return (arguments.get("app") == "api_docs" and arguments.get("api") in {
        "show_app_descriptions", "show_api_descriptions", "show_api_doc", "search_api_docs"})


def evaluate_bfcl(decoded, truth, entry, run_id):
    from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_checker import multi_turn_checker
    return multi_turn_checker(decoded, truth["ground_truth"], entry,
                              "multi_turn_base", "smoke_" + run_id)


def resolve_policy(method, override=None):
    """Resolve a method name to its executable policy; explicit overrides win."""
    try:
        policy = resolve_arm(method)
    except KeyError:
        if override is None:
            raise
        # Legacy scientist candidates use a free-form label; keep the old
        # full M1–M4 baseline and apply only the candidate's explicit fields.
        policy = resolve_arm("rw_merchant")
        policy["arm"] = method
    if override is not None:
        policy.update(override)
    # A scheduler spec (preset name, file or dict) becomes a full validated policy.
    return apply_to_arm(policy)


def active_time(task_time_s, queue_wait_s):
    """Task time with model queue wait removed; never negative."""
    return max(0.0, float(task_time_s) - float(queue_wait_s or 0))


# The Claude CLI attaches notes about the machine it runs on (today's date, the
# signed-in account's e-mail address, the working directory) to every call and
# has no switch for it. A dataset has its own dates and identities, so every
# arm is told, in the same words, that those notes are not part of the task.
RELAY_NOTE = ("" if MODEL_BACKEND != "claude" else
              "The program that relays this conversation may attach notes about its "
              "own machine (a current date, an account e-mail address, a working "
              "directory). They are not part of the task: never use them as today's "
              "date, as the user's identity or in a tool argument. Dates, names and "
              "addresses come only from the user's messages and from tool results. ")


# Every arm is told not to repeat a read whose result the conversation already
# shows. MCP_NO_REUSE_HINT=1 leaves that sentence out, to see how much of an
# agent's reuse of a value it read earlier comes from being told so.
REUSE_HINT = ("" if os.environ.get("MCP_NO_REUSE_HINT") == "1" else
              "If a tool result for the same tool and arguments is already present "
              "in this conversation, do not call it again unless you have evidence "
              "the state changed or the user explicitly asks you to re-confirm that "
              "exact read. ")
COMPACTED = {"role": "system", "content":
             "Earlier turns are kept as question and answer only; their tool results "
             "are no longer in this conversation."}


def protocol_text(dataset, supervisor=None):
    """Benchmark protocol sentences shared by every arm."""
    if dataset == "appworld":
        return (
            " This is an AppWorld task. Read the task with api_call "
            "app=supervisor api=show_active_task. Most app APIs need a "
            "token: read supervisor.show_account_passwords, then call "
            "api_call app=<app> api=login with username and password to get "
            "access_token for that app. When every requirement is met, call "
            "api_call app=supervisor api=complete_task, with answer only if "
            "the task asks a question; do not set done before complete_task "
            "reports the task is complete.")
    if dataset == "bfcl":
        return (
            " This benchmark requires at least one real tool call in every "
            "original user turn; a turn with no tool call is invalid even when "
            "the answer looks derivable from earlier results.")
    return ""


def unknown_tool_result(name, tools):
    """Error payload for a tool name outside the exposed catalog."""
    return {"error": f"Unknown tool name '{name}'. Valid tool names: "
                     + ", ".join(tool["name"] for tool in tools)}


_TOOL_FAILURE_CLAIM = re.compile(
    r"\bunsupported\b|\bnot\s+supported\b|\breturned\s+(?:an?\s+)?['\"]?error\b"
    r"|\bfailed\s+with\b|\brejected\s+the\s+request\b"
    r"|\btools?\s+(?:are|were|is|was)\s+unavailable\b", re.IGNORECASE)


def claims_tool_failure(answer):
    """True when the answer attributes an outcome to tool failure."""
    return bool(_TOOL_FAILURE_CLAIM.search(answer))


def done_correction(dataset, turn_index, original_turn_count, turn_calls, answer,
                    executed_ok, executed_failed, unknown_attempts,
                    publish_operations, grounding_corrections, protocol_corrections):
    """Correction a done answer must receive before its turn may end."""
    if (publish_operations and not executed_failed and not unknown_attempts
            and claims_tool_failure(answer) and grounding_corrections < 2):
        names = ", ".join(sorted(executed_ok)) or "(none)"
        return "grounding", (
            "Runtime execution log check: no tool call in this run returned an "
            "error, and no tool name was rejected. Tools that returned results: "
            f"{names}. Do not report a tool failure that did not occur. Continue "
            "the task with the listed tools, or finish without claiming failures.")
    if (dataset == "bfcl" and turn_index < original_turn_count and not turn_calls
            and protocol_corrections < 1):
        return "bfcl", (
            "This benchmark requires at least one real tool call in every original "
            "user turn. Answering without any tool call makes this turn invalid. "
            "Call the tool needed for this turn now, then set done=true.")
    return None, ""


def run_task(dataset, task_id, model, method, output_dir, max_steps=16, policy=None,
             delay_ms=0, jitter_ms=0, seed=20261116, case_id=None, scenario=None,
             augmented=False, case_split=None, augmentation=None,
             run_config_sha256=None, dataset_sha256=None, suite_sha256=None,
             lab_options=None):
    # Lab experiment factors, identical for every method: how often the agent
    # framework asks for the tool list, how much of the contract hosts are
    # given, and whether deliveries are audited against the server.
    lab_options = {"listing": "per_turn", "contract": "full", "audit": False,
                   **(lab_options or {})}
    virtual = {"now": 0}  # the lab world's clock, refreshed at every decision step
    lab_clock = {"clock": lambda: virtual["now"]} if dataset == "lab" else {}
    run_id = uuid.uuid4().hex
    output_dir.mkdir(parents=True, exist_ok=True)
    trace = output_dir / f"{run_id}.jsonl"
    arm_policy = resolve_policy(method, policy)
    row = {"run_id": run_id, "case_id": case_id, "dataset": dataset,
           "task_id": task_id, "scenario": scenario, "augmented": augmented,
           "model": model, "method": method, "arm": arm_policy["arm"],
           "split": None, "status": "running",
           "model_calls": 0, "input_tokens": 0, "output_tokens": 0,
           "cached_input_tokens": 0, "reasoning_output_tokens": 0,
           "task_success": None, "transport": "localhost Streamable HTTP",
           "model_backend": f"{MODEL_BACKEND} subprocess (includes CLI system context)",
           "max_steps_per_turn": max_steps, "policy": arm_policy,
           "delay_ms": delay_ms, "jitter_ms": jitter_ms, "seed": seed,
           "logical_tool_calls": 0,
           "publish_events": 0, "publish_bytes": 0, "publish_results_count": 0,
           "publish_operations_count": 0, "publish_empty_state_events": 0,
           "grounding_corrections": 0, "protocol_corrections": 0,
           "published_results_used": 0,
           "published_only_steps": 0, "published_cache_hits": 0,
           "exact_repeat_calls": 0,
           "context_dedup_saved_bytes": 0, "context_dedup_events": 0,
           "large_repeat_saved_bytes": 0, "large_repeat_events": 0,
           "task_time_s": None, "active_time_s": None,
           "api_inference_requests": None,
           "inference_observability": "CLI completed turns only; underlying API requests unobserved"}
    row["run_config_sha256"] = run_config_sha256
    row["dataset_sha256"] = dataset_sha256
    row["suite_sha256"] = suite_sha256
    row["method_info"] = METHOD_INFO.get(arm_policy["arm"], {
        "reproduction_status": "development_candidate"})
    row["policy_sha256"] = policy_sha256(arm_policy)
    row["augmented"] = bool(augmentation) or augmented
    if augmentation:
        row["augmentation_sha256"] = hashlib.sha256(augmentation.encode()).hexdigest()
    row["runtime_source_sha256"] = hashlib.sha256((ROOT / "exp_core/runtime.py").read_bytes()).hexdigest()
    row["access_contract_sha256"] = hashlib.sha256((CONTRACTS / "access.json").read_bytes()).hexdigest()
    row["execution_source_sha256"] = row["code_sha256"] = code_sha256(ROOT)
    if case_split is not None:
        row["split"] = case_split
    else:
        manifest = json.loads((DATA / "manifest.json").read_text())
        row["split"] = "dev" if task_id in manifest["datasets"][dataset]["dev"] else "heldout"
    scheduler_policy = arm_policy.get("scheduler")
    if scheduler_policy:
        runtime = Scheduler(scheduler_policy, disabled=arm_policy["disabled_modules"],
                            m4_failure_only=arm_policy["m4_failure_only"], seed=seed,
                            **lab_clock)
        row["scheduler_preset"] = scheduler_policy.get("preset")
        row["scheduler_policy_sha256"] = policy_sha256(scheduler_policy)
        row["scheduler_source_sha256"] = hashlib.sha256(
            (ROOT / "exp_core/scheduler.py").read_bytes()).hexdigest()
        publish_cfg = scheduler_policy["publish"]
        reference_limit = min(3, publish_cfg["max_items"])
        operation_limit = publish_cfg["max_items"]
        publish_mode = publish_cfg["mode"]
        publish_select = publish_cfg["select"]
        prefetch_cfg = scheduler_policy["prefetch"]
        full_max_bytes = publish_cfg["full_max_bytes"]
        ops_first = publish_cfg["ops_first"]
        failure_form = publish_cfg["failure_form"]
    else:
        runtime = Runtime(disabled=arm_policy["disabled_modules"],
                          ttl=arm_policy["ttl_seconds"],
                          m4_failure_only=arm_policy["m4_failure_only"], **lab_clock)
        reference_limit, operation_limit = 3, 20
        publish_mode, full_max_bytes, ops_first = "ref_only", 0, False
        publish_select = "current_turn"
        prefetch_cfg = {"enabled": False}
        failure_form = "verbatim"
    # Evaluation switch: compare every cache hit with the backend through an
    # unmetered control call. It never changes what the agent sees.
    shadow_verify = bool(scheduler_policy and arm_policy.get("eval_shadow_verify"))
    row["eval_shadow_verify"] = shadow_verify
    published_identities, used_published, seen_results = set(), set(), set()
    # Large results are handed to the model once (scheduler arms only).
    seen_large = {}
    large_repeat_min_bytes = ((arm_policy.get("scheduler") or {}).get("publish") or {}).get(
        "large_repeat_min_bytes", 0)
    visible_identities, called_identities = set(), set()
    # Version of each read result the conversation shows, and every read the
    # agent has made (for reading ahead): identity -> name, arguments, turns.
    visible_versions, read_history = {}, {}
    row["prefetch_reads"] = row["prefetch_requests"] = row["prefetch_published"] = 0
    # Ordered most-recent-first; the augmentation turn names up to three.
    recent_reads = []
    port = free_port()
    SCRATCH.mkdir(parents=True, exist_ok=True)
    server_log = (SCRATCH / f"server-{run_id}.log").open("w")
    process = subprocess.Popen([sys.executable, "-m", "exp_core.adapters.server", "--port", str(port),
                               "--delay-ms", str(delay_ms), "--jitter-ms", str(jitter_ms),
                               "--seed", str(seed),
                               "--response-cache-ttl", str(arm_policy["server_cache_ttl"])],
                               stdout=server_log, stderr=server_log, cwd=ROOT)
    start = time.monotonic()
    client = MCP(port, trace)
    try:
        wait_server(process, port)
        client.initialize()
        info = client.tool("benchmark_init", {
            "dataset": dataset, "task_id": task_id, "run_id": run_id}, control=True)
        setup_end = time.monotonic()
        lab = dataset == "lab"
        if lab:
            # The lab server's catalog is listed at the start of every user turn,
            # as a host does at the start of every agent run (see the turn loop).
            tools, lab_catalog_version, lab_step = [], None, 0
        else:
            # Both policies use the normal one-time catalog fetch; no artificial re-listing.
            runtime.catalog(client.url, "adapter-v1", lambda: client.rpc("tools/list"))
            tools = APP_TOOLS if dataset == "appworld" else info["tools"]
        scope = f"{dataset}:{task_id}:{run_id}"
        shadow_arguments = {"shadow": True} if lab else {}

        def presented():
            # Lab: the host's sign-in tokens travel with every request.
            return {"credentials": dict(access.tokens)} if lab else {}

        def shadow_call(name, arguments):
            return client.tool("invoke", {"name": name, "arguments": arguments,
                                          **shadow_arguments, **presented()}, control=True)
        access = Access(runtime, dataset, scope,
                        lambda name, arguments: client.tool("invoke", {
                            "name": name, "arguments": arguments, **presented()}),
                        read_contract=info.get("readonly"),
                        invoke_keyed=(lambda name, arguments, key: client.tool("invoke", {
                            "name": name, "arguments": arguments, "key": key,
                            **presented()}))
                        if lab else None,
                        lab_contract=degrade_contract(info["contract"], lab_options["contract"])
                        if lab else None,
                        shadow_invoke=shadow_call if shadow_verify else None)
        audit = (DeliveryAudit(shadow_call, info["contract"])
                 if lab and lab_options["audit"] else None)
        row["lab_options"] = lab_options if lab else None
        turn_answers, lab_served = [], []
        turns = [[{"role": "user", "content": text}] for text in info["turns"]] if lab else \
            info["question"] if dataset == "bfcl" else [[{"role": "user", "content":
            info["instruction"] + ("\nSupervisor: " + json.dumps(info["supervisor"])
                                   if dataset == "appworld" else
                                   "\nUse JSON done=true in place of the done tool.")}]]
        if lab and info.get("instructions"):
            # Standing instructions of a rebuilt benchmark precede the first turn.
            turns[0] = [{"role": "user", "content":
                         info["instructions"] + "\n\n" + turns[0][0]["content"]}]
        original_turn_count = len(turns)
        if augmentation:
            # The extra user turn follows every original turn for all datasets.
            # The official checkers read only the original ground-truth turns or
            # the final environment state, so this turn cannot rewrite the task.
            turns = list(turns) + [[{"role": "user", "content": augmentation}]]
        if dataset == "tau2":
            row["benchmark_mode"] = info["mode"]
            row["executor"] = info["executor"]
            row["initial_oracle"] = client.tool("tau2_oracle", {}, control=True)
        # Benchmark protocol, identical for every arm: AppWorld defines
        # completion through supervisor.complete_task and requires per-app
        # login. Without these two protocol steps no AppWorld case can pass,
        # so stating them is not a hint about the task itself. BFCL requires a
        # real call in every original turn, which its checker enforces.
        protocol = protocol_text(dataset)
        executed_ok, executed_failed, unknown_attempts = set(), set(), set()
        grounding_corrections = 0
        messages, decoded = [], []
        last_block = ""
        # Lab tasks may say how the conversation is kept. "answers_only": when a
        # turn ends its tool results leave the context and only the question
        # and the answer stay, as hosts do to keep long tasks within the
        # window. "agents": several agents take turns through this one host;
        # each sees only its own conversation. Both apply to every arm alike;
        # what the host itself remembers (its caches) is the arm's business.
        compact = lab and info.get("context") == "answers_only"
        agents = info.get("agents") if lab else None
        histories, last_blocks, speaker = {}, {}, None
        for turn_index, question in enumerate(turns):
            if compact or agents:
                last_blocks[speaker] = last_block
                speaker = agents[turn_index] if agents else None
                messages = histories.setdefault(speaker, [])
                # Nothing shown earlier counts as visible to this decision maker.
                visible_identities.clear()
                visible_versions.clear()
                seen_results.clear()
                seen_large.clear()
                # An agent's own conversation still holds the last runtime block
                # it was given: when nothing in it is current any more, that
                # agent has to be told. A compacted history holds no block.
                last_block = "" if compact else last_blocks.get(speaker, "")
            turn_start = len(messages)
            if compact and messages and messages[0] is not COMPACTED:
                messages.insert(0, COMPACTED)
                turn_start += 1
            messages.extend(question)
            turn_calls = []
            # Design 4.3: only reads made in the current user turn are offered
            # before a decision, and a repeat within the turn gets a notice.
            recent_reads, turn_identities = [], set()
            # Reads the host made ahead in this turn and the agent has not
            # asked for since; ``wrote`` marks a step that changed something.
            ahead_identities, wrote = set(), False
            lab_served.append([])
            protocol_corrections = 0
            turn_answer = ""
            for step in range(max_steps):
                if scheduler_policy:
                    runtime.advance_step()
                if lab:
                    # World time follows the agent: events fire at turn starts.
                    lab_step += 1
                    world = client.tool("lab_advance", {"step": lab_step, "turn": turn_index,
                                                        **presented()}, control=True)
                    virtual["now"] = world["now"]
                    changed = world["catalog_version"] != lab_catalog_version
                    if changed and lab_catalog_version is not None and scheduler_policy:
                        runtime.notify_list_changed("lab")  # MCP list_changed
                    wanted = (step == 0 if lab_options["listing"] == "per_turn"
                              else lab_catalog_version is None)
                    if changed or wanted:
                        lab_catalog_version = world["catalog_version"]
                        listing = runtime.catalog(
                            "lab", lab_catalog_version,
                            lambda: client.tool("lab_list_tools", presented()))
                        tools = [{"name": tool["name"], "description": tool["description"],
                                  "parameters": tool["parameters"]}
                                 for tool in listing["tools"]]
                preamble = (
                    "You are a task agent. Choose only tools described below. "
                    "Never use any built-in CLI tools, Shell, filesystem or web. "
                    "The tools below are not functions you can invoke yourself: the "
                    "only way to use one is to list it in the calls array of your "
                    "JSON reply, and its result arrives in the next message. "
                    "Return JSON {calls:[{name,arguments}],done,answer}. "
                    "Reply with one bare JSON object and never wrap it in a code fence. "
                    "Only the tool results shown in this conversation are real; do not "
                    "assume a tool failed unless its own result reports an error. "
                    "Every call uses literal JSON arguments. You may call several tools "
                    "only when their arguments are already known. Do not invent results. "
                    + REUSE_HINT +
                    "Messages with role 'runtime' come from the host, not from "
                    "the user: the latest one replaces all earlier ones. Read results listed "
                    "there are current for the published state; use their values instead of "
                    "calling again when they answer the question, and call the tool when a "
                    "value you need is not included. Still perform every action the user "
                    "requested. " + RELAY_NOTE +
                    "Mark done after completing this user turn. Available tools:\n" +
                    json.dumps(tools) + protocol + "\nConversation:\n")
                block = ""
                published = []
                step_published = set()
                published_bytes = 0
                operations = []
                budget = arm_policy["publish_budget_bytes"]

                def gather_operations(remaining):
                    return collect_operations(runtime, limit=operation_limit,
                                              budget_bytes=max(0, remaining),
                                              failure_form=failure_form)

                if ops_first and arm_policy["publish_operations"]:
                    operations = gather_operations(budget)
                    budget -= sum(len(json.dumps(item, default=str).encode())
                                  for item in operations)
                fresh = {}
                if (prefetch_cfg["enabled"] and arm_policy["publish_read_results"]
                        and publish_select == "replaces_call"
                        and (step == 0 or (wrote and prefetch_cfg["after_write"]))):
                    # Read ahead: repeat reads the agent made before and is about
                    # to need again, so this decision can be the answer.
                    for identity in choose_prefetch(
                            read_history, question[0]["content"], turn_index,
                            lambda key: runtime.valid_read(key)[0],
                            prefetch_cfg["max_per_decision"]):
                        record = read_history[identity]
                        before = client.metrics["mcp_requests"]
                        try:
                            value = access.call(record["name"], record["arguments"],
                                                str(uuid.uuid4()))
                        except Exception as error:
                            value = failure_result(error, "")
                        row["prefetch_reads"] += 1
                        row["prefetch_requests"] += client.metrics["mcp_requests"] - before
                        if isinstance(value, dict) and "error" in value:
                            continue
                        fresh[identity] = value
                        record["fetched"] = turn_index
                    ahead_identities.update(fresh)
                wrote = False
                if arm_policy["publish_read_results"]:
                    # Scope, auth and data validity and TTL are re-checked here.
                    if publish_select == "replaces_call":
                        # Only reads the model cannot see in this turn: a
                        # published value has to be able to save a call.
                        ahead = [ahead_item(read_history[key]["name"],
                                            read_history[key]["arguments"], value, full_max_bytes)
                                 for key, value in fresh.items()]
                        ahead_bytes = sum(len(json.dumps(item, default=str).encode())
                                          for item in ahead)
                        published, published_bytes, identities = collect_replacements(
                            runtime, scope, budget - ahead_bytes,
                            (turn_identities - ahead_identities) | set(fresh),
                            visible_versions, limit=operation_limit,
                            full_max_bytes=full_max_bytes)
                        if audit:
                            for key, value in fresh.items():
                                audit.referenced(read_history[key]["name"],
                                                 read_history[key]["arguments"], value)
                        row["prefetch_published"] += len(ahead)
                        published = ahead + published
                        published_bytes += ahead_bytes
                        fresh_keys = list(fresh)
                    else:
                        published, published_bytes, identities = collect_references(
                            runtime, scope, budget,
                            visible_identities, limit=reference_limit, latest=recent_reads,
                            mode=publish_mode, full_max_bytes=full_max_bytes)
                        if not recent_reads:
                            published, published_bytes, identities = [], 0, []
                    if published and audit and hasattr(runtime, "valid_read"):
                        # The "still valid" channel is a delivery too: audit it.
                        for identity in identities:
                            valid, value = runtime.valid_read(identity)
                            if valid:
                                _, ref_name, ref_arguments = json.loads(identity)
                                audit.referenced(ref_name, ref_arguments, value)
                    if publish_select == "replaces_call":
                        identities = fresh_keys + list(identities)
                    if published and publish_select == "replaces_call":
                        block += ("Reads the host still holds a current result for; use "
                                  "these instead of calling the tool again (they do not "
                                  "replace any requested action). 'values' are the result's "
                                  "fields, without those listed under _omitted; "
                                  "'shown_earlier' means the result is in this conversation "
                                  "and still current:\n"
                                  + json.dumps(published, default=str, separators=(",", ":")))
                    elif published:
                        block += ("Runtime-validated read results for this decision "
                                  "(they do not replace any requested action):\n"
                                  + json.dumps(published, default=str))
                        step_published = set(identities)
                        published_identities.update(step_published)
                empty_state = False
                if arm_policy["publish_operations"]:
                    if not ops_first:
                        operations = gather_operations(budget - published_bytes)
                    if operations:
                        block += (
                            "\nFailed or uncertain operations already attempted under the "
                            "current authorization and data state (do not repeat them without "
                            "new evidence; uncertain entries must be reconciled before retry, "
                            "not retried directly):\n" + json.dumps(operations, default=str))
                    elif executed_ok or executed_failed:
                        empty_state = (
                            "\nFailed or uncertain operations already attempted under "
                            "the current authorization and data state: [] (none). This "
                            "list is exhaustive; do not report a tool failure unless a "
                            "tool result or this list says so.")
                        block += empty_state
                block = block.strip()
                if not block and last_block:
                    block = ("No earlier read result is confirmed as current any more; "
                             "disregard earlier runtime messages.")
                if block != last_block:
                    # Design 4.3: a new block only when what is published changed.
                    last_block = block
                    messages.append({"role": "runtime", "content": block})
                    row["publish_events"] += 1
                    row["publish_bytes"] += len(block.encode())
                    row["publish_results_count"] += len(published)
                    row["publish_operations_count"] += len(operations)
                    row["publish_empty_state_events"] += bool(empty_state)
                context = preamble + json.dumps(messages)
                response, usage = generate(model, context, trace)
                row["model_calls"] += 1
                for key in ("input_tokens", "output_tokens", "cached_input_tokens", "reasoning_output_tokens"):
                    row[key] += usage.get(key, 0)
                messages.append({"role": "assistant", "content": json.dumps(response)})
                calls = response["calls"]
                step_called, literals = set(), []
                for call in calls:
                    row["logical_tool_calls"] += 1
                    name, arguments = call["name"], call["arguments"]
                    if not all(k.isidentifier() and not keyword.iskeyword(k) for k in arguments):
                        raise ValueError("Tool parameter name is not a valid literal keyword")
                    if name not in {tool["name"] for tool in tools}:
                        result = unknown_tool_result(name, tools)
                        unknown_attempts.add(name)
                    else:
                        # Host generates UUID v4; internal retries keep this identity.
                        operation_id = str(uuid.uuid4())
                        identity = json.dumps([scope, name, arguments], sort_keys=True)
                        step_called.add(identity)
                        repeated_in_turn = identity in turn_identities
                        turn_identities.add(identity)
                        if identity in called_identities:
                            row["exact_repeat_calls"] += 1
                        called_identities.add(identity)
                        hits_before = sum(runtime.stats[key]
                                          for key in ("M1_hits", "M2_hits", "M4_hits"))
                        requests_before = client.metrics["mcp_requests"]
                        try:
                            result = access.call(name, arguments, operation_id)
                        except Exception as error:
                            result = failure_result(error, operation_id)
                        if isinstance(result, dict) and "error" in result:
                            executed_failed.add(name)
                        else:
                            executed_ok.add(name)
                        # The result is now part of the visible conversation.
                        visible_identities.add(identity)
                        ahead_identities.discard(identity)
                        failed = isinstance(result, dict) and "error" in result
                        if access.readonly(name, arguments):
                            if not failed:
                                visible_versions[identity] = version_of(result)
                                read_history[identity] = {
                                    "name": name, "arguments": arguments, "asked": turn_index,
                                    "fetched": read_history.get(identity, {}).get("fetched", -9)}
                        elif not failed:
                            wrote = True
                        catalog_read = (dataset == "appworld"
                                        and readonly(dataset, name, arguments))
                        if catalog_read:
                            recent = ("catalog", json.dumps(
                                [scope, arguments], sort_keys=True))
                        elif access.readonly(name, arguments):
                            recent = ("result", identity)
                        else:
                            recent = None
                        if recent is not None:
                            recent_reads = [item for item in recent_reads if item != recent]
                            recent_reads.insert(0, recent)
                            del recent_reads[5:]
                        from_cache = sum(runtime.stats[key] for key in (
                            "M1_hits", "M2_hits", "M4_hits")) > hits_before
                        if from_cache:
                            if identity in published_identities:
                                used_published.add(identity)
                        no_news = (repeated_in_turn and from_cache
                                   and arm_policy["publish_read_results"])
                        literals.append(name + "(" + ", ".join(
                            f"{k}={v!r}" for k, v in arguments.items()) + ")")
                    payload = {"name": name, "result": result}
                    if name in {tool["name"] for tool in tools} and no_news:
                        # Same read, same turn, state unchanged: say so (design 4.3).
                        payload["note"] = ("same call as earlier in this turn and the state "
                                           "it reads has not changed: no new information")
                        row["repeat_notices"] = row.get("repeat_notices", 0) + 1
                    if lab:
                        lab_served[-1].append(served_entry(name, arguments, result))
                    if audit and name in {tool["name"] for tool in tools}:
                        audit.delivered(name, arguments, result, reached_server=(
                            client.metrics["mcp_requests"] > requests_before))
                    if lab:
                        # Tell the world the call is over, cached or not, so that
                        # events tied to it fire at the same point for every method.
                        client.tool("lab_note", {"name": name, **presented()}, control=True)
                    if arm_policy["context_dedup"]:
                        content, saved, duplicated = dedupe_result(payload, seen_results)
                        if duplicated:
                            row["context_dedup_events"] += 1
                            row["context_dedup_saved_bytes"] += saved
                    else:
                        content, saved = reference_large_repeat(
                            name, arguments, result, seen_large, large_repeat_min_bytes)
                        if saved:
                            row["large_repeat_events"] += 1
                            row["large_repeat_saved_bytes"] += saved
                        if "note" in payload and "result" in content:
                            content = {**content, "note": payload["note"]}
                    # The trace below always keeps the full result.
                    messages.append({"role": "tool", "content": json.dumps(
                        content, default=str)})
                    with trace.open("a") as output:
                        output.write(json.dumps({"kind": "action", "turn": turn_index,
                                                "step": step, "name": name,
                                                "arguments": redact(arguments), "result": redact(result),
                                                "runtime_stats": runtime.stats}, default=str) + "\n")
                if step_published and not step_called:
                    # The model answered this decision entirely from published
                    # state: one reference each replaced a network request.
                    row["published_results_used"] += len(step_published)
                    row["published_only_steps"] += 1
                if literals:
                    turn_calls.append(literals)
                if response["done"]:
                    correction, text = done_correction(
                        dataset, turn_index, original_turn_count, turn_calls,
                        response["answer"], executed_ok, executed_failed,
                        unknown_attempts, arm_policy["publish_operations"],
                        grounding_corrections, protocol_corrections)
                    if correction is None:
                        turn_answer = str(response.get("answer", ""))
                        break
                    messages.append({"role": "user", "content": text})
                    if correction == "grounding":
                        grounding_corrections += 1
                        row["grounding_corrections"] += 1
                    else:
                        protocol_corrections += 1
                        row["protocol_corrections"] += 1
                    continue
            else:
                row["budget_exhausted"] = True
            decoded.append(turn_calls)
            turn_answers.append(turn_answer)
            if compact:
                messages[turn_start:] = [*question, {"role": "assistant", "content": json.dumps(
                    {"calls": [], "done": True, "answer": turn_answer})}]
            if scheduler_policy:
                runtime.end_turn()
        row["task_time_s"] = time.monotonic() - setup_end
        if lab:
            oracle = client.tool("lab_oracle", {"answers": turn_answers,
                                                "served": lab_served}, control=True)
            row["task_success"] = oracle["success"]
            row["turn_answers"] = turn_answers
            row["oracle"] = oracle
            if audit:
                row["delivery"] = {**audit.stats, "stale": audit.stale}
        elif dataset in {"appworld", "tau2"}:
            oracle = client.tool(dataset + "_oracle", {}, control=True)
            row["task_success"] = oracle["success"]
        else:
            entry = next(x for x in json.loads((ROOT / "data/bfcl-tasks.json").read_text())
                         if x["id"] == task_id)
            truth = json.loads((ROOT / "data/bfcl-oracle.json").read_text())[task_id]
            oracle = evaluate_bfcl(decoded, truth, entry, run_id)
            row["task_success"] = oracle["valid"]
        row["oracle"] = oracle
        row["status"] = "completed"
    except Exception as error:
        row["status"] = "error"
        row["error"] = str(error)
        row["traceback"] = traceback.format_exc()
    finally:
        try:
            row["server_cache"] = client.tool("server_cache_stats", {}, control=True)
        except Exception as error:
            row["server_cache"] = {"error": str(error)}
        try:
            client.tool("benchmark_close", {}, control=True)
        except Exception:
            pass
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
        server_log.close()
        row["total_time_s"] = time.monotonic() - start
        trace_records = ([json.loads(line) for line in trace.read_text().splitlines()
                          if line.strip()] if trace.exists() else [])
        model_records = [record for record in trace_records
                         if record.get("kind") == "model"]
        usages = [event["usage"] for record in model_records for event in record["events"]
                  if event.get("type") == "turn.completed"]
        row["model_attempts"] = len(model_records)
        row["model_process_failures"] = sum(1 for record in model_records
                                            if record.get("error"))
        row["model_retries"] = sum(1 for record in trace_records
                                   if record.get("kind") == "model_retry")
        row["model_calls"] = len(usages)
        row["usage_unknown_attempts"] = len(model_records) - len(usages)
        for key in ("input_tokens", "output_tokens", "cached_input_tokens", "reasoning_output_tokens"):
            row[key] = sum(usage.get(key, 0) for usage in usages)
        row["harness_queue_wait_s"] = sum(record.get("queue_wait_s", 0) for record in model_records)
        row["published_cache_hits"] = len(used_published)
        # Shadow checks are measurement, not method work: exclude their time.
        row["shadow_time_s"] = runtime.stats.get("shadow_time_s", 0.0)
        if row["task_time_s"] is not None:
            row["active_time_s"] = active_time(
                row["task_time_s"], row["harness_queue_wait_s"] + row["shadow_time_s"])
        row.update(client.metrics)
        row["runtime"] = runtime.stats
        row["cli_completed_turns"] = row["model_calls"]
        row["M1"] = {"directory_fetches": runtime.stats["M1_lookups"] - runtime.stats["M1_hits"],
                     "cache_hits": runtime.stats["M1_hits"],
                     "note": ("the host lists the tools once per user turn and on every "
                              "announced change; these lookups come from the runner, "
                              "not from the model") if lab else
                     "normal tools/list plus original api_docs calls, no forced re-listing"}
        row["M3"] = {"status": "issuer_verified_AppWorld_login_reuse" if dataset == "appworld"
                     else "lab_server_issues_sessions" if lab
                     else "no_credential_issuer_in_this_task",
                     "cache_hits": runtime.stats["M3_hits"]}
        row["M4"] = {"operations": runtime.operation_audit()}
        if scheduler_policy:
            row["scheduler_usage"] = runtime.usage()
            checks = runtime.stats["shadow_checks"]
            row["stale_hits"] = {
                "shadow_checks": checks, "shadow_stale": runtime.stats["shadow_stale"],
                "stale_hit_rate": (runtime.stats["shadow_stale"] / checks) if checks else None,
                "sampled_verifications": runtime.stats["M2_verifications"],
                "sampled_stale": runtime.stats["M2_verify_stale"]}
        (output_dir / f"{run_id}.result.json").write_text(json.dumps(row, indent=2, default=str))
    print(json.dumps({k: row.get(k) for k in [
        "run_id", "dataset", "task_id", "model", "method", "status",
        "task_success", "model_calls", "input_tokens", "output_tokens", "error"]}), flush=True)
    return row


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["bfcl", "appworld", "tau2", "lab"],
                        required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--model", default="DeepSeek-V4.1-Flash")
    parser.add_argument("--method", choices=[*ARM_NAMES, "fixed"], default="bare")
    parser.add_argument("--max-steps", type=int, default=16)
    parser.add_argument("--output", type=Path, default=RESULTS / "tasks")
    parser.add_argument("--delay-ms", type=float, default=0)
    parser.add_argument("--jitter-ms", type=float, default=0)
    args = parser.parse_args(argv)
    row = run_task(args.dataset, args.task, args.model, args.method, args.output, args.max_steps,
                   delay_ms=args.delay_ms, jitter_ms=args.jitter_ms)
    return 0 if row["status"] == "completed" else 1


if __name__ == "__main__":
    sys.exit(main())
