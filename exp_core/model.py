"""Authenticated CLI model calls with real usage and no token estimates."""
import json
import fcntl
import os
import re
from pathlib import Path
import signal
import subprocess
import tempfile
import time
from exp_core.redaction import redact
from exp_core.settings import (MODEL_BACKEND, MODEL_CLI, MODEL_CONCURRENCY,
                                MODEL_EFFORT, SCRATCH)

TEMP = SCRATCH
BACKENDS = ("traex", "codex", "claude")
# Codex exec exposes these agent tools by default; the task agent must only
# answer with JSON, so every tool surface is switched off rather than merely
# discouraged in the prompt.
CODEX_DISABLED_FEATURES = (
    "shell_tool", "unified_exec", "apps", "plugins", "browser_use",
    "browser_use_external", "in_app_browser", "computer_use",
    "image_generation", "view_image", "sleep_tool", "tool_suggest")
# The CLI backend rejects unbounded concurrent traffic. Eight slots keep real
# model traffic bounded while letting independent task processes overlap; the
# time a call spends waiting for a slot is recorded as queue wait and excluded
# from active_time_s, so a busy machine never inflates a run's own metrics.
SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "calls": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "properties": {"name": {"type": "string"}, "arguments": {"type": "object"}},
            "required": ["name", "arguments"]}},
        "done": {"type": "boolean"}, "answer": {"type": "string"}},
    "required": ["calls", "done", "answer"]}


def _message_texts(events):
    return [event["item"].get("text", "") for event in events
            if event.get("type") == "item.completed"
            and (event.get("item") or {}).get("type") == "agent_message"]


def _strip_fence(text):
    value = text.strip()
    if not value.startswith("```"):
        return value
    body = value.split("\n", 1)[1] if "\n" in value else ""
    if body.rstrip().endswith("```"):
        body = body.rstrip()[:-3]
    return body.strip()


def parse_answer(answer_path, events):
    """Parse the structured answer, falling back to the visible final message.

    The CLI occasionally leaves ``answer.json`` empty while the final
    ``agent_message`` already contains the same JSON object, sometimes wrapped
    in a markdown fence. Both forms come from the model; parsing the visible
    message does not add a model call and is applied identically to every arm.
    """
    text = answer_path.read_text() if answer_path.exists() else ""
    candidates = [text] + [_strip_fence(item) for item in reversed(_message_texts(events))]
    for candidate in candidates:
        if not candidate.strip():
            continue
        try:
            value = json.loads(candidate)
        except ValueError:
            continue
        if (isinstance(value, dict) and isinstance(value.get("calls"), list)
                and isinstance(value.get("done"), bool)
                and isinstance(value.get("answer"), str)):
            return value
    raise RuntimeError("Model produced no structured final output")


_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+")


def mask_foreign_emails(value, context):
    """Replace e-mail addresses the prompt never contained: ``(value, count)``.

    The CLI adds the signed-in account's address to the model's context. An
    address in the reply that the prompt does not contain therefore came from
    the machine, not from the task (or was made up); either way it must not
    reach a tool argument, an answer or a stored trace.
    """
    known = {address.lower() for address in _EMAIL.findall(context)}
    count = 0

    def clean(item):
        nonlocal count
        if isinstance(item, dict):
            return {key: clean(part) for key, part in item.items()}
        if isinstance(item, list):
            return [clean(part) for part in item]
        if isinstance(item, str):
            def swap(match):
                nonlocal count
                if match.group(0).lower() in known:
                    return match.group(0)
                count += 1
                return "[address-not-in-task]"
            return _EMAIL.sub(swap, item)
        return item
    return clean(value), count


def generate(model, context, trace_path, schema=SCHEMA, attempts=2):
    """Call the backend, retrying one transient model-interface failure.

    Only failures before any tool execution are retried (empty process exit,
    missing usage, unusable structured output). Timeouts are not retried. The
    same policy applies to every arm and each failed attempt is traced.
    """
    for attempt in range(attempts):
        try:
            return _generate_once(model, context, trace_path, schema)
        except RuntimeError as error:
            transient = ("model process exit" in str(error)
                         or "no structured final output" in str(error)
                         or "Missing or ambiguous" in str(error)
                         or "Unexpected CLI tool use" in str(error)
                         or "model timeout" in str(error))
            if not transient or attempt + 1 >= attempts:
                raise
            with trace_path.open("a") as output:
                output.write(json.dumps({"kind": "model_retry",
                    "failed_attempt": attempt + 1, "error": str(error)}) + "\n")
            time.sleep(2)


def _acquire_slot():
    """Take one of the bounded model-traffic slots; block only when all are busy."""
    TEMP.mkdir(parents=True, exist_ok=True)
    for index in range(MODEL_CONCURRENCY):
        handle = (TEMP / f"model.lock.{index}").open("w")
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return handle
        except BlockingIOError:
            handle.close()
    handle = (TEMP / f"model.lock.{MODEL_CONCURRENCY - 1}").open("w")
    fcntl.flock(handle, fcntl.LOCK_EX)
    return handle


def strict_schema(schema):
    """OpenAI strict structured output rejects free-form objects.

    Only the task schema's ``arguments`` object is free-form, so the Codex
    backend asks for it as a JSON-encoded string and ``decode_arguments``
    restores the object before the runner sees it. Other schemas pass through.
    """
    if schema is not SCHEMA:
        return schema
    value = json.loads(json.dumps(schema))
    value["properties"]["calls"]["items"]["properties"]["arguments"] = {
        "type": "string", "description": "JSON-encoded object of literal arguments"}
    return value


def decode_arguments(answer):
    for call in answer["calls"]:
        if isinstance(call.get("arguments"), str):
            try:
                call["arguments"] = json.loads(call["arguments"])
            except ValueError:
                raise RuntimeError("Model produced no structured final output") from None
        if not isinstance(call.get("arguments"), dict):
            raise RuntimeError("Model produced no structured final output")
    return answer


def build_command(backend, model, path, schema):
    """Return (command, cwd) for one backend call; the prompt goes to stdin."""
    if backend == "claude":
        command = [
            MODEL_CLI, "-p", "--output-format", "stream-json", "--verbose",
            "--json-schema", json.dumps(schema), "--tools", "",
            "--strict-mcp-config", "--setting-sources", "",
            "--disable-slash-commands", "--no-session-persistence",
            "--model", model]
        if MODEL_EFFORT:
            command.extend(["--effort", MODEL_EFFORT])
        return command, path
    command = [
        MODEL_CLI, "exec", "--ignore-user-config",
        "--skip-git-repo-check", "--ephemeral", "--sandbox", "read-only"]
    if backend == "traex":
        command.extend(["--disallowed-tool", "Shell", "--disallowed-tool", "Bash"])
    else:
        for feature in CODEX_DISABLED_FEATURES:
            command.extend(["--disable", feature])
        command.extend(["-c", 'web_search="disabled"'])
        if MODEL_EFFORT:
            command.extend(["-c", f'model_reasoning_effort="{MODEL_EFFORT}"'])
    command.extend([
        "--json", "-C", str(path), "-m", model,
        "--output-schema", str(path / "schema.json"),
        "-o", str(path / "answer.json"), "-"])
    return command, None


def normalize_claude(raw_events, schema=None):
    """Map Claude stream-json to the Codex-style events the runner aggregates.

    ``input_tokens`` keeps the experiment's meaning (all prompt tokens the model
    read), so cache reads and cache writes are added back; Claude reports them
    separately. Quota utilization from ``rate_limit_event`` is kept for audit.

    The task tools exist only as text in the prompt, but the model sometimes
    invokes one as if it were a native function. For the task schema that first
    request is taken at its word and becomes the reply's ``calls``; whatever the
    CLI did afterwards (an error result and a second attempt) is discarded. The
    usage still covers the whole exchange, so the cost is over- not
    under-counted, and the conversion is noted in the trace.
    """
    results = [e for e in raw_events if e.get("type") == "result"]
    if len(results) != 1:
        return [], None, None
    result = results[0]
    if result.get("is_error"):
        raise RuntimeError(f"model process exit: {result.get('subtype')} "
                           f"{result.get('api_error_status') or ''}".strip())
    native, answered = [], False
    for event in raw_events:
        if event.get("type") == "user" and (native or answered):
            break  # only the model's first request counts
        if event.get("type") != "assistant":
            continue
        for block in (event.get("message") or {}).get("content") or []:
            if block.get("type") != "tool_use":
                continue
            if block.get("name") == "StructuredOutput":
                answered = True
            else:
                native.append({"name": block.get("name"),
                               "arguments": block.get("input") or {}})
    structured = result.get("structured_output")
    if native:
        if schema is not SCHEMA or answered:
            raise RuntimeError("Unexpected CLI tool use; this run cannot count as one model turn")
        structured = {"calls": native, "done": False, "answer": ""}
    raw = result.get("usage") or {}
    cache_read = raw.get("cache_read_input_tokens", 0)
    cache_write = raw.get("cache_creation_input_tokens", 0)
    usage = {
        "input_tokens": raw.get("input_tokens", 0) + cache_read + cache_write,
        "cached_input_tokens": cache_read,
        "cache_write_input_tokens": cache_write,
        "output_tokens": raw.get("output_tokens", 0),
        "reasoning_output_tokens": (raw.get("output_tokens_details") or {}).get("thinking_tokens", 0),
        "list_cost_usd": result.get("total_cost_usd"),
    }
    events = []
    if isinstance(structured, dict):
        events.append({"type": "item.completed", "item": {
            "type": "agent_message", "text": json.dumps(structured)}})
    if native:
        events.append({"type": "native_calls_converted", "count": len(native)})
    events.append({"type": "turn.completed", "usage": usage})
    quota = [e["rate_limit_info"] for e in raw_events if e.get("type") == "rate_limit_event"]
    return events, usage, (quota[-1] if quota else None)


def _generate_once(model, context, trace_path, schema=SCHEMA, backend=None):
    backend = backend or MODEL_BACKEND
    if backend not in BACKENDS:
        raise ValueError(f"unknown model backend: {backend}")
    TEMP.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=TEMP) as folder:
        path = Path(folder)
        sent_schema = strict_schema(schema) if backend == "codex" else schema
        (path / "schema.json").write_text(json.dumps(sent_schema))
        start = time.monotonic()
        command, cwd = build_command(backend, model, path, sent_schema)
        # Bounded concurrency: the backend tolerates several calls but not
        # unbounded ones. Queue wait is measured, never hidden.
        lock = _acquire_slot()
        try:
            model_start = time.monotonic()
            # The CLI spawns helper processes that inherit the stdout/stderr
            # pipes. A plain ``subprocess.run(timeout=...)`` kills only the
            # direct child and then blocks forever waiting for those pipes to
            # close, so the CLI runs in its own session and the whole group is
            # killed on timeout.
            process = subprocess.Popen(
                command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, text=True, start_new_session=True,
                cwd=cwd)
            timed_out = False
            try:
                stdout, stderr = process.communicate(context, timeout=300)
            except subprocess.TimeoutExpired:
                timed_out = True
                try:
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    process.kill()
                stdout, stderr = process.communicate()
            if timed_out:
                events = []
                for line in (stdout or "").splitlines():
                    try:
                        events.append(json.loads(line))
                    except ValueError:
                        pass
                with trace_path.open("a") as output:
                    output.write(json.dumps({"kind": "model", "model": model,
                        "backend": backend,
                        "elapsed_s": time.monotonic() - start,
                        "queue_wait_s": model_start - start,
                        "events": redact(mask_foreign_emails(events, context)[0]),
                        "error": "model timeout after 300 seconds",
                        "usage_status": "unknown unless turn.completed present"}) + "\n")
                raise RuntimeError("model timeout after 300 seconds")
        finally:
            lock.close()
        events = []
        for line in (stdout or "").splitlines():
            if line.startswith("{"):
                events.append(json.loads(line))
        record = {"kind": "model", "model": model, "backend": backend,
                  "elapsed_s": time.monotonic() - start,
                  "queue_wait_s": model_start - start}
        if os.environ.get("MCP_TRACE_CONTEXT"):
            # Debugging aid: what the model was shown (holds task credentials).
            record["context"] = context
        if backend == "claude":
            raw_events = events
            try:
                events, _, quota = normalize_claude(raw_events, schema)
            except RuntimeError as error:
                record.update(events=[], error=str(error), raw_events=redact(
                    mask_foreign_emails(raw_events, context)[0]))
                with trace_path.open("a") as output:
                    output.write(json.dumps(record) + "\n")
                raise
            raw_events, masked = mask_foreign_emails(raw_events, context)
            events, _ = mask_foreign_emails(events, context)
            record.update(raw_events=redact(raw_events), quota=quota,
                          foreign_emails_masked=masked)
        record["events"] = redact(events)
        with trace_path.open("a") as output:
            output.write(json.dumps(record) + "\n")
        if process.returncode:
            errors = [x for x in (stderr or "").splitlines() if x.startswith("Error:")]
            errors.extend(x.get("message", "") for x in events if x.get("type") == "error")
            raise RuntimeError("; ".join(errors) or f"model process exit {process.returncode}")
        for event in events:
            item = event.get("item", {})
            # Codex emits transport warnings as ``error`` items (for example a
            # WebSocket-to-HTTPS fallback) and may still finish the turn; the
            # return code and usage checks decide whether the call failed.
            if item.get("type") not in ("agent_message", "reasoning", "error", None):
                raise RuntimeError("Unexpected CLI tool use; this run cannot count as one model turn")
        usage = [x["usage"] for x in events if x.get("type") == "turn.completed"]
        if len(usage) != 1:
            raise RuntimeError("Missing or ambiguous actual model usage")
        answer = parse_answer(path / "answer.json", events)
        if backend == "codex" and schema is SCHEMA:
            answer = decode_arguments(answer)
        return answer, usage[0]
