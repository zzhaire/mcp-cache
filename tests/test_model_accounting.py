"""A timed-out backend attempt must leave an auditable unknown-usage record."""
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from exp_core import model

ROOT = Path(__file__).resolve().parents[1]


class ModelAccounting(unittest.TestCase):
    def test_timeout_is_recorded_before_error_propagates(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            trace = Path(folder) / "trace.jsonl"

            class HangingProcess:
                pid, returncode = 424242, None

                def communicate(self, input=None, timeout=None):
                    if timeout is None:
                        return '{"type":"turn.started"}\n', ""
                    raise subprocess.TimeoutExpired("traex", 180)

                def kill(self):
                    pass

            with patch.object(model, "TEMP", Path(folder)), \
                    patch.object(model.subprocess, "Popen",
                                 return_value=HangingProcess()), \
                    patch.object(model.os, "getpgid", return_value=424242), \
                    patch.object(model.os, "killpg") as killpg:
                with self.assertRaisesRegex(RuntimeError, "timeout after 300"):
                    model.generate("test-model", "test-context", trace)
            # A timeout is retried once, so both attempts are killed and traced.
            self.assertEqual(killpg.call_count, 2)
            rows = [json.loads(line) for line in trace.read_text().splitlines()]
            self.assertEqual([row["kind"] for row in rows], ["model", "model_retry", "model"])
            self.assertIn("unknown", rows[0]["usage_status"])
            self.assertFalse(any(e.get("type") == "turn.completed"
                                 for e in rows[0]["events"]))


def message(text):
    return {"type": "item.completed", "item": {"type": "agent_message", "text": text}}


class AnswerParsing(unittest.TestCase):
    def test_structured_file_is_used(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            path = Path(folder) / "answer.json"
            path.write_text(json.dumps({"calls": [], "done": True, "answer": "ok"}))
            value = model.parse_answer(path, [message('{"calls": [], "done": false, "answer": "x"}')])
            self.assertTrue(value["done"])
            self.assertEqual(value["answer"], "ok")

    def test_empty_file_falls_back_to_fenced_message(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            path = Path(folder) / "answer.json"
            path.write_text("")
            fenced = '```json\n{"calls": [{"name": "ls", "arguments": {}}], "done": false, "answer": ""}\n```'
            value = model.parse_answer(path, [message("thinking"), message(fenced)])
            self.assertEqual(value["calls"], [{"name": "ls", "arguments": {}}])
            self.assertFalse(value["done"])

    def test_invalid_schema_is_rejected_without_a_valid_message(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            path = Path(folder) / "answer.json"
            path.write_text(json.dumps({"calls": "no", "done": True, "answer": ""}))
            with self.assertRaisesRegex(RuntimeError, "no structured final output"):
                model.parse_answer(path, [message("plain prose")])


class GenerateRetry(unittest.TestCase):
    def test_transient_failure_retries_once_and_is_traced(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            trace = Path(folder) / "trace.jsonl"
            calls = []

            def fake(*args, **kwargs):
                calls.append(1)
                if len(calls) == 1:
                    raise RuntimeError("model process exit 1")
                return {"calls": [], "done": True, "answer": ""}, {"input_tokens": 1}

            with patch.object(model, "_generate_once", side_effect=fake), \
                    patch.object(model.time, "sleep"):
                value, usage = model.generate("test-model", "context", trace)
            self.assertEqual(len(calls), 2)
            self.assertEqual(value["done"], True)
            self.assertEqual(usage["input_tokens"], 1)
            self.assertEqual(json.loads(trace.read_text())["kind"], "model_retry")


class Backends(unittest.TestCase):
    def test_codex_schema_encodes_free_form_arguments(self):
        strict = model.strict_schema(model.SCHEMA)
        arguments = strict["properties"]["calls"]["items"]["properties"]["arguments"]
        self.assertEqual(arguments["type"], "string")
        self.assertEqual(model.SCHEMA["properties"]["calls"]["items"]["properties"]
                         ["arguments"]["type"], "object")
        other = {"type": "object"}
        self.assertIs(model.strict_schema(other), other)

    def test_decode_arguments_restores_objects_and_rejects_bad_json(self):
        answer = {"calls": [{"name": "f", "arguments": '{"a": 1}'}], "done": False, "answer": ""}
        self.assertEqual(model.decode_arguments(answer)["calls"][0]["arguments"], {"a": 1})
        with self.assertRaisesRegex(RuntimeError, "no structured final output"):
            model.decode_arguments({"calls": [{"name": "f", "arguments": "[1"}]})
        with self.assertRaisesRegex(RuntimeError, "no structured final output"):
            model.decode_arguments({"calls": [{"name": "f", "arguments": "[1]"}]})

    def test_commands_disable_tools_per_backend(self):
        path = Path("/tmp/x")
        traex, _ = model.build_command("traex", "m", path, model.SCHEMA)
        self.assertIn("--disallowed-tool", traex)
        codex, _ = model.build_command("codex", "m", path, model.SCHEMA)
        self.assertNotIn("--disallowed-tool", codex)
        self.assertIn("shell_tool", codex)
        self.assertIn("unified_exec", codex)
        claude, cwd = model.build_command("claude", "m", path, model.SCHEMA)
        self.assertEqual(claude[claude.index("--tools") + 1], "")
        self.assertIn("--strict-mcp-config", claude)
        self.assertEqual(cwd, path)

    def test_claude_usage_counts_cache_reads_and_writes_as_input(self):
        raw = [
            {"type": "assistant", "message": {"content": [
                {"type": "tool_use", "name": "StructuredOutput", "input": {}}]}},
            {"type": "rate_limit_event", "rate_limit_info": {"unifiedWindows": {
                "seven_day": {"utilization": 0.3}}}},
            {"type": "result", "is_error": False, "total_cost_usd": 0.5,
             "structured_output": {"calls": [], "done": True, "answer": "ok"},
             "usage": {"input_tokens": 2, "cache_read_input_tokens": 100,
                       "cache_creation_input_tokens": 30, "output_tokens": 7,
                       "output_tokens_details": {"thinking_tokens": 3}}}]
        events, usage, quota = model.normalize_claude(raw)
        self.assertEqual(usage["input_tokens"], 132)
        self.assertEqual(usage["cached_input_tokens"], 100)
        self.assertEqual(usage["reasoning_output_tokens"], 3)
        self.assertEqual(quota["unifiedWindows"]["seven_day"]["utilization"], 0.3)
        self.assertEqual([e["type"] for e in events], ["item.completed", "turn.completed"])
        self.assertTrue(json.loads(events[0]["item"]["text"])["done"])

    def test_claude_other_tool_use_is_rejected(self):
        raw = [{"type": "assistant", "message": {"content": [
                   {"type": "tool_use", "name": "Bash", "input": {}}]}},
               {"type": "result", "is_error": False, "usage": {}}]
        with self.assertRaisesRegex(RuntimeError, "Unexpected CLI tool use"):
            model.normalize_claude(raw)

    def test_claude_native_task_tool_request_becomes_calls(self):
        raw = [{"type": "assistant", "message": {"content": [
                   {"type": "tool_use", "name": "ref_get_rate", "input": {"currency": "EUR"}}]}},
               {"type": "user", "message": {"content": [{"type": "tool_result", "is_error": True}]}},
               {"type": "assistant", "message": {"content": [
                   {"type": "tool_use", "name": "StructuredOutput", "input": {}}]}},
               {"type": "result", "is_error": False, "usage": {"input_tokens": 7},
                "structured_output": {"calls": [], "done": True, "answer": "it failed"}}]
        events, usage, _ = model.normalize_claude(raw, model.SCHEMA)
        self.assertEqual(json.loads(events[0]["item"]["text"]), {
            "calls": [{"name": "ref_get_rate", "arguments": {"currency": "EUR"}}],
            "done": False, "answer": ""})
        self.assertEqual(events[1], {"type": "native_calls_converted", "count": 1})
        self.assertEqual(usage["input_tokens"], 7)     # the whole exchange is charged
        # Any other schema has no calls field to carry the request.
        with self.assertRaisesRegex(RuntimeError, "Unexpected CLI tool use"):
            model.normalize_claude(raw, {"type": "object"})

    def test_claude_missing_result_has_no_usage(self):
        self.assertEqual(model.normalize_claude([{"type": "system"}]), ([], None, None))


class TransportWarnings(unittest.TestCase):
    def test_codex_error_item_is_not_treated_as_tool_use(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            trace = Path(folder) / "trace.jsonl"
            answer = {"calls": [{"name": "ls", "arguments": "{}"}], "done": False, "answer": ""}
            stdout = "\n".join(json.dumps(e) for e in [
                {"type": "item.completed", "item": {"type": "error", "message": "Falling back from WebSockets to HTTPS transport."}},
                {"type": "item.completed", "item": {"type": "agent_message", "text": json.dumps(answer)}},
                {"type": "turn.completed", "usage": {"input_tokens": 5, "output_tokens": 1}}])

            class Done:
                pid, returncode = 1, 0

                def communicate(self, input=None, timeout=None):
                    return stdout, ""

            with patch.object(model, "TEMP", Path(folder)), \
                    patch.object(model.subprocess, "Popen", return_value=Done()):
                value, usage = model._generate_once("m", "c", trace, backend="codex")
            self.assertEqual(value["calls"][0]["arguments"], {})
            self.assertEqual(usage["input_tokens"], 5)
