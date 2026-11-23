"""Shared protocol helpers from run.py; no model, network or benchmark calls."""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from exp_core.runner import protocol_text, unknown_tool_result

APPWORLD_PROTOCOL = (
    " This is an AppWorld task. Read the task with api_call "
    "app=supervisor api=show_active_task. Most app APIs need a "
    "token: read supervisor.show_account_passwords, then call "
    "api_call app=<app> api=login with username and password to get "
    "access_token for that app. When every requirement is met, call "
    "api_call app=supervisor api=complete_task, with answer only if "
    "the task asks a question; do not set done before complete_task "
    "reports the task is complete.")
BFCL_PROTOCOL = (
    " This benchmark requires at least one real tool call in every original "
    "user turn; a turn with no tool call is invalid even when the answer looks "
    "derivable from earlier results.")


class ProtocolText(unittest.TestCase):
    def test_appworld_text_is_unchanged(self):
        self.assertEqual(protocol_text("appworld"), APPWORLD_PROTOCOL)

    def test_tau2_has_no_protocol_text(self):
        self.assertEqual(protocol_text("tau2"), "")

    def test_bfcl_text_exact(self):
        self.assertEqual(protocol_text("bfcl"), BFCL_PROTOCOL)

    def test_bfcl_text_is_only_used_for_bfcl(self):
        self.assertIn("at least one real tool call", protocol_text("bfcl"))
        for dataset in ("appworld", "tau2"):
            self.assertNotIn("at least one real tool call",
                             protocol_text(dataset))

    def test_supervisor_argument_does_not_change_the_text(self):
        self.assertEqual(protocol_text("appworld", supervisor={"id": "s"}),
                         APPWORLD_PROTOCOL)
        self.assertEqual(protocol_text("bfcl", supervisor={"id": "s"}),
                         BFCL_PROTOCOL)


class UnknownToolResult(unittest.TestCase):
    def test_error_lists_every_valid_tool_name(self):
        tools = [{"name": "get_customer_by_phone"}, {"name": "transfer_to_human_agents"}]
        result = unknown_tool_result("get_details_by_id", tools)
        self.assertEqual(result, {"error": "Unknown tool name 'get_details_by_id'. "
                                          "Valid tool names: get_customer_by_phone, "
                                          "transfer_to_human_agents"})

    def test_single_tool_catalog(self):
        self.assertEqual(unknown_tool_result("x", [{"name": "api_call"}]),
                         {"error": "Unknown tool name 'x'. Valid tool names: api_call"})

    def test_payload_has_no_extra_keys(self):
        self.assertEqual(set(unknown_tool_result("x", [{"name": "a"}])), {"error"})

    def test_name_is_quoted_verbatim(self):
        message = unknown_tool_result("bad name", [{"name": "a"}])["error"]
        self.assertIn("'bad name'", message)


if __name__ == "__main__":
    unittest.main()
