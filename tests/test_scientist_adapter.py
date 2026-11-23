"""Exercise real query/spec routing without a remote model call."""
import json
import unittest
from unittest.mock import patch
from ours.scientist import query, FunctionSpec, traex_backend


class ScientistAdapter(unittest.TestCase):
    def test_numeric_feedback_passes_as_json_string(self):
        payload = {"initial_policy": {"ttl_seconds": 60, "publish_read_results": False},
                   "feedback": [{"task_success": None, "input_tokens": 10}]}
        schema = {"type": "object", "properties": {"ttl_seconds": {"type": "integer"}}}
        with patch.object(traex_backend, "traex_chat_json", return_value={"ttl_seconds": 60}) as call:
            result = query("Use the contract.", json.dumps(payload), model="DeepSeek-V4.1-Flash",
                           func_spec=FunctionSpec("policy", schema, "test"))
        self.assertEqual(result, {"ttl_seconds": 60})
        self.assertEqual(json.loads(call.call_args.args[1][0]["content"]), payload)
        self.assertEqual(call.call_args.kwargs["output_schema"], schema)
