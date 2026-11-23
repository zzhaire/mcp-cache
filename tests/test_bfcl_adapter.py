"""Verify that HTTP adapter dispatch and BFCL reference semantics agree."""
import json
import unittest
import uuid
import copy
from pathlib import Path
from exp_core.adapters.server import Benchmark
from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils import execute_multi_turn_func_call

ROOT = Path(__file__).resolve().parents[1]


class BFCLAdapter(unittest.TestCase):
    def test_numeric_run_id_cannot_hide_incorrect_ticket(self):
        from exp_core.runner import evaluate_bfcl
        entry = next(x for x in json.loads((ROOT / "data/bfcl-tasks.json").read_text())
                     if x["id"] == "multi_turn_base_176")
        truth = json.loads((ROOT / "data/bfcl-oracle.json").read_text())[entry["id"]]
        calls = copy.deepcopy(truth["ground_truth"])
        calls[1] = [call.replace("2023", "2026") for call in calls[1]]
        result = evaluate_bfcl([[turn] for turn in calls], truth, entry, "1234" + uuid.uuid4().hex)
        self.assertFalse(result["valid"])

    def test_initial_state_and_literal_call_match_reference(self):
        benchmark = Benchmark()
        info = benchmark.initialize("bfcl", "multi_turn_base_176", uuid.uuid4().hex)
        result = benchmark.invoke("get_credit_card_balance",
                                  {"access_token": "ABCD1234", "card_id": "id_1234"})
        entry = next(x for x in json.loads((ROOT / "data/bfcl-tasks.json").read_text())
                     if x["id"] == "multi_turn_base_176")
        reference, _ = execute_multi_turn_func_call(
            ["get_credit_card_balance(access_token='ABCD1234',card_id='id_1234')"],
            entry["initial_config"], entry["involved_classes"],
            "test_" + uuid.uuid4().hex, entry["id"])
        self.assertEqual(result, json.loads(reference[0]))
        self.assertIn("question", info)

    def test_unexposed_method_is_rejected(self):
        benchmark = Benchmark()
        benchmark.initialize("bfcl", "multi_turn_base_176", uuid.uuid4().hex)
        with self.assertRaises(ValueError):
            benchmark.invoke("__init__", {})


if __name__ == "__main__":
    unittest.main()
