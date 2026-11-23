"""Done-claim grounding and zero-call corrections; pure helpers only."""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from exp_core.runner import claims_tool_failure, done_correction

GROUNDING_PREFIX = ("Runtime execution log check: no tool call in this run returned "
                    "an error, and no tool name was rejected. Tools that returned "
                    "results: ")
GROUNDING_SUFFIX = (". Do not report a tool failure that did not occur. Continue the "
                    "task with the listed tools, or finish without claiming failures.")
BFCL_CORRECTION = ("This benchmark requires at least one real tool call in every "
                   "original user turn. Answering without any tool call makes this "
                   "turn invalid. Call the tool needed for this turn now, then set "
                   "done=true.")


def decide(dataset="tau2", turn_index=0, original_turn_count=1, turn_calls=None,
           answer="", executed_ok=(), executed_failed=(), unknown_attempts=(),
           publish_operations=True, grounding_corrections=0,
           protocol_corrections=0):
    return done_correction(
        dataset, turn_index, original_turn_count,
        [["x()"]] if turn_calls is None else turn_calls, answer,
        set(executed_ok), set(executed_failed), set(unknown_attempts),
        publish_operations, grounding_corrections, protocol_corrections)


class FailureClaims(unittest.TestCase):
    def test_explicit_tool_failure_claims_match(self):
        for answer in (
                "get_details_by_id was an unsupported call.",
                "check_status_bar is not supported by the server.",
                "That operation is unsupported.",
                "The tool returned an error.",
                "It returned 'error' twice.",
                "The request failed with AccessFailure.",
                "The server rejected the request.",
                "The tools are unavailable right now.",
                "This tool was unavailable."):
            with self.subTest(answer=answer):
                self.assertTrue(claims_tool_failure(answer))

    def test_ordinary_prose_does_not_match(self):
        for answer in ("The bill is overdue.",
                       "No errors occurred during the lookup.",
                       "I could not be sure which account is correct.",
                       "The task is complete.",
                       "The API supports the operation.",
                       "The invoice was paid."):
            with self.subTest(answer=answer):
                self.assertFalse(claims_tool_failure(answer))


class GroundingCorrection(unittest.TestCase):
    def test_hallucinated_failure_gets_a_grounding_message(self):
        kind, text = decide(answer="get_details_by_id returned an error.",
                            executed_ok=("get_customer_by_phone",))
        self.assertEqual(kind, "grounding")
        self.assertEqual(text, GROUNDING_PREFIX + "get_customer_by_phone"
                         + GROUNDING_SUFFIX)

    def test_executed_tool_names_are_sorted(self):
        _, text = decide(answer="The tools are unavailable.",
                         executed_ok=("transfer_to_human_agents", "get_details_by_id"))
        self.assertEqual(text, GROUNDING_PREFIX
                         + "get_details_by_id, transfer_to_human_agents"
                         + GROUNDING_SUFFIX)

    def test_no_results_yet_reports_none(self):
        _, text = decide(answer="That call is unsupported.", executed_ok=())
        self.assertEqual(text, GROUNDING_PREFIX + "(none)" + GROUNDING_SUFFIX)

    def test_real_failure_blocks_the_correction(self):
        kind, _ = decide(answer="delete_message returned an error.",
                         executed_ok=("ls",), executed_failed=("delete_message",))
        self.assertIsNone(kind)

    def test_rejected_unknown_name_blocks_the_correction(self):
        kind, _ = decide(answer="That call is unsupported.",
                         executed_ok=("ls",), unknown_attempts=("get_details_by_id",))
        self.assertIsNone(kind)

    def test_non_ours_arm_never_gets_grounding(self):
        kind, _ = decide(answer="The tool returned an error.", publish_operations=False)
        self.assertIsNone(kind)

    def test_ordinary_answer_is_not_corrected(self):
        kind, text = decide(answer="The appointment is confirmed.",
                            executed_ok=("get_customer_by_phone",))
        self.assertEqual((kind, text), (None, ""))

    def test_grounding_is_capped_at_two(self):
        answer = "The tools are unavailable."
        self.assertEqual(decide(answer=answer, grounding_corrections=1)[0],
                         "grounding")
        self.assertIsNone(decide(answer=answer, grounding_corrections=2)[0])

    def test_grounding_wins_over_the_bfcl_correction(self):
        kind, _ = decide(dataset="bfcl", turn_calls=[],
                         answer="delete_message returned an error.",
                         executed_ok=("ls",))
        self.assertEqual(kind, "grounding")


class BFCLZeroCallCorrection(unittest.TestCase):
    def test_empty_original_turn_is_corrected(self):
        kind, text = decide(dataset="bfcl", turn_calls=[], answer="Done.")
        self.assertEqual(kind, "bfcl")
        self.assertEqual(text, BFCL_CORRECTION)

    def test_turn_with_a_real_call_is_accepted(self):
        kind, _ = decide(dataset="bfcl", turn_calls=[["ls(path='/')"]], answer="")
        self.assertIsNone(kind)

    def test_augmentation_turn_is_exempt(self):
        kind, _ = decide(dataset="bfcl", turn_index=2, original_turn_count=2,
                         turn_calls=[], answer="")
        self.assertIsNone(kind)

    def test_every_original_turn_is_covered(self):
        kind, _ = decide(dataset="bfcl", turn_index=1, original_turn_count=3,
                         turn_calls=[], answer="")
        self.assertEqual(kind, "bfcl")

    def test_one_correction_per_turn(self):
        kind, _ = decide(dataset="bfcl", turn_calls=[], answer="",
                         protocol_corrections=1)
        self.assertIsNone(kind)

    def test_non_bfcl_datasets_are_never_corrected(self):
        for dataset in ("appworld", "tau2"):
            with self.subTest(dataset=dataset):
                self.assertIsNone(decide(dataset=dataset, turn_calls=[],
                                         publish_operations=False)[0])


if __name__ == "__main__":
    unittest.main()
