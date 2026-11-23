import unittest
from ours.selection import accept_candidate


class CandidateGate(unittest.TestCase):
    def row(self, success=True, seconds=10, **kwargs):
        return dict(status="completed", task_success=success, task_time_s=seconds,
                    dataset="bfcl", task_id="one", model="model", split="dev",
                    max_steps_per_turn=16, runtime_source_sha256="same", **kwargs)

    def test_faster_failure_is_rejected(self):
        self.assertFalse(accept_candidate([self.row(False, 1, policy={"ttl_seconds": 30})],
                                          [self.row(False, 10)]))
        self.assertFalse(accept_candidate([self.row(False, 1, policy={"ttl_seconds": 30})],
                                          [self.row(True, 10)]))

    def test_unknown_usage_and_exhaustion_are_rejected(self):
        self.assertFalse(accept_candidate([self.row(seconds=1, usage_unknown_attempts=1,
                                                    policy={"ttl_seconds": 30})],
                                          [self.row()]))
        self.assertFalse(accept_candidate([self.row(seconds=1, policy={"ttl_seconds": 30})],
                                          [self.row(budget_exhausted=True)]))
        baseline = self.row()
        baseline["status"] = "error"
        self.assertFalse(accept_candidate([self.row(seconds=1, policy={"ttl_seconds": 30})], [baseline]))

    def test_valid_paired_improvement_is_accepted(self):
        candidate = self.row(seconds=1, policy={"ttl_seconds": 30})
        self.assertTrue(accept_candidate([candidate], [self.row()]))
        self.assertTrue(accept_candidate([candidate], [self.row(False, 1)]))
        self.assertFalse(accept_candidate([self.row(policy={"ttl_seconds": 30})], [self.row()]))
        changed = self.row(seconds=1, policy={"ttl_seconds": 30})
        changed["task_id"] = "different"
        self.assertFalse(accept_candidate([changed], [self.row()]))

    def test_identical_policy_cannot_win_by_run_variation(self):
        candidate = self.row(seconds=1, policy={
            "ttl_seconds": 60, "publish_read_results": False,
            "publish_budget_bytes": 0, "rationale": "retain baseline"})
        self.assertFalse(accept_candidate([candidate], [self.row()]))
