"""Matrix identity, resume and CLI checks; no model or server is started."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from exp_core import matrix, runner  # noqa: E402
from exp_core.methods import ARM_NAMES, policy_sha256, resolve_arm  # noqa: E402

SCENARIOS = ROOT / "data/scenarios-36.json"
RESULTS = ROOT / "results" / "smoke"


def load_cases():
    return matrix.validate_cases(matrix.load_document(SCENARIOS))


def dry_run(*args):
    result = subprocess.run(
        [sys.executable, str(ROOT / "run"), "smoke", "--dry-run", "--split", "all", *args],
        capture_output=True, text=True, cwd=ROOT)
    if result.returncode != 0:
        raise AssertionError(result.stderr)
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    return json.loads(lines[-1])


def snapshot(path):
    if not path.exists():
        return None
    return {str(item.relative_to(path)): item.read_bytes()
            for item in path.rglob("*") if item.is_file()}


class BuildJobs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = load_cases()
        cls.code_sha = runner.code_sha256(ROOT)

    def jobs(self, *args, **kwargs):
        kwargs.setdefault("code_sha", self.code_sha)
        return matrix.build_jobs(self.cases, *args, **kwargs)

    def test_default_matrix_has_252_unique_identities(self):
        jobs = self.jobs()
        identities = [job["identity"] for job in jobs]
        self.assertEqual(len(jobs), 252)
        self.assertEqual(len(set(identities)), 252)
        self.assertEqual(len({job["case_id"] for job in jobs}), 36)
        self.assertEqual({job["arm"] for job in jobs}, set(ARM_NAMES))
        for job in jobs:
            self.assertEqual(job["identity"][0], job["case_id"])
            self.assertEqual(job["identity"][1], job["arm"])
            self.assertEqual(len(job["policy_sha256"]), 64)
            self.assertEqual(len(job["code_sha256"]), 64)

    def test_identity_is_stable_across_builds(self):
        first = [job["identity"] for job in self.jobs()]
        second = [job["identity"] for job in self.jobs()]
        self.assertEqual(first, second)

    def test_fixed_normalizes_to_rw_merchant(self):
        fixed = matrix.build_jobs(self.cases[:1], ["fixed"], code_sha=self.code_sha)
        merchant = matrix.build_jobs(self.cases[:1], ["rw_merchant"],
                                     code_sha=self.code_sha)
        self.assertEqual(len(fixed), 1)
        self.assertEqual(fixed[0]["arm"], "rw_merchant")
        self.assertEqual(fixed[0]["identity"], merchant[0]["identity"])
        self.assertEqual(fixed[0]["policy_sha256"],
                         policy_sha256(resolve_arm("rw_merchant")))
        both = matrix.build_jobs(self.cases[:1], ["fixed", "rw_merchant"],
                                 code_sha=self.code_sha)
        self.assertEqual(len(both), 1)

    def test_split_and_arm_filters(self):
        dev = self.jobs(split="dev")
        self.assertEqual(len(dev), 126)
        self.assertEqual({job["split"] for job in dev}, {"dev"})
        self.assertEqual(len({job["case_id"] for job in dev}), 18)
        ours = self.jobs(["ours"])
        self.assertEqual(len(ours), 36)
        self.assertEqual({job["arm"] for job in ours}, {"ours"})

    def test_ours_policy_override_changes_only_ours(self):
        base = {job["arm"]: job for job in matrix.build_jobs(
            self.cases[:1], code_sha=self.code_sha)}
        override = {"publish_budget_bytes": 2048, "ttl_seconds": 30}
        changed = {job["arm"]: job for job in matrix.build_jobs(
            self.cases[:1], ours_policy=override, code_sha=self.code_sha)}
        self.assertEqual(set(base), set(ARM_NAMES))
        self.assertNotEqual(base["ours"]["policy_sha256"],
                            changed["ours"]["policy_sha256"])
        self.assertEqual(changed["ours"]["policy"]["publish_budget_bytes"], 2048)
        self.assertEqual(changed["ours"]["policy"]["ttl_seconds"], 30)
        self.assertEqual(changed["ours"]["arm"], "ours")
        for arm in ARM_NAMES:
            if arm != "ours":
                self.assertEqual(base[arm]["identity"], changed[arm]["identity"])
        retagged = matrix.build_jobs(
            self.cases[:1], ["ours"], ours_policy={"arm": "candidate"},
            code_sha=self.code_sha)[0]
        self.assertEqual(retagged["arm"], "ours")


class Resume(unittest.TestCase):
    def test_completed_and_error_results_are_skipped_without_overwrite(self):
        cases = load_cases()[:2]
        with tempfile.TemporaryDirectory() as temp:
            results = Path(temp) / "matrix"
            jobs = matrix.build_jobs(cases, ["bare"], results_root=results,
                                     code_sha=runner.code_sha256(ROOT))
            self.assertEqual(len(jobs), 2)
            for job, status in zip(jobs, ("completed", "error")):
                job["output_dir"].mkdir(parents=True)
                row = {key: job[key]
                       for key in matrix.IDENTITY_FIELDS}
                row.update({"status": status, "run_id": status, "task_success": None})
                (job["output_dir"] / f"{status}.result.json").write_text(
                    json.dumps(row), encoding="utf-8")
            before = snapshot(results)
            done = matrix.completed_identities(results)
            self.assertEqual(done, {job["identity"] for job in jobs})
            self.assertEqual(matrix.pending_jobs(jobs, done), [])
            self.assertEqual(snapshot(results), before)

    def test_running_result_is_not_terminal(self):
        case = load_cases()[0]
        with tempfile.TemporaryDirectory() as temp:
            results = Path(temp) / "matrix"
            job = matrix.build_jobs([case], ["bare"], results_root=results,
                                    code_sha=runner.code_sha256(ROOT))[0]
            job["output_dir"].mkdir(parents=True)
            row = {key: job[key]
                   for key in matrix.IDENTITY_FIELDS}
            row["status"] = "running"
            (job["output_dir"] / "running.result.json").write_text(
                json.dumps(row), encoding="utf-8")
            self.assertEqual(matrix.completed_identities(results), set())
            self.assertEqual(matrix.pending_jobs([job], set()), [job])


class CommandLine(unittest.TestCase):
    def test_dry_run_reports_252_without_creating_results(self):
        before = snapshot(RESULTS)
        payload = dry_run()
        self.assertTrue(payload["dry_run"])
        self.assertEqual(payload["jobs"], 252)
        self.assertEqual(len(payload["identities"]), 252)
        identities = {(item["case_id"], item["arm"], item["policy_sha256"],
                       item["code_sha256"]) for item in payload["identities"]}
        self.assertEqual(len(identities), 252)
        self.assertEqual(snapshot(RESULTS), before)

    def test_dry_run_split_and_arm_filters(self):
        self.assertEqual(dry_run("--split", "dev")["jobs"], 126)
        self.assertEqual(dry_run("--arms", "ours")["jobs"], 36)

    def test_ours_policy_file_changes_only_ours_hash(self):
        case = load_cases()[0]
        with tempfile.TemporaryDirectory() as temp:
            policy_file = Path(temp) / "ours.json"
            policy_file.write_text(json.dumps({"publish_budget_bytes": 2048}),
                                   encoding="utf-8")
            payload = dry_run("--cases", case["case_id"], "--arms", "ours",
                              "--arms", "bare", "--ours-policy-file",
                              str(policy_file))
        self.assertEqual(payload["jobs"], 2)
        by_arm = {item["arm"]: item for item in payload["identities"]}
        self.assertEqual(by_arm["bare"]["policy_sha256"],
                         policy_sha256(resolve_arm("bare")))
        self.assertNotEqual(by_arm["ours"]["policy_sha256"],
                            policy_sha256(resolve_arm("ours")))


if __name__ == "__main__":
    unittest.main()
