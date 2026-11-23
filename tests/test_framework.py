"""Regression tests for the public experiment entry and reproducibility boundary."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
ENTRY = ROOT / "run"


class ExperimentFrameworkTests(unittest.TestCase):
    def invoke(self, *arguments, cwd=None):
        self.assertTrue(ENTRY.is_file(), "the unified experiment run entry is missing")
        return subprocess.run(
            [sys.executable, str(ENTRY), *arguments],
            cwd=cwd or ROOT.parent, capture_output=True, text=True, timeout=30)

    def plan(self, *arguments):
        result = self.invoke("smoke", "--dry-run", "--methods", "bare",
                             "--split", "all", *arguments)
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout)

    def test_dry_run_works_from_other_directory_without_writing_results(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "results"
            plan = self.plan("--results-root", str(output))
            self.assertEqual(plan["jobs"], 36)
            self.assertFalse(output.exists())
            self.assertEqual(plan["profile"], "smoke")

    def test_different_execution_settings_have_distinct_resume_identities(self):
        base = self.plan()
        for arguments in (("--model", "other-model"), ("--seed", "1234"),
                          ("--delay-ms", "90"), ("--max-steps", "2")):
            changed = self.plan(*arguments)
            self.assertNotEqual(base["run_config_sha256"],
                                changed["run_config_sha256"], arguments)

    def test_partial_baselines_are_identified_as_adaptations(self):
        result = self.invoke("list")
        self.assertEqual(result.returncode, 0, result.stderr)
        methods = {item["name"]: item for item in json.loads(result.stdout)["methods"]}
        self.assertEqual(methods["openai"]["reproduction_status"], "partial_adaptation")
        self.assertEqual(methods["ours"]["reproduction_status"], "prototype")
        self.assertEqual(len(methods), 7)

    def test_formal_profile_does_not_relabel_observed_smoke_data(self):
        result = self.invoke("full", "--dry-run")
        self.assertEqual(result.returncode, 2)
        self.assertIn("unregistered", result.stderr)

    def test_modified_dataset_is_rejected_before_execution(self):
        self.assertTrue((ROOT / "data" / "bfcl-tasks.json").is_file(),
                        "frozen task data has not been migrated")
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "data"
            shutil.copytree(ROOT / "data", target)
            task_file = target / "bfcl-tasks.json"
            task_file.write_bytes(task_file.read_bytes() + b"\n")
            result = self.invoke("validate", "--data-dir", str(target))
            self.assertEqual(result.returncode, 2)
            self.assertIn("sha256", result.stderr)


if __name__ == "__main__":
    unittest.main()
