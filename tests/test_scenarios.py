import hashlib
import json
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from data import select_cases  # noqa: E402

DATA = HERE / "data"
SCENARIOS = DATA / "scenarios-36.json"

REQUIRED_CASE_FIELDS = {
    "case_id",
    "dataset",
    "task_id",
    "scenario",
    "split",
    "augmented",
    "augmentation",
    "source_task_id",
    "expected_modules",
    "selection_reason",
    "augmentation_rule",
    "task_file_sha256",
}
EXPECTED_MODULES = {
    ("appworld", "discovery"): ["M1"],
    ("appworld", "read"): ["M2"],
    ("appworld", "auth_operation"): ["M3"],
    ("bfcl", "discovery"): ["M1", "M2"],
    ("bfcl", "read"): ["M2"],
    ("bfcl", "auth_operation"): ["M4"],
    ("tau2", "discovery"): ["M1", "M2"],
    ("tau2", "read"): ["M2"],
    ("tau2", "auth_operation"): ["M4"],
}
PERFORMANCE_KEYS = {
    "task_success",
    "task_time",
    "task_time_s",
    "active_time_s",
    "mcp_requests",
    "cache_hits",
    "m1_hits",
    "m2_hits",
    "m3_hits",
    "m4_hits",
    "input_tokens",
    "output_tokens",
    "cached_input_tokens",
    "publish_bytes",
    "retries",
}


def load_manifest():
    return json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))


def load_scenarios():
    return json.loads(SCENARIOS.read_text(encoding="utf-8"))


def load_tasks(dataset):
    return json.loads((DATA / select_cases.TASK_FILES[dataset]).read_text(encoding="utf-8"))


def walk_keys(value):
    if isinstance(value, dict):
        for key, child in value.items():
            yield key
            yield from walk_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk_keys(child)


class Scenarios36Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.document = load_scenarios()
        cls.cases = cls.document["cases"]
        cls.manifest = load_manifest()
        cls.tasks = {
            dataset: {task["id"]: task for task in load_tasks(dataset)}
            for dataset in select_cases.DATASETS
        }

    def test_top_level_fields(self):
        for field in ("version", "seed", "generated_at", "manifest_sha256", "cells", "cases"):
            self.assertIn(field, self.document)
        manifest_bytes = (DATA / "manifest.json").read_bytes()
        self.assertEqual(self.document["manifest_sha256"],
                         hashlib.sha256(manifest_bytes).hexdigest())
        self.assertEqual(self.document["generated_at"], "2026-11-16T00:00:00+08:00")

    def test_36_cases_and_nine_cells(self):
        self.assertEqual(len(self.cases), 36)
        self.assertEqual(len(self.document["cells"]), 9)
        cell_keys = {(cell["dataset"], cell["scenario"]) for cell in self.document["cells"]}
        self.assertEqual(cell_keys, set(EXPECTED_MODULES))

    def test_each_cell_has_two_dev_and_two_heldout(self):
        by_cell = {}
        for case in self.cases:
            by_cell.setdefault((case["dataset"], case["scenario"]), []).append(case)
        self.assertEqual(set(by_cell), set(EXPECTED_MODULES))
        for key, cases in by_cell.items():
            self.assertEqual(len(cases), 4, key)
            splits = [case["split"] for case in cases]
            self.assertEqual(splits.count("dev"), 2, key)
            self.assertEqual(splits.count("heldout"), 2, key)
        for cell in self.document["cells"]:
            self.assertEqual(cell["counts"], {"total": 4, "dev": 2, "heldout": 2})
            self.assertEqual(len(cell["case_ids"]), 4)

    def test_case_fields_are_complete(self):
        for case in self.cases:
            self.assertEqual(set(case), REQUIRED_CASE_FIELDS, case["case_id"])
            self.assertIsInstance(case["selection_reason"], str)
            self.assertTrue(case["selection_reason"], case["case_id"])
            self.assertIsInstance(case["augmentation_rule"], str)
            self.assertTrue(case["augmentation_rule"], case["case_id"])

    def test_case_ids_are_unique_and_stable(self):
        pattern = re.compile(
            r"^(appworld|bfcl|tau2)-(discovery|read|auth_operation)-(dev|heldout)-([12])$")
        case_ids = [case["case_id"] for case in self.cases]
        self.assertEqual(len(case_ids), len(set(case_ids)))
        for case in self.cases:
            match = pattern.match(case["case_id"])
            self.assertIsNotNone(match, case["case_id"])
            self.assertEqual(match.group(1), case["dataset"])
            self.assertEqual(match.group(2), case["scenario"])
            self.assertEqual(match.group(3), case["split"])
        for cell in self.document["cells"]:
            suffixes = [case_id.split("-")[-1] for case_id in cell["case_ids"]]
            self.assertEqual(sorted(suffixes), ["1", "1", "2", "2"])

    def test_task_ids_exist_in_frozen_files_and_declared_split(self):
        for case in self.cases:
            tasks = self.tasks[case["dataset"]]
            self.assertIn(case["task_id"], tasks, case["case_id"])
            declared = self.manifest["datasets"][case["dataset"]][case["split"]]
            self.assertIn(case["task_id"], declared, case["case_id"])

    def test_identity_and_uniform_augmentation_state(self):
        texts = set()
        for case in self.cases:
            self.assertTrue(case["augmented"], case["case_id"])
            self.assertEqual(case["augmentation"], select_cases.AUGMENTATION_TEXT,
                             case["case_id"])
            self.assertEqual(case["source_task_id"], case["task_id"], case["case_id"])
            texts.add(case["augmentation"])
        self.assertEqual(len(texts), 1)

    def test_expected_modules(self):
        for case in self.cases:
            expected = EXPECTED_MODULES[(case["dataset"], case["scenario"])]
            self.assertEqual(case["expected_modules"], expected, case["case_id"])

    def test_task_file_hashes_match_actual_files(self):
        for case in self.cases:
            path = DATA / select_cases.TASK_FILES[case["dataset"]]
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(case["task_file_sha256"], actual, case["case_id"])
            declared = self.manifest["datasets"][case["dataset"]]["task_file_sha256"]
            self.assertEqual(case["task_file_sha256"], declared, case["case_id"])

    def test_no_performance_fields(self):
        keys = set(walk_keys(self.document))
        self.assertFalse(keys & PERFORMANCE_KEYS, keys & PERFORMANCE_KEYS)

    def test_repeated_runs_are_byte_identical(self):
        with tempfile.TemporaryDirectory() as temp:
            outputs = [Path(temp) / "first.json", Path(temp) / "second.json"]
            for output in outputs:
                subprocess.run(
                    [sys.executable, str(HERE / "run"), "select", "--output", str(output)],
                    check=True,
                    cwd=str(HERE),
                    capture_output=True,
                )
            first = outputs[0].read_bytes()
            second = outputs[1].read_bytes()
            self.assertEqual(first, second)
            self.assertEqual(first, SCENARIOS.read_bytes())


if __name__ == "__main__":
    unittest.main()
