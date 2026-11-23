"""Safety and accounting checks for the smoke runtime, not benchmark evidence."""
import unittest
from exp_core.runtime import Runtime


class RuntimeSafety(unittest.TestCase):
    def test_catalog_requires_same_version_and_identity(self):
        runtime = Runtime()
        calls = []
        fetch = lambda: calls.append(1) or ["read"]
        runtime.catalog("server", "v1", fetch)
        runtime.catalog("server", "v1", fetch)
        runtime.catalog("server", "v2", fetch)
        runtime.auth_epoch += 1
        runtime.catalog("server", "v2", fetch)
        self.assertEqual(len(calls), 3)

    def test_expired_or_revoked_credential_is_unavailable(self):
        now = [0]
        runtime = Runtime(clock=lambda: now[0])
        runtime.save_credential("service", "alice", ["read"], "opaque", valid_until=10)
        self.assertEqual(runtime.credential("service", "alice", ["read"]), "opaque")
        self.assertIsNone(runtime.credential("service", "bob", ["read"]))
        runtime.auth_epoch += 1
        self.assertIsNone(runtime.credential("service", "alice", ["read"]))
        runtime.save_credential("service", "alice", ["read"], "opaque", valid_until=10)
        now[0] = 11
        self.assertIsNone(runtime.credential("service", "alice", ["read"]))

    def test_default_never_reuses_undeclared_result(self):
        runtime = Runtime()
        calls = []
        for _ in range(2):
            runtime.call("read", {}, lambda: calls.append(1) or {"x": 1})
        self.assertEqual(len(calls), 2)

    def test_read_contract_reuses_until_write_or_auth_change(self):
        runtime = Runtime()
        calls = []
        fetch = lambda: calls.append(1) or {"value": len(calls)}
        self.assertEqual(runtime.call("read", {}, fetch, readonly=True), {"value": 1})
        self.assertEqual(runtime.call("read", {}, fetch, readonly=True), {"value": 1})
        runtime.call("write", {}, lambda: "ok")
        self.assertEqual(runtime.call("read", {}, fetch, readonly=True), {"value": 2})
        runtime.auth_epoch += 1
        self.assertEqual(runtime.call("read", {}, fetch, readonly=True), {"value": 3})

    def test_expired_read_refetches(self):
        now = [0]
        runtime = Runtime(clock=lambda: now[0], ttl=1)
        calls = []
        fetch = lambda: calls.append(1) or len(calls)
        runtime.call("read", {}, fetch, readonly=True)
        now[0] = 2
        self.assertEqual(runtime.call("read", {}, fetch, readonly=True), 2)

    def test_distinct_identical_writes_are_executed(self):
        runtime = Runtime()
        calls = []
        fetch = lambda: calls.append(1) or len(calls)
        runtime.call("write", {}, fetch, operation_id="one")
        runtime.call("write", {}, fetch, operation_id="two")
        self.assertEqual(len(calls), 2)

    def test_same_operation_result_is_not_reexecuted(self):
        runtime = Runtime()
        calls = []
        fetch = lambda: calls.append(1) or {"completed": True}
        runtime.call("write", {}, fetch, operation_id="one")
        runtime.call("write", {}, fetch, operation_id="one")
        self.assertEqual(len(calls), 1)

    def test_operation_id_collision_is_rejected(self):
        runtime = Runtime()
        runtime.call("write", {"x": 1}, lambda: "ok", operation_id="one")
        with self.assertRaises(ValueError):
            runtime.call("write", {"x": 2}, lambda: "ok", operation_id="one")

    def test_errors_are_not_successfully_cached(self):
        runtime = Runtime()
        calls = []
        def fetch():
            calls.append(1)
            raise RuntimeError("temporary failure")
        for _ in range(2):
            with self.assertRaises(RuntimeError):
                runtime.call("read", {}, fetch, readonly=True, operation_id="one")
        self.assertEqual(len(calls), 2)


if __name__ == "__main__":
    unittest.main()
