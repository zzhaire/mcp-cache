"""Real MCP module coverage, with explicit controlled cases and business assertions."""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import subprocess
import sys
import time
import uuid
import jsonschema
from exp_core.access import Access
from exp_core.runner import MCP, free_port, wait_server
from exp_core.runtime import AccessFailure, Runtime
from exp_core.settings import ROOT, CONTRACTS, SCRATCH, RESULTS



@contextmanager
def service(dataset, task, *fault_flags):
    SCRATCH.mkdir(parents=True, exist_ok=True)
    run_id, port = uuid.uuid4().hex, free_port()
    trace = SCRATCH / f"foundation-{run_id}.jsonl"
    with (SCRATCH / f"foundation-{run_id}.server.log").open("w") as log:
        process = subprocess.Popen([
            sys.executable, "-m", "exp_core.adapters.server", "--port", str(port),
            "--delay-ms", "20", "--jitter-ms", "5", "--seed", "20261116",
            *fault_flags], stdout=log, stderr=log, cwd=ROOT)
        client = MCP(port, trace)
        try:
            wait_server(process, port)
            client.initialize()
            info = client.tool("benchmark_init", {
                "dataset": dataset, "task_id": task, "run_id": run_id}, control=True)
            yield client, info, trace
        finally:
            try:
                client.tool("benchmark_close", {}, control=True)
            except Exception:
                pass
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def action(access, app, api, arguments, operation_id=None):
    return access.call("api_call", {"app": app, "api": api, "arguments": arguments},
                       operation_id or str(uuid.uuid4()))


def verify():
    cases, audits = [], []
    with service("appworld", "e7a10f8_2") as (client, info, trace):
        now = [time.monotonic()]
        runtime = Runtime(clock=lambda: now[0], ttl=60)
        access = Access(runtime, "appworld", "isolated-appworld-task",
                        lambda name, args: client.tool("invoke", {"name": name, "arguments": args}))
        doc_args = {"app_name": "spotify", "api_name": "show_playlist_library"}
        a = action(access, "api_docs", "show_api_doc", doc_args)
        before = client.metrics["mcp_requests"]
        b = action(access, "api_docs", "show_api_doc", doc_args)
        assert a == b and client.metrics["mcp_requests"] == before
        assert runtime.stats["M1_hits"] == 1
        cases.append({"module": "M1", "case": "original_api_docs_reuse", "passed": True})
        # Version is declared by the adapter. Refetch is checked over real HTTP.
        fetch = lambda: client.rpc("tools/list")
        runtime.catalog(client.url, "v1", fetch)
        runtime.catalog(client.url, "v1", fetch)
        before = client.metrics["mcp_requests"]
        runtime.catalog(client.url, "v2", fetch)
        assert client.metrics["mcp_requests"] == before + 1
        cases.append({"module": "M1", "case": "declared_version_change_refetches", "passed": True})
        passwords = action(access, "supervisor", "show_account_passwords", {})
        if isinstance(passwords, list):
            password = next(x["password"] for x in passwords if x.get("account_name") == "spotify")
        else:
            password = passwords["spotify"]
        login = {"username": info["supervisor"]["email"], "password": password}
        first = action(access, "spotify", "login", login)
        assert runtime.credentials, "Real issuer contract was not produced"
        before = client.metrics["mcp_requests"]
        second = action(access, "spotify", "login", login)
        assert first == second and client.metrics["mcp_requests"] == before
        assert runtime.stats["M3_hits"] == 1
        cases.append({"module": "M3", "case": "issuer_verified_login_reused", "passed": True})
        token = first["access_token"]
        query = {"access_token": token, "page_index": 0, "page_limit": 20}
        playlists = action(access, "spotify", "show_playlist_library", query)
        assert isinstance(playlists, list) and playlists, "No actual playlists were read"
        before = client.metrics["mcp_requests"]
        assert action(access, "spotify", "show_playlist_library", query) == playlists
        assert client.metrics["mcp_requests"] == before and runtime.stats["M2_hits"] == 1
        cases.append({"module": "M2", "case": "original_authenticated_data_read_reused", "passed": True})
        now[0] += 61
        before = client.metrics["mcp_requests"]
        action(access, "spotify", "show_playlist_library", query)
        assert client.metrics["mcp_requests"] == before + 1
        cases.append({"module": "M2", "case": "read_expiry_refetches", "passed": True})
        # Conservative local cache expiry: the issuer clock remains task-frozen.
        now[0] = max(value[1] for value in runtime.credentials.values()) + 1
        before = client.metrics["mcp_requests"]
        fresh = action(access, "spotify", "login", login)
        assert client.metrics["mcp_requests"] == before + 1
        cases.append({"module": "M3", "case": "credential_cache_expiry_refetches", "passed": True,
                      "clock": "local cache clock exceeds issuer-declared lifetime"})
        action(access, "spotify", "logout", {"access_token": fresh["access_token"]})
        assert not runtime.credentials and not runtime.results
        try:
            action(access, "spotify", "show_playlist_library",
                   dict(query, access_token=fresh["access_token"]))
        except AccessFailure:
            pass
        else:
            raise AssertionError("Original issuer accepted a logged-out credential")
        action(access, "spotify", "login", login)
        cases.append({"module": "M3", "case": "logout_clears_store_and_server_rejects_old_token",
                      "passed": True})
        cases.append({"module": "M2", "case": "authorization_change_clears_data", "passed": True})
        audits.append({"dataset": "appworld", "runtime": runtime.stats,
                       "http": client.metrics.copy(), "operations": runtime.operation_audit(),
                       "trace": str(trace)})

    with service("bfcl", "multi_turn_base_38", "--inject-transient-once",
                 "--inject-uncertain-write", "mkdir") as (client, _, trace):
        runtime = Runtime()
        access = Access(runtime, "bfcl", "isolated-bfcl-task",
                        lambda name, args: client.tool("invoke", {"name": name, "arguments": args}))
        operation = str(uuid.uuid4())
        value = access.call("pwd", {}, operation)
        assert value and runtime.operations[operation]["attempts"] == 2
        assert runtime.stats["retries"] == 1
        cases.append({"module": "M4", "case": "transient_retries_once_with_same_uuid",
                      "passed": True, "controlled_fault": "before original action execution"})
        missing = str(uuid.uuid4())
        for index in range(2):
            before = client.metrics["mcp_requests"]
            try:
                access.call("cat", {"file_name": "mcp_smoke_missing"}, missing)
            except AccessFailure:
                pass
            else:
                raise AssertionError("A missing file was cached as a successful read")
            if index:
                assert client.metrics["mcp_requests"] == before
        assert runtime.stats["M4_failure_hits"] == 1
        cases.append({"module": "M4", "case": "same_failed_operation_waits_for_condition_change",
                      "passed": True})
        access.call("touch", {"file_name": "mcp_smoke_missing"}, str(uuid.uuid4()))
        access.call("cat", {"file_name": "mcp_smoke_missing"}, missing)
        assert runtime.operations[missing]["state"] == "completed"
        cases.append({"module": "M4", "case": "resource_change_allows_recovery", "passed": True})
        write_op = str(uuid.uuid4())
        before = client.metrics["mcp_requests"]
        for _ in range(2):
            try:
                access.call("mkdir", {"dir_name": "mcp_smoke_new_directory"}, write_op)
            except AccessFailure:
                pass
            else:
                raise AssertionError("Unknown write outcome was not blocked")
        assert client.metrics["mcp_requests"] == before + 1
        inventory = access.call("ls", {"a": True}, str(uuid.uuid4()))
        assert "mcp_smoke_new_directory" in json.dumps(inventory)
        assert runtime.operations[write_op]["state"] == "uncertain"
        cases.append({"module": "M4", "case": "lost_write_response_is_not_reexecuted",
                      "passed": True, "controlled_fault": "after the original write committed"})
        audits.append({"dataset": "bfcl", "runtime": runtime.stats,
                       "http": client.metrics.copy(), "operations": runtime.operation_audit(),
                       "trace": str(trace)})
    schema = json.loads((CONTRACTS / "operation.schema.json").read_text())
    for audit in audits:
        for operation in audit["operations"]:
            jsonschema.validate(operation, schema)
    return {"status": "passed", "type": "controlled_real_MCP_coverage_not_natural_gain",
            "transport": "localhost Streamable HTTP", "model_calls": 0,
            "checks_per_run": len(cases), "executions_per_check_per_run": 1,
            "independent_environment_count": 2, "environment_reset_per_check": False,
            "seed": 20261116,
            "cases": cases, "audits": audits}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=RESULTS / "checks" / "foundation.json")
    args = parser.parse_args()
    start = time.monotonic()
    try:
        result = verify()
    except Exception as error:
        result = {"status": "failed", "error": str(error)}
    result["time_s"] = time.monotonic() - start
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps({"status": result["status"], "passed_cases": len(result.get("cases", [])),
                      "error": result.get("error"), "output": str(args.output)}), flush=True)
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
