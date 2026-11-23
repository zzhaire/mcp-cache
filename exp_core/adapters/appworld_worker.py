"""AppWorld subprocess: separate Pydantic 1 environment, no model code execution."""
import json
import sys
from contextlib import redirect_stdout
from appworld import AppWorld

world = None
for line in sys.stdin:
    try:
        command = json.loads(line)
        credential_contract = None
        with redirect_stdout(sys.stderr):
            if command["action"] == "init":
                world = AppWorld(command["id"], experiment_name=command["run_id"],
                                 random_seed=20261116)
                value = {"instruction": world.task.instruction,
                         "supervisor": dict(world.task.supervisor)}
            elif command["action"] == "call":
                value = world.requester.request(
                    _app_name=command["app"], _api_name=command["api"], **command["arguments"])
                if command["api"] == "login" and isinstance(value, dict) and "access_token" in value:
                    # Verify through the original issuer, not an unverified JWT decode.
                    # AppWorld freezes time to task.datetime, so use its clock.
                    from appworld.common.datetime import DateTime
                    from appworld.common.utils import import_apis_module
                    manager = import_apis_module(command["app"]).logging_manager
                    claims = manager._get_payload(value["access_token"])
                    manager.get_current_user(value["access_token"])
                    credential_contract = {
                        "subject": claims["sub"], "scopes": ["account"],
                        "valid_for_seconds": max(0, claims["exp"] - DateTime.now().timestamp()),
                        "source": "verified AppWorld issuer, isolated task clock"}
                world.save_logs()
                world.save_state()
            elif command["action"] == "oracle":
                value = world.evaluate().to_dict()
            elif command["action"] == "close":
                world.close()
                value = {}
            else:
                raise ValueError("Unknown command")
        response = {"ok": True, "value": value}
        if credential_contract:
            response["credential_contract"] = credential_contract
        print(json.dumps(response, default=str), flush=True)
    except Exception as error:
        print(json.dumps({"ok": False, "error": str(error)}), flush=True)
