"""tau2 telecom environment for the lab server layer (runs in the tau2 venv).

Unlike ``tau2_worker`` this keeps no conversation history and uses no tau2
evaluator: the lab oracle compares the whole environment state. One process
serves many tasks; ``init`` replaces the environment.
"""
import json
import sys
from contextlib import redirect_stdout

with redirect_stdout(sys.stderr):
    from pydantic import BaseModel
    from tau2.data_model.tasks import Task
    from tau2.domains.telecom.environment import get_environment

from exp_core.settings import DATA

TASKS = {}
env = None


def plain(value):
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    return value


def locate(path):
    """Follow ``path`` from a database root; a dict step selects the list item
    whose fields match it. Returns ``(holder, last step)``."""
    node = env.tools.db if path[0] == "crm" else env.user_tools.db
    for step in path[1:-1]:
        if isinstance(step, dict):
            node = next(item for item in node
                        if all(getattr(item, k) == v for k, v in step.items()))
        elif isinstance(node, dict):
            node = node[step]
        else:
            node = getattr(node, step)
    return node, path[-1]


def handle(command):
    global env
    action = command["action"]
    if action == "init":
        if not TASKS:
            TASKS.update({x["id"]: x for x in json.loads((DATA / "tau2-tasks.json").read_text())})
        task = Task.model_validate(TASKS[command["id"]])
        env = get_environment(solo_mode=True)
        initial = task.initial_state
        env.set_state(
            initialization_data=initial.initialization_data if initial else None,
            initialization_actions=initial.initialization_actions if initial else None,
            message_history=[])
        user = {tool.name for tool in env.get_user_tools()}
        return {"policy": env.get_policy(), "ticket": task.ticket,
                "tools": [{**tool.openai_schema["function"],
                           "mutating": env._is_mutating_tool(tool.name),
                           "side": "device" if tool.name in user else "crm"}
                          for tool in [*env.get_tools(), *env.get_user_tools()]]}
    if action in ("call", "shadow"):
        value = env.make_tool_call(command["name"], requestor="assistant",
                                   **command["arguments"])
        if action == "call":
            env.sync_tools()
        return plain(value)
    if action == "event":
        event = command["event"]
        if event["kind"] == "call":  # someone else uses the same service
            env.make_tool_call(event["name"], requestor="assistant", **event["arguments"])
        else:
            holder, last = locate(event["path"])
            current = holder[last] if isinstance(holder, dict) else getattr(holder, last)
            new = event["value"] if event["kind"] == "set" else current + event["value"]
            if isinstance(holder, dict):
                holder[last] = new
            else:
                setattr(holder, last, new)
        env.sync_tools()
        return {}
    if action == "state":
        return {"crm": env.tools.db.model_dump(mode="json"),
                "device": env.user_tools.db.model_dump(mode="json")}
    raise ValueError(f"unknown action {action}")


for line in sys.stdin:
    command = json.loads(line)
    try:
        with redirect_stdout(sys.stderr):
            value = handle(command)
        print(json.dumps({"ok": True, "value": value}, default=str), flush=True)
    except Exception as error:  # noqa: BLE001 - reported to the caller
        refused = command.get("action") in ("call", "shadow")
        print(json.dumps({"ok": False, "refused": refused,
                          "error": f"{type(error).__name__}: {error}"}), flush=True)
