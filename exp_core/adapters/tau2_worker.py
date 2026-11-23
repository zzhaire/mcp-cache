"""Official tau2 telecom solo environment and strict original state evaluator."""
import json
from pathlib import Path
import sys
import uuid
from contextlib import redirect_stdout

# Third-party imports can log during registry initialization.
with redirect_stdout(sys.stderr):
    from tau2.agent.llm_agent import LLMSoloAgent
    from tau2.data_model.message import AssistantMessage, ToolCall
    from tau2.data_model.tasks import Task
    from tau2.domains.telecom.environment import get_environment
    from tau2.evaluator.evaluator_env import EnvironmentEvaluator

from exp_core.settings import DATA
for line in sys.stdin:
    try:
        command = json.loads(line)
        with redirect_stdout(sys.stderr):
            if command["action"] == "init":
                raw = next(x for x in json.loads((DATA / "tau2-tasks.json").read_text())
                           if x["id"] == command["id"])
                task = Task.model_validate(raw)
                environment = get_environment(solo_mode=True)
                initial = task.initial_state
                history = list(initial.message_history or []) if initial else []
                environment.set_state(
                    initialization_data=initial.initialization_data if initial else None,
                    initialization_actions=initial.initialization_actions if initial else None,
                    message_history=history)
                tools = environment.get_tools() + environment.get_user_tools()
                agent = LLMSoloAgent(tools=list(tools), domain_policy=environment.get_policy(),
                                     task=task, llm="not-used-for-inference")
                value = {
                    "instruction": agent.system_prompt,
                    "tools": [tool.openai_schema["function"] for tool in tools],
                    "readonly": {tool.name: not environment._is_mutating_tool(tool.name)
                                 for tool in tools},
                    "mode": "official telecom solo; JSON done substitutes the stop tool",
                    "executor": "tau2-bench v1.0.1",
                }
            elif command["action"] == "call":
                tool_call = ToolCall(id=str(uuid.uuid4()), name=command["name"],
                                     arguments=command["arguments"], requestor="assistant")
                response = environment.get_response(tool_call)
                history.extend([AssistantMessage(role="assistant", tool_calls=[tool_call]), response])
                if response.error:
                    raise ValueError(response.content)
                try:
                    value = json.loads(response.content)
                except (TypeError, json.JSONDecodeError):
                    value = response.content
            elif command["action"] == "oracle":
                value = EnvironmentEvaluator.calculate_reward(
                    get_environment, task, history, solo_mode=True,
                    strict_replay=True).model_dump(mode="json")
                value["success"] = value["reward"] == 1
            elif command["action"] == "close":
                value = {}
            else:
                raise ValueError("Unknown command")
        print(json.dumps({"ok": True, "value": value}, default=str), flush=True)
    except Exception as error:
        print(json.dumps({"ok": False, "error": str(error)}), flush=True)
