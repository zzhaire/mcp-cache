"""Explicit method registry; implementations live in related_work/ and ours/."""
import copy
from exp_core.policies import make_policy, policy_sha256
from related_work.openai import method as openai
from related_work.fastmcp import method as fastmcp
from related_work.gateway import method as gateway
from related_work.mehan import method as mehan
from related_work.merchant import method as merchant
from related_work.strong import method as strong
from ours import method as ours

ARM_NAMES = ("bare", "rw_openai", "rw_fastmcp", "rw_gateway", "rw_mehan",
             "rw_merchant", "ours")
# Arms that can be asked for by name but are not part of the frozen default
# matrix: the capability-equal baseline used on the lab server.
EXTRA_ARMS = ("strong",)
LABELS = ("baseline", "官方原生", "Host 侧适配", "本文方法")
ARMS = {"bare": make_policy(
    "bare", "baseline", "禁用 M1–M4、状态提供与服务器缓存。",
    ("M1", "M2", "M3", "M4"), ttl_seconds=0)}
METHOD_INFO = {"bare": {
    "name": "bare", "method_id": "bare", "reproduction_status": "reference",
    "source": "shared task runner", "scope": "无复用参考；共同执行协议相同"}}
for module in (openai, fastmcp, gateway, mehan, merchant, strong, ours):
    ARMS[module.POLICY["arm"]] = module.POLICY
    METHOD_INFO[module.POLICY["arm"]] = module.INFO
ARM_ALIASES = {info["name"]: name for name, info in METHOD_INFO.items()}
ARM_ALIASES["fixed"] = "rw_merchant"


def resolve_arm(name):
    try:
        return copy.deepcopy(ARMS[ARM_ALIASES.get(name, name)])
    except KeyError:
        raise KeyError(f"unknown method: {name}") from None


def describe_methods():
    return [copy.deepcopy(METHOD_INFO[name]) for name in ARM_NAMES]
