"""tau2-bench telecom (official environment, solo mode) as a lab domain.

Two apps: ``crm`` holds the agent-side tools (customers, lines, bills),
``device`` the customer's phone. They are coupled - suspending a line changes
what the phone shows, paying a bill changes the account - and the contract
says so. Phone state is ``volatile``: the customer can change it at any time.
"""
from exp_core.lab.domains import Domain, _tool
from exp_core.lab.domains._worker import shared
from exp_core.settings import TAU2_PYTHON, TAU2_ROOT

DEVICE = "device/state"
# crm/all: touched by actions whose target account the caller does not name.
CRM = {
    "get_customer_by_phone": ("read", "slow", ["crm/customers", "crm/all"]),
    "get_customer_by_id": ("read", "slow", ["crm/customer/{customer_id}", "crm/all"]),
    "get_customer_by_name": ("read", "slow", ["crm/customers", "crm/all"]),
    "get_details_by_id": ("read", "slow", ["crm/object/{id}", "crm/all"]),
    "get_bills_for_customer": ("read", "slow", ["crm/bills/{customer_id}", "crm/all"]),
    "get_data_usage": ("read", "volatile", ["crm/usage/{line_id}", "crm/all"]),
    "suspend_line": ("write", None, ["crm/object/{line_id}", "crm/customer/{customer_id}",
                                     DEVICE]),
    "resume_line": ("write", None, ["crm/object/{line_id}", "crm/customer/{customer_id}",
                                    DEVICE]),
    "send_payment_request": ("write", None, ["crm/object/{bill_id}",
                                             "crm/bills/{customer_id}", DEVICE]),
    "enable_roaming": ("write", None, ["crm/object/{line_id}", DEVICE]),
    "disable_roaming": ("write", None, ["crm/object/{line_id}", DEVICE]),
    "refuel_data": ("write", None, ["crm/object/{line_id}", "crm/usage/{line_id}",
                                    "crm/bills/{customer_id}", DEVICE]),
    "transfer_to_human_agents": ("write", None, []),
}
KEYED = {"refuel_data"}


class Tau2Domain(Domain):
    login_apps = frozenset({"crm", "device"})
    event_kinds = frozenset({"set", "add", "call"})

    def __init__(self, task):
        self.worker = shared("tau2", TAU2_PYTHON, "exp_core.adapters.tau2_domain_worker",
                             {"TAU2_DATA_DIR": str(TAU2_ROOT / "data")})
        info = self.worker.request("init", id=task["source"]["task_id"])
        self.instructions = info["policy"]
        self.ticket = info["ticket"]
        self.tools = {}
        for tool in info["tools"]:
            name, schema = tool["name"], tool["parameters"]
            if tool["side"] == "crm":
                kind, fresh, resources = CRM[name]
            elif name == "make_payment":  # the customer pays: the account changes too
                kind, fresh, resources = "write", None, [DEVICE, "crm/all"]
            else:
                kind = "write" if tool["mutating"] else "read"
                fresh, resources = (None if tool["mutating"] else "volatile"), [DEVICE]
            spec = _tool(tool["side"], kind, tool["description"], schema.get("properties", {}),
                         fresh=fresh, reads=resources if kind == "read" else (),
                         affects=resources if kind == "write" else (), key=name in KEYED)
            spec["required"] = list(schema.get("required", []))
            self.tools[name] = spec

    def call(self, name, arguments, shadow=False):
        return self.worker.request("shadow" if shadow else "call", name=name,
                                   arguments=arguments)

    def apply(self, event):
        self.worker.request("event", event=event)

    def state(self):
        return self.worker.request("state")
