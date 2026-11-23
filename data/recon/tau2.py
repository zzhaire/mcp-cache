"""Rebuild the selected tau2 telecom tasks on the lab server layer.

The original task is one support ticket with a list of required actions and
no reads. The rebuilt first turn is that ticket: the agent looks the customer
up, inspects the line and the phone, performs the original actions in their
original order and looks at the phone again. Overlays add the rest.
"""
import json

from data.recon.common import PASSWORD, USER, login_step
from data.recon.overlays import overlays
from exp_core.lab.domains.tau2 import Tau2Domain
from exp_core.settings import DATA

CREDENTIALS = (f"Credentials for every app: username {USER}, password {PASSWORD}. "
               "Sign in to an app before using it. ")
# A second application changes the outcome: a toggle undoes itself, a refuel
# is billed twice.
NON_IDEMPOTENT = {"refuel_data", "toggle_airplane_mode", "toggle_data", "toggle_roaming",
                  "toggle_data_saver_mode", "toggle_wifi_calling", "toggle_wifi"}


KEYED = {"refuel_data"}
_NETWORK = {"name": "check_network_status", "arguments": {}}
# The read that shows whether an unkeyed toggle was applied.
VERIFY = {"toggle_airplane_mode": _NETWORK, "toggle_data": _NETWORK,
          "toggle_roaming": _NETWORK,
          "toggle_data_saver_mode": {"name": "check_data_restriction_status", "arguments": {}},
          "toggle_wifi_calling": {"name": "check_wifi_calling_status", "arguments": {}},
          "toggle_wifi": {"name": "check_wifi_status", "arguments": {}}}


def identity(entry, domain):
    """``(phone, customer id, line id)`` of the ticket's customer."""
    actions = (entry.get("initial_state") or {}).get("initialization_actions") or []
    phone = next((a["arguments"]["phone_number"] for a in actions
                  if a["func_name"] == "set_user_info"), "555-123-2002")
    customer = domain.call("get_customer_by_phone", {"phone_number": phone}, shadow=True)
    line = next(line for line in customer["line_ids"]
                if domain.call("get_details_by_id", {"id": line}, shadow=True)
                .get("phone_number") == phone)
    return phone, customer["customer_id"], line


def base_task(entry):
    task = {"id": "recon-tau2-" + str(entry["_index"]), "domain": "tau2",
            "source": {"dataset": "tau2", "task_id": entry["id"]}, "initial": {},
            "users": {USER: PASSWORD}}
    domain = Tau2Domain(task)
    phone, customer, line = identity(entry, domain)
    look = [[{"name": "get_customer_by_phone", "arguments": {"phone_number": phone}}],
            [{"name": "get_details_by_id", "arguments": {"id": line}},
             {"name": "get_data_usage", "arguments": {"customer_id": customer,
                                                      "line_id": line}},
             {"name": "check_status_bar", "arguments": {}},
             {"name": "check_network_status", "arguments": {}}]]
    actions = [[{"name": a["name"], "arguments": a["arguments"]}]
               for a in entry["evaluation_criteria"]["actions"]]
    after = [[{"name": "check_status_bar", "arguments": {}}]]
    task["turns"] = [CREDENTIALS + "You are the support agent for this ticket; you can "
                     "also operate the customer's phone through the device tools. "
                     + domain.ticket]
    task["gold"] = [{"steps": [login_step(domain.login_apps), *look, *actions, *after],
                     "answer": "done"}]
    task["identity"] = {"phone": phone, "customer_id": customer, "line_id": line}
    return task, domain


def probes(task):
    customer, line = task["identity"]["customer_id"], task["identity"]["line_id"]
    return [
        {"call": {"name": "get_data_usage",
                  "arguments": {"customer_id": customer, "line_id": line}},
         "field": ["data_used_gb"], "label": "used", "fresh": "volatile", "app": "crm",
         "ask": f"how many GB line {line} has used in this billing cycle",
         # The customer keeps using data.
         "event": {"kind": "add", "path": ["crm", "lines", {"line_id": line}, "data_used_gb"],
                   "value": 0.5}},
        # A value only the line's own record holds: the usage read (volatile)
        # does not return it, so an agent cannot get it by another route.
        {"call": {"name": "get_details_by_id", "arguments": {"id": line}},
         "field": ["contract_end_date"], "label": "contract", "fresh": "slow", "app": "crm",
         "ask": f"the date the contract of line {line} ends",
         # A colleague extends the contract from another console.
         "event": {"kind": "set",
                   "path": ["crm", "lines", {"line_id": line}, "contract_end_date"],
                   "value": "2027-06-30"}},
        {"call": {"name": "check_status_bar", "arguments": {}},
         "field": [], "label": "bar", "fresh": "volatile", "app": "device",
         "ask": "what the phone's status bar shows",
         # The customer flips airplane mode on their own.
         "event": {"kind": "call", "name": "toggle_airplane_mode", "arguments": {}}},
    ]


def build():
    from data.recon.common import reference
    entries = json.loads((DATA / "tau2-tasks.json").read_text())
    manifest = json.loads((DATA / "manifest.json").read_text())["datasets"]["tau2"]
    tasks, cases = [], []
    for index, entry in enumerate(entries):
        base, domain = base_task({**entry, "_index": index})
        errors = len(reference({**base, "events": []})["errors"])  # the original's own
        writes = [(t, s, call["name"]) for t, turn in enumerate(base["gold"])
                  for s, step in enumerate(turn["steps"]) for call in step
                  if domain.tools.get(call["name"], {}).get("kind") == "write"]
        split = "dev" if entry["id"] in manifest["dev"] else "heldout"
        for task in overlays(base, probes(base), writes, NON_IDEMPOTENT, errors,
                             keyed=KEYED, verify=VERIFY):
            tasks.append(task)
            cases.append({"case_id": task["id"], "dataset": "lab", "task_id": task["id"],
                          "scenario": task["family"], "split": split,
                          "source_dataset": "tau2", "source_task_id": entry["id"],
                          "augmented": False, "augmentation": None})
    return tasks, cases
