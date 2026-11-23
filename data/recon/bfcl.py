"""Rebuild the selected BFCL multi-turn tasks on the lab server layer.

``./run recon-build bfcl`` writes ``data/recon-bfcl-tasks.json`` and
``data/recon-bfcl-cases.json``. Each original task keeps its backend state,
user turns and reference calls; ``data.recon.overlays`` adds the scenarios.
"""
import ast
import json

from data.recon.common import PASSWORD, USER, login_step
from data.recon.overlays import overlays
from exp_core.lab.domains.bfcl import APPS, PUBLIC_APPS, BfclDomain
from exp_core.settings import DATA

CREDENTIALS = (f"Credentials for every app: username {USER}, password {PASSWORD}. "
               "Sign in to an app before using it. ")
# Writes a second application of which changes the outcome.
NON_IDEMPOTENT = {"book_flight", "purchase_insurance", "place_order", "fund_account",
                  "withdraw_funds", "send_message", "create_ticket", "fillFuelTank"}


KEYED = {"book_flight", "purchase_insurance", "place_order", "fund_account", "withdraw_funds"}
# The read that shows whether an unkeyed write was applied.
VERIFY = {"send_message": {"name": "view_messages_sent", "arguments": {}},
          "create_ticket": {"name": "get_user_tickets", "arguments": {}},
          "fillFuelTank": {"name": "displayCarStatus", "arguments": {"option": "fuel"}}}


def parse(text):
    node = ast.parse(text, mode="eval").body
    return {"name": node.func.id,
            "arguments": {k.arg: ast.literal_eval(k.value) for k in node.keywords}}


def base_task(entry, truth):
    apps = {APPS[c] for c in entry["involved_classes"]} - PUBLIC_APPS
    gold = []
    for index, calls in enumerate(truth):
        steps = [[parse(call)] for call in calls]
        if index == 0:
            steps.insert(0, login_step(apps))
        gold.append({"steps": steps, "answer": "done"})
    turns = [" ".join(m["content"] for m in turn) for turn in entry["question"]]
    turns[0] = CREDENTIALS + turns[0]
    return {"id": "recon-bfcl-" + entry["id"].rsplit("_", 1)[-1], "domain": "bfcl",
            "source": {"dataset": "bfcl", "task_id": entry["id"]},
            "involved_classes": entry["involved_classes"],
            "excluded_function": entry.get("excluded_function", []),
            "initial": entry["initial_config"], "users": {USER: PASSWORD},
            "turns": turns, "gold": gold}


def _argument(gold, name):
    for turn in gold:
        for step in turn["steps"]:
            for call in step:
                if name in call["arguments"]:
                    return call["arguments"][name]
    return None


def probes(task):
    """Reads of this task's own state whose value a user could ask for."""
    domain = BfclDomain({**task})
    out = []

    def probe(name, arguments, field, label, ask, fresh, app, event=None):
        if name in domain.tools:
            out.append({"call": {"name": name, "arguments": arguments}, "field": field,
                        "label": label, "ask": ask, "fresh": fresh, "app": app,
                        "event": event})

    trading = domain.instances.get("TradingBot")
    if trading:
        symbol = _argument(task["gold"], "symbol") or sorted(trading.stocks)[0]
        if symbol in trading.stocks:
            probe("get_stock_info", {"symbol": symbol}, ["price"], symbol,
                  f"the current price of {symbol}", "volatile", "trading",
                  {"kind": "add", "target": "TradingBot", "path": ["stocks", symbol, "price"],
                   "value": 7.5})
        probe("get_account_info", {}, ["balance"], "balance",
              "my trading account balance", "slow", "trading",
              {"kind": "add", "target": "TradingBot", "path": ["account_info", "balance"],
               "value": 500})
    travel = domain.instances.get("TravelAPI")
    if travel:
        token = _argument(task["gold"], "access_token") or travel.access_token
        card = _argument(task["gold"], "card_id") or next(iter(travel.credit_card_list), None)
        if token and card in travel.credit_card_list:
            probe("get_credit_card_balance", {"access_token": token, "card_id": card},
                  ["card_balance"], card, f"the balance of credit card {card}", "slow",
                  "travel", {"kind": "add", "target": "TravelAPI",
                             "path": ["credit_card_list", card, "balance"], "value": -120})
    ticket = domain.instances.get("TicketAPI")
    if ticket:
        known = [t for t in ticket.ticket_queue if isinstance(t.get("id"), int)
                 and isinstance(t.get("priority"), int)]
        if known:
            index = ticket.ticket_queue.index(known[0])
            number, priority = known[0]["id"], known[0]["priority"]
            probe("get_ticket", {"ticket_id": number}, ["priority"], "priority",
                  f"the priority of ticket {number}", "slow", "ticket",
                  {"kind": "set", "target": "TicketAPI",
                   "path": ["ticket_queue", index, "priority"],
                   "value": 4 if priority == 5 else 5})
    vehicle = domain.instances.get("VehicleControlAPI")
    if vehicle:
        probe("displayCarStatus", {"option": "fuel"}, ["fuelLevel"], "fuel",
              "the fuel level of the car", "volatile", "vehicle",
              {"kind": "add", "target": "VehicleControlAPI", "path": ["fuelLevel"],
               "value": 2.5 if vehicle.fuelLevel < 40 else -2.5})
        probe("check_tire_pressure", {}, ["frontLeftTirePressure"], "tire",
              "the front left tire pressure", "volatile", "vehicle")
    message = domain.instances.get("MessageAPI")
    if message:
        probe("get_message_stats", {}, ["stats", "total_contacts"], "contacts",
              "how many contacts I have messaged", "slow", "message")
    if domain.instances.get("GorillaFileSystem"):
        probe("pwd", {}, ["current_working_directory"], "directory",
              "the current working directory", "slow", "fs")
    return out


def write_calls(task):
    domain = BfclDomain({**task})
    return [(t, s, call["name"]) for t, turn in enumerate(task["gold"])
            for s, step in enumerate(turn["steps"]) for call in step
            if domain.tools.get(call["name"], {}).get("kind") == "write"]


def build():
    from data.recon.common import reference
    entries = json.loads((DATA / "bfcl-tasks.json").read_text())
    oracle = json.loads((DATA / "bfcl-oracle.json").read_text())
    manifest = json.loads((DATA / "manifest.json").read_text())["datasets"]["bfcl"]
    tasks, cases = [], []
    for entry in entries:
        base = base_task(entry, oracle[entry["id"]]["ground_truth"])
        errors = len(reference({**base, "events": []})["errors"])  # the original's own
        split = "dev" if entry["id"] in manifest["dev"] else "heldout"
        for task in overlays(base, probes(base), write_calls(base), NON_IDEMPOTENT, errors,
                             keyed=KEYED, verify=VERIFY):
            tasks.append(task)
            cases.append({"case_id": task["id"], "dataset": "lab", "task_id": task["id"],
                          "scenario": task["family"], "split": split,
                          "source_dataset": "bfcl", "source_task_id": entry["id"],
                          "augmented": False, "augmentation": None})
    return tasks, cases
