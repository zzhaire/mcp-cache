"""BFCL multi-turn backends (bfcl-eval, unmodified) as a lab domain.

The original Python classes run as they are; this module only adds what the
benchmark does not have: which app a tool belongs to, whether it reads or
writes, how fresh a read stays and which resources it touches. Reads that draw
from the backend's random generator (sensor values, the clock) are ``strict``.
"""
import copy
import importlib
import inspect
import json
from pathlib import Path

from exp_core.lab.domains import BusinessError, Domain, _tool

APPS = {"TravelAPI": "travel", "VehicleControlAPI": "vehicle", "TicketAPI": "ticket",
        "TradingBot": "trading", "MathAPI": "math", "MessageAPI": "message",
        "GorillaFileSystem": "fs"}
PUBLIC_APPS = {"math"}


def R(fresh, *reads):
    return {"kind": "read", "fresh": fresh, "reads": reads}


def W(*affects, key=False):
    return {"kind": "write", "affects": affects, "key": key}


TRADING_ALL = ("trading/account", "trading/orders", "trading/watchlist",
               "trading/transactions")
CONTRACT = {
    # ---- travel: one aggregate ("cards") plus one resource per card
    "authenticate_travel": W("travel/token"),
    "book_flight": W("travel/bookings", "travel/cards", "travel/card/{card_id}", key=True),
    "cancel_booking": W("travel/bookings", "travel/cards", "travel/cards/refund"),
    "compute_exchange_rate": R("static"),
    "contact_customer_support": W(),
    "get_all_credit_cards": R("slow", "travel/cards"),
    "get_booking_history": R("slow", "travel/bookings"),
    "get_budget_fiscal_year": R("static"),
    "get_credit_card_balance": R("slow", "travel/card/{card_id}", "travel/cards/refund"),
    "get_flight_cost": R("static"),
    "get_nearest_airport_by_city": R("static"),
    "list_all_airports": R("static"),
    "purchase_insurance": W("travel/bookings", "travel/cards", "travel/card/{card_id}",
                            key=True),
    "register_credit_card": W("travel/cards"),
    "retrieve_invoice": R("slow", "travel/bookings"),
    "set_budget_limit": W("travel/budget"),
    "travel_get_login_status": R("strict"),
    "verify_traveler_information": R("static"),
    # ---- vehicle
    "activateParkingBrake": W("vehicle/state"),
    "adjustClimateControl": W("vehicle/state"),
    "check_tire_pressure": R("volatile", "vehicle/tires"),
    "displayCarStatus": R("volatile", "vehicle/state"),
    "display_log": W(),
    "estimate_distance": R("static"),
    "estimate_drive_feasibility_by_mileage": R("volatile", "vehicle/state"),
    "fillFuelTank": W("vehicle/state"),
    "find_nearest_tire_shop": R("static"),
    "gallon_to_liter": R("static"),
    "get_current_speed": R("strict"),
    "get_outside_temperature_from_google": R("strict"),
    "get_outside_temperature_from_weather_com": R("strict"),
    "get_zipcode_based_on_city": R("static"),
    "liter_to_gallon": R("static"),
    "lockDoors": W("vehicle/state"),
    "pressBrakePedal": W("vehicle/state"),
    "releaseBrakePedal": W("vehicle/state"),
    "setCruiseControl": W("vehicle/state"),
    "setHeadlights": W("vehicle/state"),
    "set_navigation": W("vehicle/state"),
    "startEngine": W("vehicle/state"),
    # ---- tickets
    "close_ticket": W("ticket/list", "ticket/{ticket_id}"),
    "create_ticket": W("ticket/list"),
    "edit_ticket": W("ticket/list", "ticket/{ticket_id}"),
    "get_ticket": R("slow", "ticket/{ticket_id}"),
    "get_user_tickets": R("slow", "ticket/list"),
    "logout": W("ticket/list"),
    "resolve_ticket": W("ticket/list", "ticket/{ticket_id}"),
    "ticket_get_login_status": R("strict"),
    "ticket_login": W("ticket/list"),
    # ---- trading
    "add_to_watchlist": W("trading/watchlist"),
    "cancel_order": W("trading/orders", "trading/order/{order_id}", "trading/account"),
    "filter_stocks_by_price": R("volatile", "trading/prices"),
    "fund_account": W("trading/account", "trading/transactions", key=True),
    "get_account_info": R("slow", "trading/account"),
    "get_available_stocks": R("static"),
    "get_current_time": R("strict"),
    "get_order_details": R("slow", "trading/order/{order_id}", "trading/orders"),
    "get_order_history": R("slow", "trading/orders"),
    "get_stock_info": R("volatile", "trading/prices"),
    "get_symbol_by_name": R("static"),
    "get_transaction_history": R("slow", "trading/transactions"),
    "get_watchlist": R("slow", "trading/watchlist"),
    "notify_price_change": R("volatile", "trading/prices"),
    "place_order": W("trading/orders", "trading/account", "trading/transactions", key=True),
    "remove_stock_from_watchlist": W("trading/watchlist"),
    "trading_get_login_status": R("strict"),
    "trading_login": W(*TRADING_ALL),
    "trading_logout": W(*TRADING_ALL),
    "withdraw_funds": W("trading/account", "trading/transactions", key=True),
    # ---- messages
    "add_contact": W("message/users"),
    "delete_message": W("message/messages"),
    "get_message_stats": R("slow", "message/messages", "message/users"),
    "get_user_id": R("slow", "message/users"),
    "list_users": R("slow", "message/users"),
    "message_get_login_status": R("strict"),
    "message_login": W("message/messages"),
    "search_messages": R("slow", "message/messages"),
    "send_message": W("message/messages"),
    "view_messages_sent": R("slow", "message/messages"),
    # ---- file system: every result depends on the working directory
    **{name: R("slow", "fs/tree") for name in
       ("cat", "diff", "du", "find", "grep", "ls", "pwd", "sort", "tail", "wc")},
    **{name: W("fs/tree") for name in
       ("cd", "cp", "echo", "mkdir", "mv", "rm", "rmdir", "touch")},
}


def _docs(class_name):
    import bfcl_eval
    from bfcl_eval.constants.executable_backend_config import MULTI_TURN_FUNC_DOC_FILE_MAPPING
    path = (Path(bfcl_eval.__file__).parent / "data/multi_turn_func_doc"
            / MULTI_TURN_FUNC_DOC_FILE_MAPPING[class_name])
    return [json.loads(line) for line in path.read_text().splitlines()]


def _spec(class_name, doc):
    app, name = APPS[class_name], doc["name"]
    declared = R("static") if class_name == "MathAPI" else CONTRACT[name]
    schema = doc["parameters"]
    spec = _tool(app, declared["kind"], doc["description"], schema.get("properties", {}),
                 fresh=declared.get("fresh"), reads=declared.get("reads", ()),
                 affects=declared.get("affects", ()), auth=app not in PUBLIC_APPS,
                 key=declared.get("key", False))
    spec["required"] = list(schema.get("required", []))
    return spec


def plain(value):
    """A backend object as JSON, the way BFCL compares states: public
    attributes only (no generator, no timestamps, no parent links)."""
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        rows = [plain(item) for item in value]
        return sorted(rows, key=json.dumps) if isinstance(value, set) else rows
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "__dict__"):
        return {key: plain(item) for key, item in vars(value).items()
                if not key.startswith("_") and key != "parent"}
    return str(value)


class BfclDomain(Domain):
    event_kinds = frozenset({"set", "add", "append", "call"})

    def __init__(self, task):
        from bfcl_eval.constants.executable_backend_config import (
            CLASS_FILE_PATH_MAPPING, STATELESS_CLASSES)
        self.instances, self.methods, self.tools = {}, {}, {}
        excluded = set(task.get("excluded_function", []))
        for class_name in task["involved_classes"]:
            instance = getattr(importlib.import_module(CLASS_FILE_PATH_MAPPING[class_name]),
                               class_name)()
            if class_name not in STATELESS_CLASSES:
                instance._load_scenario(copy.deepcopy(task["initial"].get(class_name, {})))
            self.instances[class_name] = instance
            methods = dict(inspect.getmembers(instance, inspect.ismethod))
            for doc in _docs(class_name):
                if doc["name"] in excluded or doc["name"] not in methods:
                    continue
                self.tools[doc["name"]] = _spec(class_name, doc)
                self.methods[doc["name"]] = (class_name, doc["name"])
        self.login_apps = frozenset(APPS[c] for c in task["involved_classes"]) - PUBLIC_APPS

    def call(self, name, arguments, shadow=False):
        class_name, method = self.methods[name]
        instance = self.instances[class_name]
        if shadow:  # a copy: the real backend, its generator included, is untouched
            instance = copy.deepcopy(instance)
        try:
            value = getattr(instance, method)(**copy.deepcopy(arguments))
        except Exception as error:  # noqa: BLE001 - BFCL backends raise on bad input
            raise BusinessError(f"{type(error).__name__}: {error}") from error
        # BFCL reports a refusal as {"error": ...} instead of raising.
        if isinstance(value, dict) and "error" in value:
            raise BusinessError(str(value["error"]))
        return plain(value)

    def apply(self, event):
        instance = self.instances[event["target"]]
        if event["kind"] == "call":  # another party uses the same service
            getattr(instance, event["name"])(**copy.deepcopy(event["arguments"]))
            return
        *path, last = event["path"]
        node = getattr(instance, path[0]) if path else instance
        for part in path[1:]:
            node = node[part]
        holder_is_object = not path
        current = getattr(node, last) if holder_is_object else node[last]
        if event["kind"] == "set":
            new = copy.deepcopy(event["value"])
        elif event["kind"] == "add":
            new = current + event["value"]
        else:
            new = [*current, copy.deepcopy(event["value"])]
        if holder_is_object:
            setattr(node, last, new)
        else:
            node[last] = new

    def state(self):
        return {name: plain(instance) for name, instance in sorted(self.instances.items())}
