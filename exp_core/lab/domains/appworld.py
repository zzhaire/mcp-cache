"""AppWorld (official environment) as a lab domain.

Every API of the apps a task needs becomes one tool, ``<app>__<api>``. The
original in-band sign-in (a token passed as an argument to every call) moves
into the server layer: ``auth_login`` checks the password with the app's own
login API, and the domain attaches the app's token to each call.

The contract is derived from the API documentation - GET is a read, anything
else a write - at the granularity of one resource per app; a few entries are
refined by hand below.
"""
import json
import re

from exp_core.lab.domains import BusinessError, Domain, _tool
from exp_core.lab.domains._worker import shared
from exp_core.settings import APPWORLD_PYTHON, APPWORLD_ROOT

SEP = "__"
SKIPPED = {"login", "logout"}                      # replaced by the server layer
VOLATILE = {"spotify__show_volume", "spotify__show_current_song",
            "venmo__show_notifications_count"}    # changed from other devices / by others
KEYED = {"venmo__create_transaction", "venmo__create_payment_request"}
TYPES = {"string": "string", "integer": "integer", "number": "number", "boolean": "boolean"}


def api_docs(app):
    path = APPWORLD_ROOT / "data" / "api_docs" / "standard" / f"{app}.json"
    return json.loads(path.read_text())


def spec_of(app, doc):
    name = app + SEP + doc["api_name"]
    read = doc["method"] == "GET"
    takes_token = any(p["name"] == "access_token" for p in doc["parameters"])
    static = app == "supervisor"
    resource = f"{app}/state"
    parameters = {p["name"]: {"type": TYPES.get(p["type"], "string"),
                              "description": p["description"]}
                  for p in doc["parameters"] if p["name"] != "access_token"}
    spec = _tool(app, "read" if read else "write", doc["description"], parameters,
                 fresh=("static" if static else "volatile" if name in VOLATILE else "slow")
                 if read else None,
                 reads=[resource] if read else (), affects=() if read else [resource],
                 # An API that takes no token is public (the song catalog, say).
                 auth=takes_token, key=name in KEYED)
    spec["required"] = [p["name"] for p in doc["parameters"]
                        if p["required"] and p["name"] != "access_token"]
    spec["takes_token"] = takes_token
    return name, spec


class AppWorldDomain(Domain):
    event_kinds = frozenset({"set", "add"})

    def __init__(self, task):
        self.apps = list(task["apps"])
        self.worker = shared("appworld", APPWORLD_PYTHON,
                             "exp_core.adapters.appworld_domain_worker",
                             {"APPWORLD_ROOT": str(APPWORLD_ROOT)})
        self.info = self.worker.request("init", id=task["source"]["task_id"],
                                        apps=["supervisor", *self.apps])
        self.login_apps = frozenset(self.apps)
        self.tokens, self.tools = {}, {}
        for app in ["supervisor", *self.apps]:
            for doc in api_docs(app).values():
                if doc["api_name"] not in SKIPPED:
                    name, spec = spec_of(app, doc)
                    self.tools[name] = spec

    def login(self, app, username, password):
        """Check a sign-in with the app's own login API and keep its token."""
        token = self.worker.request("call", app=app, api="login",
                                    arguments={"username": username, "password": password})
        self.tokens[app] = token["access_token"]

    def call(self, name, arguments, shadow=False):
        app, api = name.split(SEP, 1)
        arguments = dict(arguments)
        if self.tools[name]["takes_token"]:
            if app not in self.tokens:
                raise BusinessError(f"sign in to {app} first")
            arguments["access_token"] = self.tokens[app]
        return self.worker.request("call", app=app, api=api, arguments=arguments)

    def apply(self, event):
        self.worker.request("event", event=event)

    def state(self):
        return self.worker.request("state")


def route(method, url):
    """``(tool name, path arguments)`` of one recorded HTTP call."""
    app = url.split("/")[1]
    best = None
    for doc in api_docs(app).values():
        if doc["method"].lower() != method.lower():
            continue
        pattern = re.sub(r"\{(\w+)\}", r"(?P<\1>[^/]+)", doc["path"])
        match = re.fullmatch(pattern, url)
        if match and (best is None or len(match.groupdict()) < len(best[1])):
            types = {p["name"]: p["type"] for p in doc["parameters"]}
            best = (doc, {key: int(value) if types.get(key) == "integer" else value
                          for key, value in match.groupdict().items()})
    if best is None:
        raise ValueError(f"no API for {method} {url}")
    return best[0]["api_name"], best[1]
