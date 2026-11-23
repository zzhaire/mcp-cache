"""Other task environments behind the same server layer.

The built-in lab world is a bank, a shop and a rate table. A *domain* swaps
that business logic for the original backend of a public benchmark while the
server layer stays the same: sign-in tokens that expire and can be revoked,
operation keys, injected faults, external changes, a versioned tool catalog,
the virtual clock, and the per-tool contract a host may rely on.

A task selects a domain with ``"domain": "<name>"``; without it the built-in
world runs. ``make_world(task)`` is the only constructor callers need.
"""
import copy
import importlib

from exp_core.lab.world import TOOLS, BusinessError, LabWorld, _tool  # noqa: F401

DOMAINS = {"bfcl": "exp_core.lab.domains.bfcl:BfclDomain",
           "tau2": "exp_core.lab.domains.tau2:Tau2Domain",
           "appworld": "exp_core.lab.domains.appworld:AppWorldDomain"}
SERVER_EVENTS = frozenset({"expire_session", "catalog_update"})


class Domain:
    """The business logic of one environment.

    ``tools``/``later_tools`` map a tool name to a ``_tool`` spec whose
    ``parameters`` are JSON-schema properties and whose ``required`` lists the
    mandatory ones. ``login_apps`` are the apps that need a sign-in.
    """
    tools, later_tools = {}, {}
    login_apps = frozenset()
    event_kinds = frozenset()

    def call(self, name, arguments, shadow=False):
        """Run one tool and return its value; raise ``BusinessError`` when the
        service refuses. ``shadow`` is a measurement read: it must leave no
        trace in the environment."""
        raise NotImplementedError

    def apply(self, event):
        """Apply one external change."""
        raise NotImplementedError

    def state(self):
        """Everything a run can change, as plain JSON."""
        raise NotImplementedError

    def close(self):
        pass


def load_domain(name):
    module, cls = DOMAINS[name].split(":")
    return getattr(importlib.import_module(module), cls)


class DomainWorld(LabWorld):
    """``LabWorld`` with its business logic replaced by a ``Domain``."""

    def __init__(self, task):
        self.domain = load_domain(task["domain"])(copy.deepcopy(task))
        self.login_apps = frozenset(self.domain.login_apps)
        self.event_kinds = SERVER_EVENTS | frozenset(self.domain.event_kinds)
        super().__init__({**task, "initial": {"users": task.get("users", {})}})

    def _tool_tables(self):
        apps = ", ".join(sorted(self.login_apps))
        server = {name: TOOLS[name] for name in ("auth_login", "auth_logout", "op_status")}
        server["auth_login"] = {**server["auth_login"], "description":
                                f"Sign in to an app ({apps}). Needed before using it."}
        return {**server, **self.domain.tools}, dict(self.domain.later_tools)

    def _schema(self, spec):
        return {"type": "object", "properties": spec["parameters"],
                "required": list(spec.get("required", spec["parameters"]))}

    def _invalid(self, name, spec, arguments):
        if not isinstance(arguments, dict):
            return f"{name} takes an object of arguments"
        unknown = sorted(set(arguments) - set(spec["parameters"]))
        missing = sorted(set(spec.get("required", spec["parameters"])) - set(arguments))
        if unknown or missing:
            return (f"{name}: " + (f"unknown arguments {unknown} " if unknown else "")
                    + (f"missing arguments {missing}" if missing else "")).strip()
        return None

    def _handler(self, name):
        if name in self.domain.tools or name in self.domain.later_tools:
            return lambda **arguments: self.domain.call(name, arguments, shadow=self._shadow)
        return super()._handler(name)

    def _check_credentials(self, app, username, password):
        if hasattr(self.domain, "login"):  # the environment has its own accounts
            self.domain.login(app, username, password)
        else:
            super()._check_credentials(app, username, password)

    def _apply_domain(self, event):
        self.domain.apply(event)

    def snapshot(self):
        return self.domain.state()

    def final_state(self):
        return self.domain.state()

    def close(self):
        self.domain.close()


def make_world(task):
    """The world a task runs in: its domain's, or the built-in lab world."""
    return DomainWorld(task) if task.get("domain") else LabWorld(task)
