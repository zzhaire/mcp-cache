"""Connect M1–M4 to actual adapter responses using explicit contracts."""
import hashlib
import json
from pathlib import Path
from exp_core.runtime import AccessFailure
from exp_core.settings import CONTRACTS

CONTRACT = json.loads((CONTRACTS / "access.json").read_text())
SCHEDULER_CONTRACT = json.loads((CONTRACTS / "scheduler.json").read_text())


def _expand(templates, params):
    """Format resource templates; a template with a missing field is dropped."""
    out = []
    for template in templates:
        try:
            out.append(template.format(**params))
        except (KeyError, IndexError, ValueError):
            continue
    return out


def scheduler_meta(dataset, name, arguments):
    """Contract facts the scheduler needs; conservative defaults when undeclared."""
    app = arguments.get("app") if dataset == "appworld" else None
    tool = f"{app}.{arguments.get('api')}" if dataset == "appworld" else name
    params = arguments.get("arguments", {}) if dataset == "appworld" else arguments
    if not isinstance(params, dict):
        params = {}
    app_source = SCHEDULER_CONTRACT["app_source"].get(dataset, "dataset")
    principal_source = SCHEDULER_CONTRACT["principal_source"].get(dataset)
    resources = SCHEDULER_CONTRACT["resources"].get(dataset, {}).get(tool, {})
    return {
        "server": dataset,
        "app": str(app) if app_source == "app" and app else dataset,
        "tool": tool,
        "principal": str(app) if principal_source == "app" and app else None,
        "fresh_class": SCHEDULER_CONTRACT["fresh_classes"].get(dataset, {}).get(
            tool, SCHEDULER_CONTRACT["default_read_class"]),
        "reads": _expand(resources.get("reads", []), params),
        "affects": _expand(resources.get("affects", []), params),
        "idempotency_key": tool in SCHEDULER_CONTRACT.get("idempotency_key", {}).get(
            dataset, []),
    }


class Access:
    def __init__(self, runtime, dataset, scope, invoke, read_contract=None,
                 invoke_keyed=None, shadow_invoke=None, lab_contract=None):
        self.runtime, self.dataset, self.scope, self.invoke = runtime, dataset, scope, invoke
        # invoke_keyed(name, arguments, key): same call carrying an idempotency key.
        # shadow_invoke(name, arguments): unmetered read used only to measure staleness.
        self.invoke_keyed, self.shadow_invoke = invoke_keyed, shadow_invoke
        # Lab server: the per-tool contract comes from the server itself.
        self.lab_contract = lab_contract or {}
        self._subjects = {}  # app -> the user last signed in through this host
        # Lab server: the sign-in token per app, presented with every request.
        # Carrying it is part of the transport, the same for every method.
        self.tokens = {}
        if (dataset == "lab" and getattr(runtime, "supports_meta", False)
                and runtime.reconciler is None):
            runtime.reconciler = self._reconcile
        self.read_contract = read_contract or {}
        self.credential_lookups = 0

    def readonly(self, name, arguments):
        if self.dataset == "lab":
            # A strict read is "undeclared" for caches that have no strict class.
            spec = self.lab_contract.get(name, {})
            return spec.get("kind") == "read" and spec.get("fresh_class") != "strict"
        if self.dataset == "tau2":
            return self.read_contract.get(name, False)
        if self.dataset == "bfcl":
            return name in CONTRACT["bfcl_readonly"]
        return arguments.get("api") in CONTRACT["appworld_readonly"].get(arguments.get("app"), [])

    # ------------------------------------------------------------ lab server
    def _payload_value(self, payload):
        if not payload["ok"]:
            raise AccessFailure(payload.get("error_code", "adapter_error"),
                                retryable=payload.get("retryable", False),
                                outcome=payload.get("outcome", "unknown"),
                                retry_after=payload.get("retry_after", 0),
                                detail=payload.get("error"))
        return payload["value"]

    def _reconcile(self, request):
        """Ask the server whether a write sent under this operation key was applied."""
        try:
            status = self._payload_value(self.invoke("op_status",
                                                     {"key": request["operation_id"]}))
        except Exception:
            return None
        if status.get("state") == "executed":
            return "executed", status.get("result")
        if status.get("state") == "not_executed":
            return "not_executed", None
        return None

    def _lab_meta(self, name, arguments, spec):
        return {"server": "lab", "app": spec["app"], "tool": name,
                "principal": spec.get("principal"),
                "fresh_class": spec.get("fresh_class") or "slow",
                "reads": _expand(spec.get("reads", []), arguments),
                "affects": _expand(spec.get("affects", []), arguments),
                "idempotency_key": bool(spec.get("idempotency_key") and self.invoke_keyed)}

    def _hold(self, app, login_value):
        if isinstance(login_value, dict) and login_value.get("session"):
            self.tokens[app] = login_value["session"]

    def _call_lab(self, name, arguments, operation_id):
        runtime = self.runtime
        scheduled = getattr(runtime, "supports_meta", False)
        spec = self.lab_contract.get(name) or {"app": "lab", "kind": "write"}
        kind = spec["kind"]
        scopes = CONTRACT["credential_contract"]["scopes"]
        if kind in {"login", "logout"}:
            app = str(arguments.get("app"))
            principal_kw = {"principal": app} if scheduled else {}
            if kind == "logout":
                try:
                    return runtime.execute(name, arguments,
                                           lambda: self._payload_value(self.invoke(name, arguments)),
                                           operation_id=operation_id, scope=self.scope,
                                           max_attempts=CONTRACT["retry_budget"], invalidate=False)
                finally:
                    # The user signed out on purpose: drop everything cached for
                    # this app and never sign back in behind their back.
                    self.tokens.pop(app, None)
                    self._subjects.pop(app, None)
                    if scheduled:
                        runtime.refreshers.pop(app, None)
                    runtime.invalidate_authorization(**principal_kw)
            subject = f"{app}+{arguments.get('username')}"
            previous = self._subjects.get(app)
            if previous is not None and previous != subject:
                # A different user is signing in to the same app: everything
                # cached for the previous user must go, or it would leak across.
                runtime.invalidate_authorization(**principal_kw)
            self._subjects[app] = subject
            service = self.scope + ":" + app + ":" + hashlib.sha256(
                str(arguments.get("password", "")).encode()).hexdigest()
            self.credential_lookups += 1
            cached = runtime.credential(service, subject, scopes, **principal_kw)
            if cached is not None:
                self._hold(app, cached)
                return cached

            def login():
                payload = self.invoke(name, arguments)
                value = self._payload_value(payload)
                self._hold(app, value)
                contract = payload.get("credential_contract", {})
                remaining = contract.get("valid_for_seconds", 0)
                if contract.get("subject") == subject and remaining > 0:
                    runtime.save_credential(service, subject, scopes, value,
                                            runtime.clock() + remaining, **principal_kw)
                return value, runtime.clock() + remaining
            if scheduled:
                # The host can sign in again on its own when the server drops the session.
                runtime.refreshers[app] = login
            return runtime.execute(name, arguments, lambda: login()[0],
                                   operation_id=operation_id, scope=self.scope,
                                   max_attempts=CONTRACT["retry_budget"], invalidate=False)
        readonly = self.readonly(name, arguments)
        neutral = kind in {"status"} or (kind == "read" and not readonly)
        meta = self._lab_meta(name, arguments, spec) if scheduled else None
        keyed = bool(scheduled and kind == "write" and self.invoke_keyed)

        def fetch():
            # Every scheduled write carries its operation id, so the server can
            # answer op_status; only tools that declare a key deduplicate on it.
            return self._payload_value(self.invoke_keyed(name, arguments, operation_id)
                                       if keyed else self.invoke(name, arguments))
        if scheduled and kind == "read" and self.shadow_invoke is not None:
            meta["shadow_fetch"] = lambda: self._payload_value(
                self.shadow_invoke(name, arguments))
        if scheduled and kind == "read" and not readonly:
            readonly_arg = True  # the scheduler knows the strict class itself
        else:
            readonly_arg = readonly
        try:
            return runtime.execute(name, arguments, fetch, readonly=readonly_arg,
                                   operation_id=operation_id, scope=self.scope,
                                   max_attempts=CONTRACT["retry_budget"],
                                   invalidate=not neutral,
                                   **({"meta": meta} if scheduled else {}))
        except AccessFailure as error:
            if error.code == "unauthorized" and spec.get("principal"):
                # The server no longer accepts the session: cached authorization
                # state for it is wrong, whatever its stated lifetime was.
                runtime.invalidate_authorization(
                    **({"principal": spec["principal"]} if scheduled else {}))
            raise

    def call(self, name, arguments, operation_id):
        if self.dataset == "lab":
            return self._call_lab(name, arguments, operation_id)
        runtime = self.runtime
        # The scheduler takes contract facts; the legacy Runtime keeps its old calls.
        scheduled = getattr(runtime, "supports_meta", False)
        meta = scheduler_meta(self.dataset, name, arguments) if scheduled else None
        principal_kw = {"principal": meta["principal"]} if scheduled else {}
        app, api = arguments.get("app"), arguments.get("api")
        params = arguments.get("arguments", {})
        catalog = (self.dataset == "appworld" and app == "api_docs"
                   and api in CONTRACT["appworld_catalog_apis"])
        login = self.dataset == "appworld" and api == "login"
        service = self.scope + ":" + str(app)
        subject = str(app) + "+" + str(params.get("username"))
        # Changing the supplied password cannot hit a previous login entry.
        auth_service = service + ":" + hashlib.sha256(
            str(params.get("password", "")).encode()).hexdigest()
        scopes = CONTRACT["credential_contract"]["scopes"]
        if login:
            self.credential_lookups += 1
            cached = runtime.credential(auth_service, subject, scopes, **principal_kw)
            if cached is not None:
                return cached

        keyed = bool(scheduled and meta["idempotency_key"] and self.invoke_keyed)
        if scheduled and not keyed:
            # Without a transport for the key the write is not idempotent for us.
            meta["idempotency_key"] = False

        def fetch():
            # The operation id doubles as the idempotency key across internal retries.
            payload = (self.invoke_keyed(name, arguments, operation_id) if keyed
                       else self.invoke(name, arguments))
            if not payload["ok"]:
                raise AccessFailure(payload.get("error_code", "adapter_error"),
                                    retryable=payload.get("retryable", False),
                                    outcome=payload.get("outcome", "unknown"),
                                    retry_after=payload.get("retry_after", 0),
                                    detail=payload.get("error"))
            value = payload["value"]
            # BFCL returns business failures as dictionaries rather than exceptions.
            if isinstance(value, dict) and "error" in value:
                raise AccessFailure("business_error", outcome="not_executed",
                                    detail=value.get("error"))
            if login:
                contract = payload.get("credential_contract", {})
                remaining = contract.get("valid_for_seconds", 0)
                if (contract.get("subject") == subject and remaining > 0
                        and contract.get("scopes") == scopes):
                    runtime.save_credential(auth_service, subject, scopes, value,
                                            runtime.clock() + remaining, **principal_kw)
            return value

        if catalog:
            identity = json.dumps([self.scope, arguments], sort_keys=True)
            if scheduled:
                return runtime.catalog(
                    identity, CONTRACT["version"], fetch,
                    public=SCHEDULER_CONTRACT["public_catalogs"].get(self.dataset, False))
            return runtime.catalog(identity, CONTRACT["version"], fetch)
        readonly = self.readonly(name, arguments)
        if scheduled and readonly and self.shadow_invoke is not None:
            def shadow():
                payload = self.shadow_invoke(name, arguments)
                if not payload["ok"]:
                    raise AccessFailure(payload.get("error_code", "adapter_error"))
                return payload["value"]
            meta["shadow_fetch"] = shadow
        neutral = (self.dataset == "appworld"
                   and api in CONTRACT.get("appworld_neutral_apis", []))
        try:
            return runtime.execute(name, arguments, fetch, readonly=readonly,
                                   operation_id=operation_id, scope=self.scope,
                                   max_attempts=CONTRACT["retry_budget"],
                                   invalidate=not neutral,
                                   **({"meta": meta} if scheduled else {}))
        finally:
            if (self.dataset == "appworld"
                    and api in CONTRACT["appworld_auth_invalidation_apis"]):
                runtime.invalidate_authorization(**principal_kw)
