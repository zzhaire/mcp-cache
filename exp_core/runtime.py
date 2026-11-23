"""Conservative fixed policy; benchmark adapters must declare read contracts."""
import copy
import hashlib
import json
import time


class AccessFailure(RuntimeError):
    """Adapter-declared failure. Unknown write outcomes are never retryable."""
    def __init__(self, code, *, retryable=False, outcome="unknown", retry_after=0,
                 detail=None):
        super().__init__(code)
        self.code, self.retryable = code, retryable
        self.outcome, self.retry_after = outcome, retry_after
        # The server's own explanation ("amount must be a positive number").
        # The agent needs it to correct a malformed call; the code alone is a category.
        self.detail = detail


def failure_result(error, operation_id=None):
    """What the agent is shown for a failed call: the category, the server's
    explanation when it gave one, and the operation id."""
    result = {"error": str(error)}
    if getattr(error, "detail", None):
        result["detail"] = str(error.detail)[:500]
    if operation_id is not None:
        result["operation_id"] = operation_id
    return result


class Runtime:
    def __init__(self, clock=time.monotonic, ttl=60, disabled=(),
                 m4_failure_only=False):
        self.clock, self.ttl, self.disabled = clock, ttl, set(disabled)
        # Failure-audit arms reuse failed operations but never successful results.
        self.m4_failure_only = m4_failure_only
        self.data_epoch = self.auth_epoch = 0
        self.results, self.operations = {}, {}
        self.catalogs, self.credentials = {}, {}
        self.stats = {"logical_calls": 0, "M2_eligible": 0, "M2_hits": 0,
                      "M4_hits": 0, "backend_calls": 0, "M1_lookups": 0,
                      "M1_hits": 0, "M3_lookups": 0, "M3_hits": 0,
                      "M4_failure_hits": 0, "retries": 0}

    def catalog(self, server, version, fetch):
        self.stats["M1_lookups"] += 1
        key = (server, version, self.auth_epoch)
        if "M1" not in self.disabled and key in self.catalogs:
            self.stats["M1_hits"] += 1
            return copy.deepcopy(self.catalogs[key])
        value = fetch()
        if "M1" not in self.disabled:
            self.catalogs[key] = copy.deepcopy(value)
        return value

    def save_credential(self, service, subject, scopes, token, valid_until):
        # valid_until is supplied by the execution adapter's explicit contract.
        # No opaque token receives an invented TTL.
        key = (service, subject, tuple(sorted(scopes)), self.auth_epoch)
        self.credentials[key] = (token, valid_until)

    def credential(self, service, subject, scopes):
        self.stats["M3_lookups"] += 1
        key = (service, subject, tuple(sorted(scopes)), self.auth_epoch)
        value = self.credentials.get(key)
        if "M3" not in self.disabled and value and self.clock() < value[1]:
            self.stats["M3_hits"] += 1
            return value[0]
        return None

    def invalidate_authorization(self):
        self.auth_epoch += 1
        self.credentials.clear()
        self.results.clear()
        self.catalogs.clear()

    def operation_audit(self):
        """JSON-safe operation state, without arguments, credentials or results."""
        return [{k: copy.deepcopy(v) for k, v in record.items()
                 if k not in {"result", "failure", "request"}}
                for record in self.operations.values()]

    def execute(self, name, arguments, fetch, *, readonly=False, operation_id=None,
                scope="", max_attempts=2, invalidate=True):
        # Retry cap is identical for bare and fixed. M4 adds state, not a larger
        # retry budget. The Host keeps one operation_id across internal retries.
        for attempt in range(max_attempts):
            try:
                return self.call(name, arguments, fetch, readonly, operation_id, scope,
                                 invalidate)
            except AccessFailure as error:
                if (not error.retryable or error.outcome != "not_executed"
                        or attempt + 1 >= max_attempts):
                    raise
                if error.retry_after:
                    time.sleep(error.retry_after)
                self.stats["retries"] += 1

    def call(self, name, arguments, fetch, readonly=False, operation_id=None, scope="",
             invalidate=True):
        self.stats["logical_calls"] += 1
        identity = json.dumps([scope, name, arguments], sort_keys=True)
        fingerprint = hashlib.sha256(identity.encode()).hexdigest()
        key = (self.auth_epoch, self.data_epoch, identity)
        now = self.clock()
        old = self.operations.get(operation_id) if operation_id else None
        if old:
            if old["identity_sha256"] != fingerprint:
                raise ValueError("operation_id was reused for a different request")
            # These checks protect side effects even in the ablated baseline.
            if old["state"] == "uncertain":
                raise AccessFailure("write_outcome_unknown", outcome="unknown")
            if not readonly and old["auth_epoch"] != self.auth_epoch:
                raise AccessFailure("operation_authorization_changed")
            valid = (old["auth_epoch"] == self.auth_epoch
                     and (not readonly or (old["data_epoch"] == self.data_epoch
                                           and now < old["valid_until"])))
            if ("M4" not in self.disabled and not self.m4_failure_only
                    and valid and old["state"] == "completed"):
                self.stats["M4_hits"] += 1
                return copy.deepcopy(old["result"])
            same_conditions = (old["auth_epoch"] == self.auth_epoch
                               and old["data_epoch"] == self.data_epoch)
            if "M4" not in self.disabled and same_conditions and old.get("failure"):
                error = old["failure"]
                if not error.retryable or now < old["retry_at"]:
                    self.stats["M4_failure_hits"] += 1
                    raise AccessFailure(error.code, retryable=error.retryable,
                                        outcome=error.outcome,
                                        retry_after=max(0, old["retry_at"] - now),
                                        detail=error.detail)
        if readonly:
            self.stats["M2_eligible"] += 1
            cached = self.results.get(key)
            if "M2" not in self.disabled and cached and now < cached[0]:
                self.stats["M2_hits"] += 1
                return copy.deepcopy(cached[1])
        self.stats["backend_calls"] += 1
        record = {
            "operation_id": operation_id, "identity_sha256": fingerprint,
            "auth_epoch": self.auth_epoch, "data_epoch": self.data_epoch,
            "readonly": readonly, "state": "running",
            "attempts": (old or {}).get("attempts", 0) + 1,
            "created_at": (old or {}).get("created_at", now),
            "updated_at": now, "valid_until": now + self.ttl,
            "retry_at": now, "error_code": None, "outcome": None,
            "history": copy.deepcopy((old or {}).get("history", [])),
            # Kept out of operation_audit(); decision-time publication needs
            # name+arguments to describe the failed request without its result.
            "request": {"name": name, "arguments": copy.deepcopy(arguments)}}
        try:
            value = fetch()
        except Exception as error:
            outcome = getattr(error, "outcome", "unknown")
            if not readonly and invalidate and outcome == "unknown":
                self.data_epoch += 1
                self.results.clear()
            record.update(state="uncertain" if not readonly and outcome == "unknown" else "failed",
                          outcome=outcome, updated_at=self.clock(),
                          error_code=error.code if isinstance(error, AccessFailure) else "unknown_error",
                          retry_at=self.clock() + getattr(error, "retry_after", 0))
            record["history"].append({"attempt": record["attempts"],
                                      "state": record["state"],
                                      "error_code": record["error_code"]})
            if isinstance(error, AccessFailure):
                record["failure"] = error
            if operation_id:
                self.operations[operation_id] = record
            raise
        if not readonly and invalidate:
            self.data_epoch += 1
            self.results.clear()
        elif "M2" not in self.disabled:
            self.results[key] = (self.clock() + self.ttl, copy.deepcopy(value))
        if operation_id:
            record.update(state="completed", result=copy.deepcopy(value),
                          updated_at=self.clock(), valid_until=self.clock() + self.ttl,
                          outcome="completed", data_epoch=self.data_epoch)
            record["history"].append({"attempt": record["attempts"], "state": "completed",
                                      "error_code": None})
            self.operations[operation_id] = record
        return value
