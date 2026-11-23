"""Experiment factors and measurements that apply to every method alike.

- ``degrade_contract``: how much the server tells hosts about its tools.
- ``DeliveryAudit``: what was actually handed to the agent, compared with what
  the server would have said at that moment - the same check for a method with
  no cache, a legacy cache or the scheduler.
"""
import copy
import json

LISTINGS = ("per_turn", "per_connection")
CONTRACT_LEVELS = ("full", "annotations", "none", "wrong")


def degrade_contract(contract, level):
    """The contract a host is given at one of four levels.

    full         everything the server publishes: freshness class, the
                 resources a read depends on and a write affects, key support.
    annotations  only what MCP tool annotations carry: read or write. No
                 freshness class and no resources, so a host can cache reads
                 but must invalidate by app.
    none         nothing: every tool may change state. Nothing is cacheable.
    wrong        the full contract with two mistakes a server could really
                 make: the transfer forgets that it changes the target's
                 balance, and the price is declared slow although others
                 change it.
    """
    if level not in CONTRACT_LEVELS:
        raise ValueError(f"unknown contract level {level}")
    contract = copy.deepcopy(contract)
    if level == "full":
        return contract
    for name, spec in contract.items():
        if level == "annotations":
            if spec["kind"] == "read":
                spec["fresh_class"] = None
            spec["reads"], spec["affects"] = [], []
            spec["idempotency_key"] = False
        elif level == "none":
            if spec["kind"] in {"read", "status"}:
                spec.update(kind="write", fresh_class=None)
            spec["reads"], spec["affects"] = [], []
            spec["idempotency_key"] = False
    if level == "wrong":
        transfer = contract.get("bank_transfer")
        if transfer:
            transfer["affects"] = [r for r in transfer["affects"] if "{target}" not in r]
        if "shop_get_price" in contract:
            contract["shop_get_price"]["fresh_class"] = "slow"
    return contract


def _same(left, right):
    return json.dumps(left, sort_keys=True, default=str) == \
        json.dumps(right, sort_keys=True, default=str)


class DeliveryAudit:
    """Counts, at the point where a result is handed to the agent, whether it
    is what the server would say right now.

    ``truth(name, arguments)`` is an unmetered read of the current state that
    returns the adapter payload.
    """

    def __init__(self, truth, contract):
        self.truth, self.contract = truth, contract
        self.stats = {"delivered_reads": 0, "from_cache": 0, "stale": 0,
                      "after_revocation": 0, "check_failed": 0,
                      "references": 0, "stale_references": 0}
        self.stale = []

    def _is_read(self, name):
        return (self.contract.get(name) or {}).get("kind") == "read"

    def _compare(self, name, arguments, value, where):
        try:
            payload = self.truth(name, arguments)
        except Exception:  # noqa: BLE001 - a failed check is counted, never hidden
            self.stats["check_failed"] += 1
            return None
        if not payload.get("ok"):
            if payload.get("error_code") == "unauthorized":
                # The server no longer accepts the sign-in; the host served anyway.
                self.stats["after_revocation"] += 1
                self.stale.append({"where": where, "tool": name, "arguments": arguments,
                                   "kind": "after_revocation"})
                return False
            self.stats["check_failed"] += 1
            return None
        if _same(payload["value"], value):
            return True
        self.stale.append({"where": where, "tool": name, "arguments": arguments,
                           "kind": "stale"})
        return False

    def delivered(self, name, arguments, result, reached_server):
        """A tool result handed to the agent as the answer to its call."""
        if not self._is_read(name) or (isinstance(result, dict) and "error" in result):
            return
        self.stats["delivered_reads"] += 1
        self.stats["from_cache"] += not reached_server
        # Compared even when a request was sent: a server-side cache can hand
        # back an old reply. An event triggered by this very call takes effect
        # only before the next request, so a fresh reply never looks stale.
        if self._compare(name, arguments, result, "result") is False and \
                self.stale[-1]["kind"] == "stale":
            self.stats["stale"] += 1

    def referenced(self, name, arguments, value):
        """A cached value the host told the agent is still valid, without a call."""
        if not self._is_read(name):
            return
        self.stats["references"] += 1
        if self._compare(name, arguments, value, "reference") is False and \
                self.stale[-1]["kind"] == "stale":
            self.stats["stale_references"] += 1
