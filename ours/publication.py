"""Decision-time publication and duplicate-result references for the Ours arm.

Pure functions only: no network, no model, no filesystem access. ``run.py``
calls them inside the decision loop, and ``test_publish.py`` exercises them
directly. Publication never bypasses the Runtime validity contract: scope,
auth epoch, data epoch and TTL are re-checked at decision time.
"""
import copy
import hashlib
import json
import re

from exp_core.redaction import redact


def _json_bytes(value):
    return len(json.dumps(value, default=str).encode())


def collect_results(runtime, scope, budget_bytes):
    """Collect read results that are valid for the next decision.

    Returns ``(items, used_bytes, identities)`` where each item is
    ``{"name", "arguments", "result"}``. Results are ordered by identity so the
    same state publishes identical bytes; the first item that would exceed the
    remaining budget stops collection instead of silently overspending.
    """
    if budget_bytes <= 0:
        return [], 0, []
    if _scheduled(runtime):
        valid = sorted(runtime.iter_valid_reads())
        items, identities, used = [], [], 0
        for identity, value in valid:
            result_scope, name, arguments = json.loads(identity)
            if result_scope != scope:
                continue
            item = {"name": name, "arguments": redact(copy.deepcopy(arguments)),
                    "result": redact(copy.deepcopy(value))}
            size = _json_bytes(item)
            if used + size > budget_bytes:
                break
            items.append(item)
            identities.append(identity)
            used += size
        return items, used, identities
    keys = sorted(runtime.results, key=lambda key: (key[2], key[0], key[1]))
    items, identities, used = [], [], 0
    for key in keys:
        auth_epoch, data_epoch, identity = key
        if auth_epoch != runtime.auth_epoch or data_epoch != runtime.data_epoch:
            continue
        expiry, value = runtime.results[key]
        if runtime.clock() >= expiry:
            continue
        result_scope, name, arguments = json.loads(identity)
        if result_scope != scope:
            continue
        item = {"name": name, "arguments": redact(copy.deepcopy(arguments)),
                "result": redact(copy.deepcopy(value))}
        size = _json_bytes(item)
        if used + size > budget_bytes:
            break
        items.append(item)
        identities.append(key)
        used += size
    return items, used, identities


def _scalar(value):
    return value is None or isinstance(value, (str, int, float, bool))


def key_values(value, max_bytes):
    """The key values of a read result, small enough to restate in the context.

    A record keeps its scalar fields; a list keeps its records the same way.
    What does not fit in ``max_bytes`` is left out and named, so the model can
    tell a complete summary from one that needs the tool again.
    """
    if _scalar(value):
        return value
    if isinstance(value, dict):
        kept = {k: v for k, v in value.items() if _scalar(v)}
        omitted = sorted(str(k) for k in value if k not in kept)
        while kept and _json_bytes(kept) > max_bytes:
            omitted.append(str(kept.popitem()[0]))
        return {**kept, "_omitted": sorted(omitted)} if omitted else kept
    if isinstance(value, (list, tuple)):
        rows = [key_values(row, max_bytes) for row in value]
        if _json_bytes(rows) <= max_bytes:
            return rows
        return {"count": len(rows), "_omitted": ["rows"]}
    return {"_omitted": ["value"]}


def version_of(value):
    """A short fingerprint of a read result."""
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def ahead_item(name, arguments, value, full_max_bytes=1024):
    """The published record of a read the host has just made on its own."""
    return {"call": name, "arguments": redact(copy.deepcopy(arguments)),
            "values": redact(key_values(copy.deepcopy(value), full_max_bytes))}


def _reference(name, arguments, value, mode="ref_only", full_max_bytes=0):
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"),
                           default=str)
    digest = hashlib.sha256(canonical.encode()).hexdigest()
    if mode == "key_values":
        # The values themselves, so the model can go on without calling again;
        # ``version`` identifies the result they were taken from.
        return {"name": name, "arguments": redact(copy.deepcopy(arguments)),
                "state": "valid", "version": digest[:16],
                "values": redact(key_values(copy.deepcopy(value), full_max_bytes)),
                "note": "current for the published state; fields under _omitted "
                        "are not included here"}
    item = {"name": name, "arguments": redact(copy.deepcopy(arguments)),
            "state": "valid", "result_ref": digest[:16],
            "note": "result already shown earlier in this conversation; "
                    "valid for the current state"}
    # Scheduler publish.mode: attach the (redacted) result itself.
    if mode == "full" or (mode == "hybrid" and len(canonical.encode()) <= full_max_bytes):
        item["result"] = redact(copy.deepcopy(value))
    return item


def _scheduled(runtime):
    return getattr(runtime, "supports_meta", False)


def _latest_reference(runtime, scope, budget_bytes, store, identity, mode="ref_only",
                      full_max_bytes=0):
    """One reference for exactly one read, or ``(None, 0, None)`` if invalid.

    The returned identity is the tool-call identity the model would produce if
    it repeated that read, so publication accounting and cache-hit attribution
    compare like with like for both data reads and M1 catalog reads.
    """
    key, item = None, None
    if _scheduled(runtime):
        if store == "result":
            found, value = runtime.valid_read(identity)
            result_scope, name, arguments = json.loads(identity)
            if found and result_scope == scope:
                key = identity
                item = _reference(name, arguments, value, mode, full_max_bytes)
        elif store == "catalog":
            found, value = runtime.valid_catalog(identity)
            parsed = json.loads(identity) if found else None
            if (isinstance(parsed, list) and len(parsed) == 2 and parsed[0] == scope
                    and isinstance(parsed[1], dict)):
                arguments = parsed[1]
                key = json.dumps([scope, "api_call", arguments], sort_keys=True)
                item = _reference("api_call", arguments, value, mode, full_max_bytes)
        else:
            raise ValueError(f"unknown latest store {store!r}")
    elif store == "result":
        for candidate in runtime.results:
            if candidate[2] != identity:
                continue
            auth_epoch, data_epoch, _ = candidate
            if auth_epoch != runtime.auth_epoch or data_epoch != runtime.data_epoch:
                continue
            expiry, value = runtime.results[candidate]
            if runtime.clock() >= expiry:
                continue
            result_scope, name, arguments = json.loads(identity)
            if result_scope != scope:
                continue
            key, item = identity, _reference(name, arguments, value)
            break
    elif store == "catalog":
        for candidate, value in runtime.catalogs.items():
            identity_text, _version, auth_epoch = candidate
            if identity_text != identity or auth_epoch != runtime.auth_epoch:
                continue
            try:
                parsed = json.loads(identity_text)
            except (TypeError, ValueError):
                continue
            if not (isinstance(parsed, list) and len(parsed) == 2):
                continue
            result_scope, arguments = parsed
            if result_scope != scope or not isinstance(arguments, dict):
                continue
            # Catalog reads are reached through the adapter's api_call tool.
            key = json.dumps([scope, "api_call", arguments], sort_keys=True)
            item = _reference("api_call", arguments, value)
            break
    else:
        raise ValueError(f"unknown latest store {store!r}")
    if item is None:
        return None, 0, None
    size = _json_bytes(item)
    if size > budget_bytes:
        return None, 0, None
    return item, size, key


def collect_references(runtime, scope, budget_bytes, visible, limit=2, latest=None,
                       mode="ref_only", full_max_bytes=0):
    """Publish compact validity references for reads already shown to the model.

    The result content is already in the conversation; repeating it would add
    input tokens without adding information. ``visible`` holds the identities
    whose results the conversation already shows. A reference states that the
    runtime still validates that exact read under the current scope, auth
    epoch, data epoch and TTL, so the agent can restate it instead of calling
    again. Newest reads are published first and the list is capped by both
    ``limit`` and ``budget_bytes``.

    ``latest`` selects exact reads instead: a single ``(store, identity)``
    tuple or an ordered list of them, where ``store`` is ``"result"`` for a
    data read or ``"catalog"`` for an M1 directory read. The caller passes the
    reads it wants re-confirmed, most recent first, and gets one reference per
    read that the runtime still validates.
    """
    if budget_bytes <= 0 or limit <= 0:
        return [], 0, []
    if latest:
        entries = [latest] if isinstance(latest, tuple) else list(latest)
        items, identities, used = [], [], 0
        for store, identity in entries:
            item, size, key = _latest_reference(runtime, scope, budget_bytes, store, identity,
                                                mode, full_max_bytes)
            if item is None:
                continue
            if used + size > budget_bytes:
                break
            items.append(item)
            identities.append(key)
            used += size
            if len(items) >= limit:
                break
        return items, used, identities
    items, identities, used = [], [], 0
    if _scheduled(runtime):
        for identity, value in runtime.iter_valid_reads():
            if identity not in visible:
                continue
            result_scope, name, arguments = json.loads(identity)
            if result_scope != scope:
                continue
            item = _reference(name, arguments, value, mode, full_max_bytes)
            size = _json_bytes(item)
            if used + size > budget_bytes:
                break
            items.append(item)
            identities.append(identity)
            used += size
            if len(items) >= limit:
                break
        return items, used, identities
    for key in reversed(list(runtime.results)):
        auth_epoch, data_epoch, identity = key
        if identity not in visible:
            continue
        if auth_epoch != runtime.auth_epoch or data_epoch != runtime.data_epoch:
            continue
        expiry, value = runtime.results[key]
        if runtime.clock() >= expiry:
            continue
        result_scope, name, arguments = json.loads(identity)
        if result_scope != scope:
            continue
        item = _reference(name, arguments, value)
        size = _json_bytes(item)
        if used + size > budget_bytes:
            break
        items.append(item)
        identities.append(identity)
        used += size
        if len(items) >= limit:
            break
    return items, used, identities


def collect_replacements(runtime, scope, budget_bytes, shown_this_turn, visible, limit=12,
                         full_max_bytes=1024):
    """Publish only what can stand in for a call.

    A read made in the current turn has just been handed to the model as a
    tool result; saying it again adds bytes and replaces nothing, so it is
    left out (``shown_this_turn``). Every other read the runtime still
    validates is one the model would otherwise have to ask for again:

    - its result is still in the conversation from an earlier turn
      (``visible``): a short "still valid" reference is enough;
    - its result is not in this conversation at all - the history was
      compacted, or another agent on this host made the read: the key values
      themselves are published.

    Newest reads first, capped by ``limit`` and ``budget_bytes``. Validity is
    the runtime's own (scope, authorization, data state, TTL): nothing is
    published that a repeated call would not be answered with from the cache.
    """
    if budget_bytes <= 0 or limit <= 0 or not _scheduled(runtime):
        return [], 0, []
    items, identities, used = [], [], 0
    for identity, value in runtime.iter_valid_reads():
        if identity in shown_this_turn:
            continue
        result_scope, name, arguments = json.loads(identity)
        if result_scope != scope:
            continue
        # One short record per read; what the fields mean is said once, in the
        # sentence the runner puts in front of the list.
        item = {"call": name, "arguments": redact(copy.deepcopy(arguments))}
        # ``visible`` may map an identity to the version the conversation shows:
        # a pointer is right only while that is the version the host holds.
        shown = visible.get(identity) if isinstance(visible, dict) else (
            True if identity in visible else None)
        if shown is True or (shown is not None and shown == version_of(value)):
            item["shown_earlier"] = True
        else:
            item["values"] = redact(key_values(copy.deepcopy(value), full_max_bytes))
        size = _json_bytes(item)
        if used + size > budget_bytes:
            break
        items.append(item)
        identities.append(identity)
        used += size
        if len(items) >= limit:
            break
    return items, used, identities


def _named(arguments, text):
    """Whether ``text`` names everything a read is about: each string argument
    appears in it as a word of its own. A read without arguments names nothing."""
    words = [value for value in arguments.values() if isinstance(value, str)]
    if not words or len(words) != len(arguments):
        return False
    lowered = text.lower()
    return all(re.search(rf"(?<![\w-]){re.escape(word.lower())}(?![\w-])", lowered)
               for word in words)


def choose_prefetch(history, text, turn, is_valid, limit):
    """The reads worth making before the model decides: ``[identity]``.

    ``history`` maps a read's identity to ``{"name", "arguments", "asked",
    "fetched"}`` - the last turn the agent asked for it and the last turn the
    host fetched it ahead. A read is chosen when the host holds no current
    result for it and either the user's message names what it reads, or the
    agent asked for it in the preceding turn (a value being followed). A read
    fetched ahead last turn stays followed for one more turn, no longer: one
    the agent stopped asking about is dropped. Named reads come first, then
    the most recently asked; at most ``limit``.
    """
    chosen = []
    for identity, record in history.items():
        if is_valid(identity):
            continue
        named = _named(record["arguments"], text)
        followed = record["asked"] == turn - 1 or (
            record["fetched"] == turn - 1 and record["asked"] == turn - 2)
        if named or followed or record["asked"] == turn:
            chosen.append((not named, -record["asked"], identity))
    return [identity for _, _, identity in sorted(chosen)[:max(0, limit)]]


def describe_failure(record):
    """One runtime-written sentence about a failed/uncertain operation.

    It names the tool and when it was tried but does not repeat the literal
    arguments: showing the verbatim failed call can make a model re-emit it.
    """
    request = record.get("request") or {}
    step = record.get("_step")
    when = f" at decision step {step}" if step is not None else ""
    if record.get("state") == "uncertain":
        return (f"A write through tool '{request.get('name')}'{when} got no confirmed "
                "outcome; it may or may not have taken effect. Check the resulting "
                "state before trying it again.")
    failure = record.get("failure")
    head = (f"A call to tool '{request.get('name')}'{when} failed with "
            f"'{record.get('error_code')}'")
    executed = "" if record.get("outcome") != "not_executed" else " and was not executed"
    if getattr(failure, "retryable", False):
        # Rate limits, timeouts and the like: saying "it will fail again" would be false.
        return head + executed + ". The failure was temporary; the same request may succeed later."
    return (head + executed + ". The state it depends on has not changed since, so the "
            "same request would fail again.")


def collect_operations(runtime, limit=20, budget_bytes=None, failure_form="verbatim"):
    """Summarize failed/uncertain operations under the current epochs.

    The summary never contains results or failure objects. An uncertain write
    advanced the data epoch itself, so it stays visible under the current
    epoch until the Host reconciles it; a plain failure is only published while
    the data epoch still matches the one it was observed in. ``budget_bytes``
    caps the published bytes; the first item that would overflow stops
    collection instead of silently overspending.
    """
    items, used = [], 0
    ordered = sorted(runtime.operations.items(),
                     key=lambda pair: (pair[1].get("created_at", 0), str(pair[0])))
    for _, record in ordered:
        state = record.get("state")
        if state not in {"failed", "uncertain"}:
            continue
        if record.get("error_code") == "invalid_arguments":
            # A malformed request is not a fact about the tool or the data: an
            # agent shown it tends to give up on the tool altogether.
            continue
        if _scheduled(runtime):
            # Tag-versioned conditions replace the global epochs.
            if not runtime.operation_current(record):
                continue
        elif record.get("auth_epoch") != runtime.auth_epoch:
            continue
        elif state == "failed" and record.get("data_epoch") != runtime.data_epoch:
            continue
        request = record.get("request") or {}
        item = {"name": request.get("name"),
                "arguments": redact(copy.deepcopy(request.get("arguments"))),
                "state": state,
                "error_code": record.get("error_code"),
                "outcome": record.get("outcome"),
                "attempts": record.get("attempts"),
                "retry_at": record.get("retry_at")}
        if state == "uncertain":
            item["note"] = "reconcile before retry"
        if failure_form == "described":
            item = {"state": state, "description": describe_failure(record)}
        if budget_bytes is not None:
            size = _json_bytes(item)
            if used + size > budget_bytes:
                break
            used += size
        items.append(item)
        if len(items) >= limit:
            break
    return items


def reference_large_repeat(name, arguments, result, seen, min_bytes):
    """Hand a large result to the model once; later byte-identical copies
    become a short pointer to the call that first returned it.

    A cache hit costs no request, but its result would still be written into
    the context in full every time. ``seen`` maps a fingerprint to the first
    call that produced it. Results below ``min_bytes`` are returned unchanged
    and are not remembered: repeating a small value costs little, and keeping
    it next to the question is safer. Returns ``(payload, saved_bytes)``.
    """
    payload = {"name": name, "result": result}
    size = _json_bytes(payload)
    if not min_bytes or size < min_bytes:
        return payload, 0
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    fingerprint = hashlib.sha256(canonical.encode()).hexdigest()
    if fingerprint not in seen:
        seen[fingerprint] = {"name": name, "arguments": arguments}
        return payload, 0
    reference = {"name": name, "result_ref": fingerprint[:16],
                 "same_result_as": seen[fingerprint],
                 "note": "byte-identical to the result that earlier call returned in "
                         "this conversation; it is current, read the values there"}
    return reference, max(0, size - _json_bytes(reference))


def dedupe_result(result, seen):
    """Replace a byte-identical repeat with a reference to the earlier result.

    Returns ``(payload, saved_bytes, duplicated)``. The fingerprint is added
    to ``seen`` on first use; ``saved_bytes`` is the serialized byte difference
    between the original payload and the reference, never negative.
    """
    canonical = json.dumps(result, sort_keys=True, separators=(",", ":"), default=str)
    fingerprint = hashlib.sha256(canonical.encode()).hexdigest()
    if fingerprint not in seen:
        seen.add(fingerprint)
        return result, 0, False
    reference = {"name": result.get("name"), "result_ref": fingerprint[:16],
                 "note": "identical to earlier result"}
    saved = max(0, _json_bytes(result) - _json_bytes(reference))
    return reference, saved, True
