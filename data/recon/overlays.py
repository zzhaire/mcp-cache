"""Scenario overlays shared by the three rebuilt benchmarks.

A rebuilt task keeps an original benchmark task (its environment, its user
turns, its reference calls) and adds what the original environment cannot
show: the same read needed again, a write followed by a re-read, a change
made by someone else, a sign-in the server drops or lets expire, a reply that
gets lost, and a large catalog that changes. Every overlay is the same code
for every benchmark; a benchmark only supplies its *probes* - reads whose
value can be asked for, with the external change that would move it.

Probe: ``{"call": {"name", "arguments"}, "field": <key path in the result>,
"label": <one word naming the value>, "ask": <how a user asks for it>,
"fresh": "slow" | "volatile", "event": <external change, optional>}``.
"""
import copy

from data.recon.common import SIGN_IN, finish

FAMILIES = ("repeat_read", "write_then_read", "external_volatile", "external_slow",
            "auth_revoked", "auth_expiry", "lost_ack_keyed", "lost_request_keyed",
            "lost_ack_unkeyed", "lost_request_unkeyed", "transient", "catalog")
RETRY_NOTE = (" If a call reports a problem, make sure the action ends up applied "
              "exactly once.")


def _field(path):
    return "".join(f"[{part}]" for part in path)


def probe_turn(probes, again=False):
    """One user turn that asks for every probe value: ``(text, gold turn)``."""
    asked = ", ".join(p["ask"] for p in probes)
    text = (f"Check again and tell me the current values: {asked}." if again
            else f"Tell me: {asked}.")
    calls = [{**copy.deepcopy(p["call"]), "as": f"p{i}"} for i, p in enumerate(probes)]
    answer = "; ".join(f"{p['label']}: {{p{i}{_field(p['field'])}}}"
                       for i, p in enumerate(probes))
    return text, {"steps": [calls], "answer": answer}


def _checks(run, turn, probes):
    """What the reference answer of a probe turn states, as oracle checks."""
    out = []
    for probe, segment in zip(probes, run["answers"][turn].split("; ")):
        text = segment.split(": ", 1)[1]
        try:
            token = float(text)
            token = int(token) if token.is_integer() and "." not in text else token
        except ValueError:
            token = text
        out.append((turn, token, probe["label"]))
    return out


def _build(base, family, turns, gold, probe_turns, base_errors, **extra):
    """Assemble a task and derive its oracle; None if the overlay does not fit
    (the reference run meets an error the original task does not have)."""
    task = {**{k: v for k, v in base.items() if k not in ("turns", "gold", "prelude")},
            "id": f"{base['id']}-{family}", "family": family, "turns": turns,
            "gold": gold, "events": [], "faults": [], **extra}
    task["reference_errors_ok"] = True
    try:
        done, run = finish(task)
    except RuntimeError:
        # The external change has nothing to act on in this task (the original
        # trace deleted the account, for example).
        return None
    # A refused request after the server dropped a sign-in is part of the scenario.
    errors = [e for e in run["errors"] if e["error"] != "unauthorized"]
    if len(errors) > base_errors or len(run["answers"]) != len(turns):
        return None
    checks = [c for turn, probes in probe_turns for c in _checks(run, turn, probes)]
    if any(c[1] == "<none>" for c in checks):
        return None
    done, _ = finish(task, checks)
    done.pop("reference_errors_ok")
    # What a real model's run is judged on (``judge_overlay``): the reads the
    # added turns ask for, so the server can note their true value each turn.
    done["overlay"] = {"probes": [
        {"turn": turn, "call": probe["call"], "field": probe["field"],
         "label": probe["label"]} for turn, probes in probe_turns for probe in probes]}
    return done


def _retry(gold, turn, step, alias="w", verify=None):
    """Make the write at ``gold[turn].steps[step]`` recoverable.

    Without ``verify`` the agent simply sends it again when it reports a
    problem - correct only for a write that carries an operation key. With
    ``verify`` (a read whose result changes when the write is applied) the
    agent reads before the write, reads again after a problem, and sends the
    write again only if nothing changed.
    """
    gold = copy.deepcopy(gold)
    call = gold[turn]["steps"][step][0]
    call["as"] = alias
    again = {**copy.deepcopy(call), "if_error": alias}
    if verify is None:
        gold[turn]["steps"].insert(step + 1, [again])
        return gold
    before = {**copy.deepcopy(verify), "as": "pre"}
    after = {**copy.deepcopy(verify), "as": "chk", "if_error": alias}
    gold[turn]["steps"][step:step + 1] = [[before], [call], [after],
                                          [{**again, "if": "chk == pre"}]]
    return gold


def overlays(base, probes, writes, non_idempotent, base_errors=0, keyed=(), verify=None):
    """Every rebuilt task of one original task.

    ``base``: the original task in lab form (first gold step signs in).
    ``writes``: ``[(turn, step, tool)]`` of the reference calls that write.
    ``non_idempotent``: writes a second application of which changes the
    outcome. Of those, ``keyed`` accept an operation key; for the others
    ``verify[tool]`` is the read that shows whether the write was applied.
    """
    verify = verify or {}
    prelude = base.get("prelude", 1)  # leading reference steps that sign in
    out = []
    turns, gold = base["turns"], base["gold"]
    n = len(turns)
    slow = [p for p in probes if p["fresh"] == "slow"]
    volatile = [p for p in probes if p["fresh"] == "volatile"]

    def add(task):
        if task is not None:
            out.append(task)

    def build(family, turns, gold, probe_turns, **extra):
        return _build(base, family, turns, gold, probe_turns, base_errors, **extra)

    if probes:
        ask, first = probe_turn(probes)
        again, second = probe_turn(probes, again=True)
        once, third = probe_turn(probes[:1], again=True)
        add(build("repeat_read", turns + [ask, again, once], gold + [first, second, third],
                  [(n, probes), (n + 1, probes), (n + 2, probes[:1])]))
        if writes:
            # The probe is read before the original task and again after it.
            login, rest = gold[0]["steps"][:prelude], gold[0]["steps"][prelude:]
            opening = {"steps": [*login, *first["steps"]], "answer": first["answer"]}
            head = turns[0].split(SIGN_IN, 1)
            credentials = head[0] + SIGN_IN
            add(build("write_then_read",
                      [credentials + ask, head[-1], *turns[1:], again],
                      [opening, {"steps": rest, "answer": gold[0]["answer"]}, *gold[1:], second],
                      [(0, probes), (n + 1, probes)]))
    for family, pool in (("external_volatile", volatile), ("external_slow", slow)):
        for index, changing in enumerate(p for p in pool if p.get("event")):
            probe = [changing]
            ask, first = probe_turn(probe)
            again, second = probe_turn(probe, again=True)
            task = build(family, turns + [ask, again], gold + [first, second],
                         [(n, probe), (n + 1, probe)],
                         events=[{"at_turn": n + 1, **changing["event"]}])
            if task is not None and index:
                task["id"] += f"-{index + 1}"
            add(task)
    private = [p for p in slow if p.get("app")]
    if private:
        probe = private[:1]
        ask, first = probe_turn(probe)
        again, second = probe_turn(probe, again=True)
        add(build("auth_revoked", turns + [ask, again], gold + [first, second],
                  [(n, probe), (n + 1, probe)],
                  events=[{"at_turn": n + 1, "kind": "expire_session",
                           "app": probe[0]["app"]}]))
        add(build("auth_expiry", turns + [ask, again], gold + [first, second],
                  [(n, probe), (n + 1, probe)], session_ttl_s=60))
    # One lost reply (and one lost request) on a write with an operation key,
    # and one on a write without: there the agent has to look before it resends.
    with_key = [w for w in writes if w[2] in non_idempotent and w[2] in keyed]
    without_key = [w for w in writes if w[2] in non_idempotent and w[2] not in keyed
                   and w[2] in verify]
    tail = ([probe_turn(probes)] if probes else [])
    more_turns, more_gold = [t for t, _ in tail], [g for _, g in tail]
    checked = [(n, probes)] if probes else []
    for suffix, risky in (("keyed", with_key), ("unkeyed", without_key)):
        if not risky:
            continue
        turn, step, tool = risky[0]
        check = None
        if suffix == "unkeyed":
            check = verify[tool]
            if callable(check):  # the read depends on what is being written
                check = check(gold[turn]["steps"][step][0]["arguments"])
        retried = _retry(gold, turn, step, verify=check)
        texts = list(turns)
        texts[turn] += RETRY_NOTE
        for kind in ("lost_ack", "lost_request"):
            add(build(f"{kind}_{suffix}", texts + more_turns, retried + more_gold, checked,
                      faults=[{"tool": tool, "kind": kind, "on_calls": [1]}]))
    if writes or probes:
        # A transient failure of the first reference call after signing in.
        first_step = gold[0]["steps"][prelude] if len(gold[0]["steps"]) > prelude else None
        if first_step:
            retried = _retry(gold, 0, prelude, alias="t")
            add(build("transient", turns, retried, [],
                      faults=[{"tool": first_step[0]["name"], "kind": "unavailable",
                               "on_calls": [1]}]))
    ask, first = probe_turn(probes) if probes else (None, None)
    extra_turns = [ask] if probes else []
    extra_gold = [first] if probes else []
    add(build("catalog", turns + extra_turns, gold + extra_gold,
              [(n, probes)] if probes else [], catalog_padding=40,
              events=[{"at_turn": max(1, n - 1), "kind": "catalog_update", "version": "v2",
                       "pad": 2}]))
    return out
