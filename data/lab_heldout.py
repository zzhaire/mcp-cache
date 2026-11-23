"""Held-out lab cases, produced by a seeded generator and then frozen.

``python -m data.lab_heldout`` writes ``data/lab-heldout-tasks.json``,
``data/lab-heldout-cases.json`` and ``data/lab-heldout.lock.json``.

The development cases (``data/lab_stress.py``) were written by hand, one at a
time, while the methods were being built. These are different in kind: eight
scenario *structures*, each instantiated three times with accounts, amounts,
items, the number of rounds, and the moment and kind of every disturbance drawn
from a random generator. The values each answer must state are not written by
hand either: they are read off the reference solution, run with no cache.

Use: generate once, freeze (the lock file holds the hash of the task file and
the seed), and do not look at any method's result on them while choosing
parameters. The lock is checked by the tests; regenerating must give the same
bytes. What freezing cannot undo: the generator's author knew how the methods
work, and the default parameters were already chosen on the development cases.
"""
import copy
import hashlib
import json
import random
import re
from pathlib import Path

from data.lab_stress import (BASE, CRED, ONCE, after, at_time, balance, call, careful_transfer,
                             catalog, expire, fault, login, price, says, served, stock, transfer,
                             turn)

DATA = Path(__file__).resolve().parent
SEED = 20261120
VARIANTS = 3
MINE = ["checking", "savings", "travel", "bills"]          # alice's accounts
START = {name: BASE["bank"]["accounts"][name]["balance"] for name in MINE}
SKUS = {"sku-pen": 2.5, "sku-lamp": 40, "sku-desk": 250, "sku-chair": 60}
STOCK = {"sku-pen": 100, "sku-lamp": 8, "sku-desk": 3, "sku-chair": 20}


def make(task_id, family, note, turns, *, events=(), faults=(), served_rules=(), **flags):
    return {"id": task_id, "family": family, "note": note, "initial": copy.deepcopy(BASE),
            "turns": [t["text"] for t in turns], "events": list(events), "faults": list(faults),
            "expected": {"answers": [], "served": list(served_rules)},
            "gold": [{k: v for k, v in t.items() if k != "text"} for t in turns], **flags}


def ask(names):
    return ", ".join(names[:-1]) + (" and " if len(names) > 1 else "") + names[-1]


# ------------------------------------------------------------- structures
def interleaved_accounts(rng, n):
    """Transfers between varying pairs; each is followed by reads of accounts
    it did and did not touch. The bank session is dropped once, mid-turn."""
    money, turns = dict(START), []
    rounds = rng.choice([3, 4])
    turns.append(turn(f"{CRED} Sign in to the bank and tell me the {ask(MINE)} balances.",
                      [[login("bank")], [balance(a, a[0]) for a in MINE]],
                      ", ".join(f"{a} {{{a[0]}[balance]}}" for a in MINE)))
    reads_so_far = len(MINE)
    drop_after = reads_so_far + rng.randint(1, 2 * rounds - 1)
    for _ in range(rounds):
        source, target = rng.sample(MINE, 2)
        amount = rng.choice([10, 20, 25, 40, 60])
        money[source] -= amount
        money[target] += amount
        watch = rng.sample(MINE, 2)
        turns.append(turn(
            f"Move {amount} from {source} to {target}, then tell me the {ask(watch)} "
            "balances. Sign in again if the bank asks for it.",
            [[transfer(source, target, amount)], [balance(a, a[0]) for a in watch]],
            ", ".join(f"{a} {{{a[0]}[balance]}}" for a in watch)))
    return make(f"ho-accounts-{n}", "ho_accounts", interleaved_accounts.__doc__, turns,
                events=[after("bank_get_balance", drop_after, "expire_session", app="bank")])


def price_watch(rng, n):
    """A price that others change, sometimes between two reads of one turn;
    an order is placed only while the price is at or under a limit."""
    sku = rng.choice(["sku-lamp", "sku-chair"])
    name, base = sku[4:], SKUS[sku]
    limit = base + rng.choice([2, 5])
    turns = [turn(f"{CRED} Sign in to the shop and tell me the price and the stock of {sku}.",
                  [[login("shop")], [call("shop_get_price", "p", sku=sku),
                                     call("shop_get_stock", "s", sku=sku)]],
                  f"{name} {{p[price]}}, stock {{s[stock]}}")]
    events, reads = [], 1
    for index in range(1, rng.choice([3, 4]) + 1):
        new = base + rng.choice([-6, -3, 3, 8, 12])
        if rng.random() < 0.5:
            events.append(price(index, sku, new))                       # at the turn start
        else:
            events.append(after("shop_get_price", reads + 1, "set_price", sku=sku, price=new))
        qty = rng.choice([1, 2])
        turns.append(turn(
            f"Check the {sku} price, then its stock, then the price once more. If that last "
            f"price is {limit} or less, order {qty}. Tell me the last price and the stock at "
            "the end.",
            [[call("shop_get_price", "p1", sku=sku)], [call("shop_get_stock", "s1", sku=sku)],
             [call("shop_get_price", "p", sku=sku)],
             [call("shop_place_order", when=f"p[price] <= {limit}", sku=sku, quantity=qty)],
             [call("shop_get_stock", "s", sku=sku)]],
            f"{name} {{p[price]}}, stock {{s[stock]}}"))
        reads += 2
    return make(f"ho-price-{n}", "ho_price", price_watch.__doc__, turns, events=events)


def restock_race(rng, n):
    """Stock that runs out and comes back; an order can be refused, after
    which the stock has to be read again."""
    sku = rng.choice(["sku-desk", "sku-lamp"])
    name = sku[4:]
    turns = [turn(f"{CRED} Sign in to the shop. How many {sku} are in stock?",
                  [[login("shop")], [call("shop_get_stock", "s", sku=sku)]],
                  f"{name} stock {{s[stock]}}")]
    events, reads = [], 1
    for index in range(1, rng.choice([3, 4]) + 1):
        level = rng.choice([0, 0, 1, 2, 5])
        trigger = rng.random()
        if trigger < 0.4:
            events.append(stock(index, sku, level))
        elif trigger < 0.8:
            events.append(after("shop_get_stock", reads + 1, "set_stock", sku=sku, stock=level))
        turns.append(turn(
            f"How many {sku} are in stock now? If there is at least one, order 1. Then tell "
            "me the stock and whether an order was placed.",
            [[call("shop_get_stock", "s1", sku=sku)],
             [call("shop_place_order", "o", when="s1[stock] > 0", sku=sku, quantity=1)],
             [call("shop_get_stock", "s", sku=sku)]],
            f"{name} stock {{s[stock]}}"))
        reads += 2
    return make(f"ho-stock-{n}", "ho_stock", restock_race.__doc__, turns, events=events)


def catalog_rollout(rng, n):
    """The server changes its catalog while the task runs: a tool appears and is
    needed at once, and the meaning of a field changes without a new version."""
    appear, silent = rng.sample([1, 2, 3], 2)
    sku, qty = rng.choice(["sku-desk", "sku-chair"]), rng.choice([10, 12])
    turns = [turn(f"{CRED} Sign in to the shop. Which category is sku-lamp in, and what does "
                  f"{sku} cost?",
                  [[login("shop")], [call("shop_get_item", "i", sku="sku-lamp"),
                                     call("shop_get_price", "p", sku=sku)]],
                  f"category {{i[category]}}, {sku[4:]} {{p[price]}}")]
    events = []
    for index in (1, 2, 3):
        if index == appear:
            events.append(catalog(index, version=f"v{index + 1}", add=["shop_get_quote"]))
            turns.append(turn(f"The shop says it now offers quotes. Get a quote for {qty} "
                              f"{sku} and tell me the total.",
                              [[call("shop_get_quote", "q", sku=sku, quantity=qty)]],
                              "total {q[total]}"))
        elif index == silent:
            events.append(catalog(index, behavior={"category_codes": True}, describe={
                "shop_get_item": "Name and department code of one item."}))
            turns.append(turn("The shop reorganised its categories. What category do sku-lamp "
                              "and sku-pen report now?",
                              [[call("shop_get_item", "i", sku="sku-lamp"),
                                call("shop_get_item", "j", sku="sku-pen")]],
                              "lamp {i[category]}, pen {j[category]}"))
        else:
            turns.append(turn("What category does sku-desk report, and what is the EUR rate?",
                              [[call("shop_get_item", "i", sku="sku-desk"),
                                call("ref_get_rate", "r", currency="EUR")]],
                              "desk {i[category]}, EUR {r[rate]}"))
    return make(f"ho-catalog-{n}", "ho_catalog", catalog_rollout.__doc__, turns, events=events,
                catalog_ttl_ms=0, catalog_padding=rng.choice([0, 20, 40]))


def unclear_writes(rng, n):
    """Several transfers, each of which may lose its request, lose its reply
    or be turned away as busy. Each must end up applied exactly once."""
    money, turns, faults, calls = dict(START), [], [], 0
    turns.append(turn(f"{CRED} Sign in to the bank and tell me the checking balance.",
                      [[login("bank")], [balance("checking", "c")]], "checking {c[balance]}"))
    for _ in range(rng.choice([2, 3])):
        source, target = rng.sample(MINE, 2)
        amount = rng.choice([15, 30, 45, 70])
        kind = rng.choice(["lost_ack", "lost_request", "rate_limit", "lost_ack", None])
        calls += 1
        if kind:
            faults.append(fault("bank_transfer", kind, calls))
            # A dropped request is repeated by the agent, a busy reply by the host.
            calls += kind in ("lost_request", "rate_limit")
        steps = careful_transfer(source, target, amount, money[source])
        money[source] -= amount
        money[target] += amount
        turns.append(turn(
            f"Move {amount} from {source} to {target}. {ONCE} Then tell me the {source} and "
            f"{target} balances.",
            [*steps, [balance(source, "a"), balance(target, "b")]],
            f"{source} {{a[balance]}}, {target} {{b[balance]}}"))
    turns.append(turn(f"Finally, tell me the {ask(MINE)} balances.",
                      [[balance(a, a[0]) for a in MINE]],
                      ", ".join(f"{a} {{{a[0]}[balance]}}" for a in MINE)))
    return make(f"ho-writes-{n}", "ho_writes", unclear_writes.__doc__, turns, faults=faults)


def session_lifetimes(rng, n):
    """Sign-ins that run out on the clock, with a pause between turns."""
    ttl, gap = rng.choice([50, 80, 120]), rng.choice([0, 200])
    money, turns = dict(START), []
    turns.append(turn(f"{CRED} Sign in to the bank. Tell me the {ask(MINE[:3])} balances, one "
                      "request at a time.",
                      [[login("bank")], *[[balance(a, a[0])] for a in MINE[:3]]],
                      ", ".join(f"{a} {{{a[0]}[balance]}}" for a in MINE[:3])))
    for _ in range(rng.choice([2, 3])):
        if rng.random() < 0.5:
            source, target = rng.sample(MINE, 2)
            amount = rng.choice([5, 35, 55])
            money[source] -= amount
            money[target] += amount
            turns.append(turn(
                f"Move {amount} from {source} to {target} and tell me both balances. Sign in "
                "again if the bank asks for it.",
                [[transfer(source, target, amount)], [balance(source, "a"), balance(target, "b")]],
                f"{source} {{a[balance]}}, {target} {{b[balance]}}"))
        else:
            watch = rng.sample(MINE, 2)
            turns.append(turn(
                f"Sign in to the bank (same credentials) and tell me the {ask(watch)} balances.",
                [[login("bank")], [balance(a, a[0]) for a in watch]],
                ", ".join(f"{a} {{{a[0]}[balance]}}" for a in watch)))
    return make(f"ho-session-{n}", "ho_session", session_lifetimes.__doc__, turns,
                session_ttl_s=ttl, turn_gap_s=gap)


def two_users(rng, n):
    """Two people use the same bank through one host, one after the other."""
    mine, order = rng.choice(MINE), rng.choice(["ab", "aba", "abab"])
    turns, rules = [], []
    for index, who in enumerate(order):
        if who == "a":
            turns.append(turn(
                f"Sign in to the bank as alice (password pw-alice) and tell me the {mine} "
                "balance and the first account in the account list.",
                [[login("bank")], [balance(mine, "c"), call("bank_list_accounts", "l")]],
                f"{mine} {{c[balance]}}, first account {{l[0][account]}}"))
        else:
            turns.append(turn(
                f"Now sign in to the bank as bob (password pw-bob) without signing out. As bob, "
                f"read the bob-main balance and the {mine} balance, and tell me exactly what "
                "the service answers for each.",
                [[login("bank", user="bob")], [balance("bob-main", "m"), balance(mine, "c")]],
                f"bob-main {{m[balance]}}; {mine} balance: {{c[balance]}}; error: {{c[error]}}"))
            rules.append(served(index, "bank_get_balance", refused=True, account=mine))
    return make(f"ho-users-{n}", "ho_users", two_users.__doc__, turns, served_rules=rules,
                strict_ownership=True)


def mixed_day(rng, n):
    """Several of the above in one task: a transfer with an unclear reply, a
    price changed by others, a dropped session, a pause between turns."""
    sku = rng.choice(["sku-lamp", "sku-chair"])
    name, base = sku[4:], SKUS[sku]
    source, target = rng.sample(MINE, 2)
    amount, new = rng.choice([20, 50, 80]), base + rng.choice([-5, 6, 11])
    limit, kind = base + 4, rng.choice(["lost_ack", "lost_request"])
    turns = [
        turn(f"{CRED} Sign in to the bank and the shop. Tell me the {source} balance and the "
             f"{sku} price.",
             [[login("bank"), login("shop")],
              [balance(source, "a"), call("shop_get_price", "p", sku=sku)]],
             f"{source} {{a[balance]}}, {name} {{p[price]}}"),
        turn(f"Move {amount} from {source} to {target}. {ONCE} Then tell me both balances.",
             [*careful_transfer(source, target, amount, START[source]),
              [balance(source, "a"), balance(target, "b")]],
             f"{source} {{a[balance]}}, {target} {{b[balance]}}"),
        turn(f"Check the {sku} price again and order 1 if it is {limit} or less. Tell me the "
             f"price and the {sku} stock. Sign in again wherever needed.",
             [[call("shop_get_price", "p", sku=sku)],
              [call("shop_place_order", when=f"p[price] <= {limit}", sku=sku, quantity=1)],
              [call("shop_get_stock", "s", sku=sku)]],
             f"{name} {{p[price]}}, stock {{s[stock]}}"),
        turn(f"What are the {source} and {target} balances now, and the EUR rate?",
             [[balance(source, "a"), balance(target, "b"),
               call("ref_get_rate", "r", currency="EUR")]],
             f"{source} {{a[balance]}}, {target} {{b[balance]}}, EUR {{r[rate]}}")]
    events = [rng.choice([price(2, sku, new),
                          at_time(rng.choice([150, 250]), "set_price", sku=sku, price=new)]),
              expire(rng.choice([2, 3]), rng.choice(["bank", "shop"]))]
    return make(f"ho-mixed-{n}", "ho_mixed", mixed_day.__doc__, turns, events=events,
                faults=[fault("bank_transfer", kind, 1)], turn_gap_s=100)


STRUCTURES = (interleaved_accounts, price_watch, restock_race, catalog_rollout, unclear_writes,
              session_lifetimes, two_users, mixed_day)


# ------------------------------------------------- oracle from the reference
def _stated(template, answer):
    """The values the reference answer puts in the template's placeholders."""
    literal = re.split(r"\{[^{}]+\}", template)
    pattern = "(.+?)".join(re.escape(part) for part in literal) + "$"
    match = re.match(pattern, answer, re.DOTALL)
    fields, values = re.findall(r"\{([^{}]+)\}", template), []
    for field, text in zip(fields, match.groups() if match else ()):
        # An error's wording is not a value: a refusal is judged on what the
        # host handed over (the "served" rules), not on the words of the answer.
        if text in ("<none>", "") or field.endswith("[error]"):
            continue
        try:
            number = float(text)
            values.append(int(number) if number == int(number) else number)
        except ValueError:
            values.append(text)
    return values


def build():
    from data.lab_expect import BARE, enrich
    from exp_core.lab.harness import run_gold
    rng, tasks = random.Random(SEED), []
    for structure in STRUCTURES:
        for n in range(1, VARIANTS + 1):
            task = structure(rng, n)
            reference = run_gold({**copy.deepcopy(task), "faults": [],
                                  "expected": {"answers": []}}, BARE)
            task["expected"]["answers"] = [
                says(index, *_stated(gold["answer"], reference["answers"][index]))
                for index, gold in enumerate(task["gold"])]
            task = enrich(task)
            # The task must be solvable by a host with no cache, faults included.
            solved = run_gold(task, BARE)
            if not solved["oracle"]["success"]:
                bad = [c for c in solved["oracle"]["checks"] if not c["ok"]]
                raise ValueError(f"{task['id']} is not solvable without a cache: {bad}")
            tasks.append(task)
    return tasks


def main():
    tasks = build()
    text = json.dumps(tasks, ensure_ascii=False, indent=1) + "\n"
    (DATA / "lab-heldout-tasks.json").write_text(text, encoding="utf-8")
    cases = [{"case_id": t["id"], "dataset": "lab", "task_id": t["id"], "scenario": t["family"],
              "split": "heldout", "augmented": False, "augmentation": None} for t in tasks]
    (DATA / "lab-heldout-cases.json").write_text(json.dumps({
        "version": "lab-heldout-cases-v0", "suite": "lab",
        "note": "留出测试集：8 种场景结构，每种 3 个随机实例，由 data/lab_heldout.py 按固定种子生成后冻结。"
                "选参数时不得查看任何方法在这批 case 上的结果。不要手工修改。",
        "cases": cases}, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    (DATA / "lab-heldout.lock.json").write_text(json.dumps({
        "seed": SEED, "structures": [s.__name__ for s in STRUCTURES], "variants": VARIANTS,
        "tasks": len(tasks),
        "tasks_sha256": hashlib.sha256(text.encode()).hexdigest()}, indent=1) + "\n",
        encoding="utf-8")
    print(f"wrote {len(tasks)} held-out tasks; sha256 "
          f"{hashlib.sha256(text.encode()).hexdigest()[:16]}")


if __name__ == "__main__":
    main()
