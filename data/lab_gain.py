"""Cases built to show where a host-side cache can save model work, and one
where it cannot.

``./run gain-build`` writes ``data/lab-gain-tasks.json`` and
``data/lab-gain-cases.json``.

The two cases of ``lab_long`` showed fewer requests but no fewer model rounds:
a model sends the reads of a turn in one round, so dropping some of them saves
no round. A round is saved only when a whole turn needs no call, or when a
lookup that had to come first (which accounts are mine, which sku is the desk)
is already known. These cases differ in how often that happens:

- ``lab-gain-revisit``: one agent, compacted history; in half the turns the
  user comes back to values that have not changed since they were read.
- ``lab-gain-chain``: one agent, compacted history; questions name things
  indirectly, so a list has to be read before the value can be.
- ``lab-gain-handoff``: three agents on one host; a reporting agent is asked
  for what the other two have just read.
- ``lab-gain-control``: same length, nothing to reuse - every question is about
  a volatile value, a value a write has just changed, or a new one.

What can be reused is decided by the server's contract and by the writes and
turns of the task, not by the method under test.
"""
import copy
import json

from data.lab_long import (ACCOUNT, CATEGORY, PRICE, RATE, SHARED, STOCK, build_task, order,
                           read, transfer, turn)
from data.lab_stress import CRED, call, login
from exp_core.settings import DATA

MINE = ("checking", "savings", "travel", "bills")
SIGN_IN = [[login("bank"), login("shop")]]

REVISIT = [
    turn(f"{CRED} Sign in to the bank and the shop.", read(ACCOUNT, *MINE), before=SIGN_IN),
    turn("", read(CATEGORY, "sku-pen", "sku-lamp", "sku-desk", "sku-chair")
         + read(RATE, "EUR", "GBP")),
    turn("", read(ACCOUNT, "checking", "savings") + read(RATE, "EUR")),
    turn("", read(CATEGORY, "sku-desk", "sku-chair") + read(ACCOUNT, "travel")),
    turn("Move 100 from checking to bills. Then:", read(ACCOUNT, "checking", "bills"),
         before=[transfer("checking", "bills", 100)]),
    turn("", read(ACCOUNT, "savings", "travel") + read(RATE, "GBP")
         + read(CATEGORY, "sku-lamp")),
    turn("", read(PRICE, "sku-chair") + read(STOCK, "sku-chair")),
    turn("", read(ACCOUNT, "checking", "bills", "savings")),
    turn("Order 1 of sku-lamp. Then:", read(STOCK, "sku-lamp"),
         before=[order("sku-lamp", 1)]),
    turn("One last round.", read(ACCOUNT, *MINE) + read(RATE, "EUR", "GBP")
         + read(CATEGORY, "sku-pen")),
]

CONTROL = [
    turn(f"{CRED} Sign in to the bank and the shop.", read(ACCOUNT, "checking", "savings"),
         before=SIGN_IN),
    turn("", read(PRICE, "sku-lamp", "sku-chair") + read(STOCK, "sku-lamp", "sku-chair")),
    turn("", read(PRICE, "sku-lamp") + read(STOCK, "sku-chair")),
    turn("Move 50 from checking to savings. Then:", read(ACCOUNT, "checking", "savings"),
         before=[transfer("checking", "savings", 50)]),
    turn("", read(PRICE, "sku-desk", "sku-pen") + read(STOCK, "sku-desk")),
    turn("Move 50 from savings to checking. Then:", read(ACCOUNT, "checking", "savings"),
         before=[transfer("savings", "checking", 50)]),
    turn("", read(STOCK, "sku-lamp", "sku-pen") + read(PRICE, "sku-chair")),
    turn("Order 1 of sku-chair. Then:", read(STOCK, "sku-chair") + read(PRICE, "sku-chair"),
         before=[order("sku-chair", 1)]),
    turn("Move 20 from checking to travel. Then:", read(ACCOUNT, "checking", "travel"),
         before=[transfer("checking", "travel", 20)]),
    turn("One last round.", read(PRICE, "sku-lamp", "sku-desk")
         + read(STOCK, "sku-lamp", "sku-desk")),
]


def chain(text, probes, lookups=(), before=()):
    """A turn whose question names things indirectly: ``lookups`` are the reads
    that resolve the names, ``probes`` the reads that hold the answer."""
    calls = [{**copy.deepcopy(p["call"]), "as": f"p{i}"} for i, p in enumerate(probes)]
    answer = "; ".join(f"{p['label']}: {{p{i}[{p['field'][0]}]}}" for i, p in enumerate(probes))
    steps = [*before, *([[call(name) for name in lookups]] if lookups else []), calls]
    return {"text": text, "probes": probes, "gold": {"steps": steps, "answer": answer}}


ACCOUNTS, ITEMS, CURRENCIES = "bank_list_accounts", "shop_list_items", "ref_list_currencies"
CHAIN = [
    chain(f"{CRED} Sign in to the bank and the shop. Tell me the balance of every bank "
          "account that belongs to me.", read(ACCOUNT, *MINE), [ACCOUNTS], before=SIGN_IN),
    chain("Tell me the category of the shop items named Desk and Chair.",
          read(CATEGORY, "sku-desk", "sku-chair"), [ITEMS]),
    chain("Tell me the exchange rate of every currency the reference app lists.",
          read(RATE, "EUR", "GBP", "JPY"), [CURRENCIES]),
    chain("Which of my bank accounts holds the most, and how much is in it?",
          read(ACCOUNT, "savings"), [ACCOUNTS]),
    chain("Tell me the category of the shop item named Lamp, and the exchange rate of "
          "every listed currency.", read(CATEGORY, "sku-lamp") + read(RATE, "EUR", "GBP", "JPY"),
          [ITEMS, CURRENCIES]),
    chain("Move 100 from savings to travel. Then tell me the balance of every bank account "
          "that belongs to me.", read(ACCOUNT, *MINE), [ACCOUNTS],
          before=[transfer("savings", "travel", 100)]),
    chain("Tell me the category of the shop item named Desk and the exchange rates of the "
          "listed currencies.", read(CATEGORY, "sku-desk") + read(RATE, "EUR", "GBP", "JPY"),
          [ITEMS, CURRENCIES]),
    chain("One last round: the balance of every bank account that belongs to me, and the "
          "category of the shop items named Chair and Lamp.",
          read(ACCOUNT, *MINE) + read(CATEGORY, "sku-chair", "sku-lamp"), [ACCOUNTS, ITEMS]),
]

F, S, R = "finance", "shopping", "reporting"
HANDOFF = [
    (F, turn(f"{CRED} Sign in to the bank.", read(ACCOUNT, "checking", "savings", "travel")
             + read(RATE, "EUR"), before=[[login("bank")]])),
    (S, turn(f"{CRED} {SHARED} Sign in to the shop.",
             read(CATEGORY, "sku-desk", "sku-chair", "sku-lamp") + read(PRICE, "sku-desk"),
             before=[[login("shop")]])),
    (R, turn(f"{CRED} {SHARED} For the report:", read(ACCOUNT, "checking", "savings")
             + read(RATE, "EUR") + read(CATEGORY, "sku-desk", "sku-chair"))),
    (F, turn("Move 200 from savings to checking. Then:", read(ACCOUNT, "savings", "checking"),
             before=[transfer("savings", "checking", 200)])),
    (R, turn("For the report:", read(ACCOUNT, "checking", "savings", "travel")
             + read(CATEGORY, "sku-lamp"))),
    (S, turn("Order 1 of sku-desk. Then:", read(STOCK, "sku-desk"),
             before=[order("sku-desk", 1)])),
    (R, turn("For the report:", read(ACCOUNT, "checking") + read(RATE, "EUR")
             + read(CATEGORY, "sku-desk") + read(STOCK, "sku-desk"))),
    (F, turn("", read(RATE, "GBP", "JPY") + read(ACCOUNT, "travel"))),
    (S, turn(SHARED, read(ACCOUNT, "checking", "savings") + read(RATE, "GBP"))),
    (R, turn("One last round for the report.", read(ACCOUNT, "checking", "savings", "travel")
             + read(RATE, "EUR", "GBP", "JPY") + read(CATEGORY, "sku-chair"))),
]


def build():
    return [
        build_task("lab-gain-revisit", "gain_revisit",
                   "一个智能体、10 个 turn，历史被压缩。一半的 turn 回头问之前读过、之后没有变过的值。",
                   REVISIT, context="answers_only"),
        build_task("lab-gain-chain", "gain_chain",
                   "一个智能体、8 个 turn，历史被压缩。问题用间接的说法指东西（我名下的账户、"
                   "叫 Desk 的商品），要先读列表才能读到值。", CHAIN, context="answers_only"),
        build_task("lab-gain-handoff", "gain_handoff",
                   "三个智能体共用一个 Host。汇报智能体被问到的，是另外两个刚读过的值。",
                   [t for _, t in HANDOFF], agents=[agent for agent, _ in HANDOFF]),
        build_task("lab-gain-control", "gain_control",
                   "对照：长度相同，但没有可复用的内容。每个问题问的都是易变值、刚被写改过的值或新的值。",
                   CONTROL, context="answers_only"),
    ]


def main(argv=None):
    tasks = build()
    (DATA / "lab-gain-tasks.json").write_text(
        json.dumps(tasks, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    cases = [{"case_id": t["id"], "dataset": "lab", "task_id": t["id"], "scenario": t["family"],
              "split": "dev", "augmented": False, "augmentation": None} for t in tasks]
    (DATA / "lab-gain-cases.json").write_text(json.dumps({
        "version": "lab-gain-cases-v0", "suite": "lab",
        "note": "为模型侧开销设计的四条：回访、多步查询、三智能体交接、无可复用内容的对照。"
                "由 data/lab_gain.py 生成。",
        "cases": cases}, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    for task in tasks:
        print(f"{task['id']}: {len(task['turns'])} turns, "
              f"{len(task['overlay']['probes'])} asked values, "
              f"{len(task['expected']['write_log'])} writes")
    return 0
