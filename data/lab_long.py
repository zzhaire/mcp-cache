"""Two lab cases for what a single short conversation cannot show.

``./run long-build`` writes ``data/lab-long-tasks.json`` and
``data/lab-long-cases.json``.

- ``lab-long-compacted``: one agent, twelve turns. Its history is compacted:
  when a turn ends, only the question and the answer stay, the tool results
  are gone (``"context": "answers_only"``). Values read earlier are needed
  again in later turns, between writes and one change made by someone else.
- ``lab-two-agents``: two agents work for the same user through one host and
  take turns (``"agents"``). Each sees only its own conversation, so one
  agent's reads are not in the other's context; the host's cache is the only
  thing they share.

Every question asks for values a single read returns, so a real model's run
can be judged against the server's own state (``judge_overlay``).
"""
import copy
import json

from data.lab_stress import BASE, CRED, call, deposit, login, price  # noqa: F401
from data.recon.common import finish
from data.recon.overlays import _checks
from exp_core.settings import DATA

ACCOUNT = {"name": "bank_get_balance", "field": ["balance"], "key": "account",
           "ask": "the {0} balance", "label": "{0}"}
RATE = {"name": "ref_get_rate", "field": ["rate"], "key": "currency",
        "ask": "the {0} exchange rate", "label": "{0}"}
PRICE = {"name": "shop_get_price", "field": ["price"], "key": "sku",
         "ask": "the current price of {0}", "label": "{0} price"}
STOCK = {"name": "shop_get_stock", "field": ["stock"], "key": "sku",
         "ask": "the stock of {0}", "label": "{0} stock"}
CATEGORY = {"name": "shop_get_item", "field": ["category"], "key": "sku",
            "ask": "the category of {0}", "label": "{0} category"}


def read(kind, *names):
    return [{"call": {"name": kind["name"], "arguments": {kind["key"]: name}},
             "field": kind["field"], "label": kind["label"].format(name),
             "ask": kind["ask"].format(name)} for name in names]


def turn(lead, probes, before=()):
    """A user turn: optional writes or sign-ins (``before``), then one question
    for every probe value."""
    asked = ", ".join(p["ask"] for p in probes)
    calls = [{**copy.deepcopy(p["call"]), "as": f"p{i}"} for i, p in enumerate(probes)]
    answer = "; ".join(f"{p['label']}: {{p{i}[{p['field'][0]}]}}" for i, p in enumerate(probes))
    return {"text": f"{lead} Tell me: {asked}.".strip(), "probes": probes,
            "gold": {"steps": [*before, calls], "answer": answer}}


def build_task(task_id, family, note, turns, **extra):
    task = {"id": task_id, "family": family, "note": note, "initial": copy.deepcopy(BASE),
            "turns": [t["text"] for t in turns], "gold": [t["gold"] for t in turns],
            "events": [], "faults": [], **extra}
    _, run = finish(task)
    if run["errors"]:
        raise ValueError(f"{task_id}: reference run has errors: {run['errors']}")
    checks = [c for index, t in enumerate(turns) for c in _checks(run, index, t["probes"])]
    task, _ = finish(task, checks)
    task["overlay"] = {"probes": [
        {"turn": index, "call": p["call"], "field": p["field"], "label": p["label"]}
        for index, t in enumerate(turns) for p in t["probes"]]}
    return task


def transfer(source, target, amount):
    return [call("bank_transfer", source=source, target=target, amount=amount)]


def order(sku, quantity):
    return [call("shop_place_order", sku=sku, quantity=quantity)]


ALL_ACCOUNTS = ("checking", "savings", "travel", "bills")

LONG = [
    turn(f"{CRED} Sign in to the bank and the shop.", read(ACCOUNT, *ALL_ACCOUNTS),
         before=[[login("bank"), login("shop")]]),
    turn("", read(CATEGORY, "sku-pen", "sku-lamp", "sku-desk", "sku-chair")),
    turn("", read(RATE, "EUR", "GBP") + read(ACCOUNT, "checking")),
    turn("", read(PRICE, "sku-chair", "sku-lamp") + read(STOCK, "sku-chair", "sku-lamp")),
    turn("Move 100 from checking to bills. Then:", read(ACCOUNT, "checking", "bills"),
         before=[transfer("checking", "bills", 100)]),
    turn("", read(CATEGORY, "sku-chair", "sku-pen") + read(PRICE, "sku-chair", "sku-pen")),
    turn("Order 2 of sku-chair. Then:", read(STOCK, "sku-chair"),
         before=[order("sku-chair", 2)]),
    turn("", read(ACCOUNT, *ALL_ACCOUNTS)),
    turn("", read(RATE, "GBP", "JPY") + read(ACCOUNT, "savings")),
    turn("", read(PRICE, "sku-lamp") + read(ACCOUNT, "travel")),
    turn("Move 250 from checking to rent. Then:", read(ACCOUNT, "checking"),
         before=[transfer("checking", "rent", 250)]),
    turn("One last round.", read(ACCOUNT, *ALL_ACCOUNTS) + read(RATE, "EUR")
         + read(CATEGORY, "sku-desk") + read(STOCK, "sku-chair")),
]

A, B = "finance", "shopping"
SHARED = ("You work for the same user as another agent, through the same host. "
          "The other agent may already have signed in; sign in only if an app asks for it.")
TWO = [
    (A, turn(f"{CRED} Sign in to the bank.", read(ACCOUNT, "checking", "savings")
             + read(RATE, "EUR"), before=[[login("bank")]])),
    (B, turn(f"{CRED} {SHARED} Sign in to the shop.",
             read(CATEGORY, "sku-desk", "sku-chair") + read(PRICE, "sku-desk", "sku-chair")
             + read(ACCOUNT, "checking"), before=[[login("shop")]])),
    (A, turn("Move 200 from savings to checking. Then:", read(ACCOUNT, "savings", "checking"),
             before=[transfer("savings", "checking", 200)])),
    (B, turn("Order 1 of sku-desk. Then:", read(STOCK, "sku-desk") + read(ACCOUNT, "checking"),
             before=[order("sku-desk", 1)])),
    (A, turn(SHARED, read(CATEGORY, "sku-desk", "sku-chair") + read(RATE, "EUR", "GBP"))),
    (B, turn("", read(ACCOUNT, "checking", "savings") + read(RATE, "GBP"))),
    (A, turn("", read(STOCK, "sku-desk") + read(PRICE, "sku-chair")
             + read(ACCOUNT, "checking"))),
    (B, turn("One last round.", read(CATEGORY, "sku-desk") + read(RATE, "EUR")
             + read(ACCOUNT, "savings", "travel"))),
]


def build():
    long_task = build_task(
        "lab-long-compacted", "long_compacted",
        "一个智能体、12 个 turn。每个 turn 结束后历史被压缩，只留问题和回答，工具结果不再可见。"
        "前面读过的值在后面的 turn 里被再次需要，中间有两次转账、一次下单，"
        "第 9 个 turn 开始时台灯价格被别人改了。", LONG,
        context="answers_only", events=[price(9, "sku-lamp", 44)])
    two = build_task(
        "lab-two-agents", "two_agents",
        "两个智能体为同一个用户工作，共用一个 Host，轮流回答。各自只看得到自己的对话，"
        "一方读过的值不在另一方的上下文里；一方的写会让另一方之前读到的值过期。",
        [t for _, t in TWO], agents=[agent for agent, _ in TWO])
    return [long_task, two]


def main(argv=None):
    tasks = build()
    (DATA / "lab-long-tasks.json").write_text(
        json.dumps(tasks, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    cases = [{"case_id": t["id"], "dataset": "lab", "task_id": t["id"], "scenario": t["family"],
              "split": "dev", "augmented": False, "augmentation": None} for t in tasks]
    (DATA / "lab-long-cases.json").write_text(json.dumps({
        "version": "lab-long-cases-v0", "suite": "lab",
        "note": "长任务（历史被压缩）和双智能体共用 Host 各一条。由 data/lab_long.py 生成。",
        "cases": cases}, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    for task in tasks:
        print(f"{task['id']}: {len(task['turns'])} turns, "
              f"{len(task['overlay']['probes'])} asked values, "
              f"{len(task['expected']['write_log'])} writes")
    return 0
