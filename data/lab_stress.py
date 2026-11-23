"""Stress cases for the lab server: one extreme scenario per module (M1 catalog,
M2 read results, M3 credentials) and one mixed scenario, five cases each.

Run ``python -m data.lab_stress`` to (re)write the tasks into
``data/lab-tasks.json`` and the case list into ``data/lab-stress-cases.json``.
Each task has user turns, a scripted reference solution (``gold``) and an
oracle (``expected``): final state, number of writes, and tokens every turn's
answer must contain or must not contain.
"""
import copy
import json
from pathlib import Path

DATA = Path(__file__).resolve().parent
BASE = {
    "users": {"alice": "pw-alice", "bob": "pw-bob"},
    "bank": {"accounts": {
        "checking": {"owner": "alice", "balance": 500},
        "savings": {"owner": "alice", "balance": 1200},
        "travel": {"owner": "alice", "balance": 300},
        "bills": {"owner": "alice", "balance": 80},
        "bob-main": {"owner": "bob", "balance": 900},
        "rent": {"owner": "landlord", "balance": 0}}},
    "shop": {"items": {
        "sku-pen": {"name": "Pen", "category": "office", "price": 2.5, "stock": 100},
        "sku-lamp": {"name": "Lamp", "category": "home", "price": 40, "stock": 8},
        "sku-desk": {"name": "Desk", "category": "home", "price": 250, "stock": 3},
        "sku-chair": {"name": "Chair", "category": "office", "price": 60, "stock": 20}}},
    "ref": {"rates": {"EUR": 0.9, "GBP": 0.8, "JPY": 150}},
}
CRED = "Credentials for the bank and the shop: username alice, password pw-alice."
TASKS = []


def call(name, alias=None, when=None, if_error=None, **arguments):
    """One scripted tool call; ``when`` / ``if_error`` make it conditional."""
    item = {"name": name, "arguments": arguments}
    if alias:
        item["as"] = alias
    if when:
        item["if"] = when
    if if_error:
        item["if_error"] = if_error
    return item


def login(app, user="alice", **kw):
    return call("auth_login", app=app, username=user, password=f"pw-{user}", **kw)


def balance(account, alias, **kw):
    return call("bank_get_balance", alias, account=account, **kw)


def transfer(source, target, amount, alias="tr", **kw):
    return call("bank_transfer", alias, source=source, target=target, amount=amount, **kw)


def turn(text, steps, answer, no_recover=False):
    """``no_recover``: the scripted agent must not sign in again in this turn."""
    return {"text": text, "steps": steps, "answer": answer,
            **({"no_recover": True} if no_recover else {})}


def says(index, *contains, excludes=()):
    rule = {"turn": index, "contains": list(contains)}
    if excludes:
        rule["excludes"] = list(excludes)
    return rule


def served(index, tool, *, refused=False, excludes=(), **arguments):
    """A rule on what the agent was handed in a turn, not on how it words its
    answer: matching calls must all be refused, or must not contain a value."""
    rule = {"turn": index, "tool": tool, "arguments": arguments}
    rule.update({"refused": True} if refused else {"excludes": list(excludes)})
    return rule


def task(task_id, scenario, extreme, note, turns, *, state=None, writes=None, answers=(),
         served=(), events=(), faults=(), initial=None, **flags):
    world = copy.deepcopy(BASE)
    for section, patch in (initial or {}).items():
        world[section].update(patch)
    expected = {"writes": {"bank_transfer": 0, "shop_place_order": 0, **(writes or {})},
                "answers": list(answers)}
    if served:
        expected["served"] = list(served)
    if state:
        expected["state"] = state
    TASKS.append({
        "id": task_id, "family": scenario, "extreme": extreme, "note": note,
        "initial": world, "turns": [t["text"] for t in turns],
        "events": list(events), "faults": list(faults), "expected": expected,
        "gold": [{key: value for key, value in t.items() if key != "text"} for t in turns],
        **flags})


def catalog(at_turn, **change):
    return {"at_turn": at_turn, "kind": "catalog_update", **change}


def expire(at_turn, app):
    return {"at_turn": at_turn, "kind": "expire_session", "app": app}


def price(at_turn, sku, value):
    return {"at_turn": at_turn, "kind": "set_price", "sku": sku, "price": value}


def stock(at_turn, sku, value):
    return {"at_turn": at_turn, "kind": "set_stock", "sku": sku, "stock": value}


def deposit(at_turn, account, amount):
    return {"at_turn": at_turn, "kind": "deposit", "account": account, "amount": amount}


# =====================================================================  M1
RATES = [("EUR", 0.9), ("GBP", 0.8), ("JPY", 150), ("EUR", 0.9),
         ("GBP", 0.8), ("JPY", 150), ("EUR", 0.9), ("GBP", 0.8)]
task("m1-relist-large-stable", "m1_catalog", "目录很大且稳定，被反复列出",
     "8 个很短的 turn，每个 turn 开始都要列一次 78 个工具的目录，目录始终不变。"
     "考验：目录缓存能省掉多少次列目录，以及多少响应字节。",
     [turn(f"What is the {cur} exchange rate?" if i else
           f"No sign-in is needed for this. What is the {cur} exchange rate?",
           [[call("ref_get_rate", "r", currency=cur)]], "rate {r[rate]}")
      for i, (cur, _) in enumerate(RATES)],
     answers=[says(i, rate) for i, (_, rate) in enumerate(RATES)],
     catalog_padding=60, catalog_ttl_ms=600000)

task("m1-version-churn", "m1_catalog", "目录每个 turn 都变，新工具马上要用",
     "每个 turn 开始目录都升一个版本：新增工具、扩容或改描述。后面的问题只能用刚出现的工具回答。"
     "考验：目录缓存不能把变化挡在外面。",
     [turn(f"{CRED} Sign in to the shop and tell me the price of sku-desk.",
           [[login("shop")], [call("shop_get_price", "p", sku="sku-desk")]], "desk {p[price]}"),
      turn("The shop added a bulk-price tool. What is the bulk price of sku-desk?",
           [[call("shop_get_bulk_price", "b", sku="sku-desk")]], "bulk {b[bulk_price]}"),
      turn("The shop added a quote tool. What is the total for 10 sku-chair?",
           [[call("shop_get_quote", "q", sku="sku-chair", quantity=10)]], "total {q[total]}"),
      turn("What is the price of sku-chair?",
           [[call("shop_get_price", "p", sku="sku-chair")]], "chair {p[price]}"),
      turn("How many sku-chair are in stock?",
           [[call("shop_get_stock", "s", sku="sku-chair")]], "stock {s[stock]}")],
     answers=[says(0, 250), says(1, 225), says(2, 540), says(3, 60), says(4, 20)],
     events=[catalog(1, version="v2", add=["shop_get_bulk_price"]),
             catalog(2, version="v3", add=["shop_get_quote"]),
             catalog(3, version="v4", pad=5),
             catalog(4, version="v5", describe={
                 "shop_get_stock": "Units of one item in stock right now."})],
     catalog_ttl_ms=600000)

task("m1-silent-semantic-change", "m1_catalog", "版本号不变，工具的含义悄悄变了",
     "服务器声明目录有效期为 0（每次都要重新确认）。第 2 个 turn 开始时，商品信息工具的分类字段"
     "从名称改成了部门代码，描述随之改变，但版本号没变，也没有通知。"
     "考验：是否遵守服务器给的有效期，并靠内容比对发现变化、让该工具的旧结果失效。",
     [turn(f"{CRED} Sign in to the shop. Which category is sku-lamp in?",
           [[login("shop")], [call("shop_get_item", "i", sku="sku-lamp")]],
           "category {i[category]}"),
      turn("The shop reorganised its categories. What category does sku-lamp report now?",
           [[call("shop_get_item", "i", sku="sku-lamp")]], "category {i[category]}"),
      turn("And what categories do sku-desk and sku-pen report?",
           [[call("shop_get_item", "d", sku="sku-desk"), call("shop_get_item", "p", sku="sku-pen")]],
           "desk {d[category]}, pen {p[category]}")],
     answers=[says(0, "home"), says(1, "H-12"), says(2, "H-12", "O-07")],
     events=[catalog(1, describe={"shop_get_item": "Name and department code of one item."},
                     behavior={"category_codes": True})],
     catalog_ttl_ms=0)

READS = [balance("checking", "c"), balance("savings", "s"), balance("travel", "t"),
         call("shop_get_item", "i", sku="sku-lamp")]
task("m1-notification-noise", "m1_catalog", "目录变更通知不断，内容其实没变",
     "每个 turn 开始目录版本号都加一并发出变更通知，但工具一个没变。"
     "考验：收到通知要重新确认目录，但不该因此丢掉无关的读取结果。",
     [turn(f"{CRED} Sign in to the bank and the shop. Tell me the checking, savings and travel "
           "balances and the name of sku-lamp.",
           [[login("bank"), login("shop")], READS],
           "checking {c[balance]}, savings {s[balance]}, travel {t[balance]}, {i[name]}")] +
     [turn("Tell me the same four things again.", [READS],
           "checking {c[balance]}, savings {s[balance]}, travel {t[balance]}, {i[name]}")
      for _ in range(4)],
     answers=[says(i, 500, 1200, 300, "Lamp") for i in range(5)],
     events=[catalog(i, version=f"v{i + 1}") for i in range(1, 5)],
     catalog_ttl_ms=600000)

task("m1-tool-retired", "m1_catalog", "旧工具被悄悄下线，换成了新工具",
     "服务器声明目录有效期为 0。第 2 个 turn 开始时批量价格工具被下线，换成报价工具，版本号不变。"
     "考验：拿着旧目录既看不到新工具，又会去调用已经不存在的工具。",
     [turn(f"{CRED} Sign in to the shop. What is the bulk price of sku-desk?",
           [[login("shop")], [call("shop_get_bulk_price", "b", sku="sku-desk")]],
           "bulk {b[bulk_price]}"),
      turn("The shop replaced its bulk-price tool. Using whatever it offers now, what is the "
           "total for 10 sku-desk?",
           [[call("shop_get_bulk_price", "old", sku="sku-desk")],
            [call("shop_get_quote", "q", if_error="old", sku="sku-desk", quantity=10)]],
           "total {q[total]}"),
      turn("And the total for 10 sku-chair?",
           [[call("shop_get_quote", "q", sku="sku-chair", quantity=10)]], "total {q[total]}")],
     answers=[says(0, 225), says(1, 2250), says(2, 540)],
     events=[catalog(1, remove=["shop_get_bulk_price"], add=["shop_get_quote"])],
     tools_add=["shop_get_bulk_price"], catalog_ttl_ms=0)

# =====================================================================  M2
STORM = [balance("checking", "c"), balance("savings", "s"), balance("travel", "t"),
         call("shop_get_item", "i", sku="sku-lamp"), call("shop_list_orders", "o"),
         call("ref_get_rate", "r", currency="EUR")]
STORM_ANSWER = ("checking {c[balance]}, savings {s[balance]}, travel {t[balance]}, "
                "{i[name]}, EUR {r[rate]}")
task("m2-read-storm", "m2_reads", "同一批读取被反复请求，没有任何变化",
     "6 个 turn，每个 turn 都要同样的 6 项只读信息；没有写操作，也没有外部变化。"
     "考验：读取缓存的收益上限。",
     [turn(f"{CRED} Sign in to the bank and the shop. Tell me the checking, savings and travel "
           "balances, the name of sku-lamp, whether I have any orders, and the EUR rate.",
           [[login("bank"), login("shop")], STORM], STORM_ANSWER)] +
     [turn("Give me all of that once more.", [STORM], STORM_ANSWER) for _ in range(5)],
     answers=[says(i, 500, 1200, 300, "Lamp", 0.9) for i in range(6)])

FOUR = [balance("checking", "c"), balance("savings", "s"), balance("travel", "t"),
        balance("bills", "b")]
FOUR_ANSWER = "checking {c[balance]}, savings {s[balance]}, travel {t[balance]}, bills {b[balance]}"
task("m2-write-interleave", "m2_reads", "每个 turn 都写，写完立刻重读全部",
     "4 次转账分别涉及不同的账户对，每次转账后都要重读全部 4 个账户的余额。"
     "考验：每次写只应让受影响的两个账户失效；失效少了答案错，失效多了白白回源。",
     [turn(f"{CRED} Sign in to the bank. Tell me the checking, savings, travel and bills balances.",
           [[login("bank")], FOUR], FOUR_ANSWER),
      turn("Transfer 50 from checking to bills, then tell me all four balances.",
           [[transfer("checking", "bills", 50)], FOUR], FOUR_ANSWER),
      turn("Transfer 200 from savings to travel, then tell me all four balances.",
           [[transfer("savings", "travel", 200)], FOUR], FOUR_ANSWER),
      turn("Transfer 100 from travel to checking, then tell me all four balances.",
           [[transfer("travel", "checking", 100)], FOUR], FOUR_ANSWER),
      turn("Transfer 30 from bills to rent, then tell me all four balances and the rent balance.",
           [[transfer("bills", "rent", 30)], FOUR + [balance("rent", "r")]],
           FOUR_ANSWER + ", rent {r[balance]}")],
     state={"bank.accounts.checking.balance": 550, "bank.accounts.savings.balance": 1000,
            "bank.accounts.travel.balance": 400, "bank.accounts.bills.balance": 100,
            "bank.accounts.rent.balance": 30},
     writes={"bank_transfer": 4},
     answers=[says(0, 500, 1200, 300, 80), says(1, 450, 1200, 300, 130),
              says(2, 450, 1000, 500, 130), says(3, 550, 1000, 400, 130),
              says(4, 550, 1000, 400, 100, 30)])

task("m2-volatile-churn", "m2_reads", "易变数据每个 turn 都被外部改动，决策依赖最新值",
     "价格和库存每个 turn 开始都被外部改变，而且要根据最新值决定是否下单、下哪一个。"
     "考验：用了任何一个旧值都会下错单。",
     [turn(f"{CRED} Sign in to the shop. What do sku-lamp and sku-chair cost?",
           [[login("shop")],
            [call("shop_get_price", "l", sku="sku-lamp"), call("shop_get_price", "ch", sku="sku-chair")]],
           "lamp {l[price]}, chair {ch[price]}"),
      turn("Prices moved. Check both again and order 2 of whichever is cheaper now. "
           "Tell me both prices.",
           [[call("shop_get_price", "l", sku="sku-lamp"), call("shop_get_price", "ch", sku="sku-chair")],
            [call("shop_place_order", when="ch[price] < l[price]", sku="sku-chair", quantity=2),
             call("shop_place_order", when="l[price] <= ch[price]", sku="sku-lamp", quantity=2)]],
           "lamp {l[price]}, chair {ch[price]}"),
      turn("Order 1 sku-desk if it is in stock. If it is not, order 1 sku-lamp, but only if the "
           "lamp costs 35 or less. Tell me the desk stock and the lamp price.",
           [[call("shop_get_stock", "d", sku="sku-desk"), call("shop_get_price", "l", sku="sku-lamp")],
            [call("shop_place_order", when="d[stock] > 0", sku="sku-desk", quantity=1),
             call("shop_place_order", when="d[stock] == 0 and l[price] <= 35",
                  sku="sku-lamp", quantity=1)]],
           "desk stock {d[stock]}, lamp {l[price]}"),
      turn("How many sku-pen are in stock, and what does sku-chair cost now?",
           [[call("shop_get_stock", "p", sku="sku-pen"), call("shop_get_price", "ch", sku="sku-chair")]],
           "pen stock {p[stock]}, chair {ch[price]}")],
     state={"shop.orders.ord-1.sku": "sku-chair", "shop.orders.ord-1.unit_price": 45,
            "shop.orders.ord-2.sku": "sku-lamp", "shop.orders.ord-2.unit_price": 30,
            "shop.items.sku-desk.stock": 0},
     writes={"shop_place_order": 2},
     answers=[says(0, 40, 60), says(1, 55, 45), says(2, 0, 30), says(3, 5, 80)],
     events=[price(1, "sku-lamp", 55), price(1, "sku-chair", 45),
             stock(2, "sku-desk", 0), price(2, "sku-lamp", 30),
             stock(3, "sku-pen", 5), price(3, "sku-chair", 80)])

task("m2-contract-drift-repeated", "m2_reads", "契约说不会变的数据，每个 turn 都在变（对抗性）",
     "余额被声明为只随自己的写操作变化，但每个 turn 开始都有一笔外部存款。"
     "这是故意设的对抗性场景：照契约缓存的方法都会答错，只有回源校验才能发现。"
     "考验：抽样校验和有效期自适应能挽回多少。",
     [turn(f"{CRED} Sign in to the bank and tell me the checking balance.",
           [[login("bank")], [balance("checking", "c")]], "checking {c[balance]}")] +
     [turn("Deposits keep arriving. What is the checking balance now?",
           [[balance("checking", "c")]], "checking {c[balance]}") for _ in range(3)],
     state={"bank.accounts.checking.balance": 800},
     answers=[says(0, 500), says(1, 600), says(2, 700), says(3, 800)],
     events=[deposit(i, "checking", 100) for i in (1, 2, 3)], adversarial=True)

HISTORY = [call("bank_list_transfers", "c", account="checking"),
           call("bank_list_transfers", "t", account="travel")]
task("m2-large-payloads", "m2_reads", "返回结果很大",
     "checking 有 150 条转账记录（约 10 KB），travel 有 600 条（约 40 KB）。4 个 turn 都要这两份记录。"
     "考验：大结果能缓存的要缓存并答对。超过单条上限的情形放在机制测试里（把上限调低到 20 KB），"
     "因为真正超过默认上限 256 KB 的结果放不进模型的上下文。",
     [turn(f"{CRED} Sign in to the bank. Give me the id of the first transfer on checking and "
           "the first on travel.",
           [[login("bank")], HISTORY], "checking {c[0][id]}, travel {t[0][id]}")] +
     [turn("Look both histories up again and repeat those two ids.", [HISTORY],
           "checking {c[0][id]}, travel {t[0][id]}") for _ in range(3)],
     answers=[says(i, "chk-1", "trv-1") for i in range(4)],
     initial={"bank": {"seed_transfers": [
         {"prefix": "chk", "source": "checking", "target": "savings", "count": 150},
         {"prefix": "trv", "source": "travel", "target": "bills", "count": 600}]}})

# =====================================================================  M3
task("m3-relogin-storm", "m3_credentials", "每个 turn 都重新登录一遍",
     "6 个 turn，每个 turn 开头都先登录银行和商店，再读一项信息。会话始终有效。"
     "考验：凭据缓存能省掉多少次多余的登录。",
     [turn((f"{CRED} " if i == 0 else "") + "Sign in to the bank and the shop first, then tell me "
           + ("the checking balance." if i % 2 == 0 else "the name of sku-lamp."),
           [[login("bank"), login("shop")],
            [balance("checking", "x") if i % 2 == 0 else call("shop_get_item", "x", sku="sku-lamp")]],
           "checking {x[balance]}" if i % 2 == 0 else "{x[name]}")
      for i in range(6)],
     answers=[says(i, 500 if i % 2 == 0 else "Lamp") for i in range(6)])

CHURN = ["savings", "travel", "bills", "savings"]
CHURN_BAL = [1200, 300, 80, 1200]
task("m3-session-churn", "m3_credentials", "银行会话每个 turn 都被服务端作废",
     "从第 2 个 turn 起，银行会话在每个 turn 开始时被作废；商店会话一直有效。"
     "每个 turn 读一个银行余额和商店订单。考验：能否自动重新登录，以及是否只丢银行这一侧的缓存。",
     [turn(f"{CRED} Sign in to the bank and the shop. Tell me the checking balance and whether "
           "I have any orders.",
           [[login("bank"), login("shop")], [balance("checking", "b"), call("shop_list_orders", "o")]],
           "checking {b[balance]}")] +
     [turn(f"Tell me the {account} balance and check my orders again. Sign in again if needed.",
           [[balance(account, "b"), call("shop_list_orders", "o")]],
           f"{account} {{b[balance]}}") for account in CHURN],
     answers=[says(0, 500)] + [says(i + 1, value) for i, value in enumerate(CHURN_BAL)],
     events=[expire(i, "bank") for i in range(1, 5)])

task("m3-logout-must-not-serve", "m3_credentials", "登出之后不得再返回缓存的数据",
     "先读余额，再主动登出银行，然后再读同一个余额。正确的结果是服务拒绝，而不是返回登出前缓存的值。"
     "考验：登出必须让该账号的缓存立即失效，这是安全问题而不只是新鲜度问题。",
     [turn(f"{CRED} Sign in to the bank and tell me the checking and savings balances.",
           [[login("bank")], [balance("checking", "c"), balance("savings", "s")]],
           "checking {c[balance]}, savings {s[balance]}"),
      turn("Sign out of the bank. Then try to read the checking balance anyway and tell me "
           "exactly what the service answers.",
           [[call("auth_logout", app="bank")], [balance("checking", "c")]],
           "balance: {c[balance]}; error: {c[error]}", no_recover=True),
      turn("Sign in to the bank again and tell me the checking and savings balances.",
           [[login("bank")], [balance("checking", "c"), balance("savings", "s")]],
           "checking {c[balance]}, savings {s[balance]}")],
     answers=[says(0, 500, 1200), says(1), says(2, 500, 1200)],
     served=[served(1, "bank_get_balance", refused=True, account="checking")])

task("m3-user-switch", "m3_credentials", "同一个应用里换了用户，不能把上一个用户的数据带过去",
     "先以 alice 登录并读取她的账户，然后不登出、直接以 bob 登录。bob 无权查看 alice 的账户。"
     "考验：缓存按应用区分还不够，换了用户必须清掉上一个用户的缓存，否则就是跨用户泄露。",
     [turn("Sign in to the bank as alice (password pw-alice). Tell me the checking balance and "
           "the first account in my account list.",
           [[login("bank")], [balance("checking", "c"), call("bank_list_accounts", "a")]],
           "checking {c[balance]}, first account {a[0][account]}"),
      turn("Now sign in to the bank as bob (password pw-bob) without signing out first. As bob, "
           "read the bob-main balance and the checking balance, and tell me exactly what the "
           "service answers for each.",
           [[login("bank", user="bob")], [balance("bob-main", "m"), balance("checking", "c")]],
           "bob-main {m[balance]}; checking balance: {c[balance]}; error: {c[error]}"),
      turn("Still as bob: what is the first account in the account list?",
           [[call("bank_list_accounts", "a")]], "first account {a[0][account]}"),
      turn("Sign in as alice again and tell me the checking balance.",
           [[login("bank")], [balance("checking", "c")]], "checking {c[balance]}")],
     answers=[says(0, 500, "bills"), says(1, 900), says(2, "bob-main"), says(3, 500)],
     served=[served(1, "bank_get_balance", refused=True, account="checking"),
             served(2, "bank_list_accounts", excludes=["bills", "checking"])],
     strict_ownership=True)

task("m3-expiry-before-write", "m3_credentials", "会话恰好在写操作之前过期",
     "两次转账之前，银行会话都刚被服务端作废。转账第一次会被拒绝（未执行），重新登录后再发。"
     "考验：授权失败后的重试必须安全，转账只能生效一次。",
     [turn(f"{CRED} Sign in to the bank and tell me the checking balance.",
           [[login("bank")], [balance("checking", "c")]], "checking {c[balance]}"),
      turn("Transfer 100 from checking to rent (sign in again if the bank asks), then tell me "
           "the checking and rent balances.",
           [[transfer("checking", "rent", 100)],
            [balance("checking", "c"), balance("rent", "r")]],
           "checking {c[balance]}, rent {r[balance]}"),
      turn("Transfer 50 from checking to bills (sign in again if needed), then tell me the "
           "checking and bills balances.",
           [[transfer("checking", "bills", 50)],
            [balance("checking", "c"), balance("bills", "b")]],
           "checking {c[balance]}, bills {b[balance]}")],
     state={"bank.accounts.checking.balance": 350, "bank.accounts.rent.balance": 100,
            "bank.accounts.bills.balance": 130},
     writes={"bank_transfer": 2},
     answers=[says(0, 500), says(1, 400, 100), says(2, 350, 130)],
     events=[expire(1, "bank"), expire(2, "bank")])

# ==================================================================  mixed
task("mix-office-day", "mixed", "一段较长的日常操作：目录、读取、凭据、写操作都出现",
     "6 个 turn、47 个工具的目录。其间有一次转账、一次按价格决定的下单、一次目录更新、"
     "一次外部调价、银行和商店各一次会话作废、一次库存外部变化。",
     [turn(f"{CRED} Sign in to the bank and the shop. Tell me the checking and savings balances "
           "and the price and stock of sku-chair.",
           [[login("bank"), login("shop")],
            [balance("checking", "c"), balance("savings", "s"),
             call("shop_get_price", "p", sku="sku-chair"), call("shop_get_stock", "k", sku="sku-chair")]],
           "checking {c[balance]}, savings {s[balance]}, chair {p[price]}, stock {k[stock]}"),
      turn("Transfer 100 from checking to bills. Then tell me the checking, bills and savings balances.",
           [[transfer("checking", "bills", 100)],
            [balance("checking", "c"), balance("bills", "b"), balance("savings", "s")]],
           "checking {c[balance]}, bills {b[balance]}, savings {s[balance]}"),
      turn("Check the sku-chair price again and order 3 if it is 55 or less. Tell me the price "
           "and my checking balance. Sign in again wherever needed.",
           [[call("shop_get_price", "p", sku="sku-chair"), balance("checking", "c")],
            [call("shop_place_order", when="p[price] <= 55", sku="sku-chair", quantity=3)]],
           "chair {p[price]}, checking {c[balance]}"),
      turn("The shop added a quote tool. What is the total for 10 sku-chair?",
           [[call("shop_get_quote", "q", sku="sku-chair", quantity=10)]], "total {q[total]}"),
      turn("Tell me the savings and bills balances and the EUR rate.",
           [[balance("savings", "s"), balance("bills", "b"), call("ref_get_rate", "r", currency="EUR")]],
           "savings {s[balance]}, bills {b[balance]}, EUR {r[rate]}"),
      turn("How many sku-chair are in stock now? Sign in again if needed.",
           [[call("shop_get_stock", "k", sku="sku-chair")]],
           "stock {k[stock]}")],
     state={"bank.accounts.checking.balance": 400, "bank.accounts.bills.balance": 180,
            "shop.orders.ord-1.sku": "sku-chair", "shop.orders.ord-1.quantity": 3,
            "shop.orders.ord-1.unit_price": 50},
     writes={"bank_transfer": 1, "shop_place_order": 1},
     answers=[says(0, 500, 1200, 60, 20), says(1, 400, 180, 1200), says(2, 50, 400),
              says(3, 450), says(4, 1200, 180, 0.9), says(5, 2)],
     events=[price(2, "sku-chair", 50), expire(2, "bank"),
             catalog(3, version="v2", add=["shop_get_quote"]),
             expire(5, "shop"), stock(5, "sku-chair", 2)],
     catalog_padding=30, catalog_ttl_ms=600000)

task("mix-private-catalog-user-switch", "mixed", "目录随登录状态变化，中途还换了用户",
     "目录是私有的：只有登录了某个应用才看得到它的工具。其间登录、换用户、登出交替发生。"
     "考验：目录、读取结果、凭据三者都要跟着身份变化走，不能串。",
     [turn("No sign-in yet: what is the EUR rate? Then sign in to the bank as alice (password "
           "pw-alice) and tell me the checking balance.",
           [[call("ref_get_rate", "r", currency="EUR")], [login("bank")], [balance("checking", "c")]],
           "EUR {r[rate]}, checking {c[balance]}"),
      turn("Sign in to the shop as alice too (same password, pw-alice) and tell me the price "
           "of sku-desk.",
           [[login("shop")], [call("shop_get_price", "p", sku="sku-desk")]], "desk {p[price]}"),
      turn("Now sign in to the bank as bob (password pw-bob) without signing out. As bob, tell "
           "me the bob-main balance and exactly what the service answers for the checking balance.",
           [[login("bank", user="bob")], [balance("bob-main", "m"), balance("checking", "c")]],
           "bob-main {m[balance]}; checking balance: {c[balance]}; error: {c[error]}"),
      turn("Sign out of the shop. Then try to get the price of sku-desk anyway and tell me what "
           "happens. Also tell me the EUR rate.",
           [[call("auth_logout", app="shop")],
            [call("shop_get_price", "p", sku="sku-desk"), call("ref_get_rate", "r", currency="EUR")]],
           "desk price: {p[price]}; error: {p[error]}; EUR {r[rate]}", no_recover=True),
      turn("Sign in to the bank as alice again. Tell me the checking balance and the first "
           "account in the account list.",
           [[login("bank")], [balance("checking", "c"), call("bank_list_accounts", "a")]],
           "checking {c[balance]}, first account {a[0][account]}")],
     answers=[says(0, 0.9, 500), says(1, 250), says(2, 900), says(3, 0.9),
              says(4, 500, "bills")],
     served=[served(2, "bank_get_balance", refused=True, account="checking"),
             # The catalog is private: after signing out the price tool is no longer
             # listed, so not calling it is a correct answer too. What must not
             # happen is that the old price is handed over.
             served(3, "shop_get_price", excludes=[250], sku="sku-desk")],
     strict_ownership=True, catalog_private=True, catalog_ttl_ms=600000)

MIX3 = lambda: [  # noqa: E731 - one turn's script, rebuilt per turn
    [transfer("checking", "savings", 10)],
    [balance("checking", "c"), balance("savings", "s"), call("shop_get_price", "p", sku="sku-lamp")]]
task("mix-everything-churns", "mixed", "每个 turn 所有东西都在变",
     "每个 turn 开始：目录版本号加一（内容没变）、灯的价格被外部调高、至少一个会话被作废；"
     "每个 turn 还要转一次账。考验：三类状态同时抖动时仍然全部答对，转账恰好 4 次。",
     [turn(f"{CRED} Sign in to the bank and the shop. Tell me the checking and savings balances, "
           "the price of sku-lamp and its name.",
           [[login("bank"), login("shop")],
            [balance("checking", "c"), balance("savings", "s"),
             call("shop_get_price", "p", sku="sku-lamp"), call("shop_get_item", "i", sku="sku-lamp")]],
           "checking {c[balance]}, savings {s[balance]}, lamp {p[price]}, {i[name]}")] +
     [turn(f"New request, separate from any earlier one: this is transfer {i} of the 4 I will "
           "ask for. Transfer 10 from checking to savings exactly once now, then tell me the "
           "checking and savings balances and the current sku-lamp price. Sign in again "
           "wherever needed.", MIX3(),
           "checking {c[balance]}, savings {s[balance]}, lamp {p[price]}") for i in range(1, 5)],
     state={"bank.accounts.checking.balance": 460, "bank.accounts.savings.balance": 1240},
     writes={"bank_transfer": 4},
     answers=[says(0, 500, 1200, 40, "Lamp")] +
             [says(i, 500 - 10 * i, 1200 + 10 * i, 40 + 2 * i) for i in range(1, 5)],
     events=[catalog(i, version=f"v{i + 1}") for i in range(1, 5)] +
            [price(i, "sku-lamp", 40 + 2 * i) for i in range(1, 5)] +
            [expire(1, "bank"), expire(2, "bank"), expire(3, "shop"),
             expire(4, "bank"), expire(4, "shop")],
     catalog_ttl_ms=600000)

task("mix-silent-change-under-load", "mixed", "工具含义两次悄悄改变，同时有写操作和会话作废",
     "服务器声明目录有效期为 0，目录有 37 个工具。分类字段和价格单位先后被悄悄改掉（版本号不变），"
     "其间银行会话被作废一次，还有一次转账和一次下单。",
     [turn(f"{CRED} Sign in to the bank and the shop. Which category is sku-lamp in, and what is "
           "my checking balance?",
           [[login("bank"), login("shop")],
            [call("shop_get_item", "i", sku="sku-lamp"), balance("checking", "c")]],
           "category {i[category]}, checking {c[balance]}"),
      turn("What category does sku-lamp report now, and what is my checking balance? Sign in "
           "again if needed.",
           [[call("shop_get_item", "i", sku="sku-lamp"), balance("checking", "c")]],
           "category {i[category]}, checking {c[balance]}"),
      turn("Transfer 50 from checking to travel. Then tell me the category of sku-desk and my "
           "checking balance.",
           [[transfer("checking", "travel", 50)],
            [call("shop_get_item", "i", sku="sku-desk"), balance("checking", "c")]],
           "category {i[category]}, checking {c[balance]}"),
      turn("What does the price tool report for sku-lamp now? Give the number as reported.",
           [[call("shop_get_price", "p", sku="sku-lamp")]], "lamp {p[price]}"),
      turn("Order 1 sku-pen and tell me which category sku-pen reports.",
           [[call("shop_place_order", sku="sku-pen", quantity=1)],
            [call("shop_get_item", "i", sku="sku-pen")]], "category {i[category]}")],
     state={"bank.accounts.checking.balance": 450, "bank.accounts.travel.balance": 350,
            "shop.orders.ord-1.sku": "sku-pen"},
     writes={"bank_transfer": 1, "shop_place_order": 1},
     answers=[says(0, "home", 500), says(1, "H-12", 500),
              says(2, "H-12", 450), says(3, 4000), says(4, "O-07")],
     events=[catalog(1, describe={"shop_get_item": "Name and department code of one item."},
                     behavior={"category_codes": True}),
             expire(1, "bank"),
             catalog(3, describe={"shop_get_price": "Current unit price of one item, in cents."},
                     behavior={"price_unit": "cents"})],
     catalog_padding=20, catalog_ttl_ms=0)

DESK = [call("shop_get_stock", "st", sku="sku-desk"), call("shop_get_price", "p", sku="sku-desk")]
task("mix-write-heavy-expiry-volatile", "mixed", "跨应用的连环写操作，依赖最新价格和库存，会话中途作废",
     "两次“先从储蓄转出货款、再下单”的连环操作，是否执行取决于桌子当时的价格和库存；"
     "价格、库存被外部改了三次，银行和商店会话各被作废一次。",
     [turn(f"{CRED} Sign in to the bank and the shop. Tell me the stock and price of sku-desk and "
           "my checking and savings balances.",
           [[login("bank"), login("shop")],
            DESK + [balance("checking", "c"), balance("savings", "s")]],
           "stock {st[stock]}, price {p[price]}, checking {c[balance]}, savings {s[balance]}"),
      turn("Check the sku-desk price again. If it is 220 or less, move 200 from savings to "
           "checking and order 1 sku-desk. Tell me the price you saw.",
           [[call("shop_get_price", "p", sku="sku-desk")],
            [transfer("savings", "checking", 200, when="p[price] <= 220")],
            [call("shop_place_order", when="p[price] <= 220", sku="sku-desk", quantity=1)]],
           "price {p[price]}"),
      turn("Order another sku-desk only if one is in stock. Tell me the desk stock and my "
           "checking balance. Sign in again if needed.",
           [[call("shop_get_stock", "st", sku="sku-desk"), balance("checking", "c")],
            [call("shop_place_order", when="st[stock] > 0", sku="sku-desk", quantity=1)]],
           "stock {st[stock]}, checking {c[balance]}"),
      turn("Check sku-desk once more (sign in again if needed). If it is in stock and costs 190 "
           "or less, move 180 from savings to checking and order 1. Tell me the price and stock "
           "you saw.",
           [DESK,
            [transfer("savings", "checking", 180, when="st[stock] > 0 and p[price] <= 190")],
            [call("shop_place_order", when="st[stock] > 0 and p[price] <= 190",
                  sku="sku-desk", quantity=1)]],
           "price {p[price]}, stock {st[stock]}"),
      turn("Tell me my final checking and savings balances.",
           [[balance("checking", "c"), balance("savings", "s")]],
           "checking {c[balance]}, savings {s[balance]}")],
     state={"bank.accounts.checking.balance": 880, "bank.accounts.savings.balance": 820,
            "shop.items.sku-desk.stock": 4, "shop.orders.ord-1.unit_price": 200,
            "shop.orders.ord-2.unit_price": 180},
     writes={"bank_transfer": 2, "shop_place_order": 2},
     answers=[says(0, 3, 250, 500, 1200), says(1, 200), says(2, 0, 700),
              says(3, 180, 5), says(4, 880, 820)],
     events=[price(1, "sku-desk", 200), stock(2, "sku-desk", 0), expire(2, "bank"),
             stock(3, "sku-desk", 5), price(3, "sku-desk", 180), expire(3, "shop")])

# ==================================================================  timing
# Changes that do not wait for a turn boundary: between two calls of the same
# turn, or at a point in virtual time. Nothing here lines up with a host's
# end-of-turn housekeeping.
def after(tool, nth, kind, **change):
    """An event that takes effect right after the n-th call of ``tool``."""
    return {"after_call": {"tool": tool, "nth": nth}, "kind": kind, **change}


def at_time(seconds, kind, **change):
    return {"at_time": seconds, "kind": kind, **change}


task("time-midturn-price-change", "timing", "价格在同一轮里、两次读取之间被改掉",
     "同一轮里先看一次价格，做别的事，再看一次价格并据此决定是否下单。第一次读取之后价格被外部调高。"
     "考验：轮内的外部变化。只在轮次结束时清理易变数据的做法在这里不起作用。",
     [turn(f"{CRED} Sign in to the bank and the shop. Tell me the price of sku-lamp. Then tell "
           "me my checking balance and the stock of sku-lamp. After that, check the sku-lamp "
           "price once more and order 2 only if it is still 40 or less. Tell me the first "
           "price, the last price and the lamp stock at the end.",
           [[login("bank"), login("shop")],
            [call("shop_get_price", "p1", sku="sku-lamp")],
            [balance("checking", "c"), call("shop_get_stock", "s1", sku="sku-lamp")],
            [call("shop_get_price", "p2", sku="sku-lamp")],
            [call("shop_place_order", when="p2[price] <= 40", sku="sku-lamp", quantity=2)],
            [call("shop_get_stock", "s2", sku="sku-lamp")]],
           "first {p1[price]}, last {p2[price]}, stock {s2[stock]}"),
      turn("What does sku-lamp cost now, and how many are in stock?",
           [[call("shop_get_price", "p", sku="sku-lamp"), call("shop_get_stock", "s", sku="sku-lamp")]],
           "lamp {p[price]}, stock {s[stock]}"),
      turn("And my checking balance?", [[balance("checking", "c")]], "checking {c[balance]}")],
     answers=[says(0, 40, 52, 8), says(1, 52, 8), says(2, 500)],
     events=[after("shop_get_price", 1, "set_price", sku="sku-lamp", price=52)])

task("time-stock-gone-before-order", "timing", "读到有货，下单前被别人买走",
     "读库存时还有 3 件，紧接着库存被外部清零，下单被拒。之后要报告当前库存。"
     "考验：写操作失败后，之前缓存的库存不能再当作当前值。",
     [turn(f"{CRED} Sign in to the shop. How many sku-desk are in stock? If there is at least "
           "one, order 1. Then tell me the stock now and whether the order went through.",
           [[login("shop")],
            [call("shop_get_stock", "s1", sku="sku-desk")],
            [call("shop_place_order", "o", when="s1[stock] > 0", sku="sku-desk", quantity=1)],
            [call("shop_get_stock", "s2", sku="sku-desk")]],
           "stock before {s1[stock]}, stock now {s2[stock]}, order error: {o[error]}"),
      turn("Check the sku-desk stock again and list my orders.",
           [[call("shop_get_stock", "s", sku="sku-desk"), call("shop_list_orders", "o")]],
           "desk stock {s[stock]}"),
      turn("The shop restocked. How many sku-desk now? Order 1 if there is any.",
           [[call("shop_get_stock", "s", sku="sku-desk")],
            [call("shop_place_order", when="s[stock] > 0", sku="sku-desk", quantity=1)],
            [call("shop_get_stock", "s2", sku="sku-desk")]],
           "desk stock {s2[stock]}")],
     state={"shop.items.sku-desk.stock": 3},
     writes={"shop_place_order": 1},
     answers=[says(0, 3, 0), says(1, 0), says(2, 3)],
     events=[after("shop_get_stock", 1, "set_stock", sku="sku-desk", stock=0),
             stock(2, "sku-desk", 4)])

task("time-midturn-contract-drift", "timing", "契约说只随自己的写变化，轮内却来了一笔外部存款",
     "同一轮里读两次余额，中间有一笔外部存款。契约把余额声明为只随自己的写变化。"
     "考验：对抗性场景，照契约缓存必然答错。",
     [turn(f"{CRED} Sign in to the bank. Tell me the checking balance, then the savings "
           "balance, then read the checking balance again: a deposit may have just arrived.",
           [[login("bank")], [balance("checking", "c1")], [balance("savings", "s")],
            [balance("checking", "c2")]],
           "first {c1[balance]}, savings {s[balance]}, again {c2[balance]}"),
      turn("What is the checking balance now?", [[balance("checking", "c")]],
           "checking {c[balance]}"),
      turn("Move 50 from checking to savings and give me both balances.",
           [[transfer("checking", "savings", 50)],
            [balance("checking", "c"), balance("savings", "s")]],
           "checking {c[balance]}, savings {s[balance]}")],
     state={"bank.accounts.checking.balance": 650},
     writes={"bank_transfer": 1},
     answers=[says(0, 500, 1200, 700), says(1, 700), says(2, 650, 1250)],
     events=[after("bank_get_balance", 1, "deposit", account="checking", amount=200)],
     adversarial=True)

task("time-token-expires-mid-task", "timing", "登录凭据按时间到期，不靠事件作废",
     "登录有效期只有 60 秒虚拟时间（约 6 个决策步）。任务比这长，后半段还要转账。"
     "考验：凭据的真实到期。Host 记住的凭据必须按时间失效，到期后不能再拿旧凭据去请求。",
     [turn(f"{CRED} Sign in to the bank. Tell me the checking, savings, travel and bills "
           "balances, one request at a time.",
           [[login("bank")], [balance("checking", "c")], [balance("savings", "s")],
            [balance("travel", "t")], [balance("bills", "b")]],
           "checking {c[balance]}, savings {s[balance]}, travel {t[balance]}, bills {b[balance]}"),
      turn("Sign in to the bank again if you need to. Move 30 from checking to bills, then "
           "tell me both balances.",
           [[transfer("checking", "bills", 30)],
            [balance("checking", "c"), balance("bills", "b")]],
           "checking {c[balance]}, bills {b[balance]}"),
      turn("Sign in to the bank (same credentials) and tell me the travel balance.",
           [[login("bank")], [balance("travel", "t")]], "travel {t[balance]}"),
      turn("And the savings balance?", [[balance("savings", "s")]], "savings {s[balance]}")],
     state={"bank.accounts.checking.balance": 470, "bank.accounts.bills.balance": 110},
     writes={"bank_transfer": 1},
     answers=[says(0, 500, 1200, 300, 80), says(1, 470, 110), says(2, 300), says(3, 1200)],
     session_ttl_s=60)

task("time-think-time-between-turns", "timing", "用户两轮之间隔了十分钟",
     "每两轮之间有 600 秒的间隔。登录有效期 900 秒，所以第三轮时已经过期；灯的价格在第 400 秒被调高。"
     "考验：按真实时间流逝的过期。按步数计的有效期在这里不会走动，按墙钟计的会。",
     [turn(f"{CRED} Sign in to the bank and the shop. Tell me the checking balance, the "
           "sku-lamp price and the EUR rate.",
           [[login("bank"), login("shop")],
            [balance("checking", "c"), call("shop_get_price", "p", sku="sku-lamp"),
             call("ref_get_rate", "r", currency="EUR")]],
           "checking {c[balance]}, lamp {p[price]}, EUR {r[rate]}"),
      turn("Ten minutes later: what is the sku-lamp price, and the checking balance?",
           [[call("shop_get_price", "p", sku="sku-lamp"), balance("checking", "c")]],
           "lamp {p[price]}, checking {c[balance]}"),
      turn("Another ten minutes later: order 1 sku-lamp if it costs 50 or less, and tell me "
           "the price, the checking balance and the EUR rate. Sign in again if needed.",
           [[call("shop_get_price", "p", sku="sku-lamp"), balance("checking", "c"),
             call("ref_get_rate", "r", currency="EUR")],
            [call("shop_place_order", when="p[price] <= 50", sku="sku-lamp", quantity=1)]],
           "lamp {p[price]}, checking {c[balance]}, EUR {r[rate]}")],
     state={"shop.items.sku-lamp.stock": 7},
     writes={"shop_place_order": 1},
     answers=[says(0, 500, 40, 0.9), says(1, 46, 500), says(2, 46, 500, 0.9)],
     events=[at_time(400, "set_price", sku="sku-lamp", price=46)],
     turn_gap_s=600, session_ttl_s=900)

# ======================================================================  M4
# Writes whose outcome the host cannot see. The reference solution is a careful
# agent: after an unclear reply it looks at the state and repeats the write only
# if it did not take effect.
def careful_transfer(source, target, amount, before, tries=1):
    """Transfer, then up to ``tries`` rounds of "check the source balance and
    repeat only if it is unchanged"."""
    steps = [[transfer(source, target, amount)]]
    for _ in range(tries):
        steps.append([balance(source, "chk", if_error="tr")])
        steps.append([transfer(source, target, amount, when=f"chk[balance] == {before}",
                               if_error="tr")])
    return steps


def fault(tool, kind, *calls, **extra):
    return {"tool": tool, "kind": kind, "on_calls": list(calls), **extra}


ONCE = ("If a transfer or an order reports a problem, check whether it took effect before "
        "trying again: it must end up applied exactly once.")

task("m4-lost-request-and-lost-reply", "m4_operations", "请求丢了和回复丢了，Host 看到的一模一样",
     "两次转账：第一次请求在到达服务器之前丢失（没生效），第二次生效了但回复丢失。"
     "Host 两次看到的都是“没有回复”。考验：两种情况都要恰好生效一次。",
     [turn(f"{CRED} Sign in to the bank. Move 100 from checking to rent. {ONCE} Then tell me "
           "the checking balance.",
           [[login("bank")], *careful_transfer("checking", "rent", 100, 500),
            [balance("checking", "c")]], "checking {c[balance]}"),
      turn(f"Now move 50 from savings to travel. {ONCE} Then tell me the savings and travel "
           "balances.",
           [*careful_transfer("savings", "travel", 50, 1200),
            [balance("savings", "s"), balance("travel", "t")]],
           "savings {s[balance]}, travel {t[balance]}"),
      turn("Give me the checking and rent balances once more.",
           [[balance("checking", "c"), balance("rent", "r")]],
           "checking {c[balance]}, rent {r[balance]}")],
     state={"bank.accounts.checking.balance": 400, "bank.accounts.rent.balance": 100,
            "bank.accounts.savings.balance": 1150, "bank.accounts.travel.balance": 350},
     writes={"bank_transfer": 2},
     answers=[says(0, 400), says(1, 1150, 350), says(2, 400, 100)],
     faults=[fault("bank_transfer", "lost_request", 1), fault("bank_transfer", "lost_ack", 3)])

ORDER_CHECK = [[call("shop_place_order", "o", sku="sku-pen", quantity=5)],
               [call("shop_get_stock", "chk", sku="sku-pen", if_error="o")]]
task("m4-order-without-idempotency", "m4_operations", "下单不支持幂等键，回复丢了不能直接重发",
     "下单工具不支持幂等键。第一轮下单生效但回复丢失；第二轮下单请求在途中丢失。"
     "考验：没有幂等键时不能靠重发来恢复，必须先确认状态。",
     [turn(f"{CRED} Sign in to the shop and order 5 sku-pen. {ONCE} Then tell me the sku-pen "
           "stock.",
           [[login("shop")], *ORDER_CHECK,
            [call("shop_place_order", "o", when="chk[stock] == 100", if_error="o",
                  sku="sku-pen", quantity=5)],
            [call("shop_get_stock", "s", sku="sku-pen")]], "pen stock {s[stock]}"),
      turn(f"Order 5 more sku-pen. {ONCE} Then tell me the sku-pen stock.",
           [*ORDER_CHECK,
            [call("shop_place_order", "o", when="chk[stock] == 95", if_error="o",
                  sku="sku-pen", quantity=5)],
            [call("shop_get_stock", "s", sku="sku-pen")]], "pen stock {s[stock]}"),
      turn("How many sku-pen are left?", [[call("shop_get_stock", "s", sku="sku-pen")]],
           "pen stock {s[stock]}")],
     state={"shop.items.sku-pen.stock": 90},
     writes={"shop_place_order": 2},
     answers=[says(0, 95), says(1, 90), says(2, 90)],
     faults=[fault("shop_place_order", "lost_ack", 1),
             fault("shop_place_order", "lost_request", 2)])

task("m4-status-query-unavailable", "m4_operations", "回复丢了，状态查询也不可用",
     "转账生效但回复丢失，同时按操作键查询状态的接口不可用。"
     "考验：对账手段失效时不能把写当成未执行，要从业务状态确认。",
     [turn(f"{CRED} Sign in to the bank. Move 120 from checking to bills. {ONCE} Then tell me "
           "the checking and bills balances.",
           [[login("bank")], *careful_transfer("checking", "bills", 120, 500),
            [balance("checking", "c"), balance("bills", "b")]],
           "checking {c[balance]}, bills {b[balance]}"),
      turn("What is the checking balance now?", [[balance("checking", "c")]],
           "checking {c[balance]}"),
      turn("Move 20 from checking to bills and give me both balances.",
           [[transfer("checking", "bills", 20)],
            [balance("checking", "c"), balance("bills", "b")]],
           "checking {c[balance]}, bills {b[balance]}")],
     state={"bank.accounts.checking.balance": 360, "bank.accounts.bills.balance": 220},
     writes={"bank_transfer": 2},
     answers=[says(0, 380, 200), says(1, 380), says(2, 360, 220)],
     faults=[fault("bank_transfer", "lost_ack", 1),
             fault("op_status", "unavailable", 1, 2, 3, 4)])

task("m4-consecutive-faults", "m4_operations", "同一次转账连续遇到限流、不可用和回复丢失",
     "一次转账先被限流，再遇到服务不可用，最后一次生效但回复丢失。"
     "考验：多种故障叠加时，重试必须安全，转账恰好生效一次。",
     [turn(f"{CRED} Sign in to the bank. Move 75 from savings to rent, retrying if the service "
           f"is busy. {ONCE} Then tell me the savings and rent balances.",
           [[login("bank")], *careful_transfer("savings", "rent", 75, 1200, tries=3),
            [balance("savings", "s"), balance("rent", "r")]],
           "savings {s[balance]}, rent {r[balance]}"),
      turn("Check the savings balance again.", [[balance("savings", "s")]],
           "savings {s[balance]}"),
      turn("And the rent balance?", [[balance("rent", "r")]], "rent {r[balance]}")],
     state={"bank.accounts.savings.balance": 1125, "bank.accounts.rent.balance": 75},
     writes={"bank_transfer": 1},
     answers=[says(0, 1125, 75), says(1, 1125), says(2, 75)],
     faults=[fault("bank_transfer", "rate_limit", 1), fault("bank_transfer", "unavailable", 2),
             fault("bank_transfer", "lost_ack", 3)])

task("m4-unknown-then-session-dropped", "m4_operations", "写的结果未知，紧接着会话被作废",
     "转账生效但回复丢失，随后银行会话立刻被服务器作废。要确认结果就得先重新登录。"
     "考验：结果未知的写不能因为授权变化而被当成失败重发。",
     [turn(f"{CRED} Sign in to the bank. Move 90 from checking to travel. {ONCE} Sign in "
           "again if the bank asks for it. Then tell me the checking and travel balances.",
           [[login("bank")], *careful_transfer("checking", "travel", 90, 500),
            [balance("checking", "c"), balance("travel", "t")]],
           "checking {c[balance]}, travel {t[balance]}"),
      turn("Move 10 from checking to travel as well and give me both balances.",
           [[transfer("checking", "travel", 10)],
            [balance("checking", "c"), balance("travel", "t")]],
           "checking {c[balance]}, travel {t[balance]}"),
      turn("What is the travel balance?", [[balance("travel", "t")]], "travel {t[balance]}")],
     state={"bank.accounts.checking.balance": 400, "bank.accounts.travel.balance": 400},
     writes={"bank_transfer": 2},
     answers=[says(0, 410, 390), says(1, 400, 400), says(2, 400)],
     faults=[fault("bank_transfer", "lost_ack", 1)],
     events=[after("bank_transfer", 1, "expire_session", app="bank")])


def main():
    path = DATA / "lab-tasks.json"
    existing = json.loads(path.read_text(encoding="utf-8"))
    ours = {t["id"] for t in TASKS}
    from data.lab_expect import enrich
    # Every task, hand-written or generated, gets the oracle derived from its
    # reference solution (final state, committed writes, entity of each value).
    merged = [enrich(t) for t in [t for t in existing if t["id"] not in ours] + TASKS]
    path.write_text(json.dumps(merged, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    cases = [{"case_id": t["id"], "dataset": "lab", "task_id": t["id"], "scenario": t["family"],
              "split": "dev", "augmented": False, "augmentation": None} for t in TASKS]
    (DATA / "lab-stress-cases.json").write_text(json.dumps({
        "version": "lab-stress-cases-v0", "suite": "lab",
        "note": "M1、M2、M3、M4、时间点五类极端场景加一个复合场景，每个场景 5 条。全部为开发材料。"
                "由 data/lab_stress.py 生成，不要手工修改。",
        "cases": cases}, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {len(TASKS)} stress tasks ({len(merged)} lab tasks in total)")


if __name__ == "__main__":
    main()
