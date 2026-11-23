"""Rebuild the selected AppWorld tasks on the lab server layer.

The reference calls are the benchmark's own recorded solution trace
(``ground_truth/api_calls.json``): every HTTP call becomes one tool call, the
in-band logins become server-layer sign-ins at the start, and consecutive
reads of the same API are sent in batches, as an agent that already knows the
ids would.
"""
import json

from data.recon.common import SIGN_IN
from data.recon.overlays import overlays
from exp_core.lab.domains.appworld import SEP, AppWorldDomain, route
from exp_core.settings import APPWORLD_ROOT, DATA

BATCH = 8
PREFIX = ("Your app passwords are listed by supervisor__show_account_passwords; your own "
          "name, e-mail address and phone number by supervisor__show_profile. " + SIGN_IN)
NON_IDEMPOTENT = {"venmo__create_transaction", "venmo__create_payment_request",
                  "venmo__create_transaction_comment"}
KEYED = {"venmo__create_transaction", "venmo__create_payment_request"}
# The read that shows whether an unkeyed write was applied.
VERIFY = {"venmo__create_transaction_comment": lambda arguments: {
    "name": "venmo__show_transaction",
    "arguments": {"transaction_id": arguments["transaction_id"]}}}


def reference_calls(task_id):
    path = APPWORLD_ROOT / "data" / "tasks" / task_id / "ground_truth"
    recorded = json.loads((path / "api_calls.json").read_text())
    logins, calls = [], []
    for record in recorded:
        app = record["url"].split("/")[1]
        api, path_arguments = route(record["method"], record["url"])
        data = {k: v for k, v in record["data"].items() if k != "access_token"}
        if api == "login":
            logins.append({"name": "auth_login", "arguments": {"app": app, **data}})
        else:
            calls.append({"name": app + SEP + api, "arguments": {**path_arguments, **data}})
    return logins, calls, json.loads((path / "required_apps.json").read_text()), \
        json.loads((path / "answer.json").read_text())


def base_task(entry):
    logins, calls, apps, answer = reference_calls(entry["id"])
    task = {"id": "recon-appworld-" + entry["id"], "domain": "appworld",
            "source": {"dataset": "appworld", "task_id": entry["id"]}, "apps": apps,
            "initial": {}}
    domain = AppWorldDomain(task)
    # Sign in to every app the task needs, also where the recorded solution
    # only used public APIs: the added turns read the user's own data.
    passwords = {row["account_name"]: row["password"] for row in
                 domain.call("supervisor__show_account_passwords", {}, shadow=True)}
    supervisor = entry["spec"]["supervisor"]
    logins = [{"name": "auth_login", "arguments": {
        "app": app, "password": passwords[app],
        "username": supervisor["phone_number" if app == "phone" else "email"]}}
        for app in sorted(apps)]
    # Who the user is and the passwords first, then every sign-in, then the work.
    opening = [{"name": "supervisor__show_profile", "arguments": {}},
               {"name": "supervisor__show_account_passwords", "arguments": {}}]
    rest = [c for c in calls if c not in opening]
    steps = [[c] for c in opening] + [logins]
    for call in rest:
        last = steps[-1]
        if (domain.tools[call["name"]]["kind"] == "read" and len(steps) > len(opening) + 1
                and last[0]["name"] == call["name"] and len(last) < BATCH
                and call not in last):
            last.append(call)
        else:
            steps.append([call])
    task["turns"] = [PREFIX + entry["spec"]["instruction"]]
    task["gold"] = [{"steps": steps, "answer": "done" if answer is None else str(answer)}]
    task["prelude"] = len(opening) + 1
    task["email"] = entry["spec"]["supervisor"]["email"]
    return task, domain


def probes(task, domain):
    out, email = [], task["email"]

    def probe(name, arguments, field, label, ask, fresh, event=None):
        out.append({"call": {"name": name, "arguments": arguments}, "field": field,
                    "label": label, "ask": ask, "fresh": fresh,
                    "app": name.split(SEP)[0] if domain.tools[name]["auth"] else None,
                    "event": event})

    if "venmo" in task["apps"]:
        probe("venmo__show_venmo_balance", {}, ["venmo_balance"], "balance",
              "my Venmo balance", "slow",
              # A friend pays me.
              {"kind": "add", "app": "venmo", "model": "User", "where": {"email": email},
               "field": "venmo_balance", "value": 25.0})
    if "spotify" in task["apps"]:
        probe("spotify__show_volume", {}, ["volume"], "volume",
              "the volume of my Spotify player", "volatile",
              # I change the volume on another device.
              {"kind": "set", "app": "spotify", "model": "MusicPlayer",
               "where": {"user_id": {"of": ["User", {"email": email}, "id"]}},
               "field": "volume", "value": 3, "else": 7})
        song = next((call["arguments"]["song_id"] for turn in task["gold"]
                     for step in turn["steps"] for call in step
                     if call["name"] == "spotify__show_song"), None)
        if song is not None:
            probe("spotify__show_song", {"song_id": song}, ["play_count"], "plays",
                  f"how many times song {song} has been played", "slow",
                  # Other listeners play it.
                  {"kind": "add", "app": "spotify", "model": "Song", "where": {"id": song},
                   "field": "play_count", "value": 5})
    # Without a private read that stays valid for a while, the sign-in
    # scenarios have nothing to ask for: fall back to the account itself.
    for app, field in (("spotify", "email"), ("phone", "phone_number"),
                       ("simple_note", "email"), ("file_system", "email")):
        if app in task["apps"] and not any(p["fresh"] == "slow" and p["app"] for p in out):
            probe(f"{app}__show_account", {}, [field], "account",
                  f"the {field.replace('_', ' ')} of my {app.replace('_', ' ')} account", "slow")
    return out


def build():
    from data.recon.common import reference
    entries = json.loads((DATA / "appworld-tasks.json").read_text())
    manifest = json.loads((DATA / "manifest.json").read_text())["datasets"]["appworld"]
    tasks, cases = [], []
    for entry in entries:
        base, domain = base_task(entry)
        own = reference({**{k: v for k, v in base.items() if k != "prelude"}, "events": []})
        errors = len(own["errors"])  # errors of the original trace itself
        writes = [(t, s, call["name"]) for t, turn in enumerate(base["gold"])
                  for s, step in enumerate(turn["steps"]) for call in step
                  if domain.tools.get(call["name"], {}).get("kind") == "write"]
        split = "dev" if entry["id"] in manifest["dev"] else "heldout"
        built = overlays(base, probes(base, domain), writes, NON_IDEMPOTENT, errors,
                         keyed=KEYED, verify=VERIFY)
        print(f"  {entry['id']}: {len(base['gold'][0]['steps'])} steps, "
              f"{own['logical_calls']} calls, {errors} errors in the original trace, "
              f"{len(built)} cases", flush=True)
        for task in built:
            tasks.append(task)
            cases.append({"case_id": task["id"], "dataset": "lab", "task_id": task["id"],
                          "scenario": task["family"], "split": split,
                          "source_dataset": "appworld", "source_task_id": entry["id"],
                          "augmented": False, "augmentation": None})
    return tasks, cases
