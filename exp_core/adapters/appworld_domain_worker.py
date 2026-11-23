"""AppWorld environment for the lab server layer (runs in the AppWorld venv).

One process serves many tasks; ``init`` replaces the world. Nothing is saved
per call. The state reported to the oracle is one digest per database table
of the apps in use, so a difference names the table that differs.
"""
import hashlib
import json
import os
import shutil
import sys
from contextlib import redirect_stdout

with redirect_stdout(sys.stderr):
    from appworld import AppWorld
    from sqlmodel import Session

EXPERIMENT = f"recon-{os.getpid()}"
world, apps = None, []


def table_digest(model):
    table = model.__tablename__
    digest, rows = hashlib.sha256(), 0
    with Session(model.db.engine) as session:
        connection = session.connection()
        for row in connection.exec_driver_sql(f'SELECT * FROM "{table}" ORDER BY 1'):
            digest.update(repr(tuple(row)).encode())
            rows += 1
    return f"{rows} rows {digest.hexdigest()[:16]}"


def models_of(app):
    return {name: model for name, model in sorted(getattr(world.models, app).items())
            if name != "SQLModel"}


def handle(command):
    global world, apps
    action = command["action"]
    if action == "init":
        if world is not None:
            world.close()
        world = AppWorld(command["id"], experiment_name=EXPERIMENT, random_seed=20261116)
        apps = command["apps"]
        return {"instruction": world.task.instruction,
                "supervisor": dict(world.task.supervisor)}
    if action == "call":
        return world.requester.request(_app_name=command["app"], _api_name=command["api"],
                                       **command["arguments"])
    if action == "event":
        event = command["event"]
        model = models_of(event["app"])[event["model"]]
        # A condition {"of": [Model, where, field]} is looked up first, e.g.
        # the player of the user with a given e-mail address.
        where = {key: (getattr(models_of(event["app"])[value["of"][0]].find_one(
            **value["of"][1]), value["of"][2]) if isinstance(value, dict) else value)
            for key, value in event["where"].items()}
        record = model.find_one(**where)
        current = getattr(record, event["field"])
        if event["kind"] == "add":
            new = current + event["value"]
        else:  # "else" is used when the value is already the one to set
            new = event["else"] if current == event["value"] and "else" in event \
                else event["value"]
        setattr(record, event["field"], new)
        record.save()
        return {}
    if action == "state":
        return {app: {name: table_digest(model) for name, model in models_of(app).items()}
                for app in apps}
    if action == "models":
        return {app: {name: sorted(model.schema()["properties"])
                      for name, model in models_of(app).items()} for app in apps}
    raise ValueError(f"unknown action {action}")


try:
    for line in sys.stdin:
        command = json.loads(line)
        try:
            with redirect_stdout(sys.stderr):
                value = handle(command)
            print(json.dumps({"ok": True, "value": value}, default=str), flush=True)
        except Exception as error:  # noqa: BLE001 - reported to the caller
            print(json.dumps({"ok": False, "refused": command.get("action") == "call",
                              "error": str(error).replace("\n", " ")[:500]}), flush=True)
finally:
    if world is not None:
        world.close()
    root = os.environ.get("APPWORLD_ROOT")
    if root:
        shutil.rmtree(os.path.join(root, "experiments", "outputs", EXPERIMENT),
                      ignore_errors=True)
