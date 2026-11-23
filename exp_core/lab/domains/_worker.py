"""A long-lived helper process in another Python environment.

AppWorld and tau2 need their own interpreters. Starting one costs seconds, a
task's environment milliseconds, so one worker per kind is kept for the life
of this process and re-initialised for every task.
"""
import atexit
import json
import os
import subprocess

from exp_core.lab.world import BusinessError
from exp_core.settings import ROOT

_workers = {}


class Worker:
    def __init__(self, python, module, env):
        self.process = subprocess.Popen(
            [str(python), "-m", module], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, env={**os.environ, **env}, cwd=ROOT)

    def alive(self):
        return self.process.poll() is None

    def request(self, action, **fields):
        self.process.stdin.write(json.dumps({"action": action, **fields}) + "\n")
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError(f"worker exited during {action}")
        response = json.loads(line)
        if response["ok"]:
            return response["value"]
        if response.get("refused"):  # the service said no; the environment is unchanged
            raise BusinessError(response["error"])
        raise RuntimeError(f"worker failed during {action}: {response['error']}")

    def stop(self):
        if self.alive():
            self.process.terminate()


def shared(kind, python, module, env):
    worker = _workers.get(kind)
    if worker is None or not worker.alive():
        worker = _workers[kind] = Worker(python, module, env)
    return worker


@atexit.register
def _stop_all():
    for worker in _workers.values():
        worker.stop()
