"""Local worker-group ownership; never restart with stale session handles."""
import os
import signal
import subprocess
import time


class WorkerGroup:
    def __init__(self):
        self.processes = []
        self.records = []

    def start(self, argv, *, environment, cwd, log, name=None, restartable=False):
        process = subprocess.Popen(
            argv, env=environment, cwd=cwd, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True)
        self.processes.append(process)
        self.records.append({
            "name": name or str(process.pid), "argv": tuple(argv),
            "environment": environment, "cwd": cwd, "log": log,
            "restartable": bool(restartable), "process": process,
            "restarts": 0,
        })
        return process

    def check(self, *, restart=False):
        events = []
        for record in self.records:
            process = record["process"]
            code = process.poll()
            if code is None:
                continue
            if not restart or not record["restartable"]:
                raise RuntimeError(
                    f"worker exited; stopping local deployment: "
                    f"{[(record['name'], process.pid, code)]}")
            # torchrun can leave rank children alive after its parent exits.
            # Clear the complete old process group before reusing its GPUs or
            # listening port for the replacement.
            try:
                os.killpg(process.pid, signal.SIGTERM)
                time.sleep(.1)
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            replacement = subprocess.Popen(
                record["argv"], env=record["environment"], cwd=record["cwd"],
                stdout=record["log"], stderr=subprocess.STDOUT,
                start_new_session=True)
            record["process"] = replacement
            record["restarts"] += 1
            self.processes.append(replacement)
            events.append({"name": record["name"], "previous_pid": process.pid,
                           "returncode": code, "pid": replacement.pid,
                           "restart": record["restarts"]})
        return tuple(events)

    def close(self, *, grace_seconds=15):
        # A torchrun parent can exit while children survive in its process
        # group. Signal the group even if the parent already has an exit code.
        for process in self.processes:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        deadline = time.monotonic() + grace_seconds
        for process in self.processes:
            try:
                process.wait(timeout=max(.01, deadline-time.monotonic()))
            except subprocess.TimeoutExpired:
                pass
        for process in self.processes:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
