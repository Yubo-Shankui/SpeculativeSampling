"""Coordinator-side proxy for a restartable distributed DiT service."""
from __future__ import annotations

import time
from pathlib import Path

from speculative_sampling.runtime.generation_journal import GenerationJournal
from speculative_sampling.runtime.rpc import RemoteCallError, RemoteWorkerRestarted, TensorClient


class RemoteLtxDitWorker:
    mock = False

    def __init__(self, args, demo):
        self.args = args
        self.demo = demo
        token = Path(args.service_token_file).read_text().strip()
        self.client = TensorClient(args.dit_endpoint, token, timeout=300)
        self.recovery_timeout = float(args.worker_recovery_timeout)
        if self.recovery_timeout <= 0:
            raise ValueError("worker recovery timeout must be positive")

    def warmup(self):
        health = self.client.call("health")
        if health.get("role") != "dit" or health.get("ready") is not True:
            raise RuntimeError("remote DiT service is not ready")

    def _session(self, task):
        with self.demo.lock:
            session = self.demo.session
            return session if session is not None and session.id == task["session"] else None

    def _begin_recovery(self, task, message):
        with self.demo.lock:
            session = self.demo.session
            if session is not None and session.id == task["session"]:
                session.begin_recovery("dit", message)

    def _finish_recovery(self, task, report):
        with self.demo.lock:
            session = self.demo.session
            if session is not None and session.id == task["session"]:
                session.finish_recovery("dit", report)

    def _records(self, session):
        journal = GenerationJournal(self.args.output)
        document = journal.load(session)
        return list(document["completed"])

    def _recover(self, task, restart=None):
        if restart is not None:
            self.client.accept_restart(restart)
        records = self._records(task["session"])
        report = self.client.call("recover", records=records)
        if report.get("recovered") is not True:
            raise RuntimeError("DiT service did not confirm recovery")
        if records and records[-1].get("dit_native_state"):
            expected = records[-1].get("dit_native_state")
            if not expected or report.get("rank_native_state") != expected:
                raise RuntimeError("recovered DiT native state differs from journal")
        self._finish_recovery(task, report)
        return report

    def _wait_available(self, task, deadline):
        while True:
            if time.monotonic() >= deadline:
                raise TimeoutError("DiT worker did not recover before deadline")
            try:
                self.client.call("health")
                return None
            except RemoteWorkerRestarted as restart:
                return self._recover(task, restart)
            except Exception:
                time.sleep(.5)

    def generate(self, task, on_block, prompt_source=None):
        del prompt_source
        deadline = time.monotonic() + self.recovery_timeout
        recovering = False
        recovery_started = None
        while True:
            attempt_started = time.monotonic()
            try:
                result = self.client.call("generate", task=task)
                if recovering:
                    self._finish_recovery(task, {
                        "recovered": True, "same_epoch_retry": True})
                event = result.get("event")
                if on_block is not None and not result.get("early_handoff", False):
                    if event is None:
                        raise RuntimeError("DiT response lacks latent handoff")
                    on_block(event, None)
                stages = dict(result["stages"])
                if result.get("rank_native_state") is not None:
                    stages["dit_native_state"] = result["rank_native_state"]
                if recovery_started is not None:
                    # The successful retry is ordinary production time. Only
                    # exclude the failed call, wait, load and replay interval.
                    stages["worker_recovery_seconds"] = max(
                        0., attempt_started - recovery_started)
                return (None, stages, result["engine_elapsed_seconds"])
            except RemoteWorkerRestarted as restart:
                if recovery_started is None:
                    recovery_started = attempt_started
                recovering = True
                self._begin_recovery(task, str(restart))
                self._recover(task, restart)
            except RemoteCallError:
                raise
            except Exception as exc:
                if recovery_started is None:
                    recovery_started = attempt_started
                recovering = True
                self._begin_recovery(task, f"{type(exc).__name__}: {exc}")
                self._wait_available(task, deadline)

    def close(self):
        pass
