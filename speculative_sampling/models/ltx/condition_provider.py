"""Thin restart-aware remote-condition adapter."""
import time

from speculative_sampling.runtime.rpc import RemoteCallError, RemoteWorkerRestarted, TensorClient


class LtxConditionProvider:
    def __init__(self, endpoint, token, *, coordinator=None, recovery_timeout=600):
        self.client = TensorClient(endpoint, token)
        # Establish the original process epoch before the first encode.  This
        # makes a bootstrap text failure observable as a typed restart instead
        # of silently treating the replacement as the first worker ever seen.
        health = self.client.call("health")
        if health.get("role") != "text" or health.get("ready") is not True:
            raise RuntimeError("text service is not ready")
        self.coordinator = coordinator
        self.recovery_timeout = float(recovery_timeout)

    def _session(self):
        if self.coordinator is None:
            return None
        with self.coordinator.lock:
            return self.coordinator.session

    def _begin(self, session, message):
        if session is not None:
            with self.coordinator.lock:
                if self.coordinator.session is session:
                    session.begin_recovery("text", message)

    def _finish(self, session):
        if session is not None:
            with self.coordinator.lock:
                if self.coordinator.session is session:
                    session.finish_recovery("text", {
                        "recovered": True, "epoch": self.client.epoch})

    def encode_condition(self, value, version):
        session = self._session()
        deadline = time.monotonic() + self.recovery_timeout
        recovering = False
        while True:
            try:
                encoded = self.client.call("encode_handle", prompt=value)
                if recovering:
                    self._finish(session)
                return {
                    "owner": "ltx_text", "epoch": self.client.epoch,
                    "key": encoded["key"], "fingerprint": encoded["fingerprint"],
                    "condition_version": version,
                }
            except RemoteWorkerRestarted as restart:
                if not recovering:
                    self._begin(session, str(restart))
                recovering = True
                self.client.accept_restart(restart)
            except RemoteCallError:
                raise
            except Exception as exc:
                if not recovering:
                    self._begin(session, f"{type(exc).__name__}: {exc}")
                recovering = True
                if time.monotonic() >= deadline:
                    raise TimeoutError("text worker did not recover before deadline") from exc
                time.sleep(.5)
