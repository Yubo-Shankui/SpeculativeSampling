"""Single-flight latest-condition encoding, outside the generator slot.

The coordinator serializes session access with its own lock. Requests arriving
while encoding coalesce to the latest unencoded version. No tensors are owned
by the controller; only a versioned provider handle is installed.
"""
from __future__ import annotations

import threading


class ConditionPump:
    def __init__(self, coordinator, provider, *, interval=.02):
        self.coordinator = coordinator
        self.provider = provider
        self.interval = interval
        self.stopping = threading.Event()
        self.thread = threading.Thread(target=self.run, name="condition-pump", daemon=True)
        self.thread.start()

    def tick(self):
        """Encode at most one version. Kept synchronous for deterministic tests."""
        with self.coordinator.lock:
            session = self.coordinator.session
            if session is None or session.stopped or session.error:
                return False
            controller = session.controller
            version = controller.pending_condition_version
            if version is None or version in controller.condition_ready:
                return False
            value = session.prompts[version]
        try:
            handle = self.provider.encode_condition(value, version)
        except Exception as exc:
            with self.coordinator.lock:
                # A superseded encoding failure must not kill the newer input.
                current = self.coordinator.session
                if current is session and not current.stopped:
                    if controller.pending_condition_version == version:
                        current.fail(f"condition worker: {type(exc).__name__}: {exc}")
                    else:
                        current.event("condition_superseded_failure", prompt_version=version,
                                      message=str(exc))
            return True
        with self.coordinator.lock:
            current = self.coordinator.session
            if current is session and not session.stopped and not session.error:
                if controller.pending_condition_version == version:
                    mark_ready = getattr(session, "mark_condition_ready", None)
                    if mark_ready is not None:
                        mark_ready(version, handle)
                    else:
                        controller.mark_condition_ready(version, handle)
                        import time
                        if hasattr(session, "condition_ready_at"):
                            session.condition_ready_at[version] = time.monotonic()
                        session.event("condition_ready", prompt_version=version)
                else:
                    session.event("condition_superseded", prompt_version=version)
        return True

    def run(self):
        while not self.stopping.wait(self.interval):
            self.tick()

    def close(self):
        self.stopping.set()
        self.thread.join(timeout=5)
