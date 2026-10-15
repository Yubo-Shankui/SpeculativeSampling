"""Coordinator endpoint for early, idempotent LTX latent handoff."""
from __future__ import annotations

import threading
from collections import OrderedDict


class CoordinatorLatentHandoff:
    """Accept installed proposal latents before the DiT generate RPC returns.

    The media tensors already live in the media worker.  This endpoint carries
    only their transaction identity and immediately makes the proposal visible
    to the coordinator decode pump.  Repeated delivery is harmless, which is
    required when an RPC response is lost after the event was applied.
    """

    operations = {"health", "latents_ready"}

    def __init__(self, demo, media_pump, media_journal, generation_journal):
        self.demo = demo
        self.media_pump = media_pump
        self.media_journal = media_journal
        self.generation_journal = generation_journal
        self.lock = threading.Lock()
        self.applied = OrderedDict()

    def health(self):
        return {"ready": True, "role": "ltx-coordinator-handoff"}

    @staticmethod
    def _identity(event):
        if not isinstance(event, dict) or event.get("event") != "latents_ready":
            raise ValueError("expected an LTX latents_ready event")
        task = event.get("task")
        install = event.get("media_install")
        checkpoint = event.get("media_checkpoint")
        if not all(isinstance(value, dict) for value in (task, install, checkpoint)):
            raise ValueError("incomplete LTX latent handoff")
        session = task.get("session")
        proposal = task.get("proposal")
        content = install.get("content_fingerprint")
        checkpoint_fingerprint = checkpoint.get("checkpoint_fingerprint")
        if (not isinstance(session, str) or not session
                or type(proposal) is not int or proposal < 1
                or not isinstance(content, str) or len(content) != 64
                or not isinstance(checkpoint_fingerprint, str)
                or len(checkpoint_fingerprint) != 64):
            raise ValueError("invalid LTX latent handoff identity")
        return ((session, proposal), (content, checkpoint_fingerprint),
                task, install, checkpoint)

    def latents_ready(self, event):
        key, identity, task, install, checkpoint = self._identity(event)
        with self.lock:
            previous = self.applied.get(key)
            if previous is not None:
                if previous != identity:
                    raise ValueError("repeated latent handoff changed content identity")
                self.applied.move_to_end(key)
                return {"accepted": True, "duplicate": True}

            # Persist transaction identity before exposing it to asynchronous
            # decode.  A durable media checkpoint is optional on the NCCL-only
            # development path, but the generation journal is always updated.
            if not checkpoint.get("volatile"):
                self.media_journal.save(checkpoint)
            self.generation_journal.latents_installed(task, install, checkpoint)
            with self.demo.lock:
                session = self.demo.session
                active = (session is not None and session.id == task["session"]
                          and session.latents_ready(task))
            if active:
                self.media_pump.submit(task)
            self.applied[key] = identity
            self.applied.move_to_end(key)
            while len(self.applied) > 512:
                self.applied.popitem(last=False)
            return {"accepted": True, "duplicate": False, "active": bool(active)}
