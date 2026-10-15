"""Durable coordinator command journal for exact DiT replay after restart."""
from __future__ import annotations

import json
import os
from pathlib import Path

from .recovery_journal import SESSION


TASK_FIELDS = (
    "session", "proposal", "retain_until", "chunk_latents", "prompt",
    "prompt_version", "seed", "reference", "condition_handle",
)


def replay_task(task):
    missing = [name for name in TASK_FIELDS if name not in task]
    if missing:
        raise ValueError(f"generation task missing replay fields: {missing}")
    value = {name: task[name] for name in TASK_FIELDS}
    if (not isinstance(value["session"], str) or not value["session"]
            or type(value["proposal"]) is not int or value["proposal"] < 1
            or type(value["retain_until"]) is not int or value["retain_until"] < 0
            or type(value["chunk_latents"]) is not int or value["chunk_latents"] < 1
            or not isinstance(value["prompt"], str) or not value["prompt"]
            or type(value["prompt_version"]) is not int or value["prompt_version"] < 1
            or type(value["seed"]) is not int
            or not isinstance(value["reference"], str) or not value["reference"]
            or not isinstance(value["condition_handle"], dict)):
        raise ValueError("generation task contains invalid replay fields")
    return value


class GenerationJournal:
    def __init__(self, output):
        self.root = Path(output) / "recovery"
        self.path = None
        self.document = None

    @staticmethod
    def _empty(session):
        return {
            "schema_version": 1,
            "session": session,
            "completed": [],
            "inflight": None,
        }

    def load(self, session):
        if not isinstance(session, str) or not SESSION.fullmatch(session):
            raise ValueError("invalid recovery session identity")
        path = self.root / f"dit-generation-{session}.json"
        if self.path == path and self.document is not None:
            return self.document
        document = self._empty(session)
        self.path = path
        if self.path.is_file():
            loaded = json.loads(self.path.read_text())
            if not isinstance(loaded, dict) or set(loaded) != set(document):
                raise ValueError("invalid DiT generation journal structure")
            if loaded["session"] != session:
                raise ValueError("DiT generation journal session mismatch")
            document = loaded
        self.document = document
        return document

    def _save(self):
        self.root.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(
            self.document, indent=2, sort_keys=True) + "\n")
        os.chmod(temporary, 0o600)
        temporary.replace(self.path)

    def begin(self, task):
        task = replay_task(task)
        self.load(task["session"])
        inflight = self.document["inflight"]
        if inflight is not None:
            if inflight["task"] == task:
                return inflight
            raise RuntimeError("another generation command is already in flight")
        if any(item["task"]["proposal"] == task["proposal"]
               for item in self.document["completed"]):
            raise ValueError("completed proposal cannot begin again")
        self.document["inflight"] = {
            "task": task,
            "media_content_fingerprint": None,
            "media_checkpoint_fingerprint": None,
            "dit_native_state": None,
        }
        self._save()
        return self.document["inflight"]

    def latents_installed(self, task, media_install, media_checkpoint):
        task = replay_task(task)
        if self.document is None or self.document["session"] != task["session"]:
            self.load(task["session"])
        inflight = self.document["inflight"]
        if inflight is None or inflight["task"] != task:
            raise ValueError("latent callback does not match in-flight command")
        content = media_install.get("content_fingerprint")
        checkpoint = media_checkpoint.get("checkpoint_fingerprint")
        if not isinstance(content, str) or len(content) != 64:
            raise ValueError("media install lacks a content fingerprint")
        if not isinstance(checkpoint, str) or len(checkpoint) != 64:
            raise ValueError("media checkpoint lacks a fingerprint")
        previous = (inflight["media_content_fingerprint"],
                    inflight["media_checkpoint_fingerprint"])
        current = (content, checkpoint)
        if previous != (None, None) and previous != current:
            raise ValueError("repeated latent callback changed content identity")
        inflight["media_content_fingerprint"] = content
        inflight["media_checkpoint_fingerprint"] = checkpoint
        self._save()

    def complete(self, task, dit_native_state=None):
        task = replay_task(task)
        if self.document is None or self.document["session"] != task["session"]:
            self.load(task["session"])
        inflight = self.document["inflight"]
        if inflight is None or inflight["task"] != task:
            raise ValueError("completed command does not match in-flight journal")
        if not inflight["media_content_fingerprint"]:
            raise ValueError("cannot complete command before latent installation")
        if dit_native_state is not None:
            if (not isinstance(dit_native_state, list) or not dit_native_state
                    or any(not isinstance(item, dict)
                           or not isinstance(item.get("sha256"), str)
                           or len(item["sha256"]) != 64
                           for item in dit_native_state)):
                raise ValueError("invalid per-rank DiT native state fingerprint")
            inflight["dit_native_state"] = dit_native_state
        expected = len(self.document["completed"]) + 1
        if task["proposal"] != expected:
            raise ValueError("generation journal proposals must be contiguous")
        self.document["completed"].append(inflight)
        self.document["inflight"] = None
        self._save()
        return inflight

    def replay_records(self, session=None):
        if session is not None:
            self.load(session)
        if self.document is None:
            return ()
        return tuple(item["task"] for item in self.document["completed"])
