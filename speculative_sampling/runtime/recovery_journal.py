"""Atomic local journals for worker state; files contain no service tokens."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re


SESSION = re.compile(r"^[A-Za-z0-9_-]{1,128}$")


class MediaCheckpointJournal:
    def __init__(self, output):
        self.root = Path(output) / "recovery"

    def _paths(self, session):
        if not isinstance(session, str) or not SESSION.fullmatch(session):
            raise ValueError("invalid recovery session identity")
        return (self.root / f"media-{session}.pt",
                self.root / f"media-{session}.json")

    def save(self, checkpoint):
        import torch
        session = checkpoint.get("session")
        state_path, manifest_path = self._paths(session)
        self.root.mkdir(parents=True, exist_ok=True)
        state_temp = state_path.with_suffix(".pt.tmp")
        manifest_temp = manifest_path.with_suffix(".json.tmp")
        torch.save(checkpoint, state_temp)
        os.chmod(state_temp, 0o600)
        manifest = {
            "schema_version": 1,
            "session": session,
            "checkpoint_fingerprint": checkpoint["checkpoint_fingerprint"],
            "official_until": checkpoint["official_until"],
            "proposal_ids": [item["proposal_id"] for item in checkpoint["proposals"]],
            "state_file": state_path.name,
        }
        manifest_temp.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        os.chmod(manifest_temp, 0o600)
        state_temp.replace(state_path)
        manifest_temp.replace(manifest_path)
        return manifest

    def load(self, session):
        import torch
        state_path, manifest_path = self._paths(session)
        manifest = json.loads(manifest_path.read_text())
        checkpoint = torch.load(state_path, map_location="cpu", weights_only=True)
        if (manifest.get("session") != session
                or manifest.get("checkpoint_fingerprint")
                != checkpoint.get("checkpoint_fingerprint")):
            raise ValueError("media recovery journal manifest mismatch")
        return checkpoint
