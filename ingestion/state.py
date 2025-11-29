"""Resumable ingestion state tracking."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict


@dataclass
class StateTracker:
    path: Path
    state: Dict[str, dict]

    @classmethod
    def load(cls, path: Path) -> "StateTracker":
        if path.exists():
            data = json.loads(path.read_text())
        else:
            data = {}
        return cls(path=path, state=data)

    def mark(self, digest: str, status: str, meta: dict | None = None) -> None:
        self.state[digest] = {"status": status, **(meta or {})}
        self.persist()

    def persist(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.state, indent=2))
