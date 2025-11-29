"""Conversation memory backed by Supabase."""
from __future__ import annotations

from typing import List
from threading import RLock

from backend.database.clients import db_clients

_local_memory_store: dict[str, List[dict[str, str]]] = {}
_store_lock = RLock()


class ConversationMemory:
    def __init__(self, window: int = 6) -> None:
        self.window = window

    def fetch(self, session_id: str) -> List[dict]:
        try:
            response = (
                db_clients.supabase.table("messages")
                .select("role, content")
                .eq("conversation_id", session_id)
                .order("created_at", desc=True)
                .limit(self.window)
                .execute()
            )
            return list(reversed(response.data))
        except Exception:
            return self._fetch_local(session_id)

    def append(self, session_id: str, role: str, content: str) -> None:
        payload = {
            "conversation_id": session_id,
            "role": role,
            "content": content,
        }
        try:
            self._ensure_conversation(session_id)
            db_clients.supabase.table("messages").insert(payload).execute()
        except Exception:
            self._append_local(session_id, payload)

    def _ensure_conversation(self, session_id: str) -> None:
        try:
            db_clients.supabase.table("conversations").upsert({"id": session_id}).execute()
        except Exception:
            # Conversation table missing or Supabase unavailable; local store needs no setup.
            pass

    def _fetch_local(self, session_id: str) -> List[dict]:
        with _store_lock:
            history = _local_memory_store.get(session_id, [])
            if not history:
                return []
            return history[-self.window :]

    def _append_local(self, session_id: str, payload: dict[str, str]) -> None:
        with _store_lock:
            entries = _local_memory_store.setdefault(session_id, [])
            entries.append({"role": payload["role"], "content": payload["content"]})
