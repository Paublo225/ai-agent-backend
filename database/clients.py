"""Database client initializers for Supabase and Pinecone."""
from __future__ import annotations

from typing import Any

from pinecone import Pinecone
from supabase import Client, create_client

from backend.app.config import settings


class DatabaseClients:
    def __init__(self) -> None:
        self._supabase: Client | None = None
        self._pinecone: Pinecone | None = None

    @property
    def supabase(self) -> Client:
        if self._supabase is None:
            if not settings.supabase_url or not settings.supabase_key:
                raise RuntimeError("Supabase credentials are missing")
            self._supabase = create_client(settings.supabase_url, settings.supabase_key)
        return self._supabase

    @property
    def pinecone(self) -> Any:
        if self._pinecone is None:
            if not settings.pinecone_api_key:
                raise RuntimeError("Pinecone API key missing")
            self._pinecone = Pinecone(api_key=settings.pinecone_api_key)
        return self._pinecone


db_clients = DatabaseClients()
