"""Web search integration for Google Custom Search and Bing."""
from __future__ import annotations

import httpx
from pydantic import BaseModel

from backend.app.config import settings


class SearchResult(BaseModel):
    title: str
    link: str
    snippet: str
    source: str


class SearchTool:
    def __init__(self) -> None:
        self._client = httpx.AsyncClient(timeout=10)

    async def search(self, query: str) -> list[SearchResult]:
        google_results = await self._google_search(query)
        bing_results = await self._bing_search(query)
        return google_results + bing_results

    async def _google_search(self, query: str) -> list[SearchResult]:
        if not settings.google_search_api_key or not settings.google_search_cx:
            return []
        
        # IMPROVEMENT: Boost trusted sites
        trusted_sites = "site:partselect.com OR site:repairclinic.com OR site:searspartsdirect.com"
        augmented_query = f"{query} ({trusted_sites})"
        
        params = {
            "key": settings.google_search_api_key,
            "cx": settings.google_search_cx,
            "q": augmented_query, # Use the new query
        }
        resp = await self._client.get(
            "https://www.googleapis.com/customsearch/v1", params=params
        )
        resp.raise_for_status()
        payload = resp.json()
        items = payload.get("items", [])
        return [
            SearchResult(
                title=item.get("title", ""),
                link=item.get("link", ""),
                snippet=item.get("snippet", ""),
                source="google",
            )
            for item in items
        ]

    async def _bing_search(self, query: str) -> list[SearchResult]:
        if not settings.bing_search_api_key:
            return []
        headers = {"Ocp-Apim-Subscription-Key": settings.bing_search_api_key}
        params = {"q": query, "mkt": "en-US"}
        resp = await self._client.get(
            "https://api.bing.microsoft.com/v7.0/search", params=params, headers=headers
        )
        resp.raise_for_status()
        payload = resp.json()
        web_pages = payload.get("webPages", {}).get("value", [])
        return [
            SearchResult(
                title=item.get("name", ""),
                link=item.get("url", ""),
                snippet=item.get("snippet", ""),
                source="bing",
            )
            for item in web_pages
        ]
