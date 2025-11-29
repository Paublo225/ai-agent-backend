"""Vision tool for image understanding via Gemini 2.5 Flash."""
from __future__ import annotations

from typing import List

import httpx

from backend.app.config import settings


class VisionTool:
    def __init__(self) -> None:
        self._client = httpx.AsyncClient(timeout=30)
        self._api_url = "https://generativelanguage.googleapis.com/v1beta/models"
        self._model = "gemini-2.5-flash"

    async def describe_images(self, image_ids: List[str]) -> List[dict]:
        if not settings.gemini_api_key:
            return []
        responses: List[dict] = []
        for image_id in image_ids:
            resp = await self._client.post(
                f"{self._api_url}/{self._model}:generateContent",
                params={"key": settings.gemini_api_key},
                json={
                    "contents": [
                        {
                            "parts": [
                                {"text": "Extract appliance model, part numbers, and labels."},
                                {"fileData": {"mimeType": "image/png", "fileUri": image_id}},
                            ]
                        }
                    ]
                },
            )
            resp.raise_for_status()
            responses.append(resp.json())
        return responses
