"""Tool routing and structured outputs for the agent."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from backend.app.models import ToolCall
from backend.tools.search import SearchTool
from backend.tools.vision import VisionTool


@dataclass
class ToolRouter:
    search_tool: SearchTool
    vision_tool: VisionTool

    async def route(self, message: str, metadata: Dict[str, Any] | None) -> List[ToolCall]:
        """Choose which tools to call based on message metadata."""

        events: List[ToolCall] = []
        metadata = metadata or {}

        if metadata.get("image_ids"):
            vision_result = await self.vision_tool.describe_images(metadata["image_ids"])
            events.append(
                ToolCall(name="vision", input={"image_ids": metadata["image_ids"]}, output=vision_result)
            )

        if metadata.get("requires_search"):
            search_result = await self.search_tool.search(message)
            events.append(ToolCall(name="web_search", input={"query": message}, output=search_result))

        return events
