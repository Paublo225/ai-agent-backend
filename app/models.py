"""Pydantic schemas shared across API layers."""
from __future__ import annotations

from datetime import datetime
from typing import Any, List, Literal, Optional

from pydantic import BaseModel, Field


class DocumentChunk(BaseModel):
    document_id: str
    source: str
    page_number: Optional[int] = None
    appliance_type: Optional[str] = None
    summary: Optional[str] = None
    part_numbers: List[str] = Field(default_factory=list)


class ToolCall(BaseModel):
    name: str
    input: dict
    output: Optional[Any] = None


class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    citations: List[DocumentChunk] = Field(default_factory=list)
    tool_calls: List[ToolCall] = Field(default_factory=list)


class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    message: str
    image_ids: List[str] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)


class ChatResponse(BaseModel):
    session_id: str
    messages: List[ChatMessage]
    latency_ms: float


class ImageUploadResponse(BaseModel):
    image_id: str
    filename: str
    content_type: str


class HealthResponse(BaseModel):
    status: str
    environment: str
    version: str = "0.1.0"
