"""Model routing for Groq (Llama 70B/8B) and Qwen extraction."""
from __future__ import annotations

from dataclasses import dataclass
from langchain_core.language_models import BaseLanguageModel

from backend.app.models import ChatRequest


@dataclass
class ModelRouter:
    primary: BaseLanguageModel
    fast: BaseLanguageModel
    extractor: BaseLanguageModel

    def pick(self, request: ChatRequest) -> BaseLanguageModel:
        metadata = request.metadata or {}
        if metadata.get("mode") == "extract" or metadata.get("requires_part_numbers"):
            return self.extractor
        if len(request.message) < 320:
            return self.fast
        return self.primary
