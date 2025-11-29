"""Agent orchestration layer built on LangChain."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from textwrap import dedent
from typing import Any, List
from urllib.parse import urlparse

from langchain.prompts import ChatPromptTemplate
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from backend.agent.memory import ConversationMemory
from backend.agent.model_router import ModelRouter
from backend.agent.prompts import RESPONSE_TEMPLATE, SYSTEM_PROMPT
from backend.app.models import ChatMessage, ChatRequest
from backend.rag.pipeline import RetrievalPipeline
from backend.tools.router import ToolRouter


@dataclass
class AgentContext:
    model_router: ModelRouter
    retrieval_pipeline: RetrievalPipeline | None
    tools: ToolRouter
    memory: ConversationMemory


class AgentOrchestrator:
    """Coordinates conversations, retrieval, tools, and model calls."""

    def __init__(self, context: AgentContext) -> None:
        self.context = context
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT),
                ("human", RESPONSE_TEMPLATE),
            ]
        )

    async def run(self, request: ChatRequest) -> ChatMessage:
        session_id = request.session_id or "temp"
        history = self.context.memory.fetch(session_id)
        history_str = self._format_history(history)

        metadata = dict(request.metadata or {})

        # 1. Start Retrieval
        if self.context.retrieval_pipeline:
            retrieved_chunks = await self.context.retrieval_pipeline.retrieve(query=request.message)
        else:
            retrieved_chunks = []

        # 2. Check if we need web search (Fallback Logic)
        should_search = metadata.get("requires_search", False)
        
        # FALLBACK: If no manuals found, force web search
        if not retrieved_chunks: 
            should_search = True
        elif await self._should_search(request.message): # Or if the question inherently needs web (e.g. "price")
            should_search = True

        # 3. Execute Search if needed
        tool_events = []
        if should_search:
            # Route to search tool
            tool_events = await self.context.tools.route(message=request.message, metadata={"requires_search": True})

        # 4. Format Context (Combine Manuals + Web)
        context_block = self._format_context(retrieved_chunks, tool_events)

        prompt_messages = self.prompt.format_messages(
            context=context_block,
            history=history_str,
            question=request.message,
        )

        model = self.context.model_router.pick(request)
        response = await model.ainvoke(prompt_messages)

        message = ChatMessage(
            role="assistant",
            content=response.content,
            citations=[chunk.to_model() for chunk in retrieved_chunks],
            tool_calls=tool_events,
        )

        self.context.memory.append(session_id, "user", request.message)
        self.context.memory.append(session_id, "assistant", response.content)

        return message

    def _format_history(self, history: List[dict]) -> str:
        if not history:
            return "(empty)"
        return "\n".join(f"{item['role']}: {item['content']}" for item in history)

    async def _should_search(self, question: str) -> bool:
        """Lightweight classifier that decides whether web search is needed."""
        fast_model = self.context.model_router.fast
        prompt = [
            SystemMessage(
                content=(
                    "You determine if a customer question needs up-to-date or general web knowledge. "
                    "Respond with exactly YES or NO."
                )
            ),
            HumanMessage(content=question),
        ]

        try:
            result = await fast_model.ainvoke(prompt)
            answer = getattr(result, "content", str(result)).strip().lower()
            if answer.startswith("yes") or answer.startswith("y"):
                return True
            if answer.startswith("no") or answer.startswith("n"):
                return False
        except Exception:
            pass

        lowered = question.lower()
        heuristics = ["error code", "recall", "price", "warranty", "meaning", "news"]
        return any(term in lowered for term in heuristics)

    def _format_context(self, chunks, tool_events) -> str:
        sections: List[str] = []

        manual_section = self._format_manual_chunks(chunks)
        if manual_section:
            sections.append("Manual Excerpts:\n" + manual_section)

        search_section = self._format_search_results(tool_events)
        if search_section:
            sections.append("Web Search Results:\n" + search_section)

        if sections:
            return "\n\n".join(sections)
        return "No manual excerpts retrieved."

    def _format_manual_chunks(self, chunks) -> str:
        if not chunks:
            return ""
        parts = []
        for chunk in chunks:
            summary_source = chunk.summary or getattr(chunk, "text", "")[:400]
            parts.append(
                dedent(
                    f"""
                    Source: {chunk.source} (page {chunk.page_number})
                    Snippet: {summary_source}
                    Part numbers: {', '.join(chunk.part_numbers) or 'N/A'}
                    """
                ).strip()
            )
        return "\n\n".join(parts)

    def _format_search_results(self, tool_events) -> str:
        if not tool_events:
            return ""

        entries: List[str] = []
        for event in tool_events:
            if event.name != "web_search" or not event.output:
                continue
            for index, result in enumerate(event.output, start=1):
                title = self._safe_get(result, "title")
                link = self._safe_get(result, "link")
                snippet = self._safe_get(result, "snippet")
                source = self._safe_get(result, "source") or "web"
                host = urlparse(link).netloc if link else "unknown"
                entries.append(
                    dedent(
                        f"""
                        [{index}] ({source} | {host})
                        Title: {title}
                        URL: {link}
                        Snippet: {snippet}
                        """
                    ).strip()
                )
        return "\n\n".join(entries)

    def _safe_get(self, result: Any, field: str) -> str:
        if hasattr(result, field):
            return str(getattr(result, field) or "")
        if isinstance(result, dict):
            return str(result.get(field, "") or "")
        return ""
