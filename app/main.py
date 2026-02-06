"""FastAPI application entry point."""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import List, Optional

from fastapi import Depends, FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from langchain_core.messages import AIMessage
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, generate_latest

from backend.agent.memory import ConversationMemory
from backend.agent.model_router import ModelRouter
from backend.agent.service import AgentContext, AgentOrchestrator
from backend.app.config import Settings, get_settings, settings
from backend.app.models import ChatRequest, ChatResponse, ChatMessage, HealthResponse, ImageUploadResponse
from backend.monitoring.observability import CHAT_LATENCY, CHAT_REQUESTS, setup_observability
from backend.rag.pipeline import RetrievalPipeline
from backend.tools.router import ToolRouter
from backend.tools.search import SearchTool
from backend.tools.vision import VisionTool

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

app = FastAPI(title=settings.app_name, version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[str(origin) for origin in settings.allowed_origins] or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_langsmith = setup_observability()
_registry = CollectorRegistry()


class LocalEchoModel:
    async def ainvoke(self, messages):
        content = ""
        if messages:
            last = messages[-1]
            content = getattr(last, "content", str(last))
        return AIMessage(content=f"[mock-response] {content}")


def _build_agent(settings: Settings) -> AgentOrchestrator:
    # Prioritize Gemini if GEMINI_API_KEY is set
    if settings.gemini_api_key:
        logger.info("Using Google Gemini models")
        def build_model(model_name: str, temperature: float) -> object:
            return ChatGoogleGenerativeAI(
                google_api_key=settings.gemini_api_key,
                model=model_name,
                temperature=temperature,
                convert_system_message_to_human=True,
            )
        primary_llm = build_model("gemini-2.5-flash", 0.1)
        fast_llm = build_model("gemini-2.5-flash", 0.3)
        extraction_llm = build_model("gemini-1.5-flash", 0.0)
    elif settings.groq_api_key:
        logger.info("Using Groq models")
        def build_model(model_name: str, temperature: float) -> object:
            return ChatGroq(
                groq_api_key=settings.groq_api_key,
                model_name=model_name,
                temperature=temperature,
            )
        primary_llm = build_model("llama-3.1-70b-versatile", 0.1)
        fast_llm = build_model("llama-3.1-8b-instant", 0.3)
        extraction_llm = build_model("qwen-2.5-7b-instruct", 0.0)
    else:
        logger.warning("No API keys found (GEMINI_API_KEY or GROQ_API_KEY). Using LocalEchoModel.")
        primary_llm = LocalEchoModel()
        fast_llm = LocalEchoModel()
        extraction_llm = LocalEchoModel()

    router = ModelRouter(primary=primary_llm, fast=fast_llm, extractor=extraction_llm)
    
    # Initialize RAG pipeline only if Pinecone is configured
    try:
        if settings.pinecone_api_key and settings.pinecone_index:
            retrieval = RetrievalPipeline()
        else:
            retrieval = None
    except Exception as e:
        logger.warning(f"RAG pipeline disabled: {e}")
        retrieval = None
    
    tools = ToolRouter(search_tool=SearchTool(), vision_tool=VisionTool())

    context = AgentContext(
        model_router=router,
        retrieval_pipeline=retrieval,
        tools=tools,
        memory=ConversationMemory(),
    )
    return AgentOrchestrator(context)


agent = _build_agent(settings)


@app.get("/")
async def root():
    return {"status": "ok", "message": "AI Agent Backend is running"}


@app.get("/health", response_model=HealthResponse)
def health(settings: Settings = Depends(get_settings)) -> HealthResponse:
    return HealthResponse(status="ok", environment=settings.environment)


@app.get("/metrics")
def metrics() -> PlainTextResponse:
    payload = generate_latest()  # type: ignore[arg-type]
    return PlainTextResponse(payload, media_type=CONTENT_TYPE_LATEST)


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    start = time.perf_counter()
    session_id = request.session_id or str(uuid.uuid4())
    request = ChatRequest(**request.model_dump(), session_id=session_id)

    CHAT_REQUESTS.labels(endpoint="chat").inc()
    message = await agent.run(request)

    latency_ms = (time.perf_counter() - start) * 1000
    CHAT_LATENCY.labels(endpoint="chat").observe(latency_ms / 1000)

    return ChatResponse(session_id=session_id, messages=[message], latency_ms=latency_ms)


@app.post("/api/upload-image", response_model=ImageUploadResponse)
async def upload_image(file: UploadFile = File(...)) -> ImageUploadResponse:
    tmp_id = f"uploaded://{uuid.uuid4()}"
    return ImageUploadResponse(image_id=tmp_id, filename=file.filename, content_type=file.content_type or "")


@app.get("/api/conversation/{session_id}", response_model=List[ChatMessage])
async def conversation_history(session_id: str) -> List[ChatMessage]:
    rows = agent.context.memory.fetch(session_id)
    return [ChatMessage(role=row["role"], content=row["content"]) for row in rows]


@app.websocket("/api/ws/chat")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    session_id = str(uuid.uuid4())
    try:
        while True:
            data = await websocket.receive_json()
            request = ChatRequest(
                session_id=session_id,
                message=data.get("message", ""),
                image_ids=data.get("image_ids", []),
                metadata=data.get("metadata", {}),
            )
            response = await agent.run(request)
            await websocket.send_json(ChatResponse(session_id=session_id, messages=[response], latency_ms=0).model_dump(mode='json'))
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected (session=%s)", session_id)
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("WebSocket error: %s", exc)
        await websocket.close(code=1011)
