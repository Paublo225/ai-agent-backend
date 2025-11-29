"""Monitoring helpers (Sentry, LangSmith, Prometheus)."""
from __future__ import annotations

import logging
import os

import sentry_sdk
from langsmith import Client as LangSmithClient
from prometheus_client import Counter, Histogram

from backend.app.config import settings

logger = logging.getLogger(__name__)

CHAT_REQUESTS = Counter(
    "chat_requests_total", "Total chat requests", labelnames=("endpoint",)
)
CHAT_LATENCY = Histogram(
    "chat_latency_seconds", "Latency of chat responses", labelnames=("endpoint",)
)


def setup_observability() -> LangSmithClient | None:
    if settings.sentry_dsn:
        sentry_sdk.init(dsn=settings.sentry_dsn, traces_sample_rate=0.2)
        logger.info("Sentry initialized")

    langsmith_client: LangSmithClient | None = None
    if settings.langsmith_api_key:
        langsmith_client = LangSmithClient(api_key=settings.langsmith_api_key)
        logger.info("LangSmith client ready (project=%s)", settings.langsmith_project)
        os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
        os.environ.setdefault("LANGCHAIN_PROJECT", settings.langsmith_project)
        os.environ.setdefault("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")

    return langsmith_client
