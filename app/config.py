"""Centralized configuration management for the backend application."""
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

try:
    from pydantic import AnyHttpUrl, Field
    from pydantic_settings import BaseSettings
except ImportError:
    from pydantic import AnyHttpUrl, BaseSettings, Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables or .env file."""

    app_name: str = Field(default="Appliance Repair Agent")
    environment: str = Field(default="development")
    api_base_url: str = Field(default="/api")

    # Model + provider credentials
    groq_api_key: str = Field(default="", repr=False)
    gemini_api_key: str = Field(default="", repr=False)

    # Vector + relational stores
    pinecone_api_key: str = Field(default="", repr=False)
    pinecone_environment: str = Field(default="")
    pinecone_index: str = Field(default="appliance-manuals")
    supabase_url: str = Field(default="", repr=False)
    supabase_key: str = Field(default="", repr=False)

    # Monitoring
    langsmith_api_key: str = Field(default="", repr=False)
    langsmith_project: str = Field(default="appliance-agent")
    # LangSmith tracing toggle (optional)
    langchain_tracing_v2: bool = Field(default=False)
    sentry_dsn: str = Field(default="", repr=False)

    # Search APIs
    google_search_cx: str = Field(default="")
    google_search_api_key: str = Field(default="", repr=False)
    bing_search_api_key: str = Field(default="", repr=False)

    allowed_origins: List[AnyHttpUrl] = Field(default_factory=list)

    class Config:
        env_file = Path(__file__).resolve().parent.parent / ".env"
        env_file_encoding = "utf-8"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached settings instance to avoid reparsing the .env file."""
    return Settings()


settings = get_settings()
