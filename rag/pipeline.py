"""Haystack-powered hybrid retrieval pipeline."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import List

from haystack import Pipeline
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.rankers import TransformersSimilarityRanker
from haystack_integrations.document_stores.pinecone import PineconeDocumentStore
from haystack_integrations.components.retrievers.pinecone import PineconeEmbeddingRetriever

from backend.app.config import settings
from backend.app.models import DocumentChunk


@dataclass
class RetrievalResult:
    document_id: str
    source: str
    page_number: int | None
    appliance_type: str | None
    summary: str | None
    part_numbers: List[str]
    score: float
    text: str

    def to_model(self) -> DocumentChunk:
        return DocumentChunk(
            document_id=self.document_id,
            source=self.source,
            page_number=self.page_number,
            appliance_type=self.appliance_type,
            summary=self.summary or self.text[:280],
            part_numbers=self.part_numbers,
        )


class RetrievalPipeline:
    def __init__(self, top_k: int = 5) -> None:
        self.top_k = top_k
        self.document_store = PineconeDocumentStore(
            api_key=settings.pinecone_api_key,
            environment=settings.pinecone_environment,
            index=settings.pinecone_index,
            namespace=settings.pinecone_index,
            dimension=768,
        )
        self.pipeline = Pipeline()
        self.pipeline.add_component(
            "text_embedder",
            SentenceTransformersTextEmbedder(model="sentence-transformers/all-mpnet-base-v2"),
        )
        self.pipeline.add_component(
            "retriever",
            PineconeEmbeddingRetriever(document_store=self.document_store, top_k=top_k * 2),
        )
        self.pipeline.add_component(
            "ranker",
            TransformersSimilarityRanker(model="cross-encoder/ms-marco-MiniLM-L-6-v2", top_k=top_k),
        )
        self.pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
        self.pipeline.connect("retriever", "ranker")

    async def retrieve(self, query: str) -> List[RetrievalResult]:
        result = await asyncio.to_thread(self.pipeline.run, data={"text_embedder": {"text": query}})
        documents = result["ranker"]["documents"]
        return [self._to_result(doc) for doc in documents]

    def _to_result(self, document) -> RetrievalResult:
        meta = document.meta or {}
        return RetrievalResult(
            document_id=meta.get("document_id", document.id),
            source=meta.get("source", "pinecone"),
            page_number=meta.get("page_number"),
            appliance_type=meta.get("appliance_type"),
            summary=meta.get("summary"),
            part_numbers=meta.get("part_numbers", []),
            score=document.score or 0.0,
            text=document.content,
        )
