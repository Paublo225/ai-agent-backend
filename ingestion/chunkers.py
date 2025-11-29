"""Chunking utilities for PDF ingestion."""
from __future__ import annotations

from typing import Iterable, List

from langchain.text_splitter import RecursiveCharacterTextSplitter

DEFAULT_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=1200,
    chunk_overlap=200,
    separators=["\n\n", "\n", ".", "!", "?"]
)


def chunk_text(text: str, splitter: RecursiveCharacterTextSplitter = DEFAULT_SPLITTER) -> List[str]:
    return splitter.split_text(text)


def chunk_iterable(pages: Iterable[str]) -> List[str]:
    chunks: List[str] = []
    for page in pages:
        chunks.extend(chunk_text(page))
    return chunks
