"""Embedding helpers (dense + sparse) for hybrid Pinecone ingest."""
from __future__ import annotations

from collections import Counter
from typing import List, Sequence

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer


class DenseEmbedder:
    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2") -> None:
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        embeddings = self.model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        return embeddings.tolist()


class SparseEmbedder:
    """Lightweight sparse encoder using TF-IDF (replace with SPLADE in production)."""

    def __init__(self) -> None:
        self.vectorizer = TfidfVectorizer(max_features=20000)

    def fit(self, texts: Sequence[str]) -> None:
        self.vectorizer.fit(texts)

    def embed(self, texts: Sequence[str]) -> List[dict]:
        matrix = self.vectorizer.transform(texts)
        results: List[dict] = []
        for row in matrix:
            coo = row.tocoo()
            indices = coo.col
            values = coo.data
            sparse_dict = {str(self.vectorizer.get_feature_names_out()[idx]): float(val) for idx, val in zip(indices, values)}
            results.append(sparse_dict)
        return results
