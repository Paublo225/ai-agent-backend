"""Utility script to create Pinecone index with hybrid (dense+sparse) support."""
from __future__ import annotations

from pinecone import Pinecone

from backend.app.config import settings


def create_index() -> None:
    pc = Pinecone(api_key=settings.pinecone_api_key)
    indexes = {index["name"] for index in pc.list_indexes()}
    if settings.pinecone_index in indexes:
        print(f"Index {settings.pinecone_index} already exists")
        return

    pc.create_index(
        name=settings.pinecone_index,
        dimension=768,
        metric="cosine",
        spec={
            "serverless": {
                "cloud": "aws",
                "region": settings.pinecone_environment or "us-east-1",
            }
        },
        pod_type="p1",
        # enable sparse vectors for SPLADE/BM25 hybrid search
        metadata_config={"indexed": ["document_id", "appliance_type", "brand", "diagram_type"]},
    )
    print(f"Index {settings.pinecone_index} created")


if __name__ == "__main__":
    create_index()
