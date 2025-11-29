"""CLI entry point for running the ingestion pipeline."""
from __future__ import annotations

import argparse
from pathlib import Path

from backend.ingestion.embeddings import DenseEmbedder, SparseEmbedder
from backend.ingestion.pipeline import IngestionConfig, IngestionPipeline, PipelineComponents
from backend.ingestion.vision import LocalVisionAnalyzer


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest appliance manuals")
    parser.add_argument("pdf_root", type=Path, help="Path to directory containing PDFs")
    parser.add_argument(
        "--state-file", type=Path, default=Path(".ingestion/state.json"), help="State checkpoint file"
    )
    parser.add_argument(
        "--with-vision",
        action="store_true",
        help="Enable local LLaVA vision analysis for diagrams",
    )
    args = parser.parse_args()

    components = PipelineComponents(
        dense_embedder=DenseEmbedder(),
        sparse_embedder=SparseEmbedder(),
        vision_analyzer=LocalVisionAnalyzer() if args.with_vision else None,
    )

    pipeline = IngestionPipeline(
        config=IngestionConfig(pdf_root=args.pdf_root, state_file=args.state_file),
        components=components,
    )
    pipeline.ingest()


if __name__ == "__main__":
    main()
