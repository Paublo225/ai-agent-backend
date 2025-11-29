"""Privacy-first PDF ingestion pipeline for manuals and diagrams."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List

import fitz  # PyMuPDF
import re

from backend.app.config import settings
from backend.database.clients import db_clients
from backend.ingestion.chunkers import chunk_iterable
from backend.ingestion.embeddings import DenseEmbedder, SparseEmbedder
from backend.ingestion.state import StateTracker
from backend.ingestion.vision import LocalVisionAnalyzer


@dataclass
class IngestionConfig:
    pdf_root: Path
    state_file: Path
    namespace: str = "appliance_manuals"
    batch_size: int = 32
    tmp_dir: Path = Path(".ingestion/tmp")


@dataclass
class PipelineComponents:
    dense_embedder: DenseEmbedder = field(default_factory=DenseEmbedder)
    sparse_embedder: SparseEmbedder = field(default_factory=SparseEmbedder)
    vision_analyzer: LocalVisionAnalyzer | None = field(default=None)


class IngestionPipeline:
    def __init__(self, config: IngestionConfig, components: PipelineComponents) -> None:
        self.config = config
        self.components = components
        self.state = StateTracker.load(config.state_file)
        self.tmp_dir = config.tmp_dir
        self.tmp_dir.mkdir(parents=True, exist_ok=True)

    def ingest(self) -> None:
        for pdf_path in self._iter_pdfs():
            digest = self._digest(pdf_path)
            status = self.state.state.get(digest, {}).get("status")
            if status == "completed":
                continue

            self.state.mark(digest, "processing", {"filename": pdf_path.name})
            self._process_pdf(pdf_path)
            self.state.mark(digest, "completed", {"filename": pdf_path.name})

    def _iter_pdfs(self) -> Iterable[Path]:
        yield from sorted(self.config.pdf_root.glob("**/*.pdf"))

    def _digest(self, pdf_path: Path) -> str:
        sha = hashlib.sha256()
        sha.update(pdf_path.read_bytes())
        return sha.hexdigest()

    def _process_pdf(self, pdf_path: Path) -> None:
        doc_id = pdf_path.stem
        
        # NEW: Extract metadata from directory structure
        # Assuming structure: root / appliance_type / brand / filename.pdf
        try:
            # relative_to(self.config.pdf_root) gives "Refrigerator/Samsung/model.pdf"
            # parts gives ("Refrigerator", "Samsung", "model.pdf")
            path_parts = pdf_path.relative_to(self.config.pdf_root).parts
            if len(path_parts) >= 2:
                appliance_type = path_parts[0]  # e.g. "Refrigerator"
                brand = path_parts[1]           # e.g. "Samsung"
            else:
                appliance_type = "unknown"
                brand = "unknown"
        except ValueError:
            appliance_type = "unknown"
            brand = "unknown"

        text_pages: List[str] = []
        image_metadata: List[dict] = []

        with fitz.open(pdf_path) as doc:
            for page_number, page in enumerate(doc, start=1):
                text_pages.append(page.get_text("text"))
                for img_index, img in enumerate(page.get_images(), start=1):
                    image_path = self._export_image(doc, img[0], doc_id, page_number, img_index)
                    if image_path and self.components.vision_analyzer:
                        description = self.components.vision_analyzer.analyze(image_path)
                        image_metadata.append(
                            {
                                "document_id": doc_id,
                                "page_number": page_number,
                                "image_path": str(image_path),
                                "analysis": description,
                            }
                        )

        chunks = chunk_iterable(text_pages)
        if not chunks:
            return

        self.components.sparse_embedder.fit(chunks)
        dense_vectors = self.components.dense_embedder.embed(chunks)
        sparse_vectors = self.components.sparse_embedder.embed(chunks)

        pinecone = db_clients.pinecone.Index(settings.pinecone_index)
        to_upsert = []
        for idx, (dense, sparse, chunk_text) in enumerate(zip(dense_vectors, sparse_vectors, chunks)):
            part_numbers = extract_part_numbers(chunk_text)
            models_in_chunk = extract_model_numbers(chunk_text)

            metadata = {
                "document_id": doc_id,
                "chunk_index": idx,
                "text": chunk_text[:2000],
                "appliance_type": appliance_type,
                "brand": brand,
                "part_numbers": part_numbers,
                "appliance_models": models_in_chunk,
            }
            to_upsert.append(
                {
                    "id": f"{doc_id}-{idx}",
                    "values": dense,
                    "sparse_values": self._map_sparse_tokens(sparse),
                    "metadata": metadata,
                }
            )

        for start in range(0, len(to_upsert), self.config.batch_size):
            batch = to_upsert[start : start + self.config.batch_size]
            pinecone.upsert(vectors=batch, namespace=self.config.namespace)

        supabase = db_clients.supabase
        supabase.table("documents").upsert(
            {
                "document_id": doc_id,
                "filename": pdf_path.name,
                "total_pages": len(text_pages),
                "metadata": {"images": image_metadata},
            }
        ).execute()

    def _export_image(self, doc: fitz.Document, xref: int, doc_id: str, page_number: int, img_index: int) -> Path | None:
        try:
            pix = fitz.Pixmap(doc, xref)
            if pix.n - pix.alpha > 3:
                pix = fitz.Pixmap(fitz.csRGB, pix)
            image_path = self.tmp_dir / f"{doc_id}_p{page_number}_{img_index}.png"
            pix.save(image_path)
            return image_path
        except Exception:
            return None

    def _map_sparse_tokens(self, sparse: dict) -> dict:
        import zlib

        indices: List[int] = []
        values: List[float] = []
        for token, weight in sparse.items():
            indices.append(zlib.crc32(token.encode("utf-8")))
            values.append(weight)
        return {"indices": indices, "values": values}


PART_NUMBER_RE = re.compile(r"\b[A-Z0-9]{3,5}-?[0-9A-Z]{3,6}\b")

def extract_part_numbers(text: str) -> list[str]:
    # tune this regex to your brand patterns
    candidates = PART_NUMBER_RE.findall(text)
    # optional: filter obvious junk
    return sorted(set(candidates))

def extract_model_numbers(text: str) -> list[str]:
    # very rough example – customize for your manuals
    MODEL_RE = re.compile(r"\b[A-Z0-9]{3,}-[A-Z0-9/]{3,}\b")
    return sorted(set(MODEL_RE.findall(text)))
