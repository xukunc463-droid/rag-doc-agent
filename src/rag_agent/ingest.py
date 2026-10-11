"""Document ingestion: PDF/text -> chunks -> vector store."""

from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader

from rag_agent.chunking import Chunk, build_chunks
from rag_agent.config import Settings
from rag_agent.vectorstore import VectorStore


class IngestError(RuntimeError):
    """Raised when document ingestion fails."""


def extract_pdf_pages(path: Path) -> list[tuple[str, int]]:
    """Extract (text, page_number) pairs from a PDF. Page numbers are 1-based."""
    if not path.exists():
        raise IngestError(f"File not found: {path}")

    try:
        reader = PdfReader(str(path))
    except Exception as exc:  # pypdf raises a variety of parse errors
        raise IngestError(f"Failed to open PDF {path.name}: {exc}") from exc

    pages: list[tuple[str, int]] = []
    for index, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except Exception as exc:
            raise IngestError(
                f"Failed to extract text from {path.name} page {index + 1}: {exc}"
            ) from exc
        pages.append((text, index + 1))
    return pages


def chunk_document(
    path: Path,
    settings: Settings,
    strategy: str = "fixed",
) -> list[Chunk]:
    """Load a PDF or .txt file and produce chunks with provenance metadata."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        pages = extract_pdf_pages(path)
    elif suffix in {".txt", ".md"}:
        text = path.read_text(encoding="utf-8")
        pages = [(text, 1)]
    else:
        raise IngestError(f"Unsupported file type: {suffix} ({path.name})")

    chunks: list[Chunk] = []
    for text, page in pages:
        if not text.strip():
            continue
        chunks.extend(
            build_chunks(
                text=text,
                source=path.name,
                page=page,
                chunk_size=settings.chunk_size,
                overlap=settings.chunk_overlap,
                strategy=strategy,
            )
        )
    return chunks


def ingest_file(
    path: Path,
    settings: Settings,
    store: VectorStore,
    strategy: str = "fixed",
) -> int:
    """Chunk one file and write chunks into the vector store.

    Returns:
        Number of chunks written.
    """
    chunks = chunk_document(path, settings, strategy=strategy)
    if not chunks:
        raise IngestError(f"No extractable text in {path.name}")
    store.add_chunks(chunks)
    return len(chunks)


def ingest_directory(
    directory: Path,
    settings: Settings,
    store: VectorStore,
    strategy: str = "fixed",
) -> dict[str, int]:
    """Ingest every supported file under a directory (non-recursive).

    Returns:
        Mapping of filename -> chunk count written.
    """
    if not directory.is_dir():
        raise IngestError(f"Not a directory: {directory}")

    supported = {".pdf", ".txt", ".md"}
    results: dict[str, int] = {}
    for entry in sorted(directory.iterdir()):
        if entry.is_file() and entry.suffix.lower() in supported:
            results[entry.name] = ingest_file(entry, settings, store, strategy)
    return results
