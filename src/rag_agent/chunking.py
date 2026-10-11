"""Text chunking strategies for document ingestion.

Two strategies:
- Fixed-size with character overlap (default, robust for mixed-format docs)
- Sentence-aware: split on sentence boundaries, then pack into chunks

Chip datasheets mix paragraphs, tables and parameter lists. Sentence-aware
chunking keeps parameter groups together; fixed-size is the safe baseline.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Latin terminators need trailing whitespace so "3.3V" stays intact.
# CJK terminators end a sentence with no space, so they split unconditionally.
SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？；])|(?<=[.!?;])\s+|\n{2,}")


@dataclass(frozen=True)
class Chunk:
    """A single text chunk with provenance metadata."""

    text: str
    source: str
    page: int
    chunk_id: int


def _split_sentences(text: str) -> list[str]:
    parts = [p.strip() for p in SENTENCE_SPLIT_RE.split(text)]
    return [p for p in parts if p]


def chunk_fixed(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into fixed-size chunks with character overlap.

    Args:
        text: Input text.
        chunk_size: Maximum characters per chunk.
        overlap: Characters shared between consecutive chunks.

    Returns:
        List of chunk strings. Empty input returns empty list.

    Raises:
        ValueError: if chunk_size <= 0 or overlap not in [0, chunk_size).
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be > 0, got {chunk_size}")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(f"overlap must be in [0, {chunk_size}), got {overlap}")

    cleaned = text.strip()
    if not cleaned:
        return []

    step = chunk_size - overlap
    chunks: list[str] = []
    start = 0
    while start < len(cleaned):
        end = start + chunk_size
        piece = cleaned[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(cleaned):
            break
        start += step
    return chunks


def chunk_by_sentence(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split on sentence boundaries, then pack sentences up to chunk_size.

    Overlap is applied at the character level across the packed chunk,
    matching chunk_fixed semantics.
    """
    sentences = _split_sentences(text)
    if not sentences:
        return []

    packed: list[str] = []
    buffer = ""
    for sentence in sentences:
        candidate = f"{buffer} {sentence}".strip() if buffer else sentence
        if len(candidate) >= chunk_size and buffer:
            packed.append(buffer)
            tail = buffer[-overlap:] if overlap else ""
            buffer = f"{tail} {sentence}".strip() if tail else sentence
        else:
            buffer = candidate
    if buffer:
        packed.append(buffer)
    return packed


def build_chunks(
    text: str,
    source: str,
    page: int,
    chunk_size: int,
    overlap: int,
    strategy: str = "fixed",
) -> list[Chunk]:
    """Chunk a page of text and attach provenance metadata."""
    if strategy == "fixed":
        pieces = chunk_fixed(text, chunk_size, overlap)
    elif strategy == "sentence":
        pieces = chunk_by_sentence(text, chunk_size, overlap)
    else:
        raise ValueError(f"Unknown chunk strategy: {strategy!r}")

    return [
        Chunk(text=piece, source=source, page=page, chunk_id=index)
        for index, piece in enumerate(pieces)
    ]
