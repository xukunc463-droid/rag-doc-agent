"""ChromaDB vector store wrapper.

Uses ChromaDB's default embedding function (all-MiniLM-L6-v2 via onnxruntime)
so the demo runs without an extra embeddings API. Swapping in an API-based
embedder later only touches this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import chromadb
from chromadb.api.models.Collection import Collection

from rag_agent.chunking import Chunk


@dataclass(frozen=True)
class Hit:
    """A retrieval result with provenance metadata."""

    text: str
    source: str
    page: int
    score: float


class VectorStore:
    """Persistent ChromaDB collection with a minimal add/search surface."""

    def __init__(self, db_path: Path, collection_name: str) -> None:
        self._db_path = db_path
        self._collection_name = collection_name
        self._client = chromadb.PersistentClient(path=str(db_path))
        self._collection: Collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(self, chunks: list[Chunk]) -> None:
        """Write chunks to the collection. Ids are deterministic per source/page/id."""
        if not chunks:
            return

        ids = [f"{c.source}:{c.page}:{c.chunk_id}" for c in chunks]
        documents = [c.text for c in chunks]
        metadatas = [
            {"source": c.source, "page": c.page, "chunk_id": c.chunk_id}
            for c in chunks
        ]
        self._collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
        )

    def search(self, query: str, top_k: int) -> list[Hit]:
        """Return the top-k most similar chunks to the query.

        Raises:
            ValueError: if top_k <= 0.
        """
        if top_k <= 0:
            raise ValueError(f"top_k must be > 0, got {top_k}")
        if self._collection.count() == 0:
            return []

        result = self._collection.query(
            query_texts=[query],
            n_results=min(top_k, self._collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        documents = result.get("documents") or [[]]
        metadatas = result.get("metadatas") or [[]]
        distances = result.get("distances") or [[]]

        hits: list[Hit] = []
        for text, meta, dist in zip(documents[0], metadatas[0], distances[0]):
            hits.append(
                Hit(
                    text=text or "",
                    source=str(meta.get("source", "unknown")),
                    page=int(meta.get("page", 0)),
                    # cosine distance -> similarity
                    score=1.0 - float(dist),
                )
            )
        return hits

    def count(self) -> int:
        return self._collection.count()

    def reset(self) -> None:
        """Drop and recreate the collection. Used by tests and re-ingest flows."""
        self._client.delete_collection(self._collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )
