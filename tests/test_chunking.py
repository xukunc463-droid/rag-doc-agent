"""Unit tests for chunking strategies (AAA pattern)."""

from __future__ import annotations

import pytest

from rag_agent.chunking import (
    build_chunks,
    chunk_by_sentence,
    chunk_fixed,
)


class TestChunkFixed:
    def test_returns_empty_list_for_empty_text(self) -> None:
        # Arrange
        text = ""

        # Act
        result = chunk_fixed(text, chunk_size=100, overlap=20)

        # Assert
        assert result == []

    def test_returns_single_chunk_when_text_fits(self) -> None:
        # Arrange
        text = "short text"

        # Act
        result = chunk_fixed(text, chunk_size=100, overlap=20)

        # Assert
        assert result == ["short text"]

    def test_splits_into_overlapping_chunks(self) -> None:
        # Arrange
        text = "a" * 250

        # Act
        result = chunk_fixed(text, chunk_size=100, overlap=20)

        # Assert
        assert len(result) == 3
        # overlap: tail of chunk N appears at head of chunk N+1
        assert result[0][-20:] == result[1][:20]

    def test_rejects_zero_chunk_size(self) -> None:
        with pytest.raises(ValueError, match="chunk_size"):
            chunk_fixed("text", chunk_size=0, overlap=0)

    def test_rejects_overlap_not_less_than_chunk_size(self) -> None:
        with pytest.raises(ValueError, match="overlap"):
            chunk_fixed("text", chunk_size=100, overlap=100)


class TestChunkBySentence:
    def test_keeps_sentences_together_up_to_limit(self) -> None:
        # Arrange
        text = "First sentence here. Second sentence here. Third one."

        # Act
        result = chunk_by_sentence(text, chunk_size=40, overlap=0)

        # Assert
        assert len(result) >= 2
        assert all(len(c) <= 50 for c in result)  # allow minor packing slack

    def test_handles_chinese_punctuation(self) -> None:
        # Arrange
        text = "这是第一句。这是第二句。这是第三句。"

        # Act
        result = chunk_by_sentence(text, chunk_size=20, overlap=0)

        # Assert
        assert len(result) >= 2

    def test_returns_empty_for_blank_input(self) -> None:
        assert chunk_by_sentence("   \n  ", chunk_size=50, overlap=0) == []


class TestBuildChunks:
    def test_attaches_provenance_metadata(self) -> None:
        # Arrange
        text = "Parameter Vout is 3.3V. Tolerance is 2 percent."

        # Act
        chunks = build_chunks(
            text, source="demo_datasheet.pdf", page=3,
            chunk_size=20, overlap=0, strategy="fixed",
        )

        # Assert
        assert len(chunks) >= 1
        assert all(c.source == "demo_datasheet.pdf" for c in chunks)
        assert all(c.page == 3 for c in chunks)
        assert [c.chunk_id for c in chunks] == list(range(len(chunks)))

    def test_rejects_unknown_strategy(self) -> None:
        with pytest.raises(ValueError, match="strategy"):
            build_chunks(
                "text", source="a.pdf", page=1,
                chunk_size=10, overlap=0, strategy="nope",
            )
