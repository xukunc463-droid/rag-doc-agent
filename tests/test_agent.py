"""Unit tests for agent input validation and context formatting.

These tests exercise pure logic only — no network, no Anthropic API calls.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rag_agent.agent import AgentError, RagAgent
from rag_agent.config import Settings
from rag_agent.vectorstore import Hit, VectorStore


@pytest.fixture
def settings() -> Settings:
    return Settings(
        anthropic_api_key="sk-ant-test-key",
        model="claude-sonnet-4-5",
        chunk_size=1024,
        chunk_overlap=128,
        top_k=5,
        db_path=Path("./data/test_chroma"),
        collection_name="test_docs",
    )


@pytest.fixture
def empty_store(tmp_path) -> VectorStore:
    return VectorStore(tmp_path / "chroma", "test_docs")


class TestRagAgentValidation:
    def test_rejects_empty_question(self, settings, empty_store) -> None:
        # Arrange
        agent = RagAgent(settings, empty_store)

        # Act / Assert
        with pytest.raises(AgentError, match="empty"):
            agent.ask("   ")

    def test_rejects_whitespace_question(self, settings, empty_store) -> None:
        # Arrange
        agent = RagAgent(settings, empty_store)

        # Act / Assert
        with pytest.raises(AgentError, match="empty"):
            agent.ask("\n\t")


class TestContextBuilding:
    def test_build_context_includes_source_page_and_score(self, settings, empty_store) -> None:
        # Arrange
        agent = RagAgent(settings, empty_store)
        hits = [
            Hit(text="Vout = 3.3V", source="demo_datasheet.pdf", page=3, score=0.82),
            Hit(text="Tolerance 2%", source="demo_datasheet.pdf", page=4, score=0.75),
        ]

        # Act
        ctx = agent._build_context(hits)

        # Assert
        assert "demo_datasheet.pdf" in ctx
        assert "page=3" in ctx
        assert "page=4" in ctx
        assert "0.820" in ctx
        assert "Vout = 3.3V" in ctx

    def test_build_context_handles_empty_hits(self, settings, empty_store) -> None:
        # Arrange
        agent = RagAgent(settings, empty_store)

        # Act
        ctx = agent._build_context([])

        # Assert
        assert "no context" in ctx.lower()


class TestRetrieve:
    def test_retrieve_returns_empty_for_empty_store(self, settings, empty_store) -> None:
        # Arrange
        agent = RagAgent(settings, empty_store)

        # Act
        hits = agent.retrieve("anything")

        # Assert
        assert hits == []
