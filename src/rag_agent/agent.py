"""RAG question-answering agent built on the Anthropic Messages API.

Pipeline: retrieve top-k chunks -> build grounded prompt -> generate answer
with source citations. Retrieval is exposed as a tool so the model can ask
for more context when the first pass is insufficient.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import anthropic

from rag_agent.config import Settings
from rag_agent.vectorstore import Hit, VectorStore

SYSTEM_PROMPT = """\
You are a documentation assistant for chip / semiconductor engineering teams.
Answer strictly from the provided context excerpts.

Rules:
1. Cite sources as [source:page] after each factual claim.
2. If the context does not contain the answer, say so explicitly. Do not guess.
3. Prefer precise values (numbers, units, part codes) exactly as written in context.
4. Answer in the same language as the question.
"""

RETRIEVAL_TOOL: dict[str, Any] = {
    "name": "search_docs",
    "description": (
        "Search the technical document knowledge base for passages relevant "
        "to a query. Use this before answering if the provided context is "
        "insufficient."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query, in the same language as the question.",
            }
        },
        "required": ["query"],
    },
}

MAX_TOOL_ROUNDS = 3


@dataclass(frozen=True)
class Answer:
    """Final answer with the context excerpts used to produce it."""

    text: str
    sources: list[Hit]


class AgentError(RuntimeError):
    """Raised when the agent cannot produce an answer."""


class RagAgent:
    """Retrieve-augmented generation agent over a VectorStore."""

    def __init__(self, settings: Settings, store: VectorStore) -> None:
        self._settings = settings
        self._store = store
        self._client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

    def retrieve(self, query: str) -> list[Hit]:
        return self._store.search(query, self._settings.top_k)

    def _build_context(self, hits: list[Hit]) -> str:
        if not hits:
            return "(no context retrieved)"
        blocks = []
        for index, hit in enumerate(hits, start=1):
            blocks.append(
                f"[Excerpt {index}] source={hit.source} page={hit.page} "
                f"score={hit.score:.3f}\n{hit.text}"
            )
        return "\n\n".join(blocks)

    def ask(self, question: str, history: list[dict[str, str]] | None = None) -> Answer:
        """Answer a question using retrieval-augmented generation.

        Args:
            question: User question.
            history: Optional prior turns as [{"role": "user"/"assistant", "content": ...}].

        Returns:
            Answer with cited text and the retrieved source excerpts.

        Raises:
            AgentError: if the API call fails or no answer is produced.
        """
        if not question.strip():
            raise AgentError("Question must not be empty")

        initial_hits = self.retrieve(question)
        all_hits: dict[str, Hit] = {
            f"{h.source}:{h.page}:{i}": h for i, h in enumerate(initial_hits)
        }

        messages: list[dict[str, Any]] = list(history or [])
        messages.append(
            {
                "role": "user",
                "content": (
                    f"Context:\n{self._build_context(initial_hits)}\n\n"
                    f"Question: {question}"
                ),
            }
        )

        for _ in range(MAX_TOOL_ROUNDS):
            response = self._client.messages.create(
                model=self._settings.model,
                max_tokens=1024,
                system=SYSTEM_PROMPT,
                tools=[RETRIEVAL_TOOL],
                messages=messages,
            )

            if response.stop_reason == "tool_use":
                tool_blocks = [b for b in response.content if b.type == "tool_use"]
                tool_results = []
                for block in tool_blocks:
                    extra = self.retrieve(block.input.get("query", question))
                    for i, hit in enumerate(extra):
                        all_hits.setdefault(
                            f"{hit.source}:{hit.page}:{len(all_hits)}", hit
                        )
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": self._build_context(extra),
                        }
                    )
                messages.append({"role": "assistant", "content": response.content})
                messages.append({"role": "user", "content": tool_results})
                continue

            text_parts = [b.text for b in response.content if b.type == "text"]
            answer_text = "\n".join(text_parts).strip()
            if not answer_text:
                raise AgentError("Model returned no text content")
            return Answer(text=answer_text, sources=list(all_hits.values()))

        raise AgentError(f"Agent did not finish within {MAX_TOOL_ROUNDS} tool rounds")
