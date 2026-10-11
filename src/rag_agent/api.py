"""HTTP interface for rag-doc-agent.

Thin FastAPI wrapper over the same core the CLI uses: ingest documents into
the vector store, then ask questions and get cited answers back.

Run:
    uvicorn rag_agent.api:app --reload
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from rag_agent.agent import AgentError, RagAgent
from rag_agent.config import ConfigError, load_settings
from rag_agent.ingest import IngestError, ingest_directory
from rag_agent.vectorstore import VectorStore

app = FastAPI(
    title="rag-doc-agent",
    description="Enterprise document RAG agent with page-level source citations.",
    version="0.1.0",
)


class HistoryTurn(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    content: str


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    history: list[HistoryTurn] = Field(default_factory=list)


class Source(BaseModel):
    source: str
    page: int
    score: float


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]


class IngestRequest(BaseModel):
    path: str = Field(min_length=1)
    strategy: str = Field(default="fixed", pattern="^(fixed|sentence)$")


class IngestResponse(BaseModel):
    files_ingested: int
    chunks_added: int


class StatsResponse(BaseModel):
    collection: str
    chunk_count: int
    model: str
    top_k: int


def _build_agent() -> tuple[RagAgent, Any]:
    try:
        settings = load_settings()
    except ConfigError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    store = VectorStore(settings.db_path, settings.collection_name)
    return RagAgent(settings, store), settings


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    agent, _settings = _build_agent()
    history = [{"role": h.role, "content": h.content} for h in req.history]
    try:
        result = agent.ask(req.question, history=history or None)
    except AgentError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return AskResponse(
        answer=result.text,
        sources=[
            Source(source=s.source, page=s.page, score=s.score) for s in result.sources
        ],
    )


@app.post("/ingest", response_model=IngestResponse)
def ingest(req: IngestRequest) -> IngestResponse:
    agent, settings = _build_agent()
    try:
        result = ingest_directory(
            Path(req.path), settings, agent._store, strategy=req.strategy
        )
    except IngestError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return IngestResponse(files_ingested=len(result), chunks_added=sum(result.values()))


@app.get("/stats", response_model=StatsResponse)
def stats() -> StatsResponse:
    agent, settings = _build_agent()
    return StatsResponse(
        collection=settings.collection_name,
        chunk_count=agent._store.count(),
        model=settings.model,
        top_k=settings.top_k,
    )
