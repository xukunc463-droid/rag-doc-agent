# rag-doc-agent

Technical document RAG agent — ask questions over chip datasheets, design docs, and test reports, get answers with source citations.

Built for a semiconductor R&D knowledge-base scenario: documents are messy (PDF tables, parameter lists, formulas mixed with prose), and engineers need precise values with page-level provenance.

## Features

- **Document ingestion** — PDF / TXT / Markdown → chunks → ChromaDB vector store
- **Two chunking strategies** — fixed-size with overlap (baseline) and sentence-aware packing
- **Grounded answers** — model answers strictly from retrieved context, cites `[source:page]`
- **Tool-use retrieval** — agent can request more context when the first retrieval pass is thin
- **Multi-turn chat** — conversation history maintained across turns
- **Provenance tracking** — every answer lists file, page, and similarity score
- **HTTP API** — FastAPI wrapper (`/ask`, `/ingest`, `/stats`, `/healthz`)
- **Regression eval** — 210-item labeled set + offline baseline-vs-optimized report

## Architecture

```
PDF/TXT ──► ingest ──► chunking ──► ChromaDB (cosine)
                                        │
Question ──► retrieve top-k ────────────┤
                                        ▼
                              Anthropic Messages API
                              (system prompt + context + tools)
                                        │
                                        ▼
                              Answer + [source:page] citations
```

## Setup

```bash
# 1. Install
pip install -r requirements.txt

# 2. Configure
cp .env.example .env
# Edit .env: set ANTHROPIC_API_KEY

# 3. Add documents
mkdir -p data/docs
# Drop your PDF / .txt / .md files into data/docs/
```

## Usage

```bash
# Ingest a directory of documents
python -m rag_agent.cli ingest --path ./data/docs

# Ingest with sentence-aware chunking
python -m rag_agent.cli ingest --path ./data/docs --strategy sentence

# Ask one question
python -m rag_agent.cli query "What is the output voltage tolerance?"

# Interactive chat
python -m rag_agent.cli chat

# Show config / index stats
python -m rag_agent.cli stats
```

## Example session

```
$ python -m rag_agent.cli query "What is Vout tolerance?"

╭─ Answer ──────────────────────────────────────╮
│ Vout tolerance is ±2% across the full         │
│ temperature range [demo_datasheet.pdf:4].     │
╰───────────────────────────────────────────────╯

╭─ Sources ─────────────────────────────────────╮
│ File                │ Page │ Score │          │
│ demo_datasheet.pdf  │    4 │ 0.823 │          │
│ demo_datasheet.pdf  │    3 │ 0.791 │          │
╰───────────────────────────────────────────────╯
```

## Configuration

| Env var | Default | Description |
|---------|---------|-------------|
| `ANTHROPIC_API_KEY` | — (required) | Anthropic API key |
| `RAG_MODEL` | `claude-sonnet-4-5` | Generation model |
| `RAG_CHUNK_SIZE` | `1024` | Max characters per chunk |
| `RAG_CHUNK_OVERLAP` | `128` | Overlap between chunks |
| `RAG_TOP_K` | `5` | Retrieved passages per query |
| `RAG_DB_PATH` | `./data/chroma` | ChromaDB persistent directory |
| `RAG_COLLECTION` | `chip_docs` | Collection name |

## Project structure

```
src/rag_agent/
├── config.py       # Env-based settings, validation
├── chunking.py     # Fixed-size and sentence-aware chunkers
├── ingest.py       # PDF/TXT loading and ingestion pipeline
├── vectorstore.py  # ChromaDB wrapper (add / search / reset)
├── agent.py        # RAG agent with tool-use retrieval loop
├── api.py          # FastAPI HTTP interface
└── cli.py          # CLI: ingest / query / chat / stats

eval/
├── build_dataset.py  # Regenerate the labeled eval set
├── run_eval.py       # Offline baseline-vs-optimized report
└── dataset.jsonl     # 210 labeled items

tests/
├── test_chunking.py
└── test_agent.py

data/sample_docs/   # Synthetic YC-3300 datasheets for the demo
```

## HTTP API

```bash
uvicorn rag_agent.api:app --reload
```

| Method | Path | Body |
|--------|------|------|
| GET | `/healthz` | — |
| POST | `/ask` | `{"question": "...", "history": [{"role":"user","content":"..."}]}` |
| POST | `/ingest` | `{"path": "./data/docs", "strategy": "fixed"}` |
| GET | `/stats` | — |

## Evaluation

```bash
python eval/build_dataset.py     # regenerate dataset.jsonl (210 items)
python eval/run_eval.py          # baseline vs optimized, offline
```

`run_eval.py` compares a weak config (fixed chunks, no overlap, top_k=3) against
the tuned one (sentence chunks, overlap=128, top_k=5) and writes `eval/report.json`
with three metrics: retrieval hit rate, multi-hop source completeness, and
no-answer refusal rate. No API key needed — it only exercises the retrieval path.

## Testing

```bash
pytest tests/ -v
```

Unit tests cover chunking edge cases (overlap, CJK punctuation, empty input), config validation, and agent input validation. No network calls in tests.

## Design notes

- **Why fixed-size chunking is the default:** chip datasheets mix tables, parameter lists and prose. Sentence splitters mis-handle table cells; fixed-size with overlap is the robust baseline. `--strategy sentence` is available for prose-heavy docs.
- **Why cosine distance:** embeddings are compared directionally, not by magnitude — standard for semantic retrieval.
- **Why tool-use retrieval:** when the initial top-k misses the answer, the model can issue a refined search instead of hallucinating.
- **Swapping embedders:** `vectorstore.py` is the only module that touches embedding functions. Move to an API-based embedder by changing that one file.

## License

MIT
