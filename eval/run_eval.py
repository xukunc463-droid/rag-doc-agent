"""Offline retrieval evaluation for rag-doc-agent.

Measures three metrics on the labeled set in eval/dataset.jsonl:

1. hit_rate                — gold source appears in top-k retrieved chunks
2. multi_hop_completeness  — all gold sources retrieved for multi-hop items
3. no_answer_refusal       — no-answer items where retrieval returns nothing
                             above the score floor (proxy for the refusal path)

Runs fully offline — no Anthropic API calls. Compares two configs so the
baseline-vs-optimized delta is reproducible.

Usage:
    python eval/run_eval.py
    python eval/run_eval.py --dataset eval/dataset.jsonl --out eval/report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag_agent.chunking import build_chunks  # noqa: E402
from rag_agent.vectorstore import VectorStore  # noqa: E402

DEFAULT_DATASET = ROOT / "eval" / "dataset.jsonl"
DEFAULT_CORPUS = ROOT / "data" / "sample_docs"
DEFAULT_OUT = ROOT / "eval" / "report.json"

# Cosine similarity floor: below this we treat a hit as "no evidence".
SCORE_FLOOR = 0.35


@dataclass(frozen=True)
class EvalConfig:
    name: str
    strategy: str
    chunk_size: int
    chunk_overlap: int
    top_k: int


BASELINE = EvalConfig(
    name="baseline",
    strategy="fixed",
    chunk_size=512,
    chunk_overlap=0,
    top_k=3,
)

OPTIMIZED = EvalConfig(
    name="optimized",
    strategy="sentence",
    chunk_size=1024,
    chunk_overlap=128,
    top_k=5,
)


@dataclass(frozen=True)
class EvalResult:
    config: str
    n_factual: int
    n_multi_hop: int
    n_no_answer: int
    hit_rate: float
    multi_hop_completeness: float
    no_answer_correct_refusal: float


def load_dataset(path: Path) -> list[dict]:
    items: list[dict] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            items.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_no} invalid JSON: {exc}") from exc
    return items


def load_corpus(corpus_dir: Path) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for p in sorted(corpus_dir.iterdir()):
        if p.suffix.lower() in {".md", ".txt"} and p.is_file():
            pairs.append((p.name, p.read_text(encoding="utf-8")))
    return pairs


def build_index(
    corpus: list[tuple[str, str]],
    db_path: Path,
    collection: str,
    strategy: str,
    chunk_size: int,
    chunk_overlap: int,
) -> VectorStore:
    store = VectorStore(db_path, collection)
    store.reset()
    for source, text in corpus:
        chunks = build_chunks(
            text,
            source=source,
            page=1,
            chunk_size=chunk_size,
            overlap=chunk_overlap,
            strategy=strategy,
        )
        store.add_chunks(chunks)
    return store


def hit_in_topk(retrieved: list, gold_sources: list[str]) -> bool:
    gold = set(gold_sources)
    for h in retrieved:
        if f"{h.source}:{h.page}" in gold:
            return True
    return False


def all_gold_retrieved(retrieved: list, gold_sources: list[str]) -> bool:
    got = {f"{h.source}:{h.page}" for h in retrieved}
    return all(g in got for g in gold_sources)


def evaluate_config(
    config: EvalConfig,
    dataset: list[dict],
    corpus: list[tuple[str, str]],
    db_root: Path,
) -> EvalResult:
    store = build_index(
        corpus=corpus,
        db_path=db_root / config.name,
        collection=f"eval_{config.name}",
        strategy=config.strategy,
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
    )

    factual_hits = 0
    factual_total = 0
    multi_complete = 0
    multi_total = 0
    no_answer_refusals = 0
    no_answer_total = 0

    for item in dataset:
        qtype = item["answer_type"]
        hits = store.search(item["question"], top_k=config.top_k)

        if qtype == "factual":
            factual_total += 1
            if hit_in_topk(hits, item["gold_sources"]):
                factual_hits += 1
        elif qtype == "multi_hop":
            multi_total += 1
            if all_gold_retrieved(hits, item["gold_sources"]):
                multi_complete += 1
        elif qtype == "no_answer":
            no_answer_total += 1
            strong = [h for h in hits if h.score >= SCORE_FLOOR]
            if not strong:
                no_answer_refusals += 1

    return EvalResult(
        config=config.name,
        n_factual=factual_total,
        n_multi_hop=multi_total,
        n_no_answer=no_answer_total,
        hit_rate=round(factual_hits / factual_total, 4) if factual_total else 0.0,
        multi_hop_completeness=round(multi_complete / multi_total, 4) if multi_total else 0.0,
        no_answer_correct_refusal=round(no_answer_refusals / no_answer_total, 4)
        if no_answer_total
        else 0.0,
    )


def render_report(results: list[EvalResult]) -> str:
    lines = ["# rag-doc-agent eval report", ""]
    header = (
        f"{'config':<12} {'hit_rate':>10} {'multi_hop':>10} "
        f"{'no_ans_ref':>11} {'n_f':>5} {'n_m':>5} {'n_n':>5}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for r in results:
        lines.append(
            f"{r.config:<12} {r.hit_rate:>10.1%} {r.multi_hop_completeness:>10.1%} "
            f"{r.no_answer_correct_refusal:>11.1%} "
            f"{r.n_factual:>5} {r.n_multi_hop:>5} {r.n_no_answer:>5}"
        )
    if len(results) == 2:
        base, opt = results[0], results[1]
        lines.append("")
        lines.append(
            f"delta hit_rate       : {opt.hit_rate - base.hit_rate:+.1%}"
        )
        lines.append(
            f"delta multi_hop      : {opt.multi_hop_completeness - base.multi_hop_completeness:+.1%}"
        )
        lines.append(
            f"delta no_answer_ref  : {opt.no_answer_correct_refusal - base.no_answer_correct_refusal:+.1%}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--db-root", type=Path, default=ROOT / "data" / "eval_chroma")
    args = parser.parse_args(argv)

    if not args.dataset.is_file():
        print(f"[eval] dataset not found: {args.dataset}", file=sys.stderr)
        return 1
    if not args.corpus.is_dir():
        print(f"[eval] corpus not found: {args.corpus}", file=sys.stderr)
        return 1

    dataset = load_dataset(args.dataset)
    corpus = load_corpus(args.corpus)
    if not corpus:
        print(f"[eval] no .md/.txt files in {args.corpus}", file=sys.stderr)
        return 1

    results = [
        evaluate_config(BASELINE, dataset, corpus, args.db_root),
        evaluate_config(OPTIMIZED, dataset, corpus, args.db_root),
    ]

    print(render_report(results))

    report = {
        "n_dataset": len(dataset),
        "corpus_files": [name for name, _ in corpus],
        "score_floor": SCORE_FLOOR,
        "configs": [asdict(BASELINE), asdict(OPTIMIZED)],
        "results": [asdict(r) for r in results],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[eval] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
