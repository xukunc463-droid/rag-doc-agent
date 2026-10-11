"""Command-line entry point for the RAG document agent.

Usage:
    python -m rag_agent.cli ingest --path ./data
    python -m rag_agent.cli query "What is the output voltage range?"
    python -m rag_agent.cli chat
    python -m rag_agent.cli stats
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from rag_agent.agent import AgentError, RagAgent
from rag_agent.config import ConfigError, Settings, load_settings
from rag_agent.ingest import IngestError, ingest_directory, ingest_file
from rag_agent.vectorstore import VectorStore

console = Console()


def _build_store(settings: Settings) -> VectorStore:
    return VectorStore(settings.db_path, settings.collection_name)


def cmd_ingest(args: argparse.Namespace, settings: Settings) -> int:
    store = _build_store(settings)
    target = Path(args.path)

    try:
        if target.is_dir():
            results = ingest_directory(target, settings, store, strategy=args.strategy)
        else:
            count = ingest_file(target, settings, store, strategy=args.strategy)
            results = {target.name: count}
    except IngestError as exc:
        console.print(f"[red]Ingest failed:[/red] {exc}")
        return 1

    table = Table(title="Ingest complete")
    table.add_column("File")
    table.add_column("Chunks", justify="right")
    total = 0
    for name, count in results.items():
        table.add_row(name, str(count))
        total += count
    table.add_row("[bold]Total[/bold]", f"[bold]{total}[/bold]")
    console.print(table)
    console.print(f"Vector store now holds [cyan]{store.count()}[/cyan] chunks.")
    return 0


def cmd_query(args: argparse.Namespace, settings: Settings) -> int:
    store = _build_store(settings)
    if store.count() == 0:
        console.print("[yellow]Vector store is empty. Run `ingest` first.[/yellow]")
        return 1

    agent = RagAgent(settings, store)
    try:
        answer = agent.ask(args.question)
    except AgentError as exc:
        console.print(f"[red]Agent error:[/red] {exc}")
        return 1

    console.print(Panel(answer.text, title="Answer", border_style="green"))
    if answer.sources:
        table = Table(title="Sources")
        table.add_column("File")
        table.add_column("Page", justify="right")
        table.add_column("Score", justify="right")
        for hit in answer.sources:
            table.add_row(hit.source, str(hit.page), f"{hit.score:.3f}")
        console.print(table)
    return 0


def cmd_chat(args: argparse.Namespace, settings: Settings) -> int:
    store = _build_store(settings)
    if store.count() == 0:
        console.print("[yellow]Vector store is empty. Run `ingest` first.[/yellow]")
        return 1

    agent = RagAgent(settings, store)
    history: list[dict[str, str]] = []
    console.print("[bold]Chat mode[/bold] — type `exit` to quit.\n")

    while True:
        try:
            question = console.input("[bold blue]you>[/bold blue] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\nbye.")
            return 0

        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            console.print("bye.")
            return 0

        try:
            answer = agent.ask(question, history=history)
        except AgentError as exc:
            console.print(f"[red]Agent error:[/red] {exc}")
            continue

        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": answer.text})

        console.print(Panel(answer.text, title="agent", border_style="green"))
        for hit in answer.sources:
            console.print(f"  [dim]- {hit.source} p.{hit.page} ({hit.score:.3f})[/dim]")
        console.print()


def cmd_stats(args: argparse.Namespace, settings: Settings) -> int:
    store = _build_store(settings)
    console.print(f"Collection:    [cyan]{settings.collection_name}[/cyan]")
    console.print(f"DB path:       [cyan]{settings.db_path}[/cyan]")
    console.print(f"Chunks:        [cyan]{store.count()}[/cyan]")
    console.print(f"Chunk size:    [cyan]{settings.chunk_size}[/cyan]")
    console.print(f"Chunk overlap: [cyan]{settings.chunk_overlap}[/cyan]")
    console.print(f"Top-K:         [cyan]{settings.top_k}[/cyan]")
    console.print(f"Model:         [cyan]{settings.model}[/cyan]")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag_agent",
        description="RAG knowledge-base agent for technical documents.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="Ingest a PDF/txt file or a directory of them")
    p_ingest.add_argument("--path", required=True, help="File or directory to ingest")
    p_ingest.add_argument(
        "--strategy",
        choices=["fixed", "sentence"],
        default="fixed",
        help="Chunking strategy (default: fixed)",
    )
    p_ingest.set_defaults(func=cmd_ingest)

    p_query = sub.add_parser("query", help="Ask one question and exit")
    p_query.add_argument("question", help="Question to answer")
    p_query.set_defaults(func=cmd_query)

    p_chat = sub.add_parser("chat", help="Interactive multi-turn chat")
    p_chat.set_defaults(func=cmd_chat)

    p_stats = sub.add_parser("stats", help="Show vector store and config stats")
    p_stats.set_defaults(func=cmd_stats)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        settings = load_settings()
    except ConfigError as exc:
        console.print(f"[red]Config error:[/red] {exc}")
        return 2

    return args.func(args, settings)


if __name__ == "__main__":
    sys.exit(main())
