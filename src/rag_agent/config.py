"""Runtime configuration loaded from environment / .env."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

DEFAULT_CHUNK_SIZE = 1024
DEFAULT_CHUNK_OVERLAP = 128
DEFAULT_TOP_K = 5
DEFAULT_MODEL = "claude-sonnet-4-5"
DEFAULT_DB_PATH = "./data/chroma"
DEFAULT_COLLECTION = "chip_docs"


@dataclass(frozen=True)
class Settings:
    """Immutable runtime settings. Create new instances to change values."""

    anthropic_api_key: str
    model: str
    chunk_size: int
    chunk_overlap: int
    top_k: int
    db_path: Path
    collection_name: str


class ConfigError(RuntimeError):
    """Raised when required configuration is missing or invalid."""


def load_settings(env_file: str | None = ".env") -> Settings:
    """Load settings from environment, optionally sourcing an env file first.

    Raises:
        ConfigError: if ANTHROPIC_API_KEY is missing or numeric fields are invalid.
    """
    if env_file and Path(env_file).exists():
        load_dotenv(env_file)

    api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        raise ConfigError(
            "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and fill it in."
        )

    try:
        chunk_size = int(os.getenv("RAG_CHUNK_SIZE", DEFAULT_CHUNK_SIZE))
        chunk_overlap = int(os.getenv("RAG_CHUNK_OVERLAP", DEFAULT_CHUNK_OVERLAP))
        top_k = int(os.getenv("RAG_TOP_K", DEFAULT_TOP_K))
    except ValueError as exc:
        raise ConfigError(f"Numeric config value is invalid: {exc}") from exc

    if chunk_size <= 0:
        raise ConfigError(f"RAG_CHUNK_SIZE must be > 0, got {chunk_size}")
    if chunk_overlap < 0:
        raise ConfigError(f"RAG_CHUNK_OVERLAP must be >= 0, got {chunk_overlap}")
    if chunk_overlap >= chunk_size:
        raise ConfigError(
            f"RAG_CHUNK_OVERLAP ({chunk_overlap}) must be < RAG_CHUNK_SIZE ({chunk_size})"
        )
    if top_k <= 0:
        raise ConfigError(f"RAG_TOP_K must be > 0, got {top_k}")

    return Settings(
        anthropic_api_key=api_key,
        model=os.getenv("RAG_MODEL", DEFAULT_MODEL),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        top_k=top_k,
        db_path=Path(os.getenv("RAG_DB_PATH", DEFAULT_DB_PATH)),
        collection_name=os.getenv("RAG_COLLECTION", DEFAULT_COLLECTION),
    )
