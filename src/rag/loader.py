"""Knowledge-base loading.

Uses LangChain's document abstraction when LangChain is installed, and a
compatible local `Document` otherwise, so the RAG pipeline (and its tests) run
with or without the optional `ai` extra.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.utils.config import settings
from src.utils.logging import get_logger

logger = get_logger(__name__)

try:  # pragma: no cover - exercised only when the optional extra is installed
    from langchain_core.documents import Document as _LCDocument

    LANGCHAIN_AVAILABLE = True
except ImportError:  # pragma: no cover
    _LCDocument = None  # type: ignore[assignment]
    LANGCHAIN_AVAILABLE = False


@dataclass
class Document:
    """Local stand-in with the same shape as `langchain_core.documents.Document`."""

    page_content: str
    metadata: dict[str, Any] = field(default_factory=dict)


def make_document(page_content: str, metadata: dict[str, Any] | None = None) -> Any:
    metadata = metadata or {}
    if LANGCHAIN_AVAILABLE:
        return _LCDocument(page_content=page_content, metadata=metadata)
    return Document(page_content=page_content, metadata=metadata)


def load_knowledge_base(directory: str | Path | None = None) -> list[Any]:
    """Load every markdown article in the knowledge base as one document."""
    root = Path(directory) if directory else settings.knowledge_base_path
    if not root.exists():
        raise FileNotFoundError(f"Knowledge base directory not found: {root}")

    documents: list[Any] = []
    for path in sorted(root.glob("*.md")):
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            continue
        documents.append(
            make_document(
                text,
                {
                    "source": path.name,
                    "topic": path.stem,
                    "path": str(path.relative_to(settings.project_root)),
                },
            )
        )
    logger.info("knowledge_base_loaded", extra={"documents": len(documents), "dir": str(root)})
    if not documents:
        raise ValueError(f"No markdown articles found in {root}")
    return documents
