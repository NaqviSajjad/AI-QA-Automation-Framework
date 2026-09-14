"""Chunking.

Markdown support articles are structured by heading, so chunks are cut on
heading boundaries first and only then on size. That keeps a policy statement
and its conditions in the same chunk - the single biggest driver of context
precision in this corpus.
"""

from __future__ import annotations

import re
from typing import Any

from src.rag.loader import make_document
from src.utils.config import settings

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)


def _split_by_heading(text: str) -> list[tuple[str, str]]:
    """Return [(heading, section_text)] preserving document order."""
    matches = list(HEADING_RE.finditer(text))
    if not matches:
        return [("", text)]

    sections: list[tuple[str, str]] = []
    preamble = text[: matches[0].start()].strip()
    if preamble:
        sections.append(("", preamble))
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        heading = match.group(2).strip()
        body = text[match.end() : end].strip()
        if body:
            sections.append((heading, body))
    return sections


def _window(text: str, size: int, overlap: int) -> list[str]:
    """Sentence-aware sliding window."""
    if len(text) <= size:
        return [text]
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        if current and len(current) + len(sentence) + 1 > size:
            chunks.append(current.strip())
            tail = current[-overlap:] if overlap else ""
            current = f"{tail} {sentence}".strip()
        else:
            current = f"{current} {sentence}".strip()
    if current.strip():
        chunks.append(current.strip())
    return chunks


def chunk_documents(
    documents: list[Any],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Any]:
    size = chunk_size or settings.rag.chunk_size
    overlap = chunk_overlap or settings.rag.chunk_overlap
    if overlap >= size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    chunks: list[Any] = []
    for document in documents:
        for heading, section in _split_by_heading(document.page_content):
            for position, piece in enumerate(_window(section, size, overlap)):
                # The heading is prefixed as its own sentence so it contributes
                # to the embedding without being mistaken for prose downstream.
                header = f"{heading.rstrip('.')}. " if heading else ""
                metadata = dict(document.metadata)
                metadata.update(
                    {
                        "heading": heading,
                        "chunk_index": len(chunks),
                        "section_part": position,
                    }
                )
                chunks.append(make_document(f"{header}{piece}".strip(), metadata))
    return chunks
