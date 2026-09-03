"""JSONL document loaders compatible with CRUCIBLE collections."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

from rag_framework.models.document import Document


def stream_documents_from_jsonl(
    path: Path,
    wanted_doc_ids: set[str] | None = None,
) -> Iterator[Document]:
    """Stream documents from a JSONL file, optionally filtering by ID.

    Parameters
    ----------
    path : Path
        Path to a JSONL collection file (one JSON object per line).
    wanted_doc_ids : set[str] | None
        If provided, only yield documents whose ``id`` is in this set.

    Yields
    ------
    Document
        Parsed documents. Malformed lines are skipped with no exception.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Collection not found: {path}")

    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue

            record = json.loads(stripped)
            doc_id = str(record.get("id") or record.get("doc_id") or "")
            if wanted_doc_ids is not None and doc_id not in wanted_doc_ids:
                continue

            if "text" not in record:
                continue

            yield Document.from_jsonl_record(record)


def load_documents_from_jsonl(
    path: Path,
    wanted_doc_ids: set[str] | None = None,
) -> list[Document]:
    """Load all documents from a JSONL file into memory.

    Parameters
    ----------
    path : Path
        Path to JSONL collection.
    wanted_doc_ids : set[str] | None
        Optional ID filter.

    Returns
    -------
    list[Document]
        All matching documents in file order.
    """
    return list(stream_documents_from_jsonl(path, wanted_doc_ids=wanted_doc_ids))
