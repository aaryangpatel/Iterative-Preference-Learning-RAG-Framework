"""Query models for retrieval."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class Query(BaseModel):
    """A retrieval query, optionally with TREC-style report-request fields.

    Parameters
    ----------
    query_id : str
        Unique query/topic identifier.
    text : str
        Primary query text sent to the retriever.
    title : str | None
        Short query title (RAGTIME ``request_title``).
    background : str | None
        User background context (RAGTIME ``background``).
    problem_statement : str | None
        Problem to address (RAGTIME ``problem_statement``).
    metadata : dict[str, Any]
        Additional fields for downstream iterative RAG stages.
    """

    query_id: str
    text: str
    title: str | None = None
    background: str | None = None
    problem_statement: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    def retrieval_text(self) -> str:
        """Build a single string for retrieval from available query fields.

        Concatenates title, background, problem statement, and primary text
        when those fields are present. This mirrors how CRUCIBLE report
        requests combine context for document lookup.

        Returns
        -------
        str
            Space-joined non-empty query components.
        """
        parts = [
            self.title,
            self.background,
            self.problem_statement,
            self.text,
        ]
        return " ".join(part.strip() for part in parts if part and part.strip())

    @classmethod
    def from_report_request(cls, record: dict[str, Any]) -> Query:
        """Build a ``Query`` from a RAGTIME-style report request JSON object.

        Parameters
        ----------
        record : dict[str, Any]
            Parsed report request with ``topic_id`` or ``query_id``.

        Returns
        -------
        Query
            Query with RAGTIME field mapping applied.
        """
        query_id = str(record.get("topic_id") or record.get("query_id") or record.get("id"))
        title = record.get("request_title") or record.get("title")
        background = record.get("background")
        problem = record.get("problem_statement") or record.get("problem")

        text = record.get("query") or record.get("text") or problem or title or ""
        if not text:
            raise ValueError(f"Query {query_id} has no searchable text")

        return cls(
            query_id=query_id,
            text=text,
            title=title,
            background=background,
            problem_statement=problem,
            metadata={k: v for k, v in record.items() if k not in {
                "topic_id", "query_id", "id", "request_title", "title",
                "background", "problem_statement", "problem", "query", "text",
            }},
        )
