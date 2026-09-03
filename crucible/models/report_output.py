"""Full CRUCIBLE report output including query and source documents."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from crucible.models.citation import CitedReport
from crucible.models.nugget import CrucibleNuggetBank
from rag_framework.models.document import Document
from rag_framework.models.query import Query


class CrucibleReportBundle(BaseModel):
    """Complete report artifact: initial query, sources, nuggets, and cited report.

    Parameters
    ----------
    query : Query
        Original user query / RAGTIME request fields.
    report : CitedReport
        Generated cited report sentences.
    source_documents : list[Document]
        RAGTIME (or other) documents used for generation.
    nugget_bank : CrucibleNuggetBank | None
        Nugget bank guiding extraction.
    metadata : dict[str, Any]
        Pipeline configuration and provenance.
    """

    query: Query
    report: CitedReport
    source_documents: list[Document] = Field(default_factory=list)
    nugget_bank: CrucibleNuggetBank | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    def report_text(self) -> str:
        """Concatenate all cited report sentences."""
        return " ".join(sentence.text for sentence in self.report.sentences)

    def to_export_dict(self) -> dict[str, Any]:
        """Serialize bundle for JSON export with query, citations, and text."""
        citations = []
        for sentence in self.report.sentences:
            for citation in sentence.citations:
                citations.append(
                    {
                        "sentence_id": sentence.sentence_id,
                        "sentence_text": sentence.text,
                        "doc_id": citation.span.doc_id,
                        "source_url": citation.span.metadata.get("url"),
                        "source_passage": citation.span.text,
                        "confidence": citation.confidence,
                    }
                )

        return {
            "query": self.query.model_dump(),
            "report_text": self.report_text(),
            "report": self.report.model_dump(),
            "citations": citations,
            "source_documents": [document.model_dump() for document in self.source_documents],
            "nugget_bank": self.nugget_bank.to_v3_dict() if self.nugget_bank else None,
            "metadata": self.metadata,
        }
