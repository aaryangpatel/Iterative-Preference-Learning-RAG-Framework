"""PrefNugget TREC RAG run response models."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class RunDocument(BaseModel):
    """Document bundled inside a TREC RAG run record.

    Parameters
    ----------
    id : str
        Document key referenced by response citations.
    text : str
        Document body.
    title : str | None
        Optional title.
    url : str | None
        Optional source URL.
    metadata : dict[str, Any]
        Extra fields.
    """

    id: str
    text: str
    title: str | None = None
    url: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ResponseSentence(BaseModel):
    """One generated response sentence with citation keys.

    Parameters
    ----------
    text : str
        Response sentence text.
    citations : list[str]
        Keys into ``RagRunRecord.documents``.
    metadata : dict[str, Any]
        Extra fields.
    """

    text: str
    citations: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RunMetadata(BaseModel):
    """Run-level metadata in kiddie TREC RAG format.

    Parameters
    ----------
    team_id : str
        Team identifier.
    run_id : str
        Run identifier.
    topic_id : str
        Topic / request id.
    """

    team_id: str
    run_id: str
    topic_id: str


class RagRunRecord(BaseModel):
    """One topic's RAG response in PrefNugget kiddie format.

    Parameters
    ----------
    metadata : RunMetadata
        Run and topic ids.
    responses : list[ResponseSentence]
        Generated sentences with citation keys.
    documents : dict[str, RunDocument]
        Cited documents keyed by citation id.
    """

    metadata: RunMetadata
    responses: list[ResponseSentence] = Field(default_factory=list)
    documents: dict[str, RunDocument] = Field(default_factory=dict)

    def response_text(self) -> str:
        """Concatenate all response sentences."""
        return " ".join(response.text for response in self.responses)

    def cited_paragraphs(self) -> list[str]:
        """Return cited document texts (for docs-grading variant)."""
        paragraphs: list[str] = []
        for response in self.responses:
            for citation_key in response.citations:
                document = self.documents.get(citation_key)
                if document is not None:
                    paragraphs.append(document.text)
        return paragraphs
