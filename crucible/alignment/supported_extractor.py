"""Extract supporting sentences from documents for CRUCIBLE report generation."""

from __future__ import annotations

from rag_framework.llm.client import OpenRouterLLM, get_llm_client
from rag_framework.models.document import Document

from crucible.models.alignment import NuggetAlignmentResult
from crucible.models.nugget import CrucibleNugget

_SUPPORTED_SYSTEM = """Given a question and answer, find sections in the source document that support
and validate the answer. If the document does not support the answer, set confidence to 0.0 and
is_match to false.

Provide the supporting text segment using complete sentences directly from the document.
Then condense the extracted text into one concise summary sentence that answers the question
without referring to the source document.

Respond with JSON only:
{"extracted_text_segment": "...", "summary": "...", "confidence": 0.0 to 1.0,
 "is_match": true/false, "reasoning": "..."}"""

_SUPPORTED_REQUEST_SYSTEM = """Given a question and answer on the given topic/background/problem statement,
find sections in the source document that support and validate the answer. If the document does not
support the answer, set confidence to 0.0 and is_match to false.

Provide the supporting text segment using complete sentences directly from the document.
Then condense the extracted text into one concise summary sentence that answers the question
without referring to the source document.

Respond with JSON only:
{"extracted_text_segment": "...", "summary": "...", "confidence": 0.0 to 1.0,
 "is_match": true/false, "reasoning": "..."}"""


class SupportedAnswerExtractor:
    """Align nuggets to documents and extract cited report sentences.

    Mimics CRUCIBLE ``SupportedAnswerExtractor`` / ``SupportedAnswerExtractorAll``.

    Parameters
    ----------
    match_threshold : float
        Minimum confidence for ``is_match=True``.
    llm : OpenRouterLLM | None
        LLM client. Defaults to environment-configured OpenRouter client.
    use_request_context : bool
        When True, include title/background/problem in the prompt.
    max_doc_len : int
        Truncate document text to this many characters.
    """

    def __init__(
        self,
        match_threshold: float = 0.3,
        llm: OpenRouterLLM | None = None,
        use_request_context: bool = False,
        max_doc_len: int = 1000,
    ) -> None:
        self._match_threshold = match_threshold
        self._llm = llm or get_llm_client()
        self._use_request_context = use_request_context
        self._max_doc_len = max_doc_len

    @property
    def name(self) -> str:
        suffix = "request" if self._use_request_context else "all"
        return f"crucible_llm_supported_extractor_{suffix}"

    def extract(
        self,
        nugget: CrucibleNugget,
        document: Document,
        query_id: str | None = None,
        request: dict | None = None,
    ) -> NuggetAlignmentResult:
        """Extract a supporting segment and summary for one nugget-document pair.

        Parameters
        ----------
        nugget : CrucibleNugget
            Nugget with question and gold answers.
        document : Document
            Source document to scan.
        query_id : str | None
            Optional topic id.
        request : dict | None
            RAGTIME request with title, background, problem_statement.

        Returns
        -------
        NuggetAlignmentResult
            Extraction result with summary sentence and citation metadata.
        """
        answer = next(iter(nugget.answers.values())).answer if nugget.answers else nugget.text
        doc_text = document.text[: self._max_doc_len]
        system = _SUPPORTED_REQUEST_SYSTEM if self._use_request_context else _SUPPORTED_SYSTEM

        user_parts = [
            f"question: {nugget.alignment_text()}",
            f"answer: {answer}",
            f"source_document:\n{doc_text}",
        ]
        if self._use_request_context and request:
            user_parts.insert(
                0,
                (
                    f"title_query: {request.get('title', '')}\n"
                    f"background: {request.get('background', '')}\n"
                    f"problem_statement: {request.get('problem_statement', '')}"
                ),
            )
        payload: dict | list
        try:
            payload = self._llm.complete_json(system=system, user="\n\n".join(user_parts))
        except ValueError:
            return NuggetAlignmentResult(
                nugget_text=nugget.alignment_text(),
                nugget_id=nugget.nugget_id,
                query_id=query_id,
                source_document=doc_text,
                is_match=False,
                confidence=0.0,
                answer=answer,
                metadata={
                    "aligner": self.name,
                    "document.id": document.doc_id,
                    "llm_model": self._llm.model,
                    "parse_error": True,
                },
            )
        if not isinstance(payload, dict):
            payload = {}

        segment = str(payload.get("extracted_text_segment", "") or "").strip()
        summary = str(payload.get("summary", "") or "").strip()
        confidence = float(payload.get("confidence", 0.0) or 0.0)
        is_match = bool(payload.get("is_match", confidence >= self._match_threshold))
        if confidence < self._match_threshold:
            is_match = False

        return NuggetAlignmentResult(
            nugget_text=nugget.alignment_text(),
            nugget_id=nugget.nugget_id,
            query_id=query_id,
            source_document=doc_text,
            extracted_text_segment=segment or None,
            sentence=summary or segment or None,
            summary=summary or None,
            answer=answer,
            is_match=is_match,
            match_score=confidence,
            confidence=confidence,
            reasoning=str(payload.get("reasoning", "") or ""),
            metadata={
                "aligner": self.name,
                "document.id": document.doc_id,
                "llm_model": self._llm.model,
            },
        )

    def nuggetize_documents(
        self,
        nuggets: list[CrucibleNugget],
        documents: list[Document],
        query_id: str | None = None,
        request: dict | None = None,
        max_docs_per_nugget: int | None = None,
    ) -> list[NuggetAlignmentResult]:
        """Run extraction for nugget-document pairs.

        Parameters
        ----------
        nuggets : list[CrucibleNugget]
            Nuggets to align.
        documents : list[Document]
            Documents to scan.
        query_id : str | None
            Topic id.
        request : dict | None
            Optional RAGTIME request context.
        max_docs_per_nugget : int | None
            Limit documents scanned per nugget (uses top-ranked docs first).

        Returns
        -------
        list[NuggetAlignmentResult]
            All alignment results (matches and non-matches).
        """
        doc_pool = documents
        if max_docs_per_nugget is not None:
            doc_pool = documents[:max_docs_per_nugget]

        results: list[NuggetAlignmentResult] = []
        for nugget in nuggets:
            for document in doc_pool:
                results.append(
                    self.extract(
                        nugget=nugget,
                        document=document,
                        query_id=query_id,
                        request=request,
                    )
                )
        return results
