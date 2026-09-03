"""Generate cited CRUCIBLE reports from nugget banks and retrieved documents."""

from __future__ import annotations

from dataclasses import dataclass, field

from rag_framework.models.document import Document

from crucible.alignment.supported_extractor import SupportedAnswerExtractor
from crucible.loaders import build_cited_report_from_ragtime
from crucible.models.alignment import NuggetAlignmentResult
from crucible.models.citation import CitedReport
from crucible.models.nugget import CrucibleNugget, CrucibleNuggetBank


@dataclass
class ReportGeneratorConfig:
    """Configuration for CRUCIBLE report generation.

    Parameters
    ----------
    team_id : str
        Submitter team id.
    run_id : str
        Run identifier for output reports.
    use_request_context : bool
        Pass RAGTIME request fields into the extraction prompt.
    match_threshold : float
        Minimum extraction confidence for ``is_match``.
    max_doc_len : int
        Document truncation length for LLM prompts.
    """

    team_id: str = "crucible"
    run_id: str = "crucible-run"
    use_request_context: bool = True
    match_threshold: float = 0.3
    max_doc_len: int = 1000


class CrucibleReportGenerator:
    """Assemble cited reports from nugget-document alignments.

    Mimics ``crucible_document_set`` + ``crucible_documents`` from scale25-crucible.

    Parameters
    ----------
    config : ReportGeneratorConfig | None
        Generator settings.
    extractor : SupportedAnswerExtractor | None
        Alignment extractor. Built from config when omitted.
    """

    def __init__(
        self,
        config: ReportGeneratorConfig | None = None,
        extractor: SupportedAnswerExtractor | None = None,
    ) -> None:
        self.config = config or ReportGeneratorConfig()
        self._extractor = extractor or SupportedAnswerExtractor(
            match_threshold=self.config.match_threshold,
            use_request_context=self.config.use_request_context,
            max_doc_len=self.config.max_doc_len,
        )

    def align_nuggets(
        self,
        bank: CrucibleNuggetBank,
        documents: list[Document],
        request: dict | None = None,
        max_docs_per_nugget: int | None = None,
    ) -> list[NuggetAlignmentResult]:
        """Run nugget-document extraction for all pairs.

        Parameters
        ----------
        bank : CrucibleNuggetBank
            v3 nugget bank for one topic.
        documents : list[Document]
            Retrieved or reference documents.
        request : dict | None
            RAGTIME request with title/background/problem_statement.

        Returns
        -------
        list[NuggetAlignmentResult]
            Raw alignment results.
        """
        return self._extractor.nuggetize_documents(
            nuggets=bank.nuggets,
            documents=documents,
            query_id=bank.query_id,
            request=request,
            max_docs_per_nugget=max_docs_per_nugget,
        )

    def build_report(
        self,
        bank: CrucibleNuggetBank,
        alignment_results: list[NuggetAlignmentResult],
        document_lookup: dict[str, Document],
    ) -> CitedReport:
        """Convert matched alignments into a cited report.

        Parameters
        ----------
        bank : CrucibleNuggetBank
            Source nugget bank.
        alignment_results : list[NuggetAlignmentResult]
            Extraction results from ``align_nuggets``.
        document_lookup : dict[str, Document]
            doc_id -> document for citation span text.

        Returns
        -------
        CitedReport
            Report with one sentence candidate per matched alignment.
        """
        matches = [result for result in alignment_results if result.is_match and result.summary]
        matches_by_nugget: dict[str, list[NuggetAlignmentResult]] = {}
        for result in matches:
            matches_by_nugget.setdefault(result.nugget_text, []).append(result)

        sentence_dicts: list[dict] = []
        for nugget in bank.nuggets:
            question = nugget.alignment_text()
            for result in matches_by_nugget.get(question, []):
                doc_id = result.metadata.get("document.id", "")
                if not doc_id:
                    continue
                confidence = (result.confidence or 0.0) * 100.0
                sentence_dicts.append(
                    {
                        "text": result.summary,
                        "citations": {doc_id: confidence},
                        "metadata": {
                            "question": result.nugget_text,
                            "extracted_text_segment": result.extracted_text_segment,
                            "answer": result.answer,
                            "confidence": result.confidence or 0.0,
                            "reasoning": result.reasoning,
                        },
                    }
                )

        doc_records = {
            doc_id: {"text": document.text, "url": document.metadata.get("url")}
            for doc_id, document in document_lookup.items()
        }
        report = build_cited_report_from_ragtime(
            topic_id=bank.query_id,
            team_id=self.config.team_id,
            run_id=self.config.run_id,
            sentences=sentence_dicts,
            document_lookup=doc_records,
        )
        report.metadata["generator"] = self._extractor.name
        report.metadata["num_candidates"] = len(sentence_dicts)
        return report

    def generate(
        self,
        bank: CrucibleNuggetBank,
        documents: list[Document],
        request: dict | None = None,
        max_docs_per_nugget: int | None = None,
    ) -> tuple[CitedReport, list[NuggetAlignmentResult]]:
        """Align nuggets to documents and assemble a cited report.

        Parameters
        ----------
        bank : CrucibleNuggetBank
            v3 nugget bank.
        documents : list[Document]
            Documents to scan.
        request : dict | None
            Optional RAGTIME request context.

        Returns
        -------
        tuple[CitedReport, list[NuggetAlignmentResult]]
            Generated report and raw alignment results.
        """
        alignment_results = self.align_nuggets(
            bank, documents, request=request, max_docs_per_nugget=max_docs_per_nugget
        )
        document_lookup = {document.doc_id: document for document in documents}
        report = self.build_report(bank, alignment_results, document_lookup)
        return report, alignment_results
