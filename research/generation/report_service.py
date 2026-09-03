"""CRUCIBLE report generation with configurable profiles."""

from __future__ import annotations

from crucible.alignment.supported_extractor import SupportedAnswerExtractor
from crucible.models.nugget import CrucibleNuggetBank
from crucible.models.report_output import CrucibleReportBundle
from crucible.pipeline.refinement import RefinementConfig, ReportRefinement
from crucible.pipeline.report_generator import CrucibleReportGenerator, ReportGeneratorConfig
from rag_framework.models.document import Document
from rag_framework.models.query import Query

from research.config import GenerationProfile


class ReportGenerationService:
    """Generate cited reports from a nugget bank, documents, and generation profile.

    Parameters
    ----------
    team_id : str
        CRUCIBLE team id.
    char_limit : int
        Default report character limit.
    max_docs_per_nugget : int
        Documents scanned per nugget during alignment.
    """

    def __init__(
        self,
        team_id: str = "research",
        char_limit: int = 2000,
        max_docs_per_nugget: int = 5,
    ) -> None:
        self._team_id = team_id
        self._char_limit = char_limit
        self._max_docs_per_nugget = max_docs_per_nugget

    def generate(
        self,
        query: Query,
        nugget_bank: CrucibleNuggetBank,
        documents: list[Document],
        profile: GenerationProfile,
        run_id: str,
        request: dict | None = None,
    ) -> CrucibleReportBundle:
        """Build one cited report variant.

        Parameters
        ----------
        query : Query
            User query.
        nugget_bank : CrucibleNuggetBank
            Nuggets guiding extraction.
        documents : list[Document]
            Evidence documents.
        profile : GenerationProfile
            Generation/refinement variant settings.
        run_id : str
            Unique run identifier for this report.
        request : dict | None
            Optional RAGTIME request dict for extraction context.

        Returns
        -------
        CrucibleReportBundle
            Complete report bundle with query and sources attached.
        """
        request_payload = request or {
            "request_id": query.query_id,
            "title": query.title,
            "background": query.background,
            "problem_statement": query.problem_statement,
        }

        extractor = SupportedAnswerExtractor(
            match_threshold=profile.match_threshold,
            use_request_context=profile.use_request_context,
        )
        generator = CrucibleReportGenerator(
            config=ReportGeneratorConfig(team_id=self._team_id, run_id=run_id),
            extractor=extractor,
        )
        raw_report, alignments = generator.generate(
            nugget_bank,
            documents,
            request=request_payload,
            max_docs_per_nugget=self._max_docs_per_nugget,
        )

        if profile.extractive:
            raw_report = self._apply_extractive_mode(raw_report)

        report = raw_report
        if profile.refine:
            refiner = ReportRefinement(
                config=RefinementConfig(
                    char_limit=self._char_limit,
                    confidence_threshold=profile.confidence_threshold,
                    sentences_per_nugget=profile.sentences_per_nugget,
                    prefer_shorter=profile.prefer_shorter,
                    filter_citations=profile.filter_citations,
                    filter_nugget_coverage=profile.filter_nugget_coverage,
                )
            )
            report = refiner.refine(raw_report, nugget_bank)

        report.collection_id = "research"
        report.metadata.update(
            {
                "generation_profile": profile.name,
                "query_text": query.retrieval_text(),
            }
        )

        document_lookup = {document.doc_id: document for document in documents}
        for sentence in report.sentences:
            for citation in sentence.citations:
                source = document_lookup.get(citation.span.doc_id)
                if source is not None:
                    citation.span.metadata.setdefault("url", source.metadata.get("url"))
                    citation.method = f"align_{profile.name}"

        return CrucibleReportBundle(
            query=query,
            report=report,
            source_documents=documents,
            nugget_bank=nugget_bank,
            metadata={
                "profile": profile.name,
                "run_id": run_id,
                "num_alignments": len(alignments),
                "num_matches": sum(1 for result in alignments if result.is_match),
            },
        )

    @staticmethod
    def _apply_extractive_mode(report):
        """Replace abstractive summaries with extracted source segments."""
        from crucible.models.citation import CitedReport

        updated_sentences = []
        for sentence in report.sentences:
            segment = sentence.metadata.get("extracted_text_segment")
            if segment:
                updated_sentences.append(sentence.model_copy(update={"text": str(segment)}))
            else:
                updated_sentences.append(sentence)
        updated = report.model_copy(update={"sentences": updated_sentences})
        return updated
