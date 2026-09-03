"""Multi-round preference-learning loop for CRUCIBLE + PrefNugget."""

from __future__ import annotations

from typing import Any

from crucible.models.nugget import CrucibleNuggetBank
from crucible.models.report_output import CrucibleReportBundle
from crucible.nuggets.request_extractor import RequestNuggetExtractor
from prefnugget.judges.extract_contrastive import ContrastiveNuggetExtractor
from prefnugget.models.nugget import NuggetQuestionBank
from rag_framework.llm.client import OpenRouterLLM, get_llm_client
from rag_framework.models.document import Document
from rag_framework.models.query import Query

from research.adapters import (
    bundle_to_rag_run,
    make_run_id,
    merge_crucible_banks,
    query_to_topic,
    question_bank_to_crucible_bank,
)
from research.config import ExperimentConfig, GenerationProfile, DEFAULT_SYNTHESIS_PROFILE
from research.convergence import ConvergenceChecker
from research.document_source import DocumentSource
from research.generation.report_service import ReportGenerationService
from research.judges.report_pairwise import ReportPairwiseJudge
from research.models import ExperimentResult, RoundResult
from research.storage import ExperimentStorage


class PreferenceLearningLoop:
    """Iterative CRUCIBLE report improvement via PrefNugget contrastive nuggets.

    Pipeline:
        Round 0 — generate two reports with different profiles, judge, extract contrastive nuggets.
        Round 1+ — merge base + contrastive nuggets, generate one improved challenger report,
                   judge vs the prior round's champion, extract new contrastive nuggets,
                   repeat until convergence or ``max_rounds``.

    Parameters
    ----------
    config : ExperimentConfig
        Experiment configuration.
    llm : OpenRouterLLM | None
        Shared LLM client.
    """

    def __init__(
        self,
        config: ExperimentConfig,
        llm: OpenRouterLLM | None = None,
    ) -> None:
        self.config = config
        self._llm = llm or get_llm_client()
        self._document_source = DocumentSource(config.document_source)
        self._report_service = ReportGenerationService(
            team_id=config.team_id,
            char_limit=config.char_limit,
            max_docs_per_nugget=config.document_source.max_docs_per_nugget,
        )
        self._nugget_extractor = RequestNuggetExtractor(max_nuggets=config.base_nuggets_count)
        self._contrastive_extractor = ContrastiveNuggetExtractor(
            max_questions=config.max_contrastive_nuggets,
            max_per_pair=3,
            llm=self._llm,
        )
        self._report_judge = ReportPairwiseJudge(llm=self._llm)
        self._convergence = ConvergenceChecker(config.iteration)
        self._storage = ExperimentStorage(config)

    def run(
        self,
        query: Query,
        request: dict | None = None,
        base_nugget_bank: CrucibleNuggetBank | None = None,
        documents: list[Document] | None = None,
    ) -> ExperimentResult:
        """Run the full preference-learning loop for one query.

        Parameters
        ----------
        query : Query
            User query with title/background/problem fields.
        request : dict | None
            Optional RAGTIME-style request dict for nugget extraction.
        base_nugget_bank : CrucibleNuggetBank | None
            Optional pre-built nugget bank (auto-generated when omitted).
        documents : list[Document] | None
            Optional pre-fetched documents (skips remote retrieval when provided).

        Returns
        -------
        ExperimentResult
            All round results and the final champion report.
        """
        request_payload = request or self._query_to_request(query)
        if documents is None:
            documents = self._document_source.fetch(query)
        topic = query_to_topic(query)

        base_bank = base_nugget_bank
        if base_bank is None:
            base_bank = self._nugget_extractor.extract(request_payload)
        base_bank.title_query = query.title

        profile_a, profile_b = self._initial_profiles()
        report_a = self._report_service.generate(
            query=query,
            nugget_bank=base_bank,
            documents=documents,
            profile=profile_a,
            run_id=make_run_id(self.config.experiment_id, 0, profile_a.run_id_suffix),
            request=request_payload,
        )
        report_b = self._report_service.generate(
            query=query,
            nugget_bank=base_bank,
            documents=documents,
            profile=profile_b,
            run_id=make_run_id(self.config.experiment_id, 0, profile_b.run_id_suffix),
            request=request_payload,
        )

        preference = self._report_judge.judge(
            topic,
            bundle_to_rag_run(report_a),
            bundle_to_rag_run(report_b),
        )
        champion, runner_up = self._resolve_pair(preference, report_a, report_b)

        contrastive_bank = self._contrastive_extractor.extract(
            topic,
            bundle_to_rag_run(champion),
            bundle_to_rag_run(runner_up),
        )
        merged_bank = self._build_merged_bank(base_bank, contrastive_bank, query.title)
        new_nuggets_count = len(contrastive_bank.questions)

        stop = False
        stop_reason = ""
        stop, stop_reason = self._convergence.should_stop(0, champion.report.run_id, new_nuggets_count)
        round0 = RoundResult(
            round_index=0,
            candidate_bundles=[report_a, report_b],
            champion=champion,
            runner_up=runner_up,
            preference=preference,
            merged_nugget_bank=merged_bank,
            contrastive_nugget_bank=contrastive_bank,
            new_nuggets_count=new_nuggets_count,
            converged=stop,
            metadata={
                "phase": "initial_dual_reports",
                "profile_a": profile_a.name,
                "profile_b": profile_b.name,
                "stop_reason": stop_reason if stop else None,
            },
        )
        rounds = [round0]
        self._storage.save_round(round0)

        for round_index in range(1, self.config.iteration.max_rounds):
            if stop:
                break

            champion_before = champion
            previous_count = len(contrastive_bank.questions)
            contrastive_bank = self._contrastive_extractor.extract(
                topic,
                bundle_to_rag_run(champion),
                bundle_to_rag_run(runner_up),
                existing=contrastive_bank,
            )
            new_nuggets_count = len(contrastive_bank.questions) - previous_count
            merged_bank = self._build_merged_bank(base_bank, contrastive_bank, query.title)

            improved_profile = self.config.improved_profile
            challenger = self._report_service.generate(
                query=query,
                nugget_bank=merged_bank,
                documents=documents,
                profile=improved_profile,
                run_id=make_run_id(
                    self.config.experiment_id,
                    round_index,
                    improved_profile.run_id_suffix,
                ),
                request=request_payload,
            )

            preference = self._report_judge.judge(
                topic,
                bundle_to_rag_run(challenger),
                bundle_to_rag_run(champion_before),
            )
            if self._should_adopt_challenger(challenger, champion_before, preference):
                runner_up = champion_before
                champion = challenger
            else:
                runner_up = challenger

            stop, stop_reason = self._convergence.should_stop(
                round_index,
                champion.report.run_id,
                new_nuggets_count,
            )
            round_result = RoundResult(
                round_index=round_index,
                candidate_bundles=[champion_before, challenger],
                champion=champion,
                runner_up=runner_up,
                challenger=challenger,
                preference=preference,
                merged_nugget_bank=merged_bank,
                contrastive_nugget_bank=contrastive_bank,
                new_nuggets_count=new_nuggets_count,
                converged=stop,
                metadata={
                    "phase": "single_challenger",
                    "improved_profile": improved_profile.name,
                    "stop_reason": stop_reason if stop else None,
                    "challenger_run_id": challenger.report.run_id,
                    "contrastive_questions": [question.text for question in contrastive_bank.questions],
                },
            )
            rounds.append(round_result)
            self._storage.save_round(round_result)

        merged_bank = self._build_merged_bank(base_bank, contrastive_bank, query.title)
        if self.config.iteration.enable_final_synthesis:
            synthesis = self._report_service.generate(
                query=query,
                nugget_bank=merged_bank,
                documents=documents,
                profile=DEFAULT_SYNTHESIS_PROFILE,
                run_id=make_run_id(self.config.experiment_id, len(rounds), DEFAULT_SYNTHESIS_PROFILE.run_id_suffix),
                request=request_payload,
            )
            candidates = self._collect_candidates(rounds, synthesis)
            final_champion = self._select_best_candidate(candidates)
            selection_mode = "structural_with_synthesis"
            selected_run_id = final_champion.report.run_id
            candidate_run_ids = [bundle.report.run_id for bundle in candidates]
        else:
            final_champion = champion
            selection_mode = "round_champion"
            selected_run_id = champion.report.run_id
            candidate_run_ids = [champion.report.run_id]

        experiment = ExperimentResult(
            experiment_id=self.config.experiment_id,
            query_id=query.query_id,
            rounds=rounds,
            final_champion=final_champion,
            total_rounds=len(rounds),
            converged=stop,
            metadata={
                "stop_reason": stop_reason if stop else "incomplete",
                "document_provider": self.config.document_source.provider,
                "num_documents": len(documents),
                "base_nuggets_count": len(base_bank.nuggets),
                "contrastive_nuggets_count": len(contrastive_bank.questions),
                "final_selection": selection_mode,
                "candidate_run_ids": candidate_run_ids,
                "selected_run_id": selected_run_id,
            },
        )
        self._storage.save_experiment(experiment)
        return experiment

    def _initial_profiles(self) -> tuple[GenerationProfile, GenerationProfile]:
        profiles = list(self.config.generation_profiles)
        if len(profiles) < 2:
            profiles.append(GenerationProfile(name="extractive", run_id_suffix="b", extractive=True))
        return profiles[0], profiles[1]

    @staticmethod
    def _collect_candidates(
        rounds: list[RoundResult],
        synthesis: CrucibleReportBundle,
    ) -> list[CrucibleReportBundle]:
        seen: set[str] = set()
        candidates: list[CrucibleReportBundle] = []
        for round_result in rounds:
            for bundle in round_result.candidate_bundles:
                run_id = bundle.report.run_id
                if run_id not in seen:
                    seen.add(run_id)
                    candidates.append(bundle)
            if round_result.challenger is not None:
                run_id = round_result.challenger.report.run_id
                if run_id not in seen:
                    seen.add(run_id)
                    candidates.append(round_result.challenger)
        synthesis_id = synthesis.report.run_id
        if synthesis_id not in seen:
            candidates.append(synthesis)
        return candidates

    @staticmethod
    def _report_quality_score(bundle: CrucibleReportBundle) -> tuple[float, float, float, float]:
        """Rank reports by nugget coverage, valid citations, and length (higher is better)."""
        report = bundle.report
        valid_doc_ids = {document.doc_id for document in bundle.source_documents}
        questions: set[str] = set()
        cited_sentences = 0
        valid_cited = 0
        for sentence in report.sentences:
            question = str(sentence.metadata.get("question", "")).strip()
            if question:
                questions.add(question)
            if sentence.citations:
                cited_sentences += 1
                citation_doc_ids = {citation.span.doc_id for citation in sentence.citations}
                if citation_doc_ids.issubset(valid_doc_ids):
                    valid_cited += 1
        text_len = min(len(bundle.report_text()), 2000)
        return (
            float(len(questions)),
            float(valid_cited),
            float(cited_sentences),
            float(text_len),
        )

    def _should_adopt_challenger(self, challenger, champion, preference) -> bool:
        """Adopt challenger when the judge prefers it or structural quality is higher."""
        challenger_score = self._report_quality_score(challenger)
        champion_score = self._report_quality_score(champion)
        if preference is not None and preference.winner_run_id == challenger.report.run_id:
            return True
        if preference is not None and preference.winner_run_id == champion.report.run_id:
            if challenger_score > champion_score:
                return True
            return False
        return challenger_score > champion_score

    def _select_best_candidate(
        self,
        candidates: list[CrucibleReportBundle],
    ) -> CrucibleReportBundle:
        if not candidates:
            raise ValueError("No candidate reports available for final selection")
        return max(candidates, key=self._report_quality_score)

    @staticmethod
    def _resolve_pair(preference, first, second):
        if preference is not None and preference.winner_run_id == second.report.run_id:
            return second, first
        if preference is not None and preference.winner_run_id == first.report.run_id:
            return first, second
        return first, second

    @staticmethod
    def _build_merged_bank(
        base_bank: CrucibleNuggetBank,
        contrastive_bank: NuggetQuestionBank,
        title_query: str | None,
    ) -> CrucibleNuggetBank:
        contrastive_crucible = question_bank_to_crucible_bank(contrastive_bank, title_query=title_query)
        return merge_crucible_banks(base_bank, contrastive_crucible)

    @staticmethod
    def _query_to_request(query: Query) -> dict[str, Any]:
        return {
            "request_id": query.query_id,
            "title": query.title,
            "background": query.background,
            "problem_statement": query.problem_statement,
        }
