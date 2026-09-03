"""Benchmark system identifiers and generation wrappers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from crucible.models.report_output import CrucibleReportBundle
from crucible.nuggets.request_extractor import RequestNuggetExtractor
from crucible.pipeline.refinement import RefinementConfig, ReportRefinement
from crucible.pipeline.report_generator import CrucibleReportGenerator, ReportGeneratorConfig
from prefnugget.judges.extract_queryonly import QueryOnlyNuggetExtractor
from rag_framework.llm.client import OpenRouterLLM, get_llm_client
from rag_framework.models.document import Document
from rag_framework.models.query import Query

from research.adapters import query_to_topic
from research.benchmark.datasets import RagtimeTopicRecord
from research.config import DEFAULT_IMPROVED_PROFILE, DEFAULT_PROFILE_A, ExperimentConfig
from research.document_source import DocumentSource
from research.generation.report_service import ReportGenerationService
from research.loop import PreferenceLearningLoop
from research.models import ExperimentResult


SYSTEM_IDS = (
    "preference_loop_full",
    "preference_loop_1round",
    "crucible_single",
    "crucible_dual_best",
    "vanilla_rag",
)


class BenchmarkSystem(Protocol):
    """Protocol for benchmark report generators."""

    system_id: str

    def generate(
        self,
        topic: RagtimeTopicRecord,
        documents: list[Document],
    ) -> CrucibleReportBundle | ExperimentResult:
        ...


@dataclass
class SystemRunResult:
    """Normalized output from any benchmark system.

    Parameters
    ----------
    system_id : str
        System identifier.
    topic_id : str
        Topic id.
    bundle : CrucibleReportBundle
        Final report bundle for evaluation.
    experiment : ExperimentResult | None
        Full loop result when applicable.
    """

    system_id: str
    topic_id: str
    bundle: CrucibleReportBundle
    experiment: ExperimentResult | None = None


class PreferenceLoopSystem:
    """Run the full or single-round preference-learning loop."""

    def __init__(
        self,
        system_id: str,
        config: ExperimentConfig,
        llm: OpenRouterLLM | None = None,
    ) -> None:
        if system_id not in {"preference_loop_full", "preference_loop_1round"}:
            raise ValueError(f"Unsupported loop system id: {system_id}")
        self.system_id = system_id
        self._config = config
        self._llm = llm or get_llm_client()
        self._loop = PreferenceLearningLoop(config, llm=self._llm)

    def generate(
        self,
        topic: RagtimeTopicRecord,
        documents: list[Document],
    ) -> SystemRunResult:
        result = self._loop.run(topic.query, request=topic.request, documents=documents)
        return SystemRunResult(
            system_id=self.system_id,
            topic_id=topic.query.query_id,
            bundle=result.final_champion,
            experiment=result,
        )


class CrucibleSingleSystem:
    """Single-pass CRUCIBLE report from auto-generated base nuggets."""

    system_id = "crucible_single"

    def __init__(self, config: ExperimentConfig, llm: OpenRouterLLM | None = None) -> None:
        self._config = config
        self._llm = llm or get_llm_client()
        self._document_source = DocumentSource(config.document_source)
        self._report_service = ReportGenerationService(
            team_id=config.team_id,
            char_limit=config.char_limit,
            max_docs_per_nugget=config.document_source.max_docs_per_nugget,
        )
        self._nugget_extractor = RequestNuggetExtractor(max_nuggets=config.base_nuggets_count)

    def generate(
        self,
        topic: RagtimeTopicRecord,
        documents: list[Document],
    ) -> SystemRunResult:
        base_bank = self._nugget_extractor.extract(topic.request)
        base_bank.title_query = topic.query.title
        bundle = self._report_service.generate(
            query=topic.query,
            nugget_bank=base_bank,
            documents=documents,
            profile=DEFAULT_PROFILE_A,
            run_id=f"{self._config.experiment_id}-single-{topic.query.query_id}",
            request=topic.request,
        )
        bundle.metadata["benchmark_system"] = self.system_id
        return SystemRunResult(
            system_id=self.system_id,
            topic_id=topic.query.query_id,
            bundle=bundle,
        )


class CrucibleDualBestSystem:
    """Best of dual reports (round 0 only, no improvement iterations)."""

    system_id = "crucible_dual_best"

    def __init__(self, config: ExperimentConfig, llm: OpenRouterLLM | None = None) -> None:
        one_round = ExperimentConfig.from_dict(_experiment_dict(config, max_rounds=1, enable_final_synthesis=False))
        self._inner = PreferenceLoopSystem("preference_loop_1round", one_round, llm=llm)

    def generate(
        self,
        topic: RagtimeTopicRecord,
        documents: list[Document],
    ) -> SystemRunResult:
        result = self._inner.generate(topic, documents)
        result.system_id = self.system_id
        result.bundle.metadata["benchmark_system"] = self.system_id
        return result


class VanillaRAGSystem:
    """Retrieve documents then produce one-shot cited summary (non-nugget baseline)."""

    system_id = "vanilla_rag"

    def __init__(self, config: ExperimentConfig, llm: OpenRouterLLM | None = None) -> None:
        self._config = config
        self._llm = llm or get_llm_client()
        self._generator = CrucibleReportGenerator(
            config=ReportGeneratorConfig(
                team_id=config.team_id,
                run_id=f"{config.experiment_id}-vanilla",
            )
        )
        self._refiner = ReportRefinement(config=RefinementConfig(char_limit=config.char_limit))

    def generate(
        self,
        topic: RagtimeTopicRecord,
        documents: list[Document],
    ) -> SystemRunResult:
        queryonly = QueryOnlyNuggetExtractor(max_questions=min(8, self._config.base_nuggets_count))
        bank = queryonly.extract(query_to_topic(topic.query))
        from research.adapters import question_bank_to_crucible_bank

        crucible_bank = question_bank_to_crucible_bank(bank, title_query=topic.query.title)
        raw_report, _alignments = self._generator.generate(
            crucible_bank,
            documents,
            request=topic.request,
            max_docs_per_nugget=self._config.document_source.max_docs_per_nugget,
        )
        report = self._refiner.refine(raw_report, crucible_bank)
        report.run_id = f"{self._config.experiment_id}-vanilla-{topic.query.query_id}"
        report.metadata["benchmark_system"] = self.system_id
        bundle = CrucibleReportBundle(
            query=topic.query,
            report=report,
            source_documents=documents,
            nugget_bank=crucible_bank,
            metadata={"benchmark_system": self.system_id},
        )
        return SystemRunResult(
            system_id=self.system_id,
            topic_id=topic.query.query_id,
            bundle=bundle,
        )


def build_system(system_id: str, config: ExperimentConfig, llm: OpenRouterLLM | None = None) -> BenchmarkSystem:
    """Instantiate a benchmark system by id.

    Parameters
    ----------
    system_id : str
        One of ``SYSTEM_IDS``.
    config : ExperimentConfig
        Shared experiment configuration.
    llm : OpenRouterLLM | None
        Optional shared LLM client.

    Returns
    -------
    BenchmarkSystem
        Configured generator.
    """
    if system_id == "preference_loop_full":
        return PreferenceLoopSystem(system_id, config, llm=llm)
    if system_id == "preference_loop_1round":
        one_round = ExperimentConfig.from_dict(_experiment_dict(config, max_rounds=1, enable_final_synthesis=False))
        return PreferenceLoopSystem(system_id, one_round, llm=llm)
    if system_id == "crucible_single":
        return CrucibleSingleSystem(config, llm=llm)
    if system_id == "crucible_dual_best":
        return CrucibleDualBestSystem(config, llm=llm)
    if system_id == "vanilla_rag":
        return VanillaRAGSystem(config, llm=llm)
    raise ValueError(f"Unknown system id: {system_id}. Expected one of {SYSTEM_IDS}")


def _experiment_dict(config: ExperimentConfig, max_rounds: int, enable_final_synthesis: bool = False) -> dict:
    return {
        "experiment_id": config.experiment_id,
        "team_id": config.team_id,
        "char_limit": config.char_limit,
        "base_nuggets_count": config.base_nuggets_count,
        "max_contrastive_nuggets": config.max_contrastive_nuggets,
        "output_dir": str(config.output_dir),
        "document_source": {
            "provider": config.document_source.provider,
            "max_documents": config.document_source.max_documents,
            "max_docs_per_nugget": config.document_source.max_docs_per_nugget,
            "collection_path": str(config.document_source.collection_path)
            if config.document_source.collection_path
            else None,
            "ragtime_pipeline": config.document_source.ragtime_pipeline,
            "ragtime_collection": config.document_source.ragtime_collection,
            "cache_dir": str(config.document_source.cache_dir)
            if config.document_source.cache_dir
            else None,
        },
        "iteration": {
            "max_rounds": max_rounds,
            "min_improvement_rounds": config.iteration.min_improvement_rounds if max_rounds > 1 else 0,
            "enable_final_synthesis": enable_final_synthesis,
            "min_new_nuggets_to_continue": config.iteration.min_new_nuggets_to_continue,
            "stable_rounds_for_convergence": config.iteration.stable_rounds_for_convergence,
        },
    }
