"""Batch benchmark execution over RAGTIME topics and systems."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from crucible.loaders import save_report_bundle
from rag_framework.llm.client import OpenRouterLLM, get_llm_client

from research.benchmark.baselines import SYSTEM_IDS, SystemRunResult, build_system
from research.benchmark.config import BenchmarkConfig
from research.benchmark.datasets import RagtimeBenchmarkData, load_benchmark_data
from research.document_source import DocumentSource


@dataclass
class BenchmarkRunPlan:
    """Dry-run plan describing what would be executed.

    Parameters
    ----------
    benchmark_id : str
        Benchmark identifier.
    topic_ids : list[str]
        Topics to process.
    systems : list[str]
        Systems to run.
    dry_run : bool
        Whether live APIs are disabled.
    document_provider : str
        Document source provider name.
    output_dir : str
        Target output directory.
    """

    benchmark_id: str
    topic_ids: list[str]
    systems: list[str]
    dry_run: bool
    document_provider: str
    output_dir: str


class BenchmarkRunner:
    """Generate reports for all configured systems and topics.

    Parameters
    ----------
    config : BenchmarkConfig
        Benchmark configuration.
    llm : OpenRouterLLM | None
        Shared LLM client for generation systems.
    """

    def __init__(
        self,
        config: BenchmarkConfig,
        llm: OpenRouterLLM | None = None,
    ) -> None:
        self.config = config
        self._llm = llm or get_llm_client()
        self._data = load_benchmark_data(config)
        self._document_source = DocumentSource(config.experiment.document_source)
        self._output_root = config.benchmark_output_dir()

    @property
    def data(self) -> RagtimeBenchmarkData:
        return self._data

    def run_generate(self) -> list[SystemRunResult]:
        """Generate reports for all topic × system pairs.

        Returns
        -------
        list[SystemRunResult]
            Generated run results (empty when ``dry_run`` writes plan only).
        """
        self._output_root.mkdir(parents=True, exist_ok=True)
        if self.config.dry_run:
            self._write_run_plan()
            return []

        results: list[SystemRunResult] = []
        total = len(self._data.topics) * len(self.config.systems)
        completed = 0
        for topic in self._data.topics:
            topic_id = topic.query.query_id
            run_path = self._run_path
            print(f"\n[benchmark] Topic {topic_id}: fetching documents...", flush=True)
            documents = self._document_source.fetch(topic.query)
            print(f"[benchmark] Topic {topic_id}: {len(documents)} documents", flush=True)
            for system_id in self.config.systems:
                if system_id not in SYSTEM_IDS:
                    raise ValueError(f"Unknown system id in config: {system_id}")
                output_path = run_path(system_id, topic_id)
                if output_path.exists():
                    completed += 1
                    print(
                        f"[benchmark] Skip {completed}/{total} {system_id}/{topic_id} (already saved)",
                        flush=True,
                    )
                    continue
                print(f"[benchmark] Run {completed + 1}/{total} {system_id}/{topic_id}...", flush=True)
                system = build_system(system_id, self.config.experiment, llm=self._llm)
                run_result = None
                last_error: Exception | None = None
                for attempt in range(3):
                    try:
                        run_result = system.generate(topic, documents)
                        break
                    except Exception as error:
                        last_error = error
                        if attempt + 1 >= 3:
                            break
                        print(
                            f"[benchmark] Retry {attempt + 2}/3 {system_id}/{topic_id}: {error}",
                            flush=True,
                        )
                        import time

                        time.sleep(10 * (attempt + 1))
                if run_result is None:
                    self._log_failure(system_id, topic_id, last_error)
                    completed += 1
                    print(
                        f"[benchmark] FAILED {system_id}/{topic_id}: {last_error}",
                        flush=True,
                    )
                    continue
                self._save_run(run_result)
                results.append(run_result)
                completed += 1
                print(f"[benchmark] Saved {system_id}/{topic_id}", flush=True)
        return results

    def _log_failure(self, system_id: str, topic_id: str, error: Exception | None) -> None:
        failures_path = self._output_root / "failures.jsonl"
        failures_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "system_id": system_id,
            "topic_id": topic_id,
            "error": str(error) if error is not None else "unknown",
        }
        with failures_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

    def _run_path(self, system_id: str, topic_id: str) -> Path:
        return self._output_root / "runs" / system_id / f"{topic_id}.json"

    def _save_run(self, run_result: SystemRunResult) -> Path:
        run_dir = self._output_root / "runs" / run_result.system_id
        run_dir.mkdir(parents=True, exist_ok=True)
        path = run_dir / f"{run_result.topic_id}.json"
        save_report_bundle(run_result.bundle, path)
        if run_result.experiment is not None:
            experiment_path = run_dir / f"{run_result.topic_id}.experiment.json"
            experiment_path.write_text(
                run_result.experiment.model_dump_json(indent=2),
                encoding="utf-8",
            )
        return path

    def _write_run_plan(self) -> Path:
        plan = BenchmarkRunPlan(
            benchmark_id=self.config.benchmark_id,
            topic_ids=[topic.query.query_id for topic in self._data.topics],
            systems=list(self.config.systems),
            dry_run=True,
            document_provider=self.config.experiment.document_source.provider,
            output_dir=str(self._output_root),
        )
        path = self._output_root / "run_plan.json"
        path.write_text(json.dumps(asdict(plan), indent=2), encoding="utf-8")
        return path
