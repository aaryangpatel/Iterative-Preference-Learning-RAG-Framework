"""PrefNugget-style LLM judge for evaluating RAG systems."""

from prefnugget.models.judgment import GradeResult, PreferenceResult
from prefnugget.models.nugget import NuggetQuestion, NuggetQuestionBank
from prefnugget.models.response import RagRunRecord, RunDocument
from prefnugget.models.topic import PrefNuggetTopic
from prefnugget.judges.pref_ranker import PrefRanker
from prefnugget.judges.extract_contrastive import ContrastiveNuggetExtractor
from prefnugget.judges.extract_queryonly import QueryOnlyNuggetExtractor
from prefnugget.judges.grader import NuggetGrader
from prefnugget.alignment.bank_aligner import PrefNuggetBankAligner
from prefnugget.pipeline.workflow import PrefNuggetWorkflow

__all__ = [
    "ContrastiveNuggetExtractor",
    "GradeResult",
    "NuggetGrader",
    "NuggetQuestion",
    "NuggetQuestionBank",
    "PrefNuggetBankAligner",
    "PrefNuggetTopic",
    "PrefNuggetWorkflow",
    "PrefRanker",
    "PreferenceResult",
    "QueryOnlyNuggetExtractor",
    "RagRunRecord",
    "RunDocument",
]

__version__ = "0.1.0"
