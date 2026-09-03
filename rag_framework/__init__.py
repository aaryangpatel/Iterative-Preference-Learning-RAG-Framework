"""Modular research framework for iterative retrieval-augmented generation."""

from rag_framework.llm import LLMConfig, OpenRouterLLM, get_llm_client
from rag_framework.models.document import Document, RankedDocument, RetrievalResult
from rag_framework.models.query import Query
from rag_framework.pipeline.retrieval import RetrievalPipeline, RetrievalPipelineConfig

__all__ = [
    "Document",
    "LLMConfig",
    "OpenRouterLLM",
    "Query",
    "RankedDocument",
    "RetrievalPipeline",
    "RetrievalPipelineConfig",
    "RetrievalResult",
    "get_llm_client",
]

__version__ = "0.1.0"
