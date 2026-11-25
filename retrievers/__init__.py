"""Minimal retrievers package for the BIRD quick run (no heavy deps)."""
from retrievers.base_retriever import (BaseRetriever, RetrieverConfig,
                                       RetrievalResult, EvaluationMetrics)
from retrievers.bm25_retriever import BM25Retriever as BM25SchemaRetriever
