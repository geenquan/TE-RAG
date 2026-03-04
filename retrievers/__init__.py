"""
检索器模块

提供可插拔的检索器实现，支持：
1. 基类定义
2. 多种检索器实现
3. 工厂模式创建检索器
4. 配置化管理
"""

from retrievers.base_retriever import (
    BaseRetriever,
    RetrieverConfig,
    RetrievalResult,
    EvaluationMetrics
)
from retrievers.bm25_retriever import BM25Retriever
from retrievers.vector_retriever import VectorRetriever
from retrievers.llm_retriever import LLMRetriever
from retrievers.terag_retriever import TERAGRetriever
from retrievers.retriever_factory import RetrieverFactory, RetrieverManager

__all__ = [
    'BaseRetriever',
    'RetrieverConfig',
    'RetrievalResult',
    'EvaluationMetrics',
    'BM25Retriever',
    'VectorRetriever',
    'LLMRetriever',
    'TERAGRetriever',
    'RetrieverFactory',
    'RetrieverManager'
]
