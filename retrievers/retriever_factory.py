"""
检索器工厂

提供统一的检索器创建和管理接口
"""

import os
from typing import Dict, List, Optional, Type, Callable
import pandas as pd

from retrievers.base_retriever import (
    BaseRetriever, RetrieverConfig, EvaluationMetrics, RetrievalResult
)


class RetrieverFactory:
    """
    检索器工厂类

    支持：
    1. 注册新的检索器类型
    2. 通过配置创建检索器
    3. 批量创建和管理检索器
    """

    # 已注册的检索器类型
    _registry: Dict[str, Type[BaseRetriever]] = {}

    # 默认配置
    _default_configs: Dict[str, RetrieverConfig] = {}

    @classmethod
    def register(cls, name: str, retriever_class: Type[BaseRetriever],
                 default_config: Optional[RetrieverConfig] = None):
        """
        注册检索器类型

        Args:
            name: 检索器名称
            retriever_class: 检索器类
            default_config: 默认配置

        Example:
            >>> @RetrieverFactory.register("my_retriever", MyRetriever)
            >>> class MyRetriever(BaseRetriever):
            >>>     pass
        """
        cls._registry[name] = retriever_class
        if default_config:
            cls._default_configs[name] = default_config
        print(f"Registered retriever: {name}")

    @classmethod
    def unregister(cls, name: str):
        """
        注销检索器类型

        Args:
            name: 检索器名称
        """
        if name in cls._registry:
            del cls._registry[name]
            if name in cls._default_configs:
                del cls._default_configs[name]
            print(f"Unregistered retriever: {name}")

    @classmethod
    def list_available(cls) -> List[str]:
        """
        列出所有可用的检索器类型

        Returns:
            检索器名称列表
        """
        return list(cls._registry.keys())

    @classmethod
    def create(cls, name: str, field_csv: str, table_csv: str,
               config: Optional[RetrieverConfig] = None,
               **kwargs) -> BaseRetriever:
        """
        创建检索器实例

        Args:
            name: 检索器名称
            field_csv: 字段CSV路径
            table_csv: 表CSV路径
            config: 检索器配置
            **kwargs: 额外参数

        Returns:
            检索器实例

        Raises:
            ValueError: 如果检索器类型未注册
        """
        if name not in cls._registry:
            available = cls.list_available()
            raise ValueError(
                f"Unknown retriever: {name}. "
                f"Available retrievers: {available}"
            )

        retriever_class = cls._registry[name]

        # 使用默认配置（如果未提供）
        if config is None and name in cls._default_configs:
            config = cls._default_configs[name]

        return retriever_class(field_csv, table_csv, config=config, **kwargs)

    @classmethod
    def create_all(cls, field_csv: str, table_csv: str,
                   names: Optional[List[str]] = None,
                   configs: Optional[Dict[str, RetrieverConfig]] = None) -> Dict[str, BaseRetriever]:
        """
        批量创建检索器

        Args:
            field_csv: 字段CSV路径
            table_csv: 表CSV路径
            names: 要创建的检索器名称列表（None表示全部）
            configs: 检索器配置字典

        Returns:
            检索器实例字典 {name: retriever}
        """
        if names is None:
            names = cls.list_available()

        configs = configs or {}
        retrievers = {}

        for name in names:
            config = configs.get(name)
            try:
                retrievers[name] = cls.create(name, field_csv, table_csv, config)
            except Exception as e:
                print(f"Failed to create retriever {name}: {e}")

        return retrievers

    @classmethod
    def get_config_template(cls, name: str) -> Optional[RetrieverConfig]:
        """
        获取检索器的配置模板

        Args:
            name: 检索器名称

        Returns:
            配置模板
        """
        return cls._default_configs.get(name)


class RetrieverManager:
    """
    检索器管理器

    管理、训练和评估多个检索器
    """

    def __init__(self, field_csv: str, table_csv: str, qa_csv: str = None):
        """
        初始化管理器

        Args:
            field_csv: 字段CSV路径
            table_csv: 表CSV路径
            qa_csv: QA数据CSV路径
        """
        self.field_csv = field_csv
        self.table_csv = table_csv
        self.qa_csv = qa_csv

        self.retrievers: Dict[str, BaseRetriever] = {}
        self.qa_df = pd.read_csv(qa_csv) if qa_csv else None

    def add_retriever(self, name: str, config: Optional[RetrieverConfig] = None):
        """
        添加检索器

        Args:
            name: 检索器名称
            config: 检索器配置
        """
        retriever = RetrieverFactory.create(name, self.field_csv, self.table_csv, config)
        self.retrievers[name] = retriever

    def remove_retriever(self, name: str):
        """移除检索器"""
        if name in self.retrievers:
            del self.retrievers[name]

    def get_retriever(self, name: str) -> Optional[BaseRetriever]:
        """获取检索器"""
        return self.retrievers.get(name)

    def fit_all(self, train_data: pd.DataFrame = None):
        """
        训练所有检索器

        Args:
            train_data: 训练数据
        """
        for name, retriever in self.retrievers.items():
            print(f"Fitting {name}...")
            retriever.fit(train_data)

    def fit(self, name: str, train_data: pd.DataFrame = None):
        """训练指定检索器"""
        if name in self.retrievers:
            self.retrievers[name].fit(train_data)

    def evaluate_all(self, test_data: pd.DataFrame, k: int = 5) -> Dict[str, EvaluationMetrics]:
        """
        评估所有检索器

        Args:
            test_data: 测试数据
            k: top-k结果

        Returns:
            评估结果字典 {name: metrics}
        """
        results = {}
        for name, retriever in self.retrievers.items():
            print(f"Evaluating {name}...")
            if retriever.is_fitted:
                results[name] = retriever.evaluate(test_data, k)
            else:
                print(f"  Skipping {name} (not fitted)")
        return results

    def evaluate(self, name: str, test_data: pd.DataFrame, k: int = 5) -> Optional[EvaluationMetrics]:
        """评估指定检索器"""
        if name in self.retrievers and self.retrievers[name].is_fitted:
            return self.retrievers[name].evaluate(test_data, k)
        return None

    def compare(self, test_data: pd.DataFrame, k: int = 5) -> pd.DataFrame:
        """
        对比所有检索器的性能

        Args:
            test_data: 测试数据
            k: top-k结果

        Returns:
            对比结果DataFrame
        """
        results = self.evaluate_all(test_data, k)

        rows = []
        for name, metrics in results.items():
            rows.append({
                'Method': name,
                'Table Accuracy': metrics.table_accuracy,
                'SQL Accuracy': metrics.sql_accuracy,
                'Avg Query Time (s)': metrics.avg_query_time,
                'Avg Memory (MB)': metrics.avg_memory_mb,
                'Total Queries': metrics.total_queries
            })

        return pd.DataFrame(rows)

    def run_experiment(self, train_ratio: float = 0.8,
                       k: int = 5) -> Dict[str, pd.DataFrame]:
        """
        运行完整实验

        Args:
            train_ratio: 训练集比例
            k: top-k结果

        Returns:
            实验结果
        """
        if self.qa_df is None:
            raise ValueError("QA data not provided")

        import numpy as np

        # 分割数据
        indices = np.random.permutation(len(self.qa_df))
        n_train = int(len(indices) * train_ratio)

        train_data = self.qa_df.iloc[indices[:n_train]]
        test_data = self.qa_df.iloc[indices[n_train:]]

        # 训练
        self.fit_all(train_data)

        # 评估
        comparison_df = self.compare(test_data, k)

        return {
            'comparison': comparison_df,
            'train_size': len(train_data),
            'test_size': len(test_data)
        }


# 自动注册内置检索器
def _auto_register():
    """自动注册内置检索器"""
    try:
        from retrievers.bm25_retriever import BM25Retriever
        from retrievers.vector_retriever import VectorRetriever
        from retrievers.llm_retriever import LLMRetriever
        from retrievers.terag_retriever import TERAGRetriever

        # 注册BM25
        if 'BM25' not in RetrieverFactory._registry:
            RetrieverFactory.register(
                'BM25', BM25Retriever,
                RetrieverConfig(name='BM25', description='BM25 keyword-based retriever')
            )

        # 注册Vector
        if 'Vector' not in RetrieverFactory._registry:
            RetrieverFactory.register(
                'Vector', VectorRetriever,
                RetrieverConfig(name='Vector', description='TF-IDF vector retriever')
            )

        # 注册LLM
        if 'LLM' not in RetrieverFactory._registry:
            RetrieverFactory.register(
                'LLM', LLMRetriever,
                RetrieverConfig(name='LLM', description='LLM-based retriever')
            )

        # 注册TE-RAG
        if 'TE-RAG' not in RetrieverFactory._registry:
            RetrieverFactory.register(
                'TE-RAG', TERAGRetriever,
                RetrieverConfig(name='TE-RAG', description='Table-Enhanced RAG')
            )

    except ImportError as e:
        print(f"Warning: Failed to auto-register retrievers: {e}")

    # 注册 TE-RAG V2 (论文版)
    try:
        from terag.terag_retriever_v2 import TERAGRetrieverV2
        from terag.config import TERAGConfig

        # TE-RAG V2 需要特殊处理，因为它需要配置文件
        # 我们创建一个适配器类
        class TERAGV2Adapter(BaseRetriever):
            """TE-RAG V2 适配器，使其兼容旧接口"""

            def __init__(self, field_csv: str, table_csv: str,
                         config: Optional[RetrieverConfig] = None,
                         terag_config_path: str = None):
                if config is None:
                    config = RetrieverConfig(
                        name="TE-RAG-V2",
                        description="Table-Enhanced RAG V2 (论文版)"
                    )
                super().__init__(field_csv, table_csv, config)

                # 加载 TE-RAG 配置
                import os
                from pathlib import Path

                if terag_config_path is None:
                    terag_config_path = Path(field_csv).parent.parent / 'config.yaml'

                if os.path.exists(terag_config_path):
                    self.terag_config = TERAGConfig.from_yaml(terag_config_path)
                else:
                    # 使用默认配置
                    self.terag_config = TERAGConfig()

                self.terag_retriever = None

            def fit(self, train_data: pd.DataFrame = None):
                """训练"""
                if train_data is not None:
                    self.terag_retriever = TERAGRetrieverV2(self.terag_config)
                    self.terag_retriever.fit(train_data)
                else:
                    # 尝试从 artifacts 加载
                    try:
                        self.terag_retriever = TERAGRetrieverV2.from_artifacts(self.terag_config)
                    except Exception:
                        pass

                self._is_fitted = True

            def _retrieve(self, query: str, k: int = 5) -> List[RetrievalResult]:
                """检索"""
                if not self.terag_retriever:
                    return []

                results = self.terag_retriever.retrieve(query, k)

                # 转换格式
                return [
                    RetrievalResult(
                        table=r.table,
                        table_score=r.table_score,
                        columns=r.columns,
                        metadata=r.metadata
                    )
                    for r in results
                ]

        if 'TE-RAG-V2' not in RetrieverFactory._registry:
            RetrieverFactory.register(
                'TE-RAG-V2', TERAGV2Adapter,
                RetrieverConfig(name='TE-RAG-V2', description='Table-Enhanced RAG V2 (论文版)')
            )

    except ImportError as e:
        print(f"Warning: Failed to register TE-RAG V2: {e}")


# 执行自动注册
_auto_register()
