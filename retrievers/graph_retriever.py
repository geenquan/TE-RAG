"""
Graph-only检索器实现

基于 TE-RAG-V2 裁剪 pattern 分支，只保留基础检索与 graph propagation 的排序能力
"""

import os
import pandas as pd
from typing import Dict, List, Optional
from pathlib import Path

from retrievers.base_retriever import (
    BaseRetriever, RetrieverConfig, RetrievalResult
)


class GraphOnlyRetriever(BaseRetriever):
    """
    Graph-only检索器

    基于 TE-RAG-V2 实现，但只保留：
    - 基础检索（BM25）
    - 图传播增强（Graph Propagation）

    不使用：
    - Pattern mining
    - Query expansion / annotation enhancement

    本质是"结构感知检索，但不使用模式增强"
    """

    def __init__(self, field_csv: str, table_csv: str,
                 config: Optional[RetrieverConfig] = None,
                 terag_config_path: str = None):
        """
        初始化Graph-only检索器

        Args:
            field_csv: 字段CSV路径
            table_csv: 表CSV路径
            config: 检索器配置
            terag_config_path: TE-RAG配置文件路径
        """
        if config is None:
            config = RetrieverConfig(
                name="Graph",
                description="Graph-only retriever (BM25 + Graph, no Pattern)"
            )

        super().__init__(field_csv, table_csv, config)

        # 加载 TE-RAG 配置
        if terag_config_path is None:
            terag_config_path = Path(field_csv).parent.parent / 'config.yaml'

        self.terag_config = self._load_terag_config(terag_config_path)

        # 内部 TE-RAG 检索器
        self.terag_retriever = None

    def _load_terag_config(self, config_path: str):
        """加载 TE-RAG 配置，并设置消融开关"""
        from terag.config import TERAGConfig

        if os.path.exists(config_path):
            config = TERAGConfig.from_yaml(config_path)
        else:
            config = TERAGConfig()

        # 关键：设置消融开关
        # 保留图传播，关闭模板挖掘
        config.ablation.use_graph_weight = True
        config.ablation.use_template_mining = False
        config.ablation.use_pattern_generalization = False

        # 更新特征权重：只使用 BM25 和 graph_score
        config.feature.table_weights = {
            'bm25_score': 0.5,
            'graph_score': 0.5,
            'pattern_score': 0.0  # 不使用 pattern
        }

        # 更新字段权重：不使用 role_prior（与 pattern 相关）
        config.feature.field_weights = {
            'direct_match': 0.50,
            'graph_propagation': 0.35,
            'role_prior': 0.0,      # 不使用
            'train_recommend': 0.15
        }

        return config

    def fit(self, train_data: pd.DataFrame = None):
        """
        训练/构建索引

        构建图和索引，但不进行 pattern mining

        Args:
            train_data: 训练数据
        """
        from terag.terag_retriever_v2 import TERAGRetrieverV2

        print("  Graph: 构建图结构（无Pattern）...")

        if train_data is not None:
            self.terag_retriever = TERAGRetrieverV2(self.terag_config)
            self.terag_retriever.fit(train_data)
        else:
            # 尝试从 artifacts 加载
            try:
                self.terag_retriever = TERAGRetrieverV2.from_artifacts(self.terag_config)
                print("  Graph: 从 artifacts 加载成功")
            except Exception as e:
                print(f"  Graph: 无法从 artifacts 加载: {e}")
                self.terag_retriever = None

        self._is_fitted = True

    def _retrieve(self, query: str, k: int = 5) -> List[RetrievalResult]:
        """
        Graph-only检索

        只基于基础检索 + graph score 排序

        Args:
            query: 查询文本
            k: 返回的top-k结果

        Returns:
            检索结果列表
        """
        if not self.terag_retriever:
            # 如果没有 TE-RAG 检索器，回退到简单的 BM25
            return self._fallback_retrieve(query, k)

        # 使用 TE-RAG 的带消融检索
        results = self.terag_retriever.retrieve_with_ablation(
            query, k,
            use_graph_weight=True,
            use_template_mining=False
        )

        # 转换为标准 RetrievalResult
        return [
            RetrievalResult(
                table=r.table,
                table_score=r.table_score,
                columns=r.columns,
                metadata={
                    'method': 'Graph-only',
                    'components': ['BM25', 'Graph']
                }
            )
            for r in results
        ]

    def _fallback_retrieve(self, query: str, k: int = 5) -> List[RetrievalResult]:
        """
        回退检索（当 TE-RAG 不可用时）

        使用简单的 BM25 检索
        """
        from retrievers.bm25_retriever import BM25Retriever

        if not hasattr(self, '_fallback_bm25'):
            self._fallback_bm25 = BM25Retriever(
                self.field_df.to_csv(index=False),
                self.table_df.to_csv(index=False)
            )
            # 直接使用已有数据
            self._fallback_bm25.field_df = self.field_df
            self._fallback_bm25.table_df = self.table_df
            self._fallback_bm25.fit()

        return self._fallback_bm25.retrieve(query, k)


class GraphOnlyRetrieverAdapter(BaseRetriever):
    """
    Graph-only 检索器适配器（简化版）

    直接继承 BaseRetriever，手动实现 graph 增强的检索逻辑
    不依赖 TERAGRetrieverV2 的完整实现

    关键：作为 baseline，应该比 TE-RAG 弱，不使用 pattern mining
    """

    def __init__(self, field_csv: str, table_csv: str,
                 config: Optional[RetrieverConfig] = None,
                 train_data: pd.DataFrame = None):
        """
        初始化

        Args:
            field_csv: 字段CSV路径
            table_csv: 表CSV路径
            config: 检索器配置
        """
        if config is None:
            config = RetrieverConfig(
                name="Graph",
                description="Graph-only retriever (BM25 + Graph similarity)"
            )

        super().__init__(field_csv, table_csv, config)

        # 内部 BM25 检索器
        from retrievers.bm25_retriever import BM25Retriever
        self.bm25 = BM25Retriever(field_csv, table_csv)

        # 训练数据（用于图传播计算）
        self.train_data = train_data
        self.train_tokens_cache: Dict[int, set] = {}

        # 图传播参数 - 使用更保守的参数
        # 作为 baseline，不应该太强
        self.graph_propagation_threshold = 0.25  # 提高阈值，减少传播
        self.graph_propagation_weight = 1.5       # 降低权重

    def fit(self, train_data: pd.DataFrame = None):
        """
        训练

        Args:
            train_data: 训练数据
        """
        # 构建 BM25 索引
        print("  Graph: 构建BM25索引...")
        self.bm25.fit(train_data)

        # 缓存训练数据分词结果
        if train_data is not None:
            self.train_data = train_data
            import jieba
            for idx in range(len(train_data)):
                question = train_data.iloc[idx]['question']
                self.train_tokens_cache[idx] = set(jieba.cut(question))

        self._is_fitted = True

    def _retrieve(self, query: str, k: int = 5) -> List[RetrievalResult]:
        """
        检索

        结合 BM25 分数和图传播分数

        Args:
            query: 查询文本
            k: 返回的top-k结果

        Returns:
            检索结果列表
        """
        import jieba
        from collections import defaultdict

        # 1. BM25 检索
        bm25_results = self.bm25.retrieve(query, k * 2)

        # 2. 计算图传播分数
        query_tokens = set(jieba.cut(query))
        graph_scores = defaultdict(float)

        if self.train_data is not None:
            for idx in range(len(self.train_data)):
                hist_tokens = self.train_tokens_cache.get(idx, set())
                if not hist_tokens:
                    continue

                # 计算 Jaccard 相似度
                intersection = len(query_tokens & hist_tokens)
                if intersection == 0:
                    continue

                union = len(query_tokens | hist_tokens)
                sim = intersection / union if union > 0 else 0

                if sim > self.graph_propagation_threshold:
                    # 获取该历史查询对应的表
                    table = self.train_data.iloc[idx].get('table', '')
                    if pd.notna(table):
                        table_simple = table.split('.')[-1]
                        graph_scores[table_simple] += sim * self.graph_propagation_weight

        # 3. 融合分数
        table_scores = {}
        bm25_columns = {}

        for r in bm25_results:
            table_scores[r.table] = r.table_score
            bm25_columns[r.table] = r.columns

        # 归一化 BM25 分数
        if table_scores:
            values = list(table_scores.values())
            min_val, max_val = min(values), max(values)
            if max_val > min_val:
                table_scores = {
                    t: (s - min_val) / (max_val - min_val)
                    for t, s in table_scores.items()
                }

        # 归一化图分数
        if graph_scores:
            values = list(graph_scores.values())
            min_val, max_val = min(values), max(values)
            if max_val > min_val:
                graph_scores = {
                    t: (s - min_val) / (max_val - min_val)
                    for t, s in graph_scores.items()
                }

        # 加权融合
        final_scores = {}
        all_tables = set(table_scores.keys()) | set(graph_scores.keys())

        for table in all_tables:
            bm25_s = table_scores.get(table, 0.0)
            graph_s = graph_scores.get(table, 0.0)
            # 50% BM25 + 50% Graph
            final_scores[table] = 0.5 * bm25_s + 0.5 * graph_s

        # 4. 排序并返回结果
        sorted_tables = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)[:k]

        results = []
        for table_name, score in sorted_tables:
            columns = bm25_columns.get(table_name, [])

            results.append(RetrievalResult(
                table=table_name,
                table_score=score,
                columns=columns,
                metadata={
                    'method': 'Graph-only',
                    'components': ['BM25', 'Graph'],
                    'bm25_score': table_scores.get(table_name, 0.0),
                    'graph_score': graph_scores.get(table_name, 0.0)
                }
            ))

        return results
