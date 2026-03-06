"""
TE-RAG V2 检索器

论文版 TE-RAG 检索器，整合所有组件，支持加载预构建 artifacts
"""

import os
import json
import pickle
import pandas as pd
import networkx as nx
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from terag.config import TERAGConfig
from terag.graph_builder import BipartiteGraphBuilder
from terag.pattern_miner import PatternMiner
from terag.index_builder import IndexBuilder
from terag.feature_extractor import FeatureExtractor
from terag.ranker import TableRanker, FieldRanker, CombinedRanker, RankingResult
from terag.weight_learner import WeightLearner


@dataclass
class RetrievalResultV2:
    """检索结果 V2"""
    table: str
    table_score: float
    columns: List[Tuple[str, float]]
    metadata: Dict = field(default_factory=dict)


class TERAGRetrieverV2:
    """
    TE-RAG V2 检索器

    论文版 TE-RAG，整合所有组件：
    - 二分图（Bipartite Graph）
    - 模式挖掘（Pattern Mining）
    - BM25 索引
    - 特征提取
    - 学习权重排序

    特点：
    - 支持加载预构建 artifacts
    - 支持消融开关
    - 统一配置管理

    使用方式:
        # 方式1：从 artifacts 加载
        retriever = TERAGRetrieverV2.from_artifacts(config)

        # 方式2：从头构建
        retriever = TERAGRetrieverV2(config)
        retriever.fit(train_data)

        # 检索
        results = retriever.retrieve(query, k=5)
    """

    def __init__(self, config: TERAGConfig):
        """
        初始化检索器

        Args:
            config: TE-RAG 配置
        """
        self.config = config

        # 组件
        self.graph = None
        self.patterns = None
        self.index = None
        self.feature_extractor = None
        self.ranker = None

        # 数据
        self.train_data = None
        self.field_df = pd.read_csv(config.data.field_csv)
        self.table_df = pd.read_csv(config.data.table_csv)

        # 状态
        self._is_fitted = False

    @classmethod
    def from_artifacts(cls, config: TERAGConfig, artifacts_dir: str = None) -> 'TERAGRetrieverV2':
        """
        从预构建 artifacts 创建检索器

        Args:
            config: TE-RAG 配置
            artifacts_dir: artifacts 目录路径（默认使用配置中的路径）

        Returns:
            TERAGRetrieverV2 实例
        """
        artifacts_dir = artifacts_dir or config.output.artifacts_dir

        retriever = cls(config)

        # 加载图
        graph_path = os.path.join(artifacts_dir, 'graph.pkl')
        if os.path.exists(graph_path):
            with open(graph_path, 'rb') as f:
                retriever.graph = pickle.load(f)
            print(f"加载图: {graph_path}")

        # 加载模式
        patterns_path = os.path.join(artifacts_dir, 'patterns.jsonl')
        if os.path.exists(patterns_path):
            miner = PatternMiner(config)
            retriever.patterns = miner.load(patterns_path)
            print(f"加载模式: {patterns_path}")

        # 加载索引
        index_dir = os.path.join(artifacts_dir, 'bm25_index')
        if os.path.exists(index_dir):
            builder = IndexBuilder(config)
            retriever.index = builder.load(index_dir)
            print(f"加载索引: {index_dir}")

        # 加载排序器
        table_ranker = TableRanker(config)
        field_ranker = FieldRanker(config)

        table_ranker_path = os.path.join(artifacts_dir, 'table_ranker.pkl')
        if os.path.exists(table_ranker_path):
            table_ranker.load(table_ranker_path)

        field_ranker_path = os.path.join(artifacts_dir, 'field_ranker.pkl')
        if os.path.exists(field_ranker_path):
            field_ranker.load(field_ranker_path)

        retriever.ranker = CombinedRanker(config, table_ranker, field_ranker)

        # 加载训练数据（如果存在）
        train_path = config.get_split_path('train')
        if os.path.exists(train_path):
            train_data = []
            with open(train_path, 'r', encoding='utf-8') as f:
                for line in f:
                    train_data.append(json.loads(line))
            retriever.train_data = pd.DataFrame(train_data)

        # 创建特征提取器
        if retriever.graph and retriever.index:
            train_df = retriever.train_data if retriever.train_data is not None and not retriever.train_data.empty else pd.DataFrame()
            patterns_dict = retriever.patterns if retriever.patterns is not None else {}

            retriever.feature_extractor = FeatureExtractor(
                config,
                retriever.graph,
                retriever.index,
                patterns_dict,
                train_df,
                retriever.field_df,
                retriever.table_df
            )

        retriever._is_fitted = True
        return retriever

    def fit(self, train_data: pd.DataFrame):
        """
        训练检索器

        Args:
            train_data: 训练数据
        """
        self.train_data = train_data

        # 1. 构建图
        print("构建二分图...")
        graph_builder = BipartiteGraphBuilder(self.config)
        self.graph = graph_builder.build(train_data)

        # 2. 挖掘模式
        print("挖掘查询模式...")
        element_to_queries = graph_builder.get_element_to_queries(self.graph, train_data)
        pattern_miner = PatternMiner(self.config)
        self.patterns = pattern_miner.mine(element_to_queries)

        # 3. 构建索引
        print("构建索引...")
        index_builder = IndexBuilder(self.config)
        self.index = index_builder.build()

        # 4. 创建特征提取器
        self.feature_extractor = FeatureExtractor(
            self.config,
            self.graph,
            self.index,
            self.patterns,
            train_data,
            self.field_df,
            self.table_df
        )

        # 5. 创建排序器（使用默认权重或加载学习到的权重）
        self.ranker = CombinedRanker(self.config)

        # 尝试加载学习到的权重
        table_ranker_path = self.config.get_artifact_path('table_ranker.pkl')
        field_ranker_path = self.config.get_artifact_path('field_ranker.pkl')

        if os.path.exists(table_ranker_path):
            self.ranker.table_ranker.load(table_ranker_path)

        if os.path.exists(field_ranker_path):
            self.ranker.field_ranker.load(field_ranker_path)

        self._is_fitted = True
        print("TE-RAG V2 训练完成")

    def retrieve(self, query: str, k: int = 5) -> List[RetrievalResultV2]:
        """
        检索

        Args:
            query: 查询文本
            k: 返回的 top-k 结果

        Returns:
            [RetrievalResultV2, ...]
        """
        if not self._is_fitted:
            raise RuntimeError("Retriever has not been fitted. Call fit() first.")

        # 执行排序
        ranking_results = self.ranker.rank(
            self.feature_extractor,
            query,
            k_tables=k,
            k_fields=10
        )

        # 转换为 RetrievalResultV2
        results = []
        for r in ranking_results:
            results.append(RetrievalResultV2(
                table=r.table,
                table_score=r.table_score,
                columns=r.columns,
                metadata={'method': 'TE-RAG-V2'}
            ))

        return results

    def retrieve_with_ablation(
        self,
        query: str,
        k: int = 5,
        **ablation_overrides
    ) -> List[RetrievalResultV2]:
        """
        带消融开关的检索

        Args:
            query: 查询文本
            k: 返回的 top-k 结果
            **ablation_overrides: 覆盖消融开关

        Returns:
            [RetrievalResultV2, ...]
        """
        # 临时覆盖消融开关
        original_values = {}
        for key, value in ablation_overrides.items():
            if hasattr(self.config.ablation, key):
                original_values[key] = getattr(self.config.ablation, key)
                setattr(self.config.ablation, key, value)

        try:
            return self.retrieve(query, k)
        finally:
            # 恢复原始值
            for key, value in original_values.items():
                setattr(self.config.ablation, key, value)

    def build_with_ablation(
        self,
        train_data: pd.DataFrame,
        ablation_config: Dict[str, bool],
        output_dir: str
    ) -> 'TERAGRetrieverV2':
        """
        根据消融配置重建 artifacts

        Args:
            train_data: 训练数据
            ablation_config: 消融配置，例如 {'use_graph_weight': False, 'use_template_mining': False}
            output_dir: 输出目录路径

        Returns:
            重建后的 TERAGRetrieverV2 实例
        """
        from terag.graph_builder import BipartiteGraphBuilder
        from terag.pattern_miner import PatternMiner
        from terag.index_builder import IndexBuilder

        print(f"为消融配置重建 artifacts: {ablation_config}")
        print(f"输出目录: {output_dir}")

        # 临时修改消融配置
        original_values = {}
        for key, value in ablation_config.items():
            if hasattr(self.config.ablation, key):
                original_values[key] = getattr(self.config.ablation, key)
                setattr(self.config.ablation, key, value)

        try:
            # 1. 构建图
            print("\n[1/3] 构建二分图...")
            graph_builder = BipartiteGraphBuilder(self.config)
            graph = graph_builder.build(train_data)

            graph_path = os.path.join(output_dir, 'graph.pkl')
            graph_builder.save(graph, graph_path)

            # 2. 挖掘模式
            print("\n[2/3] 挖掘查询模式...")
            element_to_queries = graph_builder.get_element_to_queries(graph, train_data)
            pattern_miner = PatternMiner(self.config)
            patterns = pattern_miner.mine(element_to_queries)

            patterns_path = os.path.join(output_dir, 'patterns.jsonl')
            pattern_miner.save(patterns, patterns_path)

            # 3. 构建索引
            print("\n[3/3] 构建索引...")
            index_builder = IndexBuilder(self.config)
            index = index_builder.build()

            index_dir = os.path.join(output_dir, 'bm25_index')
            index_builder.save(index, index_dir)

            # 4. 创建特征提取器
            feature_extractor = FeatureExtractor(
                self.config,
                graph,
                index,
                patterns,
                train_data,
                self.field_df,
                self.table_df
            )

            # 5. 创建排序器
            ranker = CombinedRanker(self.config)

            # 创建新的 retriever
            new_retriever = TERAGRetrieverV2(self.config)
            new_retriever.graph = graph
            new_retriever.patterns = patterns
            new_retriever.index = index
            new_retriever.feature_extractor = feature_extractor
            new_retriever.ranker = ranker
            new_retriever.train_data = train_data
            new_retriever._is_fitted = True

            print(f"\n消融 artifacts 构建完成: {output_dir}")

            return new_retriever

        finally:
            # 恢复原始配置
            for key, value in original_values.items():
                setattr(self.config.ablation, key, value)

    def get_table_weights(self) -> Dict[str, float]:
        """获取表排序权重"""
        return self.ranker.table_ranker.get_weights()

    def get_field_weights(self) -> Dict[str, float]:
        """获取字段排序权重"""
        return self.ranker.field_ranker.get_weights()

    @property
    def is_fitted(self) -> bool:
        """是否已训练"""
        return self._is_fitted


def main():
    """演示 TE-RAG V2 检索"""
    import sys
    from pathlib import Path

    # 添加项目根目录
    sys.path.insert(0, str(Path(__file__).parent.parent))

    # 加载配置
    config_path = Path(__file__).parent.parent / 'config.yaml'
    config = TERAGConfig.from_yaml(str(config_path))

    print("=" * 60)
    print("TE-RAG V2 检索器演示")
    print("=" * 60)

    # 尝试从 artifacts 加载
    artifacts_dir = config.output.artifacts_dir
    if os.path.exists(os.path.join(artifacts_dir, 'graph.pkl')):
        print("\n从 artifacts 加载...")
        retriever = TERAGRetrieverV2.from_artifacts(config, artifacts_dir)
    else:
        print("\n从头构建...")
        retriever = TERAGRetrieverV2(config)

        # 加载训练数据
        train_data = []
        with open(config.get_split_path('train'), 'r', encoding='utf-8') as f:
            for line in f:
                train_data.append(json.loads(line))
        train_df = pd.DataFrame(train_data)

        retriever.fit(train_df)

    # 测试检索
    print("\n测试检索:")
    test_queries = [
        "查询公司的售电量",
        "统计用户的电费金额",
        "分析供电所的回收率",
    ]

    for query in test_queries:
        print(f"\n查询: {query}")
        results = retriever.retrieve(query, k=3)

        for i, result in enumerate(results, 1):
            print(f"  {i}. {result.table} (score: {result.table_score:.4f})")
            for col, score in result.columns[:3]:
                print(f"      - {col}: {score:.4f}")

    # 打印权重
    print("\n排序权重:")
    print(f"  表排序: {retriever.get_table_weights()}")
    print(f"  字段排序: {retriever.get_field_weights()}")


if __name__ == "__main__":
    main()
