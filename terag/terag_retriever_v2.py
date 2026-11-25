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

        # 记录实际加载的外部资产来源（P1-A：默认 fit 不加载任何旧资产）
        self.asset_sources: List[str] = []

        # 状态
        self._is_fitted = False

        # P2-D3：B/C 注解通道产物（fit 后可用；默认 None=关闭）
        self.miner = None
        self.transfer_result = None
        self._annotation_terms = {}
        self._diagnostics = None

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
            retriever.asset_sources.append(graph_path)
            print(f"加载图: {graph_path}")

        # 加载模式
        patterns_path = os.path.join(artifacts_dir, 'patterns.jsonl')
        if os.path.exists(patterns_path):
            miner = PatternMiner(config)
            retriever.patterns = miner.load(patterns_path)
            retriever.asset_sources.append(patterns_path)
            print(f"加载模式: {patterns_path}")

        # 加载索引
        index_dir = os.path.join(artifacts_dir, 'bm25_index')
        if os.path.exists(index_dir):
            builder = IndexBuilder(config)
            retriever.index = builder.load(index_dir)
            retriever.asset_sources.append(index_dir)
            print(f"加载索引: {index_dir}")

        # 加载排序器
        table_ranker = TableRanker(config)
        field_ranker = FieldRanker(config)

        table_ranker_path = os.path.join(artifacts_dir, 'table_ranker.pkl')
        if os.path.exists(table_ranker_path):
            table_ranker.load(table_ranker_path)
            retriever.asset_sources.append(table_ranker_path)

        field_ranker_path = os.path.join(artifacts_dir, 'field_ranker.pkl')
        if os.path.exists(field_ranker_path):
            field_ranker.load(field_ranker_path)
            retriever.asset_sources.append(field_ranker_path)

        retriever.ranker = CombinedRanker(config, table_ranker, field_ranker)

        # 加载训练数据（如果存在）
        train_path = config.get_split_path('train')
        if os.path.exists(train_path):
            train_data = []
            with open(train_path, 'r', encoding='utf-8') as f:
                for line in f:
                    train_data.append(json.loads(line))
            retriever.train_data = pd.DataFrame(train_data)
            retriever.asset_sources.append(train_path)

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

        # P1-A：关键资产缺失时不得宣告训练成功
        if retriever.graph is None or retriever.index is None:
            raise ValueError(
                f"from_artifacts: {artifacts_dir} 缺少 graph.pkl 或 bm25_index/，"
                f"无法构成可检索实例（拒絶静默降级）")

        retriever._is_fitted = True
        return retriever

    def fit(self, train_data: pd.DataFrame, load_artifact_weights: bool = False,
            diagnostics=None):
        """
        训练检索器

        仅使用传入的 train_data 与当前 config 中的固定参数，从零构建
        图/模式/索引/特征。默认不加载 artifacts/ 中任何历史产物
        （P1-A 泄漏修复：旧实现会隐式加载 table_ranker.pkl / field_ranker.pkl，
        这些权重可能来自包含本次测试集的历史划分）。

        P2-D3：B/C 注解通道接入主线（config.annotations）——
        use_history=True 时同一次 fit 内：训练日志 → 直接注解 B →（可选）
        受约束迁移 C → IndexBuilder（extra_table_texts 内存传递）→
        特征提取器（列排序注解加成）。开关支持消融；B/C 生成一次、去重注入；
        不读取评价 SQL/gold、不加载全局历史资产。

        Args:
            train_data: 训练数据（当前运行允许使用的全部监督信号）
            diagnostics: 默认 None；显式 DiagnosticExport 仅导出本次新计算证据
            load_artifact_weights: 显式选择加载 artifacts 中的排序器权重，
                仅用于复现历史行为；启用时会记录资产来源
        """
        from terag.config import AnnotationsConfig
        self._diagnostics = diagnostics
        ann_cfg = getattr(self.config, 'annotations',
                          AnnotationsConfig())  # 缺省关闭（兼容旧行为）
        self.train_data = train_data

        # 1. 构建图
        print("构建二分图...")
        graph_builder = BipartiteGraphBuilder(self.config)
        self.graph = graph_builder.build(train_data)

        # 2. 挖掘模式（旧关键词管线——与 B/C 注解并行存在，不互相注入；
        #    其 _generalize_patterns 不参与新迁移路径）
        print("挖掘查询模式...")
        element_to_queries = graph_builder.get_element_to_queries(self.graph, train_data)
        pattern_miner = PatternMiner(self.config)
        self.patterns = pattern_miner.mine(element_to_queries)

        # 2b. B/C 注解（P2-D3 主线接线：一次生成、内存传递）
        self.miner = None
        self.transfer_result = None
        extra_texts = None
        texts_b, texts_c = {}, {}
        self._annotation_terms = {}   # {table.column: [term, ...]} 列排序加成
        if ann_cfg.use_history:
            from retrievers.annotation_miner import DirectAnnotationMiner
            print("挖掘直接注解 (B)...")
            self.miner = DirectAnnotationMiner(self.field_df, self.table_df,
                                               dialect='mysql').mine(train_data)
            texts_b = self.miner.table_annotation_texts(
                self.config.index.field_weights.pattern)
            # B 类列加成项
            for a in self.miner.annotations:
                if a['type'] == 'TERM' and a.get('status') == 'confirmed' \
                        and a.get('target'):
                    self._annotation_terms.setdefault(a['target'], []).append(
                        str(a['content']))
            extra_texts = dict(texts_b)
            if ann_cfg.use_transfer:
                from retrievers.annotation_transfer import (
                    SchemaInfo, transfer_annotations, transfer_index_texts,
                    transfer_column_bonus_map)
                print("受约束迁移注解 (C)...")
                columns = {}
                for t, g in self.field_df.groupby('table'):
                    columns[t] = (g[['field_name', 'field_name_desc',
                                    'field_type']]
                                  .rename(columns={'field_name_desc': 'field_desc'})
                                  .to_dict('records'))
                table_desc = {r['table']: r.get('table_desc', '')
                              for _, r in self.table_df.iterrows()}
                self.transfer_result = transfer_annotations(
                    self.miner.annotations,
                    SchemaInfo(columns=columns, table_desc=table_desc))
                texts_c = transfer_index_texts(
                    self.transfer_result, self.config.index.field_weights.pattern)
                for t, items in texts_c.items():
                    merged = list(extra_texts.get(t, []))
                    existing = {txt for txt, _w in merged}
                    for txt, w in items:
                        if txt not in existing:   # B/C 同文本去重
                            merged.append((txt, w))
                            existing.add(txt)
                    extra_texts[t] = merged
                for key, terms in transfer_column_bonus_map(
                        self.transfer_result).items():
                    lst = self._annotation_terms.setdefault(key, [])
                    for term in terms:
                        if term not in lst:
                            lst.append(term)

        # 3. 构建索引（B/C 经 extra_table_texts 内存注入）
        print("构建索引...")
        index_builder = IndexBuilder(self.config)
        self.index = index_builder.build(extra_table_texts=extra_texts)

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
        # 列排序注解加成接口（P2-D3）：特征提取器可查询 B/C 术语命中
        self.feature_extractor.annotation_terms = dict(self._annotation_terms)

        # 5. 创建排序器（默认使用 config.feature 中声明的固定权重）
        self.ranker = CombinedRanker(self.config)

        # 显式 opt-in 才加载 artifacts 中的历史权重（P1-A 泄漏修复）
        if load_artifact_weights:
            table_ranker_path = self.config.get_artifact_path('table_ranker.pkl')
            field_ranker_path = self.config.get_artifact_path('field_ranker.pkl')

            if os.path.exists(table_ranker_path):
                self.ranker.table_ranker.load(table_ranker_path)
                self.asset_sources.append(table_ranker_path)

            if os.path.exists(field_ranker_path):
                self.ranker.field_ranker.load(field_ranker_path)
                self.asset_sources.append(field_ranker_path)

        if self._diagnostics is not None:
            self._diagnostics.record_fit(self, train_data, texts_b, texts_c, extra_texts)
        self._is_fitted = True
        print(f"TE-RAG V2 训练完成（排序权重来源: {'artifacts(显式加载)' if load_artifact_weights else 'config 固定默认值'}）")

    def retrieve(self, query: str, k: int = 5, *, diagnostic_qid=None) -> List[RetrievalResultV2]:
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

        if diagnostic_qid is not None and self._diagnostics is None:
            raise ValueError('diagnostic_qid requires explicit diagnostics in fit')
        trace = {} if diagnostic_qid is not None else None
        rank_kwargs = {'k_tables': k, 'k_fields': 10}
        if trace is not None:
            rank_kwargs['diagnostics'] = trace
        ranking_results = self.ranker.rank(self.feature_extractor, query, **rank_kwargs)

        # 转换为 RetrievalResultV2
        results = []
        for r in ranking_results:
            results.append(RetrievalResultV2(
                table=r.table,
                table_score=r.table_score,
                columns=r.columns,
                metadata={'method': 'TE-RAG-V2'}
            ))

        if trace is not None:
            self._diagnostics.record_query(diagnostic_qid, trace)
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
