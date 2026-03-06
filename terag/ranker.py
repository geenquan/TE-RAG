"""
排序器

表排序器和字段排序器
"""

import os
import pickle
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from terag.config import TERAGConfig
from terag.feature_extractor import FeatureExtractor


@dataclass
class RankingResult:
    """排序结果"""
    table: str
    table_score: float
    columns: List[Tuple[str, float]]  # [(column_name, score), ...]


class TableRanker:
    """
    表排序器

    使用学习到的权重对表进行排序
    """

    def __init__(self, config: TERAGConfig):
        """
        初始化表排序器

        Args:
            config: TE-RAG 配置
        """
        self.config = config
        self.weights = config.feature.table_weights.copy()
        self._model = None

    def set_weights(self, weights: Dict[str, float]):
        """设置权重"""
        self.weights = weights.copy()

    def get_weights(self) -> Dict[str, float]:
        """获取权重"""
        return self.weights.copy()

    def rank(
        self,
        table_features: Dict[str, Dict[str, float]],
        k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        对表进行排序

        Args:
            table_features: {table_node: {'bm25_score': float, ...}}
            k: 返回的 top-k 结果

        Returns:
            [(table_node, score), ...]
        """
        scores = {}

        for table_node, features in table_features.items():
            score = 0.0
            for feature_name, feature_value in features.items():
                weight = self.weights.get(feature_name, 0.0)
                score += weight * feature_value
            scores[table_node] = score

        # 排序并返回 top-k
        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:k]

    def save(self, output_path: str):
        """保存排序器"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'wb') as f:
            pickle.dump({
                'weights': self.weights,
                'model': self._model,
            }, f)
        print(f"表排序器已保存到: {output_path}")

    def load(self, input_path: str):
        """加载排序器"""
        with open(input_path, 'rb') as f:
            data = pickle.load(f)
            self.weights = data['weights']
            self._model = data.get('model')
        print(f"表排序器已从 {input_path} 加载")


class FieldRanker:
    """
    字段排序器

    使用学习到的权重对字段进行排序
    """

    def __init__(self, config: TERAGConfig):
        """
        初始化字段排序器

        Args:
            config: TE-RAG 配置
        """
        self.config = config
        self.weights = config.feature.field_weights.copy()
        self._model = None

    def set_weights(self, weights: Dict[str, float]):
        """设置权重"""
        self.weights = weights.copy()

    def get_weights(self) -> Dict[str, float]:
        """获取权重"""
        return self.weights.copy()

    def rank(
        self,
        field_features: Dict[str, Dict[str, float]],
        k: int = 10
    ) -> List[Tuple[str, float]]:
        """
        对字段进行排序

        Args:
            field_features: {field_name: {'direct_match': float, ...}}
            k: 返回的 top-k 结果

        Returns:
            [(field_name, score), ...]
        """
        scores = {}

        for field_name, features in field_features.items():
            score = 0.0
            for feature_name, feature_value in features.items():
                weight = self.weights.get(feature_name, 0.0)
                score += weight * feature_value
            scores[field_name] = score

        # 排序并返回 top-k
        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:k]

    def save(self, output_path: str):
        """保存排序器"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'wb') as f:
            pickle.dump({
                'weights': self.weights,
                'model': self._model,
            }, f)
        print(f"字段排序器已保存到: {output_path}")

    def load(self, input_path: str):
        """加载排序器"""
        with open(input_path, 'rb') as f:
            data = pickle.load(f)
            self.weights = data['weights']
            self._model = data.get('model')
        print(f"字段排序器已从 {input_path} 加载")


class CombinedRanker:
    """
    组合排序器

    整合表排序和字段排序
    """

    def __init__(
        self,
        config: TERAGConfig,
        table_ranker: Optional[TableRanker] = None,
        field_ranker: Optional[FieldRanker] = None
    ):
        """
        初始化组合排序器

        Args:
            config: TE-RAG 配置
            table_ranker: 表排序器
            field_ranker: 字段排序器
        """
        self.config = config
        self.table_ranker = table_ranker or TableRanker(config)
        self.field_ranker = field_ranker or FieldRanker(config)

    def rank(
        self,
        feature_extractor: FeatureExtractor,
        query: str,
        k_tables: int = 5,
        k_fields: int = 10
    ) -> List[RankingResult]:
        """
        执行完整排序

        Args:
            feature_extractor: 特征提取器
            query: 查询文本
            k_tables: 返回的表数量
            k_fields: 每个表返回的字段数量

        Returns:
            [RankingResult, ...]
        """
        results = []

        # 1. 提取表级特征并排序
        table_features = feature_extractor.extract_table_features(query)
        top_tables = self.table_ranker.rank(table_features, k=k_tables)

        # 2. 对每个表提取字段级特征并排序
        for table_node, table_score in top_tables:
            table_name = table_node.replace('T:', '')

            # 提取字段级特征
            field_features = feature_extractor.extract_field_features(query, table_name)

            # 字段排序
            top_fields = self.field_ranker.rank(field_features, k=k_fields)

            # 构建结果
            results.append(RankingResult(
                table=table_name,
                table_score=table_score,
                columns=[(f"C:{table_name}.{field}", score) for field, score in top_fields]
            ))

        return results

    def save(self, artifacts_dir: str):
        """保存排序器"""
        self.table_ranker.save(os.path.join(artifacts_dir, 'table_ranker.pkl'))
        self.field_ranker.save(os.path.join(artifacts_dir, 'field_ranker.pkl'))

    def load(self, artifacts_dir: str):
        """加载排序器"""
        self.table_ranker.load(os.path.join(artifacts_dir, 'table_ranker.pkl'))
        self.field_ranker.load(os.path.join(artifacts_dir, 'field_ranker.pkl'))
