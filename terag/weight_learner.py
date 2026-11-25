"""
权重学习器

使用 LogisticRegression 学习排序权重
"""

import os
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from terag.config import TERAGConfig
from terag.feature_extractor import FeatureExtractor
from terag.ranker import TableRanker, FieldRanker


class WeightLearner:
    """
    权重学习器

    使用 LogisticRegression 学习表排序和字段排序的权重。

    使用方式:
        learner = WeightLearner(config, feature_extractor)
        learner.fit(train_data, dev_data)
        learner.save("artifacts/")

        # 获取学习到的权重
        table_weights = learner.get_table_weights()
        field_weights = learner.get_field_weights()
    """

    def __init__(self, config: TERAGConfig, feature_extractor: FeatureExtractor):
        """
        初始化权重学习器

        Args:
            config: TE-RAG 配置
            feature_extractor: 特征提取器
        """
        self.config = config
        self.feature_extractor = feature_extractor

        # 模型
        self.table_model = None
        self.field_model = None
        self.table_scaler = None
        self.field_scaler = None

        # 学习到的权重
        self.table_weights = {}
        self.field_weights = {}

    def fit(
        self,
        train_data: pd.DataFrame,
        dev_data: Optional[pd.DataFrame] = None
    ):
        """
        训练权重

        Args:
            train_data: 训练数据
            dev_data: 验证数据（可选，用于早停）
        """
        print("训练表排序权重...")
        self._fit_table_ranker(train_data)

        print("训练字段排序权重...")
        self._fit_field_ranker(train_data)

    def _fit_table_ranker(self, train_data: pd.DataFrame):
        """训练表排序器"""
        X = []
        y = []

        feature_names = ['bm25_score', 'graph_score', 'pattern_score']

        for idx, row in train_data.iterrows():
            query = row['question']
            gt_table = row.get('table', '')

            if pd.isna(gt_table):
                continue

            gt_table_simple = gt_table.split('.')[-1]

            # 提取特征
            table_features = self.feature_extractor.extract_table_features(query)

            for table_node, features in table_features.items():
                table_simple = table_node.replace('T:', '')
                label = 1 if table_simple == gt_table_simple else 0

                feature_vector = [features.get(fn, 0.0) for fn in feature_names]
                X.append(feature_vector)
                y.append(label)

        if not X:
            print("  警告：没有足够的训练数据")
            return

        X = np.array(X)
        y = np.array(y)

        # 标准化
        self.table_scaler = StandardScaler()
        X_scaled = self.table_scaler.fit_transform(X)

        # 训练
        self.table_model = LogisticRegression(
            class_weight='balanced',
            max_iter=1000,
            random_state=self.config.seed
        )
        self.table_model.fit(X_scaled, y)

        # 提取权重
        # LogisticRegression 的系数反映了每个特征的重要性
        coef = self.table_model.coef_[0]

        # 将系数转换为正权重
        weights = np.abs(coef)
        weights = weights / weights.sum()  # 归一化

        self.table_weights = {fn: float(w) for fn, w in zip(feature_names, weights)}

        print(f"  学习到的表排序权重: {self.table_weights}")

    def _fit_field_ranker(self, train_data: pd.DataFrame):
        """训练字段排序器"""
        X = []
        y = []

        feature_names = ['direct_match', 'graph_propagation', 'role_prior', 'train_recommend']

        for idx, row in train_data.iterrows():
            query = row['question']
            gt_table = row.get('table', '')
            gt_fields = row.get('field', '')

            if pd.isna(gt_table):
                continue

            gt_table_simple = gt_table.split('.')[-1]

            gt_field_set = set()
            if pd.notna(gt_fields) and isinstance(gt_fields, str):
                gt_field_set = set(f.strip() for f in gt_fields.split('|') if f.strip())

            # 提取特征
            field_features = self.feature_extractor.extract_field_features(query, gt_table_simple)

            for field_name, features in field_features.items():
                label = 1 if field_name in gt_field_set else 0

                feature_vector = [features.get(fn, 0.0) for fn in feature_names]
                X.append(feature_vector)
                y.append(label)

        if not X:
            print("  警告：没有足够的训练数据")
            return

        X = np.array(X)
        y = np.array(y)

        # 标准化
        self.field_scaler = StandardScaler()
        X_scaled = self.field_scaler.fit_transform(X)

        # 训练
        self.field_model = LogisticRegression(
            class_weight='balanced',
            max_iter=1000,
            random_state=self.config.seed
        )
        self.field_model.fit(X_scaled, y)

        # 提取权重
        coef = self.field_model.coef_[0]
        weights = np.abs(coef)
        weights = weights / weights.sum()

        self.field_weights = {fn: float(w) for fn, w in zip(feature_names, weights)}

        print(f"  学习到的字段排序权重: {self.field_weights}")

    def get_table_weights(self) -> Dict[str, float]:
        """获取表排序权重"""
        return self.table_weights.copy()

    def get_field_weights(self) -> Dict[str, float]:
        """获取字段排序权重"""
        return self.field_weights.copy()

    def get_table_ranker(self) -> TableRanker:
        """获取配置了学习权重的表排序器"""
        ranker = TableRanker(self.config)
        if self.table_weights:
            ranker.set_weights(self.table_weights)
        return ranker

    def get_field_ranker(self) -> FieldRanker:
        """获取配置了学习权重的字段排序器"""
        ranker = FieldRanker(self.config)
        if self.field_weights:
            ranker.set_weights(self.field_weights)
        return ranker

    def save(self, artifacts_dir: str):
        """保存权重学习器"""
        os.makedirs(artifacts_dir, exist_ok=True)

        with open(os.path.join(artifacts_dir, 'weight_learner.pkl'), 'wb') as f:
            pickle.dump({
                'table_model': self.table_model,
                'field_model': self.field_model,
                'table_scaler': self.table_scaler,
                'field_scaler': self.field_scaler,
                'table_weights': self.table_weights,
                'field_weights': self.field_weights,
            }, f)

        # 同时保存权重为 JSON 格式（方便查看）
        import json
        with open(os.path.join(artifacts_dir, 'learned_weights.json'), 'w', encoding='utf-8') as f:
            json.dump({
                'table_weights': self.table_weights,
                'field_weights': self.field_weights,
            }, f, indent=2, ensure_ascii=False)

        print(f"权重学习器已保存到: {artifacts_dir}")

    def load(self, artifacts_dir: str):
        """加载权重学习器"""
        with open(os.path.join(artifacts_dir, 'weight_learner.pkl'), 'rb') as f:
            data = pickle.load(f)
            self.table_model = data['table_model']
            self.field_model = data['field_model']
            self.table_scaler = data['table_scaler']
            self.field_scaler = data['field_scaler']
            self.table_weights = data['table_weights']
            self.field_weights = data['field_weights']

        print(f"权重学习器已从 {artifacts_dir} 加载")
        print(f"  表排序权重: {self.table_weights}")
        print(f"  字段排序权重: {self.field_weights}")


def main():
    """演示权重学习"""
    import sys
    import json
    from pathlib import Path
    import networkx as nx

    # 添加项目根目录
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from terag.config import TERAGConfig
    from terag.graph_builder import BipartiteGraphBuilder
    from terag.pattern_miner import PatternMiner
    from terag.index_builder import IndexBuilder

    # 加载配置
    config_path = Path(__file__).parent.parent / 'config.yaml'
    config = TERAGConfig.from_yaml(str(config_path))

    # 加载训练数据
    train_data = []
    with open(config.get_split_path('train'), 'r', encoding='utf-8') as f:
        for line in f:
            train_data.append(json.loads(line))
    train_df = pd.DataFrame(train_data)

    print(f"加载训练数据: {len(train_df)} 条")

    # 构建组件
    print("\n构建图...")
    graph_builder = BipartiteGraphBuilder(config)
    graph = graph_builder.build(train_df)

    print("\n挖掘模式...")
    element_to_queries = graph_builder.get_element_to_queries(graph, train_df)
    pattern_miner = PatternMiner(config)
    patterns = pattern_miner.mine(element_to_queries)

    print("\n构建索引...")
    index_builder = IndexBuilder(config)
    index = index_builder.build()

    # 加载字段表和数据表
    field_df = pd.read_csv(config.data.field_csv)
    table_df = pd.read_csv(config.data.table_csv)

    # 创建特征提取器
    feature_extractor = FeatureExtractor(
        config, graph, index, patterns,
        train_df, field_df, table_df
    )

    # 训练权重
    print("\n训练权重...")
    learner = WeightLearner(config, feature_extractor)
    learner.fit(train_df)

    # 保存
    learner.save(config.output.artifacts_dir)

    # 打印学习到的权重
    print("\n学习到的权重:")
    print(f"  表排序: {learner.get_table_weights()}")
    print(f"  字段排序: {learner.get_field_weights()}")


if __name__ == "__main__":
    main()
