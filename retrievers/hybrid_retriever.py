"""
Hybrid检索器实现

复用 BM25 和 Vector 两个检索器，做线性分数融合
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional

from retrievers.base_retriever import (
    BaseRetriever, RetrieverConfig, RetrievalResult
)
from retrievers.bm25_retriever import BM25Retriever
from retrievers.vector_retriever import VectorRetriever


class HybridRetriever(BaseRetriever):
    """
    Hybrid检索器

    复用 BM25 和 Vector 两个检索器，做线性分数融合

    final_score = alpha * bm25_score + (1 - alpha) * vector_score
    """

    def __init__(self, field_csv: str, table_csv: str,
                 config: Optional[RetrieverConfig] = None,
                 alpha: float = 0.5):
        """
        初始化Hybrid检索器

        Args:
            field_csv: 字段CSV路径
            table_csv: 表CSV路径
            config: 检索器配置
            alpha: BM25分数权重 (0-1), 默认0.5表示等权重
                   建议值: 0.5 (均衡), 0.6 (偏关键词), 0.4 (偏语义)
        """
        if config is None:
            config = RetrieverConfig(
                name="Hybrid",
                description="Hybrid retriever combining BM25 and Vector"
            )

        super().__init__(field_csv, table_csv, config)

        # 融合权重
        self.alpha = alpha

        # 创建子检索器
        self.bm25_retriever = BM25Retriever(field_csv, table_csv)
        self.vector_retriever = VectorRetriever(field_csv, table_csv)

    def fit(self, train_data: pd.DataFrame = None):
        """
        训练/构建索引

        分别构建 BM25 和 Vector 的索引

        Args:
            train_data: 训练数据（BM25和Vector不需要，保持接口一致）
        """
        print("  Hybrid: 构建BM25索引...")
        self.bm25_retriever.fit(train_data)

        print("  Hybrid: 构建Vector索引...")
        self.vector_retriever.fit(train_data)

        self._is_fitted = True

    def _retrieve(self, query: str, k: int = 5) -> List[RetrievalResult]:
        """
        Hybrid检索

        1. 分别调用 BM25 和 Vector 的检索结果
        2. 对候选结果按表名对齐
        3. 计算融合分数
        4. 按融合分数重新排序
        5. 返回 top-k 结果

        Args:
            query: 查询文本
            k: 返回的top-k结果

        Returns:
            检索结果列表
        """
        # 分别获取两个检索器的结果（获取更多候选以便融合）
        retrieve_k = min(k * 3, 20)  # 获取更多候选用于融合

        bm25_results = self.bm25_retriever.retrieve(query, retrieve_k)
        vector_results = self.vector_retriever.retrieve(query, retrieve_k)

        # 收集所有候选表名
        all_tables = set()
        bm25_scores: Dict[str, float] = {}
        vector_scores: Dict[str, float] = {}
        bm25_columns: Dict[str, List[tuple]] = {}
        vector_columns: Dict[str, List[tuple]] = {}

        for r in bm25_results:
            all_tables.add(r.table)
            bm25_scores[r.table] = r.table_score
            bm25_columns[r.table] = r.columns

        for r in vector_results:
            all_tables.add(r.table)
            vector_scores[r.table] = r.table_score
            vector_columns[r.table] = r.columns

        # 分数归一化（Min-Max归一化）
        def normalize_scores(scores: Dict[str, float]) -> Dict[str, float]:
            if not scores:
                return {}
            values = list(scores.values())
            min_val, max_val = min(values), max(values)
            if max_val == min_val:
                return {k: 1.0 for k in scores}
            return {k: (v - min_val) / (max_val - min_val) for k, v in scores.items()}

        norm_bm25 = normalize_scores(bm25_scores)
        norm_vector = normalize_scores(vector_scores)

        # 计算融合分数
        final_scores: Dict[str, float] = {}
        for table in all_tables:
            bm25_s = norm_bm25.get(table, 0.0)
            vector_s = norm_vector.get(table, 0.0)
            # 线性加权融合
            final_scores[table] = self.alpha * bm25_s + (1 - self.alpha) * vector_s

        # 按融合分数排序
        sorted_tables = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)[:k]

        # 构建结果
        results = []
        query_terms = [t for t in self.tokenize(query) if len(t) > 1]
        query_term_set = set(query_terms)

        for table_name, score in sorted_tables:
            # 合并字段列表（优先使用BM25的字段，如果不存在则使用Vector的）
            if table_name in bm25_columns:
                columns = bm25_columns[table_name]
            elif table_name in vector_columns:
                columns = vector_columns[table_name]
            else:
                # 如果两个检索器都没有字段信息，从原始数据获取
                columns = self._get_default_columns(table_name, query_term_set, query_terms)

            results.append(RetrievalResult(
                table=table_name,
                table_score=score,
                columns=columns,
                metadata={
                    'method': 'Hybrid',
                    'alpha': self.alpha,
                    'bm25_score': bm25_scores.get(table_name, 0.0),
                    'vector_score': vector_scores.get(table_name, 0.0)
                }
            ))

        return results

    def _get_default_columns(self, table_name: str, query_term_set: set, query_terms: list) -> List[tuple]:
        """获取默认字段列表"""
        columns = self.field_df[self.field_df['table'] == table_name]

        column_scores = []
        for _, row in columns.iterrows():
            field_name = row['field_name']
            field_desc = str(row.get('field_name_desc', ''))

            field_text = f"{field_name} {field_desc}"
            field_terms = set(t for t in self.tokenize(field_text) if len(t) > 1)

            overlap = len(query_term_set & field_terms)
            col_score = overlap / len(query_terms) if query_terms else 0

            if any(term in field_name for term in query_term_set):
                col_score += 0.5

            column_scores.append((f"C:{table_name}.{field_name}", col_score))

        column_scores.sort(key=lambda x: x[1], reverse=True)
        return column_scores[:8]
