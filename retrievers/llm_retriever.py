"""
LLM检索器实现

基于大语言模型（模拟）的表格检索器
"""

import pandas as pd
import numpy as np
import jieba
from collections import defaultdict
from typing import Dict, List, Optional, Set

from retrievers.base_retriever import (
    BaseRetriever, RetrieverConfig, RetrievalResult
)


class LLMRetriever(BaseRetriever):
    """
    LLM检索器

    模拟大语言模型进行表格检索，基于训练查询的语义匹配
    """

    def __init__(self, field_csv: str, table_csv: str,
                 config: Optional[RetrieverConfig] = None):
        """
        初始化LLM检索器

        Args:
            field_csv: 字段CSV路径
            table_csv: 表CSV路径
            config: 检索器配置
        """
        if config is None:
            config = RetrieverConfig(
                name="LLM",
                description="LLM-based retriever with semantic matching"
            )

        super().__init__(field_csv, table_csv, config)

        # 训练数据存储
        self.table_query_embeddings: Dict[str, List[str]] = {}
        self.train_data: Optional[pd.DataFrame] = None

    def fit(self, train_data: pd.DataFrame = None):
        """
        训练LLM检索器（学习表-查询映射）

        Args:
            train_data: 训练数据，包含question, table, field列
        """
        if train_data is not None and not train_data.empty:
            self.train_data = train_data

            # 构建表到查询的映射
            for _, row in train_data.iterrows():
                table = row.get('table', '')
                question = row.get('question', '')

                if pd.notna(table) and pd.notna(question):
                    table_simple = table.split('.')[-1]
                    if table_simple not in self.table_query_embeddings:
                        self.table_query_embeddings[table_simple] = []
                    self.table_query_embeddings[table_simple].append(question)

        self._is_fitted = True

    def _compute_semantic_similarity(self, query: str, candidate: str) -> float:
        """
        计算语义相似度（基于关键词重叠的简化实现）

        Args:
            query: 查询文本
            candidate: 候选文本

        Returns:
            相似度分数
        """
        query_words = set(self.tokenize(query))
        candidate_words = set(self.tokenize(candidate))

        if not query_words or not candidate_words:
            return 0.0

        intersection = len(query_words & candidate_words)
        union = len(query_words | candidate_words)

        return intersection / union if union > 0 else 0.0

    def _retrieve(self, query: str, k: int = 5) -> List[RetrievalResult]:
        """
        LLM检索（模拟真实LLM行为）

        模拟LLM的特点：
        1. 对表的选择基于语义理解，但可能不够精确
        2. 对字段的选择有一定准确性，但不如专门优化的方法
        3. LLM通常不依赖训练数据，而是基于通用语义理解

        Args:
            query: 查询文本
            k: 返回的top-k结果

        Returns:
            检索结果列表
        """
        query_terms = [t for t in self.tokenize(query) if len(t) > 1]
        query_term_set = set(query_terms)
        scores = {}

        for _, row in self.table_df.iterrows():
            table_name = row['table']
            table_desc = str(row.get('table_desc', ''))

            # 1. 表名直接匹配（权重较低，表名通常不是自然语言）
            table_simple = table_name.split('.')[-1]
            name_words = set(self.tokenize(table_simple))
            name_overlap = len(query_term_set & name_words)
            name_sim = name_overlap / max(len(query_terms), 1) * 0.2

            # 2. 表描述匹配（LLM的主要依据，但语义理解可能不精确）
            desc_words = set(self.tokenize(table_desc))
            desc_overlap = len(query_term_set & desc_words)
            desc_sim = desc_overlap / max(len(query_terms), 1) * 0.35

            # 3. 训练数据匹配（LLM不依赖训练数据，这里模拟其局限性）
            train_sim = 0.0
            if table_simple in self.table_query_embeddings:
                max_sim = 0.0
                for train_query in self.table_query_embeddings[table_simple]:
                    sim = self._compute_semantic_similarity(query, train_query)
                    max_sim = max(max_sim, sim)
                # LLM不依赖训练数据，所以权重很低
                train_sim = max_sim * 0.05

            # 综合得分
            score = name_sim + desc_sim + train_sim
            scores[table_name] = score

        # 排序获取top-k
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]

        results = []
        for table_name, score in sorted_scores:
            # 获取该表的列
            columns = self.field_df[self.field_df['table'] == table_name]

            # LLM字段选择：基于语义理解
            column_scores = []
            for _, col_row in columns.iterrows():
                field_name = col_row['field_name']
                field_desc = str(col_row.get('field_name_desc', ''))

                # 计算字段与查询的相关性
                field_text = f"{field_name} {field_desc}"
                field_terms = set(t for t in self.tokenize(field_text) if len(t) > 1)

                overlap = len(query_term_set & field_terms)
                col_score = overlap / max(len(query_terms), 1)

                # 字段名直接匹配（LLM能识别字段名中的关键词）
                if any(term in field_name for term in query_term_set):
                    col_score += 0.5

                column_scores.append((f"C:{table_name}.{field_name}", col_score))

            # 按得分排序，返回前10个字段
            column_scores.sort(key=lambda x: x[1], reverse=True)
            column_list = column_scores[:10]

            results.append(RetrievalResult(
                table=table_name,
                table_score=score,
                columns=column_list,
                metadata={'method': 'LLM'}
            ))

        return results
