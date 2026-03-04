"""
BM25检索器实现

基于BM25算法的表格检索器
"""

import pandas as pd
import numpy as np
import jieba
import math
from collections import defaultdict
from typing import Dict, List, Optional

from retrievers.base_retriever import (
    BaseRetriever, RetrieverConfig, RetrievalResult
)


class BM25Retriever(BaseRetriever):
    """
    BM25检索器

    使用BM25算法进行表格检索，基于关键词匹配
    """

    def __init__(self, field_csv: str, table_csv: str,
                 config: Optional[RetrieverConfig] = None):
        """
        初始化BM25检索器

        Args:
            field_csv: 字段CSV路径
            table_csv: 表CSV路径
            config: 检索器配置
        """
        if config is None:
            config = RetrieverConfig(
                name="BM25",
                description="BM25 keyword-based retriever"
            )

        super().__init__(field_csv, table_csv, config)

        # BM25索引
        self.table_inverted_index: Dict[str, List[tuple]] = {}
        self.table_doc_freq: Dict[str, int] = {}
        self.table_doc_lengths: List[int] = []
        self.total_table_docs: int = 0
        self.avg_table_doc_length: float = 1.0

    def fit(self, train_data: pd.DataFrame = None):
        """
        构建BM25索引

        Args:
            train_data: 训练数据（BM25不需要，保持接口一致）
        """
        # 创建倒排索引
        self.table_inverted_index = {}
        self.table_doc_lengths = []

        for idx, row in self.table_df.iterrows():
            # 合并表名和描述作为文档
            text = f"{row['table']} {row.get('table_desc', '')}"
            words = self.tokenize(text)
            self.table_doc_lengths.append(len(words))

            # 统计词频
            word_count = defaultdict(int)
            for word in words:
                if len(word) > 1:
                    word_count[word] += 1

            # 添加到倒排索引
            for word, count in word_count.items():
                if word not in self.table_inverted_index:
                    self.table_inverted_index[word] = []
                self.table_inverted_index[word].append((idx, count))

        # 计算文档频率
        self.table_doc_freq = {}
        for word, postings in self.table_inverted_index.items():
            self.table_doc_freq[word] = len(postings)

        # 计算统计信息
        self.total_table_docs = len(self.table_df)
        self.avg_table_doc_length = np.mean(self.table_doc_lengths) if self.table_doc_lengths else 1.0

        self._is_fitted = True

    def _retrieve(self, query: str, k: int = 5) -> List[RetrievalResult]:
        """
        BM25检索

        Args:
            query: 查询文本
            k: 返回的top-k结果

        Returns:
            检索结果列表
        """
        query_terms = [t for t in self.tokenize(query) if len(t) > 1]
        query_term_set = set(query_terms)
        scores: Dict[int, float] = {}

        # 计算BM25得分
        for term in query_terms:
            if term not in self.table_inverted_index:
                continue

            df = self.table_doc_freq[term]
            idf = math.log((self.total_table_docs - df + 0.5) / (df + 0.5) + 1)

            for doc_idx, tf in self.table_inverted_index[term]:
                doc_length = self.table_doc_lengths[doc_idx]

                # BM25公式
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_length / self.avg_table_doc_length)
                score = idf * numerator / denominator

                if doc_idx not in scores:
                    scores[doc_idx] = 0
                scores[doc_idx] += score

        # 排序获取top-k
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]

        results = []
        for doc_idx, score in sorted_scores:
            table_name = self.table_df.iloc[doc_idx]['table']

            # 获取该表的列，并根据与查询的相关性排序
            columns = self.field_df[self.field_df['table'] == table_name]

            # 计算每个字段与查询的相关性得分
            column_scores = []
            for _, row in columns.iterrows():
                field_name = row['field_name']
                field_desc = str(row.get('field_name_desc', ''))

                # 计算字段名和描述与查询的匹配度
                field_text = f"{field_name} {field_desc}"
                field_terms = set(t for t in self.tokenize(field_text) if len(t) > 1)

                # 匹配得分：交集大小 / 查询词数
                overlap = len(query_term_set & field_terms)
                col_score = overlap / len(query_terms) if query_terms else 0

                # 加入字段名本身的匹配（如查询中包含字段名关键词）
                if any(term in field_name for term in query_term_set):
                    col_score += 0.5

                column_scores.append((f"C:{table_name}.{field_name}", col_score))

            # 按得分排序，取前8个字段
            column_scores.sort(key=lambda x: x[1], reverse=True)
            column_list = column_scores[:8]

            results.append(RetrievalResult(
                table=table_name,
                table_score=score,
                columns=column_list,
                metadata={'method': 'BM25'}
            ))

        return results
