"""
向量检索器实现

基于TF-IDF向量和余弦相似度的表格检索器
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


class VectorRetriever(BaseRetriever):
    """
    向量检索器

    使用TF-IDF向量和余弦相似度进行表格检索
    """

    def __init__(self, field_csv: str, table_csv: str,
                 config: Optional[RetrieverConfig] = None):
        """
        初始化向量检索器

        Args:
            field_csv: 字段CSV路径
            table_csv: 表CSV路径
            config: 检索器配置
        """
        if config is None:
            config = RetrieverConfig(
                name="Vector",
                description="TF-IDF vector-based retriever with cosine similarity"
            )

        super().__init__(field_csv, table_csv, config)

        # 向量相关
        self.vectors: Dict[str, np.ndarray] = {}
        self.vocab: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}

    def fit(self, train_data: pd.DataFrame = None):
        """
        构建向量索引

        Args:
            train_data: 训练数据（向量方法不需要，保持接口一致）
        """
        # 构建词汇表
        all_words = set()
        documents = []

        for _, row in self.table_df.iterrows():
            text = f"{row['table']} {row.get('table_desc', '')}"
            words = set(w for w in self.tokenize(text) if len(w) > 1)
            all_words.update(words)
            documents.append(words)

        self.vocab = {word: idx for idx, word in enumerate(all_words)}

        # 计算IDF
        n_docs = len(documents)
        word_doc_count = defaultdict(int)
        for doc_words in documents:
            for word in doc_words:
                word_doc_count[word] += 1

        self.idf = {
            word: math.log(n_docs / (count + 1))
            for word, count in word_doc_count.items()
        }

        # 构建文档向量
        for idx, doc_words in enumerate(documents):
            vector = np.zeros(len(self.vocab))

            # 计算TF
            word_count = defaultdict(int)
            for word in doc_words:
                word_count[word] += 1

            # 计算TF-IDF
            for word, count in word_count.items():
                if word in self.vocab:
                    vector[self.vocab[word]] = self.idf.get(word, 1.0) * count

            # L2归一化
            norm = np.linalg.norm(vector)
            if norm > 0:
                vector = vector / norm

            table_name = self.table_df.iloc[idx]['table']
            self.vectors[table_name] = vector

        self._is_fitted = True

    def _retrieve(self, query: str, k: int = 5) -> List[RetrievalResult]:
        """
        向量检索

        Args:
            query: 查询文本
            k: 返回的top-k结果

        Returns:
            检索结果列表
        """
        # 构建查询向量
        query_words = [w for w in self.tokenize(query) if len(w) > 1]
        query_term_set = set(query_words)
        query_vector = np.zeros(len(self.vocab))

        # 计算查询TF
        word_count = defaultdict(int)
        for word in query_words:
            word_count[word] += 1

        # 计算查询TF-IDF
        for word, count in word_count.items():
            if word in self.vocab:
                query_vector[self.vocab[word]] = self.idf.get(word, 1.0) * count

        # 归一化
        norm = np.linalg.norm(query_vector)
        if norm > 0:
            query_vector = query_vector / norm

        # 计算余弦相似度
        scores = {}
        for table_name, vector in self.vectors.items():
            similarity = np.dot(query_vector, vector)
            scores[table_name] = similarity

        # 排序获取top-k
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]

        results = []
        for table_name, score in sorted_scores:
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

                # 匹配得分
                overlap = len(query_term_set & field_terms)
                col_score = overlap / len(query_words) if query_words else 0

                # 加入字段名本身的匹配
                if any(term in field_name for term in query_term_set):
                    col_score += 0.5

                column_scores.append((f"C:{table_name}.{field_name}", col_score))

            # 按得分排序，取前8个字段
            column_scores.sort(key=lambda x: x[1], reverse=True)
            column_list = column_scores[:8]

            results.append(RetrievalResult(
                table=table_name,
                table_score=float(score),
                columns=column_list,
                metadata={'method': 'Vector', 'similarity': 'cosine'}
            ))

        return results
