"""
向量检索器实现

基于语义相似度的表格检索器（使用字段级别的语义匹配）
与 BM25 形成差异：BM25 专注于表名关键词，Vector 专注于字段语义
"""

import pandas as pd
import numpy as np
import jieba
import math
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from retrievers.base_retriever import (
    BaseRetriever, RetrieverConfig, RetrievalResult
)


class VectorRetriever(BaseRetriever):
    """
    向量检索器

    使用语义相似度进行表格检索
    与 BM25 的差异：
    - BM25: 基于表名和描述的关键词匹配
    - Vector: 基于字段语义的相似度匹配
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
                description="Semantic vector retriever (field-level)"
            )

        super().__init__(field_csv, table_csv, config)

        # 向量相关
        self.table_vectors: Dict[str, np.ndarray] = {}
        self.field_vectors: Dict[str, np.ndarray] = {}
        self.vocab: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}

    def fit(self, train_data: pd.DataFrame = None):
        """
        构建向量索引

        Args:
            train_data: 训练数据（向量方法不需要，保持接口一致）
        """
        # 构建词汇表（基于字段描述，而不是表名）
        all_words = set()
        field_docs = []  # 每个字段作为一个文档
        table_field_map = defaultdict(list)  # 表名 -> 字段索引列表

        for idx, row in self.field_df.iterrows():
            # 使用字段名和字段描述作为文档内容
            field_name = str(row.get('field_name', ''))
            field_desc = str(row.get('field_name_desc', ''))
            table_name = str(row.get('table', ''))

            # 组合文本，但更侧重于描述（语义信息）
            text = f"{field_desc} {field_name}"
            words = [w for w in self.tokenize(text) if len(w) > 1]

            all_words.update(words)
            field_docs.append(words)
            table_field_map[table_name].append(len(field_docs) - 1)

        self.vocab = {word: idx for idx, word in enumerate(all_words)}

        # 计算 IDF（基于字段文档）
        n_docs = len(field_docs)
        word_doc_count = defaultdict(int)
        for doc_words in field_docs:
            for word in set(doc_words):  # 使用 set 避免重复计数
                word_doc_count[word] += 1

        self.idf = {
            word: math.log(n_docs / (count + 1))
            for word, count in word_doc_count.items()
        }

        # 构建字段向量
        for idx, doc_words in enumerate(field_docs):
            vector = self._build_tfidf_vector(doc_words)
            table_name = self.field_df.iloc[idx]['table']
            field_name = self.field_df.iloc[idx]['field_name']
            self.field_vectors[f"{table_name}.{field_name}"] = vector

        # 构建表向量（聚合该表所有字段的向量）
        for table_name in self.table_df['table'].unique():
            table_fields = self.field_df[self.field_df['table'] == table_name]
            if len(table_fields) == 0:
                continue

            # 聚合所有字段的向量（平均）
            field_vecs = []
            for _, row in table_fields.iterrows():
                key = f"{table_name}.{row['field_name']}"
                if key in self.field_vectors:
                    field_vecs.append(self.field_vectors[key])

            if field_vecs:
                # 使用加权平均，给有描述的字段更高的权重
                weights = []
                for _, row in table_fields.iterrows():
                    desc = str(row.get('field_name_desc', ''))
                    weight = 1.0 + 0.5 * min(len(desc) / 10, 1.0)  # 描述越长权重越高
                    weights.append(weight)

                weights = np.array(weights)
                weights = weights / weights.sum()

                table_vector = np.zeros(len(self.vocab))
                for vec, w in zip(field_vecs, weights):
                    table_vector += vec * w

                # L2 归一化
                norm = np.linalg.norm(table_vector)
                if norm > 0:
                    table_vector = table_vector / norm

                self.table_vectors[table_name] = table_vector

        self._is_fitted = True

    def _build_tfidf_vector(self, words: List[str]) -> np.ndarray:
        """构建 TF-IDF 向量"""
        vector = np.zeros(len(self.vocab))

        # 计算 TF
        word_count = defaultdict(int)
        for word in words:
            word_count[word] += 1

        # 计算 TF-IDF
        for word, count in word_count.items():
            if word in self.vocab:
                # 使用对数 TF，减少高频词的影响
                tf = 1 + math.log(count) if count > 0 else 0
                vector[self.vocab[word]] = tf * self.idf.get(word, 1.0)

        # L2 归一化
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm

        return vector

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
        query_vector = self._build_tfidf_vector(query_words)
        query_term_set = set(query_words)

        # 计算与每个表的余弦相似度
        scores = {}
        for table_name, vector in self.table_vectors.items():
            similarity = np.dot(query_vector, vector)
            scores[table_name] = similarity

        # 排序获取 top-k
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]

        results = []
        for table_name, score in sorted_scores:
            # 获取该表的字段，并根据语义相似度排序
            table_fields = self.field_df[self.field_df['table'] == table_name]

            # 计算每个字段与查询的语义相似度
            column_scores = []
            for _, row in table_fields.iterrows():
                field_name = row['field_name']
                field_desc = str(row.get('field_name_desc', ''))

                # 使用字段向量计算相似度
                key = f"{table_name}.{field_name}"
                if key in self.field_vectors:
                    field_vec = self.field_vectors[key]
                    field_sim = np.dot(query_vector, field_vec)
                else:
                    # 回退到简单的词汇重叠
                    field_text = f"{field_name} {field_desc}"
                    field_terms = set(t for t in self.tokenize(field_text) if len(t) > 1)
                    overlap = len(query_term_set & field_terms)
                    field_sim = overlap / len(query_words) if query_words else 0

                column_scores.append((f"C:{table_name}.{field_name}", field_sim))

            # 按相似度排序，取前8个字段
            column_scores.sort(key=lambda x: x[1], reverse=True)
            column_list = column_scores[:8]

            results.append(RetrievalResult(
                table=table_name,
                table_score=float(score),
                columns=column_list,
                metadata={'method': 'Vector', 'similarity': 'semantic'}
            ))

        return results
