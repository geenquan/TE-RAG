"""
TE-RAG检索器实现

表格增强检索增强生成框架

性能优化版本：
1. 预计算训练数据的分词结果
2. 缓存查询分词结果
3. 减少不必要的图查询
"""

import pandas as pd
import numpy as np
import jieba
import math
import re
import networkx as nx
from collections import defaultdict
from functools import lru_cache
from typing import Dict, List, Optional, Set

from retrievers.base_retriever import (
    BaseRetriever, RetrieverConfig, RetrievalResult
)


class TERAGRetriever(BaseRetriever):
    """
    TE-RAG检索器

    基于二分图、模板挖掘和模式匹配的增强检索器
    """

    def __init__(self, field_csv: str, table_csv: str,
                 config: Optional[RetrieverConfig] = None):
        """
        初始化TE-RAG检索器

        Args:
            field_csv: 字段CSV路径
            table_csv: 表CSV路径
            config: 检索器配置
        """
        if config is None:
            config = RetrieverConfig(
                name="TE-RAG",
                description="Table-Enhanced RAG with bipartite graph and template mining"
            )

        super().__init__(field_csv, table_csv, config)

        # 二分图
        self.graph = nx.Graph()

        # 角色权重（优化版本）
        self.role_weights = {
            'SELECT': 1.2,
            'WHERE': 2.0,    # WHERE子句权重最高
            'JOIN': 1.8,
            'GROUP_BY': 1.5,
            'ORDER_BY': 1.3,
            'FROM': 1.0,
        }

        # 元素到查询的映射
        self.element_to_queries: Dict[str, Set[str]] = defaultdict(set)

        # 模板库
        self.template_library: Dict[str, List[Dict]] = defaultdict(list)

        # 注解库
        self.annotations: Dict[str, Dict] = {}

        # 倒排索引
        self.inverted_index: Dict[str, List[tuple]] = defaultdict(list)
        self.document_norms: Dict[str, float] = {}

        # 训练数据
        self.train_data: Optional[pd.DataFrame] = None

        # 性能优化：预计算分词结果缓存
        self._train_tokens_cache: Dict[int, Set[str]] = {}  # idx -> tokens
        self._query_tokens_cache: Dict[str, Set[str]] = {}  # query -> tokens

    def fit(self, train_data: pd.DataFrame = None):
        """
        训练TE-RAG

        Args:
            train_data: 训练数据，包含question, table, field, sql列
        """
        if train_data is not None and not train_data.empty:
            self.train_data = train_data

            # 性能优化：预计算所有训练数据的分词结果
            print("  预计算分词结果...")
            for idx in range(len(train_data)):
                question = train_data.iloc[idx]['question']
                self._train_tokens_cache[idx] = set(self.tokenize(question))

            self._build_bipartite_graph(train_data)
            self._extract_templates()
            self._build_inverted_index()

        self._is_fitted = True

    def _build_bipartite_graph(self, train_data: pd.DataFrame):
        """
        构建二分图

        Args:
            train_data: 训练数据
        """
        # 添加查询节点
        for idx in range(len(train_data)):
            self.graph.add_node(f"Q_{idx}", bipartite=0, node_type='query')

        # 添加表节点
        for _, row in self.table_df.iterrows():
            table_node = f"T:{row['table']}"
            self.graph.add_node(table_node, bipartite=1, node_type='table')

        # 添加列节点
        for _, row in self.field_df.iterrows():
            col_node = f"C:{row['table']}.{row['field_name']}"
            self.graph.add_node(col_node, bipartite=1, node_type='column')

        # 添加边
        for idx, row in train_data.iterrows():
            query_node = f"Q_{idx}"
            table = row.get('table', '')
            fields = row.get('field', '')
            sql = row.get('sql', '')

            # 根据SQL角色确定权重（优化版本）
            weight = 1.0
            if pd.notna(sql):
                sql_upper = str(sql).upper()
                if 'WHERE' in sql_upper:
                    weight = max(weight, 2.0)
                if 'JOIN' in sql_upper:
                    weight = max(weight, 1.8)
                if 'GROUP BY' in sql_upper:
                    weight = max(weight, 1.5)
                if 'ORDER BY' in sql_upper:
                    weight = max(weight, 1.3)
                if 'SELECT' in sql_upper:
                    weight = max(weight, 1.2)

            # 添加列边
            if pd.notna(fields) and isinstance(fields, str):
                for field in fields.split('|'):
                    field = field.strip()
                    if field:
                        table_simple = table.split('.')[-1] if pd.notna(table) else ''
                        col_node = f"C:{table_simple}.{field}"
                        if self.graph.has_node(col_node):
                            self.graph.add_edge(query_node, col_node, weight=weight)
                            self.element_to_queries[col_node].add(row['question'])

            # 添加表边
            if pd.notna(table):
                table_simple = table.split('.')[-1]
                table_node = f"T:{table_simple}"
                if self.graph.has_node(table_node):
                    self.graph.add_edge(query_node, table_node, weight=1.0)
                    self.element_to_queries[table_node].add(row['question'])

    def _extract_templates(self):
        """提取模板"""
        for element_node in self.element_to_queries:
            queries = list(self.element_to_queries[element_node])

            for query in queries[:10]:
                # 提取实体-属性模式
                words = list(jieba.cut(query))
                pattern = self._extract_pattern(words, query)

                if pattern:
                    self.template_library[element_node].append({
                        'pattern': pattern,
                        'query': query
                    })

    def _extract_pattern(self, words: List[str], query: str) -> str:
        """提取查询模式"""
        entity_keywords = ['公司', '供电所', '单位', '用户', '客户']
        attribute_keywords = ['售电量', '电费', '欠费', '回收率', '户数']

        entities = [w for w in words if any(k in w for k in entity_keywords)]
        attributes = [w for w in words if any(k in w for k in attribute_keywords)]

        parts = []
        if entities:
            parts.append(f"ENTITY:{'|'.join(entities)}")
        if attributes:
            parts.append(f"ATTR:{'|'.join(attributes)}")

        # 时间模式
        if re.search(r'\d{4}年\d{1,2}月', query):
            parts.append("TIME:年月")

        return ' '.join(parts) if parts else ''

    def _build_inverted_index(self):
        """构建倒排索引"""
        for table_node in [n for n in self.graph.nodes if n.startswith('T:')]:
            table_name = table_node.replace('T:', '')

            # 获取表信息
            table_info = self.table_df[self.table_df['table'] == table_name]
            table_desc = table_info.iloc[0]['table_desc'] if not table_info.empty else ''

            # 获取列
            columns = self.field_df[self.field_df['table'] == table_name]

            # 构建文档
            document_parts = []
            document_parts.append((table_name, 2.0))
            if pd.notna(table_desc):
                document_parts.append((table_desc, 1.5))
            for _, col_row in columns.iterrows():
                document_parts.append((col_row['field_name'], 1.0))
                if pd.notna(col_row.get('field_name_desc', '')):
                    document_parts.append((col_row['field_name_desc'], 0.8))

            # 添加模板
            if table_node in self.template_library:
                for template in self.template_library[table_node]:
                    document_parts.append((template['pattern'], 0.5))

            self._index_document(table_node, document_parts)

    def _index_document(self, doc_id: str, document_parts: List[tuple]):
        """索引文档"""
        term_frequencies = defaultdict(float)

        for text, weight in document_parts:
            words = self.tokenize(str(text))
            for word in words:
                if len(word) > 1:
                    term_frequencies[word] += weight

        # 计算文档长度
        doc_length = math.sqrt(sum(tf ** 2 for tf in term_frequencies.values()))
        self.document_norms[doc_id] = doc_length if doc_length > 0 else 1.0

        # 添加到索引
        for term, tf in term_frequencies.items():
            self.inverted_index[term].append((doc_id, tf, doc_length))

    def _bm25_retrieval(self, query_terms: List[str]) -> Dict[str, float]:
        """BM25检索"""
        scores = defaultdict(float)

        if not self.document_norms:
            return scores

        avgdl = np.mean(list(self.document_norms.values()))
        N = len(self.document_norms)

        for term in query_terms:
            if term not in self.inverted_index:
                continue

            postings = self.inverted_index[term]
            df = len(postings)
            idf = math.log((N - df + 0.5) / (df + 0.5) + 1)

            for doc_id, tf, doc_length in postings:
                # BM25公式
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_length / avgdl)
                score = idf * numerator / denominator

                scores[doc_id] += score

        return scores

    def _template_match_score(self, table_node: str, query: str) -> float:
        """模板匹配得分（优化版本）"""
        if table_node not in self.template_library:
            return 0.0

        query_words = self._get_query_tokens(query)  # 使用缓存
        max_score = 0.0

        for template in self.template_library[table_node]:
            pattern = template['pattern']
            pattern_words = set(self.tokenize(pattern))

            if pattern_words:
                overlap = len(query_words & pattern_words) / len(pattern_words)
                max_score = max(max_score, overlap)

        return max_score

    def _get_query_tokens(self, query: str) -> Set[str]:
        """获取查询的分词结果（带缓存）"""
        if query not in self._query_tokens_cache:
            self._query_tokens_cache[query] = set(self.tokenize(query))
        return self._query_tokens_cache[query]

    def _graph_propagation_score(self, col_node: str, query: str) -> float:
        """图传播得分（优化版本）"""
        if self.train_data is None:
            return 0.0

        score = 0.0
        query_words = self._get_query_tokens(query)  # 使用缓存

        # 性能优化：先检查列节点是否存在
        if not self.graph.has_node(col_node):
            return 0.0

        for idx in range(len(self.train_data)):
            # 使用预计算的分词结果
            hist_words = self._train_tokens_cache.get(idx, set())

            if not hist_words:
                continue

            # 快速计算Jaccard相似度
            intersection = len(query_words & hist_words)
            if intersection == 0:
                continue

            union = len(query_words | hist_words)
            sim = intersection / union if union > 0 else 0

            if sim > 0.3:
                query_node = f"Q_{idx}"
                if self.graph.has_edge(query_node, col_node):
                    weight = self.graph[query_node][col_node].get('weight', 1.0)
                    score += sim * weight

        return min(score, 5.0)

    def _retrieve(self, query: str, k: int = 5) -> List[RetrievalResult]:
        """
        TE-RAG检索（优化版本）

        Args:
            query: 查询文本
            k: 返回的top-k结果

        Returns:
            检索结果列表
        """
        query_terms = [t for t in self.tokenize(query) if len(t) > 1]
        query_term_set = set(query_terms)

        # 1. BM25检索
        table_scores = self._bm25_retrieval(query_terms)

        # 2. 模板匹配增强
        for table_node in list(table_scores.keys()):
            template_score = self._template_match_score(table_node, query)
            table_scores[table_node] += template_score * 0.6  # 增加模板权重

        # 3. 图传播增强（基于训练数据中的相似查询）- 这是TE-RAG的核心优势
        if self.train_data is not None:
            for idx in range(len(self.train_data)):
                hist_question = self.train_data.iloc[idx]['question']
                hist_words = self._train_tokens_cache.get(idx, set())

                # 计算相似度
                intersection = len(query_term_set & hist_words)
                if intersection > 0:
                    union = len(query_term_set | hist_words)
                    sim = intersection / union

                    if sim > 0.15:  # 降低阈值，增加召回
                        # 获取该历史查询对应的表
                        table = self.train_data.iloc[idx].get('table', '')
                        if pd.notna(table):
                            table_simple = table.split('.')[-1]
                            table_node = f"T:{table_simple}"
                            if table_node in table_scores:
                                # 图传播增强：利用训练数据中的表-查询关联
                                table_scores[table_node] += sim * 3.0

        # 排序
        top_tables = sorted(table_scores.items(), key=lambda x: x[1], reverse=True)[:k]

        results = []
        for table_node, table_score in top_tables:
            table_name = table_node.replace('T:', '')

            # 获取该表的列
            columns = self.field_df[self.field_df['table'] == table_name]

            # 计算每个字段的得分
            column_scores = {}
            for _, col_row in columns.iterrows():
                field_name = col_row['field_name']
                field_desc = str(col_row.get('field_name_desc', ''))
                col_node = f"C:{table_name}.{field_name}"

                # 1. 直接匹配得分（字段名和描述）
                field_text = f"{field_name} {field_desc}"
                field_terms = set(t for t in self.tokenize(field_text) if len(t) > 1)
                overlap = len(query_term_set & field_terms)
                s_direct = overlap / max(len(query_terms), 1) * 1.5

                # 字段名直接包含查询词
                for term in query_terms:
                    if term in field_name or term in field_desc:
                        s_direct += 0.4

                # 2. 图传播得分
                s_graph = self._graph_propagation_score(col_node, query)

                # 3. 模式匹配得分
                s_pattern = self._pattern_match_score(col_node, query)

                # 4. 训练数据中的字段推荐
                s_train = self._train_field_score(table_name, field_name, query)

                # 加权组合（优化权重，确保最佳效果）
                base_score = 0.40 * s_direct + 0.25 * s_train
                enhancement = 0.20 * min(s_graph, 2.0) + 0.15 * s_pattern
                column_scores[col_node] = base_score + enhancement

            # 排序列，返回更多字段
            top_columns = sorted(column_scores.items(), key=lambda x: x[1], reverse=True)[:10]

            results.append(RetrievalResult(
                table=table_name,
                table_score=table_score,
                columns=top_columns,
                metadata={'method': 'TE-RAG'}
            ))

        return results

    def _train_field_score(self, table_name: str, field_name: str, query: str) -> float:
        """基于训练数据的字段推荐得分"""
        if self.train_data is None:
            return 0.0

        score = 0.0
        query_words = self._get_query_tokens(query)

        for idx in range(len(self.train_data)):
            row = self.train_data.iloc[idx]

            # 检查是否是同一个表
            train_table = row.get('table', '')
            if pd.isna(train_table):
                continue
            train_table_simple = train_table.split('.')[-1]

            if train_table_simple != table_name:
                continue

            # 计算查询相似度
            hist_words = self._train_tokens_cache.get(idx, set())
            if not hist_words:
                continue

            sim = len(query_words & hist_words) / len(query_words | hist_words) if (query_words | hist_words) else 0

            if sim > 0.3:
                # 检查该字段是否在训练数据的字段列表中
                fields = row.get('field', '')
                if pd.notna(fields) and isinstance(fields, str):
                    field_list = [f.strip() for f in fields.split('|') if f.strip()]
                    if field_name in field_list:
                        score += sim * 1.5

        return min(score, 3.0)

    def _direct_match_score(self, col_node: str, query_terms: List[str]) -> float:
        """直接匹配得分"""
        score = 0.0

        if col_node in self.annotations:
            desc = self.annotations[col_node].get('description', '')
            if pd.notna(desc) and isinstance(desc, str):
                desc_words = set(self.tokenize(desc))
                score += len(desc_words & set(query_terms)) * 0.5

        return score

    def _pattern_match_score(self, col_node: str, query: str) -> float:
        """模式匹配得分（优化版本）"""
        if col_node not in self.template_library:
            return 0.0

        query_words = self._get_query_tokens(query)  # 使用缓存
        max_sim = 0.0

        for template in self.template_library[col_node]:
            pattern = template['pattern']
            pattern_words = set(self.tokenize(pattern))

            if pattern_words:
                sim = len(query_words & pattern_words) / len(pattern_words)
                max_sim = max(max_sim, sim)

        return max_sim
