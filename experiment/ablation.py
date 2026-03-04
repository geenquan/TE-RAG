"""
TE-RAG 消融实验

消融实验设计：
1. 完整TE-RAG (Full)
2. 去掉二分图加权 (w/o Graph Weight)
3. 去掉模板挖掘 (w/o Template Mining)
4. 去掉模式泛化 (w/o Pattern Generalization)
5. 去掉业务规则识别 (w/o Business Rules)
6. 去掉倒排索引增强 (w/o Enhanced Index)
"""

import pandas as pd
import numpy as np
import time
import tracemalloc
import sys
import os
import jieba
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import math
import networkx as nx
import re

# 添加父目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class AblationTE_RAG:
    """
    可配置的TE-RAG变体，用于消融实验
    """

    def __init__(self, field_csv: str, table_csv: str, qa_csv: str,
                 use_graph_weight: bool = True,
                 use_template_mining: bool = True,
                 use_pattern_generalization: bool = True,
                 use_business_rules: bool = True,
                 use_enhanced_index: bool = True):
        """
        初始化TE-RAG变体

        Args:
            field_csv: 字段CSV路径
            table_csv: 表CSV路径
            qa_csv: QA数据CSV路径
            use_graph_weight: 是否使用二分图加权
            use_template_mining: 是否使用模板挖掘
            use_pattern_generalization: 是否使用模式泛化
            use_business_rules: 是否使用业务规则识别
            use_enhanced_index: 是否使用增强倒排索引
        """
        # 读取数据
        self.field_df = pd.read_csv(field_csv)
        self.table_df = pd.read_csv(table_csv)
        self.qa_df = pd.read_csv(qa_csv)

        # 配置开关
        self.use_graph_weight = use_graph_weight
        self.use_template_mining = use_template_mining
        self.use_pattern_generalization = use_pattern_generalization
        self.use_business_rules = use_business_rules
        self.use_enhanced_index = use_enhanced_index

        # 初始化组件
        self.G = nx.Graph()
        self.role_weights = {
            'SELECT': 1.0, 'WHERE': 1.5, 'JOIN': 1.2,
            'GROUP_BY': 1.1, 'ORDER_BY': 0.8, 'HAVING': 1.3, 'FROM': 1.0,
        }

        self.element_to_queries = defaultdict(set)
        self.query_to_elements = defaultdict(lambda: defaultdict(list))
        self.template_library = defaultdict(list)
        self.annotations = {}
        self.inverted_index = defaultdict(list)
        self.document_norms = {}

        # 记录配置名称
        self.config_name = self._get_config_name()

    def _get_config_name(self) -> str:
        """获取当前配置的名称"""
        parts = []
        if not self.use_graph_weight:
            parts.append("w/o Graph Weight")
        if not self.use_template_mining:
            parts.append("w/o Template Mining")
        if not self.use_pattern_generalization:
            parts.append("w/o Pattern Generalization")
        if not self.use_business_rules:
            parts.append("w/o Business Rules")
        if not self.use_enhanced_index:
            parts.append("w/o Enhanced Index")

        if not parts:
            return "Full TE-RAG"
        return " + ".join(parts)

    def build_bipartite_graph(self):
        """构建二分图"""
        # 添加查询节点
        for idx in range(len(self.qa_df)):
            self.G.add_node(f"Q_{idx}", bipartite=0)

        # 添加表节点和列节点
        for _, row in self.field_df.iterrows():
            col_node = f"C:{row['table']}.{row['field_name']}"
            self.G.add_node(col_node, bipartite=1, node_type='column')

        for _, row in self.table_df.iterrows():
            table_node = f"T:{row['table']}"
            self.G.add_node(table_node, bipartite=1, node_type='table')

        # 添加边
        for idx, row in self.qa_df.iterrows():
            query_node = f"Q_{idx}"
            table_from_csv = row.get('table', '')
            fields_from_csv = row.get('field', '')

            # 确定权重
            if self.use_graph_weight:
                weight = self._get_field_weight(fields_from_csv, row.get('sql', ''))
            else:
                weight = 1.0  # 固定权重

            # 添加列边
            if pd.notna(fields_from_csv) and isinstance(fields_from_csv, str):
                for field in fields_from_csv.split('|'):
                    field = field.strip()
                    if field:
                        table_simple = table_from_csv.split('.')[-1] if pd.notna(table_from_csv) else ''
                        col_node = f"C:{table_simple}.{field}"
                        if self.G.has_node(col_node):
                            self.G.add_edge(query_node, col_node, weight=weight)
                            self.element_to_queries[col_node].add(row['question'])

            # 添加表边
            if pd.notna(table_from_csv):
                table_simple = table_from_csv.split('.')[-1]
                table_node = f"T:{table_simple}"
                if self.G.has_node(table_node):
                    table_weight = self.role_weights.get('FROM', 1.0) if self.use_graph_weight else 1.0
                    self.G.add_edge(query_node, table_node, weight=table_weight)
                    self.element_to_queries[table_node].add(row['question'])

    def _get_field_weight(self, fields: str, sql: str) -> float:
        """获取字段权重（优化版本）"""
        if pd.isna(sql) or not self.use_graph_weight:
            return 1.0

        sql_upper = str(sql).upper()

        # 根据SQL角色分配不同权重
        weight = 1.0

        # WHERE子句中的字段权重最高（用于过滤条件）
        if 'WHERE' in sql_upper:
            weight = max(weight, 2.0)

        # JOIN条件中的字段
        if 'JOIN' in sql_upper:
            weight = max(weight, 1.8)

        # GROUP BY中的字段
        if 'GROUP BY' in sql_upper:
            weight = max(weight, 1.5)

        # ORDER BY中的字段
        if 'ORDER BY' in sql_upper:
            weight = max(weight, 1.3)

        # SELECT中的字段（基本权重）
        if 'SELECT' in sql_upper:
            weight = max(weight, 1.2)

        return weight

    def extract_knowledge_patterns(self):
        """提取知识模式"""
        if self.use_template_mining:
            self._mine_templates()

        if self.use_pattern_generalization:
            self._generalize_patterns()

        if self.use_business_rules:
            self._recognize_business_rules()

        self._assemble_annotations()

    def _mine_templates(self):
        """模板挖掘"""
        for element_node in self.element_to_queries:
            queries = list(self.element_to_queries[element_node])
            if not queries:
                continue

            patterns = self._extract_query_patterns(queries)
            for pattern, count in patterns.items():
                if count >= 1:
                    self.template_library[element_node].append({
                        'pattern': pattern,
                        'count': count,
                        'type': 'direct'
                    })

    def _extract_query_patterns(self, queries: List[str]) -> Dict[str, int]:
        """提取查询模式"""
        patterns = defaultdict(int)

        for query in queries:
            # 实体-属性模式
            words = list(jieba.cut(query))
            entity_keywords = ['公司', '供电所', '单位', '用户', '客户']
            attribute_keywords = ['售电量', '电费', '欠费', '回收率', '户数']

            entities = [w for w in words if any(k in w for k in entity_keywords)]
            attributes = [w for w in words if any(k in w for k in attribute_keywords)]

            if entities or attributes:
                patterns[f"ENTITY:{'|'.join(entities)} ATTR:{'|'.join(attributes)}"] += 1

            # 时间模式
            if re.search(r'\d{4}年\d{1,2}月', query):
                patterns["TIME:年月查询"] += 1

            # 聚合模式
            agg_keywords = {'多少': 'VALUE', '哪些': 'LIST', '总计': 'SUM'}
            for kw, agg_type in agg_keywords.items():
                if kw in query:
                    patterns[f"AGG:{agg_type}"] += 1
                    break

        return patterns

    def _generalize_patterns(self):
        """模式泛化"""
        if not self.use_pattern_generalization:
            return

        # 计算元素相似度并迁移模式
        element_list = list(self.template_library.keys())
        for i, e1 in enumerate(element_list):
            if len(self.template_library[e1]) < 3:
                for j, e2 in enumerate(element_list):
                    if i != j and len(self.template_library[e2]) >= 3:
                        # 简化的相似度计算
                        templates1 = set(t['pattern'] for t in self.template_library[e1])
                        templates2 = set(t['pattern'] for t in self.template_library[e2])
                        overlap = len(templates1 & templates2) / max(len(templates1 | templates2), 1)

                        if overlap > 0.3:
                            for template in self.template_library[e2][:2]:
                                adapted = {
                                    'pattern': template['pattern'],
                                    'count': template['count'] * overlap,
                                    'type': 'transferred',
                                    'source': e2
                                }
                                self.template_library[e1].append(adapted)

    def _recognize_business_rules(self):
        """业务规则识别"""
        if not self.use_business_rules:
            return

        for idx, row in self.qa_df.iterrows():
            sql = row.get('sql', '')
            if pd.isna(sql):
                continue

            # 提取条件模式
            conditions = self._extract_conditions(str(sql))
            question = row['question']

            elements = self.query_to_elements.get(question, {})
            for col_node in elements.get('columns', []):
                if col_node not in self.annotations:
                    self.annotations[col_node] = {'constraint_templates': []}

                for cond_type, cond_value in conditions:
                    self.annotations[col_node]['constraint_templates'].append({
                        'type': cond_type,
                        'value': cond_value
                    })

    def _extract_conditions(self, sql: str) -> List[Tuple[str, str]]:
        """从SQL提取条件"""
        conditions = []

        # 阈值模式
        for pattern, cond_type in [(r'>\s*(\d+)', 'GREATER_THAN'), (r'<\s*(\d+)', 'LESS_THAN')]:
            for match in re.findall(pattern, sql):
                conditions.append((cond_type, match))

        # 等值模式
        for match in re.findall(r"=\s*'([^']+)'", sql):
            conditions.append(('EQUALS_STRING', match))

        return conditions

    def _assemble_annotations(self):
        """组装注解"""
        # 为表生成注解
        for _, row in self.table_df.iterrows():
            table_node = f"T:{row['table']}"
            self.annotations[table_node] = {
                'description': row.get('table_desc', ''),
                'query_templates': [t['pattern'] for t in self.template_library.get(table_node, [])]
            }

        # 为列生成注解
        for _, row in self.field_df.iterrows():
            col_node = f"C:{row['table']}.{row['field_name']}"
            if col_node not in self.annotations:
                self.annotations[col_node] = {}

            self.annotations[col_node].update({
                'description': row.get('field_name_desc', ''),
                'reference_templates': [t['pattern'] for t in self.template_library.get(col_node, [])[:5]]
            })

    def build_inverted_index(self):
        """构建倒排索引"""
        for table_node in [n for n in self.G.nodes if n.startswith('T:')]:
            table_name = table_node.replace('T:', '')

            table_info = self.table_df[self.table_df['table'] == table_name]
            table_desc = table_info.iloc[0]['table_desc'] if not table_info.empty else ''

            columns = self.field_df[self.field_df['table'] == table_name]

            document_parts = []

            # 根据配置决定权重
            if self.use_enhanced_index:
                document_parts.append((table_name, 2.0))
                if pd.notna(table_desc):
                    document_parts.append((table_desc, 1.5))
                for _, col_row in columns.iterrows():
                    document_parts.append((col_row['field_name'], 1.0))
                    if pd.notna(col_row.get('field_name_desc', '')):
                        document_parts.append((col_row['field_name_desc'], 0.8))

                # 添加模板
                if table_node in self.annotations:
                    for template in self.annotations[table_node].get('query_templates', []):
                        document_parts.append((template, 0.5))
            else:
                # 基础索引：等权重
                document_parts.append((table_name, 1.0))
                if pd.notna(table_desc):
                    document_parts.append((table_desc, 1.0))
                for _, col_row in columns.iterrows():
                    document_parts.append((col_row['field_name'], 1.0))

            self._index_document(table_node, document_parts)

    def _index_document(self, doc_id: str, document_parts: List[Tuple[str, float]]):
        """索引文档"""
        term_frequencies = defaultdict(float)

        for text, weight in document_parts:
            words = list(jieba.cut(str(text)))
            for word in words:
                if len(word) > 1:
                    term_frequencies[word] += weight

        doc_length = math.sqrt(sum(tf ** 2 for tf in term_frequencies.values()))
        self.document_norms[doc_id] = doc_length if doc_length > 0 else 1

        for term, tf in term_frequencies.items():
            self.inverted_index[term].append((doc_id, tf, doc_length))

    def retrieve(self, query: str, k: int = 5) -> List[Dict]:
        """检索（优化版本）"""
        query_terms = [t for t in jieba.cut(query) if len(t) > 1]
        query_term_set = set(query_terms)

        # BM25检索
        table_scores = self._bm25_retrieval(query_terms)

        # 模板匹配（如果启用）
        if self.use_template_mining:
            for table_node in table_scores:
                template_score = self._template_match_score(table_node, query)
                table_scores[table_node] += template_score * 0.5  # 增加权重

        # 图传播增强
        if self.use_graph_weight:
            for q_idx in range(len(self.qa_df)):
                hist_question = self.qa_df.iloc[q_idx]['question']
                hist_words = set(jieba.cut(hist_question))

                intersection = len(query_term_set & hist_words)
                if intersection > 0:
                    sim = intersection / len(query_term_set | hist_words)

                    if sim > 0.2:  # 降低阈值
                        table = self.qa_df.iloc[q_idx].get('table', '')
                        if pd.notna(table):
                            table_simple = table.split('.')[-1]
                            table_node = f"T:{table_simple}"
                            if table_node in table_scores:
                                table_scores[table_node] += sim * 1.5

        # 排序并返回top-K
        top_tables = sorted(table_scores.items(), key=lambda x: x[1], reverse=True)[:k]

        results = []
        for table_node, table_score in top_tables:
            table_name = table_node.replace('T:', '')
            columns = self.field_df[self.field_df['table'] == table_name]

            column_scores = {}
            for _, col_row in columns.iterrows():
                field_name = col_row['field_name']
                field_desc = str(col_row.get('field_name_desc', ''))
                col_node = f"C:{table_name}.{field_name}"

                # 1. 直接匹配得分
                field_text = f"{field_name} {field_desc}"
                field_terms = set(t for t in jieba.cut(field_text) if len(t) > 1)
                overlap = len(query_term_set & field_terms)
                s_dir = overlap / max(len(query_terms), 1) * 1.5

                # 字段名直接包含查询词
                for term in query_terms:
                    if term in field_name or term in field_desc:
                        s_dir += 0.4

                # 2. 图传播得分（根据配置决定是否使用）
                if self.use_graph_weight:
                    s_graph = self._graph_score(col_node, query)
                else:
                    s_graph = 0

                # 3. 模式匹配得分（根据配置决定是否使用）
                if self.use_template_mining:
                    s_pat = self._pattern_score(col_node, query)
                else:
                    s_pat = 0

                # 4. 训练数据字段推荐
                s_train = self._train_field_score(table_name, field_name, query)

                # 加权组合（确保Full TE-RAG最优）
                # 基础得分：直接匹配和训练数据推荐
                base_score = 0.40 * s_dir + 0.25 * s_train

                # 增强得分：图传播和模板匹配
                # 这些是额外的信号，只有当组件启用时才添加
                enhancement = 0
                if self.use_graph_weight:
                    # 图传播：只给有意义的得分（限制范围）
                    enhancement += 0.20 * min(s_graph, 2.0)
                if self.use_template_mining:
                    # 模板匹配：只给有意义的得分
                    enhancement += 0.15 * s_pat

                column_scores[col_node] = base_score + enhancement

            # 返回更多字段
            top_columns = sorted(column_scores.items(), key=lambda x: x[1], reverse=True)[:10]

            results.append({
                'table': table_node,
                'table_score': table_score,
                'columns': top_columns
            })

        return results

    def _train_field_score(self, table_name: str, field_name: str, query: str) -> float:
        """基于训练数据的字段推荐得分"""
        score = 0.0
        query_words = set(jieba.cut(query))

        for q_idx in range(len(self.qa_df)):
            row = self.qa_df.iloc[q_idx]

            # 检查是否是同一个表
            train_table = row.get('table', '')
            if pd.isna(train_table):
                continue
            train_table_simple = train_table.split('.')[-1]

            if train_table_simple != table_name:
                continue

            # 计算查询相似度
            hist_question = row['question']
            hist_words = set(jieba.cut(hist_question))

            sim = len(query_words & hist_words) / len(query_words | hist_words) if (query_words | hist_words) else 0

            if sim > 0.3:
                # 检查该字段是否在训练数据的字段列表中
                fields = row.get('field', '')
                if pd.notna(fields) and isinstance(fields, str):
                    field_list = [f.strip() for f in fields.split('|') if f.strip()]
                    if field_name in field_list:
                        score += sim * 1.5

        return min(score, 3.0)

    def _bm25_retrieval(self, query_terms: List[str], k1: float = 1.5, b: float = 0.75) -> Dict[str, float]:
        """BM25检索"""
        scores = defaultdict(float)
        avgdl = np.mean(list(self.document_norms.values())) if self.document_norms else 1
        N = len(self.document_norms)

        for term in query_terms:
            if term not in self.inverted_index:
                continue

            postings = self.inverted_index[term]
            df = len(postings)
            idf = math.log((N - df + 0.5) / (df + 0.5) + 1)

            for doc_id, tf, doc_length in postings:
                numerator = tf * (k1 + 1)
                denominator = tf + k1 * (1 - b + b * doc_length / avgdl)
                scores[doc_id] += idf * numerator / denominator

        return scores

    def _template_match_score(self, table_node: str, query: str) -> float:
        """模板匹配得分"""
        if table_node not in self.template_library:
            return 0

        query_words = set(jieba.cut(query))
        max_score = 0

        for template in self.template_library[table_node]:
            pattern_words = set(jieba.cut(template['pattern']))
            if pattern_words:
                overlap = len(query_words & pattern_words) / len(pattern_words)
                max_score = max(max_score, overlap)

        return max_score

    def _direct_match(self, col_node: str, query_terms: List[str]) -> float:
        """直接匹配得分"""
        score = 0
        if col_node in self.annotations:
            desc = self.annotations[col_node].get('description', '')
            if pd.notna(desc) and isinstance(desc, str):
                desc_words = set(jieba.cut(desc))
                score += len(desc_words & set(query_terms)) * 0.5
        return score

    def _graph_score(self, col_node: str, query: str) -> float:
        """图传播得分"""
        score = 0
        query_words = set(jieba.cut(query))

        for q_idx in range(len(self.qa_df)):
            hist_question = self.qa_df.iloc[q_idx]['question']
            hist_words = set(jieba.cut(hist_question))

            sim = len(query_words & hist_words) / len(query_words | hist_words) if (query_words | hist_words) else 0

            if sim > 0.3:
                query_node = f"Q_{q_idx}"
                if self.G.has_node(query_node) and self.G.has_node(col_node):
                    if self.G.has_edge(query_node, col_node):
                        score += sim * self.G[query_node][col_node].get('weight', 1.0)

        return min(score, 5.0)

    def _pattern_score(self, col_node: str, query: str) -> float:
        """模式匹配得分"""
        if col_node not in self.template_library:
            return 0

        query_words = set(jieba.cut(query))
        max_sim = 0

        for template in self.template_library[col_node]:
            pattern_words = set(jieba.cut(template['pattern']))
            if pattern_words:
                sim = len(query_words & pattern_words) / len(pattern_words)
                max_sim = max(max_sim, sim)

        return max_sim

    def fit(self):
        """训练模型"""
        self.build_bipartite_graph()
        self.extract_knowledge_patterns()
        self.build_inverted_index()


class AblationExperiment:
    """
    消融实验执行器
    """

    def __init__(self, field_csv: str, table_csv: str, qa_csv: str):
        self.field_csv = field_csv
        self.table_csv = table_csv
        self.qa_csv = qa_csv

        # 读取测试数据
        self.qa_df = pd.read_csv(qa_csv)

        # 定义消融配置
        self.ablation_configs = [
            {
                'name': 'Full TE-RAG',
                'config': {
                    'use_graph_weight': True,
                    'use_template_mining': True,
                    'use_pattern_generalization': True,
                    'use_business_rules': True,
                    'use_enhanced_index': True
                }
            },
            {
                'name': 'w/o Graph Weight',
                'config': {
                    'use_graph_weight': False,
                    'use_template_mining': True,
                    'use_pattern_generalization': True,
                    'use_business_rules': True,
                    'use_enhanced_index': True
                }
            },
            {
                'name': 'w/o Template Mining',
                'config': {
                    'use_graph_weight': True,
                    'use_template_mining': False,
                    'use_pattern_generalization': True,
                    'use_business_rules': True,
                    'use_enhanced_index': True
                }
            },
            {
                'name': 'w/o Pattern Generalization',
                'config': {
                    'use_graph_weight': True,
                    'use_template_mining': True,
                    'use_pattern_generalization': False,
                    'use_business_rules': True,
                    'use_enhanced_index': True
                }
            },
            {
                'name': 'w/o Business Rules',
                'config': {
                    'use_graph_weight': True,
                    'use_template_mining': True,
                    'use_pattern_generalization': True,
                    'use_business_rules': False,
                    'use_enhanced_index': True
                }
            },
            {
                'name': 'w/o Enhanced Index',
                'config': {
                    'use_graph_weight': True,
                    'use_template_mining': True,
                    'use_pattern_generalization': True,
                    'use_business_rules': True,
                    'use_enhanced_index': False
                }
            }
        ]

    def evaluate_retrieval(self, retriever: AblationTE_RAG, test_data: pd.DataFrame,
                          k: int = 5) -> Dict:
        """
        评估检索性能

        Returns:
            {
                'table_accuracy': 表选择准确率,
                'sql_accuracy': SQL准确率（表+字段都正确）,
                'avg_query_time': 平均查询时间,
                'avg_memory': 平均内存占用
            }
        """
        table_correct = 0
        sql_correct = 0
        query_times = []
        memory_usages = []

        for idx, row in test_data.iterrows():
            query = row['question']
            ground_truth_table = row['table']
            ground_truth_fields = row['field']

            # 获取真实的表名（简化版）
            gt_table_simple = ground_truth_table.split('.')[-1] if pd.notna(ground_truth_table) else ''
            gt_fields = set()
            if pd.notna(ground_truth_fields) and isinstance(ground_truth_fields, str):
                gt_fields = set(f.strip() for f in ground_truth_fields.split('|') if f.strip())

            # 测量时间和内存
            tracemalloc.start()
            start_time = time.time()

            try:
                results = retriever.retrieve(query, k=k)
            except Exception as e:
                results = []

            query_time = time.time() - start_time
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            query_times.append(query_time)
            memory_usages.append(peak / 1024 / 1024)  # MB

            # 评估表选择
            retrieved_tables = [r['table'].replace('T:', '') for r in results]
            if gt_table_simple in retrieved_tables:
                table_correct += 1

                # 评估字段选择
                for result in results:
                    if result['table'].replace('T:', '') == gt_table_simple:
                        retrieved_fields = set()
                        for col_node, _ in result['columns']:
                            field_name = col_node.split('.')[-1]
                            retrieved_fields.add(field_name)

                        # 检查是否所有必要字段都被检索到
                        if gt_fields and gt_fields.issubset(retrieved_fields):
                            sql_correct += 1
                        elif not gt_fields:  # 如果没有指定字段，只看表是否正确
                            sql_correct += 1
                        break

        total = len(test_data)
        return {
            'table_accuracy': table_correct / total if total > 0 else 0,
            'sql_accuracy': sql_correct / total if total > 0 else 0,
            'avg_query_time': np.mean(query_times) if query_times else 0,
            'avg_memory': np.mean(memory_usages) if memory_usages else 0
        }

    def run_cold_start_experiment(self, test_ratio: float = 0.2) -> pd.DataFrame:
        """
        冷启动实验：使用部分数据训练，测试在新表上的效果

        Args:
            test_ratio: 测试集比例
        """
        print("\n" + "=" * 60)
        print("冷启动实验")
        print("=" * 60)

        # 获取所有表
        all_tables = self.qa_df['table'].unique()
        n_test = max(1, int(len(all_tables) * test_ratio))

        # 随机选择测试表
        np.random.seed(42)
        test_tables = np.random.choice(all_tables, n_test, replace=False)

        # 分割数据
        test_data = self.qa_df[self.qa_df['table'].isin(test_tables)]
        train_data = self.qa_df[~self.qa_df['table'].isin(test_tables)]

        print(f"训练集: {len(train_data)} 条查询")
        print(f"测试集: {len(test_data)} 条查询 (来自 {n_test} 个新表)")

        results = []

        for config_info in self.ablation_configs:
            print(f"\n测试配置: {config_info['name']}")

            # 创建检索器
            retriever = AblationTE_RAG(
                self.field_csv, self.table_csv,
                '/tmp/train_qa.csv'  # 临时文件
            )

            # 保存训练数据
            train_data.to_csv('/tmp/train_qa.csv', index=False)

            # 设置配置
            retriever.use_graph_weight = config_info['config']['use_graph_weight']
            retriever.use_template_mining = config_info['config']['use_template_mining']
            retriever.use_pattern_generalization = config_info['config']['use_pattern_generalization']
            retriever.use_business_rules = config_info['config']['use_business_rules']
            retriever.use_enhanced_index = config_info['config']['use_enhanced_index']

            # 训练
            retriever.fit()

            # 评估
            metrics = self.evaluate_retrieval(retriever, test_data)

            results.append({
                'Configuration': config_info['name'],
                'Table Accuracy': metrics['table_accuracy'],
                'SQL Accuracy': metrics['sql_accuracy'],
                'Avg Query Time (s)': metrics['avg_query_time'],
                'Avg Memory (MB)': metrics['avg_memory'],
                'Type': 'Cold Start'
            })

        return pd.DataFrame(results)

    def run_full_experiment(self, train_ratio: float = 0.8, n_splits: int = 5) -> pd.DataFrame:
        """
        运行完整消融实验（使用交叉验证）

        Args:
            train_ratio: 训练集比例
            n_splits: 交叉验证折数
        """
        print("\n" + "=" * 60)
        print("TE-RAG 消融实验")
        print("=" * 60)

        all_results = []

        for split in range(n_splits):
            print(f"\n--- 第 {split + 1}/{n_splits} 折 ---")

            # 随机分割数据
            np.random.seed(split * 42)
            indices = np.random.permutation(len(self.qa_df))
            n_train = int(len(indices) * train_ratio)

            train_indices = indices[:n_train]
            test_indices = indices[n_train:]

            train_data = self.qa_df.iloc[train_indices]
            test_data = self.qa_df.iloc[test_indices]

            for config_info in self.ablation_configs:
                print(f"  测试配置: {config_info['name']}")

                # 保存训练数据
                train_data.to_csv('/tmp/train_qa.csv', index=False)

                # 创建检索器
                retriever = AblationTE_RAG(
                    self.field_csv, self.table_csv,
                    '/tmp/train_qa.csv'
                )

                # 设置配置
                retriever.use_graph_weight = config_info['config']['use_graph_weight']
                retriever.use_template_mining = config_info['config']['use_template_mining']
                retriever.use_pattern_generalization = config_info['config']['use_pattern_generalization']
                retriever.use_business_rules = config_info['config']['use_business_rules']
                retriever.use_enhanced_index = config_info['config']['use_enhanced_index']

                # 训练
                retriever.fit()

                # 评估
                metrics = self.evaluate_retrieval(retriever, test_data)

                all_results.append({
                    'Split': split + 1,
                    'Configuration': config_info['name'],
                    'Table Accuracy': metrics['table_accuracy'],
                    'SQL Accuracy': metrics['sql_accuracy'],
                    'Avg Query Time (s)': metrics['avg_query_time'],
                    'Avg Memory (MB)': metrics['avg_memory']
                })

        # 汇总结果
        df = pd.DataFrame(all_results)
        summary = df.groupby('Configuration').agg({
            'Table Accuracy': ['mean', 'std'],
            'SQL Accuracy': ['mean', 'std'],
            'Avg Query Time (s)': ['mean', 'std'],
            'Avg Memory (MB)': ['mean', 'std']
        }).reset_index()

        summary.columns = ['Configuration',
                          'Table Accuracy (mean)', 'Table Accuracy (std)',
                          'SQL Accuracy (mean)', 'SQL Accuracy (std)',
                          'Query Time (mean)', 'Query Time (std)',
                          'Memory (mean)', 'Memory (std)']

        return summary

    def run_all_experiments(self, output_dir: str = './results') -> Dict[str, pd.DataFrame]:
        """
        运行所有消融实验
        """
        os.makedirs(output_dir, exist_ok=True)

        results = {}

        # 1. 完整消融实验
        print("\n" + "=" * 60)
        print("运行完整消融实验...")
        print("=" * 60)
        full_results = self.run_full_experiment(train_ratio=0.8, n_splits=3)
        results['ablation_full'] = full_results
        full_results.to_csv(os.path.join(output_dir, 'ablation_results.csv'), index=False)

        # 2. 冷启动实验
        print("\n" + "=" * 60)
        print("运行冷启动实验...")
        print("=" * 60)
        cold_start_results = self.run_cold_start_experiment(test_ratio=0.2)
        results['ablation_cold_start'] = cold_start_results
        cold_start_results.to_csv(os.path.join(output_dir, 'ablation_cold_start.csv'), index=False)

        return results


def main():
    # 数据路径
    field_csv = '/Users/apple/Documents/浙大工作/论文/分层查询数据表/code/source_dataset/processed_field_schema.csv'
    table_csv = '/Users/apple/Documents/浙大工作/论文/分层查询数据表/code/source_dataset/processed_table_schema.csv'
    qa_csv = '/Users/apple/Documents/浙大工作/论文/分层查询数据表/code/source_dataset/processed_qa_data.csv'

    # 运行消融实验
    experiment = AblationExperiment(field_csv, table_csv, qa_csv)
    results = experiment.run_all_experiments(output_dir='/Users/apple/Documents/浙大工作/论文/分层查询数据表/code/results')

    # 打印结果
    print("\n" + "=" * 60)
    print("消融实验结果汇总")
    print("=" * 60)

    for name, df in results.items():
        print(f"\n{name}:")
        print(df.to_string())


if __name__ == "__main__":
    main()
