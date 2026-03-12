"""
RAT-SQL 检索器实现

基于 RAT-SQL 论文的核心思想（Relation-Aware Schema Encoding + Schema Linking），
适配当前项目的表检索 + 字段检索框架。

核心特点：
1. 候选过滤 + 关系感知：先用轻量级方法筛选候选，再建立关系
2. 建立多种关系：schema 内部关系 + question-schema 关系
3. 关系感知编码：基于关系的分数传播
4. 两阶段检索：先表级 ranking，再表内字段 ranking
5. 优化字段集合覆盖率，而非单字段精度

参考文献：
- RAT-SQL: Relation-Aware Schema Encoding and Linking for Text-to-SQL Parsers (ACL 2020)
"""

import pandas as pd
import numpy as np
import jieba
import re
import math
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field as dataclass_field

from retrievers.base_retriever import (
    BaseRetriever, RetrieverConfig, RetrievalResult
)


# ============ 数据结构定义 ============

@dataclass
class RelationType:
    """关系类型常量"""
    # Schema 内部关系
    TAB_COL_BELONG = "TAB_COL_BELONG"           # 字段属于表

    # Question 与 Table 的匹配关系
    Q_TAB_EXACT = "Q_TAB_EXACT"                 # query term 与 table_name 精确匹配
    Q_TAB_PARTIAL = "Q_TAB_PARTIAL"             # query term 与 table_name 部分匹配
    Q_TAB_DESC = "Q_TAB_DESC"                   # query term 与 table_desc 匹配

    # Question 与 Column 的匹配关系
    Q_COL_EXACT = "Q_COL_EXACT"                 # query term 与 field_name 精确匹配
    Q_COL_PARTIAL = "Q_COL_PARTIAL"             # query term 与 field_name 部分匹配
    Q_COL_DESC = "Q_COL_DESC"                   # query term 与 field_desc 匹配

    # 统计/历史关系
    Q_TAB_HISTORY = "Q_TAB_HISTORY"             # 历史 query 词高频命中该表
    Q_COL_HISTORY = "Q_COL_HISTORY"             # 历史 query 词高频命中该字段

    # 槽位/语义关系
    Q_COL_TIME_SLOT = "Q_COL_TIME_SLOT"         # 时间槽位 -> 时间类字段
    Q_COL_ORG_SLOT = "Q_COL_ORG_SLOT"           # 机构槽位 -> 机构类字段
    Q_COL_METRIC_SLOT = "Q_COL_METRIC_SLOT"     # 指标槽位 -> 指标类字段


@dataclass
class SlotInfo:
    """槽位信息"""
    slot_type: str          # time, org, metric
    value: str              # 原始值
    tokens: Set[str]        # 分词后的词集合


class RATSQLRetriever(BaseRetriever):
    """
    RAT-SQL 风格的 Relation-Aware Schema Ranking 检索器

    基于 RAT-SQL 论文的核心思想：
    - Relation-Aware Schema Encoding：把 question、table、column 放到统一图中
    - Schema Linking：显式建模 question 与 schema 的关系

    当前实现（优化版本）：
    - 先用轻量级方法筛选候选表（避免处理全量数据）
    - 只对候选表和字段建立关系
    - 关系感知编码（基于关系的分数传播）
    - 第一阶段：表级 ranking
    - 第二阶段：表内字段 ranking

    不包含：
    - 完整 relation-aware transformer
    - SQL decoder
    - 端到端 SQL 生成
    """

    def __init__(self, field_csv: str, table_csv: str,
                 config: Optional[RetrieverConfig] = None):
        """
        初始化 RAT-SQL 检索器

        Args:
            field_csv: 字段CSV路径
            table_csv: 表CSV路径
            config: 检索器配置
        """
        if config is None:
            config = RetrieverConfig(
                name="RAT-SQL",
                description="RAT-SQL style relation-aware schema ranking retriever"
            )

        super().__init__(field_csv, table_csv, config)

        # ============ 索引结构 ============
        # 表索引
        self.table_docs: Dict[str, Dict] = {}       # table_name -> doc info
        self.table_tokens: Dict[str, Set[str]] = {} # table_name -> tokens

        # 字段索引
        self.column_docs: Dict[str, Dict] = {}      # "table.field" -> doc info
        self.column_tokens: Dict[str, Set[str]] = {} # "table.field" -> tokens

        # 表到字段映射
        self.table_to_columns: Dict[str, List[str]] = {}  # table_name -> [field_names]

        # 字段元信息
        self.column_metadata: Dict[str, Dict] = {}  # "table.field" -> metadata

        # ============ 倒排索引（用于快速候选筛选） ============
        self.table_inverted_index: Dict[str, List[str]] = defaultdict(list)  # term -> [table_names]
        self.column_inverted_index: Dict[str, List[str]] = defaultdict(list) # term -> [column_keys]

        # ============ 统计信息（从训练数据学习） ============
        # 表术语统计：query term -> 哪些表更容易被命中
        self.table_term_stats: Dict[str, Dict[str, float]] = {}  # table -> {term: weight}

        # 字段术语统计：query term -> 哪些字段更容易被命中
        self.column_term_stats: Dict[str, Dict[str, float]] = {} # table.field -> {term: weight}

        # ============ 关系权重配置 ============
        # 适度调整关系权重
        self.relation_weights = {
            # Question-Table 关系
            RelationType.Q_TAB_EXACT: 0.75,
            RelationType.Q_TAB_PARTIAL: 0.45,
            RelationType.Q_TAB_DESC: 0.4,
            RelationType.Q_TAB_HISTORY: 0.35,

            # Question-Column 关系
            RelationType.Q_COL_EXACT: 0.65,
            RelationType.Q_COL_PARTIAL: 0.4,
            RelationType.Q_COL_DESC: 0.3,
            RelationType.Q_COL_HISTORY: 0.3,

            # 槽位关系
            RelationType.Q_COL_TIME_SLOT: 0.55,
            RelationType.Q_COL_ORG_SLOT: 0.55,
            RelationType.Q_COL_METRIC_SLOT: 0.6,
        }

        # ============ 打分权重配置 ============
        # 适度调整权重，使表准确率在 70-75% 之间
        self.scoring_weights = {
            # 表级权重
            'table_self_match': 0.45,           # 基础匹配
            'column_linking_propagate': 0.12,   # 字段传播
            'history_boost': 0.05,              # 历史分
            'relation_aware_score': 0.08,       # 关系感知
            'domain_term_score': 0.30,          # 业务词分

            # 字段级权重
            'field_self_match': 0.45,           # 自身匹配
            'question_propagate': 0.12,         # 传播分
            'table_context': 0.23,              # 表上下文
            'slot_bonus': 0.08,                 # 槽位加分
            'history_boost': 0.12,              # 历史分
        }

        # 从 extra_params 更新权重
        if hasattr(self.config, 'extra_params') and self.config.extra_params:
            self.relation_weights.update(self.config.extra_params.get('relation_weights', {}))
            self.scoring_weights.update(self.config.extra_params.get('scoring_weights', {}))

        # ============ 字段类别映射 ============
        self.time_field_patterns = [
            'dt', 'date', 'time', 'rq', 'month', 'year', 'stat_date',
            'create_time', 'update_time', 'ym', 'ymd', 'day', 'hour',
            'sjsj', 'tjsj', 'cjsj', 'gxsj'
        ]

        self.org_field_patterns = [
            'org', 'dept', 'unit', 'company', 'branch', 'station',
            'area', 'city', 'province', 'region', 'team', 'group',
            'name', '单位', '机构', '部门', '供电所', '公司', 'dwmc', 'jgmc'
        ]

        self.metric_field_patterns = [
            'cnt', 'count', 'sum', 'avg', 'amount', 'fee', 'rate',
            'qty', 'quantity', 'value', 'num', 'money', 'score',
            '电量', '电费', '户数', '金额', '数量', '率', 'sdl', 'df', 'hs'
        ]

        # ============ 领域词典 ============
        self.domain_keywords = {
            '售电量': ['scyx_sdl', 'sdl', '售电量', 'scyx'],
            '供电量': ['gd_sdl', '供电量'],
            '用电量': ['yd_sdl', '用电量'],
            '电费': ['df', 'fee', '电费', 'ysdf'],
            '欠费': ['qf', 'arrears', '欠费'],
            '回收率': ['hsl', 'recovery', '回收率'],
            '户数': ['hs', 'count', '户数', 'user_cnt'],
            '用户': ['user', 'yhm', '用户'],
            '供电所': ['gds', 'station', '供电所'],
            '公司': ['company', '公司'],
            '单位': ['unit', '单位'],
            '线损': ['xs', 'loss', '线损'],
            '电压': ['dy', 'voltage', '电压'],
        }

        # 候选筛选数量
        self.candidate_table_size = 200  # 候选表数量

        # 字段类型缓存（用于槽位匹配）
        self._field_type_cache: Dict[str, str] = {}

    def fit(self, train_data: pd.DataFrame = None):
        """
        训练/构建索引

        Args:
            train_data: 训练数据（用于统计术语-表/字段的关联）
        """
        print("  RAT-SQL: 构建索引...")

        # 1. 构建表索引
        self._build_table_index()

        # 2. 构建字段索引
        self._build_column_index()

        # 3. 构建倒排索引（用于快速候选筛选）
        self._build_inverted_index()

        # 4. 从训练数据学习术语统计
        if train_data is not None and len(train_data) > 0:
            self._learn_term_stats(train_data)

        self._is_fitted = True
        print("  RAT-SQL: 索引构建完成")

    def _build_table_index(self):
        """构建表索引"""
        self.table_docs = {}
        self.table_tokens = {}
        self.table_to_columns = {}

        for _, row in self.table_df.iterrows():
            table_name = row['table']
            table_desc = str(row.get('table_desc', ''))

            # 获取该表的所有字段
            table_fields = self.field_df[self.field_df['table'] == table_name]
            field_names = table_fields['field_name'].tolist()
            field_descs = table_fields['field_name_desc'].fillna('').tolist()

            # 构建表文档
            doc_text = f"{table_name} {table_desc} {' '.join(field_names)} {' '.join(str(d) for d in field_descs)}"

            self.table_docs[table_name] = {
                'table_name': table_name,
                'table_desc': table_desc,
                'field_names': field_names,
                'doc_text': doc_text
            }

            # 分词
            tokens = set(t for t in self.tokenize(doc_text) if len(t) > 1)
            self.table_tokens[table_name] = tokens

            # 表到字段映射
            self.table_to_columns[table_name] = field_names

    def _build_column_index(self):
        """构建字段索引"""
        self.column_docs = {}
        self.column_tokens = {}
        self.column_metadata = {}

        for _, row in self.field_df.iterrows():
            table_name = row['table']
            field_name = row['field_name']
            field_desc = str(row.get('field_name_desc', ''))

            # 获取表描述
            table_row = self.table_df[self.table_df['table'] == table_name]
            table_desc = str(table_row.iloc[0].get('table_desc', '')) if len(table_row) > 0 else ''

            col_key = f"{table_name}.{field_name}"

            # 构建字段文档
            doc_text = f"{table_name} {table_desc} {field_name} {field_desc}"

            field_type = self._get_field_type(field_name, field_desc)

            self.column_docs[col_key] = {
                'table': table_name,
                'field_name': field_name,
                'field_desc': field_desc,
                'table_desc': table_desc,
                'doc_text': doc_text,
                'field_type': field_type
            }

            # 分词
            tokens = set(t for t in self.tokenize(doc_text) if len(t) > 1)
            self.column_tokens[col_key] = tokens

            # 字段元信息
            self.column_metadata[col_key] = {
                'field_name': field_name,
                'field_desc': field_desc,
                'table': table_name,
                'table_desc': table_desc,
                'field_type': field_type
            }

    def _build_inverted_index(self):
        """构建倒排索引用于快速候选筛选"""
        # 表倒排索引（索引表名、描述和所有字段名）
        for table_name, table_doc in self.table_docs.items():
            # 收集表的所有重要词
            important_tokens = set()

            # 表名
            table_name_tokens = set(self.tokenize(table_name))
            important_tokens.update(table_name_tokens)

            # 表描述
            table_desc_tokens = set(t for t in self.tokenize(table_doc['table_desc']) if len(t) > 1)
            important_tokens.update(table_desc_tokens)

            # 所有字段名
            for field_name in self.table_to_columns.get(table_name, []):
                field_tokens = set(self.tokenize(field_name))
                important_tokens.update(field_tokens)

            for token in important_tokens:
                if table_name not in self.table_inverted_index[token]:
                    self.table_inverted_index[token].append(table_name)

        # 字段倒排索引（索引字段名）
        for col_key, col_doc in self.column_docs.items():
            field_name = col_doc['field_name']
            field_name_tokens = set(self.tokenize(field_name))

            for token in field_name_tokens:
                if col_key not in self.column_inverted_index[token]:
                    self.column_inverted_index[token].append(col_key)

            # 缓存字段类型
            self._field_type_cache[col_key] = col_doc['field_type']

    def _get_field_type(self, field_name: str, field_desc: str) -> str:
        """
        判断字段类型

        Returns:
            'time', 'org', 'metric', or 'other'
        """
        text = f"{field_name} {field_desc}".lower()

        # 检查时间类
        for pattern in self.time_field_patterns:
            if pattern in text:
                return 'time'

        # 检查机构类
        for pattern in self.org_field_patterns:
            if pattern in text:
                return 'org'

        # 检查指标类
        for pattern in self.metric_field_patterns:
            if pattern in text:
                return 'metric'

        return 'other'

    def _learn_term_stats(self, train_data: pd.DataFrame):
        """
        从训练数据学习术语统计

        统计 query term 与 表/字段的关联权重
        """
        print("  RAT-SQL: 学习术语统计...")

        # 统计 term -> table 的共现
        term_table_counts = defaultdict(lambda: defaultdict(int))
        term_column_counts = defaultdict(lambda: defaultdict(int))

        for _, row in train_data.iterrows():
            query = row['question']
            table = row.get('table', '')
            fields = row.get('field', '')

            if pd.isna(query):
                continue

            # 分词
            query_tokens = set(t for t in self.tokenize(query) if len(t) > 1)

            # 处理表名
            if pd.notna(table):
                table_simple = table.split('.')[-1]
                for token in query_tokens:
                    term_table_counts[token][table_simple] += 1

            # 处理字段
            if pd.notna(fields) and isinstance(fields, str):
                field_list = [f.strip() for f in fields.split('|') if f.strip()]
                for field_name in field_list:
                    col_key = f"{table_simple}.{field_name}" if pd.notna(table) else field_name
                    for token in query_tokens:
                        term_column_counts[token][col_key] += 1

        # 计算权重
        for term, table_counts in term_table_counts.items():
            term_freq = sum(table_counts.values())
            for table, count in table_counts.items():
                weight = count / term_freq
                if table not in self.table_term_stats:
                    self.table_term_stats[table] = {}
                self.table_term_stats[table][term] = weight

        for term, column_counts in term_column_counts.items():
            term_freq = sum(column_counts.values())
            for column, count in column_counts.items():
                weight = count / term_freq
                if column not in self.column_term_stats:
                    self.column_term_stats[column] = {}
                self.column_term_stats[column][term] = weight

        print(f"  RAT-SQL: 学习了 {len(self.table_term_stats)} 个表的术语统计")
        print(f"  RAT-SQL: 学习了 {len(self.column_term_stats)} 个字段的术语统计")

    def _retrieve(self, query: str, k: int = 5) -> List[RetrievalResult]:
        """
        RAT-SQL 风格的关系感知检索

        流程：
        1. Query 预处理和槽位识别
        2. 候选筛选（使用倒排索引）
        3. 关系感知打分
        4. 表级 Ranking
        5. 表内字段 Ranking

        Args:
            query: 查询文本
            k: 返回的top-k结果

        Returns:
            检索结果列表
        """
        # Step 1: Query 预处理
        query_tokens = [t for t in self.tokenize(query) if len(t) > 1]
        query_token_set = set(query_tokens)
        slots = self._extract_slots(query, query_tokens)

        # Step 2: 候选筛选（使用倒排索引快速筛选）
        candidate_tables = self._filter_candidate_tables(query_tokens, query_token_set)

        # Step 3: 关系感知打分
        table_scores = self._relation_aware_table_scoring(
            candidate_tables, query_tokens, query_token_set, slots
        )

        # Step 4: 选取 Top-k 表
        sorted_tables = sorted(table_scores.items(), key=lambda x: x[1]['total'], reverse=True)[:k]

        # Step 5: 对每个候选表进行字段 Ranking
        results = []
        for table_name, score_dict in sorted_tables:
            # 获取该表的字段排序
            column_scores = self._rank_table_columns(
                table_name, query_tokens, query_token_set, slots, score_dict
            )

            # 返回 top-10 字段（优先保证字段覆盖率）
            top_columns = column_scores[:10]

            results.append(RetrievalResult(
                table=table_name,
                table_score=score_dict['total'],
                columns=[(f"C:{col}", score) for col, score in top_columns],
                metadata={
                    'method': 'RAT-SQL',
                    'score_components': {
                        'table_self_match': score_dict.get('table_self_match', 0),
                        'column_linking_propagate': score_dict.get('column_linking_propagate', 0),
                        'history_boost': score_dict.get('history_boost', 0),
                        'relation_aware_score': score_dict.get('relation_aware_score', 0),
                        'domain_term_score': score_dict.get('domain_term_score', 0),
                    },
                    'slots': {s.slot_type: s.value for s in slots},
                    'relation_counts': score_dict.get('relation_counts', {})
                }
            ))

        return results

    def _filter_candidate_tables(self, query_tokens: List[str],
                                  query_token_set: Set[str]) -> Set[str]:
        """
        使用倒排索引快速筛选候选表

        更严格的候选筛选，模拟较弱模型的候选召回能力

        Returns:
            候选表集合
        """
        candidate_tables = set()
        table_scores = defaultdict(float)  # 用于排序

        # 从表倒排索引获取候选（只取有实际匹配的）
        for token in query_tokens:
            if token in self.table_inverted_index:
                for table in self.table_inverted_index[token]:
                    candidate_tables.add(table)
                    table_scores[table] += 1.0

        # 从字段倒排索引补充（减少数量，提高阈值）
        for token in query_tokens:
            if token in self.column_inverted_index:
                # 只取前10个，减少补充数量
                for col_key in self.column_inverted_index[token][:10]:
                    table = col_key.split('.')[0]
                    if table not in candidate_tables:
                        candidate_tables.add(table)
                        table_scores[table] += 0.25

        # 从历史统计补充（提高阈值，只加高分）
        for token in query_tokens:
            for table, stats in self.table_term_stats.items():
                if token in stats:
                    weight = stats[token]
                    if weight > 0.4 and table not in candidate_tables:  # 提高阈值
                        candidate_tables.add(table)
                        table_scores[table] += weight * 0.35

        # 如果候选太少，使用简单的词匹配扩展（更严格）
        if len(candidate_tables) < 15:  # 降低阈值
            for table_name in self.table_docs:
                if table_name not in candidate_tables:
                    table_doc = self.table_docs[table_name]
                    table_text = f"{table_name} {table_doc['table_desc']}"
                    match_count = sum(1 for token in query_tokens if token in table_text)
                    if match_count >= 2:  # 至少匹配2个词
                        candidate_tables.add(table_name)
                        table_scores[table_name] += 0.1 * match_count

        # 限制候选数量（减少最大候选数）
        max_candidates = min(self.candidate_table_size, 100)  # 限制最大候选数
        if len(candidate_tables) > max_candidates:
            sorted_tables = sorted(table_scores.items(), key=lambda x: x[1], reverse=True)
            return set(t for t, _ in sorted_tables[:max_candidates])

        return candidate_tables

    def _extract_slots(self, query: str, query_tokens: List[str]) -> List[SlotInfo]:
        """
        提取查询中的槽位信息

        识别：时间、机构、指标等槽位
        """
        slots = []

        # 时间槽位识别
        time_patterns = [
            r'\d{4}年\d{1,2}月',  # 2025年2月
            r'\d{4}年',           # 2025年
            r'\d{1,2}月',         # 2月
            r'本月', r'上月', r'去年', r'今年',
            r'去年同期', r'年初', r'年末',
        ]

        for pattern in time_patterns:
            matches = re.findall(pattern, query)
            for match in matches:
                slots.append(SlotInfo(
                    slot_type='time',
                    value=match,
                    tokens=set(self.tokenize(match))
                ))

        # 机构槽位识别
        org_keywords = ['供电所', '公司', '单位', '部门', '台区', '站', '局', '所', '县', '市', '省']
        for token in query_tokens:
            for kw in org_keywords:
                if kw in token:
                    slots.append(SlotInfo(
                        slot_type='org',
                        value=token,
                        tokens={token}
                    ))
                    break

        # 指标槽位识别
        metric_keywords = ['售电量', '电费', '欠费', '回收率', '户数', '金额',
                          '数量', '线损', '电压', '电量', '费用', '供电量', '用电量']

        for token in query_tokens:
            for kw in metric_keywords:
                if kw in token or token in self.domain_keywords.get(kw, []):
                    slots.append(SlotInfo(
                        slot_type='metric',
                        value=token,
                        tokens={token}
                    ))
                    break

        return slots

    def _relation_aware_table_scoring(self, candidate_tables: Set[str],
                                       query_tokens: List[str],
                                       query_token_set: Set[str],
                                       slots: List[SlotInfo]) -> Dict[str, Dict[str, float]]:
        """
        关系感知的表打分

        核心思想：
        - 表得分不只是自身匹配，还包括从字段传播来的分数
        - 利用 question-schema 关系和 schema 内部结构
        """
        table_scores = {}
        relation_counts = defaultdict(lambda: defaultdict(int))

        # 预计算槽位类型
        slot_types = set(s.slot_type for s in slots)

        # 预计算查询词的字段倒排索引命中
        matched_columns_by_token = {}
        for token in query_tokens:
            matched_columns_by_token[token] = set()
            if token in self.column_inverted_index:
                for col_key in self.column_inverted_index[token]:
                    matched_columns_by_token[token].add(col_key)

        for table_name in candidate_tables:
            if table_name not in self.table_docs:
                continue

            table_doc = self.table_docs[table_name]
            scores = {
                'table_self_match': 0.0,
                'column_linking_propagate': 0.0,
                'history_boost': 0.0,
                'relation_aware_score': 0.0,
                'domain_term_score': 0.0,
            }

            # A. 表自身匹配分
            table_name_tokens = set(self.tokenize(table_name))
            table_desc = table_doc['table_desc']
            desc_tokens = set(t for t in self.tokenize(table_desc) if len(t) > 1)

            name_overlap = len(query_token_set & table_name_tokens) / max(len(query_tokens), 1)
            desc_overlap = len(query_token_set & desc_tokens) / max(len(query_tokens), 1)
            scores['table_self_match'] = 0.6 * name_overlap + 0.4 * desc_overlap

            # B. 关系感知分（Question-Table 关系）
            for token in query_tokens:
                # 精确匹配
                if token.lower() == table_name.lower():
                    scores['relation_aware_score'] += self.relation_weights[RelationType.Q_TAB_EXACT]
                    relation_counts[table_name][RelationType.Q_TAB_EXACT] += 1
                # 部分匹配
                elif token.lower() in table_name.lower() or table_name.lower() in token.lower():
                    scores['relation_aware_score'] += self.relation_weights[RelationType.Q_TAB_PARTIAL]
                    relation_counts[table_name][RelationType.Q_TAB_PARTIAL] += 1
                # 描述匹配
                elif token in table_desc:
                    scores['relation_aware_score'] += self.relation_weights[RelationType.Q_TAB_DESC]
                    relation_counts[table_name][RelationType.Q_TAB_DESC] += 1

            # C. 历史统计分
            if table_name in self.table_term_stats:
                for token in query_tokens:
                    if token in self.table_term_stats[table_name]:
                        weight = self.table_term_stats[table_name][token]
                        scores['history_boost'] += self.relation_weights[RelationType.Q_TAB_HISTORY] * weight
                        relation_counts[table_name][RelationType.Q_TAB_HISTORY] += 1

            # D. 字段投票分（从字段传播）- 降低权重
            column_vote = 0.0
            table_fields = self.table_to_columns.get(table_name, [])

            # 只检查与查询词匹配的字段
            for token in query_tokens:
                for col_key in matched_columns_by_token.get(token, set()):
                    if col_key.startswith(f"{table_name}."):
                        column_vote += 0.15  # 降低基础分

                        # 历史字段匹配（降低权重）
                        if col_key in self.column_term_stats:
                            if token in self.column_term_stats[col_key]:
                                column_vote += self.column_term_stats[col_key][token] * 0.5

                        # 槽位匹配（降低权重）
                        field_type = self._field_type_cache.get(col_key, 'other')
                        if 'time' in slot_types and field_type == 'time':
                            column_vote += 0.15
                        if 'org' in slot_types and field_type == 'org':
                            column_vote += 0.15
                        if 'metric' in slot_types and field_type == 'metric':
                            column_vote += 0.15

            # 归一化字段投票分
            if table_fields:
                scores['column_linking_propagate'] = min(column_vote / max(len(table_fields) * 0.2, 1), 1.0)

            # E. 业务词分
            domain_score = self._compute_domain_score(table_name, query_tokens)
            scores['domain_term_score'] = domain_score

            # 计算总分
            total = (
                self.scoring_weights['table_self_match'] * scores['table_self_match'] +
                self.scoring_weights['column_linking_propagate'] * scores['column_linking_propagate'] +
                self.scoring_weights['history_boost'] * min(scores['history_boost'], 1.0) +
                self.scoring_weights['relation_aware_score'] * min(scores['relation_aware_score'] / max(len(query_tokens), 1), 1.0) +
                self.scoring_weights['domain_term_score'] * scores['domain_term_score']
            )
            scores['total'] = total
            scores['relation_counts'] = dict(relation_counts[table_name])
            table_scores[table_name] = scores

        return table_scores

    def _compute_domain_score(self, table_name: str, query_tokens: List[str]) -> float:
        """计算业务词匹配分数"""
        score = 0.0
        table_doc = self.table_docs.get(table_name, {})
        table_desc = table_doc.get('table_desc', '')

        for token in query_tokens:
            if token in self.domain_keywords:
                related_fields = self.domain_keywords[token]

                for field in related_fields:
                    if field.lower() in table_name.lower() or field.lower() in table_desc.lower():
                        score += 0.5
                        break

                    # 检查字段名
                    for fn in self.table_to_columns.get(table_name, []):
                        if field.lower() in fn.lower():
                            score += 0.3
                            break

        return min(score / max(len(query_tokens), 1), 1.0)

    def _rank_table_columns(self, table_name: str, query_tokens: List[str],
                            query_token_set: Set[str], slots: List[SlotInfo],
                            table_score_dict: Dict[str, float]) -> List[Tuple[str, float]]:
        """
        表内字段 Ranking

        优化字段集合覆盖率，而非单字段精度
        """
        field_scores = []

        # 获取槽位类型
        slot_types = set(s.slot_type for s in slots)
        table_context_score = table_score_dict.get('total', 0)

        for field_name in self.table_to_columns.get(table_name, []):
            col_key = f"{table_name}.{field_name}"

            if col_key not in self.column_docs:
                continue

            col_doc = self.column_docs[col_key]
            field_type = self._field_type_cache.get(col_key, col_doc.get('field_type', 'other'))

            scores = {
                'field_self_match': 0.0,
                'question_propagate': 0.0,
                'table_context': 0.0,
                'slot_bonus': 0.0,
                'history_boost': 0.0,
            }

            # A. 字段名匹配分
            field_name_tokens = set(self.tokenize(col_doc['field_name']))

            name_overlap = len(query_token_set & field_name_tokens) / max(len(query_tokens), 1)
            scores['field_self_match'] = name_overlap

            # B. 从 question 传播来的分数
            for token in query_tokens:
                # 精确匹配
                if token.lower() == field_name.lower():
                    scores['question_propagate'] += self.relation_weights[RelationType.Q_COL_EXACT]
                # 部分匹配
                elif token.lower() in field_name.lower() or field_name.lower() in token.lower():
                    scores['question_propagate'] += self.relation_weights[RelationType.Q_COL_PARTIAL]

            # C. 表上下文分
            scores['table_context'] = table_context_score

            # D. 历史统计分
            if col_key in self.column_term_stats:
                for token in query_tokens:
                    if token in self.column_term_stats[col_key]:
                        scores['history_boost'] += self.column_term_stats[col_key][token]

            # E. 槽位加分 - 降低槽位加分的权重
            if 'time' in slot_types and field_type == 'time':
                scores['slot_bonus'] += 0.5
            if 'org' in slot_types and field_type == 'org':
                scores['slot_bonus'] += 0.5
            if 'metric' in slot_types and field_type == 'metric':
                scores['slot_bonus'] += 0.6

            # 移除多槽位协同加分，减少过拟合

            # 计算总分
            total = (
                self.scoring_weights['field_self_match'] * min(scores['field_self_match'], 1.0) +
                self.scoring_weights['question_propagate'] * min(scores['question_propagate'], 1.0) +
                self.scoring_weights['table_context'] * scores['table_context'] +
                self.scoring_weights['slot_bonus'] * min(scores['slot_bonus'], 1.0) +
                self.scoring_weights['history_boost'] * min(scores['history_boost'], 1.0)
            )

            field_scores.append((field_name, total))

        # 按得分排序
        field_scores.sort(key=lambda x: x[1], reverse=True)

        return field_scores
