"""
RESDSQL 检索器实现

基于 RESDSQL 论文的核心思想（Schema Ranking + Schema Linking），
适配当前项目的表检索 + 字段检索框架。

核心特点：
1. 两阶段检索：先表级 ranking，再表内字段 ranking
2. 优化字段集合覆盖率，而非单字段精度
3. 规则 + 统计方法，可解释、可调试
"""

import pandas as pd
import numpy as np
import jieba
import re
import math
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass

from retrievers.base_retriever import (
    BaseRetriever, RetrieverConfig, RetrievalResult
)


@dataclass
class SlotInfo:
    """槽位信息"""
    slot_type: str  # time, org, metric
    value: str      # 原始值
    tokens: Set[str]  # 分词后的词集合


class RESDSQLRetriever(BaseRetriever):
    """
    RESDSQL 风格的 Schema Ranking 检索器

    基于 RESDSQL 论文的核心思想：
    - Ranking-Enhanced Encoding：对 schema item 做排序、过滤
    - Schema Linking：解决"问题对应哪些表、哪些字段"

    当前实现：
    - 第一阶段：表级 ranking（多分量加权）
    - 第二阶段：表内字段 ranking（优化字段集合覆盖率）

    不包含：
    - 完整 SQL 生成
    - SQL skeleton decoder
    - Beam search
    """

    def __init__(self, field_csv: str, table_csv: str,
                 config: Optional[RetrieverConfig] = None):
        """
        初始化 RESDSQL 检索器

        Args:
            field_csv: 字段CSV路径
            table_csv: 表CSV路径
            config: 检索器配置
        """
        if config is None:
            config = RetrieverConfig(
                name="RESDSQL",
                description="RESDSQL-style schema ranking retriever"
            )

        super().__init__(field_csv, table_csv, config)

        # ============ 索引结构 ============
        # 表索引
        self.table_docs: Dict[str, Dict] = {}  # table_name -> doc info
        self.table_tokens: Dict[str, Set[str]] = {}  # table_name -> tokens

        # 字段索引
        self.column_docs: Dict[str, Dict] = {}  # "table.field" -> doc info
        self.column_tokens: Dict[str, Set[str]] = {}  # "table.field" -> tokens

        # 表到字段映射
        self.table_to_columns: Dict[str, List[str]] = {}  # table_name -> [field_names]

        # ============ 统计信息（从训练数据学习） ============
        # 表术语统计：query term -> 哪些表更容易被命中
        self.table_term_stats: Dict[str, Dict[str, float]] = {}  # table -> {term: weight}

        # 字段术语统计：query term -> 哪些字段更容易被命中
        self.column_term_stats: Dict[str, Dict[str, float]] = {}  # table.field -> {term: weight}

        # ============ 字段类别映射 ============
        # 时间类字段
        self.time_field_patterns = [
            'dt', 'date', 'time', 'rq', 'month', 'year', 'stat_date',
            'create_time', 'update_time', 'ym', 'ymd', 'day', 'hour'
        ]

        # 机构/实体类字段
        self.org_field_patterns = [
            'org', 'dept', 'unit', 'company', 'branch', 'station',
            'area', 'city', 'province', 'region', 'team', 'group',
            'name', '单位', '机构', '部门', '供电所', '公司'
        ]

        # 指标/数值类字段
        self.metric_field_patterns = [
            'cnt', 'count', 'sum', 'avg', 'amount', 'fee', 'rate',
            'qty', 'quantity', 'value', 'num', 'money', 'score',
            '电量', '电费', '户数', '金额', '数量', '率'
        ]

        # ============ 领域词典 ============
        # 业务关键词 -> 相关表的映射
        self.domain_keywords = {
            # 售电量相关
            '售电量': ['scyx_sdl', 'scyx', '售电量'],
            '供电量': ['gd_sdl', '供电量'],
            '用电量': ['yd_sdl', '用电量'],

            # 电费相关
            '电费': ['df', 'fee', '电费', 'ysdf'],
            '欠费': ['qf', 'arrears', '欠费'],
            '回收率': ['hsl', 'recovery', '回收率'],

            # 户数相关
            '户数': ['hs', 'count', '户数', 'user_cnt'],
            '用户': ['user', 'yhm', '用户'],

            # 机构相关
            '供电所': ['gds', 'station', '供电所'],
            '公司': ['company', '公司'],
            '单位': ['unit', '单位'],
        }

        # ============ 权重配置 ============
        self.weights = {
            # 表级权重
            'table_name_match': 0.25,
            'table_desc_match': 0.15,
            'column_vote_score': 0.25,
            'template_match_score': 0.20,
            'domain_term_score': 0.15,

            # 字段级权重
            'field_name_match': 0.30,
            'field_desc_match': 0.15,
            'field_template_match': 0.20,
            'table_context_score': 0.15,
            'multi_slot_bonus': 0.20,
        }

        # 从 extra_params 更新权重
        if hasattr(self.config, 'extra_params') and self.config.extra_params:
            self.weights.update(self.config.extra_params.get('weights', {}))

    def fit(self, train_data: pd.DataFrame = None):
        """
        训练/构建索引

        Args:
            train_data: 训练数据（用于统计术语-表/字段的关联）
        """
        print("  RESDSQL: 构建索引...")

        # 1. 构建表索引
        self._build_table_index()

        # 2. 构建字段索引
        self._build_column_index()

        # 3. 从训练数据学习术语统计
        if train_data is not None and len(train_data) > 0:
            self._learn_term_stats(train_data)

        self._is_fitted = True
        print("  RESDSQL: 索引构建完成")

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

            self.column_docs[col_key] = {
                'table': table_name,
                'field_name': field_name,
                'field_desc': field_desc,
                'table_desc': table_desc,
                'doc_text': doc_text,
                'field_type': self._get_field_type(field_name, field_desc)
            }

            # 分词
            tokens = set(t for t in self.tokenize(doc_text) if len(t) > 1)
            self.column_tokens[col_key] = tokens

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
        print("  RESDSQL: 学习术语统计...")

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
                for field in field_list:
                    col_key = f"{table_simple}.{field}" if pd.notna(table) else field
                    for token in query_tokens:
                        term_column_counts[token][col_key] += 1

        # 计算权重（使用 Pointwise Mutual Information 思想）
        total_queries = len(train_data)

        for term, table_counts in term_table_counts.items():
            term_freq = sum(table_counts.values())
            for table, count in table_counts.items():
                # 简化的 PMI 权重
                weight = count / term_freq  # 条件概率 P(table | term)
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

        print(f"  RESDSQL: 学习了 {len(self.table_term_stats)} 个表的术语统计")
        print(f"  RESDSQL: 学习了 {len(self.column_term_stats)} 个字段的术语统计")

    def _retrieve(self, query: str, k: int = 5) -> List[RetrievalResult]:
        """
        RESDSQL 风格的两阶段检索

        Args:
            query: 查询文本
            k: 返回的top-k结果

        Returns:
            检索结果列表
        """
        # Step 1: Query 预处理和槽位识别
        query_tokens = [t for t in self.tokenize(query) if len(t) > 1]
        query_token_set = set(query_tokens)
        slots = self._extract_slots(query, query_tokens)

        # Step 2: 表级 Ranking
        table_scores = self._rank_tables(query, query_tokens, query_token_set, slots)

        # Step 3: 选取 Top-k 表
        sorted_tables = sorted(table_scores.items(), key=lambda x: x[1]['total'], reverse=True)[:k]

        # Step 4: 对每个候选表进行字段 Ranking
        results = []
        for table_name, score_dict in sorted_tables:
            # 获取该表的字段排序
            column_scores = self._rank_columns(
                table_name, query, query_tokens, query_token_set, slots, score_dict
            )

            # 返回 top-10 字段（优先保证字段覆盖率）
            top_columns = column_scores[:10]

            results.append(RetrievalResult(
                table=table_name,
                table_score=score_dict['total'],
                columns=[(f"C:{col}", score) for col, score in top_columns],
                metadata={
                    'method': 'RESDSQL',
                    'score_components': {
                        'table_name_match': score_dict.get('table_name_match', 0),
                        'table_desc_match': score_dict.get('table_desc_match', 0),
                        'column_vote_score': score_dict.get('column_vote_score', 0),
                        'template_match_score': score_dict.get('template_match_score', 0),
                        'domain_term_score': score_dict.get('domain_term_score', 0),
                    },
                    'slots': {s.slot_type: s.value for s in slots}
                }
            ))

        return results

    def _extract_slots(self, query: str, query_tokens: List[str]) -> List[SlotInfo]:
        """
        提取查询中的槽位信息

        识别：时间、机构、指标等槽位
        """
        slots = []
        query_token_set = set(query_tokens)

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

        # 机构槽位识别（使用领域词典和规则）
        org_keywords = ['供电所', '公司', '单位', '部门', '台区', '站', '局', '所']
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
                          '数量', '线损', '电压', '电量', '费用']

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

    def _rank_tables(self, query: str, query_tokens: List[str],
                     query_token_set: Set[str], slots: List[SlotInfo]) -> Dict[str, Dict[str, float]]:
        """
        表级 Ranking

        计算每个表的综合得分，包含多个分量：
        - 表名匹配分
        - 表描述匹配分
        - 字段投票分
        - 历史模板/术语分
        - 业务词分
        """
        table_scores = {}

        for table_name, table_doc in self.table_docs.items():
            scores = {}

            # A. 表名匹配分
            table_name_tokens = set(self.tokenize(table_name))
            name_overlap = len(query_token_set & table_name_tokens)
            scores['table_name_match'] = name_overlap / max(len(query_tokens), 1)

            # B. 表描述匹配分
            table_desc = table_doc['table_desc']
            desc_tokens = set(t for t in self.tokenize(table_desc) if len(t) > 1)
            desc_overlap = len(query_token_set & desc_tokens)
            scores['table_desc_match'] = desc_overlap / max(len(query_tokens), 1)

            # C. 字段投票分（多个字段匹配则加分）
            field_vote_score = 0
            matched_fields = 0
            for field_name in self.table_to_columns.get(table_name, []):
                col_key = f"{table_name}.{field_name}"
                if col_key in self.column_tokens:
                    col_tokens = self.column_tokens[col_key]
                    overlap = len(query_token_set & col_tokens)
                    if overlap > 0:
                        matched_fields += 1
                        field_vote_score += overlap

            # 归一化
            if len(self.table_to_columns.get(table_name, [])) > 0:
                scores['column_vote_score'] = matched_fields / len(self.table_to_columns[table_name])
            else:
                scores['column_vote_score'] = 0

            # D. 历史模板/术语分
            template_score = 0
            if table_name in self.table_term_stats:
                for token in query_tokens:
                    if token in self.table_term_stats[table_name]:
                        template_score += self.table_term_stats[table_name][token]

            # 归一化
            scores['template_match_score'] = min(template_score / max(len(query_tokens), 1), 1.0)

            # E. 业务词分
            domain_score = 0
            for token in query_tokens:
                if token in self.domain_keywords:
                    # 检查该业务词是否与当前表相关
                    related_fields = self.domain_keywords[token]
                    for field in related_fields:
                        if field.lower() in table_name.lower() or field.lower() in table_desc.lower():
                            domain_score += 1
                            break
                        # 检查字段名
                        for fn in self.table_to_columns.get(table_name, []):
                            if field.lower() in fn.lower():
                                domain_score += 0.5
                                break

            scores['domain_term_score'] = min(domain_score / max(len(query_tokens), 1), 1.0)

            # 计算总分（加权和）
            total = (
                self.weights['table_name_match'] * scores['table_name_match'] +
                self.weights['table_desc_match'] * scores['table_desc_match'] +
                self.weights['column_vote_score'] * scores['column_vote_score'] +
                self.weights['template_match_score'] * scores['template_match_score'] +
                self.weights['domain_term_score'] * scores['domain_term_score']
            )

            scores['total'] = total
            table_scores[table_name] = scores

        return table_scores

    def _rank_columns(self, table_name: str, query: str, query_tokens: List[str],
                      query_token_set: Set[str], slots: List[SlotInfo],
                      table_score_dict: Dict[str, float]) -> List[Tuple[str, float]]:
        """
        表内字段 Ranking

        优化字段集合覆盖率，而非单字段精度
        """
        column_scores = []

        # 获取槽位类型
        slot_types = set(s.slot_type for s in slots)

        # 计算表上下文分数（如果表得分高，其字段也有先验加分）
        table_context_base = table_score_dict.get('total', 0)

        for field_name in self.table_to_columns.get(table_name, []):
            col_key = f"{table_name}.{field_name}"

            if col_key not in self.column_docs:
                continue

            col_doc = self.column_docs[col_key]
            scores = {}

            # A. 字段名匹配分
            field_name_tokens = set(self.tokenize(col_doc['field_name']))
            name_overlap = len(query_token_set & field_name_tokens)
            scores['field_name_match'] = name_overlap / max(len(query_tokens), 1)

            # B. 字段描述匹配分
            field_desc = col_doc['field_desc']
            desc_tokens = set(t for t in self.tokenize(field_desc) if len(t) > 1)
            desc_overlap = len(query_token_set & desc_tokens)
            scores['field_desc_match'] = desc_overlap / max(len(query_tokens), 1)

            # C. 历史术语/模板分
            template_score = 0
            if col_key in self.column_term_stats:
                for token in query_tokens:
                    if token in self.column_term_stats[col_key]:
                        template_score += self.column_term_stats[col_key][token]

            scores['field_template_match'] = min(template_score / max(len(query_tokens), 1), 1.0)

            # D. 表上下文一致性分
            scores['table_context_score'] = table_context_base

            # E. 多槽位协同加分（关键：提升字段覆盖率）
            multi_slot_bonus = 0
            field_type = col_doc['field_type']

            # 如果 query 包含时间槽位，时间类字段加分
            if 'time' in slot_types and field_type == 'time':
                multi_slot_bonus += 1.0

            # 如果 query 包含机构槽位，机构类字段加分
            if 'org' in slot_types and field_type == 'org':
                multi_slot_bonus += 1.0

            # 如果 query 包含指标槽位，指标类字段加分
            if 'metric' in slot_types and field_type == 'metric':
                multi_slot_bonus += 1.0

            # 如果 query 包含多个槽位，且该字段类型匹配任一槽位，额外加分
            if len(slot_types) >= 2 and field_type in slot_types:
                multi_slot_bonus += 0.5

            scores['multi_slot_bonus'] = min(multi_slot_bonus, 1.0)

            # 计算总分（加权和）
            total = (
                self.weights['field_name_match'] * scores['field_name_match'] +
                self.weights['field_desc_match'] * scores['field_desc_match'] +
                self.weights['field_template_match'] * scores['field_template_match'] +
                self.weights['table_context_score'] * scores['table_context_score'] +
                self.weights['multi_slot_bonus'] * scores['multi_slot_bonus']
            )

            column_scores.append((field_name, total))

        # 按得分排序
        column_scores.sort(key=lambda x: x[1], reverse=True)

        return column_scores
