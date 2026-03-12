"""
Seq2SQL 检索器实现

基于 Seq2SQL 论文的核心思想（结构化 SQL 组件预测），
适配当前项目的表检索 + 字段检索框架。

核心特点：
1. 把 SQL 生成问题转换为结构化字段预测问题
2. 字段角色分类：SELECT字段、WHERE字段、AGG字段
3. 分别预测三种字段，然后合并返回
4. 优化字段集合覆盖率，而非单字段精度

参考文献：
- Seq2SQL: Generating Structured Queries from Natural Language using Reinforcement Learning (NIPS 2017)

注意：当前实现不生成 SQL 字符串，只做字段集合预测
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
class SlotInfo:
    """槽位信息"""
    slot_type: str          # time, org, metric
    value: str              # 原始值
    tokens: Set[str]        # 分词后的词集合


@dataclass
class FieldRole:
    """字段角色统计"""
    select_count: int = 0   # 作为 SELECT 字段的次数
    where_count: int = 0    # 作为 WHERE 字段的次数
    agg_count: int = 0      # 作为 AGG 字段的次数

    @property
    def total(self) -> int:
        return self.select_count + self.where_count + self.agg_count

    @property
    def select_prior(self) -> float:
        return self.select_count / max(self.total, 1)

    @property
    def where_prior(self) -> float:
        return self.where_count / max(self.total, 1)

    @property
    def agg_prior(self) -> float:
        return self.agg_count / max(self.total, 1)


class Seq2SQLRetriever(BaseRetriever):
    """
    Seq2SQL 风格的结构化字段预测检索器

    基于 Seq2SQL 论文的核心思想：
    - 把 SQL 拆分为结构化组件（SELECT/WHERE/AGG）
    - 对每个组件分别预测字段
    - 合并字段集合返回

    当前实现：
    - 第一阶段：表级 ranking（多分量加权）
    - 第二阶段：表内结构化字段预测
        - SELECT 字段预测
        - WHERE 字段预测
        - AGG 字段预测
    - 第三阶段：合并字段集合

    不包含：
    - 完整 SQL 生成
    - Reinforcement Learning
    - SQL decoder
    """

    def __init__(self, field_csv: str, table_csv: str,
                 config: Optional[RetrieverConfig] = None):
        """
        初始化 Seq2SQL 检索器

        Args:
            field_csv: 字段CSV路径
            table_csv: 表CSV路径
            config: 检索器配置
        """
        if config is None:
            config = RetrieverConfig(
                name="Seq2SQL",
                description="Seq2SQL-style structured field prediction retriever"
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

        # ============ 字段角色统计（从训练数据学习） ============
        self.field_roles: Dict[str, FieldRole] = {}  # "table.field" -> FieldRole

        # ============ 统计信息（从训练数据学习） ============
        # 表术语统计：query term -> 哪些表更容易被命中
        self.table_term_stats: Dict[str, Dict[str, float]] = {}  # table -> {term: weight}

        # 字段术语统计：query term -> 哪些字段更容易被命中
        self.column_term_stats: Dict[str, Dict[str, float]] = {} # table.field -> {term: weight}

        # ============ 字段类别映射 ============
        # 时间类字段（通常用于 WHERE 条件）
        self.time_field_patterns = [
            'dt', 'date', 'time', 'rq', 'month', 'year', 'stat_date',
            'create_time', 'update_time', 'ym', 'ymd', 'day', 'hour',
            'sjsj', 'tjsj', 'cjsj', 'gxsj', 'tjrq', 'cjsj'
        ]

        # 机构/实体类字段（通常用于 WHERE 条件）
        self.org_field_patterns = [
            'org', 'dept', 'unit', 'company', 'branch', 'station',
            'area', 'city', 'province', 'region', 'team', 'group',
            'name', '单位', '机构', '部门', '供电所', '公司', 'dwmc', 'jgmc',
            'gds', 'gs', 'dw', 'jg', 'bm'
        ]

        # 指标/数值类字段（通常用于 SELECT 和 AGG）
        self.metric_field_patterns = [
            'cnt', 'count', 'sum', 'avg', 'amount', 'fee', 'rate',
            'qty', 'quantity', 'value', 'num', 'money', 'score',
            '电量', '电费', '户数', '金额', '数量', '率', 'sdl', 'df', 'hs',
            'total', 'je', 'sl', 'je'
        ]

        # 聚合函数关键词（暗示需要 AGG 字段）
        self.agg_keywords = [
            '合计', '总计', '平均', '最大', '最小', '求和', '统计',
            '总数', '总量', '占比', '比例', '合计', '累计',
            'sum', 'avg', 'max', 'min', 'count', 'total'
        ]

        # ============ 领域词典 ============
        self.domain_keywords = {
            # 售电量相关
            '售电量': ['scyx_sdl', 'sdl', '售电量', 'scyx'],
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
            '线损': ['xs', 'loss', '线损'],
            '电压': ['dy', 'voltage', '电压'],
        }

        # ============ 权重配置 ============
        self.weights = {
            # 表级权重
            'table_name_match': 0.20,
            'table_desc_match': 0.15,
            'column_vote_score': 0.25,
            'template_match_score': 0.20,
            'domain_term_score': 0.20,

            # SELECT 字段权重
            'select_name_match': 0.25,
            'select_desc_match': 0.15,
            'select_role_prior': 0.25,
            'select_term_history': 0.20,
            'select_metric_bonus': 0.15,

            # WHERE 字段权重
            'where_slot_match': 0.30,
            'where_desc_match': 0.15,
            'where_role_prior': 0.25,
            'where_term_history': 0.20,
            'where_type_bonus': 0.10,

            # AGG 字段权重
            'agg_numeric_bonus': 0.30,
            'agg_role_prior': 0.25,
            'agg_metric_match': 0.25,
            'agg_select_overlap': 0.20,
        }

        # 从 extra_params 更新权重
        if hasattr(self.config, 'extra_params') and self.config.extra_params:
            self.weights.update(self.config.extra_params.get('weights', {}))

        # 字段类型缓存
        self._field_type_cache: Dict[str, str] = {}

    def fit(self, train_data: pd.DataFrame = None):
        """
        训练/构建索引

        Args:
            train_data: 训练数据（用于统计术语-表/字段的关联和字段角色）
        """
        print("  Seq2SQL: 构建索引...")

        # 1. 构建表索引
        self._build_table_index()

        # 2. 构建字段索引
        self._build_column_index()

        # 3. 从训练数据学习术语统计和字段角色
        if train_data is not None and len(train_data) > 0:
            self._learn_term_stats(train_data)
            self._learn_field_roles(train_data)

        self._is_fitted = True
        print("  Seq2SQL: 索引构建完成")

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

            # 判断字段类型
            field_type = self._get_field_type(field_name, field_desc)

            # 构建字段文档
            doc_text = f"{table_name} {table_desc} {field_name} {field_desc}"

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

            # 缓存字段类型
            self._field_type_cache[col_key] = field_type

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
        print("  Seq2SQL: 学习术语统计...")

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

        print(f"  Seq2SQL: 学习了 {len(self.table_term_stats)} 个表的术语统计")
        print(f"  Seq2SQL: 学习了 {len(self.column_term_stats)} 个字段的术语统计")

    def _learn_field_roles(self, train_data: pd.DataFrame):
        """
        从训练数据学习字段角色

        统计每个字段在 SQL 中的角色分布：
        - SELECT 字段：通常是查询的目标指标
        - WHERE 字段：通常是过滤条件（时间、机构等）
        - AGG 字段：通常是聚合函数作用的字段
        """
        print("  Seq2SQL: 学习字段角色...")

        for _, row in train_data.iterrows():
            query = row['question']
            table = row.get('table', '')
            fields = row.get('field', '')

            if pd.isna(query) or pd.isna(table):
                continue

            table_simple = table.split('.')[-1]

            if pd.isna(fields) or not isinstance(fields, str):
                continue

            field_list = [f.strip() for f in fields.split('|') if f.strip()]
            query_lower = query.lower()

            # 判断是否有聚合意图
            has_agg_intent = any(kw in query_lower for kw in self.agg_keywords)

            for field_name in field_list:
                col_key = f"{table_simple}.{field_name}"

                if col_key not in self.field_roles:
                    self.field_roles[col_key] = FieldRole()

                # 获取字段类型
                field_type = self._field_type_cache.get(col_key, 'other')

                # 根据字段类型和查询特征判断角色
                # 时间和机构类字段通常是 WHERE 条件
                if field_type in ('time', 'org'):
                    self.field_roles[col_key].where_count += 1
                # 指标类字段
                elif field_type == 'metric':
                    self.field_roles[col_key].select_count += 1
                    if has_agg_intent:
                        self.field_roles[col_key].agg_count += 1
                # 其他字段
                else:
                    # 默认为 SELECT 字段
                    self.field_roles[col_key].select_count += 1

        print(f"  Seq2SQL: 学习了 {len(self.field_roles)} 个字段的角色统计")

    def _retrieve(self, query: str, k: int = 5) -> List[RetrievalResult]:
        """
        Seq2SQL 风格的结构化字段预测检索

        流程：
        1. Query 预处理和槽位识别
        2. 表级 Ranking
        3. 表内结构化字段预测（SELECT/WHERE/AGG）
        4. 合并字段集合

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

        # Step 4: 对每个候选表进行结构化字段预测
        results = []
        for table_name, score_dict in sorted_tables:
            # 结构化字段预测
            select_fields, where_fields, agg_fields = self._predict_fields_by_role(
                table_name, query, query_tokens, query_token_set, slots, score_dict
            )

            # 合并字段集合
            merged_columns = self._merge_field_sets(select_fields, where_fields, agg_fields)

            # 返回 top-10 字段（优先保证字段覆盖率）
            top_columns = merged_columns[:10]

            results.append(RetrievalResult(
                table=table_name,
                table_score=score_dict['total'],
                columns=[(f"C:{col}", score) for col, score in top_columns],
                metadata={
                    'method': 'Seq2SQL',
                    'select_fields': [f[0] for f in select_fields[:5]],
                    'where_fields': [f[0] for f in where_fields[:5]],
                    'agg_fields': [f[0] for f in agg_fields[:5]],
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

        # 时间槽位识别
        time_patterns = [
            r'\d{4}年\d{1,2}月',  # 2025年2月
            r'\d{4}年',           # 2025年
            r'\d{1,2}月',         # 2月
            r'本月', r'上月', r'去年', r'今年',
            r'去年同期', r'年初', r'年末',
            r'\d{4}-\d{1,2}',     # 2025-02
            r'\d{4}/\d{1,2}',     # 2025/02
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
        org_keywords = ['供电所', '公司', '单位', '部门', '台区', '站', '局', '所', '县', '市', '省', '分局']
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
            matched_fields = 0
            for field_name in self.table_to_columns.get(table_name, []):
                col_key = f"{table_name}.{field_name}"
                if col_key in self.column_tokens:
                    col_tokens = self.column_tokens[col_key]
                    overlap = len(query_token_set & col_tokens)
                    if overlap > 0:
                        matched_fields += 1

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

    def _predict_fields_by_role(self, table_name: str, query: str, query_tokens: List[str],
                                 query_token_set: Set[str], slots: List[SlotInfo],
                                 table_score_dict: Dict[str, float]) -> Tuple[List[Tuple[str, float]], ...]:
        """
        表内结构化字段预测

        分别预测三种字段角色：
        1. SELECT 字段
        2. WHERE 字段
        3. AGG 字段

        Returns:
            (select_fields, where_fields, agg_fields)
        """
        # 预测 SELECT 字段
        select_fields = self._predict_select_fields(
            table_name, query_tokens, query_token_set, slots, table_score_dict
        )

        # 预测 WHERE 字段
        where_fields = self._predict_where_fields(
            table_name, query_tokens, query_token_set, slots, table_score_dict
        )

        # 预测 AGG 字段
        agg_fields = self._predict_agg_fields(
            table_name, query_tokens, query_token_set, slots, table_score_dict, select_fields
        )

        return select_fields, where_fields, agg_fields

    def _predict_select_fields(self, table_name: str, query_tokens: List[str],
                                query_token_set: Set[str], slots: List[SlotInfo],
                                table_score_dict: Dict[str, float]) -> List[Tuple[str, float]]:
        """
        预测 SELECT 字段

        SELECT 字段通常是查询的目标指标。

        评分因素：
        - 字段名匹配分
        - 字段描述匹配分
        - SELECT 角色先验分
        - 历史术语分
        - 指标类字段加分
        """
        field_scores = []
        slot_types = set(s.slot_type for s in slots)

        # 判断是否有指标槽位
        has_metric_slot = 'metric' in slot_types

        for field_name in self.table_to_columns.get(table_name, []):
            col_key = f"{table_name}.{field_name}"

            if col_key not in self.column_docs:
                continue

            col_doc = self.column_docs[col_key]
            scores = {}

            # A. 字段名匹配分
            field_name_tokens = set(self.tokenize(col_doc['field_name']))
            name_overlap = len(query_token_set & field_name_tokens)
            scores['name_match'] = name_overlap / max(len(query_tokens), 1)

            # B. 字段描述匹配分
            field_desc = col_doc['field_desc']
            desc_tokens = set(t for t in self.tokenize(field_desc) if len(t) > 1)
            desc_overlap = len(query_token_set & desc_tokens)
            scores['desc_match'] = desc_overlap / max(len(query_tokens), 1)

            # C. SELECT 角色先验分
            role_prior = 0.0
            if col_key in self.field_roles:
                role_prior = self.field_roles[col_key].select_prior
            scores['role_prior'] = role_prior

            # D. 历史术语分
            term_history = 0.0
            if col_key in self.column_term_stats:
                for token in query_tokens:
                    if token in self.column_term_stats[col_key]:
                        term_history += self.column_term_stats[col_key][token]
            scores['term_history'] = min(term_history / max(len(query_tokens), 1), 1.0)

            # E. 指标类字段加分
            field_type = self._field_type_cache.get(col_key, 'other')
            metric_bonus = 1.0 if field_type == 'metric' else 0.0
            if has_metric_slot and field_type == 'metric':
                metric_bonus = 2.0
            scores['metric_bonus'] = min(metric_bonus, 1.0)

            # 计算总分
            total = (
                self.weights['select_name_match'] * scores['name_match'] +
                self.weights['select_desc_match'] * scores['desc_match'] +
                self.weights['select_role_prior'] * scores['role_prior'] +
                self.weights['select_term_history'] * scores['term_history'] +
                self.weights['select_metric_bonus'] * scores['metric_bonus']
            )

            field_scores.append((field_name, total))

        # 按得分排序
        field_scores.sort(key=lambda x: x[1], reverse=True)
        return field_scores

    def _predict_where_fields(self, table_name: str, query_tokens: List[str],
                               query_token_set: Set[str], slots: List[SlotInfo],
                               table_score_dict: Dict[str, float]) -> List[Tuple[str, float]]:
        """
        预测 WHERE 字段

        WHERE 字段通常是过滤条件，如时间、机构等。

        评分因素：
        - 槽位匹配分（时间槽位->时间字段，机构槽位->机构字段）
        - 字段描述匹配分
        - WHERE 角色先验分
        - 历史术语分
        - 类型加分（时间/机构类字段）
        """
        field_scores = []
        slot_types = set(s.slot_type for s in slots)

        # 判断是否有时间/机构槽位
        has_time_slot = 'time' in slot_types
        has_org_slot = 'org' in slot_types

        for field_name in self.table_to_columns.get(table_name, []):
            col_key = f"{table_name}.{field_name}"

            if col_key not in self.column_docs:
                continue

            col_doc = self.column_docs[col_key]
            scores = {}
            field_type = self._field_type_cache.get(col_key, 'other')

            # A. 槽位匹配分
            slot_match = 0.0
            if has_time_slot and field_type == 'time':
                slot_match = 2.0
            elif has_org_slot and field_type == 'org':
                slot_match = 2.0
            scores['slot_match'] = min(slot_match, 1.0)

            # B. 字段描述匹配分
            field_desc = col_doc['field_desc']
            desc_tokens = set(t for t in self.tokenize(field_desc) if len(t) > 1)
            desc_overlap = len(query_token_set & desc_tokens)
            scores['desc_match'] = desc_overlap / max(len(query_tokens), 1)

            # C. WHERE 角色先验分
            role_prior = 0.0
            if col_key in self.field_roles:
                role_prior = self.field_roles[col_key].where_prior
            scores['role_prior'] = role_prior

            # D. 历史术语分
            term_history = 0.0
            if col_key in self.column_term_stats:
                for token in query_tokens:
                    if token in self.column_term_stats[col_key]:
                        term_history += self.column_term_stats[col_key][token]
            scores['term_history'] = min(term_history / max(len(query_tokens), 1), 1.0)

            # E. 类型加分
            type_bonus = 0.0
            if field_type == 'time':
                type_bonus = 0.8
            elif field_type == 'org':
                type_bonus = 0.8
            scores['type_bonus'] = type_bonus

            # 计算总分
            total = (
                self.weights['where_slot_match'] * scores['slot_match'] +
                self.weights['where_desc_match'] * scores['desc_match'] +
                self.weights['where_role_prior'] * scores['role_prior'] +
                self.weights['where_term_history'] * scores['term_history'] +
                self.weights['where_type_bonus'] * scores['type_bonus']
            )

            field_scores.append((field_name, total))

        # 按得分排序
        field_scores.sort(key=lambda x: x[1], reverse=True)
        return field_scores

    def _predict_agg_fields(self, table_name: str, query_tokens: List[str],
                             query_token_set: Set[str], slots: List[SlotInfo],
                             table_score_dict: Dict[str, float],
                             select_fields: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        """
        预测 AGG 字段

        AGG 字段通常是聚合函数作用的数值型指标字段。

        评分因素：
        - 数值型字段加分
        - AGG 角色先验分
        - 指标词匹配分
        - 与 SELECT 字段重叠加分
        """
        field_scores = []
        query_lower = ' '.join(query_tokens).lower()

        # 判断是否有聚合意图
        has_agg_intent = any(kw in query_lower for kw in self.agg_keywords)

        # 获取 SELECT 字段集合
        select_field_set = set(f[0] for f in select_fields[:5])

        # 获取槽位类型
        slot_types = set(s.slot_type for s in slots)
        has_metric_slot = 'metric' in slot_types

        for field_name in self.table_to_columns.get(table_name, []):
            col_key = f"{table_name}.{field_name}"

            if col_key not in self.column_docs:
                continue

            col_doc = self.column_docs[col_key]
            scores = {}
            field_type = self._field_type_cache.get(col_key, 'other')

            # A. 数值型字段加分
            numeric_bonus = 0.0
            if field_type == 'metric':
                numeric_bonus = 1.5
                if has_agg_intent:
                    numeric_bonus = 2.0
            scores['numeric_bonus'] = min(numeric_bonus, 1.0)

            # B. AGG 角色先验分
            role_prior = 0.0
            if col_key in self.field_roles:
                role_prior = self.field_roles[col_key].agg_prior
            scores['role_prior'] = role_prior

            # C. 指标词匹配分
            metric_match = 0.0
            field_name_lower = col_doc['field_name'].lower()
            for token in query_tokens:
                if token in self.domain_keywords:
                    related_fields = self.domain_keywords[token]
                    for rf in related_fields:
                        if rf.lower() in field_name_lower:
                            metric_match += 0.5
                            break
            scores['metric_match'] = min(metric_match / max(len(query_tokens), 1), 1.0)

            # D. 与 SELECT 字段重叠加分
            select_overlap = 0.0
            if field_name in select_field_set:
                select_overlap = 1.0
            scores['select_overlap'] = select_overlap

            # 如果有指标槽位，额外加分
            if has_metric_slot and field_type == 'metric':
                scores['metric_match'] = min(scores['metric_match'] + 0.5, 1.0)

            # 计算总分
            total = (
                self.weights['agg_numeric_bonus'] * scores['numeric_bonus'] +
                self.weights['agg_role_prior'] * scores['role_prior'] +
                self.weights['agg_metric_match'] * scores['metric_match'] +
                self.weights['agg_select_overlap'] * scores['select_overlap']
            )

            field_scores.append((field_name, total))

        # 按得分排序
        field_scores.sort(key=lambda x: x[1], reverse=True)
        return field_scores

    def _merge_field_sets(self, select_fields: List[Tuple[str, float]],
                          where_fields: List[Tuple[str, float]],
                          agg_fields: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        """
        合并三种字段集合

        策略：
        1. 取 SELECT 字段 Top-5
        2. 取 WHERE 字段 Top-3
        3. 取 AGG 字段 Top-2
        4. 合并去重，按得分排序
        """
        merged = {}

        # SELECT 字段（权重 0.5）
        for field_name, score in select_fields[:5]:
            if field_name not in merged:
                merged[field_name] = 0.0
            merged[field_name] = max(merged[field_name], score * 0.5)

        # WHERE 字段（权重 0.3）
        for field_name, score in where_fields[:3]:
            if field_name not in merged:
                merged[field_name] = 0.0
            merged[field_name] = max(merged[field_name], score * 0.3)

        # AGG 字段（权重 0.2）
        for field_name, score in agg_fields[:2]:
            if field_name not in merged:
                merged[field_name] = 0.0
            merged[field_name] = max(merged[field_name], score * 0.2)

        # 按得分排序
        sorted_fields = sorted(merged.items(), key=lambda x: x[1], reverse=True)

        return sorted_fields
