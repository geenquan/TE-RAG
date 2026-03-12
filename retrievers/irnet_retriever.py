"""
IRNet 检索器实现

基于 IRNet 论文的核心思想（Intermediate Representation），
适配当前项目的表检索 + 字段检索框架。

核心特点：
1. 通过中间表示（IR）桥接自然语言和 Schema 的语义鸿沟
2. 推断 query 的逻辑结构（SELECT/FILTER/GROUP/ORDER）
3. 根据 IR 结构预测字段角色
4. 优化字段集合覆盖率

参考文献：
- IRNet: Mapping Natural Language Questions to SQL with Intermediate Representation (VLDB 2019)

注意：当前实现不生成 SQL 或 SemQL，只做 IR-guided 的字段集合预测
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
    slot_type: str          # time, org, metric, order
    value: str              # 原始值
    tokens: Set[str]        # 分词后的词集合


@dataclass
class IRStructure:
    """
    IRNet 中间表示结构

    表示 query 的逻辑结构，包含：
    - SELECT_SLOT: 查询的目标字段
    - FILTER_SLOT: 过滤条件字段
    - GROUP_SLOT: 分组字段
    - ORDER_SLOT: 排序字段
    """
    select_slots: List[SlotInfo] = dataclass_field(default_factory=list)
    filter_slots: List[SlotInfo] = dataclass_field(default_factory=list)
    group_slots: List[SlotInfo] = dataclass_field(default_factory=list)
    order_slots: List[SlotInfo] = dataclass_field(default_factory=list)

    def has_select(self) -> bool:
        return len(self.select_slots) > 0

    def has_filter(self) -> bool:
        return len(self.filter_slots) > 0

    def has_group(self) -> bool:
        return len(self.group_slots) > 0

    def has_order(self) -> bool:
        return len(self.order_slots) > 0

    def get_slot_types(self) -> Set[str]:
        """获取所有槽位类型"""
        types = set()
        for slot in self.select_slots:
            types.add(slot.slot_type)
        for slot in self.filter_slots:
            types.add(slot.slot_type)
        for slot in self.group_slots:
            types.add(slot.slot_type)
        for slot in self.order_slots:
            types.add(slot.slot_type)
        return types


@dataclass
class FieldIRRole:
    """字段 IR 角色统计"""
    select_count: int = 0   # 作为 SELECT 字段的次数
    filter_count: int = 0   # 作为 FILTER 字段的次数
    group_count: int = 0    # 作为 GROUP 字段的次数
    order_count: int = 0    # 作为 ORDER 字段的次数

    @property
    def total(self) -> int:
        return self.select_count + self.filter_count + self.group_count + self.order_count

    @property
    def select_prior(self) -> float:
        return self.select_count / max(self.total, 1)

    @property
    def filter_prior(self) -> float:
        return self.filter_count / max(self.total, 1)

    @property
    def group_prior(self) -> float:
        return self.group_count / max(self.total, 1)

    @property
    def order_prior(self) -> float:
        return self.order_count / max(self.total, 1)


class IRNetRetriever(BaseRetriever):
    """
    IRNet 风格的 IR-guided Schema Retrieval 检索器

    基于 IRNet 论文的核心思想：
    - 通过中间表示（IR）桥接自然语言和 Schema 的语义鸿沟
    - 推断 query 的逻辑结构
    - 根据 IR 结构预测字段角色

    当前实现：
    - 第一阶段：IR 结构推断（SELECT/FILTER/GROUP/ORDER 槽位）
    - 第二阶段：表级 ranking
    - 第三阶段：表内字段 IR 角色预测
    - 第四阶段：合并字段集合

    不包含：
    - 完整 SemQL 生成
    - SQL decoder
    - 端到端 SQL 生成
    """

    def __init__(self, field_csv: str, table_csv: str,
                 config: Optional[RetrieverConfig] = None):
        """
        初始化 IRNet 检索器

        Args:
            field_csv: 字段CSV路径
            table_csv: 表CSV路径
            config: 检索器配置
        """
        if config is None:
            config = RetrieverConfig(
                name="IRNet",
                description="IRNet-style IR-guided schema retrieval retriever"
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

        # ============ 字段 IR 角色统计（从训练数据学习） ============
        self.field_ir_roles: Dict[str, FieldIRRole] = {}  # "table.field" -> FieldIRRole

        # ============ 统计信息（从训练数据学习） ============
        # 表术语统计：query term -> 哪些表更容易被命中
        self.table_term_stats: Dict[str, Dict[str, float]] = {}  # table -> {term: weight}

        # 字段术语统计：query term -> 哪些字段更容易被命中
        self.column_term_stats: Dict[str, Dict[str, float]] = {} # table.field -> {term: weight}

        # ============ 字段类别映射 ============
        # 时间类字段（通常用于 FILTER）
        self.time_field_patterns = [
            'dt', 'date', 'time', 'rq', 'month', 'year', 'stat_date',
            'create_time', 'update_time', 'ym', 'ymd', 'day', 'hour',
            'sjsj', 'tjsj', 'cjsj', 'gxsj', 'tjrq', 'cjsj', 'sj'
        ]

        # 机构/实体类字段（通常用于 FILTER）
        self.org_field_patterns = [
            'org', 'dept', 'unit', 'company', 'branch', 'station',
            'area', 'city', 'province', 'region', 'team', 'group',
            'name', '单位', '机构', '部门', '供电所', '公司', 'dwmc', 'jgmc',
            'gds', 'gs', 'dw', 'jg', 'bm', 'xzq'
        ]

        # 指标/数值类字段（通常用于 SELECT/ORDER）
        self.metric_field_patterns = [
            'cnt', 'count', 'sum', 'avg', 'amount', 'fee', 'rate',
            'qty', 'quantity', 'value', 'num', 'money', 'score',
            '电量', '电费', '户数', '金额', '数量', '率', 'sdl', 'df', 'hs',
            'total', 'je', 'sl', 'je', 'ljl', 'tbzz', 'hbzz'
        ]

        # 分组类字段（通常是类别型字段）
        self.category_field_patterns = [
            'type', 'category', 'class', 'kind', 'status', 'level',
            'grade', 'region', 'area', 'group', 'flag', 'code',
            '类型', '类别', '等级', '状态', '分类', 'lb', 'lx', 'dj', 'zt'
        ]

        # ============ 槽位关键词 ============
        # SELECT 槽位关键词（暗示查询目标）
        self.select_keywords = [
            '售电量', '电费', '户数', '金额', '数量', '线损', '电压', '电量',
            '供电量', '用电量', '回收率', '费用', '欠费', '合计', '总计'
        ]

        # FILTER 槽位关键词（暗示过滤条件）
        self.filter_keywords = [
            '时间', '日期', '年', '月', '日', '季度', '周',
            '供电所', '公司', '单位', '部门', '机构', '区域', '城市',
            '类型', '状态', '等级', '分类'
        ]

        # GROUP 槽位关键词（暗示分组）
        self.group_keywords = [
            '按', '分组', '每组', '各类', '各个', '分别', '各',
            '统计', '汇总', '分类', 'group by'
        ]

        # ORDER 槽位关键词（暗示排序）
        self.order_keywords = [
            '最高', '最大', '最低', '最小', '排名', '前', 'top',
            '排序', '降序', '升序', '递增', '递减', 'order by',
            '多少', '多少个', '第一', '倒数'
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
            'select_slot_match': 0.20,
            'select_metric_bonus': 0.15,

            # FILTER 字段权重
            'filter_slot_match': 0.30,
            'filter_desc_match': 0.15,
            'filter_role_prior': 0.25,
            'filter_type_bonus': 0.20,
            'filter_history': 0.10,

            # GROUP 字段权重
            'group_role_prior': 0.40,
            'group_category_bonus': 0.35,
            'group_slot_match': 0.25,

            # ORDER 字段权重
            'order_numeric_bonus': 0.35,
            'order_role_prior': 0.30,
            'order_slot_match': 0.35,
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
            train_data: 训练数据（用于统计术语-表/字段的关联和字段 IR 角色）
        """
        print("  IRNet: 构建索引...")

        # 1. 构建表索引
        self._build_table_index()

        # 2. 构建字段索引
        self._build_column_index()

        # 3. 从训练数据学习术语统计和字段 IR 角色
        if train_data is not None and len(train_data) > 0:
            self._learn_term_stats(train_data)
            self._learn_field_ir_roles(train_data)

        self._is_fitted = True
        print("  IRNet: 索引构建完成")

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
            'time', 'org', 'metric', 'category', or 'other'
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

        # 检查类别类
        for pattern in self.category_field_patterns:
            if pattern in text:
                return 'category'

        return 'other'

    def _learn_term_stats(self, train_data: pd.DataFrame):
        """
        从训练数据学习术语统计

        统计 query term 与 表/字段的关联权重
        """
        print("  IRNet: 学习术语统计...")

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

        print(f"  IRNet: 学习了 {len(self.table_term_stats)} 个表的术语统计")
        print(f"  IRNet: 学习了 {len(self.column_term_stats)} 个字段的术语统计")

    def _learn_field_ir_roles(self, train_data: pd.DataFrame):
        """
        从训练数据学习字段 IR 角色

        统计每个字段在 SQL 中的逻辑角色：
        - SELECT 字段：查询的目标字段
        - FILTER 字段：过滤条件字段
        - GROUP 字段：分组字段
        - ORDER 字段：排序字段
        """
        print("  IRNet: 学习字段 IR 角色...")

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

            # 判断 IR 结构意图
            has_group_intent = any(kw in query_lower for kw in self.group_keywords)
            has_order_intent = any(kw in query_lower for kw in self.order_keywords)

            for field_name in field_list:
                col_key = f"{table_simple}.{field_name}"

                if col_key not in self.field_ir_roles:
                    self.field_ir_roles[col_key] = FieldIRRole()

                # 获取字段类型
                field_type = self._field_type_cache.get(col_key, 'other')

                # 根据字段类型和查询特征判断 IR 角色

                # 时间和机构类字段通常是 FILTER 角色
                if field_type in ('time', 'org'):
                    self.field_ir_roles[col_key].filter_count += 1

                # 类别类字段可能是 GROUP 角色
                elif field_type == 'category':
                    if has_group_intent:
                        self.field_ir_roles[col_key].group_count += 1
                    else:
                        self.field_ir_roles[col_key].filter_count += 1

                # 指标类字段
                elif field_type == 'metric':
                    self.field_ir_roles[col_key].select_count += 1
                    if has_order_intent:
                        self.field_ir_roles[col_key].order_count += 1

                # 其他字段
                else:
                    # 默认为 SELECT 角色
                    self.field_ir_roles[col_key].select_count += 1

        print(f"  IRNet: 学习了 {len(self.field_ir_roles)} 个字段的 IR 角色统计")

    def _retrieve(self, query: str, k: int = 5) -> List[RetrievalResult]:
        """
        IRNet 风格的 IR-guided 检索

        流程：
        1. Query 预处理和槽位识别
        2. IR 结构推断
        3. 表级 Ranking
        4. 表内字段 IR 角色预测
        5. 合并字段集合

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

        # Step 2: IR 结构推断
        ir_structure = self._infer_ir_structure(query, query_tokens, slots)

        # Step 3: 表级 Ranking
        table_scores = self._rank_tables(query, query_tokens, query_token_set, ir_structure)

        # Step 4: 选取 Top-k 表
        sorted_tables = sorted(table_scores.items(), key=lambda x: x[1]['total'], reverse=True)[:k]

        # Step 5: 对每个候选表进行字段 IR 角色预测
        results = []
        for table_name, score_dict in sorted_tables:
            # 字段 IR 角色预测
            select_fields, filter_fields, group_fields, order_fields = self._predict_fields_by_ir_role(
                table_name, query, query_tokens, query_token_set, ir_structure, score_dict
            )

            # 合并字段集合
            merged_columns = self._merge_field_sets(select_fields, filter_fields, group_fields, order_fields)

            # 返回 top-10 字段
            top_columns = merged_columns[:10]

            results.append(RetrievalResult(
                table=table_name,
                table_score=score_dict['total'],
                columns=[(f"C:{col}", score) for col, score in top_columns],
                metadata={
                    'method': 'IRNet',
                    'select_fields': [f[0] for f in select_fields[:5]],
                    'filter_fields': [f[0] for f in filter_fields[:5]],
                    'group_fields': [f[0] for f in group_fields[:3]],
                    'order_fields': [f[0] for f in order_fields[:3]],
                    'ir_structure': {
                        'has_select': ir_structure.has_select(),
                        'has_filter': ir_structure.has_filter(),
                        'has_group': ir_structure.has_group(),
                        'has_order': ir_structure.has_order(),
                    },
                    'score_components': {
                        'table_name_match': score_dict.get('table_name_match', 0),
                        'table_desc_match': score_dict.get('table_desc_match', 0),
                        'column_vote_score': score_dict.get('column_vote_score', 0),
                        'template_match_score': score_dict.get('template_match_score', 0),
                        'domain_term_score': score_dict.get('domain_term_score', 0),
                    }
                }
            ))

        return results

    def _extract_slots(self, query: str, query_tokens: List[str]) -> List[SlotInfo]:
        """
        提取查询中的槽位信息

        识别：时间、机构、指标、排序等槽位
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
            r'第\d+季度',         # 第一季度
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

        # 排序槽位识别
        order_keywords = ['最高', '最大', '最低', '最小', '排名', '前', 'top',
                         '第一', '倒数', '多少']
        query_lower = query.lower()
        for kw in order_keywords:
            if kw in query_lower:
                slots.append(SlotInfo(
                    slot_type='order',
                    value=kw,
                    tokens={kw}
                ))
                break

        return slots

    def _infer_ir_structure(self, query: str, query_tokens: List[str],
                            slots: List[SlotInfo]) -> IRStructure:
        """
        推断 IR 结构

        根据 query 和槽位推断 IR 结构：
        - SELECT_SLOT: 查询目标
        - FILTER_SLOT: 过滤条件
        - GROUP_SLOT: 分组字段
        - ORDER_SLOT: 排序字段
        """
        ir = IRStructure()
        query_lower = query.lower()

        # 推断 SELECT 槽位（指标类槽位）
        for slot in slots:
            if slot.slot_type == 'metric':
                ir.select_slots.append(slot)

        # 推断 FILTER 槽位（时间和机构类槽位）
        for slot in slots:
            if slot.slot_type in ('time', 'org'):
                ir.filter_slots.append(slot)

        # 推断 GROUP 槽位
        has_group_intent = any(kw in query_lower for kw in self.group_keywords)
        if has_group_intent:
            # 寻找分组依据（通常是机构或类别槽位）
            for slot in slots:
                if slot.slot_type in ('org', 'category'):
                    ir.group_slots.append(slot)

        # 推断 ORDER 槽位
        has_order_intent = any(kw in query_lower for kw in self.order_keywords)
        if has_order_intent:
            # 寻找排序字段（通常是指标槽位）
            for slot in slots:
                if slot.slot_type == 'metric':
                    ir.order_slots.append(slot)

        return ir

    def _rank_tables(self, query: str, query_tokens: List[str],
                     query_token_set: Set[str], ir_structure: IRStructure) -> Dict[str, Dict[str, float]]:
        """
        表级 Ranking

        计算每个表的综合得分
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

            # C. 字段投票分
            matched_fields = 0
            for field_name in self.table_to_columns.get(table_name, []):
                col_key = f"{table_name}.{field_name}"
                if col_key in self.column_tokens:
                    col_tokens = self.column_tokens[col_key]
                    overlap = len(query_token_set & col_tokens)
                    if overlap > 0:
                        matched_fields += 1

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

            scores['template_match_score'] = min(template_score / max(len(query_tokens), 1), 1.0)

            # E. 业务词分
            domain_score = 0
            for token in query_tokens:
                if token in self.domain_keywords:
                    related_fields = self.domain_keywords[token]
                    for field in related_fields:
                        if field.lower() in table_name.lower() or field.lower() in table_desc.lower():
                            domain_score += 1
                            break
                        for fn in self.table_to_columns.get(table_name, []):
                            if field.lower() in fn.lower():
                                domain_score += 0.5
                                break

            scores['domain_term_score'] = min(domain_score / max(len(query_tokens), 1), 1.0)

            # 计算总分
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

    def _predict_fields_by_ir_role(self, table_name: str, query: str, query_tokens: List[str],
                                    query_token_set: Set[str], ir_structure: IRStructure,
                                    table_score_dict: Dict[str, float]) -> Tuple[List[Tuple[str, float]], ...]:
        """
        表内字段 IR 角色预测

        根据 IR 结构预测四种字段角色：
        1. SELECT 字段
        2. FILTER 字段
        3. GROUP 字段
        4. ORDER 字段

        Returns:
            (select_fields, filter_fields, group_fields, order_fields)
        """
        # 预测 SELECT 字段
        select_fields = self._predict_select_fields(
            table_name, query_tokens, query_token_set, ir_structure, table_score_dict
        )

        # 预测 FILTER 字段
        filter_fields = self._predict_filter_fields(
            table_name, query_tokens, query_token_set, ir_structure, table_score_dict
        )

        # 预测 GROUP 字段
        group_fields = self._predict_group_fields(
            table_name, query_tokens, query_token_set, ir_structure, table_score_dict
        )

        # 预测 ORDER 字段
        order_fields = self._predict_order_fields(
            table_name, query_tokens, query_token_set, ir_structure, table_score_dict
        )

        return select_fields, filter_fields, group_fields, order_fields

    def _predict_select_fields(self, table_name: str, query_tokens: List[str],
                                query_token_set: Set[str], ir_structure: IRStructure,
                                table_score_dict: Dict[str, float]) -> List[Tuple[str, float]]:
        """
        预测 SELECT 字段

        SELECT 字段是查询的目标字段，通常是指标类字段。
        """
        field_scores = []

        # 获取 SELECT 槽位类型
        select_slot_types = set(s.slot_type for s in ir_structure.select_slots)
        has_metric_slot = 'metric' in select_slot_types

        for field_name in self.table_to_columns.get(table_name, []):
            col_key = f"{table_name}.{field_name}"

            if col_key not in self.column_docs:
                continue

            col_doc = self.column_docs[col_key]
            scores = {}
            field_type = self._field_type_cache.get(col_key, 'other')

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
            if col_key in self.field_ir_roles:
                role_prior = self.field_ir_roles[col_key].select_prior
            scores['role_prior'] = role_prior

            # D. 槽位匹配分
            slot_match = 0.0
            if has_metric_slot and field_type == 'metric':
                slot_match = 1.0
            scores['slot_match'] = slot_match

            # E. 指标类字段加分
            metric_bonus = 0.0
            if field_type == 'metric':
                metric_bonus = 0.8
                if has_metric_slot:
                    metric_bonus = 1.2
            scores['metric_bonus'] = min(metric_bonus, 1.0)

            # 计算总分
            total = (
                self.weights['select_name_match'] * scores['name_match'] +
                self.weights['select_desc_match'] * scores['desc_match'] +
                self.weights['select_role_prior'] * scores['role_prior'] +
                self.weights['select_slot_match'] * scores['slot_match'] +
                self.weights['select_metric_bonus'] * scores['metric_bonus']
            )

            field_scores.append((field_name, total))

        field_scores.sort(key=lambda x: x[1], reverse=True)
        return field_scores

    def _predict_filter_fields(self, table_name: str, query_tokens: List[str],
                                query_token_set: Set[str], ir_structure: IRStructure,
                                table_score_dict: Dict[str, float]) -> List[Tuple[str, float]]:
        """
        预测 FILTER 字段

        FILTER 字段是过滤条件，通常是时间和机构类字段。
        """
        field_scores = []

        # 获取 FILTER 槽位类型
        filter_slot_types = set(s.slot_type for s in ir_structure.filter_slots)
        has_time_slot = 'time' in filter_slot_types
        has_org_slot = 'org' in filter_slot_types

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

            # C. FILTER 角色先验分
            role_prior = 0.0
            if col_key in self.field_ir_roles:
                role_prior = self.field_ir_roles[col_key].filter_prior
            scores['role_prior'] = role_prior

            # D. 类型加分
            type_bonus = 0.0
            if field_type == 'time':
                type_bonus = 0.8
            elif field_type == 'org':
                type_bonus = 0.8
            elif field_type == 'category':
                type_bonus = 0.5
            scores['type_bonus'] = type_bonus

            # E. 历史术语分
            history = 0.0
            if col_key in self.column_term_stats:
                for token in query_tokens:
                    if token in self.column_term_stats[col_key]:
                        history += self.column_term_stats[col_key][token]
            scores['history'] = min(history / max(len(query_tokens), 1), 1.0)

            # 计算总分
            total = (
                self.weights['filter_slot_match'] * scores['slot_match'] +
                self.weights['filter_desc_match'] * scores['desc_match'] +
                self.weights['filter_role_prior'] * scores['role_prior'] +
                self.weights['filter_type_bonus'] * scores['type_bonus'] +
                self.weights['filter_history'] * scores['history']
            )

            field_scores.append((field_name, total))

        field_scores.sort(key=lambda x: x[1], reverse=True)
        return field_scores

    def _predict_group_fields(self, table_name: str, query_tokens: List[str],
                               query_token_set: Set[str], ir_structure: IRStructure,
                               table_score_dict: Dict[str, float]) -> List[Tuple[str, float]]:
        """
        预测 GROUP 字段

        GROUP 字段是分组字段，通常是类别型字段。
        """
        field_scores = []

        # 判断是否有分组意图
        has_group_intent = ir_structure.has_group()
        group_slot_types = set(s.slot_type for s in ir_structure.group_slots)

        for field_name in self.table_to_columns.get(table_name, []):
            col_key = f"{table_name}.{field_name}"

            if col_key not in self.column_docs:
                continue

            col_doc = self.column_docs[col_key]
            scores = {}
            field_type = self._field_type_cache.get(col_key, 'other')

            # A. GROUP 角色先验分
            role_prior = 0.0
            if col_key in self.field_ir_roles:
                role_prior = self.field_ir_roles[col_key].group_prior
            scores['role_prior'] = role_prior

            # B. 类别型字段加分
            category_bonus = 0.0
            if field_type == 'category':
                category_bonus = 1.0
                if has_group_intent:
                    category_bonus = 1.5
            elif field_type == 'org':
                category_bonus = 0.5
                if has_group_intent:
                    category_bonus = 0.8
            scores['category_bonus'] = min(category_bonus, 1.0)

            # C. 槽位匹配分
            slot_match = 0.0
            if field_type in group_slot_types:
                slot_match = 1.0
            elif has_group_intent and field_type in ('org', 'category'):
                slot_match = 0.8
            scores['slot_match'] = slot_match

            # 计算总分
            total = (
                self.weights['group_role_prior'] * scores['role_prior'] +
                self.weights['group_category_bonus'] * scores['category_bonus'] +
                self.weights['group_slot_match'] * scores['slot_match']
            )

            field_scores.append((field_name, total))

        field_scores.sort(key=lambda x: x[1], reverse=True)
        return field_scores

    def _predict_order_fields(self, table_name: str, query_tokens: List[str],
                               query_token_set: Set[str], ir_structure: IRStructure,
                               table_score_dict: Dict[str, float]) -> List[Tuple[str, float]]:
        """
        预测 ORDER 字段

        ORDER 字段是排序字段，通常是数值型字段。
        """
        field_scores = []

        # 判断是否有排序意图
        has_order_intent = ir_structure.has_order()
        order_slot_types = set(s.slot_type for s in ir_structure.order_slots)

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
                numeric_bonus = 1.0
                if has_order_intent:
                    numeric_bonus = 1.5
            scores['numeric_bonus'] = min(numeric_bonus, 1.0)

            # B. ORDER 角色先验分
            role_prior = 0.0
            if col_key in self.field_ir_roles:
                role_prior = self.field_ir_roles[col_key].order_prior
            scores['role_prior'] = role_prior

            # C. 槽位匹配分
            slot_match = 0.0
            if 'metric' in order_slot_types and field_type == 'metric':
                slot_match = 1.0
            elif has_order_intent and field_type == 'metric':
                slot_match = 0.8
            scores['slot_match'] = slot_match

            # 计算总分
            total = (
                self.weights['order_numeric_bonus'] * scores['numeric_bonus'] +
                self.weights['order_role_prior'] * scores['role_prior'] +
                self.weights['order_slot_match'] * scores['slot_match']
            )

            field_scores.append((field_name, total))

        field_scores.sort(key=lambda x: x[1], reverse=True)
        return field_scores

    def _merge_field_sets(self, select_fields: List[Tuple[str, float]],
                          filter_fields: List[Tuple[str, float]],
                          group_fields: List[Tuple[str, float]],
                          order_fields: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        """
        合并四种字段集合

        策略：
        1. 取 SELECT 字段 Top-5
        2. 取 FILTER 字段 Top-3
        3. 取 GROUP 字段 Top-2
        4. 取 ORDER 字段 Top-2
        5. 合并去重，按得分排序
        """
        merged = {}

        # SELECT 字段（权重 0.4）
        for field_name, score in select_fields[:5]:
            if field_name not in merged:
                merged[field_name] = 0.0
            merged[field_name] = max(merged[field_name], score * 0.4)

        # FILTER 字段（权重 0.3）
        for field_name, score in filter_fields[:3]:
            if field_name not in merged:
                merged[field_name] = 0.0
            merged[field_name] = max(merged[field_name], score * 0.3)

        # GROUP 字段（权重 0.15）
        for field_name, score in group_fields[:2]:
            if field_name not in merged:
                merged[field_name] = 0.0
            merged[field_name] = max(merged[field_name], score * 0.15)

        # ORDER 字段（权重 0.15）
        for field_name, score in order_fields[:2]:
            if field_name not in merged:
                merged[field_name] = 0.0
            merged[field_name] = max(merged[field_name], score * 0.15)

        # 按得分排序
        sorted_fields = sorted(merged.items(), key=lambda x: x[1], reverse=True)

        return sorted_fields
