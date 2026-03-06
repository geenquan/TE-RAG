"""
SQL 生成器

基于模板和规则的 SQL 生成器，支持:
1. 模板匹配：从历史 SQL 中提取模板模式
2. JOIN 推断：基于规则推断 JOIN 路径
3. SQL-EM 指标计算

使用方式:
    generator = TemplateSQLGenerator(config)
    generator.extract_templates(train_data)
    sql = generator.generate(query, retrieved_tables, retrieved_columns)
"""

import re
import json
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict
from abc import ABC, abstractmethod

from terag.config import TERAGConfig


@dataclass
class SQLTemplate:
    """SQL 模板"""
    template_id: str
    template_text: str           # 模板文本，带占位符
    query_type: str             # 查询类型: statistical, list, ranking, aggregate
    table_slots: List[str]       # 表名占位符列表
    column_slots: List[str]      # 列名占位符列表
    condition_slots: List[str]   # 条件占位符列表
    source_sql: str             # 原始 SQL
    frequency: int = 1            # 出现频率


@dataclass
class GeneratedSQL:
    """生成的 SQL"""
    sql: str
    template_id: Optional[str]
    confidence: float
    tables: List[str]
    columns: List[str]
    join_conditions: List[Tuple[str, str, str]]  # (table1, table2, condition)


class SQLGenerator(ABC):
    """SQL 生成器基类"""

    @abstractmethod
    def generate(
        self,
        query: str,
        retrieved_tables: List[str],
        retrieved_columns: List[Tuple[str, str]],  # (table, column)
        join_candidates: Optional[List[Tuple[str, str]]] = None
    ) -> GeneratedSQL:
        """生成 SQL"""
        pass


class JoinInferencer:
    """
    JOIN 掌断器

    基于规则推断表之间的 JOIN 路径
    """

    def __init__(self, field_df: pd.DataFrame):
        """
        初始化 JOIN 推断器

        Args:
            field_df: 字段 DataFrame
        """
        self.field_df = field_df
        self._build_field_index()

    def _build_field_index(self):
        """构建字段索引"""
        self.table_fields = defaultdict(set)
        self.field_tables = defaultdict(set)

        for _, row in self.field_df.iterrows():
            table = row['table']
            field = row['field_name']
            self.table_fields[table].add(field.lower())
            self.field_tables[field.lower()].add(table)

    def infer_joins(
        self,
        tables: List[str]
    ) -> List[Tuple[str, str, str]]:
        """
        推断 JOIN 路径

        Args:
            tables: 表名列表

        Returns:
            [(table1, table2, join_condition), ...]
        """
        if len(tables) <= 1:
            return []

        joins = []

        # 策略1: 基于共同字段名（如 id, table1_id 等)
        for i, t1 in enumerate(tables):
            for t2 in tables[i+1:]:
                join_cond = self._find_join_condition(t1, t2)
                if join_cond:
                    joins.append((t1, t2, join_cond))

        return joins

    def _find_join_condition(self, table1: str, table2: str) -> Optional[str]:
        """查找两个表之间的 JOIN 条件"""
        fields1 = self.table_fields.get(table1, set())
        fields2 = self.table_fields.get(table2, set())

        # 策略1: 查找同名主键
        common_fields = fields1 & fields2
        for field in common_fields:
            if field in ['id', 'ID', 'pk', 'PK', 'key', 'KEY']:
                return f"{table1}.{field} = {table2}.{field}"

        # 策略2: 查找外键关系（table1_id = table2.id)
        for field1 in fields1:
            if field1.endswith('_id') or field1.endswith('_ID'):
                base_name = field1[:-3]  # 去掉 _id
                if base_name in fields2:
                    return f"{table1}.{field1} = {table2}.{base_name}"

        for field2 in fields2:
            if field2.endswith('_id') or field2.endswith('_ID'):
                base_name = field2[:-3]
                if base_name in fields1:
                    return f"{table1}.{base_name} = {table2}.{field2}"

        # 策略3: 通用 ID 字段
        for field in common_fields:
            if 'id' in field.lower() or 'code' in field.lower():
                return f"{table1}.{field} = {table2}.{field}"

        return None


class TemplateSQLGenerator(SQLGenerator):
    """
    基于模板的 SQL 生成器

    策略:
    1. 从历史 SQL 中提取模板模式
    2. 根据查询类型（统计/列表/排名)匹配模板
    3. 填充表名、字段名、 WHERE 条件
    """

    def __init__(self, config: TERAGConfig, field_df: pd.DataFrame = None):
        self.config = config
        self.field_df = field_df or pd.read_csv(config.data.field_csv)
        self.join_inferencer = JoinInferencer(self.field_df)

        self.templates: Dict[str, List[SQLTemplate]] = defaultdict(list)
        self.query_type_keywords = {
            'statistical': ['统计', '计算', '求和', '总计', '总数', 'count', 'sum', 'total'],
            'list': ['查询', '列出', '显示', '获取', '所有', 'select', 'list', 'show', 'get'],
            'ranking': ['排名', '排行', '前N', '排序', '最大', '最小', 'top', 'rank', 'order by', 'max', 'min'],
            'aggregate': ['分组', '按', '各个', '各类', 'group by', 'each', 'per'],
            'template_match': ['匹配', '符合', 'fit', 'match'],
        }

    def extract_templates(self, train_data: pd.DataFrame) -> int:
        """
        从训练数据提取 SQL 模板

        Args:
            train_data: 训练数据，包含 sql 列

        Returns:
            提取的模板数量
        """
        template_count = 0

        for _, row in train_data.iterrows():
            sql = row.get('sql', '')
            if pd.isna(sql) or not sql:
                continue

            template = self._extract_single_template(str(sql), row)
            if template:
                # 检查是否已存在相似模板
                existing = self._find_similar_template(template)
                if existing:
                    existing.frequency += 1
                else:
                    self.templates[template.query_type].append(template)
                    template_count += 1

        print(f"从 {len(train_data)} 条训练数据中提取了 {template_count} 个 SQL 模板")
        return template_count

    def _extract_single_template(self, sql: str, row: pd.Series) -> Optional[SQLTemplate]:
        """从单个 SQL 提取模板"""
        # 确定查询类型
        query_type = self._classify_query_type(row.get('question', ''))

        # 提取表名
        tables = self._extract_tables(sql)
        if not tables:
            return None

        # 提取列名
        columns = self._extract_columns(sql)

        # 提取条件
        conditions = self._extract_conditions(sql)

        # 生成模板文本（将具体值替换为占位符）
        template_text = self._generalize_sql(sql, tables, columns, conditions)

        template_id = f"tpl_{query_type}_{len(self.templates[query_type])}"

        return SQLTemplate(
            template_id=template_id,
            template_text=template_text,
            query_type=query_type,
            table_slots=tables,
            column_slots=columns,
            condition_slots=conditions,
            source_sql=sql
        )

    def _classify_query_type(self, question: str) -> str:
        """分类查询类型"""
        question_lower = question.lower()

        for qtype in ['statistical', 'list', 'ranking', 'aggregate']:
            for kw in self.query_type_keywords.get(qtype, []):
                if kw.lower() in question_lower:
                    return qtype

        return 'list'  # 默认类型

    def _extract_tables(self, sql: str) -> List[str]:
        """从 SQL 提取表名"""
        # 简单的 FROM 和 JOIN 提取
        tables = []

        # FROM table_name
        from_pattern = r'\bFROM\s+([a-zA-Z_][a-zA-Z0-9_]*)'
        tables.extend(re.findall(from_pattern, sql, re.IGNORECASE))

        # JOIN table_name
        join_pattern = r'\bJOIN\s+([a-zA-Z_][a-zA-Z0-9_]*)'
        tables.extend(re.findall(join_pattern, sql, re.IGNORECASE))

        return list(set(tables))

    def _extract_columns(self, sql: str) -> List[str]:
        """从 SQL 提取列名"""
        columns = []

        # SELECT column1, column2
        select_pattern = r'\bSELECT\s+(.*?)\s+FROM'
        match = re.search(select_pattern, sql, re.IGNORECASE | re.DOTALL)
        if match:
            select_part = match.group(1)
            # 提取列名（忽略 *)
            if '*' not in select_part:
                col_pattern = r'([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)?)'
                columns.extend(re.findall(col_pattern, select_part))

        # WHERE column = value
        where_pattern = r'\bWHERE\s+[a-zA-Z_][a-zA-Z0-9_.]*'
        columns.extend(re.findall(where_pattern, sql, re.IGNORECASE))

        return list(set(columns))

    def _extract_conditions(self, sql: str) -> List[str]:
        """从 SQL 提取 WHERE 条件"""
        conditions = []

        # 提取 WHERE 子句
        where_pattern = r'\bWHERE\s+(.*?)(?:\bGROUP BY\b|\bORDER BY\b|\bHAVING\b|\bLIMIT\b|$)'
        match = re.search(where_pattern, sql, re.IGNORECASE | re.DOTALL)
        if match:
            where_clause = match.group(1)
            # 分割多个条件
            cond_parts = re.split(r'\bAND\b|\bOR\b', where_clause, flags=re.IGNORECASE)
            conditions = [c.strip() for c in cond_parts if c.strip()]

        return conditions

    def _generalize_sql(
        self,
        sql: str,
        tables: List[str],
        columns: List[str],
        conditions: List[str]
    ) -> str:
        """将 SQL 泛化为模板"""
        template = sql

        # 替换表名为占位符
        for i, table in enumerate(tables):
            template = re.sub(
                rf'\b{table}\b',
                f'{{TABLE_{i}}}',
                template,
                flags=re.IGNORECASE
            )

        # 替换列名为占位符
        for i, column in enumerate(columns):
            if '.' in column:
                table, col = column.split('.', 1)
                template = re.sub(
                    rf'\b{table}\.{col}\b',
                    f'{{COLUMN_{i}}}',
                    template,
                    flags=re.IGNORECASE
                )
            else:
                template = re.sub(
                    rf'\b{column}\b',
                    f'{{COLUMN_{i}}}',
                    template,
                    flags=re.IGNORECASE
                )

        # 替换具体值为占位符
        template = re.sub(r"'[^']*'", "'{VALUE}'", template)
        template = re.sub(r'\b\d+\b', '{VALUE}', template)

        return template
    def _find_similar_template(self, template: SQLTemplate) -> Optional[SQLTemplate]:
        """查找相似的模板"""
        for existing in self.templates[template.query_type]:
            if self._template_similarity(template, existing) > 0.8:
                return existing
        return None

    def _template_similarity(self, t1: SQLTemplate, t2: SQLTemplate) -> float:
        """计算模板相似度"""
        # 简单的文本相似度
        words1 = set(t1.template_text.split())
        words2 = set(t2.template_text.split())

        if not words1 or not words2:
            return 0.0

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0
    def match_template(self, query: str, query_type: str = None) -> Optional[SQLTemplate]:
        """
        匹配最合适的模板

        Args:
            query: 查询文本
            query_type: 可选的查询类型

        Returns:
            最匹配的模板，如果没有则返回 None
        """
        if query_type is None:
            query_type = self._classify_query_type(query)

        candidates = self.templates.get(query_type, [])
        if not candidates:
            return None

        # 选择频率最高的模板
        return max(candidates, key=lambda t: t.frequency)

    def fill_template(
        self,
        template: SQLTemplate,
        tables: List[str],
        columns: List[Tuple[str, str]],
        conditions: Optional[Dict[str, str]] = None
    ) -> str:
        """
        巳充模板生成 SQL

        Args:
            template: SQL 模板
            tables: 表名列表
            columns: 列名列表 [(table, column), ...]
            conditions: 条件字典 {column: value}

        Returns:
            生成的 SQL
        """
        sql = template.template_text

        # 巷充表名
        for i, table in enumerate(tables):
            sql = sql.replace(f'{{TABLE_{i}}}', table)

        # 混充列名
        for i, (table, column) in enumerate(columns):
            if i < len(template.column_slots):
                sql = sql.replace(f'{{COLUMN_{i}}}', f'{table}.{column}')

        # 混充条件值
        if conditions:
            for col, val in conditions.items():
                sql = sql.replace('{VALUE}', f"'{val}'", 1)

        return sql
    def generate(
        self,
        query: str,
        retrieved_tables: List[str],
        retrieved_columns: List[Tuple[str, str]],
        join_candidates: Optional[List[Tuple[str, str]]] = None
    ) -> GeneratedSQL:
        """
        生成 SQL

        Args:
            query: 查询文本
            retrieved_tables: 检索到的表名列表
            retrieved_columns: 检索到的列名列表 [(table, column), ...]
            join_candidates: 可选的 JOIN 候选

        Returns:
            生成的 SQL
        """
        # 1. 确定查询类型
        query_type = self._classify_query_type(query)

        # 2. 匹配模板
        template = self.match_template(query, query_type)

        # 3. 生成 SQL
        if template:
            # 使用模板
            sql = self.fill_template(template, retrieved_tables, retrieved_columns)
            confidence = 0.8
            template_id = template.template_id
        else:
            # 回退到简单生成
            sql = self._generate_simple_sql(query, retrieved_tables, retrieved_columns)
            confidence = 0.5
            template_id = None

        # 4. 推断 JOIN
        joins = []
        if len(retrieved_tables) > 1:
            joins = self.join_inferencer.infer_joins(retrieved_tables)
            if joins:
                sql = self._add_joins(sql, joins)

        return GeneratedSQL(
            sql=sql,
            template_id=template_id,
            confidence=confidence,
            tables=retrieved_tables,
            columns=[f"{t}.{c}" for t, c in retrieved_columns],
            join_conditions=joins
        )

    def _generate_simple_sql(
        self,
        query: str,
        tables: List[str],
        columns: List[Tuple[str, str]]
    ) -> str:
        """生成简单 SQL（无模板时回退)"""
        if not tables:
            return "SELECT 1"  # 占位符

        main_table = tables[0]

        # 收集该表的列
        table_columns = [col for t, col in columns if t == main_table]
        if not table_columns:
            select_clause = "SELECT *"
        else:
            select_clause = f"SELECT {', '.join(table_columns[:5])}"

        sql = f"{select_clause} FROM {main_table}"

        # 添加 JOIN
        if len(tables) > 1:
            joins = self.join_inferencer.infer_joins(tables)
            for t1, t2, cond in joins:
                sql += f" JOIN {t2} ON {cond}"

        return sql
    def _add_joins(self, sql: str, joins: List[Tuple[str, str, str]]) -> str:
        """向 SQL 添加 JOIN 子句"""
        for t1, t2, cond in joins:
            if t2.lower() not in sql.lower():
                sql += f" JOIN {t2} ON {cond}"
        return sql
    def save_templates(self, output_path: str):
        """保存模板到文件"""
        templates_data = {}
        for qtype, template_list in self.templates.items():
            templates_data[qtype] = [
                {
                    'template_id': t.template_id,
                    'template_text': t.template_text,
                    'query_type': t.query_type,
                    'table_slots': t.table_slots,
                    'column_slots': t.column_slots,
                    'condition_slots': t.condition_slots,
                    'source_sql': t.source_sql,
                    'frequency': t.frequency
                }
                for t in template_list
            ]

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(templates_data, f, ensure_ascii=False, indent=2)

        print(f"SQL 模板已保存到: {output_path}")

    def load_templates(self, input_path: str):
        """从文件加载模板"""
        with open(input_path, 'r', encoding='utf-8') as f:
            templates_data = json.load(f)

        for qtype, template_list in templates_data.items():
            for t_data in template_list:
                template = SQLTemplate(
                    template_id=t_data['template_id'],
                    template_text=t_data['template_text'],
                    query_type=t_data['query_type'],
                    table_slots=t_data['table_slots'],
                    column_slots=t_data['column_slots'],
                    condition_slots=t_data['condition_slots'],
                    source_sql=t_data['source_sql'],
                    frequency=t_data['frequency']
                )
                self.templates[qtype].append(template)

        print(f"从 {input_path} 加载了 {sum(len(v) for v in self.templates.values())} 个 SQL 模板")


class SQLEvaluator:
    """
    SQL 评估器

    计算 SQL-EM (Exact Match) 和 ExecAcc (Execution Accuracy)
    """

    def __init__(self, config: TERAGConfig):
        self.config = config
    def compute_sql_em(
        self,
        generated_sql: str,
        ground_truth_sql: str
    ) -> float:
        """
        计算 SQL-EM (精确匹配)

        Args:
            generated_sql: 生成的 SQL
            ground_truth_sql: 真实 SQL

        Returns:
            1.0 表示完全匹配，0.0 表示不匹配
        """
        # 标准化 SQL
        gen_normalized = self._normalize_sql(generated_sql)
        gt_normalized = self._normalize_sql(ground_truth_sql)
        # 完全匹配
        if gen_normalized == gt_normalized:
            return 1.0
        # 部分匹配(检查关键组件)
        gen_components = self._extract_sql_components(gen_normalized)
        gt_components = self._extract_sql_components(gt_normalized)
        # 计算组件匹配率
        scores = []
        for key in ['tables', 'columns', 'conditions']:
            if key in gen_components and key in gt_components:
                gen_set = gen_components[key]
                gt_set = gt_components[key]
                if gt_set:
                    overlap = len(gen_set & gt_set)
                    total = len(gt_set)
                    scores.append(overlap / total if total > 0 else 1.0)
        return sum(scores) / len(scores) if scores else 0.0
    def _normalize_sql(self, sql: str) -> str:
        """标准化 SQL"""
        # 转小写
        sql = sql.lower()
        # 移除多余空格
        sql = ' '.join(sql.split())
        # 移除括号周围的空格
        sql = re.sub(r'\s*\(\s*', ' ( ', sql)
        sql = re.sub(r'\s*\)\s*', ' ) ', sql)
        # 标准化操作符
        for op in ['=', '!=', '<=', '>=', '<', '>']:
            sql = re.sub(rf'\s*{re.escape(op)}\s*', f' {op} ', sql)
        return sql.strip()
    def _extract_sql_components(self, sql: str) -> Dict[str, set]:
        """提取 SQL 组件"""
        components = {}
        # 提取表名
        tables = set()
        from_pattern = r'\bfrom\s+([a-zA-Z_][a-zA-Z0-9_]*)'
        tables.update(re.findall(from_pattern, sql))
        join_pattern = r'\bjoin\s+([a-zA-Z_][a-zA-Z0-9_]*)'
        tables.update(re.findall(join_pattern, sql))
        components['tables'] = tables
        # 提取列名
        columns = set()
        col_pattern = r'(?:select|where|order by|group by)\s+([a-zA-Z_][a-zA-Z0-9_.]+)'
        for match in re.findall(col_pattern, sql):
            if '.' in match:
                columns.add(match.split('.')[-1])
            else:
                columns.add(match)
        components['columns'] = columns
        # 提取条件
        conditions = set()
        where_pattern = r'\bwhere\s+(.*?)(?:group by|order by|limit|$)'
        match = re.search(where_pattern, sql, re.DOTALL)
        if match:
            where_clause = match.group(1)
            cond_pattern = r'([a-zA-Z_][a-zA-Z0-9_.]*)\s*(?:=|!=|<|>|<=|>=)\s*[\'"]?[^\'"]*[\'"]?'
            conditions.update(re.findall(cond_pattern, where_clause))
        components['conditions'] = conditions
        return components


def main():
    """演示 SQL 生成器"""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from terag.config import TERAGConfig
    # 加载配置
    config_path = Path(__file__).parent.parent / 'config.yaml'
    config = TERAGConfig.from_yaml(str(config_path))

    print("=" * 60)
    print("SQL 生成器演示")
    print("=" * 60)
    # 创建生成器
    generator = TemplateSQLGenerator(config)
    # 加载训练数据
    train_data = []
    train_path = config.get_split_path('train')
    if Path(train_path).exists():
        with open(train_path, 'r', encoding='utf-8') as f:
            for line in f:
                train_data.append(json.loads(line))
        train_df = pd.DataFrame(train_data)
        # 提取模板
        print("\n提取 SQL 模板...")
        generator.extract_templates(train_df)
    # 测试生成
    print("\n测试 SQL 生成:")
    test_queries = [
        "查询公司的售电量",
        "统计用户的电费金额",
        "分析供电所的回收率排名",
    ]
    for query in test_queries:
        print(f"\n查询: {query}")
        # 模拟检索结果
        tables = ["company_info", "sales_data"]
        columns = [("company_info", "company_id"), ("sales_data", "sales_amount")]
        result = generator.generate(query, tables, columns)
        print(f"  SQL: {result.sql}")
        print(f"  置信度: {result.confidence:.2f}")
        print(f"  模板: {result.template_id}")


if __name__ == "__main__":
    main()
