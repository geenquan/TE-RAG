"""
二分图构建器

构建查询-表-列二分图，边权重基于 SQL 角色确定
"""

import os
import pickle
import json
import pandas as pd
import networkx as nx
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Optional

from terag.config import TERAGConfig
from terag.sql_role_parser import SQLRoleParser, ParseResult
from datasets.gt_utils import parse_ground_truth_tables, parse_ground_truth_fields


class BipartiteGraphBuilder:
    """
    二分图构建器

    构建查询-表-列二分图：
    - 左侧节点：查询 (Q_0, Q_1, ...)
    - 右侧节点：表 (T:table_name) 和列 (C:table.field)

    边权重由 SQL 角色决定：
    - WHERE: 2.0
    - JOIN: 1.8
    - GROUP_BY: 1.5
    - ORDER_BY: 1.3
    - SELECT: 1.2
    - FROM: 1.0

    使用方式:
        builder = BipartiteGraphBuilder(config)
        graph = builder.build(train_data)
        builder.save(graph, "artifacts/graph.pkl")
    """

    def __init__(self, config: TERAGConfig):
        """
        初始化构建器

        Args:
            config: TE-RAG 配置
        """
        self.config = config
        self.parser = SQLRoleParser(
            default_role_weights=config.graph.role_weights,
            dialect=config.sql_dialect,
            mode=getattr(config.graph, 'parse_mode', 'ast'),
        )

        # 加载表和字段数据
        self.field_df = pd.read_csv(config.data.field_csv)
        self.table_df = pd.read_csv(config.data.table_csv)

        # 简单表名 -> 物理 schema 列集合（SQL 解析消歧用，兼容多表/别名 SQL）
        self.schema_columns: Dict[str, Set[str]] = defaultdict(set)
        for _, row in self.field_df.iterrows():
            simple = str(row['table']).split('.')[-1]
            self.schema_columns[simple.lower()].add(str(row['field_name']))

        # 角色统计
        self.role_stats = defaultdict(lambda: defaultdict(int))

    def build(self, train_data: pd.DataFrame) -> nx.Graph:
        """
        构建二分图

        Args:
            train_data: 训练数据，包含 question, table, field, sql 列

        Returns:
            NetworkX Graph 对象
        """
        graph = nx.Graph()

        # 添加查询节点
        for idx in range(len(train_data)):
            graph.add_node(f"Q_{idx}", bipartite=0, node_type='query')

        # 添加表节点
        for _, row in self.table_df.iterrows():
            table_node = f"T:{row['table']}"
            graph.add_node(
                table_node,
                bipartite=1,
                node_type='table',
                description=row.get('table_desc', '')
            )

        # 添加列节点
        for _, row in self.field_df.iterrows():
            col_node = f"C:{row['table']}.{str(row['field_name']).lower()}"
            graph.add_node(
                col_node,
                bipartite=1,
                node_type='column',
                description=row.get('field_name_desc', '')
            )

        # 添加边
        use_role_parser = self.config.ablation.use_role_parser

        for pos in range(len(train_data)):
            row = train_data.iloc[pos]
            query_node = f"Q_{pos}"
            sql = row.get('sql', '')

            # 统一解析 ground truth（PowerGrid 单表 legacy / BIRD 多表结构化 gold 列）
            gold_tables = parse_ground_truth_tables(row)          # canonical 表 id
            gold_fields = parse_ground_truth_fields(row)          # [(canonical 表 id, 字段名)]
            simple_to_canonical = {t.split('.')[-1].lower(): t for t in gold_tables}

            # 解析 SQL 角色（多表场景不传 expected_table，依赖别名/schema 消歧）
            if use_role_parser and pd.notna(sql):
                expected = gold_tables[0].split('.')[-1] if len(gold_tables) == 1 else None
                parse_result = self.parser.parse(
                    str(sql), expected, schema_columns=dict(self.schema_columns))
            else:
                parse_result = None

            # 添加列边
            for table_id, field in gold_fields:
                col_node = f"C:{table_id}.{str(field).lower()}"

                if graph.has_node(col_node):
                    # 计算权重
                    if parse_result and parse_result.success:
                        table_simple = table_id.split('.')[-1].lower()
                        weight = parse_result.get_field_weight(
                            table_simple, field,
                            self.config.graph.role_weights
                        )
                        # One physical column can have several SQL case spellings.
                        # Merge their role evidence by maximum, never by summation.
                        for spelling, role_info in parse_result.field_roles.get(table_simple, {}).items():
                            if str(spelling).lower() == str(field).lower():
                                weight = max([weight] + [self.config.graph.role_weights.get(role, 1.0) for role in role_info.roles])
                        # 消融：角色权重统一（保留 AST 解析和边集，仅权重→1.0）
                        if hasattr(self.config.ablation, 'use_role_weight') \
                                and not self.config.ablation.use_role_weight:
                            weight = 1.0
                        # 记录角色统计
                        info = parse_result.find_field_info(table_simple, field)
                        if info is not None:
                            for role in info.roles:
                                self.role_stats[table_id][role] += 1
                    else:
                        # 回退到字符串匹配
                        weight = self._get_weight_by_string_match(str(sql) if pd.notna(sql) else '')
                        # 消融：角色权重统一（回退路径同样不恢复非统一权重）
                        if hasattr(self.config.ablation, 'use_role_weight') \
                                and not self.config.ablation.use_role_weight:
                            weight = 1.0

                    previous = graph.get_edge_data(query_node, col_node, default={}).get('weight', 0.0)
                    graph.add_edge(query_node, col_node, weight=max(previous, weight))

            # 添加表边（多表查询为每张 gold 表都建立边）
            for table_id in gold_tables:
                table_node = f"T:{table_id}"
                if graph.has_node(table_node):
                    table_weight = self.config.graph.role_weights.get('FROM', 1.0)
                    graph.add_edge(query_node, table_node, weight=table_weight)

        return graph

    def _get_weight_by_string_match(self, sql: str) -> float:
        """通过字符串匹配获取权重（回退方案）"""
        weight = 1.0
        sql_upper = sql.upper()

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

        return weight

    def save(self, graph: nx.Graph, output_path: str):
        """保存图"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'wb') as f:
            pickle.dump(graph, f)
        print(f"图已保存到: {output_path}")

    def load(self, input_path: str) -> nx.Graph:
        """加载图"""
        with open(input_path, 'rb') as f:
            return pickle.load(f)

    def save_role_stats(self, output_path: str):
        """保存角色统计"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(dict(self.role_stats), f, ensure_ascii=False, indent=2)
        print(f"角色统计已保存到: {output_path}")

    def get_element_to_queries(self, graph: nx.Graph, train_data: pd.DataFrame) -> Dict[str, Set[str]]:
        """
        获取元素到查询的映射

        Args:
            graph: 二分图
            train_data: 训练数据

        Returns:
            {element_node: set of questions}
        """
        element_to_queries = defaultdict(set)

        for pos in range(len(train_data)):
            row = train_data.iloc[pos]
            query_node = f"Q_{pos}"
            question = row['question']

            # 获取所有邻居节点
            if graph.has_node(query_node):
                for neighbor in graph.neighbors(query_node):
                    element_to_queries[neighbor].add(question)

        return element_to_queries

    def get_graph_stats(self, graph: nx.Graph) -> Dict:
        """获取图统计信息"""
        query_nodes = [n for n in graph.nodes if n.startswith('Q_')]
        table_nodes = [n for n in graph.nodes if n.startswith('T:')]
        column_nodes = [n for n in graph.nodes if n.startswith('C:')]

        return {
            'total_nodes': graph.number_of_nodes(),
            'query_nodes': len(query_nodes),
            'table_nodes': len(table_nodes),
            'column_nodes': len(column_nodes),
            'total_edges': graph.number_of_edges(),
            'density': nx.density(graph),
        }


def main():
    """演示二分图构建"""
    import sys
    from pathlib import Path

    # 添加项目根目录
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from terag.config import TERAGConfig

    # 加载配置
    config_path = Path(__file__).parent.parent / 'config.yaml'
    config = TERAGConfig.from_yaml(str(config_path))

    # 加载训练数据
    import json
    train_data = []
    with open(config.get_split_path('train'), 'r', encoding='utf-8') as f:
        for line in f:
            train_data.append(json.loads(line))
    train_df = pd.DataFrame(train_data)

    print(f"加载训练数据: {len(train_df)} 条")

    # 构建图
    builder = BipartiteGraphBuilder(config)
    graph = builder.build(train_df)

    # 打印统计
    stats = builder.get_graph_stats(graph)
    print(f"\n图统计:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # 保存
    if config.output.save_graph:
        builder.save(graph, config.get_artifact_path('graph.pkl'))

    if config.output.save_role_stats:
        builder.save_role_stats(config.get_artifact_path('role_stats.json'))


if __name__ == "__main__":
    main()
