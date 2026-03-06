"""
SQL 评估模块

提供专业的 SQL 评估指标：
1. 检索层指标：Table@k, Column@k
2. SQL 层指标：SQL-EM (Exact Match), ExecAcc (Execution Accuracy)
3. SQL Parse Rate, SQL AST Equivalence

使用方式:
    from eval.sql_eval import SQLEvaluator, SQLMetrics

    evaluator = SQLEvaluator(config)
    metrics = evaluator.evaluate_sql(predictions, ground_truths)
"""

import os
import re
import json
import pandas as pd
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict

try:
    import duckdb
    DUCKDB_AVAILABLE = True
except ImportError:
    DUCKDB_AVAILABLE = False
    print("Warning: duckdb not available, ExecAcc will be skipped")

try:
    import sqlglot
    SQLGLOT_AVAILABLE = True
except ImportError:
    SQLGLOT_AVAILABLE = False


@dataclass
class RetrievalMetrics:
    """检索层指标"""
    # 表级指标
    table_at_1: float = 0.0
    table_at_3: float = 0.0
    table_at_5: float = 0.0
    table_at_10: float = 0.0

    # 字段级指标
    column_at_5: float = 0.0
    column_at_10: float = 0.0
    column_at_20: float = 0.0

    # 综合指标
    field_coverage: float = 0.0

    # 统计
    total_queries: int = 0

    def to_dict(self) -> Dict[str, float]:
        return {
            'Table@1': self.table_at_1,
            'Table@3': self.table_at_3,
            'Table@5': self.table_at_5,
            'Table@10': self.table_at_10,
            'Column@5': self.column_at_5,
            'Column@10': self.column_at_10,
            'Column@20': self.column_at_20,
            'Field_Coverage': self.field_coverage,
            'Total_Queries': self.total_queries,
        }


@dataclass
class SQLMetrics:
    """SQL 层指标"""
    # Exact Match
    sql_em: float = 0.0

    # Execution Accuracy
    exec_acc: float = 0.0

    # SQL Parse Rate (可解析率)
    sql_parse_rate: float = 0.0

    # SQL AST Equivalence (AST结构等价率)
    sql_ast_equiv: float = 0.0

    # 统计
    total_queries: int = 0
    valid_sql_count: int = 0

    def to_dict(self) -> Dict[str, float]:
        return {
            'SQL_EM': self.sql_em,
            'SQL_Parse_Rate': self.sql_parse_rate,
            'SQL_AST_Equiv': self.sql_ast_equiv,
            'ExecAcc': self.exec_acc,
            'Total_Queries': self.total_queries,
            'Valid_SQL': self.valid_sql_count,
        }


class SQLEvaluator:
    """
    SQL 评估器

    提供多层次的评估指标：
    1. 检索层：Table@k, Column@k
    2. SQL 层：SQL-EM, Parse Rate, AST Equiv,    3. SQL 执行层：ExecAcc
    """

    def __init__(self, config=None):
        """
        初始化评估器

        Args:
            config: TE-RAG 配置（可选）
        """
        self.config = config

    def normalize_sql(self, sql: str) -> str:
        """
        标准化 SQL

        Args:
            sql: SQL 语句

        Returns:
            标准化后的 SQL
        """
        # 转小写
        sql = sql.lower()

        # 移除多余空格
        sql = re.sub(r'\s+', ' ', sql)

        # 移除别名
        sql = re.sub(r'\s+as\s+\w+', '', sql)

        # 标准化引号
        sql = sql.replace('"', "'")

        # 移除分号
        sql = sql.rstrip(';')

        return sql.strip()

    def sql_parseable(self, sql: str) -> bool:
        """
        检查 SQL 是否可解析

        Args:
            sql: SQL 语句

        Returns:
            是否可解析
        """
        if not SQLGLOT_AVAILABLE:
            return False

        try:
            import sqlglot
            sqlglot.parse_one(sql)
            return True
        except Exception:
            return False

    def sql_ast_equivalent(self, sql1: str, sql2: str) -> bool:
        """
        检查两个 SQL 的 AST 结构是否等价

        Args:
            sql1: SQL 语句 1
            sql2: SQL 语句 2

        Returns:
            是否 AST 等价
        """
        if not SQLGLOT_AVAILABLE:
            return False

        try:
            import sqlglot

            # 解析两个 SQL
            ast1 = sqlglot.parse_one(sql1)
            ast2 = sqlglot.parse_one(sql2)

            # 标准化 AST 进行比较
            ast1_normalized = self._normalize_ast(ast1)
            ast2_normalized = self._normalize_ast(ast2)

            return ast1_normalized == ast2_normalized
        except Exception:
            return False

    def _normalize_ast(self, ast) -> str:
        """
        标准化 AST 结构用于比较

        Args:
            ast: sqlglot AST

        Returns:
            标准化后的字符串表示
        """
        import sqlglot

        # 转换为字符串并标准化
        ast_str = ast.sql(pretty=False)

        # 移除别名
        ast_str = re.sub(r'\s+AS\s+\w+', '', ast_str)

        # 移除多余空格
        ast_str = re.sub(r'\s+', ' ', ast_str)

        return ast_str.lower().strip()

    def evaluate_sql(
        self,
        predictions: List[str],
        ground_truths: List[str],
        db_schema: Dict = None,
        execute: bool = False
    ) -> SQLMetrics:
        """
        评估 SQL 生成

        Args:
            predictions: 预测的 SQL 列表
            ground_truths: 真实的 SQL 列表
            db_schema: 数据库 schema（用于执行）
            execute: 是否执行 SQL 计算 ExecAcc

        Returns:
            SQLMetrics
        """
        metrics = SQLMetrics(
            total_queries=len(predictions),
            valid_sql_count=0
        )

        em_count = 0
        exec_count = 0
        parse_ok_count = 0
        ast_equiv_count = 0
        valid_count = 0

        for pred_sql, gt_sql in zip(predictions, ground_truths):
            if pd.isna(pred_sql) or pd.isna(gt_sql):
                continue

            pred_sql = str(pred_sql).strip()
            gt_sql = str(gt_sql).strip()

            if not pred_sql or not gt_sql:
                continue

            valid_count += 1

            # 1. SQL Parse Rate
            if self.sql_parseable(pred_sql):
                parse_ok_count += 1

            # 2. Exact Match
            if self.normalize_sql(pred_sql) == self.normalize_sql(gt_sql):
                em_count += 1

            # 3. AST Equivalence
            if self.sql_ast_equivalent(pred_sql, gt_sql):
                ast_equiv_count += 1

            # 4. Execution Accuracy (可选)
            if execute and DUCKDB_AVAILABLE and db_schema:
                try:
                    conn = duckdb.connect()

                    # 创建表
                    for table_name, columns in db_schema.items():
                        col_defs = ', '.join([f"{col} TEXT" for col in columns])
                        conn.execute(f"CREATE TABLE {table_name} ({col_defs})")

                    # 执行 SQL
                    pred_result = conn.execute(pred_sql).fetchall()
                    gt_result = conn.execute(gt_sql).fetchall()

                    if pred_result == gt_result:
                        exec_count += 1

                    conn.close()
                except Exception:
                    pass

        metrics.valid_sql_count = valid_count
        metrics.sql_em = em_count / valid_count if valid_count > 0 else 0
        metrics.sql_parse_rate = parse_ok_count / valid_count if valid_count > 0 else 0
        metrics.sql_ast_equiv = ast_equiv_count / valid_count if valid_count > 0 else 0
        metrics.exec_acc = exec_count / valid_count if valid_count > 0 else 0

        return metrics

    def evaluate_retrieval(
        self,
        retriever,
        test_data: pd.DataFrame,
        k_values: List[int] = None
    ) -> RetrievalMetrics:
        """
        评估检索性能

        Args:
            retriever: 检索器（需要有 retrieve 方法）
            test_data: 测试数据
            k_values: k 值列表

        Returns:
            RetrievalMetrics
        """
        if k_values is None:
            k_values = [1, 3, 5, 10]

        metrics = RetrievalMetrics(total_queries=len(test_data))

        table_correct = defaultdict(int)
        column_correct = defaultdict(int)
        field_coverage_count = 0

        for _, row in test_data.iterrows():
            query = row['question']
            gt_table = row.get('table', '')
            gt_fields = row.get('field', '')

            # 处理真实标签
            gt_table_simple = gt_table.split('.')[-1] if pd.notna(gt_table) else ''

            gt_field_set = set()
            if pd.notna(gt_fields) and isinstance(gt_fields, str):
                gt_field_set = set(f.strip() for f in gt_fields.split('|') if f.strip())

            # 检索
            try:
                results = retriever.retrieve(query, k=max(k_values))
            except Exception as e:
                print(f"检索失败: {query}, 错误: {e}")
                continue

            # 评估表选择
            retrieved_tables = [r.table for r in results]

            for k in k_values:
                if gt_table_simple in retrieved_tables[:k]:
                    table_correct[k] += 1

            # 评估字段选择
            retrieved_fields = set()
            for r in results:
                if r.table == gt_table_simple:
                    for col, _ in r.columns:
                        field_name = col.split('.')[-1]
                        retrieved_fields.add(field_name)

            # 字段覆盖率
            if gt_field_set and gt_field_set.issubset(retrieved_fields):
                field_coverage_count += 1

            # Column@k
            if gt_field_set:
                for k in [5, 10, 20]:
                    retrieved_k = set()
                    count = 0
                    for r in results:
                        if r.table == gt_table_simple:
                            for col, _ in r.columns:
                                if count >= k:
                                    break
                                field_name = col.split('.')[-1]
                                retrieved_k.add(field_name)
                                count += 1
                            if count >= k:
                                break

                    if gt_field_set.issubset(retrieved_k):
                        column_correct[k] += 1

        # 计算指标
        total = len(test_data)

        for k in k_values:
            if k == 1:
                metrics.table_at_1 = table_correct[k] / total if total > 0 else 0
            elif k == 3:
                metrics.table_at_3 = table_correct[k] / total if total > 0 else 0
            elif k == 5:
                metrics.table_at_5 = table_correct[k] / total if total > 0 else 0
            elif k == 10:
                metrics.table_at_10 = table_correct[k] / total if total > 0 else 0

        metrics.column_at_5 = column_correct[5] / total if total > 0 else 0
        metrics.column_at_10 = column_correct[10] / total if total > 0 else 0
        metrics.column_at_20 = column_correct[20] / total if total > 0 else 0

        metrics.field_coverage = field_coverage_count / total if total > 0 else 0

        return metrics


def compare_retrievers(
    retrievers: Dict[str, Any],
    test_data: pd.DataFrame,
    output_path: str = None
) -> pd.DataFrame:
    """
    对比多个检索器

    Args:
        retrievers: {name: retriever} 字典
        test_data: 测试数据
        output_path: 输出路径

    Returns:
        对比结果 DataFrame
    """
    evaluator = SQLEvaluator()
    results = []

    for name, retriever in retrievers.items():
        print(f"评估: {name}")
        metrics = evaluator.evaluate_retrieval(retriever, test_data)

        result = {'Method': name}
        result.update(metrics.to_dict())
        results.append(result)
    df = pd.DataFrame(results)
    if output_path:
        df.to_csv(output_path, index=False)
        print(f"结果已保存到: {output_path}")
    return df


def main():
    """演示评估"""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from terag.config import TERAGConfig
    from terag.terag_retriever_v2 import TERAGRetrieverV2

    config_path = Path(__file__).parent.parent / 'config.yaml'
    config = TERAGConfig.from_yaml(str(config_path))
    print("=" * 60)
    print("SQL 评估演示")
    print("=" * 60)
    test_data = []
    test_path = config.get_split_path('test')
    if os.path.exists(test_path):
        with open(test_path, 'r', encoding='utf-8') as f:
            for line in f:
                test_data.append(json.loads(line))
        test_df = pd.DataFrame(test_data)
        print(f"测试数据: {len(test_df)} 条")
        retriever = TERAGRetrieverV2.from_artifacts(config)
        evaluator = SQLEvaluator(config)
        metrics = evaluator.evaluate_retrieval(retriever, test_df)
        print("\n检索层指标:")
        for key, value in metrics.to_dict().items():
            print(f"  {key}: {value:.4f}")
    else:
        print(f"测试数据不存在: {test_path}")
        print("请先运行: python scripts/prepare_data.py")


if __name__ == "__main__":
    main()
