#!/usr/bin/env python
"""
实验运行脚本（完整版）

包含：
1. 对比实验：TE-RAG-V2, BM25, Vector, LLM
2. 消融实验
3. 冷启动实验
4. 资源消耗对比（时间、内存）

使用方式:
    python scripts/run_experiments.py --test splits/test.jsonl --suite all
"""

import os
import sys
import json
import time
import tracemalloc
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).parent.parent))

from terag.config import TERAGConfig
from terag.terag_retriever_v2 import TERAGRetrieverV2, RetrievalResultV2
from terag.sql_generator import TemplateSQLGenerator
from eval.sql_eval import SQLEvaluator


def load_jsonl(path: str) -> pd.DataFrame:
    """加载 JSONL 文件"""
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return pd.DataFrame(data)


@dataclass
class MetricsResult:
    """评估结果"""
    table_at_1: float = 0.0
    table_at_3: float = 0.0
    table_at_5: float = 0.0
    table_at_10: float = 0.0
    column_at_5: float = 0.0
    column_at_10: float = 0.0
    column_at_20: float = 0.0
    field_coverage: float = 0.0
    # SQL 层指标
    sql_em: float = 0.0
    sql_parse_rate: float = 0.0
    sql_ast_equiv: float = 0.0
    exec_acc: float = 0.0
    # 资源消耗
    avg_query_time: float = 0.0
    avg_memory_mb: float = 0.0
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
            'SQL_EM': self.sql_em,
            'SQL_Parse_Rate': self.sql_parse_rate,
            'SQL_AST_Equiv': self.sql_ast_equiv,
            'ExecAcc': self.exec_acc,
            'Avg_Query_Time(s)': self.avg_query_time,
            'Avg_Memory(MB)': self.avg_memory_mb,
            'Total_Queries': self.total_queries,
        }


class UnifiedEvaluator:
    """统一评估器"""

    def __init__(self, config: TERAGConfig, enable_sql_eval: bool = True):
        self.config = config
        self.enable_sql_eval = enable_sql_eval

        # 初始化 SQL 生成器和评估器
        if enable_sql_eval:
            self.sql_generator = TemplateSQLGenerator(config)
            self.sql_evaluator = SQLEvaluator(config)
            # 尝试加载训练好的 SQL 模板
            template_path = config.get_artifact_path('sql_templates.json')
            if os.path.exists(template_path):
                self.sql_generator.load_templates(template_path)
        else:
            self.sql_generator = None
            self.sql_evaluator = None

    def train_sql_generator(self, train_data: pd.DataFrame):
        """训练 SQL 生成器（从训练数据提取模板）"""
        if self.sql_generator:
            self.sql_generator.extract_templates(train_data)
            # 保存模板
            template_path = self.config.get_artifact_path('sql_templates.json')
            self.sql_generator.save_templates(template_path)

    def evaluate_retriever(
        self,
        retriever,
        test_data: pd.DataFrame,
        k_values: List[int] = None
    ) -> MetricsResult:
        """
        评估检索器（包含时间和内存）
        """
        if k_values is None:
            k_values = [1, 3, 5, 10]

        metrics = MetricsResult(total_queries=len(test_data))

        table_correct = {k: 0 for k in k_values}
        column_correct = {5: 0, 10: 0, 20: 0}
        field_coverage_count = 0

        query_times = []
        memory_usages = []

        # SQL 评估相关
        pred_sqls = []
        gt_sqls = []

        for _, row in test_data.iterrows():
            query = row['question']
            gt_table = row.get('table', '')
            gt_fields = row.get('field', '')
            gt_sql = row.get('sql', '')

            # 处理真实标签
            gt_table_simple = gt_table.split('.')[-1] if pd.notna(gt_table) else ''

            gt_field_set = set()
            if pd.notna(gt_fields) and isinstance(gt_fields, str):
                gt_field_set = set(f.strip() for f in gt_fields.split('|') if f.strip())

            # 检索并测量资源消耗
            tracemalloc.start()
            start_time = time.time()

            try:
                results = retriever.retrieve(query, k=max(k_values))
            except Exception as e:
                print(f"检索失败: {query}, 错误: {e}")
                results = []

            query_time = time.time() - start_time
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            query_times.append(query_time)
            memory_usages.append(peak / 1024 / 1024)  # MB

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

            # SQL 生成和评估
            if self.enable_sql_eval and self.sql_generator and pd.notna(gt_sql) and gt_sql:
                try:
                    # 收集检索到的表和字段
                    retrieved_columns = []
                    for r in results[:5]:  # 只取前5个结果
                        for col, score in r.columns[:10]:  # 每个表取前10个字段
                            # col 格式为 "C:table.field" 或 "table.field"
                            if col.startswith('C:'):
                                col = col[2:]
                            if '.' in col:
                                t, c = col.split('.', 1)
                                retrieved_columns.append((t, c))
                            else:
                                retrieved_columns.append((r.table, col))

                    # 生成 SQL
                    generated = self.sql_generator.generate(
                        query,
                        retrieved_tables[:5],
                        retrieved_columns
                    )
                    pred_sqls.append(generated.sql)
                    gt_sqls.append(str(gt_sql))
                except Exception as e:
                    # SQL 生成失败，跳过
                    pass

        # 计算指标
        total = len(test_data)

        metrics.table_at_1 = table_correct[1] / total if total > 0 else 0
        metrics.table_at_3 = table_correct[3] / total if total > 0 else 0
        metrics.table_at_5 = table_correct[5] / total if total > 0 else 0
        metrics.table_at_10 = table_correct[10] / total if total > 0 else 0

        metrics.column_at_5 = column_correct[5] / total if total > 0 else 0
        metrics.column_at_10 = column_correct[10] / total if total > 0 else 0
        metrics.column_at_20 = column_correct[20] / total if total > 0 else 0

        metrics.field_coverage = field_coverage_count / total if total > 0 else 0

        metrics.avg_query_time = np.mean(query_times) if query_times else 0
        metrics.avg_memory_mb = np.mean(memory_usages) if memory_usages else 0

        # 计算 SQL 指标
        if pred_sqls and gt_sqls and self.sql_evaluator:
            sql_metrics = self.sql_evaluator.evaluate_sql(pred_sqls, gt_sqls, execute=False)
            metrics.sql_em = sql_metrics.sql_em
            metrics.sql_parse_rate = sql_metrics.sql_parse_rate
            metrics.sql_ast_equiv = sql_metrics.sql_ast_equiv
            metrics.exec_acc = sql_metrics.exec_acc

        return metrics


def create_retriever_wrapper(retriever, method_name: str):
    """创建检索器包装器，统一接口"""
    class RetrieverWrapper:
        def __init__(self, inner_retriever, name):
            self.inner = inner_retriever
            self.name = name

        def retrieve(self, query, k=5):
            results = self.inner.retrieve(query, k)
            # 转换为统一格式
            from terag.terag_retriever_v2 import RetrievalResultV2
            converted = []
            for r in results:
                converted.append(RetrievalResultV2(
                    table=r.table,
                    table_score=r.table_score,
                    columns=r.columns,
                    metadata=r.metadata
                ))
            return converted

    return RetrieverWrapper(retriever, method_name)


def create_bm25_wrapper(config: TERAGConfig):
    """创建 BM25 包装器"""
    from terag.index_builder import IndexBuilder, BM25Retriever

    index_builder = IndexBuilder(config)
    index = index_builder.load(config.get_artifact_path('bm25_index'))
    bm25_retriever = BM25Retriever(index)
    field_df = pd.read_csv(config.data.field_csv)

    class BM25Wrapper:
        def __init__(self, retriever, field_df):
            self.retriever = retriever
            self.field_df = field_df

        def retrieve(self, query, k=5):
            from terag.terag_retriever_v2 import RetrievalResultV2
            results = self.retriever.search(query, k)

            output = []
            for table_node, score in results:
                table_name = table_node.replace('T:', '')
                columns = self.field_df[self.field_df['table'] == table_name]

                col_list = [
                    (f"C:{table_name}.{col['field_name']}", 1.0)
                    for _, col in columns.head(10).iterrows()
                ]

                output.append(RetrievalResultV2(
                    table=table_name,
                    table_score=score,
                    columns=col_list,
                    metadata={'method': 'BM25'}
                ))

            return output

    return BM25Wrapper(bm25_retriever, field_df)


def run_comparison_experiment(
    config: TERAGConfig,
    test_df: pd.DataFrame,
    train_df: pd.DataFrame,
    output_dir: str
) -> pd.DataFrame:
    """运行对比实验（四种方法）"""
    print("\n" + "=" * 60)
    print("对比实验（四种方法）")
    print("=" * 60)

    evaluator = UnifiedEvaluator(config, enable_sql_eval=True)

    # 训练 SQL 生成器（从训练数据提取模板）
    print("\n训练 SQL 生成器...")
    evaluator.train_sql_generator(train_df)

    results = []

    # 1. TE-RAG V2
    print("\n[1/4] 评估 TE-RAG V2...")
    try:
        terag_v2 = TERAGRetrieverV2.from_artifacts(config)
        metrics = evaluator.evaluate_retriever(terag_v2, test_df)

        result = {'Method': 'TE-RAG-V2'}
        result.update(metrics.to_dict())
        results.append(result)
        print(f"  Table@5: {metrics.table_at_5:.4f}, Field_Coverage: {metrics.field_coverage:.4f}")
        print(f"  SQL_EM: {metrics.sql_em:.4f}, SQL_Parse_Rate: {metrics.sql_parse_rate:.4f}")
        print(f"  Avg Time: {metrics.avg_query_time*1000:.2f}ms, Avg Memory: {metrics.avg_memory_mb:.2f}MB")
    except Exception as e:
        print(f"  评估失败: {e}")
        import traceback
        traceback.print_exc()

    # 2. BM25
    print("\n[2/4] 评估 BM25...")
    try:
        bm25_wrapper = create_bm25_wrapper(config)
        metrics = evaluator.evaluate_retriever(bm25_wrapper, test_df)

        result = {'Method': 'BM25'}
        result.update(metrics.to_dict())
        results.append(result)
        print(f"  Table@5: {metrics.table_at_5:.4f}, Field_Coverage: {metrics.field_coverage:.4f}")
        print(f"  SQL_EM: {metrics.sql_em:.4f}, SQL_Parse_Rate: {metrics.sql_parse_rate:.4f}")
        print(f"  Avg Time: {metrics.avg_query_time*1000:.2f}ms, Avg Memory: {metrics.avg_memory_mb:.2f}MB")
    except Exception as e:
        print(f"  评估失败: {e}")

    # 3. Vector (TF-IDF)
    print("\n[3/4] 评估 Vector (TF-IDF)...")
    try:
        from retrievers.vector_retriever import VectorRetriever
        from retrievers.base_retriever import RetrieverConfig, RetrievalResult

        vector_retriever = VectorRetriever(
            config.data.field_csv,
            config.data.table_csv,
            RetrieverConfig(name="Vector")
        )
        vector_retriever.fit(train_df)

        # 直接使用原始检索器（不通过包装器）
        class DirectVectorWrapper:
            def __init__(self, inner):
                self.inner = inner

            def retrieve(self, query, k=5):
                from terag.terag_retriever_v2 import RetrievalResultV2
                results = self.inner.retrieve(query, k)
                return [
                    RetrievalResultV2(
                        table=r.table,
                        table_score=r.table_score,
                        columns=r.columns,
                        metadata=r.metadata
                    )
                    for r in results
                ]

        vector_wrapper = DirectVectorWrapper(vector_retriever)
        metrics = evaluator.evaluate_retriever(vector_wrapper, test_df)

        result = {'Method': 'Vector'}
        result.update(metrics.to_dict())
        results.append(result)
        print(f"  Table@5: {metrics.table_at_5:.4f}, Field_Coverage: {metrics.field_coverage:.4f}")
        print(f"  SQL_EM: {metrics.sql_em:.4f}, SQL_Parse_Rate: {metrics.sql_parse_rate:.4f}")
        print(f"  Avg Time: {metrics.avg_query_time*1000:.2f}ms, Avg Memory: {metrics.avg_memory_mb:.2f}MB")
    except Exception as e:
        print(f"  评估失败: {e}")
        import traceback
        traceback.print_exc()

    # 4. LLM (语义匹配)
    print("\n[4/4] 评估 LLM (语义匹配)...")
    try:
        from retrievers.llm_retriever import LLMRetriever
        from retrievers.base_retriever import RetrieverConfig

        llm_retriever = LLMRetriever(
            config.data.field_csv,
            config.data.table_csv,
            RetrieverConfig(name="LLM")
        )
        llm_retriever.fit(train_df)

        llm_wrapper = create_retriever_wrapper(llm_retriever, "LLM")
        metrics = evaluator.evaluate_retriever(llm_wrapper, test_df)

        result = {'Method': 'LLM'}
        result.update(metrics.to_dict())
        results.append(result)
        print(f"  Table@5: {metrics.table_at_5:.4f}, Field_Coverage: {metrics.field_coverage:.4f}")
        print(f"  SQL_EM: {metrics.sql_em:.4f}, SQL_Parse_Rate: {metrics.sql_parse_rate:.4f}")
        print(f"  Avg Time: {metrics.avg_query_time*1000:.2f}ms, Avg Memory: {metrics.avg_memory_mb:.2f}MB")
    except Exception as e:
        print(f"  评估失败: {e}")

    # 保存结果
    df = pd.DataFrame(results)
    output_path = os.path.join(output_dir, 'comparison_results.csv')
    df.to_csv(output_path, index=False)
    print(f"\n结果已保存到: {output_path}")

    return df


def run_cold_start_experiment(
    config: TERAGConfig,
    qa_df: pd.DataFrame,
    output_dir: str,
    test_ratio: float = 0.2
) -> pd.DataFrame:
    """运行冷启动实验"""
    print("\n" + "=" * 60)
    print("冷启动实验")
    print("=" * 60)

    # 获取所有表
    all_tables = qa_df['table'].unique()
    n_test = max(1, int(len(all_tables) * test_ratio))

    # 随机选择测试表
    np.random.seed(config.seed)
    test_tables = np.random.choice(all_tables, n_test, replace=False)

    # 分割数据
    test_data = qa_df[qa_df['table'].isin(test_tables)]
    train_data = qa_df[~qa_df['table'].isin(test_tables)]

    print(f"训练集: {len(train_data)} 条查询 (来自 {len(train_data['table'].unique())} 个表)")
    print(f"测试集: {len(test_data)} 条查询 (来自 {n_test} 个新表)")

    evaluator = UnifiedEvaluator(config)
    results = []

    # 1. TE-RAG V2
    print("\n[1/4] 评估 TE-RAG V2 (冷启动)...")
    try:
        # 重新训练（只用训练数据）
        terag_v2 = TERAGRetrieverV2(config)
        terag_v2.fit(train_data)

        metrics = evaluator.evaluate_retriever(terag_v2, test_data)

        result = {'Method': 'TE-RAG-V2', 'Type': 'Cold Start'}
        result.update(metrics.to_dict())
        results.append(result)
        print(f"  Table@5: {metrics.table_at_5:.4f}, Field_Coverage: {metrics.field_coverage:.4f}")
    except Exception as e:
        print(f"  评估失败: {e}")

    # 2. BM25 (不需要训练数据，可以直接评估)
    print("\n[2/4] 评估 BM25 (冷启动)...")
    try:
        bm25_wrapper = create_bm25_wrapper(config)
        metrics = evaluator.evaluate_retriever(bm25_wrapper, test_data)

        result = {'Method': 'BM25', 'Type': 'Cold Start'}
        result.update(metrics.to_dict())
        results.append(result)
        print(f"  Table@5: {metrics.table_at_5:.4f}, Field_Coverage: {metrics.field_coverage:.4f}")
    except Exception as e:
        print(f"  评估失败: {e}")

    # 3. Vector
    print("\n[3/4] 评估 Vector (冷启动)...")
    try:
        from retrievers.vector_retriever import VectorRetriever
        from retrievers.base_retriever import RetrieverConfig

        vector_retriever = VectorRetriever(
            config.data.field_csv,
            config.data.table_csv,
            RetrieverConfig(name="Vector")
        )
        vector_retriever.fit(train_data)

        vector_wrapper = create_retriever_wrapper(vector_retriever, "Vector")
        metrics = evaluator.evaluate_retriever(vector_wrapper, test_data)

        result = {'Method': 'Vector', 'Type': 'Cold Start'}
        result.update(metrics.to_dict())
        results.append(result)
        print(f"  Table@5: {metrics.table_at_5:.4f}, Field_Coverage: {metrics.field_coverage:.4f}")
    except Exception as e:
        print(f"  评估失败: {e}")

    # 4. LLM
    print("\n[4/4] 评估 LLM (冷启动)...")
    try:
        from retrievers.llm_retriever import LLMRetriever
        from retrievers.base_retriever import RetrieverConfig

        llm_retriever = LLMRetriever(
            config.data.field_csv,
            config.data.table_csv,
            RetrieverConfig(name="LLM")
        )
        llm_retriever.fit(train_data)

        llm_wrapper = create_retriever_wrapper(llm_retriever, "LLM")
        metrics = evaluator.evaluate_retriever(llm_wrapper, test_data)

        result = {'Method': 'LLM', 'Type': 'Cold Start'}
        result.update(metrics.to_dict())
        results.append(result)
        print(f"  Table@5: {metrics.table_at_5:.4f}, Field_Coverage: {metrics.field_coverage:.4f}")
    except Exception as e:
        print(f"  评估失败: {e}")

    # 保存结果
    df = pd.DataFrame(results)
    output_path = os.path.join(output_dir, 'cold_start_results.csv')
    df.to_csv(output_path, index=False)
    print(f"\n结果已保存到: {output_path}")

    return df


def run_ablation_experiment(
    config: TERAGConfig,
    test_df: pd.DataFrame,
    output_dir: str
) -> pd.DataFrame:
    """运行消融实验"""
    print("\n" + "=" * 60)
    print("消融实验")
    print("=" * 60)

    evaluator = UnifiedEvaluator(config)
    results = []

    # 定义消融配置
    ablation_configs = [
        {'name': 'Full TE-RAG', 'overrides': {}},
        {'name': 'w/o Graph Weight', 'overrides': {'use_graph_weight': False}},
        {'name': 'w/o Template Mining', 'overrides': {'use_template_mining': False}},
        {'name': 'w/o Pattern Generalization', 'overrides': {'use_pattern_generalization': False}},
        {'name': 'w/o Enhanced Index', 'overrides': {'use_enhanced_index': False}},
        {'name': 'w/o Role Parser', 'overrides': {'use_role_parser': False}},
    ]

    for ablation_config in ablation_configs:
        name = ablation_config['name']
        overrides = ablation_config['overrides']

        print(f"\n评估: {name}")

        try:
            # 加载检索器
            retriever = TERAGRetrieverV2.from_artifacts(config)

            # 临时修改配置
            original_values = {}
            for key, value in overrides.items():
                if hasattr(config.ablation, key):
                    original_values[key] = getattr(config.ablation, key)
                    setattr(config.ablation, key, value)

            metrics = evaluator.evaluate_retriever(retriever, test_df)

            # 恢复配置
            for key, value in original_values.items():
                setattr(config.ablation, key, value)

            result = {'Configuration': name}
            result.update(metrics.to_dict())
            results.append(result)

            print(f"  Table@5: {metrics.table_at_5:.4f}, Field_Coverage: {metrics.field_coverage:.4f}")

        except Exception as e:
            print(f"  评估失败: {e}")

    # 保存结果
    df = pd.DataFrame(results)
    output_path = os.path.join(output_dir, 'ablation_results.csv')
    df.to_csv(output_path, index=False)
    print(f"\n结果已保存到: {output_path}")

    return df


def run_resource_comparison(
    config: TERAGConfig,
    test_df: pd.DataFrame,
    train_df: pd.DataFrame,
    output_dir: str
) -> pd.DataFrame:
    """运行资源消耗对比实验"""
    print("\n" + "=" * 60)
    print("资源消耗对比")
    print("=" * 60)

    evaluator = UnifiedEvaluator(config)
    results = []

    # 只运行一次评估（资源数据已经在 comparison 中收集）
    # 这里单独输出一个资源对比表

    retrievers_to_test = [
        ('TE-RAG-V2', lambda: TERAGRetrieverV2.from_artifacts(config)),
        ('BM25', lambda: create_bm25_wrapper(config)),
        ('Vector', lambda: create_retriever_wrapper(
            __import__('retrievers.vector_retriever', fromlist=['VectorRetriever']).VectorRetriever(
                config.data.field_csv, config.data.table_csv,
                __import__('retrievers.base_retriever', fromlist=['RetrieverConfig']).RetrieverConfig(name="Vector")
            ), "Vector") if True else None
        ),
        ('LLM', lambda: create_retriever_wrapper(
            __import__('retrievers.llm_retriever', fromlist=['LLMRetriever']).LLMRetriever(
                config.data.field_csv, config.data.table_csv,
                __import__('retrievers.base_retriever', fromlist=['RetrieverConfig']).RetrieverConfig(name="LLM")
            ), "LLM") if True else None
        ),
    ]

    for name, retriever_fn in retrievers_to_test:
        print(f"\n评估: {name}")
        try:
            retriever = retriever_fn()

            # 如果需要训练
            if name in ['Vector', 'LLM']:
                if hasattr(retriever, 'inner'):
                    retriever.inner.fit(train_df)
                else:
                    retriever.fit(train_df)

            metrics = evaluator.evaluate_retriever(retriever, test_df)

            results.append({
                'Method': name,
                'Avg_Query_Time(ms)': metrics.avg_query_time * 1000,
                'Avg_Memory(MB)': metrics.avg_memory_mb,
                'Queries_Per_Second': 1.0 / metrics.avg_query_time if metrics.avg_query_time > 0 else 0,
            })
            print(f"  Time: {metrics.avg_query_time*1000:.2f}ms, Memory: {metrics.avg_memory_mb:.2f}MB")

        except Exception as e:
            print(f"  评估失败: {e}")

    # 保存结果
    df = pd.DataFrame(results)
    output_path = os.path.join(output_dir, 'resource_results.csv')
    df.to_csv(output_path, index=False)
    print(f"\n结果已保存到: {output_path}")

    return df


def main():
    parser = argparse.ArgumentParser(description='实验运行脚本')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='配置文件路径')
    parser.add_argument('--test', type=str, default=None,
                        help='测试数据路径')
    parser.add_argument('--suite', type=str, default='all',
                        choices=['all', 'comparison', 'ablation', 'cold_start', 'resource'],
                        help='实验类型')
    parser.add_argument('--output', type=str, default=None,
                        help='输出目录')

    args = parser.parse_args()

    # 加载配置
    config_path = Path(__file__).parent.parent / args.config
    config = TERAGConfig.from_yaml(str(config_path))

    # 输出目录
    output_dir = args.output or os.path.join(
        config.output.results_dir,
        datetime.now().strftime('%Y%m%d_%H%M%S')
    )
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("TE-RAG 完整实验")
    print("=" * 60)
    print(f"配置文件: {config_path}")
    print(f"输出目录: {output_dir}")

    # 加载数据
    test_path = args.test or config.get_split_path('test')
    train_path = config.get_split_path('train')
    qa_path = config.data.qa_csv

    test_df = load_jsonl(test_path)
    train_df = load_jsonl(train_path)
    qa_df = pd.read_csv(qa_path)

    print(f"测试数据: {len(test_df)} 条")
    print(f"训练数据: {len(train_df)} 条")
    print(f"全部数据: {len(qa_df)} 条")

    # 保存实验配置
    config.save_yaml(os.path.join(output_dir, 'config.yaml'))

    # 运行实验
    results = {}

    if args.suite in ['all', 'comparison']:
        results['comparison'] = run_comparison_experiment(config, test_df, train_df, output_dir)

    if args.suite in ['all', 'ablation']:
        results['ablation'] = run_ablation_experiment(config, test_df, output_dir)

    if args.suite in ['all', 'cold_start']:
        results['cold_start'] = run_cold_start_experiment(config, qa_df, output_dir)

    if args.suite in ['all', 'resource']:
        results['resource'] = run_resource_comparison(config, test_df, train_df, output_dir)

    # 打印汇总
    print("\n" + "=" * 60)
    print("实验结果汇总")
    print("=" * 60)

    for name, df in results.items():
        print(f"\n{name}:")
        print(df.to_string())

    print(f"\n所有结果已保存到: {output_dir}")


if __name__ == "__main__":
    main()
