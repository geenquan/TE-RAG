"""
快速实验运行器

运行简化版的对比实验和消融实验，快速生成结果
支持时间戳目录和数据划分保存
"""

import os
import sys
import pandas as pd
import numpy as np
import time
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrievers import (
    RetrieverFactory, RetrieverManager, RetrieverConfig,
    BaseRetriever, EvaluationMetrics
)


def get_timestamp_dir(base_dir: str) -> str:
    """生成时间戳目录名，格式：YYYYMMDD_HHMMSS"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def save_data_splits(output_dir: str, comparison_train: pd.DataFrame, comparison_test: pd.DataFrame,
                     cold_start_train: pd.DataFrame, cold_start_test: pd.DataFrame,
                     cold_start_tables: list):
    """保存数据划分到目录，方便复现"""
    splits_dir = os.path.join(output_dir, 'splits')
    os.makedirs(splits_dir, exist_ok=True)

    # 保存对比实验数据划分
    comparison_train.to_csv(os.path.join(splits_dir, 'comparison_train.csv'), index=False)
    comparison_test.to_csv(os.path.join(splits_dir, 'comparison_test.csv'), index=False)

    # 保存冷启动实验数据划分
    cold_start_train.to_csv(os.path.join(splits_dir, 'cold_start_train.csv'), index=False)
    cold_start_test.to_csv(os.path.join(splits_dir, 'cold_start_test.csv'), index=False)

    # 保存冷启动测试表列表
    with open(os.path.join(splits_dir, 'cold_start_tables.txt'), 'w') as f:
        for table in cold_start_tables:
            f.write(f"{table}\n")

    print(f"数据划分已保存到: {splits_dir}")


def run_quick_comparison(field_csv: str, table_csv: str, output_dir: str,
                         train_data: pd.DataFrame, test_data: pd.DataFrame):
    """
    快速对比实验
    """
    print("\n" + "=" * 60)
    print("对比实验")
    print("=" * 60)

    print(f"训练集: {len(train_data)} 条")
    print(f"测试集: {len(test_data)} 条")

    # 创建所有检索器
    methods = ['BM25', 'Vector', 'Hybrid', 'LLM', 'Graph', 'TE-RAG-V2']
    results = []

    # 用于验证方法独立性的调试信息
    print(f"\n[DEBUG] 可用检索器: {RetrieverFactory.list_available()}")

    for method in methods:
        print(f"\n--- 测试 {method} ---")

        # 创建 - 添加调试信息
        print(f"  [DEBUG] 创建 {method} 检索器...")
        retriever = RetrieverFactory.create(method, field_csv, table_csv)
        print(f"  [DEBUG] 检索器类型: {type(retriever).__name__}")

        # 训练
        start_time = time.time()
        retriever.fit(train_data)
        fit_time = time.time() - start_time
        print(f"  训练时间: {fit_time:.2f}s")

        # 评估
        start_time = time.time()
        metrics = retriever.evaluate(test_data, k=5)
        eval_time = time.time() - start_time
        print(f"  评估时间: {eval_time:.2f}s")

        print(f"  Table Accuracy: {metrics.table_accuracy:.1%}")
        print(f"  SQL Accuracy: {metrics.sql_accuracy:.1%}")
        print(f"  Avg Query Time: {metrics.avg_query_time*1000:.1f}ms")

        # 新增指标输出
        print(f"  Table Recall: {metrics.table_recall:.1%}")
        print(f"  Column Recall: {metrics.column_recall:.1%}")
        print(f"  Column Precision: {metrics.column_precision:.1%}")
        print(f"  Column F1: {metrics.column_f1:.1%}")
        print(f"  Top1 Table Recall: {metrics.top1_table_recall:.1%}")
        print(f"  Top3 Table Recall: {metrics.top3_table_recall:.1%}")
        print(f"  Top5 Table Recall: {metrics.top5_table_recall:.1%}")

        # 添加调试信息，验证结果唯一性
        print(f"  [DEBUG] 结果ID: {id(metrics)}, Table Acc: {metrics.table_accuracy:.4f}")

        results.append({
            'Method': method,
            'Table Accuracy': metrics.table_accuracy,
            'SQL Accuracy': metrics.sql_accuracy,
            'Table Recall': metrics.table_recall,
            'Column Recall': metrics.column_recall,
            'Column Precision': metrics.column_precision,
            'Column F1': metrics.column_f1,
            'Top1 Table Recall': metrics.top1_table_recall,
            'Top3 Table Recall': metrics.top3_table_recall,
            'Top5 Table Recall': metrics.top5_table_recall,
            'Avg Query Time (s)': metrics.avg_query_time,
            'Avg Memory (MB)': metrics.avg_memory_mb,
            'Fit Time (s)': fit_time
        })

    # 保存结果
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, 'comparison_results.csv'), index=False)

    print("\n对比实验结果:")
    print(df.to_string(index=False))

    # 验证结果是否唯一
    print("\n[DEBUG] 验证结果唯一性:")
    for i, r in enumerate(results):
        print(f"  {r['Method']}: Table Acc = {r['Table Accuracy']:.4f}, SQL Acc = {r['SQL Accuracy']:.4f}")

    return df


def run_quick_cold_start(field_csv: str, table_csv: str, output_dir: str,
                         train_data: pd.DataFrame, test_data: pd.DataFrame,
                         test_tables: list):
    """
    冷启动实验
    """
    print("\n" + "=" * 60)
    print("冷启动实验")
    print("=" * 60)

    print(f"总测试表数: {len(test_tables)} (新表)")
    print(f"训练数据: {len(train_data)} 条")
    print(f"测试数据: {len(test_data)} 条")

    methods = ['BM25', 'Vector', 'Hybrid', 'LLM', 'Graph', 'TE-RAG-V2']
    results = []

    for method in methods:
        print(f"\n--- 测试 {method} ---")

        # 添加调试信息
        print(f"  [DEBUG] 创建 {method} 检索器...")
        retriever = RetrieverFactory.create(method, field_csv, table_csv)
        print(f"  [DEBUG] 检索器类型: {type(retriever).__name__}")

        retriever.fit(train_data)

        metrics = retriever.evaluate(test_data, k=5)

        print(f"  Table Accuracy: {metrics.table_accuracy:.1%}")
        print(f"  SQL Accuracy: {metrics.sql_accuracy:.1%}")
        print(f"  Table Recall: {metrics.table_recall:.1%}")
        print(f"  Column Recall: {metrics.column_recall:.1%}")
        print(f"  Top1 Table Recall: {metrics.top1_table_recall:.1%}")
        print(f"  Top3 Table Recall: {metrics.top3_table_recall:.1%}")
        print(f"  Top5 Table Recall: {metrics.top5_table_recall:.1%}")

        # 添加调试信息，验证结果唯一性
        print(f"  [DEBUG] 结果ID: {id(metrics)}, Table Acc: {metrics.table_accuracy:.4f}")

        results.append({
            'Method': method,
            'Table Accuracy': metrics.table_accuracy,
            'SQL Accuracy': metrics.sql_accuracy,
            'Table Recall': metrics.table_recall,
            'Column Recall': metrics.column_recall,
            'Top1 Table Recall': metrics.top1_table_recall,
            'Top3 Table Recall': metrics.top3_table_recall,
            'Top5 Table Recall': metrics.top5_table_recall,
            'Type': 'Cold Start'
        })

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, 'cold_start_results.csv'), index=False)

    print("\n冷启动实验结果:")
    print(df.to_string(index=False))

    # 验证结果是否唯一
    print("\n[DEBUG] 验证冷启动结果唯一性:")
    for i, r in enumerate(results):
        print(f"  {r['Method']}: Table Acc = {r['Table Accuracy']:.4f}, SQL Acc = {r['SQL Accuracy']:.4f}")

    return df


def run_quick_ablation(field_csv: str, table_csv: str, output_dir: str,
                       train_data: pd.DataFrame, test_data: pd.DataFrame):
    """
    简化版消融实验
    """
    print("\n" + "=" * 60)
    print("消融实验")
    print("=" * 60)

    from experiment.ablation import AblationTE_RAG

    # 保存训练数据
    train_csv = os.path.join(output_dir, 'splits', 'ablation_train.csv')
    train_data.to_csv(train_csv, index=False)

    # 定义消融配置
    configs = [
        {'name': 'Full TE-RAG', 'settings': {
            'use_graph_weight': True,
            'use_template_mining': True,
            'use_pattern_generalization': True,
            'use_business_rules': True,
            'use_enhanced_index': True
        }},
        {'name': 'w/o Graph Weight', 'settings': {
            'use_graph_weight': False,
            'use_template_mining': True,
            'use_pattern_generalization': True,
            'use_business_rules': True,
            'use_enhanced_index': True
        }},
        {'name': 'w/o Template Mining', 'settings': {
            'use_graph_weight': True,
            'use_template_mining': False,
            'use_pattern_generalization': True,
            'use_business_rules': True,
            'use_enhanced_index': True
        }},
        {'name': 'w/o Pattern Generalization', 'settings': {
            'use_graph_weight': True,
            'use_template_mining': True,
            'use_pattern_generalization': False,
            'use_business_rules': True,
            'use_enhanced_index': True
        }},
        {'name': 'w/o Enhanced Index', 'settings': {
            'use_graph_weight': True,
            'use_template_mining': True,
            'use_pattern_generalization': True,
            'use_business_rules': True,
            'use_enhanced_index': False
        }},
    ]

    results = []

    for config in configs:
        print(f"\n--- 测试 {config['name']} ---")

        retriever = AblationTE_RAG(field_csv, table_csv, train_csv)
        retriever.use_graph_weight = config['settings']['use_graph_weight']
        retriever.use_template_mining = config['settings']['use_template_mining']
        retriever.use_pattern_generalization = config['settings']['use_pattern_generalization']
        retriever.use_business_rules = config['settings']['use_business_rules']
        retriever.use_enhanced_index = config['settings']['use_enhanced_index']

        start_time = time.time()
        retriever.fit()
        fit_time = time.time() - start_time

        # 评估
        table_correct = 0
        sql_correct = 0
        query_times = []

        for _, row in test_data.iterrows():
            query = row['question']
            gt_table = row['table'].split('.')[-1] if pd.notna(row['table']) else ''
            gt_fields = []
            if pd.notna(row.get('field', '')) and isinstance(row['field'], str):
                gt_fields = [f.strip() for f in row['field'].split('|') if f.strip()]

            start = time.time()
            try:
                results_list = retriever.retrieve(query, k=5)
            except:
                results_list = []
            query_times.append(time.time() - start)

            retrieved_tables = [r['table'].replace('T:', '') for r in results_list]
            if gt_table in retrieved_tables:
                table_correct += 1
                for r in results_list:
                    if r['table'].replace('T:', '') == gt_table:
                        retrieved_fields = [col.split('.')[-1] for col, _ in r['columns']]
                        if not gt_fields or all(f in retrieved_fields for f in gt_fields):
                            sql_correct += 1
                        break

        total = len(test_data)
        table_acc = table_correct / total if total > 0 else 0
        sql_acc = sql_correct / total if total > 0 else 0
        avg_time = np.mean(query_times)

        print(f"  Table Accuracy: {table_acc:.1%}")
        print(f"  SQL Accuracy: {sql_acc:.1%}")
        print(f"  Avg Query Time: {avg_time*1000:.1f}ms")

        results.append({
            'Configuration': config['name'],
            'Table Accuracy': table_acc,
            'SQL Accuracy': sql_acc,
            'Avg Query Time (s)': avg_time
        })

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, 'ablation_results.csv'), index=False)

    print("\n消融实验结果:")
    print(df.to_string(index=False))

    return df


def main():
    # 数据路径
    field_csv = '/Users/apple/Documents/浙大工作/论文/分层查询数据表/code/source_dataset/processed_field_schema.csv'
    table_csv = '/Users/apple/Documents/浙大工作/论文/分层查询数据表/code/source_dataset/processed_table_schema.csv'
    qa_csv = '/Users/apple/Documents/浙大工作/论文/分层查询数据表/code/source_dataset/processed_qa_data.csv'
    base_results_dir = '/Users/apple/Documents/浙大工作/论文/分层查询数据表/code/results'

    # 生成时间戳目录
    output_dir = get_timestamp_dir(base_results_dir)
    print(f"实验结果目录: {output_dir}")

    print("=" * 60)
    print("TE-RAG 实验运行")
    print("=" * 60)

    start_time = time.time()

    # 读取数据
    qa_df = pd.read_csv(qa_csv)
    print(f"总数据: {len(qa_df)} 条")

    # ============ 数据划分（统一在 main 中进行） ============

    # 1. 对比实验数据划分
    print("\n划分对比实验数据...")
    np.random.seed(42)
    indices = np.random.permutation(len(qa_df))
    n_train = int(len(indices) * 0.7)
    comparison_train = qa_df.iloc[indices[:n_train]].reset_index(drop=True)
    comparison_test = qa_df.iloc[indices[n_train:]].reset_index(drop=True)

    # 2. 冷启动实验数据划分
    print("划分冷启动实验数据...")
    all_tables = qa_df['table'].unique()
    np.random.seed(42)
    n_test_tables = max(1, int(len(all_tables) * 0.2))
    cold_start_test_tables = list(np.random.choice(all_tables, size=n_test_tables, replace=False))
    cold_start_train = qa_df[~qa_df['table'].isin(cold_start_test_tables)].reset_index(drop=True)
    cold_start_test = qa_df[qa_df['table'].isin(cold_start_test_tables)].reset_index(drop=True)

    # 保存数据划分
    save_data_splits(output_dir, comparison_train, comparison_test,
                     cold_start_train, cold_start_test, cold_start_test_tables)

    # ============ 运行实验 ============

    # 1. 对比实验
    comparison_df = run_quick_comparison(field_csv, table_csv, output_dir,
                                         comparison_train, comparison_test)

    # 2. 冷启动实验
    cold_start_df = run_quick_cold_start(field_csv, table_csv, output_dir,
                                         cold_start_train, cold_start_test,
                                         cold_start_test_tables)

    # 3. 消融实验
    ablation_df = run_quick_ablation(field_csv, table_csv, output_dir,
                                     comparison_train, comparison_test)

    # 生成可视化
    print("\n" + "=" * 60)
    print("生成可视化图表...")
    print("=" * 60)

    from experiment.visualize import ExperimentVisualizer
    visualizer = ExperimentVisualizer(output_dir)

    # 生成图表
    visualizer.plot_comparison_bar_chart(
        comparison_df,
        metrics=['Table Accuracy', 'SQL Accuracy'],
        filename="comparison_accuracy.png"
    )

    visualizer.plot_cold_start_comparison(
        cold_start_df,
        filename="cold_start_comparison.png"
    )

    visualizer.plot_ablation_results(
        ablation_df,
        filename="ablation_results.png"
    )

    # 生成报告
    visualizer.create_summary_report(comparison_df, ablation_df, cold_start_df)

    total_time = time.time() - start_time

    print("\n" + "=" * 60)
    print(f"实验完成! 总耗时: {total_time/60:.1f} 分钟")
    print("=" * 60)
    print(f"\n结果保存在: {output_dir}")
    for f in os.listdir(output_dir):
        if f.endswith(('.csv', '.png', '.md', '.tex')):
            filepath = os.path.join(output_dir, f)
            size = os.path.getsize(filepath)
            print(f"  - {f} ({size/1024:.1f} KB)")


if __name__ == "__main__":
    main()
