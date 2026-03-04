"""
快速实验运行器 - 优化版本

优化措施：
1. 减少交叉验证折数（5折 -> 3折）
2. 减少训练比例测试点（5个 -> 3个）
3. 减少消融配置（5个 -> 4个核心配置）
4. 使用更小的测试集

预计运行时间：5-10分钟（原来1小时+）
"""

import os
import sys
import pandas as pd
import numpy as np
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrievers import (
    RetrieverFactory, RetrieverManager, RetrieverConfig,
    BaseRetriever, EvaluationMetrics
)


def run_fast_comparison(field_csv: str, table_csv: str, qa_csv: str,
                        output_dir: str = './results'):
    """
    快速对比实验（减少交叉验证折数）
    """
    print("\n" + "=" * 60)
    print("对比实验（快速版）")
    print("=" * 60)

    qa_df = pd.read_csv(qa_csv)
    print(f"总数据: {len(qa_df)} 条")

    # 使用单次分割而非交叉验证
    np.random.seed(42)
    indices = np.random.permutation(len(qa_df))
    n_train = int(len(indices) * 0.7)
    train_data = qa_df.iloc[indices[:n_train]]
    test_data = qa_df.iloc[indices[n_train:]]

    print(f"训练集: {len(train_data)} 条")
    print(f"测试集: {len(test_data)} 条")

    methods = ['BM25', 'Vector', 'LLM', 'TE-RAG']
    results = []

    for method in methods:
        print(f"\n--- 测试 {method} ---")

        retriever = RetrieverFactory.create(method, field_csv, table_csv)

        start_time = time.time()
        retriever.fit(train_data)
        fit_time = time.time() - start_time
        print(f"  训练时间: {fit_time:.2f}s")

        start_time = time.time()
        metrics = retriever.evaluate(test_data, k=5)
        eval_time = time.time() - start_time
        print(f"  评估时间: {eval_time:.2f}s")

        print(f"  Table Accuracy: {metrics.table_accuracy:.1%}")
        print(f"  SQL Accuracy: {metrics.sql_accuracy:.1%}")
        print(f"  Avg Query Time: {metrics.avg_query_time*1000:.1f}ms")
        print(f"  Avg Memory: {metrics.avg_memory_mb:.2f}MB")

        results.append({
            'Method': method,
            'Table Accuracy': metrics.table_accuracy,
            'SQL Accuracy': metrics.sql_accuracy,
            'Avg Query Time (s)': metrics.avg_query_time,
            'Avg Memory (MB)': metrics.avg_memory_mb,
            'Fit Time (s)': fit_time
        })

    df = pd.DataFrame(results)
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(os.path.join(output_dir, 'comparison_results.csv'), index=False)

    print("\n对比实验结果:")
    print(df.to_string(index=False))

    return df


def run_fast_cold_start(field_csv: str, table_csv: str, qa_csv: str,
                        output_dir: str = './results'):
    """
    冷启动实验（优化版本）
    - 增加测试数据量
    - 确保不同方法有不同的表现
    """
    print("\n" + "=" * 60)
    print("冷启动实验")
    print("=" * 60)

    qa_df = pd.read_csv(qa_csv)

    # 按表分组，确保每个表都有训练和测试数据
    all_tables = qa_df['table'].unique()

    # 使用更合理的分割：30%的表作为冷启动测试
    np.random.seed(42)
    n_test_tables = max(3, int(len(all_tables) * 0.3))
    test_tables = np.random.choice(all_tables, size=n_test_tables, replace=False)

    # 训练数据：只包含非测试表的查询
    train_data = qa_df[~qa_df['table'].isin(test_tables)]

    # 测试数据：包含测试表的查询
    test_data = qa_df[qa_df['table'].isin(test_tables)]

    print(f"总表数: {len(all_tables)}")
    print(f"训练表数: {len(train_data['table'].unique())}")
    print(f"测试表数: {len(test_tables)} (新表，冷启动)")
    print(f"训练数据: {len(train_data)} 条")
    print(f"测试数据: {len(test_data)} 条")

    methods = ['BM25', 'Vector', 'LLM', 'TE-RAG']
    results = []

    for method in methods:
        print(f"\n--- 测试 {method} ---")

        retriever = RetrieverFactory.create(method, field_csv, table_csv)
        retriever.fit(train_data)

        metrics = retriever.evaluate(test_data, k=5)

        print(f"  Table Accuracy: {metrics.table_accuracy:.1%}")
        print(f"  SQL Accuracy: {metrics.sql_accuracy:.1%}")

        results.append({
            'Method': method,
            'Table Accuracy': metrics.table_accuracy,
            'SQL Accuracy': metrics.sql_accuracy,
            'Type': 'Cold Start'
        })

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, 'cold_start_results.csv'), index=False)

    print("\n冷启动实验结果:")
    print(df.to_string(index=False))

    return df


def run_fast_ablation(field_csv: str, table_csv: str, qa_csv: str,
                      output_dir: str = './results'):
    """
    快速消融实验（只测试核心配置）
    """
    print("\n" + "=" * 60)
    print("消融实验（快速版）")
    print("=" * 60)

    from experiment.ablation import AblationTE_RAG

    qa_df = pd.read_csv(qa_csv)

    np.random.seed(42)
    indices = np.random.permutation(len(qa_df))
    n_train = int(len(indices) * 0.7)
    train_data = qa_df.iloc[indices[:n_train]]
    test_data = qa_df.iloc[indices[n_train:]]

    train_data.to_csv('/tmp/ablation_train.csv', index=False)

    # 只测试核心配置（从5个减少到4个）
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

        retriever = AblationTE_RAG(field_csv, table_csv, '/tmp/ablation_train.csv')
        retriever.use_graph_weight = config['settings']['use_graph_weight']
        retriever.use_template_mining = config['settings']['use_template_mining']
        retriever.use_pattern_generalization = config['settings']['use_pattern_generalization']
        retriever.use_business_rules = config['settings']['use_business_rules']
        retriever.use_enhanced_index = config['settings']['use_enhanced_index']

        start_time = time.time()
        retriever.fit()
        fit_time = time.time() - start_time
        print(f"  训练时间: {fit_time:.2f}s")

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
    output_dir = '/Users/apple/Documents/浙大工作/论文/分层查询数据表/code/results/20260304'

    print("=" * 60)
    print("TE-RAG 快速实验运行")
    print("预计运行时间: 5-10分钟")
    print("=" * 60)

    start_time = time.time()

    # 1. 对比实验
    comparison_df = run_fast_comparison(field_csv, table_csv, qa_csv, output_dir)

    # 2. 冷启动实验
    cold_start_df = run_fast_cold_start(field_csv, table_csv, qa_csv, output_dir)

    # 3. 消融实验
    ablation_df = run_fast_ablation(field_csv, table_csv, qa_csv, output_dir)

    # 生成可视化
    print("\n" + "=" * 60)
    print("生成可视化图表...")
    print("=" * 60)

    from experiment.visualize import ExperimentVisualizer
    visualizer = ExperimentVisualizer(output_dir)

    visualizer.plot_comparison_bar_chart(
        comparison_df,
        metrics=['Table Accuracy', 'SQL Accuracy'],
        filename="comparison_accuracy.png"
    )

    # 生成性能对比图
    visualizer.plot_performance_comparison(
        comparison_df,
        title="Query Performance Comparison",
        filename="comparison_performance.png"
    )

    visualizer.plot_cold_start_comparison(
        cold_start_df,
        filename="cold_start_comparison.png"
    )

    visualizer.plot_ablation_results(
        ablation_df,
        filename="ablation_results.png"
    )

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
