"""
单独实验运行脚本

可以灵活选择：
1. 要运行的检索方法（可选多个）
2. 是否运行冷启动实验
3. 是否运行正常对比实验

使用方法：
    python run_single_experiment.py --methods BM25,RAT-SQL --cold-start --normal
    python run_single_experiment.py --methods BM25,Vector,Hybrid --cold-start
    python run_single_experiment.py --methods RAT-SQL,RESDSQL --normal
"""

import os
import sys
import argparse
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
    """生成时间戳目录名"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def run_normal_experiment(field_csv: str, table_csv: str, output_dir: str,
                          train_data: pd.DataFrame, test_data: pd.DataFrame,
                          methods: list):
    """
    正常对比实验

    训练集和测试集来自同一批表
    """
    print("\n" + "=" * 60)
    print("正常对比实验")
    print("=" * 60)

    print(f"训练集: {len(train_data)} 条")
    print(f"测试集: {len(test_data)} 条")

    results = []

    for method in methods:
        print(f"\n--- {method} ---")

        try:
            # 创建检索器
            retriever = RetrieverFactory.create(method, field_csv, table_csv)

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
            print(f"  Table Recall: {metrics.table_recall:.1%}")
            print(f"  Column Recall: {metrics.column_recall:.1%}")
            print(f"  Column F1: {metrics.column_f1:.1%}")
            print(f"  Top1 Table Recall: {metrics.top1_table_recall:.1%}")
            print(f"  Top3 Table Recall: {metrics.top3_table_recall:.1%}")
            print(f"  Avg Query Time: {metrics.avg_query_time*1000:.1f}ms")

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
                'Fit Time (s)': fit_time,
                'Type': 'Normal'
            })
        except Exception as e:
            print(f"  错误: {e}")
            import traceback
            traceback.print_exc()

    # 保存结果
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, 'normal_results.csv'), index=False)

    print("\n" + "=" * 60)
    print("实验结果汇总")
    print("=" * 60)
    print(df[['Method', 'Table Accuracy', 'SQL Accuracy', 'Column Recall', 'Top1 Table Recall']].to_string(index=False))

    return df


def run_cold_start_experiment(field_csv: str, table_csv: str, output_dir: str,
                               train_data: pd.DataFrame, test_data: pd.DataFrame,
                               test_tables: list, methods: list):
    """
    冷启动实验

    测试集中的表不在训练集中出现
    """
    print("\n" + "=" * 60)
    print("冷启动实验")
    print("=" * 60)

    print(f"总测试表数: {len(test_tables)} (新表，冷启动)")
    print(f"训练数据: {len(train_data)} 条")
    print(f"测试数据: {len(test_data)} 条")

    results = []

    for method in methods:
        print(f"\n--- {method} (冷启动) ---")

        try:
            # 创建检索器
            retriever = RetrieverFactory.create(method, field_csv, table_csv)

            # 训练
            retriever.fit(train_data)

            # 评估
            metrics = retriever.evaluate(test_data, k=5)

            print(f"  Table Accuracy: {metrics.table_accuracy:.1%}")
            print(f"  SQL Accuracy: {metrics.sql_accuracy:.1%}")
            print(f"  Table Recall: {metrics.table_recall:.1%}")
            print(f"  Column Recall: {metrics.column_recall:.1%}")
            print(f"  Top1 Table Recall: {metrics.top1_table_recall:.1%}")
            print(f"  Top3 Table Recall: {metrics.top3_table_recall:.1%}")

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
                'Type': 'Cold Start'
            })
        except Exception as e:
            print(f"  错误: {e}")
            import traceback
            traceback.print_exc()

    # 保存结果
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, 'cold_start_results.csv'), index=False)

    print("\n" + "=" * 60)
    print("冷启动实验结果汇总")
    print("=" * 60)
    print(df[['Method', 'Table Accuracy', 'SQL Accuracy', 'Column Recall', 'Top1 Table Recall']].to_string(index=False))

    return df


def main():
    parser = argparse.ArgumentParser(description='单独运行实验脚本')
    parser.add_argument('--methods', type=str, default='BM25,RAT-SQL',
                       help='要测试的方法，逗号分隔 (默认: BM25,RAT-SQL)')
    parser.add_argument('--cold-start', action='store_true',
                       help='是否运行冷启动实验')
    parser.add_argument('--normal', action='store_true',
                       help='是否运行正常对比实验')
    parser.add_argument('--train-ratio', type=float, default=0.7,
                       help='训练集比例 (默认: 0.7)')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子 (默认: 42)')

    args = parser.parse_args()

    # 如果没有指定任何实验类型，默认运行两种
    run_cold_start = args.cold_start
    run_normal = args.normal
    if not run_cold_start and not run_normal:
        run_cold_start = True
        run_normal = True

    # 解析方法列表
    methods = [m.strip() for m in args.methods.split(',')]

    # 检查方法是否可用
    available_methods = RetrieverFactory.list_available()
    for method in methods:
        if method not in available_methods:
            print(f"警告: 方法 '{method}' 不可用")
            print(f"可用方法: {available_methods}")
            sys.exit(1)

    # 数据路径
    field_csv = '/Users/apple/Documents/浙大工作/论文/分层查询数据表/code/source_dataset/processed_field_schema.csv'
    table_csv = '/Users/apple/Documents/浙大工作/论文/分层查询数据表/code/source_dataset/processed_table_schema.csv'
    qa_csv = '/Users/apple/Documents/浙大工作/论文/分层查询数据表/code/source_dataset/processed_qa_data.csv'
    base_results_dir = '/Users/apple/Documents/浙大工作/论文/分层查询数据表/code/results'

    # 生成时间戳目录
    output_dir = get_timestamp_dir(base_results_dir)
    print(f"实验结果目录: {output_dir}")

    print("=" * 60)
    print("单独实验运行")
    print("=" * 60)
    print(f"要测试的方法: {methods}")
    print(f"运行冷启动实验: {run_cold_start}")
    print(f"运行正常实验: {run_normal}")
    print(f"训练集比例: {args.train_ratio}")
    print(f"随机种子: {args.seed}")

    start_time = time.time()

    # 读取数据
    qa_df = pd.read_csv(qa_csv)
    print(f"\n总数据: {len(qa_df)} 条")

    # 设置随机种子
    np.random.seed(args.seed)

    # ============ 数据划分 ============

    # 1. 正常对比实验数据划分
    if run_normal:
        print("\n划分正常对比实验数据...")
        indices = np.random.permutation(len(qa_df))
        n_train = int(len(indices) * args.train_ratio)
        normal_train = qa_df.iloc[indices[:n_train]].reset_index(drop=True)
        normal_test = qa_df.iloc[indices[n_train:]].reset_index(drop=True)

    # 2. 冷启动实验数据划分
    if run_cold_start:
        print("划分冷启动实验数据...")
        all_tables = qa_df['table'].unique()
        np.random.seed(args.seed)
        n_test_tables = max(1, int(len(all_tables) * 0.3))
        cold_start_test_tables = list(np.random.choice(all_tables, size=n_test_tables, replace=False))
        cold_start_train = qa_df[~qa_df['table'].isin(cold_start_test_tables)].reset_index(drop=True)
        cold_start_test = qa_df[qa_df['table'].isin(cold_start_test_tables)].reset_index(drop=True)

    # ============ 运行实验 ============

    # 1. 正常对比实验
    if run_normal:
        normal_df = run_normal_experiment(
            field_csv, table_csv, output_dir,
            normal_train, normal_test, methods
        )

    # 2. 冷启动实验
    if run_cold_start:
        cold_start_df = run_cold_start_experiment(
            field_csv, table_csv, output_dir,
            cold_start_train, cold_start_test,
            cold_start_test_tables, methods
        )

    total_time = time.time() - start_time

    print("\n" + "=" * 60)
    print(f"实验完成! 总耗时: {total_time/60:.1f} 分钟")
    print("=" * 60)
    print(f"\n结果保存在: {output_dir}")
    for f in os.listdir(output_dir):
        if f.endswith(('.csv', '.png', '.md')):
            filepath = os.path.join(output_dir, f)
            size = os.path.getsize(filepath)
            print(f"  - {f} ({size/1024:.1f} KB)")


if __name__ == "__main__":
    main()
