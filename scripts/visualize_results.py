#!/usr/bin/env python
"""
结果可视化脚本（完整版）
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def plot_comparison_results(results_dir: str):
    """绘制对比实验结果"""
    comparison_path = os.path.join(results_dir, 'comparison_results.csv')
    if not os.path.exists(comparison_path):
        print(f"文件不存在: {comparison_path}")
        return

    df = pd.read_csv(comparison_path)

    # 创建图表
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 图1: Table@k 对比
    ax1 = axes[0]
    methods = df['Method'].tolist()
    table_k = ['Table@1', 'Table@3', 'Table@5', 'Table@10']
    x = np.arange(len(methods))
    width = 0.2

    for i, k in enumerate(table_k):
        if k in df.columns:
            values = df[k].tolist()
            bars = ax1.bar(x + i * width, values, width, label=k)
            for bar, val in zip(bars, values):
                ax1.annotate(f'{val:.2f}',
                            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=8)

    ax1.set_xlabel('Method')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Table Retrieval Accuracy @k')
    ax1.set_xticks(x + width * 1.5)
    ax1.set_xticklabels(methods, rotation=15, ha='right')
    ax1.legend(loc='upper left')
    ax1.set_ylim(0, 1.0)
    ax1.grid(axis='y', alpha=0.3)

    # 图2: Field Coverage 对比
    ax2 = axes[1]
    field_metrics = ['Column@5', 'Column@10', 'Column@20', 'Field_Coverage']
    x = np.arange(len(methods))
    width = 0.2

    for i, metric in enumerate(field_metrics):
        if metric in df.columns:
            values = df[metric].tolist()
            bars = ax2.bar(x + i * width, values, width, label=metric)
            for bar, val in zip(bars, values):
                ax2.annotate(f'{val:.2f}',
                            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=8)

    ax2.set_xlabel('Method')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Field Retrieval Accuracy')
    ax2.set_xticks(x + width * 1.5)
    ax2.set_xticklabels(methods, rotation=15, ha='right')
    ax2.legend(loc='upper left')
    ax2.set_ylim(0, 1.0)
    ax2.grid(axis='y', alpha=0.3)

    # 图3: 综合对比 (Table@5 vs Field_Coverage)
    ax3 = axes[2]
    x = np.arange(len(methods))
    width = 0.35

    table_at_5 = df['Table@5'].tolist() if 'Table@5' in df.columns else [0] * len(methods)
    field_cov = df['Field_Coverage'].tolist() if 'Field_Coverage' in df.columns else [0] * len(methods)

    bars1 = ax3.bar(x - width/2, table_at_5, width, label='Table@5', color='steelblue')
    bars2 = ax3.bar(x + width/2, field_cov, width, label='Field Coverage', color='coral')

    for bar in bars1:
        ax3.annotate(f'{bar.get_height():.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

    for bar in bars2:
        ax3.annotate(f'{bar.get_height():.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

    ax3.set_xlabel('Method')
    ax3.set_ylabel('Accuracy')
    ax3.set_title('Overall Performance Comparison')
    ax3.set_xticks(x)
    ax3.set_xticklabels(methods, rotation=15, ha='right')
    ax3.legend()
    ax3.set_ylim(0, 1.0)
    ax3.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    output_path = os.path.join(results_dir, 'comparison_chart.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"对比图表已保存: {output_path}")

    plt.close()


def plot_ablation_results(results_dir: str):
    """绘制消融实验结果"""
    ablation_path = os.path.join(results_dir, 'ablation_results.csv')
    if not os.path.exists(ablation_path):
        print(f"文件不存在: {ablation_path}")
        return

    df = pd.read_csv(ablation_path)

    if len(df) == 0:
        print("消融实验结果为空")
        return

    # 创建图表
    fig, ax = plt.subplots(figsize=(14, 6))

    configs = df['Configuration'].tolist()
    table_at_5 = df['Table@5'].tolist() if 'Table@5' in df.columns else [0] * len(configs)
    field_coverage = df['Field_Coverage'].tolist() if 'Field_Coverage' in df.columns else [0] * len(configs)

    x = np.arange(len(configs))
    width = 0.35

    bars1 = ax.bar(x - width/2, table_at_5, width, label='Table@5', color='steelblue')
    bars2 = ax.bar(x + width/2, field_coverage, width, label='Field Coverage', color='coral')

    for bar in bars1:
        ax.annotate(f'{bar.get_height():.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

    for bar in bars2:
        ax.annotate(f'{bar.get_height():.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

    ax.set_xlabel('Configuration')
    ax.set_ylabel('Accuracy')
    ax.set_title('TE-RAG Ablation Study')
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=30, ha='right')
    ax.legend()
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    output_path = os.path.join(results_dir, 'ablation_chart.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"消融图表已保存: {output_path}")

    plt.close()


def plot_cold_start_results(results_dir: str):
    """绘制冷启动实验结果"""
    cold_start_path = os.path.join(results_dir, 'cold_start_results.csv')
    if not os.path.exists(cold_start_path):
        print(f"文件不存在: {cold_start_path}")
        return

    df = pd.read_csv(cold_start_path)

    if len(df) == 0:
        print("冷启动实验结果为空")
        return

    # 创建图表
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    methods = df['Method'].tolist()
    x = np.arange(len(methods))
    width = 0.35

    # 图1: Table@5 和 Field_Coverage
    ax1 = axes[0]
    table_at_5 = df['Table@5'].tolist() if 'Table@5' in df.columns else [0] * len(methods)
    field_cov = df['Field_Coverage'].tolist() if 'Field_Coverage' in df.columns else [0] * len(methods)

    bars1 = ax1.bar(x - width/2, table_at_5, width, label='Table@5', color='steelblue')
    bars2 = ax1.bar(x + width/2, field_cov, width, label='Field Coverage', color='coral')

    for bar in bars1:
        ax1.annotate(f'{bar.get_height():.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

    for bar in bars2:
        ax1.annotate(f'{bar.get_height():.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

    ax1.set_xlabel('Method')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Cold Start Performance')
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, rotation=15, ha='right')
    ax1.legend()
    ax1.set_ylim(0, 1.0)
    ax1.grid(axis='y', alpha=0.3)

    # 图2: 与常规性能对比
    ax2 = axes[1]

    # 读取常规对比结果
    comparison_path = os.path.join(results_dir, 'comparison_results.csv')
    if os.path.exists(comparison_path):
        comp_df = pd.read_csv(comparison_path)

        # 对比冷启动 vs 常规
        regular_table = []
        cold_table = []
        regular_field = []
        cold_field = []

        for method in methods:
            # 冷启动
            cold_row = df[df['Method'] == method]
            cold_table.append(cold_row['Table@5'].values[0] if len(cold_row) > 0 and 'Table@5' in cold_row else 0)
            cold_field.append(cold_row['Field_Coverage'].values[0] if len(cold_row) > 0 and 'Field_Coverage' in cold_row else 0)

            # 常规
            reg_row = comp_df[comp_df['Method'] == method]
            regular_table.append(reg_row['Table@5'].values[0] if len(reg_row) > 0 and 'Table@5' in reg_row else 0)
            regular_field.append(reg_row['Field_Coverage'].values[0] if len(reg_row) > 0 and 'Field_Coverage' in reg_row else 0)

        x = np.arange(len(methods))
        width = 0.2

        ax2.bar(x - width*1.5, regular_table, width, label='Regular Table@5', color='steelblue', alpha=0.8)
        ax2.bar(x - width/2, cold_table, width, label='Cold Start Table@5', color='steelblue', alpha=0.4, hatch='//')
        ax2.bar(x + width/2, regular_field, width, label='Regular Field Cov', color='coral', alpha=0.8)
        ax2.bar(x + width*1.5, cold_field, width, label='Cold Start Field Cov', color='coral', alpha=0.4, hatch='//')

        ax2.set_xlabel('Method')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Cold Start vs Regular Performance')
        ax2.set_xticks(x)
        ax2.set_xticklabels(methods, rotation=15, ha='right')
        ax2.legend(loc='upper right', fontsize=8)
        ax2.set_ylim(0, 1.0)
        ax2.grid(axis='y', alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'No comparison data available', ha='center', va='center', transform=ax2.transAxes)

    plt.tight_layout()

    output_path = os.path.join(results_dir, 'cold_start_chart.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"冷启动图表已保存: {output_path}")

    plt.close()


def plot_resource_results(results_dir: str):
    """绘制资源消耗对比"""
    resource_path = os.path.join(results_dir, 'resource_results.csv')
    if not os.path.exists(resource_path):
        print(f"文件不存在: {resource_path}")
        return

    df = pd.read_csv(resource_path)

    if len(df) == 0:
        print("资源消耗结果为空")
        return

    # 创建图表
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    methods = df['Method'].tolist()
    x = np.arange(len(methods))

    # 图1: 查询时间
    ax1 = axes[0]
    if 'Avg_Query_Time(ms)' in df.columns:
        times = df['Avg_Query_Time(ms)'].tolist()
        colors = plt.cm.Blues(np.linspace(0.4, 0.8, len(methods)))
        bars = ax1.bar(x, times, color=colors)
        ax1.set_ylabel('Time (ms)')
        ax1.set_title('Average Query Time')

        for bar, val in zip(bars, times):
            ax1.annotate(f'{val:.1f}',
                        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, rotation=15, ha='right')
    ax1.grid(axis='y', alpha=0.3)

    # 图2: 内存消耗
    ax2 = axes[1]
    if 'Avg_Memory(MB)' in df.columns:
        memories = df['Avg_Memory(MB)'].tolist()
        colors = plt.cm.Oranges(np.linspace(0.4, 0.8, len(methods)))
        bars = ax2.bar(x, memories, color=colors)
        ax2.set_ylabel('Memory (MB)')
        ax2.set_title('Average Memory Usage')

        for bar, val in zip(bars, memories):
            ax2.annotate(f'{val:.1f}',
                        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

    ax2.set_xticks(x)
    ax2.set_xticklabels(methods, rotation=15, ha='right')
    ax2.grid(axis='y', alpha=0.3)

    # 图3: QPS
    ax3 = axes[2]
    if 'Queries_Per_Second' in df.columns:
        qps = df['Queries_Per_Second'].tolist()
        colors = plt.cm.Greens(np.linspace(0.4, 0.8, len(methods)))
        bars = ax3.bar(x, qps, color=colors)
        ax3.set_ylabel('Queries/Second')
        ax3.set_title('Throughput (QPS)')

        for bar, val in zip(bars, qps):
            ax3.annotate(f'{val:.1f}',
                        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

    ax3.set_xticks(x)
    ax3.set_xticklabels(methods, rotation=15, ha='right')
    ax3.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    output_path = os.path.join(results_dir, 'resource_chart.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"资源消耗图表已保存: {output_path}")

    plt.close()


def plot_weights(results_dir: str, artifacts_dir: str):
    """绘制学习到的权重"""
    import json

    weights_path = os.path.join(artifacts_dir, 'learned_weights.json')
    if not os.path.exists(weights_path):
        print(f"文件不存在: {weights_path}")
        return

    with open(weights_path, 'r') as f:
        weights = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 表排序权重
    ax1 = axes[0]
    table_weights = weights['table_weights']
    labels = list(table_weights.keys())
    values = list(table_weights.values())

    colors = plt.cm.Blues(np.linspace(0.4, 0.8, len(labels)))
    bars = ax1.barh(labels, values, color=colors)
    ax1.set_xlabel('Weight')
    ax1.set_title('Table Ranking Weights')

    for bar, val in zip(bars, values):
        ax1.annotate(f'{val:.3f}',
                    xy=(bar.get_width(), bar.get_y() + bar.get_height()/2),
                    xytext=(3, 0),
                    textcoords="offset points",
                    ha='left', va='center')

    # 字段排序权重
    ax2 = axes[1]
    field_weights = weights['field_weights']
    labels = list(field_weights.keys())
    values = list(field_weights.values())

    colors = plt.cm.Oranges(np.linspace(0.4, 0.8, len(labels)))
    bars = ax2.barh(labels, values, color=colors)
    ax2.set_xlabel('Weight')
    ax2.set_title('Field Ranking Weights')

    for bar, val in zip(bars, values):
        ax2.annotate(f'{val:.3f}',
                    xy=(bar.get_width(), bar.get_y() + bar.get_height()/2),
                    xytext=(3, 0),
                    textcoords="offset points",
                    ha='left', va='center')

    plt.tight_layout()

    output_path = os.path.join(results_dir, 'weights_chart.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"权重图表已保存: {output_path}")

    plt.close()


def main():
    import argparse

    parser = argparse.ArgumentParser(description='结果可视化')
    parser.add_argument('--results', type=str, default=None,
                        help='结果目录')
    parser.add_argument('--artifacts', type=str, default='artifacts',
                        help='artifacts 目录')

    args = parser.parse_args()

    # 查找最新的结果目录
    if args.results:
        results_dir = args.results
    else:
        results_base = Path(__file__).parent.parent / 'results'
        results_dirs = sorted([d for d in results_base.iterdir() if d.is_dir()],
                             key=lambda x: x.name, reverse=True)
        if not results_dirs:
            print("未找到结果目录")
            return
        results_dir = str(results_dirs[0])

    artifacts_dir = Path(__file__).parent.parent / args.artifacts

    print(f"结果目录: {results_dir}")
    print(f"Artifacts 目录: {artifacts_dir}")

    print("\n生成可视化图表...")

    plot_comparison_results(results_dir)
    plot_ablation_results(results_dir)
    plot_cold_start_results(results_dir)
    plot_resource_results(results_dir)
    plot_weights(results_dir, str(artifacts_dir))

    print("\n可视化完成！")


if __name__ == "__main__":
    main()
