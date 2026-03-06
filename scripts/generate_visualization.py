#!/usr/bin/env python
"""
生成实验结果可视化图表
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def plot_comparison_results(df, output_dir):
    """绘制对比实验结果"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    methods = df['Method'].values
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']

    # Table Accuracy
    ax1 = axes[0]
    metrics = ['Table@1', 'Table@3', 'Table@5', 'Table@10']
    x = np.arange(len(methods))
    width = 0.2

    for i, metric in enumerate(metrics):
        values = df[metric].values * 100
        ax1.bar(x + i * width, values, width, label=metric, color=colors[i], alpha=0.8)

    ax1.set_xlabel('Method', fontsize=12)
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_title('Table Selection Accuracy', fontsize=14, fontweight='bold')
    ax1.set_xticks(x + 1.5 * width)
    ax1.set_xticklabels(methods, rotation=0)
    ax1.legend(loc='upper left')
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim(0, 100)

    # Field Coverage & SQL Metrics
    ax2 = axes[1]
    metrics2 = ['Field_Coverage', 'SQL_Parse_Rate']
    x = np.arange(len(methods))
    width = 0.35

    for i, metric in enumerate(metrics2):
        values = df[metric].values * 100
        ax2.bar(x + i * width - width/2, values, width, label=metric, color=colors[i], alpha=0.8)

    ax2.set_xlabel('Method', fontsize=12)
    ax2.set_ylabel('Rate (%)', fontsize=12)
    ax2.set_title('Field Coverage & SQL Parse Rate', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods, rotation=0)
    ax2.legend(loc='upper right')
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comparison_chart.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/comparison_chart.png")


def plot_ablation_results(df, output_dir):
    """绘制消融实验结果"""
    fig, ax = plt.subplots(figsize=(12, 6))

    configs = df['Configuration'].values
    metrics = ['Table@1', 'Table@3', 'Table@5', 'Table@10']
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']

    x = np.arange(len(configs))
    width = 0.2

    for i, metric in enumerate(metrics):
        values = df[metric].values * 100
        bars = ax.bar(x + i * width, values, width, label=metric, color=colors[i], alpha=0.8)

    # 添加基线
    baseline = df[df['Configuration'] == 'Full TE-RAG']['Table@5'].values[0] * 100
    ax.axhline(y=baseline, color='red', linestyle='--', alpha=0.5, label='Full TE-RAG (Table@5)')

    ax.set_xlabel('Configuration', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Ablation Study Results', fontsize=14, fontweight='bold')
    ax.set_xticks(x + 1.5 * width)
    ax.set_xticklabels(configs, rotation=30, ha='right')
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ablation_chart.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/ablation_chart.png")


def plot_cold_start_results(df, output_dir):
    """绘制冷启动实验结果"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    methods = df['Method'].values
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']

    # Table@5
    ax1 = axes[0]
    x = np.arange(len(methods))
    values = df['Table@5'].values * 100
    bars = ax1.bar(x, values, 0.6, color=colors, alpha=0.8)
    ax1.set_xlabel('Method', fontsize=12)
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_title('Cold Start - Table@5', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, rotation=0)
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim(0, 100)

    for bar, val in zip(bars, values):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{val:.1f}%', ha='center', va='bottom', fontsize=10)

    # Field Coverage
    ax2 = axes[1]
    values = df['Field_Coverage'].values * 100
    bars = ax2.bar(x, values, 0.6, color=colors, alpha=0.8)
    ax2.set_xlabel('Method', fontsize=12)
    ax2.set_ylabel('Coverage (%)', fontsize=12)
    ax2.set_title('Cold Start - Field Coverage', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods, rotation=0)
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_ylim(0, 100)

    for bar, val in zip(bars, values):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{val:.1f}%', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cold_start_chart.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/cold_start_chart.png")


def plot_resource_results(df, output_dir):
    """绘制资源消耗对比"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    methods = df['Method'].values
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']

    # Query Time
    ax1 = axes[0]
    x = np.arange(len(methods))
    values = df['Avg_Query_Time(ms)'].values
    bars = ax1.bar(x, values, 0.6, color=colors, alpha=0.8)
    ax1.set_xlabel('Method', fontsize=12)
    ax1.set_ylabel('Query Time (ms)', fontsize=12)
    ax1.set_title('Query Latency', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, rotation=0)
    ax1.grid(axis='y', alpha=0.3)

    for bar, val in zip(bars, values):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{val:.1f}', ha='center', va='bottom', fontsize=10)

    # Queries Per Second
    ax2 = axes[1]
    values = df['Queries_Per_Second'].values
    bars = ax2.bar(x, values, 0.6, color=colors, alpha=0.8)
    ax2.set_xlabel('Method', fontsize=12)
    ax2.set_ylabel('Queries / Second', fontsize=12)
    ax2.set_title('Throughput', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods, rotation=0)
    ax2.grid(axis='y', alpha=0.3)

    for bar, val in zip(bars, values):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{val:.2f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'resource_chart.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/resource_chart.png")


def main():
    # 找到最新的结果目录
    results_base = Path(__file__).parent.parent / 'results'
    result_dirs = sorted([d for d in results_base.iterdir() if d.is_dir() and d.name.startswith('2026')],
                        reverse=True)

    if not result_dirs:
        print("No result directories found!")
        return

    output_dir = result_dirs[0]
    print(f"Processing results from: {output_dir}")

    # 加载数据
    comparison_path = output_dir / 'comparison_results.csv'
    ablation_path = output_dir / 'ablation_results.csv'
    cold_start_path = output_dir / 'cold_start_results.csv'
    resource_path = output_dir / 'resource_results.csv'

    # 生成图表
    if comparison_path.exists():
        df = pd.read_csv(comparison_path)
        plot_comparison_results(df, str(output_dir))

    if ablation_path.exists():
        df = pd.read_csv(ablation_path)
        plot_ablation_results(df, str(output_dir))

    if cold_start_path.exists():
        df = pd.read_csv(cold_start_path)
        plot_cold_start_results(df, str(output_dir))

    if resource_path.exists():
        df = pd.read_csv(resource_path)
        plot_resource_results(df, str(output_dir))

    print(f"\nAll visualizations saved to: {output_dir}")


if __name__ == "__main__":
    main()
