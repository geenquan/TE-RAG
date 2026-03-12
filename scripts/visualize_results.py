#!/usr/bin/env python
"""
结果可视化脚本（完整版）

使用 Nature 风格科研调色板
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# 设置中文字体和科研风格
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 11

# ============================================================
# Nature 风格科研调色板 (9色渐变: 蓝→青→橙→粉→红→深红)
# ============================================================
NATURE_PALETTE = [
    "#4E74B4",  # 深蓝 (baseline)
    "#96D1E0",  # 浅蓝
    "#E9A375",  # 橙色
    "#FEE1D4",  # 浅粉
    "#FDBCA7",  # 粉橙
    "#FB9680",  # 珊瑚橙
    "#EB6E68",  # 红 (ours)
    "#C8595D",  # 深红 (ours - 主要方法)
    "#954D55",  # 酒红
]

# 方法颜色映射 - 统一使用 Nature 调色板
# baseline 用冷色 (蓝/青/橙系), ours 用暖色 (红系)
METHOD_COLORS = {
    # Baseline 方法 (冷色系)
    'BM25': '#4E74B4',       # 深蓝
    'Vector': '#96D1E0',     # 浅蓝
    'Hybrid': '#E9A375',     # 橙色
    'LLM': '#FDBCA7',        # 粉橙
    'RESDSQL': '#FEE1D4',    # 浅粉
    'Graph': '#FB9680',      # 珊瑚橙

    # 我们的方法 (暖色/红系 - 突出显示)
    'TE-RAG': '#EB6E68',     # 红色
    'TE-RAG-V2': '#C8595D',  # 深红 (主要方法)

    # 消融实验 (同色系渐变: Full最深, 去掉模块变浅)
    'Full TE-RAG': '#C8595D',           # 深红 (最完整)
    'w/o Graph Weight': '#FB9680',      # 珊瑚橙
    'w/o Template Mining': '#FDBCA7',   # 粉橙
    'w/o Pattern Generalization': '#FEE1D4',  # 浅粉
    'w/o Enhanced Index': '#E9A375',    # 橙色
}

# 我们的方法列表 (用于特殊样式处理)
OURS_METHODS = ['TE-RAG', 'TE-RAG-V2', 'Full TE-RAG']


def get_method_color(method_name: str) -> str:
    """获取方法的颜色"""
    if method_name in METHOD_COLORS:
        return METHOD_COLORS[method_name]
    # 对于未知方法，返回基于名称哈希的颜色
    import hashlib
    hash_val = int(hashlib.md5(method_name.encode()).hexdigest()[:6], 16)
    return f'#{hash_val:06x}'


def is_ours_method(method_name: str) -> bool:
    """判断是否是我们的方法"""
    return any(ours in method_name for ours in OURS_METHODS)


def plot_comparison_results(results_dir: str):
    """绘制对比实验结果 - Nature 风格简化版"""
    comparison_path = os.path.join(results_dir, 'comparison_results.csv')
    if not os.path.exists(comparison_path):
        print(f"文件不存在: {comparison_path}")
        return

    df = pd.read_csv(comparison_path)

    # 创建图表 - 只用两个子图
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    methods = df['Method'].tolist()
    x = np.arange(len(methods))
    width = 0.35

    # 图1: Table Accuracy vs SQL Accuracy
    ax1 = axes[0]
    table_acc = df['Table Accuracy'].tolist() if 'Table Accuracy' in df.columns else [0] * len(methods)
    sql_acc = df['SQL Accuracy'].tolist() if 'SQL Accuracy' in df.columns else [0] * len(methods)

    # 为每个方法单独绘制柱状图，区分 ours 和 baseline
    for i, method in enumerate(methods):
        color = get_method_color(method)
        is_ours = is_ours_method(method)

        # Table Accuracy 柱
        bar1 = ax1.bar(x[i] - width/2, table_acc[i], width,
                       color=color,
                       alpha=1.0 if is_ours else 0.8,
                       edgecolor='black' if is_ours else 'none',
                       linewidth=1.5 if is_ours else 0)

        # SQL Accuracy 柱 (带斜线)
        bar2 = ax1.bar(x[i] + width/2, sql_acc[i], width,
                       color=color,
                       alpha=0.6 if is_ours else 0.5,
                       edgecolor='black' if is_ours else 'none',
                       linewidth=1.5 if is_ours else 0,
                       hatch='//')

    # 添加数值标签
    for i, (t, s) in enumerate(zip(table_acc, sql_acc)):
        ax1.annotate(f'{t:.0%}', xy=(x[i] - width/2, t),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
        ax1.annotate(f'{s:.0%}', xy=(x[i] + width/2, s),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

    ax1.set_xlabel('Method', fontsize=11)
    ax1.set_ylabel('Accuracy', fontsize=11)
    ax1.set_title('Table & SQL Accuracy Comparison', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, rotation=15, ha='right')
    ax1.legend(['Table Accuracy', 'SQL Accuracy'], loc='upper right')
    ax1.set_ylim(0, 1.0)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')

    # 图2: Top-K Table Recall
    ax2 = axes[1]
    width_k = 0.25

    top1 = df['Top1 Table Recall'].tolist() if 'Top1 Table Recall' in df.columns else [0] * len(methods)
    top3 = df['Top3 Table Recall'].tolist() if 'Top3 Table Recall' in df.columns else [0] * len(methods)
    top5 = df['Top5 Table Recall'].tolist() if 'Top5 Table Recall' in df.columns else [0] * len(methods)

    # 使用统一的颜色
    top1_color = '#4E74B4'  # 深蓝
    top3_color = '#96D1E0'  # 浅蓝
    top5_color = '#E9A375'  # 橙色

    bars1 = ax2.bar(x - width_k, top1, width_k, label='Top-1', color=top1_color, alpha=0.9)
    bars2 = ax2.bar(x, top3, width_k, label='Top-3', color=top3_color, alpha=0.9)
    bars3 = ax2.bar(x + width_k, top5, width_k, label='Top-5', color=top5_color, alpha=0.9)

    # 为 ours 方法添加黑边
    for i, method in enumerate(methods):
        if is_ours_method(method):
            ax2.bar(x[i] - width_k, top1[i], width_k, color=top1_color, alpha=0.9,
                   edgecolor='black', linewidth=1.5)
            ax2.bar(x[i], top3[i], width_k, color=top3_color, alpha=0.9,
                   edgecolor='black', linewidth=1.5)
            ax2.bar(x[i] + width_k, top5[i], width_k, color=top5_color, alpha=0.9,
                   edgecolor='black', linewidth=1.5)

    # 添加数值标签
    for i, (t1, t3, t5) in enumerate(zip(top1, top3, top5)):
        ax2.annotate(f'{t1:.0%}', xy=(x[i] - width_k, t1),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)
        ax2.annotate(f'{t3:.0%}', xy=(x[i], t3),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)
        ax2.annotate(f'{t5:.0%}', xy=(x[i] + width_k, t5),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

    ax2.set_xlabel('Method', fontsize=11)
    ax2.set_ylabel('Recall', fontsize=11)
    ax2.set_title('Table Retrieval Recall @K', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods, rotation=15, ha='right')
    ax2.legend(loc='upper right')
    ax2.set_ylim(0, 1.0)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()

    output_path = os.path.join(results_dir, 'comparison_chart.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"对比图表已保存: {output_path}")

    plt.close()


def plot_ablation_results(results_dir: str):
    """绘制消融实验结果 - 使用渐变色系"""
    ablation_path = os.path.join(results_dir, 'ablation_results.csv')
    if not os.path.exists(ablation_path):
        print(f"文件不存在: {ablation_path}")
        return

    df = pd.read_csv(ablation_path)

    if len(df) == 0:
        print("消融实验结果为空")
        return

    # 创建图表
    fig, ax = plt.subplots(figsize=(12, 5))

    # 兼容不同的列名格式
    config_col = 'Configuration' if 'Configuration' in df.columns else 'Method'
    configs = df[config_col].tolist()

    table_col = 'Table Accuracy' if 'Table Accuracy' in df.columns else 'Table@5'
    sql_col = 'SQL Accuracy' if 'SQL Accuracy' in df.columns else 'Field_Coverage'

    table_acc = df[table_col].tolist() if table_col in df.columns else [0] * len(configs)
    sql_acc = df[sql_col].tolist() if sql_col in df.columns else [0] * len(configs)

    x = np.arange(len(configs))
    width = 0.35

    # 消融实验渐变色: Full 最深(红), 去掉模块变浅
    ablation_colors = [
        '#C8595D',  # Full TE-RAG - 深红 (最深)
        '#FB9680',  # w/o Graph - 珊瑚橙
        '#FDBCA7',  # w/o Template - 粉橙
        '#FEE1D4',  # w/o Pattern - 浅粉
        '#E9A375',  # w/o Index - 橙色
    ]

    # 为每个配置分配颜色
    config_colors = []
    for config in configs:
        if 'Full' in config:
            config_colors.append(ablation_colors[0])
        elif 'Graph' in config:
            config_colors.append(ablation_colors[1])
        elif 'Template' in config:
            config_colors.append(ablation_colors[2])
        elif 'Pattern' in config:
            config_colors.append(ablation_colors[3])
        elif 'Index' in config or 'Enhanced' in config:
            config_colors.append(ablation_colors[4])
        else:
            config_colors.append('#888888')

    # 绘制柱状图
    for i, config in enumerate(configs):
        is_full = 'Full' in config

        # Table Accuracy
        bar1 = ax.bar(x[i] - width/2, table_acc[i], width,
                     label='Table Accuracy' if i == 0 else '',
                     color=config_colors[i],
                     alpha=1.0 if is_full else 0.85,
                     edgecolor='black' if is_full else 'none',
                     linewidth=1.5 if is_full else 0)

        # SQL Accuracy
        bar2 = ax.bar(x[i] + width/2, sql_acc[i], width,
                     label='SQL Accuracy' if i == 0 else '',
                     color=config_colors[i],
                     alpha=0.6 if is_full else 0.5,
                     edgecolor='black' if is_full else 'none',
                     linewidth=1.5 if is_full else 0,
                     hatch='//')

    # 添加数值标签
    for i, (t, s) in enumerate(zip(table_acc, sql_acc)):
        ax.annotate(f'{t:.0%}', xy=(x[i] - width/2, t),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold' if 'Full' in configs[i] else 'normal')
        ax.annotate(f'{s:.0%}', xy=(x[i] + width/2, s),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold' if 'Full' in configs[i] else 'normal')

    # 添加 Full model 的基准线
    if table_acc:
        full_idx = next((i for i, c in enumerate(configs) if 'Full' in c), None)
        if full_idx is not None:
            ax.axhline(y=table_acc[full_idx], color='#C8595D', linestyle='--',
                      alpha=0.5, linewidth=1.5, label='Full Model Baseline')

    ax.set_xlabel('Configuration', fontsize=11)
    ax.set_ylabel('Accuracy', fontsize=11)
    ax.set_title('TE-RAG Ablation Study', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=25, ha='right')
    ax.legend(['Table Accuracy', 'SQL Accuracy'], loc='upper right')
    ax.set_ylim(0, 0.85)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # 添加图例说明颜色深浅含义
    ax.text(0.02, 0.98, 'Darker = More Complete Model',
            transform=ax.transAxes, fontsize=9, verticalalignment='top',
            style='italic', alpha=0.7)

    plt.tight_layout()

    output_path = os.path.join(results_dir, 'ablation_chart.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"消融图表已保存: {output_path}")

    plt.close()


def plot_cold_start_results(results_dir: str):
    """绘制冷启动实验结果 - Nature 风格"""
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

    # 图1: Table Accuracy vs SQL Accuracy (冷启动)
    ax1 = axes[0]
    table_col = 'Table Accuracy' if 'Table Accuracy' in df.columns else 'Table@5'
    sql_col = 'SQL Accuracy' if 'SQL Accuracy' in df.columns else 'Field_Coverage'

    table_acc = df[table_col].tolist() if table_col in df.columns else [0] * len(methods)
    sql_acc = df[sql_col].tolist() if sql_col in df.columns else [0] * len(methods)

    # 为每个方法单独绘制，区分 ours
    for i, method in enumerate(methods):
        color = get_method_color(method)
        is_ours = is_ours_method(method)

        # Table Accuracy
        ax1.bar(x[i] - width/2, table_acc[i], width,
               color=color,
               alpha=1.0 if is_ours else 0.8,
               edgecolor='black' if is_ours else 'none',
               linewidth=1.5 if is_ours else 0)

        # SQL Accuracy
        ax1.bar(x[i] + width/2, sql_acc[i], width,
               color=color,
               alpha=0.6 if is_ours else 0.5,
               edgecolor='black' if is_ours else 'none',
               linewidth=1.5 if is_ours else 0,
               hatch='//')

    # 添加数值标签
    for i, (t, s) in enumerate(zip(table_acc, sql_acc)):
        ax1.annotate(f'{t:.0%}', xy=(x[i] - width/2, t),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
        ax1.annotate(f'{s:.0%}', xy=(x[i] + width/2, s),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

    ax1.set_xlabel('Method', fontsize=11)
    ax1.set_ylabel('Accuracy', fontsize=11)
    ax1.set_title('Cold Start Performance', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, rotation=15, ha='right')
    ax1.legend(['Table Accuracy', 'SQL Accuracy'], loc='upper right')
    ax1.set_ylim(0, 0.65)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')

    # 图2: 与常规性能对比
    ax2 = axes[1]

    comparison_path = os.path.join(results_dir, 'comparison_results.csv')
    if os.path.exists(comparison_path):
        comp_df = pd.read_csv(comparison_path)

        # 对比冷启动 vs 常规
        regular_table = []
        cold_table = []

        for method in methods:
            cold_row = df[df['Method'] == method]
            cold_table.append(cold_row[table_col].values[0] if len(cold_row) > 0 and table_col in cold_row.columns else 0)

            reg_row = comp_df[comp_df['Method'] == method]
            regular_table.append(reg_row[table_col].values[0] if len(reg_row) > 0 and table_col in reg_row.columns else 0)

        # 为每个方法绘制对比柱
        for i, method in enumerate(methods):
            color = get_method_color(method)
            is_ours = is_ours_method(method)

            # Regular
            ax2.bar(x[i] - width/2, regular_table[i], width,
                   color=color,
                   alpha=0.9 if is_ours else 0.7,
                   edgecolor='black' if is_ours else 'none',
                   linewidth=1.5 if is_ours else 0)

            # Cold Start (带斜线)
            ax2.bar(x[i] + width/2, cold_table[i], width,
                   color=color,
                   alpha=0.6 if is_ours else 0.4,
                   edgecolor='black' if is_ours else 'none',
                   linewidth=1.5 if is_ours else 0,
                   hatch='//')

        # 添加数值标签
        for i, (reg_val, cold_val) in enumerate(zip(regular_table, cold_table)):
            ax2.annotate(f'{reg_val:.0%}', xy=(x[i] - width/2, reg_val),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)
            ax2.annotate(f'{cold_val:.0%}', xy=(x[i] + width/2, cold_val),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

        ax2.set_xlabel('Method', fontsize=11)
        ax2.set_ylabel('Table Accuracy', fontsize=11)
        ax2.set_title('Cold Start vs Regular Performance', fontsize=12, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(methods, rotation=15, ha='right')
        ax2.legend(['Regular', 'Cold Start'], loc='upper right')
        ax2.set_ylim(0, 1.0)
        ax2.grid(axis='y', alpha=0.3, linestyle='--')
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
