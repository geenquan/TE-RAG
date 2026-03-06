#!/usr/bin/env python
"""
数据准备脚本

功能：
1. 加载 CSV 数据
2. 划分 train/dev/test (70%/10%/20%)
3. 固定随机种子
4. 保存到 splits/*.jsonl

使用方式:
    python scripts/prepare_data.py --config config.yaml --seed 42

输出:
    splits/train.jsonl
    splits/dev.jsonl
    splits/test.jsonl
"""

import os
import sys
import json
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from terag.config import TERAGConfig


def load_data(config: TERAGConfig) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    加载数据

    Returns:
        (field_df, table_df, qa_df)
    """
    field_df = pd.read_csv(config.data.field_csv)
    table_df = pd.read_csv(config.data.table_csv)
    qa_df = pd.read_csv(config.data.qa_csv)

    print(f"加载数据:")
    print(f"  字段表: {len(field_df)} 行")
    print(f"  数据表: {len(table_df)} 行")
    print(f"  QA 数据: {len(qa_df)} 行")

    return field_df, table_df, qa_df


def split_data(
    qa_df: pd.DataFrame,
    train_ratio: float = 0.70,
    dev_ratio: float = 0.10,
    test_ratio: float = 0.20,
    seed: int = 42,
    stratify_by_table: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    划分数据

    Args:
        qa_df: QA 数据
        train_ratio: 训练集比例
        dev_ratio: 验证集比例
        test_ratio: 测试集比例
        seed: 随机种子
        stratify_by_table: 是否按表分层划分

    Returns:
        (train_df, dev_df, test_df)
    """
    assert abs(train_ratio + dev_ratio + test_ratio - 1.0) < 1e-6, \
        "比例之和必须为 1"

    np.random.seed(seed)

    if stratify_by_table:
        # 按表分层划分
        train_list = []
        dev_list = []
        test_list = []

        for table in qa_df['table'].unique():
            table_data = qa_df[qa_df['table'] == table].copy()
            n = len(table_data)

            # 随机打乱
            indices = np.random.permutation(n)

            n_train = int(n * train_ratio)
            n_dev = int(n * dev_ratio)

            train_list.append(table_data.iloc[indices[:n_train]])
            dev_list.append(table_data.iloc[indices[n_train:n_train + n_dev]])
            test_list.append(table_data.iloc[indices[n_train + n_dev:]])

        train_df = pd.concat(train_list, ignore_index=True)
        dev_df = pd.concat(dev_list, ignore_index=True)
        test_df = pd.concat(test_list, ignore_index=True)
    else:
        # 随机划分
        n = len(qa_df)
        indices = np.random.permutation(n)

        n_train = int(n * train_ratio)
        n_dev = int(n * dev_ratio)

        train_df = qa_df.iloc[indices[:n_train]].copy()
        dev_df = qa_df.iloc[indices[n_train:n_train + n_dev]].copy()
        test_df = qa_df.iloc[indices[n_train + n_dev:]].copy()

    print(f"\n数据划分 (seed={seed}):")
    print(f"  训练集: {len(train_df)} 条 ({len(train_df)/len(qa_df)*100:.1f}%)")
    print(f"  验证集: {len(dev_df)} 条 ({len(dev_df)/len(qa_df)*100:.1f}%)")
    print(f"  测试集: {len(test_df)} 条 ({len(test_df)/len(qa_df)*100:.1f}%)")

    # 统计每个集合的表分布
    print(f"\n表分布:")
    for name, df in [("训练集", train_df), ("验证集", dev_df), ("测试集", test_df)]:
        tables = df['table'].nunique()
        print(f"  {name}: {tables} 个表")

    return train_df, dev_df, test_df


def save_to_jsonl(df: pd.DataFrame, output_path: str):
    """
    保存为 JSONL 格式

    Args:
        df: 数据框
        output_path: 输出路径
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        for _, row in df.iterrows():
            record = {
                'question': row.get('question', ''),
                'table': row.get('table', ''),
                'field': row.get('field', ''),
                'sql': row.get('sql', ''),
            }
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    print(f"  保存: {output_path} ({len(df)} 条)")


def save_split_info(
    train_df: pd.DataFrame,
    dev_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_path: str,
    seed: int,
    config: TERAGConfig
):
    """保存划分信息"""
    info = {
        'seed': seed,
        'train_ratio': config.data.train_ratio,
        'dev_ratio': config.data.dev_ratio,
        'test_ratio': config.data.test_ratio,
        'train_size': len(train_df),
        'dev_size': len(dev_df),
        'test_size': len(test_df),
        'train_tables': train_df['table'].unique().tolist(),
        'dev_tables': dev_df['table'].unique().tolist(),
        'test_tables': test_df['table'].unique().tolist(),
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    print(f"  保存划分信息: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='数据准备脚本')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='配置文件路径')
    parser.add_argument('--seed', type=int, default=None,
                        help='随机种子（覆盖配置文件）')
    parser.add_argument('--no-stratify', action='store_true',
                        help='不按表分层划分')

    args = parser.parse_args()

    # 加载配置
    config_path = Path(__file__).parent.parent / args.config
    config = TERAGConfig.from_yaml(str(config_path))

    # 覆盖种子
    seed = args.seed if args.seed is not None else config.seed

    print("=" * 60)
    print("TE-RAG 数据准备")
    print("=" * 60)
    print(f"配置文件: {config_path}")
    print(f"随机种子: {seed}")

    # 加载数据
    field_df, table_df, qa_df = load_data(config)

    # 划分数据
    train_df, dev_df, test_df = split_data(
        qa_df,
        train_ratio=config.data.train_ratio,
        dev_ratio=config.data.dev_ratio,
        test_ratio=config.data.test_ratio,
        seed=seed,
        stratify_by_table=not args.no_stratify
    )

    # 保存
    print(f"\n保存数据到 {config.data.splits_dir}/:")

    save_to_jsonl(train_df, config.get_split_path('train'))
    save_to_jsonl(dev_df, config.get_split_path('dev'))
    save_to_jsonl(test_df, config.get_split_path('test'))

    # 保存划分信息
    save_split_info(
        train_df, dev_df, test_df,
        os.path.join(config.data.splits_dir, 'split_info.json'),
        seed, config
    )

    print("\n数据准备完成！")


if __name__ == "__main__":
    main()
