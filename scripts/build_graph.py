#!/usr/bin/env python
"""
图构建脚本

使用方式:
    python scripts/build_graph.py --train splits/train.jsonl
"""

import os
import sys
import json
import argparse
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from terag.config import TERAGConfig
from terag.graph_builder import BipartiteGraphBuilder


def load_jsonl(path: str) -> pd.DataFrame:
    """加载 JSONL 文件"""
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return pd.DataFrame(data)


def main():
    parser = argparse.ArgumentParser(description='图构建脚本')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='配置文件路径')
    parser.add_argument('--train', type=str, default=None,
                        help='训练数据路径')

    args = parser.parse_args()

    # 加载配置
    config_path = Path(__file__).parent.parent / args.config
    config = TERAGConfig.from_yaml(str(config_path))

    print("=" * 60)
    print("二分图构建")
    print("=" * 60)

    # 加载训练数据
    train_path = args.train or config.get_split_path('train')
    train_df = load_jsonl(train_path)
    print(f"训练数据: {len(train_df)} 条")

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
        output_path = config.get_artifact_path('graph.pkl')
        builder.save(graph, output_path)

    if config.output.save_role_stats:
        output_path = config.get_artifact_path('role_stats.json')
        builder.save_role_stats(output_path)

    print("\n图构建完成！")


if __name__ == "__main__":
    main()
