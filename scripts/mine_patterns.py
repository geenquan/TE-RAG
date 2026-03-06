#!/usr/bin/env python
"""
模式挖掘脚本

使用方式:
    python scripts/mine_patterns.py --train splits/train.jsonl
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
from terag.pattern_miner import PatternMiner


def load_jsonl(path: str) -> pd.DataFrame:
    """加载 JSONL 文件"""
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return pd.DataFrame(data)


def main():
    parser = argparse.ArgumentParser(description='模式挖掘脚本')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='配置文件路径')
    parser.add_argument('--train', type=str, default=None,
                        help='训练数据路径')

    args = parser.parse_args()

    # 加载配置
    config_path = Path(__file__).parent.parent / args.config
    config = TERAGConfig.from_yaml(str(config_path))

    print("=" * 60)
    print("查询模式挖掘")
    print("=" * 60)

    # 加载训练数据
    train_path = args.train or config.get_split_path('train')
    train_df = load_jsonl(train_path)
    print(f"训练数据: {len(train_df)} 条")

    # 先构建图
    print("\n构建二分图...")
    graph_builder = BipartiteGraphBuilder(config)
    graph = graph_builder.build(train_df)

    # 获取元素到查询的映射
    element_to_queries = graph_builder.get_element_to_queries(graph, train_df)
    print(f"元素数量: {len(element_to_queries)}")

    # 挖掘模式
    print("\n挖掘查询模式...")
    miner = PatternMiner(config)
    patterns = miner.mine(element_to_queries)

    # 打印统计
    stats = miner.get_pattern_stats(patterns)
    print(f"\n模式统计:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # 保存
    if config.output.save_patterns:
        output_path = config.get_artifact_path('patterns.jsonl')
        miner.save(patterns, output_path)

    print("\n模式挖掘完成！")


if __name__ == "__main__":
    main()
