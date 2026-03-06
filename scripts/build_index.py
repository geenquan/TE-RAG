#!/usr/bin/env python
"""
索引构建脚本

使用方式:
    python scripts/build_index.py
"""

import os
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from terag.config import TERAGConfig
from terag.index_builder import IndexBuilder


def main():
    parser = argparse.ArgumentParser(description='索引构建脚本')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='配置文件路径')

    args = parser.parse_args()

    # 加载配置
    config_path = Path(__file__).parent.parent / args.config
    config = TERAGConfig.from_yaml(str(config_path))

    print("=" * 60)
    print("BM25 索引构建")
    print("=" * 60)

    # 构建索引
    builder = IndexBuilder(config)

    # 尝试加载模式库
    pattern_path = config.get_artifact_path('patterns.jsonl')
    if os.path.exists(pattern_path):
        builder.load_patterns(pattern_path)

    index = builder.build()

    # 保存
    if config.output.save_index:
        output_dir = config.get_artifact_path('bm25_index')
        builder.save(index, output_dir)

    # 测试检索
    print("\n测试检索:")
    from terag.index_builder import BM25Retriever
    retriever = BM25Retriever(index)

    test_queries = [
        "查询公司的售电量",
        "统计用户的电费金额",
    ]

    for query in test_queries:
        print(f"\n查询: {query}")
        results = retriever.search(query, k=3)
        for doc_id, score in results:
            print(f"  {doc_id}: {score:.4f}")

    print("\n索引构建完成！")


if __name__ == "__main__":
    main()
