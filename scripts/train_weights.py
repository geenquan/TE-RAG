#!/usr/bin/env python
"""
权重训练脚本

使用验证集训练排序权重，支持角色权重优化

使用方式:
    python scripts/train_weights.py --config config.yaml --dev splits/dev.jsonl
    python scripts/train_weights.py --config config.yaml --optimize-role-weights --n-trials 30

输出:
    artifacts/table_ranker.pkl
    artifacts/field_ranker.pkl
    artifacts/learned_weights.json
    artifacts/role_weights.json (如果启用 --optimize-role-weights)
"""

import os
import sys
import json
import argparse
import pandas as pd
import numpy as np
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from terag.config import TERAGConfig
from terag.graph_builder import BipartiteGraphBuilder
from terag.pattern_miner import PatternMiner
from terag.index_builder import IndexBuilder
from terag.feature_extractor import FeatureExtractor
from terag.weight_learner import WeightLearner
from terag.ranker import CombinedRanker


def load_jsonl(path: str) -> pd.DataFrame:
    """加载 JSONL 文件"""
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return pd.DataFrame(data)


def main():
    parser = argparse.ArgumentParser(description='权重训练脚本')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='配置文件路径')
    parser.add_argument('--train', type=str, default=None,
                        help='训练数据路径（覆盖配置）')
    parser.add_argument('--dev', type=str, default=None,
                        help='验证数据路径（用于调参）')
    parser.add_argument('--optimize-role-weights', action='store_true',
                        help='是否优化 SQL 角色权重')
    parser.add_argument('--n-trials', type=int, default=30,
                        help='角色权重优化次数（默认 30）')

    args = parser.parse_args()

    # 加载配置
    config_path = Path(__file__).parent.parent / args.config
    config = TERAGConfig.from_yaml(str(config_path))

    print("=" * 60)
    print("TE-RAG 权重训练")
    print("=" * 60)
    print(f"配置文件: {config_path}")

    # 加载数据
    train_path = args.train or config.get_split_path('train')
    dev_path = args.dev or config.get_split_path('dev')

    train_df = load_jsonl(train_path)
    print(f"训练数据: {len(train_df)} 条")

    dev_df = None
    if os.path.exists(dev_path):
        dev_df = load_jsonl(dev_path)
        print(f"验证数据: {len(dev_df)} 条")

    # 0. 角色权重优化（如果启用）
    if args.optimize_role_weights and dev_df is not None:
        print("\n[0/6] 优化 SQL 角色权重...")
        try:
            from terag.role_weight_optimizer import RoleWeightOptimizer

            optimizer = RoleWeightOptimizer(
                config,
                metric='table_at_5',
                verbose=True
            )

            result = optimizer.optimize(
                train_df,
                dev_df,
                n_trials=args.n_trials
            )

            # 保存优化结果
            role_weights_path = config.get_artifact_path('role_weights.json')
            optimizer.save_results(result, role_weights_path)

            # 更新配置中的角色权重
            config.graph.role_weights = result.best_weights
            print(f"\n角色权重已更新并保存到: {role_weights_path}")

        except ImportError:
            print("Warning: optuna 未安装，跳过角色权重优化")
            print("请运行: pip install optuna")
        except Exception as e:
            print(f"Warning: 角色权重优化失败: {e}")

    # 1. 构建图
    print("\n[1/6] 构建二分图...")
    graph_builder = BipartiteGraphBuilder(config)
    graph = graph_builder.build(train_df)

    if config.output.save_graph:
        graph_builder.save(graph, config.get_artifact_path('graph.pkl'))

    # 2. 挖掘模式
    print("\n[2/6] 挖掘查询模式...")
    element_to_queries = graph_builder.get_element_to_queries(graph, train_df)
    pattern_miner = PatternMiner(config)
    patterns = pattern_miner.mine(element_to_queries)

    if config.output.save_patterns:
        pattern_miner.save(patterns, config.get_artifact_path('patterns.jsonl'))

    # 3. 构建索引
    print("\n[3/6] 构建索引...")
    index_builder = IndexBuilder(config)

    # 加载模式库（如果存在）
    pattern_path = config.get_artifact_path('patterns.jsonl')
    if os.path.exists(pattern_path):
        index_builder.load_patterns(pattern_path)

    index = index_builder.build()

    if config.output.save_index:
        index_builder.save(index, config.get_artifact_path('bm25_index'))

    # 4. 训练权重
    print("\n[4/6] 训练排序权重...")

    # 加载字段表和数据表
    field_df = pd.read_csv(config.data.field_csv)
    table_df = pd.read_csv(config.data.table_csv)

    # 创建特征提取器
    feature_extractor = FeatureExtractor(
        config, graph, index, patterns,
        train_df, field_df, table_df
    )

    # 训练权重
    learner = WeightLearner(config, feature_extractor)
    learner.fit(train_df, dev_df)

    # 5. 保存
    print("\n[5/6] 保存模型...")

    # 保存权重学习器
    learner.save(config.output.artifacts_dir)

    # 保存排序器
    table_ranker = learner.get_table_ranker()
    field_ranker = learner.get_field_ranker()

    combined_ranker = CombinedRanker(config, table_ranker, field_ranker)
    combined_ranker.save(config.output.artifacts_dir)

    # 6. 保存角色统计和码值映射
    print("\n[6/6] 保存角色统计和码值映射...")

    if config.output.save_role_stats:
        graph_builder.save_role_stats(config.get_artifact_path('role_stats.json'))

    # 保存码值映射（用于查询扩展）
    try:
        from terag.code_mapper import CodeMapper
        code_mapper = CodeMapper(config)
        code_mapper.mine_from_sql(train_df)
        code_mapper.save(config.get_artifact_path('code_mappings.json'))
        print("码值映射已保存")
    except Exception as e:
        print(f"Warning: 码值映射保存失败: {e}")

    print("\n" + "=" * 60)
    print("权重训练完成！")
    print("=" * 60)
    print(f"\n学习到的权重:")
    print(f"  表排序: {learner.get_table_weights()}")
    print(f"  字段排序: {learner.get_field_weights()}")

    print(f"\n产物保存在: {config.output.artifacts_dir}/")
    print("  - graph.pkl")
    print("  - patterns.jsonl")
    print("  - bm25_index/")
    print("  - table_ranker.pkl")
    print("  - field_ranker.pkl")
    print("  - learned_weights.json")
    print("  - role_stats.json")
    print("  - code_mappings.json")
    if args.optimize_role_weights:
        print("  - role_weights.json")


if __name__ == "__main__":
    main()
