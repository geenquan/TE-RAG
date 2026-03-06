"""
TE-RAG CLI 入口

提供统一的命令行接口，支持:
1. run_id 机制：每次运行生成唯一标识
2. artifacts 隔离：每个 run_id 有独立的 artifacts 目录
3. 子命令：run, prepare, build, evaluate, optimize-weights

使用方式:
    python -m terag.cli run --suite all --config config.yaml --seed 42
    python -m terag.cli prepare --config config.yaml
    python -m terag.cli build --config config.yaml
    python -m terag.cli evaluate --config config.yaml --run_id <run_id>
"""

import os
import sys
import json
import yaml
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field


@dataclass
class RunContext:
    """运行上下文"""
    run_id: str
    root_dir: Path
    artifacts_dir: Path
    results_dir: Path
    config_path: Path
    resolved_config_path: Path

    @property
    def ablation_dir(self) -> Path:
        """消融实验目录"""
        return self.artifacts_dir / "ablation"


class RunManager:
    """
    运行管理器

    负责:
    - 生成唯一 run_id
    - 创建目录结构
    - 保存 resolved config
    - 管理 artifacts 隔离
    """

    def __init__(self, config_path: str, run_id: Optional[str] = None):
        """
        初始化运行管理器

        Args:
            config_path: 配置文件路径
            run_id: 可选的 run_id（不提供则自动生成）
        """
        self.config_path = Path(config_path).resolve()
        self.root_dir = self.config_path.parent

        # 生成或使用 run_id
        if run_id:
            self.run_id = run_id
        else:
            self.run_id = self._generate_run_id()

        # 设置目录
        self.artifacts_dir = self.root_dir / "artifacts" / self.run_id
        self.results_dir = self.root_dir / "results" / self.run_id
        self.resolved_config_path = self.results_dir / "config_resolved.yaml"

    def _generate_run_id(self) -> str:
        """生成唯一 run_id"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 获取 git hash
        git_hash = self._get_git_hash()

        return f"{timestamp}_{git_hash}"

    def _get_git_hash(self) -> str:
        """获取当前 git commit hash 的前 7 位"""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=self.root_dir,
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
        return "unknown"

    def setup(self) -> RunContext:
        """
        设置运行环境

        创建必要的目录结构
        """
        # 创建目录
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        (self.artifacts_dir / "ablation").mkdir(parents=True, exist_ok=True)

        # 保存 resolved config
        self._save_resolved_config()

        return RunContext(
            run_id=self.run_id,
            root_dir=self.root_dir,
            artifacts_dir=self.artifacts_dir,
            results_dir=self.results_dir,
            config_path=self.config_path,
            resolved_config_path=self.resolved_config_path
        )

    def _save_resolved_config(self):
        """保存 resolved 配置"""
        import shutil

        # 复制原始配置
        shutil.copy(self.config_path, self.resolved_config_path)

        # 追加 run_id 信息
        with open(self.resolved_config_path, 'a', encoding='utf-8') as f:
            f.write(f"\n# Run metadata\n")
            f.write(f"run_id: \"{self.run_id}\"\n")
            f.write(f"run_timestamp: \"{datetime.now().isoformat()}\"\n")

    def get_context(self) -> RunContext:
        """获取当前运行上下文"""
        return RunContext(
            run_id=self.run_id,
            root_dir=self.root_dir,
            artifacts_dir=self.artifacts_dir,
            results_dir=self.results_dir,
            config_path=self.config_path,
            resolved_config_path=self.resolved_config_path
        )


class AblationManager:
    """
    消融实验管理器

    负责为每个消融配置创建独立的 artifacts 目录
    """

    ABLATION_CONFIGS = {
        'full': {},
        'no_graph_weight': {'use_graph_weight': False},
        'no_template_mining': {'use_template_mining': False},
        'no_pattern_generalization': {'use_pattern_generalization': False},
        'no_enhanced_index': {'use_enhanced_index': False},
        'no_role_parser': {'use_role_parser': False},
    }

    def __init__(self, run_context: RunContext):
        self.context = run_context

    def get_ablation_dir(self, ablation_name: str) -> Path:
        """获取消融实验目录"""
        ablation_dir = self.context.ablation_dir / ablation_name
        ablation_dir.mkdir(parents=True, exist_ok=True)
        return ablation_dir

    def get_ablation_config(self, ablation_name: str) -> Dict[str, Any]:
        """获取消融配置"""
        return self.ABLATION_CONFIGS.get(ablation_name, {})

    def list_ablations(self) -> List[str]:
        """列出所有消融配置"""
        return list(self.ABLATION_CONFIGS.keys())


def cmd_prepare(args):
    """准备数据（划分 train/dev/test）"""
    from terag.config import TERAGConfig

    print("=" * 60)
    print("TE-RAG 数据准备")
    print("=" * 60)

    config = TERAGConfig.from_yaml(args.config)

    # 导入并运行数据准备脚本
    sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

    from scripts.prepare_data import prepare_splits

    prepare_splits(config)

    print(f"\n数据准备完成，保存到: {config.data.splits_dir}")


def cmd_build(args):
    """构建 artifacts（graph, index, patterns）"""
    import pandas as pd
    from terag.config import TERAGConfig
    from terag.graph_builder import BipartiteGraphBuilder
    from terag.pattern_miner import PatternMiner
    from terag.index_builder import IndexBuilder

    print("=" * 60)
    print("TE-RAG 构建 Artifacts")
    print("=" * 60)

    # 设置运行管理器
    run_manager = RunManager(args.config, args.run_id)
    context = run_manager.setup()

    # 加载配置（使用 resolved config）
    config = TERAGConfig.from_yaml(str(context.resolved_config_path))

    # 更新 artifacts 目录
    config.output.artifacts_dir = str(context.artifacts_dir)

    # 加载训练数据
    train_data = []
    train_path = config.get_split_path('train')
    if not os.path.exists(train_path):
        print(f"错误: 训练数据不存在: {train_path}")
        print("请先运行: python -m terag.cli prepare")
        sys.exit(1)

    with open(train_path, 'r', encoding='utf-8') as f:
        for line in f:
            train_data.append(json.loads(line))
    train_df = pd.DataFrame(train_data)

    print(f"加载训练数据: {len(train_df)} 条")

    # 1. 构建图
    print("\n[1/3] 构建二分图...")
    graph_builder = BipartiteGraphBuilder(config)
    graph = graph_builder.build(train_df)

    graph_path = context.artifacts_dir / "graph.pkl"
    graph_builder.save(graph, str(graph_path))

    if config.output.save_role_stats:
        role_stats_path = context.artifacts_dir / "role_stats.json"
        graph_builder.save_role_stats(str(role_stats_path))

    # 2. 挖掘模式
    print("\n[2/3] 挖掘查询模式...")
    element_to_queries = graph_builder.get_element_to_queries(graph, train_df)
    pattern_miner = PatternMiner(config)
    patterns = pattern_miner.mine(element_to_queries)

    patterns_path = context.artifacts_dir / "patterns.jsonl"
    pattern_miner.save(patterns, str(patterns_path))

    # 3. 构建索引
    print("\n[3/3] 构建索引...")
    index_builder = IndexBuilder(config)
    index = index_builder.build()

    index_dir = context.artifacts_dir / "bm25_index"
    index_builder.save(index, str(index_dir))

    print(f"\n构建完成!")
    print(f"Artifacts 保存到: {context.artifacts_dir}")
    print(f"Run ID: {context.run_id}")


def cmd_evaluate(args):
    """运行评估"""
    import pandas as pd
    from terag.config import TERAGConfig
    from scripts.run_experiments import (
        run_comparison_experiment,
        run_ablation_experiment,
        run_cold_start_experiment,
        run_resource_comparison,
        load_jsonl
    )

    print("=" * 60)
    print("TE-RAG 评估")
    print("=" * 60)

    # 加载配置
    config = TERAGConfig.from_yaml(args.config)

    # 如果指定了 run_id，使用对应的 artifacts
    if args.run_id:
        artifacts_dir = Path(config.output.artifacts_dir).parent / args.run_id
        config.output.artifacts_dir = str(artifacts_dir)
        output_dir = Path(config.output.results_dir).parent / args.run_id
    else:
        output_dir = Path(config.output.results_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")

    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载数据
    test_df = load_jsonl(config.get_split_path('test'))
    train_df = load_jsonl(config.get_split_path('train'))
    qa_df = pd.read_csv(config.data.qa_csv)

    print(f"测试数据: {len(test_df)} 条")
    print(f"训练数据: {len(train_df)} 条")

    # 运行评估
    results = {}

    if args.suite in ['all', 'comparison']:
        results['comparison'] = run_comparison_experiment(
            config, test_df, train_df, str(output_dir)
        )

    if args.suite in ['all', 'ablation']:
        results['ablation'] = run_ablation_experiment(
            config, test_df, str(output_dir)
        )

    if args.suite in ['all', 'cold_start']:
        results['cold_start'] = run_cold_start_experiment(
            config, qa_df, str(output_dir)
        )

    if args.suite in ['all', 'resource']:
        results['resource'] = run_resource_comparison(
            config, test_df, train_df, str(output_dir)
        )

    # 打印汇总
    print("\n" + "=" * 60)
    print("评估结果汇总")
    print("=" * 60)

    for name, df in results.items():
        print(f"\n{name}:")
        print(df.to_string())

    print(f"\n结果保存到: {output_dir}")


def cmd_run(args):
    """完整运行流程（prepare -> build -> evaluate）"""
    print("=" * 60)
    print("TE-RAG 完整运行")
    print("=" * 60)

    # 设置运行管理器
    run_manager = RunManager(args.config, args.run_id)
    context = run_manager.setup()

    print(f"Run ID: {context.run_id}")
    print(f"Artifacts 目录: {context.artifacts_dir}")
    print(f"Results 目录: {context.results_dir}")

    # 1. 准备数据
    if args.skip_prepare:
        print("\n[跳过] 数据准备")
    else:
        print("\n[1/3] 准备数据...")
        prepare_args = argparse.Namespace(config=args.config)
        cmd_prepare(prepare_args)

    # 2. 构建 artifacts
    if args.skip_build:
        print("\n[跳过] 构建 Artifacts")
    else:
        print("\n[2/3] 构建 Artifacts...")
        build_args = argparse.Namespace(
            config=str(context.resolved_config_path),
            run_id=context.run_id
        )
        cmd_build(build_args)

    # 3. 评估
    if args.skip_evaluate:
        print("\n[跳过] 评估")
    else:
        print("\n[3/3] 运行评估...")
        eval_args = argparse.Namespace(
            config=str(context.resolved_config_path),
            run_id=context.run_id,
            suite=args.suite
        )
        cmd_evaluate(eval_args)

    print("\n" + "=" * 60)
    print("运行完成!")
    print("=" * 60)
    print(f"Run ID: {context.run_id}")
    print(f"Artifacts: {context.artifacts_dir}")
    print(f"Results: {context.results_dir}")


def cmd_optimize_weights(args):
    """优化角色权重"""
    import pandas as pd
    from terag.config import TERAGConfig

    print("=" * 60)
    print("TE-RAG 角色权重优化")
    print("=" * 60)

    # 设置运行管理器
    run_manager = RunManager(args.config, args.run_id)
    context = run_manager.setup()

    # 加载配置
    config = TERAGConfig.from_yaml(str(context.resolved_config_path))

    print(f"Run ID: {context.run_id}")
    print(f"优化配置: n_trials={args.n_trials}")

    # 加载数据
    train_data = []
    train_path = config.get_split_path('train')
    with open(train_path, 'r', encoding='utf-8') as f:
        for line in f:
            train_data.append(json.loads(line))
    train_df = pd.DataFrame(train_data)

    dev_data = []
    dev_path = config.get_split_path('dev')
    with open(dev_path, 'r', encoding='utf-8') as f:
        for line in f:
            dev_data.append(json.loads(line))
    dev_df = pd.DataFrame(dev_data)

    print(f"训练数据: {len(train_df)} 条")
    print(f"验证数据: {len(dev_df)} 条")

    try:
        from terag.role_weight_optimizer import RoleWeightOptimizer

        # 创建优化器
        optimizer = RoleWeightOptimizer(
            config,
            metric='table_at_5',
            verbose=True
        )

        # 运行优化
        result = optimizer.optimize(
            train_df,
            dev_df,
            n_trials=args.n_trials
        )

        # 保存优化结果
        learned_weights_path = context.artifacts_dir / "learned_role_weights.json"
        optimizer.save_results(result, str(learned_weights_path))

        print(f"\n权重保存到: {learned_weights_path}")
        print(f"\n建议更新 config.yaml 中的 role_weights:")
        print("```yaml")
        print("graph:")
        print("  role_weights:")
        for role, weight in result.best_weights.items():
            print(f"    {role}: {weight:.2f}")
        print("```")

    except ImportError:
        print("\n错误: optuna 未安装")
        print("请运行: pip install optuna")
        sys.exit(1)


def cmd_ablation(args):
    """运行消融实验（带 artifacts 重建）"""
    import pandas as pd
    from terag.config import TERAGConfig
    from terag.terag_retriever_v2 import TERAGRetrieverV2
    from scripts.run_experiments import UnifiedEvaluator, load_jsonl

    print("=" * 60)
    print("TE-RAG 消融实验（带 Artifacts 重建）")
    print("=" * 60)

    # 设置运行管理器
    run_manager = RunManager(args.config, args.run_id)
    context = run_manager.setup()

    config = TERAGConfig.from_yaml(str(context.resolved_config_path))

    # 加载数据
    train_df = load_jsonl(config.get_split_path('train'))
    test_df = load_jsonl(config.get_split_path('test'))

    print(f"训练数据: {len(train_df)} 条")
    print(f"测试数据: {len(test_df)} 条")

    # 创建消融管理器
    ablation_manager = AblationManager(context)

    evaluator = UnifiedEvaluator(config)
    results = []

    for ablation_name in ablation_manager.list_ablations():
        print(f"\n{'='*40}")
        print(f"消融配置: {ablation_name}")
        print(f"{'='*40}")

        # 获取消融目录
        ablation_dir = ablation_manager.get_ablation_dir(ablation_name)
        ablation_config = ablation_manager.get_ablation_config(ablation_name)

        # 临时修改配置
        original_values = {}
        for key, value in ablation_config.items():
            if hasattr(config.ablation, key):
                original_values[key] = getattr(config.ablation, key)
                setattr(config.ablation, key, value)

        try:
            # 如果需要重建 artifacts
            if args.rebuild_artifacts:
                print(f"重建 artifacts 到: {ablation_dir}")
                _build_ablation_artifacts(config, train_df, ablation_dir)
                retriever = TERAGRetrieverV2.from_artifacts(config, str(ablation_dir))
            else:
                # 使用现有 artifacts
                retriever = TERAGRetrieverV2.from_artifacts(config)

            # 评估
            metrics = evaluator.evaluate_retriever(retriever, test_df)

            result = {'Configuration': ablation_name}
            result.update(metrics.to_dict())
            results.append(result)

            print(f"Table@5: {metrics.table_at_5:.4f}")
            print(f"Field_Coverage: {metrics.field_coverage:.4f}")

        finally:
            # 恢复配置
            for key, value in original_values.items():
                setattr(config.ablation, key, value)

    # 保存结果
    df = pd.DataFrame(results)
    output_path = context.results_dir / "ablation_results.csv"
    df.to_csv(output_path, index=False)

    print(f"\n{'='*60}")
    print("消融实验完成")
    print(f"{'='*60}")
    print(df.to_string())
    print(f"\n结果保存到: {output_path}")


def _build_ablation_artifacts(config, train_df, output_dir):
    """为消融实验构建 artifacts"""
    from terag.graph_builder import BipartiteGraphBuilder
    from terag.pattern_miner import PatternMiner
    from terag.index_builder import IndexBuilder

    # 1. 构建图
    graph_builder = BipartiteGraphBuilder(config)
    graph = graph_builder.build(train_df)

    graph_path = output_dir / "graph.pkl"
    graph_builder.save(graph, str(graph_path))

    # 2. 挖掘模式
    element_to_queries = graph_builder.get_element_to_queries(graph, train_df)
    pattern_miner = PatternMiner(config)
    patterns = pattern_miner.mine(element_to_queries)

    patterns_path = output_dir / "patterns.jsonl"
    pattern_miner.save(patterns, str(patterns_path))

    # 3. 构建索引
    index_builder = IndexBuilder(config)
    index = index_builder.build()

    index_dir = output_dir / "bm25_index"
    index_builder.save(index, str(index_dir))


def main():
    """CLI 主入口"""
    parser = argparse.ArgumentParser(
        description='TE-RAG CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    # 完整运行
    python -m terag.cli run --suite all --config config.yaml

    # 只运行数据准备
    python -m terag.cli prepare --config config.yaml

    # 只构建 artifacts
    python -m terag.cli build --config config.yaml

    # 只运行评估
    python -m terag.cli evaluate --config config.yaml --suite comparison

    # 运行消融实验（带 artifacts 重建）
    python -m terag.cli ablation --config config.yaml --rebuild-artifacts
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='可用命令')

    # run 命令
    run_parser = subparsers.add_parser('run', help='完整运行流程')
    run_parser.add_argument('--config', type=str, default='config.yaml',
                           help='配置文件路径')
    run_parser.add_argument('--suite', type=str, default='all',
                           choices=['all', 'comparison', 'ablation', 'cold_start', 'resource'],
                           help='实验类型')
    run_parser.add_argument('--run-id', type=str, default=None,
                           help='指定 run_id（不提供则自动生成）')
    run_parser.add_argument('--skip-prepare', action='store_true',
                           help='跳过数据准备')
    run_parser.add_argument('--skip-build', action='store_true',
                           help='跳过 artifacts 构建')
    run_parser.add_argument('--skip-evaluate', action='store_true',
                           help='跳过评估')

    # prepare 命令
    prepare_parser = subparsers.add_parser('prepare', help='准备数据')
    prepare_parser.add_argument('--config', type=str, default='config.yaml',
                               help='配置文件路径')

    # build 命令
    build_parser = subparsers.add_parser('build', help='构建 artifacts')
    build_parser.add_argument('--config', type=str, default='config.yaml',
                             help='配置文件路径')
    build_parser.add_argument('--run-id', type=str, default=None,
                             help='指定 run_id')

    # evaluate 命令
    eval_parser = subparsers.add_parser('evaluate', help='运行评估')
    eval_parser.add_argument('--config', type=str, default='config.yaml',
                            help='配置文件路径')
    eval_parser.add_argument('--suite', type=str, default='all',
                            choices=['all', 'comparison', 'ablation', 'cold_start', 'resource'],
                            help='实验类型')
    eval_parser.add_argument('--run-id', type=str, default=None,
                            help='使用指定 run_id 的 artifacts')

    # ablation 命令
    ablation_parser = subparsers.add_parser('ablation', help='运行消融实验')
    ablation_parser.add_argument('--config', type=str, default='config.yaml',
                                help='配置文件路径')
    ablation_parser.add_argument('--run-id', type=str, default=None,
                                help='指定 run_id')
    ablation_parser.add_argument('--rebuild-artifacts', action='store_true',
                                help='为每个消融配置重建 artifacts')

    # optimize-weights 命令
    opt_parser = subparsers.add_parser('optimize-weights', help='优化角色权重')
    opt_parser.add_argument('--config', type=str, default='config.yaml',
                           help='配置文件路径')
    opt_parser.add_argument('--run-id', type=str, default=None,
                           help='指定 run_id')
    opt_parser.add_argument('--n-trials', type=int, default=100,
                           help='Optuna 优化次数')

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    # 调用对应的命令处理函数
    commands = {
        'run': cmd_run,
        'prepare': cmd_prepare,
        'build': cmd_build,
        'evaluate': cmd_evaluate,
        'ablation': cmd_ablation,
        'optimize-weights': cmd_optimize_weights,
    }

    commands[args.command](args)


if __name__ == "__main__":
    main()
