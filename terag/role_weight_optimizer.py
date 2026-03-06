"""
角色权重优化器

使用 Optuna 自动优化 SQL 角色权重

使用方式:
    optimizer = RoleWeightOptimizer(config)
    best_weights = optimizer.optimize(train_data, dev_data, n_trials=100)
"""

import os
import json
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

try:
    import optuna
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("Warning: optuna not available, role weight optimization will be skipped")

from terag.config import TERAGConfig


@dataclass
class OptimizationResult:
    """优化结果"""
    best_weights: Dict[str, float]
    best_score: float
    n_trials: int
    study: Optional[Any] = None


class RoleWeightOptimizer:
    """
    角色权重优化器

    使用 Optuna 优化 SQL 角色权重，以最大化验证集上的检索指标。

    角色包括:
    - SELECT: 选择列
    - WHERE: 条件过滤
    - JOIN: 表连接
    - GROUP_BY: 分组
    - ORDER_BY: 排序
    - HAVING: 分组后过滤
    - FROM: 数据源

    使用方式:
        optimizer = RoleWeightOptimizer(config)
        result = optimizer.optimize(train_data, dev_data, n_trials=100)
        print(f"最佳权重: {result.best_weights}")
        print(f"最佳得分: {result.best_score}")
    """

    DEFAULT_SEARCH_SPACE = {
        'SELECT': (0.5, 3.0),
        'WHERE': (0.5, 3.0),
        'JOIN': (0.5, 3.0),
        'GROUP_BY': (0.5, 3.0),
        'ORDER_BY': (0.5, 3.0),
        'HAVING': (0.5, 3.0),
        'FROM': (0.5, 2.0),
    }

    def __init__(
        self,
        config: TERAGConfig,
        metric: str = 'table_at_5',
        search_space: Dict[str, Tuple[float, float]] = None,
        verbose: bool = True
    ):
        """
        初始化优化器

        Args:
            config: TE-RAG 配置
            metric: 优化目标指标 ('table_at_5', 'table_at_1', 'field_coverage')
            search_space: 搜索空间 {role: (min, max)}
            verbose: 是否输出详细日志
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError("optuna 未安装，请运行: pip install optuna")

        self.config = config
        self.metric = metric
        self.search_space = search_space or self.DEFAULT_SEARCH_SPACE
        self.verbose = verbose

        # 验证指标
        valid_metrics = ['table_at_5', 'table_at_1', 'field_coverage']
        if metric not in valid_metrics:
            raise ValueError(f"无效指标: {metric}，可选: {valid_metrics}")

    def optimize(
        self,
        train_data: pd.DataFrame,
        dev_data: pd.DataFrame,
        n_trials: int = 100,
        timeout: Optional[int] = None,
        n_jobs: int = 1
    ) -> OptimizationResult:
        """
        优化角色权重

        Args:
            train_data: 训练数据（用于构建 retriever）
            dev_data: 验证数据（用于评估）
            n_trials: 优化次数
            timeout: 超时时间（秒）
            n_jobs: 并行数

        Returns:
            OptimizationResult
        """
        from terag.terag_retriever_v2 import TERAGRetrieverV2

        if self.verbose:
            print("=" * 60)
            print("角色权重优化")
            print("=" * 60)
            print(f"优化目标: {self.metric}")
            print(f"优化次数: {n_trials}")
            print(f"训练数据: {len(train_data)} 条")
            print(f"验证数据: {len(dev_data)} 条")

        # 创建 study
        sampler = TPESampler(seed=self.config.seed)
        study = optuna.create_study(
            direction='maximize',
            sampler=sampler,
            study_name='role_weight_optimization'
        )

        # 定义目标函数
        def objective(trial):
            # 采样权重
            weights = {}
            for role, (low, high) in self.search_space.items():
                weights[role] = trial.suggest_float(f'weight_{role}', low, high)

            # 临时修改配置
            original_weights = self.config.graph.role_weights.copy()
            self.config.graph.role_weights = weights

            try:
                # 构建 retriever
                retriever = TERAGRetrieverV2(self.config)
                retriever.fit(train_data)

                # 评估
                score = self._evaluate(retriever, dev_data)

                if self.verbose and trial.number % 10 == 0:
                    print(f"Trial {trial.number}: {self.metric}={score:.4f}")

                return score

            finally:
                # 恢复配置
                self.config.graph.role_weights = original_weights

        # 运行优化
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout,
            n_jobs=n_jobs,
            show_progress_bar=self.verbose
        )

        # 获取最佳结果
        best_weights = {}
        for role in self.search_space.keys():
            best_weights[role] = study.best_trial.params[f'weight_{role}']

        if self.verbose:
            print("\n优化完成!")
            print(f"最佳 {self.metric}: {study.best_value:.4f}")
            print(f"最佳权重:")
            for role, weight in best_weights.items():
                print(f"  {role}: {weight:.4f}")

        return OptimizationResult(
            best_weights=best_weights,
            best_score=study.best_value,
            n_trials=len(study.trials),
            study=study
        )

    def _evaluate(self, retriever, dev_data: pd.DataFrame) -> float:
        """评估 retriever"""
        # 简单评估
        table_correct = 0
        total = len(dev_data)

        for _, row in dev_data.iterrows():
            query = row['question']
            gt_table = row.get('table', '')
            gt_table_simple = gt_table.split('.')[-1] if pd.notna(gt_table) else ''

            try:
                results = retriever.retrieve(query, k=5)
                retrieved_tables = [r.table for r in results]

                if self.metric == 'table_at_1':
                    if gt_table_simple in retrieved_tables[:1]:
                        table_correct += 1
                elif self.metric == 'table_at_5':
                    if gt_table_simple in retrieved_tables[:5]:
                        table_correct += 1
                elif self.metric == 'field_coverage':
                    # 简化的字段覆盖率
                    if gt_table_simple in retrieved_tables[:5]:
                        table_correct += 1
            except Exception:
                pass

        return table_correct / total if total > 0 else 0.0

    def save_results(self, result: OptimizationResult, output_path: str):
        """
        保存优化结果

        Args:
            result: 优化结果
            output_path: 输出路径
        """
        output = {
            'best_weights': result.best_weights,
            'best_score': result.best_score,
            'n_trials': result.n_trials,
            'metric': self.metric,
        }

        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        print(f"优化结果已保存到: {output_path}")


class MultiObjectiveOptimizer:
    """
    多目标优化器

    同时优化多个指标（如 Table@5 和 Field Coverage）
    """

    def __init__(
        self,
        config: TERAGConfig,
        objectives: List[str] = None,
        verbose: bool = True
    ):
        """
        初始化多目标优化器

        Args:
            config: TE-RAG 配置
            objectives: 优化目标列表
            verbose: 是否输出详细日志
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError("optuna 未安装，请运行: pip install optuna")

        self.config = config
        self.objectives = objectives or ['table_at_5', 'field_coverage']
        self.verbose = verbose

    def optimize(
        self,
        train_data: pd.DataFrame,
        dev_data: pd.DataFrame,
        n_trials: int = 100
    ) -> List[Dict]:
        """
        多目标优化

        Args:
            train_data: 训练数据
            dev_data: 验证数据
            n_trials: 优化次数

        Returns:
            Pareto 前沿上的权重配置列表
        """
        from terag.terag_retriever_v2 import TERAGRetrieverV2

        if self.verbose:
            print("=" * 60)
            print("多目标角色权重优化")
            print("=" * 60)
            print(f"优化目标: {self.objectives}")
            print(f"优化次数: {n_trials}")

        sampler = TPESampler(seed=self.config.seed, multivariate=True)
        study = optuna.create_study(
            directions=['maximize'] * len(self.objectives),
            sampler=sampler
        )

        def objective(trial):
            # 采样权重
            weights = {}
            for role in ['SELECT', 'WHERE', 'JOIN', 'GROUP_BY', 'ORDER_BY', 'HAVING', 'FROM']:
                weights[role] = trial.suggest_float(f'weight_{role}', 0.5, 3.0)

            # 临时修改配置
            original_weights = self.config.graph.role_weights.copy()
            self.config.graph.role_weights = weights

            try:
                # 构建 retriever
                retriever = TERAGRetrieverV2(self.config)
                retriever.fit(train_data)

                # 评估
                metric_values = self._evaluate_multi(retriever, dev_data)
                return tuple(metric_values)

            finally:
                self.config.graph.role_weights = original_weights

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study.optimize(objective, n_trials=n_trials, show_progress_bar=self.verbose)

        # 提取 Pareto 前沿
        pareto_front = []
        for trial in study.best_trials:
            weights = {}
            for role in ['SELECT', 'WHERE', 'JOIN', 'GROUP_BY', 'ORDER_BY', 'HAVING', 'FROM']:
                weights[role] = trial.params[f'weight_{role}']
            pareto_front.append({
                'weights': weights,
                'scores': dict(zip(self.objectives, trial.values))
            })

        if self.verbose:
            print(f"\nPareto 前沿包含 {len(pareto_front)} 个解")

        return pareto_front

    def _evaluate_multi(self, retriever, dev_data: pd.DataFrame) -> List[float]:
        """多目标评估"""
        table_at_5 = 0
        field_coverage = 0
        total = len(dev_data)

        for _, row in dev_data.iterrows():
            query = row['question']
            gt_table = row.get('table', '')
            gt_fields = row.get('field', '')

            gt_table_simple = gt_table.split('.')[-1] if pd.notna(gt_table) else ''

            gt_field_set = set()
            if pd.notna(gt_fields) and isinstance(gt_fields, str):
                gt_field_set = set(f.strip() for f in gt_fields.split('|') if f.strip())

            try:
                results = retriever.retrieve(query, k=5)
                retrieved_tables = [r.table for r in results]

                # Table@5
                if gt_table_simple in retrieved_tables[:5]:
                    table_at_5 += 1

                    # Field coverage
                    retrieved_fields = set()
                    for r in results:
                        if r.table == gt_table_simple:
                            for col, _ in r.columns:
                                field_name = col.split('.')[-1]
                                retrieved_fields.add(field_name)

                    if gt_field_set and gt_field_set.issubset(retrieved_fields):
                        field_coverage += 1

            except Exception:
                pass

        return [
            table_at_5 / total if total > 0 else 0.0,
            field_coverage / total if total > 0 else 0.0
        ]


# For type hint
from typing import Any


def main():
    """演示角色权重优化"""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent))

    # 加载配置
    config_path = Path(__file__).parent.parent / 'config.yaml'
    config = TERAGConfig.from_yaml(str(config_path))

    print("=" * 60)
    print("角色权重优化演示")
    print("=" * 60)

    # 加载数据
    train_data = []
    with open(config.get_split_path('train'), 'r', encoding='utf-8') as f:
        for line in f:
            train_data.append(json.loads(line))
    train_df = pd.DataFrame(train_data)

    dev_data = []
    with open(config.get_split_path('dev'), 'r', encoding='utf-8') as f:
        for line in f:
            dev_data.append(json.loads(line))
    dev_df = pd.DataFrame(dev_data)

    print(f"训练数据: {len(train_df)} 条")
    print(f"验证数据: {len(dev_df)} 条")

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
        n_trials=20  # 演示用较少次数
    )

    # 保存结果
    output_path = config.get_artifact_path('learned_role_weights.json')
    optimizer.save_results(result, output_path)

    # 更新配置
    print(f"\n建议更新 config.yaml 中的 role_weights:")
    print("```yaml")
    print("graph:")
    print("  role_weights:")
    for role, weight in result.best_weights.items():
        print(f"    {role}: {weight:.2f}")
    print("```")


if __name__ == "__main__":
    main()
