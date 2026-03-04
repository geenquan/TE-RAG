"""
对比实验

使用新架构的对比实验实现
"""

import os
import sys
import pandas as pd
import numpy as np
from typing import Dict, List, Optional

# 添加父目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrievers import (
    RetrieverFactory, RetrieverManager, RetrieverConfig,
    BaseRetriever, EvaluationMetrics
)
from retrievers.terag_retriever import TERAGRetriever


class ComparisonExperiment:
    """
    对比实验执行器

    支持动态添加新的对比方法
    """

    def __init__(self, field_csv: str, table_csv: str, qa_csv: str):
        """
        初始化对比实验

        Args:
            field_csv: 字段CSV路径
            table_csv: 表CSV路径
            qa_csv: QA数据CSV路径
        """
        self.field_csv = field_csv
        self.table_csv = table_csv
        self.qa_csv = qa_csv

        self.qa_df = pd.read_csv(qa_csv)

        # 创建管理器
        self.manager = RetrieverManager(field_csv, table_csv, qa_csv)

        # 添加默认检索器
        self._add_default_retrievers()

        # 结果存储
        self.results: Dict[str, pd.DataFrame] = {}

    def _add_default_retrievers(self):
        """添加默认的对比方法"""
        # 获取可用的检索器列表
        available = RetrieverFactory.list_available()
        print(f"Available retrievers: {available}")

        # 添加所有可用的检索器
        for name in available:
            try:
                self.manager.add_retriever(name)
                print(f"Added retriever: {name}")
            except Exception as e:
                print(f"Failed to add retriever {name}: {e}")

    def add_custom_retriever(self, name: str, retriever_class: type,
                            config: Optional[RetrieverConfig] = None):
        """
        添加自定义检索器

        Args:
            name: 检索器名称
            retriever_class: 检索器类
            config: 配置

        Example:
            >>> class MyRetriever(BaseRetriever):
            >>>     def fit(self, train_data): ...
            >>>     def _retrieve(self, query, k): ...
            >>>
            >>> RetrieverFactory.register("MyMethod", MyRetriever)
            >>> experiment.add_custom_retriever("MyMethod", MyRetriever)
        """
        # 注册到工厂
        RetrieverFactory.register(name, retriever_class, config)

        # 添加到管理器
        self.manager.add_retriever(name, config)

    def remove_retriever(self, name: str):
        """移除检索器"""
        self.manager.remove_retriever(name)

    def run_comparison_experiment(self, train_ratio: float = 0.8,
                                  n_splits: int = 5) -> pd.DataFrame:
        """
        运行对比实验（交叉验证）

        Args:
            train_ratio: 训练集比例
            n_splits: 交叉验证折数

        Returns:
            结果DataFrame
        """
        print("\n" + "=" * 60)
        print("对比实验")
        print("=" * 60)
        print(f"Available methods: {list(self.manager.retrievers.keys())}")

        all_results = []

        for split in range(n_splits):
            print(f"\n--- 第 {split + 1}/{n_splits} 折 ---")

            # 随机分割
            np.random.seed(split * 42)
            indices = np.random.permutation(len(self.qa_df))
            n_train = int(len(indices) * train_ratio)

            train_indices = indices[:n_train]
            test_indices = indices[n_train:]

            train_data = self.qa_df.iloc[train_indices]
            test_data = self.qa_df.iloc[test_indices]

            # 训练所有检索器
            for name in self.manager.retrievers:
                print(f"  训练: {name}")
                self.manager.fit(name, train_data)

            # 评估所有检索器
            for name, retriever in self.manager.retrievers.items():
                if not retriever.is_fitted:
                    continue

                print(f"  评估: {name}")
                metrics = retriever.evaluate(test_data, k=5)

                all_results.append({
                    'Split': split + 1,
                    'Method': name,
                    'Table Accuracy': metrics.table_accuracy,
                    'SQL Accuracy': metrics.sql_accuracy,
                    'Avg Query Time (s)': metrics.avg_query_time,
                    'Avg Memory (MB)': metrics.avg_memory_mb
                })

        # 汇总
        df = pd.DataFrame(all_results)
        summary = df.groupby('Method').agg({
            'Table Accuracy': ['mean', 'std'],
            'SQL Accuracy': ['mean', 'std'],
            'Avg Query Time (s)': ['mean', 'std'],
            'Avg Memory (MB)': ['mean', 'std']
        }).reset_index()

        summary.columns = ['Method',
                          'Table Acc (mean)', 'Table Acc (std)',
                          'SQL Acc (mean)', 'SQL Acc (std)',
                          'Query Time (mean)', 'Query Time (std)',
                          'Memory (mean)', 'Memory (std)']

        return summary

    def run_cold_start_experiment(self, test_ratio: float = 0.2) -> pd.DataFrame:
        """
        冷启动实验

        Args:
            test_ratio: 测试表比例

        Returns:
            结果DataFrame
        """
        print("\n" + "=" * 60)
        print("冷启动实验")
        print("=" * 60)

        # 获取所有表
        all_tables = self.qa_df['table'].unique()
        n_test = max(1, int(len(all_tables) * test_ratio))

        # 随机选择测试表
        np.random.seed(42)
        test_tables = np.random.choice(all_tables, n_test, replace=False)

        # 分割数据
        test_data = self.qa_df[self.qa_df['table'].isin(test_tables)]
        train_data = self.qa_df[~self.qa_df['table'].isin(test_tables)]

        print(f"训练集: {len(train_data)} 条查询 (来自 {len(train_data['table'].unique())} 个表)")
        print(f"测试集: {len(test_data)} 条查询 (来自 {n_test} 个新表)")

        results = []

        # 训练所有检索器
        for name in self.manager.retrievers:
            print(f"  训练: {name}")
            self.manager.fit(name, train_data)

        # 评估
        for name, retriever in self.manager.retrievers.items():
            if not retriever.is_fitted:
                continue

            print(f"  评估: {name}")
            metrics = retriever.evaluate(test_data, k=5)

            results.append({
                'Method': name,
                'Table Accuracy': metrics.table_accuracy,
                'SQL Accuracy': metrics.sql_accuracy,
                'Avg Query Time (s)': metrics.avg_query_time,
                'Avg Memory (MB)': metrics.avg_memory_mb,
                'Type': 'Cold Start'
            })

        return pd.DataFrame(results)

    def run_data_efficiency_experiment(self, train_ratios: List[float] = None) -> pd.DataFrame:
        """
        数据效率实验

        测试不同训练数据量下的性能

        Args:
            train_ratios: 训练比例列表

        Returns:
            结果DataFrame
        """
        if train_ratios is None:
            train_ratios = [0.1, 0.2, 0.4, 0.6, 0.8]

        print("\n" + "=" * 60)
        print("数据效率实验")
        print("=" * 60)

        all_results = []

        for ratio in train_ratios:
            print(f"\n--- 训练比例: {ratio * 100:.0f}% ---")

            np.random.seed(42)
            indices = np.random.permutation(len(self.qa_df))
            n_train = int(len(indices) * ratio)

            train_data = self.qa_df.iloc[indices[:n_train]]
            test_data = self.qa_df.iloc[indices[n_train:]]

            # 训练和评估
            for name in self.manager.retrievers:
                self.manager.fit(name, train_data)

            for name, retriever in self.manager.retrievers.items():
                if not retriever.is_fitted:
                    continue

                metrics = retriever.evaluate(test_data, k=5)

                all_results.append({
                    'Train Ratio': ratio,
                    'Method': name,
                    'Table Accuracy': metrics.table_accuracy,
                    'SQL Accuracy': metrics.sql_accuracy
                })

        return pd.DataFrame(all_results)

    def run_all_experiments(self, output_dir: str = './results') -> Dict[str, pd.DataFrame]:
        """
        运行所有对比实验

        Args:
            output_dir: 输出目录

        Returns:
            结果字典
        """
        os.makedirs(output_dir, exist_ok=True)

        results = {}

        # 1. 对比实验
        print("\n" + "=" * 60)
        print("运行对比实验...")
        print("=" * 60)
        comparison_results = self.run_comparison_experiment(train_ratio=0.8, n_splits=3)
        results['comparison'] = comparison_results
        comparison_results.to_csv(os.path.join(output_dir, 'comparison_results.csv'), index=False)

        # 2. 冷启动实验
        print("\n" + "=" * 60)
        print("运行冷启动实验...")
        print("=" * 60)
        cold_start_results = self.run_cold_start_experiment(test_ratio=0.2)
        results['cold_start'] = cold_start_results
        cold_start_results.to_csv(os.path.join(output_dir, 'cold_start_results.csv'), index=False)

        # 3. 数据效率实验
        print("\n" + "=" * 60)
        print("运行数据效率实验...")
        print("=" * 60)
        efficiency_results = self.run_data_efficiency_experiment()
        results['data_efficiency'] = efficiency_results
        efficiency_results.to_csv(os.path.join(output_dir, 'data_efficiency_results.csv'), index=False)

        return results


# ============================================================================
# 示例：如何添加新的检索器
# ============================================================================

class ExampleCustomRetriever(BaseRetriever):
    """
    示例自定义检索器

    展示如何添加新的对比方法
    """

    def __init__(self, field_csv: str, table_csv: str,
                 config: Optional[RetrieverConfig] = None):
        if config is None:
            config = RetrieverConfig(
                name="Custom",
                description="Example custom retriever"
            )
        super().__init__(field_csv, table_csv, config)

        # 添加自定义属性
        self.custom_index = {}

    def fit(self, train_data: pd.DataFrame = None):
        """训练"""
        # 实现自定义训练逻辑
        print("  Custom retriever fitting...")

        # 简单示例：按表名首字母排序
        self.custom_index = {
            row['table']: idx
            for idx, row in self.table_df.iterrows()
        }

        self._is_fitted = True

    def _retrieve(self, query: str, k: int = 5) -> List:
        """检索"""
        from retrievers.base_retriever import RetrievalResult

        # 简单示例：返回前k个表
        results = []
        for idx in range(min(k, len(self.table_df))):
            row = self.table_df.iloc[idx]
            table_name = row['table']

            columns = self.field_df[self.field_df['table'] == table_name]
            column_list = [
                (f"C:{table_name}.{col['field_name']}", 1.0)
                for _, col in columns.head(5).iterrows()
            ]

            results.append(RetrievalResult(
                table=table_name,
                table_score=1.0 / (idx + 1),
                columns=column_list,
                metadata={'method': 'Custom'}
            ))

        return results


def demo_add_custom_retriever():
    """
    演示如何添加自定义检索器
    """
    print("=" * 60)
    print("演示：添加自定义检索器")
    print("=" * 60)

    # 方法1：先注册到工厂，然后使用
    RetrieverFactory.register(
        "CustomMethod",
        ExampleCustomRetriever,
        RetrieverConfig(name="CustomMethod", description="My custom retriever")
    )

    # 方法2：在实验中直接添加
    field_csv = '/path/to/field.csv'
    table_csv = '/path/to/table.csv'
    qa_csv = '/path/to/qa.csv'

    experiment = ComparisonExperiment(field_csv, table_csv, qa_csv)
    experiment.add_custom_retriever(
        "CustomMethod",
        ExampleCustomRetriever,
        RetrieverConfig(name="CustomMethod")
    )

    # 运行实验时会自动包含新方法
    # results = experiment.run_comparison_experiment()


def main():
    """主函数"""
    # 数据路径
    field_csv = '/Users/apple/Documents/浙大工作/论文/分层查询数据表/code/source_dataset/processed_field_schema.csv'
    table_csv = '/Users/apple/Documents/浙大工作/论文/分层查询数据表/code/source_dataset/processed_table_schema.csv'
    qa_csv = '/Users/apple/Documents/浙大工作/论文/分层查询数据表/code/source_dataset/processed_qa_data.csv'
    output_dir = '/Users/apple/Documents/浙大工作/论文/分层查询数据表/code/results/0303'

    # 创建实验
    experiment = ComparisonExperiment(field_csv, table_csv, qa_csv)

    # 可选：添加自定义检索器
    # experiment.add_custom_retriever("MyMethod", MyRetrieverClass)

    # 运行所有实验
    results = experiment.run_all_experiments(output_dir=output_dir)

    # 打印结果
    print("\n" + "=" * 60)
    print("对比实验结果汇总")
    print("=" * 60)

    for name, df in results.items():
        print(f"\n{name}:")
        print(df.to_string())


if __name__ == "__main__":
    main()
