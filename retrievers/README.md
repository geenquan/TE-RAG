# 检索器架构说明

## 架构概述

本项目采用模块化的检索器架构，支持：

1. **可插拔设计**：每种检索方法独立成一个类
2. **工厂模式**：通过 `RetrieverFactory` 统一创建检索器
3. **配置化管理**：通过 `RetrieverConfig` 管理检索器参数
4. **统一接口**：所有检索器继承自 `BaseRetriever`

## 文件结构

```
retrievers/
├── __init__.py           # 模块入口
├── base_retriever.py     # 基类定义
├── bm25_retriever.py     # BM25检索器
├── vector_retriever.py   # 向量检索器
├── llm_retriever.py      # LLM检索器
├── terag_retriever.py    # TE-RAG检索器
└── retriever_factory.py  # 工厂类和管理器
```

## 使用方法

### 1. 基本使用

```python
from retrievers import RetrieverFactory

# 列出可用的检索器
print(RetrieverFactory.list_available())
# 输出: ['BM25', 'Vector', 'LLM', 'TE-RAG']

# 创建检索器
retriever = RetrieverFactory.create('BM25', field_csv, table_csv)

# 训练
retriever.fit(train_data)

# 检索
results = retriever.retrieve("2025年2月杭州公司的售电量是多少？")

# 评估
metrics = retriever.evaluate(test_data)
```

### 2. 批量创建和管理

```python
from retrievers import RetrieverManager

# 创建管理器
manager = RetrieverManager(field_csv, table_csv, qa_csv)

# 添加所有内置检索器
for name in RetrieverFactory.list_available():
    manager.add_retriever(name)

# 训练所有检索器
manager.fit_all(train_data)

# 评估所有检索器
results = manager.evaluate_all(test_data)

# 对比结果
comparison_df = manager.compare(test_data)
```

### 3. 运行实验

```python
from experiment.comparison import ComparisonExperiment

experiment = ComparisonExperiment(field_csv, table_csv, qa_csv)

# 运行所有实验
results = experiment.run_all_experiments(output_dir='./results')
```

## 添加新的检索器

### 方法1：创建新的检索器文件

1. **创建新文件** `retrievers/my_retriever.py`：

```python
from retrievers.base_retriever import (
    BaseRetriever, RetrieverConfig, RetrievalResult
)
from typing import List, Optional
import pandas as pd


class MyRetriever(BaseRetriever):
    """
    自定义检索器
    """

    def __init__(self, field_csv: str, table_csv: str,
                 config: Optional[RetrieverConfig] = None):
        if config is None:
            config = RetrieverConfig(
                name="MyRetriever",
                description="My custom retriever"
            )
        super().__init__(field_csv, table_csv, config)

        # 添加自定义属性
        self.my_index = {}

    def fit(self, train_data: pd.DataFrame = None):
        """
        训练/构建索引

        Args:
            train_data: 训练数据
        """
        # 实现自定义训练逻辑
        print("Fitting MyRetriever...")

        # 示例：构建简单索引
        for _, row in self.table_df.iterrows():
            self.my_index[row['table']] = row.get('table_desc', '')

        self._is_fitted = True

    def _retrieve(self, query: str, k: int = 5) -> List[RetrievalResult]:
        """
        核心检索逻辑

        Args:
            query: 查询文本
            k: 返回的top-k结果

        Returns:
            检索结果列表
        """
        results = []

        # 实现自定义检索逻辑
        for table_name, desc in list(self.my_index.items())[:k]:
            # 获取列
            columns = self.field_df[self.field_df['table'] == table_name]
            column_list = [
                (f"C:{table_name}.{row['field_name']}", 1.0)
                for _, row in columns.head(5).iterrows()
            ]

            results.append(RetrievalResult(
                table=table_name,
                table_score=1.0,
                columns=column_list,
                metadata={'method': 'MyRetriever'}
            ))

        return results
```

2. **注册检索器**：

```python
from retrievers import RetrieverFactory, RetrieverConfig
from retrievers.my_retriever import MyRetriever

# 方法A：手动注册
RetrieverFactory.register(
    "MyRetriever",
    MyRetriever,
    RetrieverConfig(name="MyRetriever", description="My custom retriever")
)

# 方法B：在 __init__.py 中添加导入
# from retrievers.my_retriever import MyRetriever
```

3. **使用新检索器**：

```python
# 创建
retriever = RetrieverFactory.create('MyRetriever', field_csv, table_csv)

# 或者添加到实验中
from experiment.comparison import ComparisonExperiment
experiment = ComparisonExperiment(field_csv, table_csv, qa_csv)
experiment.add_custom_retriever("MyRetriever", MyRetriever)
```

### 方法2：在实验中动态添加

```python
from experiment.comparison import ComparisonExperiment
from retrievers.base_retriever import BaseRetriever, RetrieverConfig, RetrievalResult

# 定义检索器
class QuickRetriever(BaseRetriever):
    def __init__(self, field_csv, table_csv, config=None):
        config = config or RetrieverConfig(name="Quick")
        super().__init__(field_csv, table_csv, config)

    def fit(self, train_data=None):
        self._is_fitted = True

    def _retrieve(self, query, k=5):
        # 简单实现
        return [RetrievalResult(table=t, table_score=1.0, columns=[])
                for t in self.table_df['table'].head(k)]

# 添加到实验
experiment = ComparisonExperiment(field_csv, table_csv, qa_csv)
experiment.add_custom_retriever("Quick", QuickRetriever)

# 运行实验
results = experiment.run_comparison_experiment()
```

## 配置参数

```python
from retrievers import RetrieverConfig

config = RetrieverConfig(
    name="CustomRetriever",
    description="My custom retriever",
    k1=1.5,                    # BM25参数
    b=0.75,                    # BM25参数
    top_k=5,                   # 默认返回结果数
    use_chinese_tokenizer=True,
    extra_params={             # 自定义参数
        "custom_weight": 0.5
    }
)

retriever = RetrieverFactory.create('BM25', field_csv, table_csv, config=config)
```

## 基类接口

```python
class BaseRetriever(ABC):
    # 必须实现的方法
    @abstractmethod
    def fit(self, train_data: pd.DataFrame = None):
        """训练/构建索引"""
        pass

    @abstractmethod
    def _retrieve(self, query: str, k: int = 5) -> List[RetrievalResult]:
        """核心检索逻辑"""
        pass

    # 已提供的方法
    def retrieve(self, query: str, k: int = None) -> List[RetrievalResult]:
        """检索接口（带检查）"""
        pass

    def retrieve_with_metrics(self, query: str, k: int = None) -> tuple:
        """带性能指标的检索"""
        pass

    def evaluate(self, test_data: pd.DataFrame, k: int = 5) -> EvaluationMetrics:
        """评估检索器"""
        pass

    def tokenize(self, text: str) -> List[str]:
        """分词"""
        pass
```

## 实验结果

运行实验后会生成以下结果：

- `comparison_results.csv` - 对比实验结果
- `cold_start_results.csv` - 冷启动实验结果
- `data_efficiency_results.csv` - 数据效率实验结果
- `comparison_accuracy.png` - 准确性对比图
- `comparison_performance.png` - 性能对比图
- `experiment_summary.md` - 实验总结报告
