"""
检索器基类

定义所有检索器必须实现的接口和通用功能
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
import time
import tracemalloc
import jieba


@dataclass
class RetrieverConfig:
    """检索器配置"""
    name: str = "BaseRetriever"            # 检索器名称
    description: str = ""                  # 描述
    k1: float = 1.5                        # BM25参数k1
    b: float = 0.75                        # BM25参数b
    top_k: int = 5                         # 返回的top-k结果
    use_chinese_tokenizer: bool = True     # 是否使用中文分词

    # 额外参数
    extra_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalResult:
    """单次检索结果"""
    table: str                             # 表名
    table_score: float                     # 表得分
    columns: List[tuple]                   # 列列表 [(column_name, score), ...]
    metadata: Dict[str, Any] = field(default_factory=dict)  # 额外元数据


@dataclass
class EvaluationMetrics:
    """评估指标"""
    table_accuracy: float = 0.0            # 表格选择准确率
    sql_accuracy: float = 0.0              # SQL准确率
    avg_query_time: float = 0.0            # 平均查询时间(秒)
    avg_memory_mb: float = 0.0             # 平均内存占用(MB)
    total_queries: int = 0                 # 总查询数

    # ========== 新增 Schema Coverage 指标 ==========
    table_recall: float = 0.0              # 表召回率
    column_recall: float = 0.0             # 字段召回率
    column_precision: float = 0.0          # 字段精确率
    column_f1: float = 0.0                 # 字段F1分数

    # ========== 新增 Top-K Table Recall ==========
    top1_table_recall: float = 0.0         # Top1 表召回率
    top3_table_recall: float = 0.0         # Top3 表召回率
    top5_table_recall: float = 0.0         # Top5 表召回率

    # ========== SQL Parse Rate ==========
    sql_parse_rate: float = 0.0            # SQL可解析率


class BaseRetriever(ABC):
    """
    检索器抽象基类

    所有检索器实现都必须继承此类并实现以下方法：
    - fit(): 训练/构建索引
    - _retrieve(): 核心检索逻辑

    使用方式:
        class MyRetriever(BaseRetriever):
            def __init__(self, field_csv, table_csv, config=None):
                super().__init__(field_csv, table_csv, config)
                # 自定义初始化

            def fit(self, train_data=None):
                # 构建索引
                self._is_fitted = True

            def _retrieve(self, query, k=5):
                # 实现检索逻辑
                return results
    """

    def __init__(self, field_csv: str, table_csv: str,
                 config: Optional[RetrieverConfig] = None):
        """
        初始化检索器

        Args:
            field_csv: 字段CSV路径
            table_csv: 表CSV路径
            config: 检索器配置（可选，子类可提供默认配置）
        """
        # 配置（子类可覆盖默认值）
        self.config = config or self._get_default_config()

        # 读取数据
        self.field_df = pd.read_csv(field_csv)
        self.table_df = pd.read_csv(table_csv)

        # 状态
        self._is_fitted = False

        # BM25参数（从config获取）
        self.k1 = self.config.k1
        self.b = self.config.b

        # 数据预处理
        self._preprocess_data()

    def _get_default_config(self) -> RetrieverConfig:
        """获取默认配置（子类可覆盖）"""
        return RetrieverConfig(name=self.__class__.__name__)

    def _preprocess_data(self):
        """数据预处理"""
        # 处理NaN值
        if self.table_df is not None:
            self.table_df['table_desc'] = self.table_df['table_desc'].fillna('').astype(str)

        if self.field_df is not None:
            self.field_df['field_name_desc'] = self.field_df['field_name_desc'].fillna('').astype(str)

    @abstractmethod
    def fit(self, train_data: pd.DataFrame = None):
        """
        训练/构建索引

        Args:
            train_data: 训练数据（可选，某些方法需要）
        """
        pass

    @abstractmethod
    def _retrieve(self, query: str, k: int = 5) -> List[RetrievalResult]:
        """
        核心检索逻辑（子类实现）

        Args:
            query: 查询文本
            k: 返回的top-k结果

        Returns:
            检索结果列表
        """
        pass

    def retrieve(self, query: str, k: int = None) -> List[RetrievalResult]:
        """
        检索接口（带检查和包装）

        Args:
            query: 查询文本
            k: 返回的top-k结果

        Returns:
            检索结果列表
        """
        if not self._is_fitted:
            raise RuntimeError(f"Retriever {self.config.name} has not been fitted. Call fit() first.")

        k = k or self.config.top_k
        return self._retrieve(query, k)

    def retrieve_with_metrics(self, query: str, k: int = None) -> tuple:
        """
        带性能指标的检索

        Args:
            query: 查询文本
            k: 返回的top-k结果

        Returns:
            (results, query_time, memory_usage_mb)
        """
        tracemalloc.start()
        start_time = time.time()

        results = self.retrieve(query, k)

        query_time = time.time() - start_time
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        return results, query_time, peak / 1024 / 1024

    def tokenize(self, text: str) -> List[str]:
        """
        分词

        Args:
            text: 输入文本

        Returns:
            分词结果
        """
        if self.config.use_chinese_tokenizer:
            return list(jieba.cut(text))
        else:
            return text.split()

    def evaluate(self, test_data: pd.DataFrame, k: int = 5) -> EvaluationMetrics:
        """
        评估检索器性能

        Args:
            test_data: 测试数据，需包含question, table, field列
            k: top-k结果

        Returns:
            评估指标
        """
        table_correct = 0
        sql_correct = 0
        query_times = []
        memory_usages = []

        # 新增指标统计
        table_recall_sum = 0.0
        column_recall_sum = 0.0
        column_precision_sum = 0.0
        column_f1_sum = 0.0
        top1_recall_count = 0
        top3_recall_count = 0
        top5_recall_count = 0

        for _, row in test_data.iterrows():
            query = row['question']
            gt_table = row.get('table', '')
            gt_fields = row.get('field', '')

            # 处理真实标签
            gt_table_simple = gt_table.split('.')[-1] if pd.notna(gt_table) else ''

            gt_field_list = []
            if pd.notna(gt_fields) and isinstance(gt_fields, str):
                gt_field_list = [f.strip() for f in gt_fields.split('|') if f.strip()]

            # 检索并记录性能
            results, query_time, memory_mb = self.retrieve_with_metrics(query, k)

            query_times.append(query_time)
            memory_usages.append(memory_mb)

            # 评估表格准确性
            retrieved_tables = [r.table for r in results]

            # ========== 计算 Top-K Table Recall ==========
            if gt_table_simple:
                if gt_table_simple in retrieved_tables[:1]:
                    top1_recall_count += 1
                if gt_table_simple in retrieved_tables[:3]:
                    top3_recall_count += 1
                if gt_table_simple in retrieved_tables[:5]:
                    top5_recall_count += 1

                # Table Recall
                if gt_table_simple in retrieved_tables:
                    table_recall_sum += 1.0

            if gt_table_simple in retrieved_tables:
                table_correct += 1

                # 评估字段准确性 - 收集所有返回的字段
                retrieved_fields = []
                for r in results:
                    if r.table == gt_table_simple:
                        for col, _ in r.columns:
                            field_name = col.split('.')[-1]
                            if field_name not in retrieved_fields:
                                retrieved_fields.append(field_name)

                # 如果有ground truth字段，检查是否都被检索到
                if gt_field_list:
                    # 检查是否所有必要字段都被检索到
                    if all(f in retrieved_fields for f in gt_field_list):
                        sql_correct += 1

                    # ========== 计算 Column Recall / Precision / F1 ==========
                    gt_set = set(gt_field_list)
                    pred_set = set(retrieved_fields)

                    # 初始化变量
                    c_recall = 0.0
                    c_precision = 0.0

                    # Recall
                    if gt_set:
                        c_recall = len(gt_set & pred_set) / len(gt_set)
                        column_recall_sum += c_recall

                    # Precision
                    if pred_set:
                        c_precision = len(gt_set & pred_set) / len(pred_set)
                        column_precision_sum += c_precision

                    # F1
                    if c_recall + c_precision > 0:
                        c_f1 = 2 * c_recall * c_precision / (c_recall + c_precision)
                        column_f1_sum += c_f1
                else:
                    # 如果没有指定字段，只要表正确就算SQL正确
                    sql_correct += 1

        total = len(test_data)

        return EvaluationMetrics(
            table_accuracy=table_correct / total if total > 0 else 0,
            sql_accuracy=sql_correct / total if total > 0 else 0,
            avg_query_time=np.mean(query_times) if query_times else 0,
            avg_memory_mb=np.mean(memory_usages) if memory_usages else 0,
            total_queries=total,
            # 新增指标
            table_recall=table_recall_sum / total if total > 0 else 0,
            column_recall=column_recall_sum / total if total > 0 else 0,
            column_precision=column_precision_sum / total if total > 0 else 0,
            column_f1=column_f1_sum / total if total > 0 else 0,
            top1_table_recall=top1_recall_count / total if total > 0 else 0,
            top3_table_recall=top3_recall_count / total if total > 0 else 0,
            top5_table_recall=top5_recall_count / total if total > 0 else 0,
        )

    @property
    def name(self) -> str:
        """获取检索器名称"""
        return self.config.name

    @property
    def is_fitted(self) -> bool:
        """是否已训练"""
        return self._is_fitted

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.config.name}', fitted={self._is_fitted})"
