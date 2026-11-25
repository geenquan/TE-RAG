"""
检索器基类

定义所有检索器必须实现的接口和通用功能
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np
import time
import tracemalloc
import jieba

from terag.tokenization import tokenize_english
from datasets.gt_utils import (
    parse_ground_truth_tables,
    parse_ground_truth_fields,
    has_structured_gold,
    load_json_list,
)


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
            分词结果（use_chinese_tokenizer=True 用 jieba，否则用英文正则分词）
        """
        if self.config.use_chinese_tokenizer:
            return list(jieba.cut(text))
        else:
            return tokenize_english(text)

    @staticmethod
    def _retrieved_column_keys(results: List['RetrievalResult']) -> List[str]:
        """
        从检索结果提取 canonical 列 key（"table.column"，去掉 "C:" 前缀）

        用于多表场景（BIRD）的 Column Recall / Precision / F1 计算
        """
        keys = []
        for r in results:
            for col, _ in r.columns:
                col_name = col[2:] if col.startswith('C:') else col
                if col_name not in keys:
                    keys.append(col_name)
        return keys

    def evaluate(self, test_data: pd.DataFrame, k: int = 5) -> EvaluationMetrics:
        """
        评估检索器性能

        同时支持两种 ground truth 格式：
        - PowerGrid（单表, table/field 列）：指标语义与原实现完全一致
        - BIRD（多表, gold_tables/gold_fields 结构化列）：
          * Table Accuracy = Table Exact Match（所有 gold 表都在检索结果中）
          * Top-K Table Recall = mean(|gold ∩ topK| / |gold|)
          * Column 指标基于 "table.column" 级别的集合匹配（macro average）
          * SQL Accuracy = Schema-Grounded 准确率（表全中且 gold 列全覆盖）

        Args:
            test_data: 测试数据，需包含question列，以及 table/field 或 gold_tables/gold_fields 列
            k: top-k结果

        Returns:
            评估指标
        """
        records = self.evaluate_detailed(test_data, k)
        return self._aggregate_records(records)

    def evaluate_detailed(self, test_data: pd.DataFrame, k: int = 5) -> List[Dict[str, Any]]:
        """
        逐条评估，返回每条查询的明细记录。

        P1-A 起，每条记录同时包含：
        - 新口径指标（eval/retrieval_metrics.py 定义）：hit_at_{1,3,5}（单表）或
          table_recall_at_k / all_gold_covered（多表）、column_precision/recall/f1、
          schema_coverage_success、gold_field_status；
        - 兼容别名（旧字段名，语义 = P0 审计确认的旧实现语义，仅供旧管线内部使用）：
          table_accuracy_hit / table_recall_full（单表 = Hit@k，k 为本参数）、
          schema_sql_correct / sql_accuracy_hit（空 gold 字段记为正确的旧 SCS 变体）；
        - 状态字段：status（ok/error）、error_type、error_message（检索异常不再中断
          整个评估，也不会伪装成正常空预测）、exec_accuracy 恒为 'not_run'。

        Args:
            test_data: 测试数据
            k: top-k结果

        Returns:
            记录列表，每条包含 question/db_id/difficulty/gold/retrieved 与各项指标
        """
        import time as _time
        from eval.retrieval_metrics import (
            evaluate_single_table_query, evaluate_multi_table_query,
            normalize_column_key,
        )

        records = []

        for _, row in test_data.iterrows():
            query = row['question']

            record = {
                'question': query,
                'db_id': row.get('db_id', ''),
                'difficulty': row.get('difficulty', ''),
                'question_id': row.get('question_id', None),
                'gold_tables': parse_ground_truth_tables(row),
                'status': 'ok',
                'error_type': None,
                'error_message': None,
            }

            # ---- 检索（逐题异常捕获：error 记录不中断评估、不伪装为正常结果） ----
            t0 = _time.time()
            try:
                results, query_time, memory_mb = self.retrieve_with_metrics(query, k)
            except Exception as e:  # noqa: BLE001 - 需记录任意检索异常
                results, query_time, memory_mb = [], _time.time() - t0, 0.0
                record.update({
                    'status': 'error',
                    'error_type': type(e).__name__,
                    'error_message': str(e)[:500],
                })
                print(f"[evaluate_detailed] 检索异常 status=error: {type(e).__name__}: {str(e)[:200]}")

            retrieved_tables = [r.table for r in results]
            retrieved_column_keys = []
            retrieved_columns_detail = []
            for r in results:
                cols = [{'key': col, 'score': float(sc)} for col, sc in r.columns]
                retrieved_columns_detail.append(
                    {'table': r.table, 'table_score': float(r.table_score), 'columns': cols})
                for col, _ in r.columns:
                    key = normalize_column_key(col)
                    if key:
                        retrieved_column_keys.append(key)

            record.update({
                'retrieved_tables': retrieved_tables,
                'retrieved_column_keys': retrieved_column_keys,
                'retrieved_columns_detail': retrieved_columns_detail,
                'method_evidence': (results[0].metadata if results else None),
                'query_time': query_time,
                'memory_mb': memory_mb,
                'exec_accuracy': 'not_run',
            })

            if has_structured_gold(row):
                # ========== BIRD 多表评估 ==========
                gold_tables = record['gold_tables']
                gold_columns = [f"{t}.{f}" for t, f in parse_ground_truth_fields(row) if t]

                m = evaluate_multi_table_query(
                    gold_tables, gold_columns, retrieved_tables, retrieved_column_keys)

                # 兼容别名（旧语义：gold 为空记 0；SCS 空 gold 记为正确）
                gold_set = set(gold_tables)
                retrieved_set = set(retrieved_tables)
                table_exact = gold_set.issubset(retrieved_set) if gold_set else False
                gold_col_set = set(gold_columns)
                pred_col_set = set(retrieved_column_keys)
                inter = gold_col_set & pred_col_set
                c_recall_legacy = len(inter) / len(gold_col_set) if gold_col_set else 0.0
                schema_sql_correct_legacy = table_exact and (not gold_col_set or c_recall_legacy >= 1.0)
                recall_at = m['table_recall_at_k']

                record.update({
                    'multi_table': len(gold_tables) > 1,
                    'table_recall_at_k': recall_at,
                    'all_gold_covered': m['all_gold_covered'],
                    'column_precision': m['column_precision'],
                    'column_recall': m['column_recall'],
                    'column_f1': m['column_f1'],
                    'annotated_field_coverage_success': m['annotated_field_coverage_success'],
                    'schema_coverage_success': m['schema_coverage_success'],  # 兼容别名
                    'gold_field_status': m['gold_field_status'],
                    'gold_column_keys': m['gold_column_keys'],
                    # ---- 兼容别名（旧字段/旧语义） ----
                    'table_exact_match': table_exact,
                    'table_accuracy_hit': table_exact,
                    'table_recall_full': (len(gold_set & retrieved_set) / len(gold_set))
                                        if gold_set else 0.0,
                    'table_recall_at_1': recall_at.get(1) or 0.0,
                    'table_recall_at_3': recall_at.get(3) or 0.0,
                    'table_recall_at_5': recall_at.get(5) or 0.0,
                    'schema_sql_correct': schema_sql_correct_legacy,
                    'sql_accuracy_hit': schema_sql_correct_legacy,
                })
            else:
                # ========== PowerGrid 单表评估 ==========
                gt_table = row.get('table', '')
                gt_fields = row.get('field', '')

                m = evaluate_single_table_query(
                    gt_table if pd.notna(gt_table) else None,
                    gt_fields if isinstance(gt_fields, str) else None,
                    retrieved_tables, retrieved_column_keys)

                hit = m['hit']

                # 兼容别名（旧语义：列指标仅统计命中表返回字段；空 gold 字段记 SCS 正确）
                gt_table_simple = gt_table.split('.')[-1] if pd.notna(gt_table) else ''
                table_hit_legacy = bool(gt_table_simple) and gt_table_simple in retrieved_tables
                c_recall = c_precision = c_f1 = 0.0
                sql_hit = False
                if table_hit_legacy:
                    retrieved_fields = []
                    for r in results:
                        if r.table == gt_table_simple:
                            for col, _ in r.columns:
                                field_name = col.split('.')[-1]
                                if field_name not in retrieved_fields:
                                    retrieved_fields.append(field_name)
                    gt_field_list = m['gold_column_keys'] and [k.split('.')[-1] for k in m['gold_column_keys']]
                    if gt_field_list:
                        sql_hit = all(f.lower() in {x.lower() for x in retrieved_fields}
                                      for f in gt_field_list)
                        gt_set = {f.lower() for f in gt_field_list}
                        pred_set = {x.lower() for x in retrieved_fields}
                        inter_l = gt_set & pred_set
                        c_recall = len(inter_l) / len(gt_set) if gt_set else 0.0
                        c_precision = len(inter_l) / len(pred_set) if pred_set else 0.0
                        c_f1 = (2 * c_recall * c_precision / (c_recall + c_precision)) \
                            if (c_recall + c_precision) > 0 else 0.0
                    else:
                        sql_hit = True
                else:
                    sql_hit = False

                record.update({
                    'multi_table': False,
                    'hit_at_1': bool(hit.get(1)),
                    'hit_at_3': bool(hit.get(3)),
                    'hit_at_5': bool(hit.get(5)),
                    'column_precision': m['column_precision'],
                    'column_recall': m['column_recall'],
                    'column_f1': m['column_f1'],
                    'annotated_field_coverage_success': m['annotated_field_coverage_success'],
                    'schema_coverage_success': m['schema_coverage_success'],  # 兼容别名
                    'gold_field_status': m['gold_field_status'],
                    'gold_column_keys': m['gold_column_keys'],
                    # ---- 兼容别名（旧字段/旧语义） ----
                    'table_exact_match': table_hit_legacy,
                    'table_accuracy_hit': table_hit_legacy,
                    'table_recall_full': 1.0 if table_hit_legacy else 0.0,
                    'table_recall_at_1': 1.0 if hit.get(1) else 0.0,
                    'table_recall_at_3': 1.0 if hit.get(3) else 0.0,
                    'table_recall_at_5': 1.0 if hit.get(5) else 0.0,
                    'column_recall_legacy': c_recall,
                    'column_precision_legacy': c_precision,
                    'column_f1_legacy': c_f1,
                    'schema_sql_correct': sql_hit,
                    'sql_accuracy_hit': sql_hit,
                })

            records.append(record)

        return records

    @staticmethod
    def _aggregate_records(records: List[Dict[str, Any]]) -> EvaluationMetrics:
        """把逐条评估记录聚合为 EvaluationMetrics

        兼容说明（P1-A）：正式报表改用 eval/retrieval_metrics.aggregate_records；
        本方法仅供旧管线（--legacy、BIRD runner 兼容字段）使用。PowerGrid
        记录的 column_* 字段已是新口径（全并集键级），旧口径保存在
        column_*_legacy 字段中，此处优先取旧口径以维持旧入口行为不变。
        """
        total = len(records)
        if total == 0:
            return EvaluationMetrics(total_queries=0)

        def mean(key: str, legacy_key: str = None) -> float:
            src = legacy_key if (legacy_key and legacy_key in records[0]) else key
            return float(np.mean([r.get(src, 0.0) for r in records])) if records else 0.0

        return EvaluationMetrics(
            table_accuracy=mean('table_exact_match'),
            sql_accuracy=mean('schema_sql_correct'),
            avg_query_time=mean('query_time'),
            avg_memory_mb=mean('memory_mb'),
            total_queries=total,
            table_recall=mean('table_recall_full'),
            column_recall=mean('column_recall', 'column_recall_legacy'),
            column_precision=mean('column_precision', 'column_precision_legacy'),
            column_f1=mean('column_f1', 'column_f1_legacy'),
            top1_table_recall=mean('table_recall_at_1'),
            top3_table_recall=mean('table_recall_at_3'),
            top5_table_recall=mean('table_recall_at_5'),
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
