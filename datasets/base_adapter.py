"""
数据集适配器基类

定义外部数据集 → 实验框架标准化格式（table_df / field_df / qa_df）的转换接口。
数据集差异应集中在适配层，而不是散布到各个 Retriever 中。

标准输出格式与 PowerGrid 保持一致：
- table_df:  table, db, table_desc, field, schema
- field_df:  table, db, field_name, field_name_desc, field_type
- qa_df:     question, table, field, sql（BIRD 额外携带 db_id / gold_tables / gold_fields 等结构化列）
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import pandas as pd


class BaseDatasetAdapter(ABC):
    """数据集适配器抽象基类"""

    @abstractmethod
    def load_queries(self) -> List[Dict]:
        """加载原始查询数据"""
        pass

    @abstractmethod
    def load_schema(self) -> Dict:
        """加载原始 schema 数据"""
        pass

    @abstractmethod
    def build_table_dataframe(self) -> pd.DataFrame:
        """
        构建标准化表 schema DataFrame

        Returns:
            包含 table / db / table_desc / field / schema 列的 DataFrame
        """
        pass

    @abstractmethod
    def build_field_dataframe(self) -> pd.DataFrame:
        """
        构建标准化字段 schema DataFrame

        Returns:
            包含 table / db / field_name / field_name_desc / field_type 列的 DataFrame
        """
        pass

    @abstractmethod
    def build_qa_dataframe(self) -> pd.DataFrame:
        """
        构建标准化 QA DataFrame

        Returns:
            包含 question / table / field / sql 列的 DataFrame
            （多表数据集额外携带结构化 gold_tables / gold_fields 列）
        """
        pass

    def prepare(self) -> Dict[str, pd.DataFrame]:
        """
        执行完整的数据准备流程

        Returns:
            {'table_df': ..., 'field_df': ..., 'qa_df': ...}
        """
        self.load_schema()
        self.load_queries()
        return {
            'table_df': self.build_table_dataframe(),
            'field_df': self.build_field_dataframe(),
            'qa_df': self.build_qa_dataframe(),
        }
