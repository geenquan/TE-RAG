"""
数据集适配器模块

将外部数据集（BIRD 等）转换为实验框架标准化格式，
数据集差异集中在适配层，不散布到各个 Retriever。
"""

from datasets.base_adapter import BaseDatasetAdapter
from datasets.bird_adapter import BirdDatasetAdapter
from datasets import gt_utils
from datasets.gt_utils import (
    parse_ground_truth_tables,
    parse_ground_truth_fields,
    load_json_list,
    is_multi_table_row,
    has_structured_gold,
    gold_column_keys,
)

__all__ = [
    'BaseDatasetAdapter',
    'BirdDatasetAdapter',
    'gt_utils',
    'parse_ground_truth_tables',
    'parse_ground_truth_fields',
    'load_json_list',
    'is_multi_table_row',
    'has_structured_gold',
    'gold_column_keys',
]
