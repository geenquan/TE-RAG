"""
Ground Truth 解析工具

统一处理 PowerGrid（单表, legacy table/field 列）与 BIRD（多表, 结构化
gold_tables/gold_fields 列）两种 QA 标注格式，供检索器评估、图构建、
训练统计等模块复用，避免各处散落 `table.split('.')[-1]` 式解析。
"""

import json
import math
from typing import List, Tuple

import pandas as pd


def _get_cell(row, key: str):
    """安全读取行单元格（兼容 pandas.Series 与 dict），NaN/缺失返回 None"""
    try:
        if key not in row:
            return None
        value = row[key]
    except TypeError:
        return None
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def load_json_list(value) -> List:
    """
    把 gold_tables / gold_fields 单元格解析为 list

    兼容：原生 list（jsonl 读取）、JSON 字符串（CSV 读取）、NaN、'|' 分隔字符串
    """
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    if isinstance(value, float) and math.isnan(value):
        return []
    s = str(value).strip()
    if not s or s.lower() == 'nan':
        return []
    if s.startswith('['):
        try:
            parsed = json.loads(s)
            return parsed if isinstance(parsed, list) else []
        except (json.JSONDecodeError, ValueError):
            return []
    # legacy '|' 分隔字符串
    return [x.strip() for x in s.split('|') if x.strip()]


def parse_ground_truth_tables(row) -> List[str]:
    """
    解析一行 QA 数据的 gold 表列表（canonical table id）

    规则：
    1. 结构化 gold_tables 列（BIRD）：直接使用（canonical id，如 "db.table"）
    2. legacy table 列（PowerGrid）："db.table" → 取最后一段简单表名

    Args:
        row: QA 行（pandas.Series 或 dict）

    Returns:
        gold 表 id 列表（保持出现顺序）
    """
    gold = load_json_list(_get_cell(row, 'gold_tables'))
    if gold:
        seen = []
        for t in gold:
            t = str(t)
            if t and t not in seen:
                seen.append(t)
        return seen

    raw = _get_cell(row, 'table')
    if raw is None:
        return []
    parts = [p.strip() for p in str(raw).split('|') if p.strip()]
    return [p.split('.')[-1] for p in parts]


def parse_ground_truth_fields(row) -> List[Tuple[str, str]]:
    """
    解析一行 QA 数据的 gold 字段列表

    规则：
    1. 结构化 gold_fields 列（BIRD）：[table_id, column] 二元组列表（或 "table.column" 字符串）
    2. legacy field 列（PowerGrid）：裸字段名列表，附着到 gold 表（单表场景）

    Args:
        row: QA 行（pandas.Series 或 dict）

    Returns:
        [(table_id, field_name), ...]
    """
    gold = load_json_list(_get_cell(row, 'gold_fields'))
    if gold:
        out = []
        for item in gold:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                pair = (str(item[0]), str(item[1]))
            else:
                s = str(item)
                if '.' in s:
                    pair = tuple(s.rsplit('.', 1))
                else:
                    pair = ('', s)
            if pair[1]:
                out.append(pair)
        return out

    # legacy: 裸字段名 + 单一 gold 表
    tables = parse_ground_truth_tables(row)
    raw = _get_cell(row, 'field')
    if raw is None or not tables:
        return []
    fields = [f.strip() for f in str(raw).split('|') if f.strip()]
    return [(tables[0], f) for f in fields]


def is_multi_table_row(row) -> bool:
    """判断一行 QA 数据是否为多表查询"""
    return len(parse_ground_truth_tables(row)) > 1


def has_structured_gold(row) -> bool:
    """判断一行 QA 数据是否携带结构化 gold 标注（gold_tables / gold_fields）"""
    return bool(load_json_list(_get_cell(row, 'gold_tables'))) or \
        bool(load_json_list(_get_cell(row, 'gold_fields')))


def gold_column_keys(row) -> List[str]:
    """获取 gold 列的 canonical key 列表（"table.column"）"""
    return [f"{t}.{f}" for t, f in parse_ground_truth_fields(row) if t]
