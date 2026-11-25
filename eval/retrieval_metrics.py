"""
检索评估指标（P1-B 正式口径；P1-A 版本的历史语义见各兼容别名说明）

本模块是正式报表使用的指标唯一定义源。语义：

单表任务（PowerGrid）：
- Hit@K      : gold 表出现在按分数排序的 top-K 返回表中（布尔，K∈{1,3,5}）。
- Annotated-Field Precision/Recall/F1 : 规范化 "table.column" 键的集合级指标，
                 逐查询（macro）平均。预测列集合 = 全部 top-K 返回表携带列的
                 并集（每表列预算由检索器实际执行，见协议）；gold 列集合 =
                 "gold表简名.字段"，**仅为已有人工标注字段（SELECT 目标）**，
                 不含 WHERE/JOIN 等未标注字段。表未命中或检索失败自然产生 0。
- AFCS（Annotated-Field Coverage Success）: gold 表命中 top-K 且 gold 标注列
                 集合 ⊆ 预测列集合。名称明确限定为“已有标注字段覆盖”——不是
                 完整 SQL Schema 覆盖（gold 语义见 P1B_protocol.md）。

多表任务（BIRD）：
- Table Recall@K = |gold ∩ topK| / |gold|。
- All-Gold-Tables-Covered@K = gold ⊆ topK（子集覆盖，允许预测包含多余表，
  不得称为集合 Exact Match）。
- Column / 覆盖率指标同上。

通用（P1-B 失败处理规则）：
- gold 字段状态：present / missing（原始标注缺失）/ empty（解析后为空）/
  parse_error（解析异常）。
- gold 标注有效（present）的题：正常空预测或检索错误（status=error）时
  列 P/R/F1 与 AFCS **一律按 0 计入有效分母**，不得从分母中消失。
- 缺失标注/解析失败：与模型失败分开统计；不计为预测错误、不计为覆盖成功、
  不计入列级与 AFCS 分母。
- 预算记录：每题记录实际返回表数、每表列数与 budget_ok（表≤5、每表列≤10）。
- 检索异常：status=error，计入 Hit@K 等主指标分母（按 0 计），并单列计数。
- SQL 生成与执行：本模块不涉及，exec_accuracy 恒为 "not_run"，不填 0 或估计值。
- 兼容别名（旧字段，仅内部使用）：table_accuracy / table_recall ==
  单表 Hit@5；sql_accuracy == 旧 SCS 变体（空 gold 记为正确）；
  schema_coverage_success == AFCS（同名同值，正式名称为 AFCS）。

规范化规则：键统一小写、去首尾空白、去 "T:"/"C:" 前缀；表名取最后一个
"." 段（去掉 db 前缀）。该规则仅适用于当前单库 PowerGrid 设定
（简名唯一）；未来全库混检（跨 db 同名表）时此规则不适用，须改为
保留库名的完整标识——已记录，本阶段不改 BIRD 管线。
"""

from typing import Dict, List, Optional, Sequence, Tuple

K_VALUES = (1, 3, 5)


def normalize_table(name: str) -> str:
    """规范化表名：去前缀/空白/大小写，取最后一段（去 db 前缀）。"""
    if not name:
        return ''
    s = str(name).strip().lower()
    if s.startswith('t:'):
        s = s[2:]
    return s.split('.')[-1]


def normalize_column_key(key: str) -> str:
    """规范化列键 "table.column"：去前缀/空白/大小写，表名取最后一段。"""
    if not key:
        return ''
    s = str(key).strip().lower()
    if s.startswith('c:'):
        s = s[2:]
    parts = [p for p in s.split('.') if p]
    if len(parts) >= 2:
        table, col = parts[-2], parts[-1]
        return f"{table.split('.')[-1]}.{col}"
    return s


def classify_gold_fields(raw_value) -> Tuple[str, List[str]]:
    """分类 gold 字段标注状态。

    Returns:
        (status, fields)：status ∈ present/missing/empty/parse_error；
        fields 为原始字段名列表（仅 present 时非空）。
    """
    if raw_value is None:
        return 'missing', []
    try:
        import pandas as pd
        if pd.isna(raw_value):
            return 'missing', []
        if not isinstance(raw_value, str):
            return 'missing', []
        fields = [f.strip() for f in raw_value.split('|') if f.strip()]
        if not fields:
            return 'empty', []
        return 'present', fields
    except Exception:
        return 'parse_error', []


def hits_at_k(gold_table: str, retrieved_tables: Sequence[str],
              k_values: Sequence[int] = K_VALUES) -> Dict[int, bool]:
    """单表 Hit@K。"""
    gold = normalize_table(gold_table)
    ordered = [normalize_table(t) for t in retrieved_tables]
    return {k: bool(gold) and gold in ordered[:k] for k in k_values}


def table_recall_at_k(gold_tables: Sequence[str], retrieved_tables: Sequence[str],
                      k_values: Sequence[int] = K_VALUES) -> Dict[int, Optional[float]]:
    """多表 Table Recall@K = |gold ∩ topK| / |gold|；gold 空返回 None。"""
    gold = {normalize_table(t) for t in gold_tables if normalize_table(t)}
    if not gold:
        return {k: None for k in k_values}
    ordered = [normalize_table(t) for t in retrieved_tables]
    out = {}
    for k in k_values:
        topk = set(ordered[:k])
        out[k] = len(gold & topk) / len(gold)
    return out


def all_gold_covered(gold_tables: Sequence[str], retrieved_tables: Sequence[str],
                     k: int = 5) -> Optional[bool]:
    """All-Gold-Tables-Covered@K：gold ⊆ topK（子集覆盖，非 Exact Match）。

    gold 为空返回 None（不计入分母）。
    """
    gold = {normalize_table(t) for t in gold_tables if normalize_table(t)}
    if not gold:
        return None
    topk = {normalize_table(t) for t in retrieved_tables[:k]}
    return gold.issubset(topk)


def column_prf(gold_keys: Sequence[str], pred_keys: Sequence[str]) -> Tuple[
        Optional[float], Optional[float], Optional[float]]:
    """集合级 Column Precision/Recall/F1（P1-B 口径）。

    gold 非空（标注有效）时，预测为空或检索失败一律按 0 计——失败不得从
    有效标注分母中消失；gold 为空（缺失/解析失败标注）返回 (None,None,None)，
    不计入任何列级分母、也不计为模型错误。
    """
    gold = {normalize_column_key(k) for k in gold_keys if normalize_column_key(k)}
    pred = {normalize_column_key(k) for k in pred_keys if normalize_column_key(k)}
    if not gold:
        return None, None, None
    inter = gold & pred
    recall = len(inter) / len(gold)
    precision = len(inter) / len(pred) if pred else 0.0
    f1 = (2 * recall * precision / (recall + precision)) if (recall + precision) > 0 else 0.0
    return precision, recall, f1


def schema_coverage_success(gold_tables: Sequence[str], gold_column_keys: Sequence[str],
                            retrieved_tables: Sequence[str],
                            retrieved_column_keys: Sequence[str], k: int = 5) -> Optional[bool]:
    """SCS：gold 表全部进入 top-K 且 gold 列键被预测列键完全覆盖。

    gold 列集合为空（empty/missing/parse_error）时返回 None——
    空或缺失标注不能当作覆盖成功，也不计入 SCS 分母。
    """
    gold_t = {normalize_table(t) for t in gold_tables if normalize_table(t)}
    if not gold_t:
        return None
    gold_c = {normalize_column_key(c) for c in gold_column_keys if normalize_column_key(c)}
    if not gold_c:
        return None
    topk = [normalize_table(t) for t in retrieved_tables[:k]]
    if not gold_t.issubset(set(topk)):
        return False
    pred_c = {normalize_column_key(c) for c in retrieved_column_keys}
    return gold_c.issubset(pred_c)


def evaluate_single_table_query(gold_table: str, gold_fields_raw,
                                retrieved_tables: Sequence[str],
                                retrieved_column_keys: Sequence[str],
                                k_values: Sequence[int] = K_VALUES) -> Dict:
    """单表任务（PowerGrid）逐查询指标。

    Args:
        gold_fields_raw: gold 字段原始标注（'|' 分隔字符串或 None/NaN）。
        retrieved_tables: 按分数排序的返回表（原样，含前缀亦可）。
        retrieved_column_keys: 全部返回表携带列的 "table.column" 键并集。
    """
    status, fields = classify_gold_fields(gold_fields_raw)
    gold_keys = [f"{normalize_table(gold_table)}.{f.lower().strip()}" for f in fields]

    out = {
        'gold_field_status': status,
        'gold_column_keys': gold_keys,
        'hit': hits_at_k(gold_table, retrieved_tables, k_values),
        'column_precision': None, 'column_recall': None, 'column_f1': None,
        'annotated_field_coverage_success': None,
        'schema_coverage_success': None,  # 兼容别名（==AFCS）
    }
    if status == 'present':
        p, r, f1 = column_prf(gold_keys, retrieved_column_keys)
        out['column_precision'], out['column_recall'], out['column_f1'] = p, r, f1
        scs = schema_coverage_success(
            [gold_table], gold_keys, retrieved_tables, retrieved_column_keys,
            k=max(k_values))
        # 空预测按 0（不返回 None）：失败不得从有效分母消失
        out['annotated_field_coverage_success'] = bool(scs) if scs is not None else False
        out['schema_coverage_success'] = out['annotated_field_coverage_success']
    return out


def evaluate_multi_table_query(gold_tables: Sequence[str], gold_column_keys: Sequence[str],
                               retrieved_tables: Sequence[str],
                               retrieved_column_keys: Sequence[str],
                               k_values: Sequence[int] = K_VALUES) -> Dict:
    """多表任务（BIRD）逐查询指标。gold 列键为 "table.column" 形式。"""
    gold_keys = list(gold_column_keys)
    p, r, f1 = column_prf(gold_keys, retrieved_column_keys) if gold_keys else (None, None, None)
    scs = schema_coverage_success(
        gold_tables, gold_keys, retrieved_tables, retrieved_column_keys,
        k=max(k_values)) if gold_keys else None
    afcs = (bool(scs) if scs is not None else False) if gold_keys else None
    return {
        'gold_field_status': 'present' if gold_keys else 'missing',
        'gold_column_keys': gold_keys,
        'table_recall_at_k': table_recall_at_k(gold_tables, retrieved_tables, k_values),
        'all_gold_covered': {k: all_gold_covered(gold_tables, retrieved_tables, k)
                             for k in k_values},
        'column_precision': p, 'column_recall': r, 'column_f1': f1,
        'annotated_field_coverage_success': afcs,
        'schema_coverage_success': afcs,  # 兼容别名（==AFCS）
    }


def _mean(values: List[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


def aggregate_records(records: List[Dict]) -> Dict:
    """从逐题记录聚合方法级指标（macro）。

    主分母 = 该组全部测试题（status=error 按 0 计入 Hit 类指标）；
    列级指标与 SCS 的有效分母 = gold_field_status == 'present' 的题数，
    各类计数全部显式输出。所有输入必须是已落盘的 predictions 记录。
    """
    n_total = len(records)
    if n_total == 0:
        return {'n_total': 0}

    n_error = sum(1 for r in records if r.get('status') == 'error')
    status_counts = {}
    for r in records:
        s = r.get('gold_field_status', 'missing')
        status_counts[s] = status_counts.get(s, 0) + 1
    n_present = status_counts.get('present', 0)

    single = 'hit_at_1' in records[0]
    out = {
        'n_total': n_total,
        'n_ok': n_total - n_error,
        'n_error': n_error,
        'n_gold_present': n_present,
        'n_gold_missing': status_counts.get('missing', 0),
        'n_gold_empty': status_counts.get('empty', 0),
        'n_gold_parse_error': status_counts.get('parse_error', 0),
    }

    if single:
        for k in K_VALUES:
            hits = [1.0 if (r.get('status') == 'ok' and r.get(f'hit_at_{k}')) else 0.0
                    for r in records]
            hits_valid = [1.0 if r.get(f'hit_at_{k}') else 0.0
                          for r in records if r.get('status') == 'ok']
            out[f'hit@{k}'] = _mean(hits)
            out[f'hit@{k}_valid'] = _mean(hits_valid)
    else:
        for k in K_VALUES:
            vals, vals_valid = [], []
            for r in records:
                tr = r.get('table_recall_at_k', {})
                v = tr.get(str(k), tr.get(k))
                if v is not None:
                    vals.append(v if r.get('status') == 'ok' else 0.0)
                    vals_valid.append(v)
            out[f'table_recall@{k}'] = _mean(vals)
            out[f'table_recall@{k}_valid'] = _mean(vals_valid)
        def _cov5(r):
            agc = r.get('all_gold_covered') or {}
            return agc.get('5', agc.get(5))

        cov = [1.0 if (r.get('status') == 'ok' and _cov5(r)) else 0.0 for r in records]
        cov_valid = [1.0 if _cov5(r) else 0.0 for r in records if r.get('status') == 'ok']
        out['all_gold_covered@5'] = _mean(cov)
        out['all_gold_covered@5_valid'] = _mean(cov_valid)

    present = [r for r in records if r.get('gold_field_status') == 'present']
    # P1-B：gold 有效的题全部计入列级分母（空预测/错误按 0，不消失）
    out['afcs_denominator'] = len(present)
    for name in ('column_precision', 'column_recall', 'column_f1'):
        vals = [r.get(name) if r.get(name) is not None else 0.0 for r in present]
        out[name] = _mean(vals)
        out[f'{name}_denominator'] = len(present)
    afcs_vals = [1.0 if r.get('annotated_field_coverage_success',
                              r.get('schema_coverage_success')) else 0.0
                 for r in present]
    out['afcs'] = _mean(afcs_vals)
    # 预算记录：实际返回表数 / 列数 / 预算合规率（表≤5、每表列≤10）
    n_cols = [sum(len(t.get('columns', [])) for t in (r.get('retrieved_columns_detail') or []))
              for r in records]
    budget_ok = []
    for r in records:
        tables = r.get('retrieved_columns_detail') or []
        ok = (len(r.get('retrieved_tables') or []) <= 5
              and all(len(t.get('columns', [])) <= 10 for t in tables))
        budget_ok.append(1.0 if ok else 0.0)
    out['avg_columns_returned'] = _mean(n_cols)
    out['avg_tables_returned'] = _mean(
        [len(r.get('retrieved_tables') or []) for r in records])
    out['budget_ok_rate'] = _mean(budget_ok)
    out['exec_accuracy'] = 'not_run'
    return out
