import json
def _inject_supervision(train_df, gold, arm):
    """按监督口径为训练行注入结构化 gold 列（gt_utils 结构化格式）。

    arm: 'clean_projection'（顶层投影表列）| 'full_sql'（全部 SQL 所需表列）。
    原始 table/field 列保留；监督方法经 parse_ground_truth_* 优先读取结构化列。
    """
    df = train_df.copy()
    tables_col, fields_col = [], []
    for _, row in df.iterrows():
        qid = str(row['qid'])
        g = gold[qid]
        if arm == 'clean_projection':
            cols = g['projection_columns']
        else:
            cols = g['gold_columns']
        pairs = []
        tabs = []
        for c in cols:
            t, f = c.split('.', 1)
            pairs.append([t, f])
            if t not in tabs:
                tabs.append(t)
        tables_col.append(json.dumps(tabs, ensure_ascii=False))
        fields_col.append(json.dumps(pairs, ensure_ascii=False))
    df['gold_tables'] = tables_col
    df['gold_fields'] = fields_col
    return df
