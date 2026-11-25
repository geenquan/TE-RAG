"""
SQL 上下文构造（P2-D3 建立 / 正式实验前修正）

把一次检索结果整理为下游 SQL 生成所需的结构化上下文：
- 问题、表列及**真实 Schema 描述**（预算内实际返回的列，不暗中扩展）
- 命中的注解（B 直接 / C 迁移，含来源 qid）——只绑定实际返回的列
- 谓词槽位（操作符+值类型；**训练日期/机构实例不成为新问题默认条件**，
  一律以槽位形式给出；CODE 不给值）
- 表关系证据，分两级：
  * join_conditions：仅来自**训练 SQL 的 AST 连接谓词**（a.x=b.y）——
    同名 id 不构成连接依据；
  * cooccurrence：两表在训练 SQL 中同现（仅作参考证据，不是连接条件）。

不调用 LLM、不依 gold 补表列、不扩大预算；输出保存供后续阶段使用。
"""

from typing import Dict, List


def _train_join_evidence(train_data, schema_cols) -> Dict:
    """训练 SQL 的 AST 连接谓词：{(t1,c1,t2,c2) -> 支持题数}。"""
    from terag.sql_role_parser import SQLRoleParser
    import sqlglot
    from sqlglot import exp as _exp
    evidence = {}
    if train_data is None:
        return evidence
    parser = SQLRoleParser(dialect='mysql')
    for _, row in train_data.iterrows():
        pr = parser.parse(str(row.get('sql', '')), schema_columns=schema_cols)
        if not pr.ast_parsed:
            continue
        sql = str(row.get('sql', ''))
        tree = None
        for d in ('mysql', 'hive'):
            try:
                tree = sqlglot.parse_one(sql, dialect=d)
                break
            except Exception:  # noqa: BLE001
                continue
        if tree is None:
            continue
        # 每个 SELECT 作用域内的别名->物理表
        for select in tree.find_all(_exp.Select):
            tables = {}
            frm = select.args.get('from_') or select.args.get('from')
            sources = [frm.this] if frm is not None else []
            sources += [j.this for j in (select.args.get('joins') or [])]
            for src in sources:
                if isinstance(src, _exp.Table) and src.name:
                    tables[(src.alias or src.name).lower()] = src.name
            n_phys = len({t for t in tables.values()})
            for join in (select.args.get('joins') or []):
                on = join.args.get('on')
                if on is None:
                    continue
                for eq in on.find_all(_exp.EQ):
                    cols = [c for c in eq.find_all(_exp.Column)]
                    if len(cols) != 2:
                        continue
                    resolved = []
                    for c in cols:
                        if c.table:
                            resolved.append(tables.get(c.table.lower()))
                        elif n_phys == 1:
                            resolved.append(next(iter(
                                {t for t in tables.values()})))
                        else:
                            resolved.append(None)
                    if all(resolved) and resolved[0] != resolved[1]:
                        key = (resolved[0], cols[0].name,
                               resolved[1], cols[1].name)
                        evidence[key] = evidence.get(key, 0) + 1
    return evidence


def build_sql_context(query: str, results: List, retriever=None) -> Dict:
    """整理检索结果为 SQL 生成上下文（见模块 docstring）。"""
    miner = getattr(retriever, 'miner', None)
    transfer = getattr(retriever, 'transfer_result', None)
    train_data = getattr(retriever, 'train_data', None)
    field_df = getattr(retriever, 'field_df', None)

    # 列描述（真实 Schema；只给实际返回列）
    desc_map = {}
    if field_df is not None:
        for _, r in field_df.iterrows():
            desc_map[(str(r['table']).lower(),
                      str(r['field_name']).lower())] = str(
                r.get('field_name_desc', '') or '')

    schema_cols = miner.schema_columns if miner is not None else {}
    join_ev = _train_join_evidence(train_data, schema_cols)

    # 训练同现（参考证据，非连接条件）
    co_occur = set()
    if miner is not None and train_data is not None:
        from terag.sql_role_parser import SQLRoleParser
        parser = SQLRoleParser(dialect='mysql')
        for _, row in train_data.iterrows():
            pr = parser.parse(str(row.get('sql', '')),
                              schema_columns=schema_cols)
            if not pr.ast_parsed:
                continue
            tabs = pr.get_gold_tables()
            for i, t1 in enumerate(tabs):
                for t2 in tabs[i + 1:]:
                    co_occur.add((min(t1, t2), max(t1, t2)))

    c_terms_by_target = {}
    if transfer is not None:
        for r in transfer.get('accepted', []):
            if r.get('status') == 'accepted' and r.get('source_type') == 'TERM' \
                    and isinstance(r.get('adapted_content'), str):
                c_terms_by_target.setdefault(r['target_element'], []).append(
                    {'term': r['adapted_content'],
                     'source': r['source_element'],
                     'sim': r['similarity']['total']})

    returned_tables = [r.table for r in results]
    tables_out = []
    for r in results:
        t = r.table
        returned_cols = {c.split('.')[-1] for c, _ in r.columns}
        cols = []
        for c, s in r.columns:
            fname = c.split('.')[-1]
            cols.append({'name': fname,
                         'desc': desc_map.get((t.lower(), fname.lower()), ''),
                         'score': round(float(s), 4)})

        # 注解只绑定实际返回的列
        anns = []
        if miner is not None:
            for a in miner.annotations:
                if a['type'] == 'TERM' and a.get('status') == 'confirmed':
                    tgt = a.get('target', '')
                    tt, tc = tgt.split('.', 1) if '.' in tgt else (tgt, '')
                    if tt == t.lower() and tc in {c.lower()
                                                  for c in returned_cols}:
                        anns.append({'term': str(a['content']),
                                     'column': tc,
                                     'channel': 'B',
                                     'rule': a.get('rule'),
                                     'source_qids': a.get('source_qids', [])})
        for col_key, items in c_terms_by_target.items():
            tt, tc = col_key.split('.', 1)
            if tt == t.lower() and tc in {c.lower() for c in returned_cols}:
                for it in items:
                    anns.append({'term': it['term'], 'column': tc,
                                 'channel': 'C',
                                 'source_element': it['source'],
                                 'sim': it['sim']})

        # 谓词槽位：只绑实际返回列；训练日期/机构实例 → 槽位（值不给）
        slots = []
        if miner is not None:
            for a in miner.annotations:
                if a['type'] == 'PREDICATE' and a.get('status') == 'confirmed':
                    tgt = a.get('target', '')
                    tt, tc = tgt.split('.', 1) if '.' in tgt else (tgt, '')
                    if tt == t.lower() and tc in {c.lower()
                                                  for c in returned_cols}:
                        c_ = a['content']
                        slots.append({'column': tc,
                                      'operator': c_['operator'],
                                      'value_class': c_['value_class'],
                                      'value': None,
                                      'note': ('训练实例值不作为新问题默认'
                                               '条件；CODE 含义待目标证据')})

        # 连接条件（仅 AST 连接谓词）+ 同现参考
        join_conditions = []
        for (t1, c1, t2, c2), sup in join_ev.items():
            if t1 == t.lower() and t2 in [x.lower() for x in returned_tables]:
                join_conditions.append({'left': f"{t1}.{c1}",
                                        'right': f"{t2}.{c2}",
                                        'support': sup})
            elif t2 == t.lower() and t1 in [x.lower() for x in returned_tables]:
                join_conditions.append({'left': f"{t2}.{c2}",
                                        'right': f"{t1}.{c1}",
                                        'support': sup})
        cooccurrence = []
        for other in returned_tables:
            if other == t:
                continue
            pair = (min(t.lower(), other.lower()),
                    max(t.lower(), other.lower()))
            if pair in co_occur:
                cooccurrence.append({'table': other,
                                     'evidence': 'cooccurrence'})

        tables_out.append({'table': t,
                           'score': round(float(r.table_score), 4),
                           'columns': cols,
                           'annotations': anns,
                           'predicate_slots': slots,
                           'join_conditions': join_conditions,
                           'cooccurrence': cooccurrence})

    return {
        'query': query,
        'tables': tables_out,
        'budget_note': ('表≤5、每表列≤10（检索器实际执行的预算）；'
                        '本上下文未调用 LLM，ExecAcc=not_run'),
        'annotation_channels': {'B': miner is not None,
                                'C': transfer is not None},
    }
