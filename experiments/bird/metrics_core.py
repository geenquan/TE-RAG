"""Exact offline metric recomputation + per-record budget/scope gates.

The metric function is a faithful reimplementation of the frozen offline
recompute used across the project:
    bird_quick_v1/review_correction_v1_20261119_1550/tools/verify_inputs.py
    ::metric_values
(db-local last-segment table normalization, last-two-segment column
normalization, error records score zero on every metric).

tests/test_eval_v2_metrics.py pins parity against that reference file-by-file (400 randomized trials, re-pinned for the V2 copy);
any behavioral divergence is a test failure, not an opinion.

Gate semantics (evaluate.py treats violations as HARD errors - a method with
any violation is refused, never partially scored):
- qid uniqueness within a cell;
- the cell's qid set equals the manifest fold's test_qids EXACTLY;
- per record: at most 5 retrieved tables (raw list length);
- per record: at most 10 retrieved columns PER TABLE (from
  retrieved_columns_detail when present, else grouped by the table segment of
  the column keys);
- every retrieved table is inside the db's table_schema.csv (raw name, or
  last-segment-normalized name - identical outcome on real data).
"""
METRIC_KEYS = ['tr@1','tr@3','tr@5','covered@5','mrr','colp','colrec','colf1','fgcs']


# ------------------------------------------------------------ normalization
def norm_table(t):
    """Last-segment table normalization (reference metric_values::norm_t)."""
    s = str(t).strip().lower()
    return s.split('.')[-1] if s.count('.') >= 1 else s


def norm_column(c):
    """Last-two-segment column normalization (reference metric_values::norm_c).

    Note: the accepted reference strips the literal 'c:' ANYWHERE in the
    string (str.replace, not a prefix check). Kept bug-for-bug; the parity
    test pins this against the reference implementation.
    """
    s = str(c).strip().lower().replace('c:', '')
    parts = [p for p in s.split('.') if p]
    return f"{parts[-2]}.{parts[-1]}" if len(parts) >= 2 else s


# ------------------------------------------------------------ metrics
def metric_values(record, gold_entry):
    """Per-question metrics; EXACT reimplementation of the frozen reference.

    record: prediction record (dict) with status / retrieved_tables /
        retrieved_column_keys.
    gold_entry: {'gold_tables': [..], 'gold_columns': ['t.c', ...]} in the
        db-local simple-name view.
    Any record status other than 'ok' scores 0.0 on every metric and stays
    in the denominator.
    """
    if record.get('status') != 'ok':
        return {k: 0.0 for k in METRIC_KEYS}

    gold_t = {norm_table(t) for t in gold_entry['gold_tables']}
    pred_t = [norm_table(t) for t in record.get('retrieved_tables') or []]
    tr = {k: len(gold_t & set(pred_t[:k])) / len(gold_t) if gold_t else 0.0
          for k in (1, 3, 5)}
    cov = 1.0 if gold_t and gold_t.issubset(set(pred_t[:5])) else 0.0
    mrr = 0.0
    for i, t in enumerate(pred_t, 1):
        if t in gold_t:
            mrr = 1.0 / i
            break

    gold_c = {norm_column(c) for c in gold_entry['gold_columns']}
    pred_c = {norm_column(c) for c in record.get('retrieved_column_keys') or []}
    inter = gold_c & pred_c
    recall = len(inter) / len(gold_c) if gold_c else 0.0
    precision = len(inter) / len(pred_c) if pred_c else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    fgcs = 1.0 if (gold_t and gold_t.issubset(set(pred_t[:5])) and gold_c
                   and gold_c.issubset(pred_c)) else 0.0
    return {'tr@1': tr[1], 'tr@3': tr[3], 'tr@5': tr[5], 'covered@5': cov,
            'mrr': mrr, 'colp': precision, 'colrec': recall, 'colf1': f1,
            'fgcs': fgcs}


def actual_counts(record):
    """Actual returned sizes (for adaptive variants; disclosed, never hidden).

    Returns (n_tables, n_columns_total, per_table_columns: {table: n}).
    Error records count as 0. per_table_columns comes from
    retrieved_columns_detail when present; otherwise column keys are grouped
    by their table segment.
    """
    if record.get('status') != 'ok':
        return 0, 0, {}
    tables = list(record.get('retrieved_tables') or [])
    keys = list(record.get('retrieved_column_keys') or [])
    detail = record.get('retrieved_columns_detail') or []
    per_table = {}
    if detail:
        for td in detail:
            per_table[str(td.get('table'))] = len(td.get('columns') or [])
    else:
        for key in keys:
            t = norm_column(key).split('.')[0]
            per_table[t] = per_table.get(t, 0) + 1
    return len(tables), len(keys), per_table


# ------------------------------------------------------------ gates
def gate_problems(records, expected_qids, schema_tables):
    """Population / budget / db-scope violations for ONE (protocol, fold, db)
    cell of one method. Returns a list of human-readable problem strings;
    empty list means the cell passes.

    schema_tables: set of RAW table names from table_schema.csv (plus their
    normalized forms are accepted).
    """
    problems = []
    qids = [r.get('qid') for r in records]
    if len(qids) != len(set(qids)):
        seen, dup = set(), set()
        for q in qids:
            (dup if q in seen else seen).add(q)
        problems.append(f'duplicate qids: {sorted(str(d) for d in dup)[:5]}')
    if set(qids) != set(expected_qids):
        problems.append(
            'population mismatch '
            f'missing={sorted(set(expected_qids) - set(qids))[:3]} '
            f'extra={sorted(set(qids) - set(expected_qids))[:3]}')
    norm_schema = {norm_table(t) for t in schema_tables}
    for r in records:
        qid = r.get('qid')
        tables = r.get('retrieved_tables') or []
        if len(tables) > 5:
            problems.append(f'{qid}: {len(tables)} tables > 5 budget')
        for t in tables:
            if t not in schema_tables and norm_table(t) not in norm_schema:
                problems.append(f'{qid}: table {t!r} outside db schema')
        detail = r.get('retrieved_columns_detail') or None
        if detail:
            for td in detail:
                n = len(td.get('columns') or [])
                if n > 10:
                    problems.append(
                        f'{qid}: {n} columns > 10 in table {td.get("table")!r}')
        else:
            _, _, per_table = actual_counts(r)
            for t, n in sorted(per_table.items()):
                if n > 10:
                    problems.append(f'{qid}: {n} columns > 10 in table {t!r}')
    return problems
