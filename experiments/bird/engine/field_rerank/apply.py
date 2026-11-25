"""Execution adapter: apply a direction-B config to a frozen-engine record.

FINAL-RUN CONTRACT (for the eval pipeline / any consumer):
  1. run the frozen engine normally:
         record, metric, trace = engine.predict(qid, params, variant)
     (params carries the engine axes; unknown-to-engine keys are ignored BY
     THE ENGINE, so the layer below MUST be applied explicitly)
  2. apply this function:
         from experiments.bird.engine.field_rerank.apply import apply_config_to_record
         record = apply_config_to_record(record, params)
     It reads params['field_rerank'] (validated; absent => record unchanged)
     and stamps record['field_rerank_applied'] with the spec digest so any
     consumer can machine-check that the layer ran (absent marker on a
     method whose params declare field_rerank = INVALID output, refuse).

The gate rule implemented here is the SAME semantics as
engine.field_rerank.field_filter (k_cap/k_low/table_gate/min_k; keep_best),
applied on the engine record's retrieved_columns_detail (table_score and
ordered columns are all in the record). tests/test_field_parity.py pins the
two implementations together on snapshot data.
"""
from .replay import normalize_key_of

GATE_KEYS = ('k_cap', 'k_low', 'table_gate', 'table_gate_op', 'table_gate_tau',
             'min_k', 'empty_policy')


def field_rerank_spec(params):
    """Validated params['field_rerank'] dict or None (absent)."""
    spec = (params or {}).get('field_rerank')
    if spec is None:
        return None
    if not isinstance(spec, dict):
        raise ValueError('field_rerank spec must be a dict')
    for k in spec:
        if k not in GATE_KEYS:
            raise KeyError(f'unknown field_rerank key: {k}')
    out = {'k_cap': spec.get('k_cap', 10), 'k_low': spec.get('k_low'),
           'table_gate': spec.get('table_gate', 'table_score'),
           'table_gate_op': spec.get('table_gate_op', 'lt'),
           'table_gate_tau': spec.get('table_gate_tau', 0.0),
           'min_k': spec.get('min_k', 1),
           'empty_policy': spec.get('empty_policy', 'keep_best')}
    if out['k_cap'] > 10:
        raise ValueError('k_cap exceeds per-table budget')
    if out['table_gate'] not in ('table_score', 'bm25_score', 'graph_score',
                                 'pattern_score', 'tscore_ratio'):
        raise ValueError(f'unknown table_gate {out["table_gate"]!r}')
    if out['empty_policy'] not in ('keep_best', 'drop_table'):
        raise ValueError('empty_policy must be keep_best|drop_table')
    return out


def _gate_value(t, gate, tscore_top):
    if gate == 'table_score':
        return t['table_score']
    if gate == 'tscore_ratio':
        return (t['table_score'] / tscore_top) if tscore_top else 0.
    return 0.  # channel gates unsupported on bare records (channels not in
    # the record); documented limitation — use table_score/tscore_ratio here


def apply_config_to_record(record, params):
    """Apply params['field_rerank'] to an engine record (in-place-safe)."""
    spec = field_rerank_spec(params)
    if spec is None:
        return record
    details = record.get('retrieved_columns_detail')
    if not details:
        return record
    tscore_top = max((t['table_score'] for t in details), default=0.)
    out_details = []
    keys = []
    counts = {}
    dropped = []
    for t in details:
        k = spec['k_cap']
        v = _gate_value(t, spec['table_gate'], tscore_top)
        gated = v < spec['table_gate_tau'] if spec['table_gate_op'] == 'lt' \
            else v >= spec['table_gate_tau']
        if spec['k_low'] is not None and gated:
            k = min(k, spec['k_low'])
        cols = t['columns'][:k]
        if not cols and spec['empty_policy'] == 'drop_table':
            dropped.append(t['table'])
            continue
        if not cols:
            cols = t['columns'][:max(1, spec['min_k'])]
        out_details.append({'table': t['table'], 'table_score': t['table_score'],
                            'columns': cols})
        counts[t['table']] = len(cols)
        keys.extend(c['key'] for c in cols)
    if len(out_details) > 5 or any(len(t['columns']) > 10 for t in out_details):
        raise RuntimeError('Budget expanded by field_rerank apply')
    out = dict(record)
    out['retrieved_tables'] = [t['table'] for t in out_details]
    out['retrieved_column_keys'] = keys
    out['retrieved_columns_detail'] = out_details
    out['budget_actual'] = {'tables': len(out_details),
                            'columns_by_table': counts,
                            'dropped_tables': dropped,
                            'mean_columns_per_q': sum(counts.values())
                            / max(len(details), 1)}
    import hashlib
    import json
    out['field_rerank_applied'] = hashlib.sha256(json.dumps(
        spec, sort_keys=True).encode()).hexdigest()[:16]
    return out
