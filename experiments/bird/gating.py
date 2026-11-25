"""THIN post-hoc score-transform layer for the H search (Direction A).

Operates on ENGINE CHANNEL OUTPUTS ONLY. engine/*.py is byte-frozen v1 and is
never edited; this module calls its public methods:

  engine.table_features(q, p, variant) -> per-node {bm25_score, graph_score,
      pattern_score} (post table_graph_norm/table_pattern_norm transforms and
      optional table_normalization), terms, ledger, index key
  engine.bmscores(q, p, variant)      -> raw augmented BM25 per node
  engine.raw_fields(q, table)         -> per-field feature dicts
  engine.field_graph(q, table, field, p), engine.patterns_for(q, p)
  engine.base.patterns                -> node -> [pattern objs] (read-only
      attribute access for pattern support counts)

Engine-level state reads (documented precisely, per task instruction): the
query-vs-train token Jaccard similarities are recomputed here from
  engine.extractor._get_query_tokens(query_text)
  engine.extractor._train_tokens_cache      (idx -> train token set)
  engine.extractor._train_gold_cache        (idx -> (gold tables, columns))
which is EXACTLY the state the engine's own graph_table (bird_engine.py:512)
and the vnext graph_margin gate (vnext_engine.py:342) consume. Read-only.

Reliability signals are INFERENCE-VISIBLE ONLY: query-train match
strength/margin/support counts, channel score distributions/concentration,
schema BM25 strength. No gold/test-table truth, no qid/db identity enters any
gate. The gold of the evaluated question is touched ONLY by the metric
functions at the end (same as engine.predict).

Modes (hg namespace, validated; all default = engine-identical 'linear'):
  linear    : engine's weighted sum; optional per-table gates/caps on the
              history channels (graph/pattern)
  residual  : normalized BM25 base + capped, gated history residual
              score = bm_norm + rw_graph*min(gate*graph_norm, cap) +
                      rw_pattern*min(gate*pattern_norm, cap)
  rank      : reciprocal-rank fusion over per-channel rankings
              score = sum_c w_c / (rrf_k + rank_c(t))
Per-question channel fallback (metadata fallback under weak history): a
history channel is zeroed for the WHOLE question when its best signal across
tables is weak (graph: best train similarity < fb_graph_tau; pattern: best raw
pattern score < fb_pattern_tau).

Field stage: verbatim port of bird_engine.py predict lines 591-635 (base
path; vnext gates must be OFF here — enforced) so columns are identical to the
engine's when the table order coincides (test-pinned bit-exact).

Usage: predict_gated(engine, qid, params, variant, hg) ->
       (record, metric, trace) with the exact engine record/metric shapes.
"""
import collections
import math

from experiments.bird.engine.study_defaults import switches, DEFAULT  # noqa: F401 (DEFAULT re-export)
from experiments.bird.engine.bird_engine import match_strength, transform

HG_KEYS = {
    'mode': ('linear', 'residual', 'rank'),
    'norm': ('zscore', 'rank', 'max'),           # channel norm in residual mode
    'rw_graph': float, 'rw_pattern': float,       # residual weights
    'rrf_k': int,                                 # rank-fusion constant
    'graph_gate_margin': float,                   # per-table: need best_sim >= theta_g + m
    'graph_min_support': int,                     # per-table: need support >= k
    'pattern_min_score': float,                   # per-table: need raw pattern >= s
    'cap_graph_rel': float,                       # graph term <= r * bm term
    'cap_pattern_rel': float,                     # pattern term <= r * bm term
    'cap_graph_abs': float,                       # absolute cap on normed graph term
    'cap_pattern_abs': float,
    'fb_graph_tau': float,                        # question-level graph fallback
    'fb_pattern_tau': float,                      # question-level pattern fallback
}
HG_DEFAULT = {
    'mode': 'linear', 'norm': 'zscore', 'rw_graph': 1.0, 'rw_pattern': 1.0,
    'rrf_k': 60, 'graph_gate_margin': 0.0, 'graph_min_support': 0,
    'pattern_min_score': 0.0, 'cap_graph_rel': 0.0, 'cap_pattern_rel': 0.0,
    'cap_graph_abs': 0.0, 'cap_pattern_abs': 0.0, 'fb_graph_tau': 0.0,
    'fb_pattern_tau': 0.0,
}


class HGError(RuntimeError):
    pass


def require(cond, msg):
    if not cond:
        raise HGError(msg)


def merge_hg(raw):
    """Validated hg vector (defaults off = engine-identical)."""
    out = dict(HG_DEFAULT)
    for k, v in (raw or {}).items():
        require(k in HG_DEFAULT, f'unknown hg key: {k}')
        if k == 'mode':
            require(v in HG_KEYS['mode'], f'bad hg.mode {v!r}')
        elif k == 'norm':
            require(v in HG_KEYS['norm'], f'bad hg.norm {v!r}')
        elif k == 'rrf_k' or k == 'graph_min_support':
            require(isinstance(v, int) and v >= 0, f'bad hg.{k}')
        else:
            require(isinstance(v, (int, float)) and not isinstance(v, bool)
                    and v >= 0. and math.isfinite(v), f'bad hg.{k}')
        out[k] = v
    return out


def hg_is_identity(hg):
    """True when every knob is at its engine-identical default."""
    hg = merge_hg(hg)
    return (hg['mode'] == 'linear' and hg['graph_gate_margin'] == 0.
            and hg['graph_min_support'] == 0 and hg['pattern_min_score'] == 0.
            and hg['cap_graph_rel'] == 0. and hg['cap_pattern_rel'] == 0.
            and hg['cap_graph_abs'] == 0. and hg['cap_pattern_abs'] == 0.
            and hg['fb_graph_tau'] == 0. and hg['fb_pattern_tau'] == 0.)


# ---------------------------------------------------------------- signals
def graph_signals(engine, q, p):
    """Per-table (best_sim, support, vote) from query-vs-train Jaccard.

    Read-only recomputation over the same extractor caches the engine's
    graph_table consumes (see module docstring). Inference-visible: train
    similarities/supports only.
    """
    tokens = engine.extractor._get_query_tokens(engine.query_text[q])
    best, support, votes = (collections.defaultdict(float),
                            collections.defaultdict(int),
                            collections.defaultdict(float))
    for idx, hist in engine.extractor._train_tokens_cache.items():
        union = len(tokens | hist)
        sim = len(tokens & hist) / union if union else 0.
        if sim > p['theta_g']:
            for table in engine.extractor._train_gold_cache[idx][0]:
                node = 'T:' + table
                if sim > best[node]:
                    best[node] = sim
                support[node] += 1
                votes[node] += sim * 3.
    return dict(best), dict(support), dict(votes)


def pattern_raw_scores(engine, q, p):
    """Raw per-node pattern scores (pre table_pattern_norm transform)."""
    return {n: s for n, s in engine.patterns_for(q, p).items() if s > 0}


def pattern_support(engine, q, p):
    """Max pattern support count per node (train occurrence counts)."""
    out = {}
    scores = engine.patterns_for(q, p)
    for node, pats in engine.base.patterns.items():
        if scores.get(node, 0.) <= 0.:
            continue
        out[node] = max(int(getattr(pat, 'count', 1)) for pat in pats) if pats else 0
    return out


# ---------------------------------------------------------------- channels
def _channel_norm(values, mode):
    """Normalize positive evidence over tables (engine transform semantics)."""
    return transform({n: v for n, v in values.items()}, mode)


def _ranks(values):
    """Descending rank per node (1 = best); deterministic name tiebreak."""
    order = sorted(values, key=lambda n: (-values[n], n))
    return {n: i + 1 for i, n in enumerate(order)}


# ---------------------------------------------------------------- table stage
def table_scores(engine, q, p, variant, hg, trace_out=None):
    """Fused table scores + channel internals (reliability-gated)."""
    a = switches(p, variant)
    features, terms, ledger, key = engine.table_features(q, p, variant)
    nodes = sorted(features)
    bm = {n: features[n]['bm25_score'] for n in nodes}
    gr = {n: features[n]['graph_score'] for n in nodes if a['TableGraph']}
    pa = {n: features[n]['pattern_score'] for n in nodes if a['Pattern']}

    # reliability signals (inference-visible only)
    g_best, g_support, _votes = graph_signals(engine, q, p)
    p_raw = pattern_raw_scores(engine, q, p)

    # per-question channel fallback under weak history
    if hg['fb_graph_tau'] > 0.:
        strong = max(g_best.values(), default=0.)
        if strong < hg['fb_graph_tau']:
            gr = {}
    if hg['fb_pattern_tau'] > 0.:
        strong = max(p_raw.values(), default=0.)
        if strong < hg['fb_pattern_tau']:
            pa = {}

    # per-table gates on history channels
    if hg['graph_gate_margin'] > 0. or hg['graph_min_support'] > 0:
        gr = {n: v for n, v in gr.items()
              if g_best.get(n, 0.) >= p['theta_g'] + hg['graph_gate_margin']
              and g_support.get(n, 0) >= hg['graph_min_support']}
    if hg['pattern_min_score'] > 0.:
        pa = {n: v for n, v in pa.items() if p_raw.get(n, 0.) >= hg['pattern_min_score']}

    tw = list(p['table_weights'])
    if not a['TableGraph']:
        tw[1] = 0.
    if not a['Pattern']:
        tw[2] = 0.
    scale = sum(tw) or 1.
    tw = [w / scale for w in tw]

    if hg['mode'] == 'linear':
        def cap_rel(v, base, rel, absolute):
            if rel > 0. and base > 0.:
                v = min(v, rel * base)
            if absolute > 0.:
                v = min(v, absolute)
            return v
        scores = {}
        for n in nodes:
            g = cap_rel(gr.get(n, 0.), bm.get(n, 0.), hg['cap_graph_rel'],
                        hg['cap_graph_abs'])
            pp = cap_rel(pa.get(n, 0.), bm.get(n, 0.), hg['cap_pattern_rel'],
                         hg['cap_pattern_abs'])
            scores[n] = tw[0] * bm.get(n, 0.) + tw[1] * g + tw[2] * pp
    elif hg['mode'] == 'residual':
        bmn = _channel_norm(bm, hg['norm'])
        grn = _channel_norm(gr, hg['norm'])
        pan = _channel_norm(pa, hg['norm'])
        scores = {}
        for n in nodes:
            g = grn.get(n, 0.)
            pp = pan.get(n, 0.)
            if hg['cap_graph_abs'] > 0.:
                g = min(g, hg['cap_graph_abs'])
            if hg['cap_pattern_abs'] > 0.:
                pp = min(pp, hg['cap_pattern_abs'])
            scores[n] = bmn.get(n, 0.) + hg['rw_graph'] * g + hg['rw_pattern'] * pp
    else:  # rank
        rbm = _ranks(bm)
        rg = _ranks(gr)
        rp = _ranks(pa)
        k = max(1, int(hg['rrf_k']))
        scores = {}
        for n in nodes:
            s = tw[0] / (k + rbm.get(n, k + 1))
            if a['TableGraph'] and n in rg:
                s += tw[1] / (k + rg[n])
            if a['Pattern'] and n in rp:
                s += tw[2] / (k + rp[n])
            scores[n] = s
    if trace_out is not None:
        trace_out.update({'features': features, 'bm': bm, 'graph': gr,
                          'pattern': pa, 'graph_best_sim': dict(g_best),
                          'graph_support': dict(g_support),
                          'pattern_raw': p_raw, 'table_weights': tw})
    return scores, terms, ledger, key


# ---------------------------------------------------------------- field stage
def _field_table_block(engine, q, p, a, terms, table):
    """Verbatim port of bird_engine.py predict field block (base path) for ONE
    table. Returns (columns list [(key, normalized_score)...] in rank order,
    per-field detail for trace). vnext gates must be off (asserted by caller).
    """
    from terag.tokenization import tokenize_text  # engine's own tokenizer path
    from eval.retrieval_metrics import normalize_column_key
    query = engine.query_text[q]
    query_terms = [w for w in engine.extractor._tokenize(query) if len(w) > 1]
    tokens = set(query_terms)
    raw = engine.raw_fields(q, table)
    fields = {f: dict(v) for f, v in raw.items()}
    for field, v in fields.items():
        key = f'{table}.{field}'.lower()
        bh = ch = 0.
        for term, weight, origin in terms.get(key, []):
            strength = match_strength(term, query, tokens,
                                      p['b_match'] if origin == 'B' else p['c_match'],
                                      engine.language)
            if origin == 'B':
                bh += weight * strength * p['b_hit_weight']
            else:
                ch += weight * strength * p['c_hit_weight']
        v['BHit'] = bh
        v['CHit'] = ch
        v['annotation_hit'] = bh + ch
        v['graph_propagation'] = engine.field_graph(q, table, field, p) if a['FieldGraph'] else 0.
        if not a['Pattern']:
            v['role_prior'] = 0.
        elif p['pattern_aggregation'] != 'max' or p['pattern_smoothing'] != 0:
            v['role_prior'] = engine.patterns_for(q, p).get(
                f'C:{table}.{str(field).lower()}', 0.) / .6
        if p['description_scale'] != 1.:
            nk = (q, table, field)
            if nk not in engine.caches['name_only']:
                engine.caches['name_only'][nk] = engine.extractor._compute_direct_match_score(
                    field, '', query_terms, tokens)
            name = engine.caches['name_only'][nk]
            v['direct_match'] = name + p['description_scale'] * (v['direct_match'] - name)
        v['direct_match'] *= p['direct_scale']
    fg = transform({f: v['graph_propagation'] for f, v in fields.items()}, p['field_graph_norm'])
    pat = transform({f: v['role_prior'] for f, v in fields.items()}, p['field_pattern_norm'])
    for f in fields:
        fields[f]['graph_propagation'] = fg[f]
        fields[f]['role_prior'] = pat[f]
    field_names = ['direct_match', 'graph_propagation', 'role_prior', 'train_recommend']
    if p['field_normalization'] == 'max':
        for name in field_names:
            maximum = max((v[name] for v in fields.values()), default=0.) or 1.
            for v in fields.values():
                v[name] /= maximum
    fw = list(p['field_weights'])
    if not a['FieldGraph']:
        fw[1] = 0.
    if not a['Pattern']:
        fw[2] = 0.
    raw_scores = {f: sum(w * v[k] for k, w in zip(field_names, fw)) + v['BHit'] + v['CHit']
                  for f, v in fields.items()}
    denominator = sum(fw) + (p['b_hit_weight'] if a['BHit'] else 0.) + (p['c_hit_weight'] if a['C'] else 0.)
    field_scores = {f: v / (denominator or 1.) for f, v in raw_scores.items()}
    fo = sorted(raw_scores, key=lambda f: (-raw_scores[f], f))
    columns = [{'key': normalize_column_key(table + '.' + f),
                'score': float(field_scores[f])} for f in fo[:10]]
    return columns, {'table': table, 'fields': fields, 'raw_total': raw_scores,
                     'ordered_fields': fo, 'field_weights': fw}


# ---------------------------------------------------------------- predict
def predict_gated(engine, q, p=DEFAULT, variant='Full', hg=None, trace=False):
    """Engine-contract predict with the post-hoc fusion layer applied.

    Returns (record, metric, trace_dict) with the exact engine shapes. With
    merge_hg(hg) at identity AND no active vnext gates, output is bit-identical
    to BirdEngine.predict (test-pinned).
    """
    from experiments.bird.engine.vnext_engine import vnext_flags, _any_gate
    from eval.full_gold import full_gold_metrics
    vn = vnext_flags(p)
    require(not _any_gate(vn), 'gating layer supports base path only (vnext gates off)')
    require(variant == 'Full' or variant in ('noTableGraph', 'noPattern', 'noGraph',
                                             'noBIndex', 'noBHit', 'noB', 'noFieldGraph',
                                             'noC'),
            f'variant {variant!r} not supported by the gating layer')
    hg = merge_hg(hg)
    a = switches(p, variant)

    scores, terms, ledger, key = table_scores(engine, q, p, variant, hg)
    ordered = sorted(scores, key=lambda n: (-scores[n], n))
    tables_kept = ordered[:5]

    details = []
    ft = []
    ann_count = 0
    c_count = 0
    for node in tables_kept:
        table = node[2:]
        columns, ftr = _field_table_block(engine, q, p, a, terms, table)
        ann_count += sum(1 for v in ftr['fields'].values() if v['annotation_hit'] > 0)
        c_count += sum(1 for v in ftr['fields'].values() if v['CHit'] > 0)
        details.append({'table': table, 'table_score': float(scores[node]),
                        'columns': columns})
        if trace:
            ft.append(ftr)
    record = {'qid': q, 'fold': engine.fold, 'setting': engine.setting,
              'variant': variant, 'status': 'ok',
              'retrieved_tables': [d['table'] for d in details],
              'retrieved_column_keys': [c['key'] for d in details for c in d['columns']],
              'retrieved_columns_detail': details,
              'active_TERM_fields': ann_count, 'active_C_TERM_fields': c_count,
              'budget_actual': {'tables': len(details),
                                'columns_by_table': {d['table']: len(d['columns'])
                                                     for d in details}},
              'hg': dict(hg)}
    require(len(details) <= 5 and all(len(d['columns']) <= 10 for d in details),
            'Budget expanded')
    m = full_gold_metrics(record, engine.gold)
    ranks = {n: i + 1 for i, n in enumerate(ordered)}
    gt = engine.gold[q]['gold_tables']
    metric = {'qid': q, 'valid': engine.gold[q]['gold_status'] == 'verified_complete',
              'covered@5': float(m[1]),
              **{f'tr@{k}': float(m[0][k]) for k in [1, 3, 5]},
              **{n: float(m[i]) for i, n in enumerate(['colp', 'colrec', 'colf1', 'fgcs'], 2)},
              'mrr': max((1 / ranks['T:' + t] for t in gt if 'T:' + t in ranks), default=0.),
              'candidate_recall': sum('T:' + t in ranks for t in gt) / len(gt) if gt else 0.}
    tr = {'qid': q, 'active': a, 'hg': dict(hg),
          'table_ordered_scores': [(n, scores[n]) for n in ordered],
          'fields': ft, 'B_consumed': ledger['B_consumed'],
          'C_consumed': ledger['C_consumed'],
          'gold_table_ranks': {t: ranks.get('T:' + t) for t in gt},
          'state_reads_documented': ('extractor._train_tokens_cache/_train_gold_cache '
                                     '(same state as engine.graph_table), '
                                     'engine.base.patterns (support counts)')}
    return record, metric, tr
