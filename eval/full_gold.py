"""Full SQL gold evaluation of saved predictions; no retrievers or configuration IO.

AFCS callers keep their annotated-field gold. This module is only for the
explicitly supplied full-gold evaluation population.
"""
from collections import defaultdict

from eval.retrieval_metrics import (all_gold_covered, column_prf,
                                    schema_coverage_success, table_recall_at_k)


def full_gold_metrics(record, gold):
    """Legacy tuple shape: TR, Covered, P, R, F1, normalized FGCS.

    Runtime errors are failures even if a malformed record contains stale output.
    No population filtering is implicit here: callers supply verified qids.
    """
    g = gold[record['qid']]
    failed = record.get('status') == 'error'
    tables = [] if failed else (record.get('retrieved_tables') or [])
    columns = [] if failed else (record.get('retrieved_column_keys') or [])
    tr = table_recall_at_k(g['gold_tables'], tables)
    cov = all_gold_covered(g['gold_tables'], tables, 5)
    p, r, f1 = column_prf(g['gold_columns'], columns)
    return tr, cov, p, r, f1, bool(schema_coverage_success(
        g['gold_tables'], g['gold_columns'], tables, columns, 5))


def index_predictions(records, expected_qids=None):
    """Reject duplicates before indexing, and missing/unexpected qids when given.

    Input must represent one method/protocol. Errors and empty selections are
    records, not missing predictions, and are deliberately retained.
    """
    result = {}
    for r in records:
        qid = r['qid']
        if qid in result:
            raise ValueError(f'duplicate prediction qid: {qid}')
        result[qid] = r
    if expected_qids is not None:
        expected = set(expected_qids)
        if set(result) != expected:
            raise ValueError(f'prediction population mismatch: missing={sorted(expected-set(result))}; '
                             f'unexpected={sorted(set(result)-expected)}')
    return result


def prediction_groups(records):
    """Validate uniqueness per (setting, variant, qid), within one source file."""
    groups = defaultdict(list)
    for r in records:
        groups[(r['setting'], r['variant'])].append(r)
    return {key: index_predictions(rows) for key, rows in groups.items()}


def validate_population(gold, valid, feasible):
    valid, feasible = set(valid), set(feasible)
    if not feasible <= valid or not valid <= set(gold) or not valid:
        raise ValueError('expected nonempty valid population with feasible <= valid <= gold')
    for q in valid:
        if not gold[q]['gold_tables'] or not gold[q]['gold_columns']:
            raise ValueError(f'valid full gold lacks tables or columns: {q}')


def summarize_predictions(records, gold, valid, feasible):
    validate_population(gold, valid, feasible)
    pool = index_predictions(records, expected_qids=gold)
    ordered = sorted(valid)
    metrics = {q: full_gold_metrics(pool[q], gold) for q in ordered}
    def mean(values):
        values = list(values)
        return sum(values) / len(values) if values else None
    row = {'n_runs': len(pool), 'n': len(ordered), 'n_feasible': len(feasible)}
    row.update({f'tr@{k}': mean(metrics[q][0][k] for q in ordered) for k in (1, 3, 5)})
    for i, name in enumerate(('covered@5', 'colp', 'colrec', 'colf1', 'fgcs'), 1):
        row[name] = mean(float(metrics[q][i]) for q in ordered)
    row['fgcs_successes'] = sum(metrics[q][5] for q in ordered)
    row['fgcs_failures'] = row['n'] - row['fgcs_successes']
    row['fgcs_feasible65'] = mean(float(metrics[q][5]) for q in sorted(feasible))
    row['fgcs_feasible_successes'] = sum(metrics[q][5] for q in feasible)
    row['n_error'] = sum(pool[q].get('status') == 'error' for q in ordered)
    row['n_empty_prediction'] = sum(not pool[q].get('retrieved_tables') for q in ordered)
    return row


def paired_covered(records_a, records_b, gold, valid, clusters_by_table):
    """Historical Full - opponent Wilcoxon/cluster bootstrap; strict paired qids."""
    import numpy as np
    import scipy
    from scipy import stats
    # Validate before filtering so duplicate invalid-gold qids also cannot hide.
    pa, pb = index_predictions(records_a), index_predictions(records_b)
    if not set(valid) <= set(pa) or not set(valid) <= set(pb):
        raise ValueError('paired predictions missing valid qids')
    if not set(valid) <= set(clusters_by_table):
        raise ValueError('missing target-table cluster assignment')
    vals = [(q, float(full_gold_metrics(pa[q], gold)[1]),
             float(full_gold_metrics(pb[q], gold)[1])) for q in sorted(valid)]
    d = [a-b for _, a, b in vals]
    res = {'n': len(d), 'mean_diff': sum(d)/len(d) if d else None,
           'metric': 'covered@5', 'direction': 'Full - opponent',
           'zero_difference': bool(d) and not any(d),
           'statistical_method': {
               'test': 'scipy.stats.wilcoxon', 'alternative': 'two-sided',
               'zero_method': 'wilcox', 'correction': False, 'method': 'auto',
               'scipy_version': scipy.__version__, 'numpy_version': np.__version__,
               'cluster_key': 'full_gold.original_table',
               'n_clusters': len({clusters_by_table[q] for q in valid}),
               'seed': 42, 'resamples': 10000, 'ci': 'percentile 2.5/97.5; unadjusted'}}
    if not d or res['zero_difference']:
        return res
    res['wilcoxon_p_raw'] = float(stats.wilcoxon(
        [a for _, a, _ in vals], [b for _, _, b in vals],
        alternative='two-sided', zero_method='wilcox', correction=False,
        method='auto').pvalue)
    by_t = defaultdict(list)
    for q, a, b in vals:
        by_t[clusters_by_table[q]].append(a-b)
    clusters = sorted(by_t)
    rng = np.random.default_rng(42)
    boots = []
    for _ in range(10000):
        sample = rng.choice(len(clusters), size=len(clusters), replace=True)
        values = [v for i in sample for v in by_t[clusters[i]]]
        boots.append(sum(values)/len(values))
    res['ci95_cluster_bootstrap'] = [float(v) for v in np.percentile(boots, [2.5, 97.5])]
    return res
