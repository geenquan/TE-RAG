"""Verbatim frozen study parameters from workspace/reference_research/study_common.py.

Copied line-for-line: VARIANTS, DEFAULT, switches, numeric_identity, and the
digest/require/sha helpers. The run-specific path constants and the
powergrid-only helpers (anchor/source_hashes) of study_common are intentionally
NOT copied (BIRD construction replaces them; no scoring semantics involved).
"""
import csv
import hashlib
import json
from pathlib import Path


def load(p):
    return json.loads(Path(p).read_text())  # verbatim study_common.load


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()


def digest(v):
    return hashlib.sha256(json.dumps(v, sort_keys=True, ensure_ascii=False,
                                     default=str).encode()).hexdigest()


def require(c, m):
    if not c:
        raise RuntimeError(m)


def write(p, v):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x') as f:
        json.dump(v, f, ensure_ascii=False, indent=2, allow_nan=False)


VARIANTS = ['Full', 'noBIndex', 'noBHit', 'noB', 'noTableGraph', 'noFieldGraph', 'noGraph', 'noPattern', 'noC']
DEFAULT = {'table_weights': [.35, .35, .30], 'field_weights': [.40, .20, .15, .25],
           'b_index': .5, 'c_index': .5, 'c_confidence': 'constant',
           'k1': 1.5, 'b': .75, 'theta_g': .15, 'table_normalization': 'raw', 'field_normalization': 'raw',
           'direct_scale': 1., 'description_scale': 1., 'field_graph_cap': 5., 'min_total_sim': .5, 'source_topk': 3, 'modules': None,
           'b_index_enabled': True, 'b_hit_enabled': True, 'table_graph_enabled': True, 'field_graph_enabled': True,
           'b_hit_weight': 1., 'c_hit_weight': 1., 'b_text_norm': 'none', 'b_duplicate_penalty': 'none', 'b_source_weight': 'none',
           'b_match': 'exact', 'c_match': 'exact', 'c_admission_subset': .5, 'c_evidence_mix': [.3, .4, .3],
           'table_graph_alpha': 0., 'field_graph_alpha': 0., 'table_graph_norm': 'none', 'field_graph_norm': 'none',
           'table_graph_clip': 1000000., 'table_graph_gamma': 3., 'field_graph_gamma': 1.,
           'field_graph_threshold': .15, 'field_graph_neighbors': 0, 'field_edge_power': 1.,
           'table_pattern_norm': 'none', 'field_pattern_norm': 'none', 'pattern_aggregation': 'max', 'pattern_smoothing': 0.}


def switches(p, variant):
    require(variant in VARIANTS + ['noBGraph', 'noBPattern'], 'Unknown variant')
    a = {'BIndex': p['b_index_enabled'] and variant not in ['noBIndex', 'noB', 'noBGraph', 'noBPattern'],
         'BHit': p['b_hit_enabled'] and variant not in ['noBHit', 'noB', 'noBGraph', 'noBPattern'],
         'TableGraph': p['table_graph_enabled'] and variant not in ['noTableGraph', 'noGraph', 'noBGraph'],
         'FieldGraph': p['field_graph_enabled'] and variant not in ['noFieldGraph', 'noGraph', 'noBGraph'],
         'C': variant != 'noC', 'Pattern': variant not in ['noPattern', 'noBPattern'], 'AnnIdx': True, 'AnnHit': True}
    for k, v in (p.get('modules') or {}).items():
        require(k in a and isinstance(v, bool), 'Invalid module mask')
        a[k] = a[k] and v
    a.update(B=a['BIndex'] or a['BHit'], Graph=a['TableGraph'] or a['FieldGraph'])
    return a


def numeric_identity(value):
    """Deduplicate numeric parameter aliases without changing scored values."""
    def rounded(x):
        if isinstance(x, float):
            return round(x, 12)
        if isinstance(x, dict):
            return {k: rounded(v) for k, v in x.items()}
        if isinstance(x, list):
            return [rounded(v) for v in x]
        return x
    return digest(rounded(value))
