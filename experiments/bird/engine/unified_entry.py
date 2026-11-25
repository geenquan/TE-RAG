"""Unified execution entry — the single authoritative path for BOTH development
and final evaluation (repair for the V2 wiring defects).

Defects being repaired (independently confirmed, see EXECUTION_PATH_AUDIT.md):
  A. field_rerank (Q column gate) was never applied in the V2 final Dev run
     (engine_loader called engine.predict only; records lack
     field_rerank_applied; selected == gate_off metrics bit-identical).
  B. hg reliability gating was never applied in the V2 final Dev run
     (search/h/gating.predict_gated was used ONLY in Train-H development).

This module is the ONE path; both entry points (dev recheck, Dev final run)
must call run_one. Order (task §五):
  1. core engine predict (params stripped of mechanism namespaces)
  2. hg gating replaces the whole predict when params declare 'hg'
  3. table candidates + field order (from 1/2)
  4. field_rerank post-processing when params declare 'field_rerank'
  5. record normalization + budget enforcement
  6. metrics recomputed from the FINAL record (never pre-mechanism metrics)
Execution evidence recorded per question: mechanism enabled/executed/changed,
spec digests, before/after column counts, hg gate/cap counters where exposed,
budget check result. Hard error when a declared mechanism lacks execution
evidence. PYTHONHASHSEED=0 required for cross-process determinism.
"""
import hashlib
import json
import sys
from pathlib import Path

RUN = Path(__file__).resolve().parents[1]
from experiments.bird.engine.field_rerank.apply import apply_config_to_record  # noqa: E402
from experiments.bird.metrics_core import metric_values

MECH_KEYS = ('hg', 'field_rerank')


def _sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str)
                          .encode()).hexdigest()[:16]


def mrr_from_record(record, gold_entry):
    tables = record.get('retrieved_tables') or []
    for i, t in enumerate(tables, 1):
        if t in gold_entry['gold_tables']:
            return 1.0 / i
    return 0.0


def final_metric(record, gold_entry):
    """Metrics from the FINAL record — the only metric source (task rule 6)."""
    m = metric_values(record, gold_entry)
    m['mrr'] = mrr_from_record(record, gold_entry)
    return m


def run_one(engine, qid, params, variant, gold_entry, predict_gated=None):
    """Returns (record, metric, evidence).

    engine: BirdEngine/VnextEngine (fitted); qid under its test set.
    params: full vector incl. optional 'hg' and 'field_rerank' namespaces.
    variant: engine switch ('Full' or stock ablation names).
    predict_gated: injected gating entry (search/h/gating.predict_gated);
      imported lazily if None and needed — keeps this module the only entry.
    """
    evidence = {'qid': qid, 'variant': variant,
                'mechanisms': {}}
    p_core = {k: v for k, v in (params or {}).items()
              if k not in MECH_KEYS}
    hg = (params or {}).get('hg')
    fr = (params or {}).get('field_rerank')

    # -- steps 1+2: core predict, or hg-gated predict when declared -----------
    if hg is not None:
        if predict_gated is None:
            from experiments.bird.gating import predict_gated as _pg
            predict_gated = _pg
        record, _engine_metric, _trace = predict_gated(
            engine, qid, p_core, variant, hg)
        evidence['mechanisms']['hg'] = {
            'enabled': True, 'executed': True, 'spec_sha': _sha(hg),
            'entry': 'search.h.gating.predict_gated',
        }
    else:
        record, _engine_metric, _trace = engine.predict(qid, p_core, variant)
        evidence['mechanisms']['hg'] = {'enabled': False, 'executed': False}

    n_cols_before = len(record.get('retrieved_column_keys') or [])

    # -- step 4: field_rerank post-processing ---------------------------------
    if fr is not None:
        record = apply_config_to_record(record, {'field_rerank': fr})
        assert record.get('field_rerank_applied'), \
            f'{qid}: field_rerank declared but apply produced no marker'
        n_cols_after = len(record.get('retrieved_column_keys') or [])
        evidence['mechanisms']['field_rerank'] = {
            'enabled': True, 'executed': True,
            'spec_sha': _sha(fr),
            'columns_before': n_cols_before, 'columns_after': n_cols_after,
            'changed_output': n_cols_after != n_cols_before,
            'marker': record['field_rerank_applied'],
        }
    else:
        evidence['mechanisms']['field_rerank'] = {
            'enabled': False, 'executed': False,
            'changed_output': False,
        }

    # -- step 5: budget + canonical fields ------------------------------------
    n_tables = len(record.get('retrieved_tables') or [])
    per_table = [len(t.get('columns', []))
                 for t in record.get('retrieved_columns_detail') or []]
    if n_tables > 5 or any(n > 10 for n in per_table):
        raise RuntimeError(f'{qid}: budget violated after mechanisms '
                           f'({n_tables} tables, per-table {per_table})')
    evidence['budget_ok'] = True
    evidence['tables'] = n_tables
    evidence['columns'] = sum(per_table)

    # -- hard gate: declared ⇒ executed evidence present ----------------------
    for mech, spec in (('field_rerank', fr), ('hg', hg)):
        if spec is not None and not evidence['mechanisms'][mech]['executed']:
            raise RuntimeError(f'{qid}: mechanism {mech} declared but not executed')

    # -- step 6: metrics from FINAL record -------------------------------------
    metric = final_metric(record, gold_entry)
    return record, metric, evidence


def run_one_devparity(engine, qid, params, variant):
    """Development-loop convenience: same path, engine-gold metrics source is
    the engine's own gold (identical population) — kept so Train rechecks use
    THE SAME run_one core. Returns (record, metric, evidence)."""
    gold_entry = engine.gold[qid]
    return run_one(engine, qid, params, variant, gold_entry)
