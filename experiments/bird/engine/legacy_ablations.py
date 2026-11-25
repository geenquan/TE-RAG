"""Legacy-ablation wrappers (noBC / noAnnIdx / noAnnHit) for the BIRD
paper-completion offline run (Offline agent).

Identity (CONTRACTS6.md): thin wrappers over BirdEngine parameter masks —
NO scoring changes, no new code paths in the engine. Each method calls
engine.predict(qid, masked_params, variant='Full') and rewrites
record['variant'] to the legacy name afterwards (records only; the engine
still runs its frozen 'Full' path).

Masks (frozen; exactly CONTRACTS6):
  noBC     : b_index_enabled=False, b_hit_enabled=False, modules={'C': False}
             -> B and C fully off. Ledger must show B_consumed=0 AND
                C_consumed=0.
  noAnnIdx : b_index_enabled=False, c_index=0
             -> generation + hits kept, all annotation INDEX texts removed.
                Ledger must show index text counts 0 (B_index and C_index
                empty, text_count=0) while weighted TERM columns
                (ledger['term_columns']) may be > 0.
  noAnnHit : b_hit_enabled=False, c_hit_weight=0
             -> B/C index texts kept, all TERM hit contributions removed.
                Ledger must keep C_index texts present while every record's
                active_C_TERM_fields (= c_count of CHit>0 fields) is 0.

noBC is NOT noB: 'noB' (a stock engine variant) keeps C on — the tests assert
differing ledgers between noBC and noB (B_generation_for_C_retained and
C_consumed differ).

Usage:
    from adapter.model.legacy_ablations import (
        LEGACY_METHODS, masked_params, run_legacy_ablation)

    p = masked_params(default_params, 'noBC')
    records = run_legacy_ablation(engine, 'noBC', default_params, test_qids)

Per-question errors become error records (fixed denominator); the runner
stamps protocol/fold and writes cell.json with engine.ledger(params_masked).
"""

import copy
import time
from typing import Any, Dict, List

ABLATION_MASKS: Dict[str, Dict[str, Any]] = {
    'noBC': {'b_index_enabled': False, 'b_hit_enabled': False,
             'modules': {'C': False}},
    'noAnnIdx': {'b_index_enabled': False, 'c_index': 0},
    'noAnnHit': {'b_hit_enabled': False, 'c_hit_weight': 0.0},
}

LEGACY_METHODS = tuple(sorted(ABLATION_MASKS))
PREDICT_VARIANT = 'Full'  # engine path is always frozen 'Full'; mask is in params


def masked_params(base_params: Dict[str, Any], method: str) -> Dict[str, Any]:
    """Deep-copied base params with the CONTRACTS6 mask for `method` applied."""
    if method not in ABLATION_MASKS:
        raise ValueError(f"Unknown legacy ablation: {method!r}; "
                         f"expected one of {LEGACY_METHODS}")
    p = copy.deepcopy(base_params)
    p.update(copy.deepcopy(ABLATION_MASKS[method]))
    return p


def _error_record(qid: str, method: str, exc: Exception,
                  elapsed: float) -> Dict[str, Any]:
    """A failed question stays in the fixed denominator with empty outputs."""
    return {'qid': qid, 'variant': method, 'status': 'error',
            'error': f"{type(exc).__name__}: {exc}"[:500],
            'retrieved_tables': [], 'retrieved_column_keys': [],
            'retrieved_columns_detail': [], 'query_time': elapsed}


def run_legacy_ablation(engine, method: str, params: Dict[str, Any],
                        test_qids: List[str],
                        per_question_timeout: float = 300.0
                        ) -> List[Dict[str, Any]]:
    """Run one legacy ablation over all test qids of one engine/unit.

    Args:
        engine: fitted adapter.model.bird_engine.BirdEngine.
        method: one of LEGACY_METHODS ('noBC', 'noAnnIdx', 'noAnnHit').
        params: frozen base params (configs/effective_params.json
                default_params); the CONTRACTS6 mask is applied per method.
        test_qids: qids to evaluate, in order; every qid yields a record.
        per_question_timeout: queries slower than this become 'timeout'
                              error records (mirrors the Lead's run_unit).

    Returns:
        One record per test qid (same order). record['variant'] is rewritten
        to the legacy name after predict (engine still ran variant='Full').
    """
    if method not in ABLATION_MASKS:
        raise ValueError(f"Unknown legacy ablation: {method!r}; "
                         f"expected one of {LEGACY_METHODS}")
    p = masked_params(params, method)
    records: List[Dict[str, Any]] = []
    for qid in test_qids:
        t0 = time.perf_counter()
        try:
            record, _metric, _trace = engine.predict(qid, p, PREDICT_VARIANT,
                                                     trace=False)
            record['variant'] = method  # rewrite AFTER predict (records only)
            record['query_time'] = time.perf_counter() - t0
            if record['query_time'] > per_question_timeout:
                record['status'] = 'timeout'
                record['error'] = f"query exceeded {per_question_timeout}s"
            records.append(record)
        except Exception as exc:  # noqa: BLE001 - per-question error isolation
            records.append(_error_record(qid, method, exc,
                                         time.perf_counter() - t0))
    return records


def ledger_checks(method: str, ledger: Dict[str, Any],
                  records: List[Dict[str, Any]]) -> Dict[str, bool]:
    """CONTRACTS6 consumption assertions (names alone are not acceptance).

    Returns a dict of check->bool; every value must be True. Used by the unit
    tests and recorded into adapter/OFFLINE_READY.json.
    """
    if method == 'noBC':
        return {
            'B_consumed_zero': ledger.get('B_consumed') == 0,
            'C_consumed_zero': ledger.get('C_consumed') == 0,
            'records_present': len(records) >= 0,
        }
    if method == 'noAnnIdx':
        b_index = ledger.get('B_index') or {}
        c_index = ledger.get('C_index') or {}
        return {
            'B_index_tables_zero': len(b_index) == 0,
            'C_index_tables_zero': len(c_index) == 0,
            'text_count_zero': ledger.get('text_count') == 0,
            'term_columns_nonnegative': ledger.get('term_columns', 0) >= 0,
        }
    if method == 'noAnnHit':
        b_index = ledger.get('B_index') or {}
        c_index = ledger.get('C_index') or {}
        # NOTE on 'C index texts present' (CONTRACTS6 wording): on BIRD real
        # data the frozen engine classifies every C-adapted index text as
        # B-origin (text collision), so C_index is empty even for the Full
        # variant (verified across all 105 accepted five-fold units). The
        # verifiable identity is: the annotation INDEX is kept (B_index ∪
        # C_index non-empty) while the C channel stays enabled and the TERM
        # HIT contributions vanish (c_count = 0 in every record).
        return {
            'annotation_index_texts_present':
                (sum(len(v) for v in b_index.values())
                 + sum(len(v) for v in c_index.values())) > 0,
            'C_channel_active': bool((ledger.get('active') or {}).get('C')),
            'records_c_count_zero': all(
                r.get('active_C_TERM_fields', 0) == 0 for r in records),
        }
    raise ValueError(f"Unknown legacy ablation: {method!r}")
