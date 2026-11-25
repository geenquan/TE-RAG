"""TFIDF baseline adapter for the BIRD paper-completion offline run (Offline agent).

Identity (CONTRACTS6.md): workspace `retrievers/vector_retriever.py`
VectorRetriever — log-TF x idf field-level vectors, weighted table aggregation,
cosine scoring — with ENGLISH tokenization
(RetrieverConfig(use_chinese_tokenizer=False)). fit() is called on the unit's
train rows for interface parity (VectorRetriever's index is schema-only; it
ignores train data — recorded as the identity note in OFFLINE_READY.json).

This module also hosts the record-normalization helpers shared by the other
offline baseline wrappers (hybrid_bird, cooccur_bird, dense_bird):
  run_retriever_records()  — per-question loop, budget enforcement, error records
  _strip_c_prefix/_result_to_detail/_column_keys/_check_budgets/_question_map

Public API (mirrors adapter/baselines/run_baselines.run_baseline):
    run_tfidf(cfg, db_id, train_rows, test_qids, questions, field_csv, table_csv)
        -> list[record]

Record shape: {qid, db_id, variant:'TFIDF', status:'ok'|'error',
retrieved_tables, retrieved_column_keys, retrieved_columns_detail, query_time,
error}. Errors keep the question in the denominator (record kept, outputs
empty). Budgets: <=5 tables per question, <=10 columns per table; violations
are recorded as status='error'. The runner stamps protocol/fold afterwards.
"""

import time
from typing import Any, Dict, List, Optional

import pandas as pd

from retrievers.base_retriever import RetrieverConfig

METHOD = 'TFIDF'
K_TABLES = 5              # table budget per question (CONTRACTS6)
COLUMN_BUDGET = 10        # column budget per table (enforced, not truncated)


# ---------------------------------------------------------------------------
# Shared record normalization (used by tfidf/hybrid/cooccur/dense wrappers)
# ---------------------------------------------------------------------------

def _strip_c_prefix(col_key: str) -> str:
    """'C:{table}.{field}' -> '{table}.{field}' (keep table.field)."""
    return col_key[2:] if col_key.startswith('C:') else col_key


def _result_to_detail(results) -> List[Dict[str, Any]]:
    """RetrievalResult list -> retrieved_columns_detail (C: prefix stripped)."""
    detail = []
    for r in results:
        detail.append({
            'table': r.table,
            'table_score': float(r.table_score),
            'columns': [{'key': _strip_c_prefix(c), 'score': float(s)}
                        for c, s in r.columns],
        })
    return detail


def _column_keys(results) -> List[str]:
    """Ordered, de-duplicated simple column keys ('table.field')."""
    keys: List[str] = []
    for r in results:
        for col, _ in r.columns:
            key = _strip_c_prefix(col)
            if key not in keys:
                keys.append(key)
    return keys


def _check_budgets(results) -> Optional[str]:
    """Return a violation message, or None if within budgets."""
    if len(results) > K_TABLES:
        return f"budget violation: {len(results)} tables > {K_TABLES}"
    for r in results:
        if len(r.columns) > COLUMN_BUDGET:
            return (f"budget violation: table {r.table!r} has "
                    f"{len(r.columns)} columns > {COLUMN_BUDGET}")
    return None


def _question_map(questions) -> Dict[str, str]:
    """Accept {qid: question} mapping or DataFrame with qid/question columns."""
    if questions is None:
        return {}
    if isinstance(questions, pd.DataFrame):
        qmap = {}
        for _, row in questions.iterrows():
            qmap[str(row['qid'])] = str(row['question'])
        return qmap
    if isinstance(questions, pd.Series):
        return {str(k): str(v) for k, v in questions.items()}
    return {str(k): str(v) for k, v in dict(questions).items()}


def run_retriever_records(retriever, method: str, db_id: str,
                          test_qids: List[str], questions, stop_exceptions=()
                          ) -> List[Dict[str, Any]]:
    """Run a fitted retriever over all test questions with per-question isolation.

    One record per test qid (same order). Any per-question exception becomes
    status='error' with empty outputs (fixed denominator). Budget violations
    (tables > 5 or columns/table > 10) also become error records.
    """
    qmap = _question_map(questions)
    records: List[Dict[str, Any]] = []
    for qid in test_qids:
        record = {
            'qid': qid,
            'db_id': db_id,
            'variant': method,
            'status': 'ok',
            'retrieved_tables': [],
            'retrieved_column_keys': [],
            'retrieved_columns_detail': [],
            'query_time': 0.0,
            'error': None,
        }
        t0 = time.perf_counter()
        try:
            if hasattr(retriever, 'api_qid'):
                retriever.api_qid = str(qid)
            question = qmap.get(str(qid))
            if question is None:
                raise KeyError(f"missing question text for qid {qid!r}")
            results = retriever.retrieve(question, k=K_TABLES)
            violation = _check_budgets(results)
            if violation is not None:
                raise RuntimeError(violation)
            record['retrieved_tables'] = [r.table for r in results]
            record['retrieved_column_keys'] = _column_keys(results)
            record['retrieved_columns_detail'] = _result_to_detail(results)
        except Exception as exc:  # noqa: BLE001 - per-question error isolation
            if isinstance(exc, stop_exceptions):
                raise
            record['status'] = 'error'
            record['retrieved_tables'] = []
            record['retrieved_column_keys'] = []
            record['retrieved_columns_detail'] = []
            record['error'] = f"{type(exc).__name__}: {exc}"[:500]
        finally:
            record['query_time'] = time.perf_counter() - t0
        records.append(record)
    return records


# ---------------------------------------------------------------------------
# TFIDF identity
# ---------------------------------------------------------------------------

def build_fitted_retriever(cfg, field_csv: str, table_csv: str,
                           train_rows: pd.DataFrame):
    """Construct and fit the workspace VectorRetriever (exact TFIDF identity).

    English tokenization via RetrieverConfig(use_chinese_tokenizer=False);
    fit() receives the unit's train rows (full-SQL supervision) for interface
    parity — the VectorRetriever index itself is schema-only (identity note).
    """
    from retrievers.vector_retriever import VectorRetriever
    retriever = VectorRetriever(
        field_csv, table_csv,
        RetrieverConfig(name=METHOD, use_chinese_tokenizer=False))
    retriever.fit(train_rows)
    return retriever


def run_tfidf(cfg, db_id: str, train_rows: pd.DataFrame,
              test_qids: List[str], questions, field_csv: str,
              table_csv: str) -> List[Dict[str, Any]]:
    """Run the TFIDF baseline over all test questions of one database.

    Args:
        cfg: TERAGConfig from runner.terag_config_factory.make_bird_config
             (kept for signature parity with run_baselines.run_baseline;
             TFIDF does not consume it).
        db_id: database id for the records.
        train_rows: DataFrame with qid, question, sql, db_id, table, field,
                    gold_tables (JSON list), gold_fields (JSON list of
                    [simple_table, col]) — full-SQL physical needs.
        test_qids: qids to evaluate, in order; every qid yields a record.
        questions: {qid: question_text} mapping (or DataFrame qid/question).
        field_csv, table_csv: per-db schema CSV paths.

    Returns:
        One record per test qid (same order). Errors produce status='error'
        with empty outputs and stay in the denominator.
    """
    retriever = build_fitted_retriever(cfg, field_csv, table_csv, train_rows)
    return run_retriever_records(retriever, METHOD, db_id, test_qids, questions)
