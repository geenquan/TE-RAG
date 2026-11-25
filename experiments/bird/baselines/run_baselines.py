"""Baseline adapter for the BIRD schema-grounding quick run (Baseline agent scope).

Implements the CONTRACTS.md baseline interface by wiring the frozen workspace
retrievers with exact method identities:

- BM25            : retrievers.bm25_retriever.BM25Retriever over table name+desc
                    documents only, English tokenizer, fit ignores train data.
- MetadataMatched : retrievers.matched_schema_retriever.MatchedSchemaRetriever.
                    Builds the SAME enhanced metadata index documents as TE-RAG
                    Full's metadata index (terag.index_builder.IndexBuilder:
                    table name/desc + field name/desc, weighted, English) with
                    use_history=False and use_transfer=False. That index (and no
                    history/transfer channel) IS this baseline's definition.
- ExampleRRF      : retrievers.example_rag_retriever.ExampleRAGRetriever with
                    fusion_mode='rank_equal' (the frozen implementation applies
                    the P2-C corrected RRF, recorded as 'rank_equal_v2' in the
                    result evidence metadata: only positive-evidence tables get
                    an evidence rank; the BM25 fallback list uses fixed depth
                    BM25_RANK_DEPTH=20). Example evidence comes from the current
                    train fold's full-SQL gold tables/fields (all roles, parsed
                    via datasets.gt_utils.parse_ground_truth_fields), never from
                    SELECT-only projections.

Workspace sources are read-only for this agent. The single boundary fix (NOT an
edit to workspace files): ExampleRAGRetriever.__init__ constructs its internal
fallback BM25Retriever with a default config that enables the Chinese jieba
tokenizer; we flip `retriever._fallback.config.use_chinese_tokenizer = False`
BEFORE fit() so the fallback index is built with the English tokenizer. See
BASELINES_READY.json identity notes.

Public API (CONTRACTS.md):
    run_baseline(method, cfg, db_id, train_rows, test_qids, questions,
                 field_csv, table_csv) -> list[record]

Record shape: {qid, db_id, variant, status: 'ok'|'error', retrieved_tables,
retrieved_column_keys, retrieved_columns_detail, query_time, error}.
Errors keep the question in the denominator (record kept, outputs empty).
Budgets: <=5 tables per question, <=10 columns per table; violations are
recorded as status='error'.
"""

import time
from typing import Any, Dict, List, Optional

import pandas as pd

from retrievers.base_retriever import RetrieverConfig

METHODS = ('BM25', 'MetadataMatched', 'ExampleRRF')

K_TABLES = 5              # table budget per question
COLUMN_BUDGET = 10        # column budget per table (enforced, not truncated)

# The Lead passes cfg = make_bird_config(db_id, artifacts_dir=.../cache/baselines/<db_id>).
# We import the factory only to document provenance; cfg arrives as an argument.
# from runner.terag_config_factory import make_bird_config  # noqa: ERA001 (docs)


# ---------------------------------------------------------------------------
# Retriever construction (exact identities; deviations not allowed)
# ---------------------------------------------------------------------------

def build_fitted_retriever(method: str, cfg, field_csv: str, table_csv: str,
                           train_rows: pd.DataFrame):
    """Construct and fit the workspace retriever for `method` (exact identity).

    Exposed for tests; run_baseline() is the contract entry point.
    """
    if method not in METHODS:
        raise ValueError(f"Unknown baseline method: {method!r}; expected one of {METHODS}")

    if method == 'BM25':
        from retrievers.bm25_retriever import BM25Retriever
        retriever = BM25Retriever(
            field_csv, table_csv,
            RetrieverConfig(name='BM25', use_chinese_tokenizer=False))
        retriever.fit(train_rows)  # fit() ignores train data (identity note)
        return retriever

    if method == 'MetadataMatched':
        from retrievers.matched_schema_retriever import MatchedSchemaRetriever
        retriever = MatchedSchemaRetriever(
            field_csv, table_csv,
            RetrieverConfig(
                name='MetadataMatched',
                use_chinese_tokenizer=False,
                extra_params={'terag_config': cfg,
                              'use_history': False,
                              'use_transfer': False}))
        retriever.fit(train_rows)  # use_history=False: builds pure metadata index
        return retriever

    # method == 'ExampleRRF'
    from retrievers.example_rag_retriever import ExampleRAGRetriever
    retriever = ExampleRAGRetriever(
        field_csv, table_csv,
        RetrieverConfig(
            name='ExampleRRF',
            use_chinese_tokenizer=False,
            extra_params={'fusion_mode': 'rank_equal'}))
    # BOUNDARY FIX (do not edit workspace files): the internal fallback
    # BM25Retriever was constructed with the default config (Chinese jieba
    # tokenizer). Flip it to English BEFORE fit() so the fallback index and its
    # query tokenization use the same English tokenizer as everything else.
    retriever._fallback.config.use_chinese_tokenizer = False
    retriever.fit(train_rows)  # example evidence = full-SQL gold via gt_utils
    return retriever


# ---------------------------------------------------------------------------
# Record normalization
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


# ---------------------------------------------------------------------------
# Contract entry point
# ---------------------------------------------------------------------------

def run_baseline(method: str, cfg, db_id: str, train_rows: pd.DataFrame,
                 test_qids: List[str], questions, field_csv: str,
                 table_csv: str) -> List[Dict[str, Any]]:
    """Run one baseline over all test questions of one database.

    Args:
        method: 'BM25' | 'MetadataMatched' | 'ExampleRRF'.
        cfg: TERAGConfig from runner.terag_config_factory.make_bird_config
             (only consumed by MetadataMatched's IndexBuilder wiring).
        db_id: database id for the records.
        train_rows: DataFrame with qid, question, sql, db_id, table, field,
                    gold_tables (JSON list), gold_fields (JSON list of
                    [simple_table, col]) — full SQL physical needs.
        test_qids: qids to evaluate, in order; every qid yields a record.
        questions: {qid: question_text} mapping (or DataFrame qid/question).
        field_csv, table_csv: per-db schema CSV paths.

    Returns:
        One record per test qid (same order). Errors produce status='error'
        with empty outputs and stay in the denominator.
    """
    if method not in METHODS:
        raise ValueError(f"Unknown baseline method: {method!r}; expected one of {METHODS}")

    retriever = build_fitted_retriever(method, cfg, field_csv, table_csv, train_rows)
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
            record['status'] = 'error'
            record['retrieved_tables'] = []
            record['retrieved_column_keys'] = []
            record['retrieved_columns_detail'] = []
            record['error'] = f"{type(exc).__name__}: {exc}"[:500]
        finally:
            record['query_time'] = time.perf_counter() - t0
        records.append(record)
    return records
