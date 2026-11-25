"""Hybrid baseline adapter for the BIRD paper-completion offline run (Offline agent).

Identity (CONTRACTS6.md): workspace `retrievers/hybrid_retriever.py`
HybridRetriever — BM25 + TFIDF linear score fusion with the default
alpha=0.5 (equal weight; min-max normalization per retriever before fusion) —
with ENGLISH tokenization. The workspace constructor propagates
use_chinese_tokenizer to both sub-retrievers (BM25 + Vector), so a single
English parent config wires the whole method; fit() is called on the unit's
train rows for interface parity (both sub-indexes are schema-only).

Public API (mirrors adapter/baselines/run_baselines.run_baseline):
    run_hybrid(cfg, db_id, train_rows, test_qids, questions, field_csv, table_csv)
        -> list[record]

Record shape/budgets/error semantics: see adapter/baselines/tfidf_bird.py
(shared run_retriever_records helper). The runner stamps protocol/fold.
"""

from typing import Any, Dict, List

import pandas as pd

from retrievers.base_retriever import RetrieverConfig

from experiments.bird.baselines.tfidf_bird import run_retriever_records

METHOD = 'Hybrid'
ALPHA = 0.5  # CONTRACTS6: default alpha (equal BM25/TFIDF weight); NO tuning


def build_fitted_retriever(cfg, field_csv: str, table_csv: str,
                           train_rows: pd.DataFrame, alpha: float = ALPHA):
    """Construct and fit the workspace HybridRetriever (exact identity).

    English tokenization via the parent config; the workspace constructor
    forwards use_chinese_tokenizer to the internal BM25Retriever and
    VectorRetriever sub-retrievers (no boundary fix needed for this method).
    """
    from retrievers.hybrid_retriever import HybridRetriever
    retriever = HybridRetriever(
        field_csv, table_csv,
        config=RetrieverConfig(name=METHOD, use_chinese_tokenizer=False),
        alpha=alpha)
    retriever.fit(train_rows)
    return retriever


def run_hybrid(cfg, db_id: str, train_rows: pd.DataFrame,
               test_qids: List[str], questions, field_csv: str,
               table_csv: str) -> List[Dict[str, Any]]:
    """Run the Hybrid baseline over all test questions of one database.

    Args:
        cfg: TERAGConfig (signature parity; Hybrid does not consume it).
        db_id: database id for the records.
        train_rows: DataFrame with qid, question, sql, db_id, table, field,
                    gold_tables (JSON list), gold_fields (JSON list).
        test_qids: qids to evaluate, in order; every qid yields a record.
        questions: {qid: question_text} mapping (or DataFrame qid/question).
        field_csv, table_csv: per-db schema CSV paths.

    Returns:
        One record per test qid (same order); per-question errors keep the
        fixed denominator (status='error', empty outputs).
    """
    retriever = build_fitted_retriever(cfg, field_csv, table_csv, train_rows)
    return run_retriever_records(retriever, METHOD, db_id, test_qids, questions)
