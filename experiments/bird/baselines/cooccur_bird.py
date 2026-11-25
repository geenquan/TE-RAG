"""CooccurRRF baseline adapter for the BIRD paper-completion offline run (Offline agent).

Identity (CONTRACTS6.md): workspace `retrievers/cooccurrence_retriever.py`
CooccurrenceRetriever with fusion_mode='rank_equal' (the preregistered
rank_equal_v2 RRF: only positive-evidence tables earn an evidence rank; the
BM25 fallback list uses fixed depth BM25_RANK_DEPTH=20). Supervision = the
unit's current train rows, full-SQL gold tables/fields parsed via
datasets.gt_utils (all roles — never SELECT-only projections).

BOUNDARY FIX (identical in kind to the ExampleRRF fix in run_baselines.py;
workspace files are read-only): CooccurrenceRetriever.__init__ constructs its
internal fallback BM25Retriever with a default config that enables the Chinese
jieba tokenizer. We flip `retriever._fallback.config.use_chinese_tokenizer =
False` BEFORE fit() so the fallback index is built and queried with the same
English tokenizer as the cooccurrence channel itself.

Public API (mirrors adapter.baselines.run_baselines.run_baseline):
    run_cooccur_rrf(cfg, db_id, train_rows, test_qids, questions,
                    field_csv, table_csv) -> list[record]

Record shape/budgets/error semantics: see adapter/baselines/tfidf_bird.py
(shared run_retriever_records helper). The runner stamps protocol/fold.
"""

from typing import Any, Dict, List

import pandas as pd

from retrievers.base_retriever import RetrieverConfig

from experiments.bird.baselines.tfidf_bird import run_retriever_records

METHOD = 'CooccurRRF'


def build_fitted_retriever(cfg, field_csv: str, table_csv: str,
                           train_rows: pd.DataFrame):
    """Construct and fit the workspace CooccurrenceRetriever (exact identity).

    fusion_mode='rank_equal' + English tokenization on the cooccurrence
    channel; the internal BM25 fallback gets the English boundary fix before
    fit() (see module docstring).
    """
    from retrievers.cooccurrence_retriever import CooccurrenceRetriever
    retriever = CooccurrenceRetriever(
        field_csv, table_csv,
        RetrieverConfig(name=METHOD, use_chinese_tokenizer=False,
                        extra_params={'fusion_mode': 'rank_equal'}))
    # BOUNDARY FIX (do not edit workspace files): the internal fallback
    # BM25Retriever was constructed with the default config (Chinese jieba
    # tokenizer). Flip it to English BEFORE fit() so the fallback index and
    # its query tokenization use the same English tokenizer.
    retriever._fallback.config.use_chinese_tokenizer = False
    retriever.fit(train_rows)  # evidence = full-SQL gold via gt_utils
    return retriever


def run_cooccur_rrf(cfg, db_id: str, train_rows: pd.DataFrame,
                    test_qids: List[str], questions, field_csv: str,
                    table_csv: str) -> List[Dict[str, Any]]:
    """Run the CooccurRRF baseline over all test questions of one database.

    Args:
        cfg: TERAGConfig (signature parity; CooccurRRF does not consume it).
        db_id: database id for the records.
        train_rows: DataFrame with qid, question, sql, db_id, table, field,
                    gold_tables (JSON list), gold_fields (JSON list of
                    [simple_table, col]) — full-SQL supervision source.
        test_qids: qids to evaluate, in order; every qid yields a record.
        questions: {qid: question_text} mapping (or DataFrame qid/question).
        field_csv, table_csv: per-db schema CSV paths.

    Returns:
        One record per test qid (same order); per-question errors keep the
        fixed denominator (status='error', empty outputs).
    """
    retriever = build_fitted_retriever(cfg, field_csv, table_csv, train_rows)
    return run_retriever_records(retriever, METHOD, db_id, test_qids, questions)
