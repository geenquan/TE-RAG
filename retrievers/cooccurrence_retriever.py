"""
Term–Schema Cooccurrence（词元–Schema 共现检索）

P1-B 基线 B。准确身份：从训练问题统计 token 与 gold 表/字段的共现，按条件
概率打分。作为“简单注解增强”对照——不含 Phrase Alignment、模板挖掘、
模板迁移或图传播。

监督来源：仅当前训练折的 (question, table, field)。不读取测试数据或全局
artifacts。token 级证据（token→schema 权重及来源训练 qid）保存于检索结果
metadata['evidence']，逐题落盘可查。

打分公式（固定，实现即本文件）：
- 对每道训练问题取去重 token 集（同一问题内重复 token 不重复计数）；
  c(t,T) = 含 token t 且 gold 表为 T 的训练问题数；c(t) = 含 t 的问题数；
  c(t,C) = 含 t 且在表 T 下引用字段 C 的问题数（C 记作 "T.C"）。
- 表证据分  E(T|q) = Σ_{t ∈ tokens(q)} P(T|t)，P(T|t) = c(t,T)/c(t)
- 字段证据分 E(C|t) 同理：Σ_t c(t,C)/c(t)
- 排序键 = (E(T|q) 降序, BM25 分降序, 表名升序)：共现证据优先，零证据表由
  原始 schema BM25 兜底（复用 retrievers/bm25_retriever.py），并列按表名
  稳定排序（不使用 gold）。零证据（查询无已知 token）时退化为纯 BM25，
  未见表不会被结构性排除。
- 每表列预算 10：共现字段优先（E 降序），其余按查询词-字段重合度补齐
  （同 bm25_retriever 字段打分），去重，不足则少给。
"""

import math
from collections import defaultdict
from typing import Dict, List, Tuple

import pandas as pd

from retrievers.base_retriever import BaseRetriever, RetrievalResult, RetrieverConfig
from retrievers.bm25_retriever import BM25Retriever
from datasets.gt_utils import parse_ground_truth_tables, parse_ground_truth_fields

COLUMN_BUDGET = 10        # 固定：每表列预算（协议统一）
BM25_RANK_DEPTH = 20      # P2-C 预注册：RRF 的 BM25 榜单深度（与最终 k 分开）

from retrievers.example_rag_retriever import _rrf_rank  # 同一预注册修正规则


def _rrf_rank_co(candidates, evidence_scores, bm25_order):
    """与 example_rag._rrf_rank 同一规则（仅正证据入榜 + 独立 BM25 深度）。"""
    return _rrf_rank(candidates, evidence_scores, bm25_order)


class CooccurrenceRetriever(BaseRetriever):
    """Term–Schema Cooccurrence（见模块 docstring）"""

    def __init__(self, field_csv: str, table_csv: str,
                 config: RetrieverConfig = None):
        if config is None:
            config = RetrieverConfig(
                name='Cooccurrence',
                description='Term–Schema Cooccurrence（共现条件概率，简单注解对照）')
        super().__init__(field_csv, table_csv, config)

        self._fallback = BM25Retriever(field_csv, table_csv)

        # fit 后填充的统计量
        self.token_total: Dict[str, int] = {}                 # c(t)
        self.token_table: Dict[str, Dict[str, int]] = {}      # t -> {T: c(t,T)}
        self.token_field: Dict[str, Dict[str, int]] = {}      # t -> {"T.C": c(t,C)}
        self.token_qids: Dict[str, List[str]] = {}            # t -> 来源训练 qid
        self.train_qids: List[str] = []

        # 融合模式（P2-B）：evidence_first（P1-B 原版）| rank_equal（预注册 RRF）
        self.fusion_mode = ((config.extra_params or {}).get('fusion_mode')
                            if config else None) or 'evidence_first'

    def _rank_candidates(self, candidates, evidence_scores, bm25_order, bm25_scores):
        """与 ExampleRAG._rank_candidates 同一预注册规则（rank_equal_v2）。"""
        if self.fusion_mode == 'rank_equal':
            return _rrf_rank_co(candidates, evidence_scores, bm25_order)
        ordered = sorted(candidates,
                         key=lambda t: (-evidence_scores.get(t, 0.0),
                                        -bm25_scores.get(t, 0.0), t))
        scores = {t: (evidence_scores.get(t, 0.0) or bm25_scores.get(t, 0.0))
                  for t in candidates}
        return ordered, scores

    # ---------------- fit ----------------

    def fit(self, train_data: pd.DataFrame = None):
        if train_data is None or len(train_data) == 0:
            self._fallback.fit(train_data)
            self._is_fitted = True
            return

        self.train_qids = (
            [str(q) for q in train_data['qid'].tolist()]
            if 'qid' in train_data.columns
            else [f"ex{i:03d}" for i in range(len(train_data))])

        for i, (_, row) in enumerate(train_data.iterrows()):
            tokens = {t for t in self.tokenize(str(row['question'])) if len(t) > 1}
            tables = {t.split('.')[-1] for t in parse_ground_truth_tables(row)}
            fields = {f"{t.split('.')[-1]}.{f}"
                      for t, f in parse_ground_truth_fields(row)}
            qid = self.train_qids[i]
            for t in tokens:  # 问题内去重，每问题每 token 只计 1 次
                self.token_total[t] = self.token_total.get(t, 0) + 1
                tt = self.token_table.setdefault(t, {})
                for T in tables:
                    tt[T] = tt.get(T, 0) + 1
                tf = self.token_field.setdefault(t, {})
                for C in fields:
                    tf[C] = tf.get(C, 0) + 1
                self.token_qids.setdefault(t, []).append(qid)

        self._fallback.fit(train_data)
        self._is_fitted = True

    # ---------------- retrieve ----------------

    def _retrieve(self, query: str, k: int = 5) -> List[RetrievalResult]:
        depth = BM25_RANK_DEPTH if self.fusion_mode == 'rank_equal' else k
        fallback_results = self._fallback.retrieve(query, k=depth)
        bm25_score = {r.table: float(r.table_score) for r in fallback_results}
        bm25_order = [r.table for r in fallback_results]

        tokens = [t for t in self.tokenize(query) if len(t) > 1]
        table_evidence: Dict[str, float] = defaultdict(float)
        field_evidence: Dict[str, float] = defaultdict(float)
        token_ev = {}
        for t in tokens:
            c = self.token_total.get(t)
            if not c:
                continue
            p_tables = {T: cnt / c for T, cnt in self.token_table.get(t, {}).items()}
            p_fields = {C: cnt / c for C, cnt in self.token_field.get(t, {}).items()}
            for T, p in p_tables.items():
                table_evidence[T] += p
            for C, p in p_fields.items():
                field_evidence[C] += p
            token_ev[t] = {
                'count': c,
                'top_tables': dict(sorted(p_tables.items(), key=lambda x: (-x[1], x[0]))[:5]),
                'qids': self.token_qids.get(t, [])[:20],
            }

        candidates = set(table_evidence) | set(bm25_order)
        ordered, rank_scores = self._rank_candidates(
            candidates, table_evidence, bm25_order, bm25_score)

        evidence = {
            'tokens': dict(sorted(token_ev.items(), key=lambda x: -x[1]['count'])[:10]),
            'table_evidence': {T: round(v, 6) for T, v in table_evidence.items()},
            'train_qid_pool_size': len(self.train_qids),
            'fusion_mode': ('rank_equal_v2' if self.fusion_mode == 'rank_equal'
                            else self.fusion_mode),
        }

        results = []
        for table in ordered[:k]:
            columns = self._columns_for(query, table, field_evidence)
            results.append(RetrievalResult(
                table=table,
                table_score=rank_scores.get(
                    table, table_evidence.get(table, bm25_score.get(table, 0.0))),
                columns=columns,
                metadata={'method': 'Cooccurrence', 'evidence': evidence},
            ))
        return results

    def _columns_for(self, query: str, table: str,
                     field_evidence: Dict[str, float]) -> List[Tuple[str, float]]:
        schema_cols = self.field_df[self.field_df['table'] == table]
        query_terms = [t for t in self.tokenize(query) if len(t) > 1]
        query_term_set = set(query_terms)

        scored = []
        for _, row in schema_cols.iterrows():
            fname = str(row['field_name'])
            fdesc = str(row.get('field_name_desc', '') or '')
            ev = field_evidence.get(f"{table}.{fname}", 0.0)
            overlap = len(query_term_set & set(
                t for t in self.tokenize(f"{fname} {fdesc}") if len(t) > 1))
            overlap_score = overlap / len(query_terms) if query_terms else 0.0
            if any(t in fname for t in query_term_set):
                overlap_score += 0.5
            scored.append((fname, ev, overlap_score))

        scored.sort(key=lambda x: (-x[1], -x[2], x[0]))
        return [(f"C:{table}.{f}", e + o) for f, e, o in scored[:COLUMN_BUDGET]]
