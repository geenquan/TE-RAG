"""
Historical Example Schema Retrieval（历史样例 Schema 检索）

P1-B 基线 A。准确身份：检索与当前问题最相似的历史问题（NLQ→SQL 样例），
用其引用过的表/字段投票得到 schema 候选。**当前没有 SQL 生成器**，本方法
只做 schema 检索，不是、也不得描述为已完成的端到端 Example-RAG。

监督来源：仅当前训练折的 (question, table, field)（fit 传入的 train_data）。
不读取测试 SQL、测试标注或全局 artifacts。

与 TF-IDF 基线保持一致的处理：分词（BaseRetriever.tokenize）与 TF-IDF
公式（log-TF × idf=log(N/(df+1))，L2 归一化，余弦相似度，同
retrievers/vector_retriever.py）——本方法把这些公式作用在"历史问题"上，
而非 schema 字段上。

固定规则（P1-B 协议预先声明，未依测试效果选择）：
1. 最多取 5 条相似度 > 0 的历史样例（固定 m=5）。
2. 表投票 vote(T) = Σ_e sim(e)，e 为 gold 表含 T 的样例；
   字段投票 vote(T.c) = Σ_e sim(e)，e 为在表 T 下引用过字段 c 的样例。
3. 排序键 = (vote 降序, BM25 分降序, 表名升序)：历史证据优先，零票表由
   原始 schema 的 BM25 排序兜底（复用 retrievers/bm25_retriever.py 实例），
   并列按表名稳定排序（不使用 gold 打破并列）。
4. 零相似度（无任何票）时退化为纯 BM25 —— 未见表因此不会被结构性排除。
5. 每表列预算 10：样例投过票的字段优先（vote 降序），其余用 BM25 式
   查询词-字段重合度（同 bm25_retriever 字段打分）补齐，去重，不足则少给。
"""

import math
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from retrievers.base_retriever import BaseRetriever, RetrievalResult, RetrieverConfig
from retrievers.bm25_retriever import BM25Retriever
from datasets.gt_utils import parse_ground_truth_tables, parse_ground_truth_fields

MAX_EXAMPLES = 5          # 固定：最多检索 5 条历史样例
COLUMN_BUDGET = 10        # 固定：每表列预算（协议统一）
BM25_RANK_DEPTH = 20      # P2-C 预注册：RRF 的 BM25 榜单深度（与最终返回表
                          # 预算 k=5 分开；固定值，不依结果挑选）


def _rrf_rank(candidates, evidence_scores, bm25_order):
    """P2-C 修正版 RRF（rank_equal_v2）。

    与 P2-B 版（rank_equal）的差异（错误修正，非调参）：
    - 历史证据榜只含**实际正证据表**（score>0），按（分降序,表名）赋名次；
      零证据表不获得该榜名次或贡献（旧版对全部候选赋名次，零证据表
      获得虚假历史贡献并与 BM25 榜并列破名）。
    - BM25 榜使用独立固定深度 BM25_RANK_DEPTH（旧版深度=最终 k）。
    公式不变：score(T)=Σ 1/(60+rank_l)，缺失榜单贡献 0，并列按表名升序。
    """
    ev_list = sorted((t for t in candidates if evidence_scores.get(t, 0.0) > 0),
                     key=lambda t: (-evidence_scores.get(t, 0.0), t))
    ev_rank = {t: i + 1 for i, t in enumerate(ev_list)}
    bm_rank = {t: i + 1 for i, t in enumerate(bm25_order)}

    def rrf(t):
        s = 0.0
        if t in ev_rank:
            s += 1.0 / (60 + ev_rank[t])
        if t in bm_rank:
            s += 1.0 / (60 + bm_rank[t])
        return s

    scores = {t: rrf(t) for t in candidates}
    return sorted(candidates, key=lambda t: (-scores[t], t)), scores


class ExampleRAGRetriever(BaseRetriever):
    """Historical Example Schema Retrieval（见模块 docstring）"""

    def __init__(self, field_csv: str, table_csv: str,
                 config: RetrieverConfig = None):
        if config is None:
            config = RetrieverConfig(
                name='ExampleRAG',
                description='Historical Example Schema Retrieval（历史样例投票，非端到端 Example-RAG）')
        super().__init__(field_csv, table_csv, config)

        # 原始 schema BM25 兜底（复用现有实现，不重写）
        self._fallback = BM25Retriever(field_csv, table_csv)

        # fit 后填充
        self.train_qids: List[str] = []          # 样例 qid（按训练行序）
        self._questions: List[str] = []
        self._gold_tables: List[List[str]] = []  # 每样例 gold 表简名列表
        self._gold_fields: List[Dict[str, List[str]]] = {}  # idx -> {table: [fields]}
        self._vocab: Dict[str, int] = {}
        self._idf: Dict[str, float] = {}
        self._q_vectors: List[np.ndarray] = []

        # 融合模式（P2-B）：evidence_first（P1-B 原版）| rank_equal（预注册 RRF）
        self.fusion_mode = ((config.extra_params or {}).get('fusion_mode')
                            if config else None) or 'evidence_first'

    # ---------------- fit ----------------

    def fit(self, train_data: pd.DataFrame = None):
        """仅使用当前训练折数据。train_data 需含 question/table(/field)，
        可含 qid 列（供证据追溯；缺失时用行号 ex000 形式）。"""
        if train_data is None or len(train_data) == 0:
            # 无历史监督：完全退化为 BM25（结构性可达未见表）
            self._fallback.fit(train_data)
            self._is_fitted = True
            return

        self._questions = [str(q) for q in train_data['question'].tolist()]
        self.train_qids = (
            [str(q) for q in train_data['qid'].tolist()]
            if 'qid' in train_data.columns
            else [f"ex{i:03d}" for i in range(len(train_data))])

        gold_tables, gold_fields = [], {}
        for i, (_, row) in enumerate(train_data.iterrows()):
            tables = [t.split('.')[-1] for t in parse_ground_truth_tables(row)]
            fields_by_table = defaultdict(list)
            for t, f in parse_ground_truth_fields(row):
                fields_by_table[t.split('.')[-1]].append(f)
            gold_tables.append(tables)
            gold_fields[i] = fields_by_table
        self._gold_tables = gold_tables
        self._gold_fields = gold_fields

        # TF-IDF over 历史问题（公式同 vector_retriever.py）
        docs = [[w for w in self.tokenize(q) if len(w) > 1] for q in self._questions]
        df_count = defaultdict(int)
        for words in docs:
            for w in set(words):
                df_count[w] += 1
        n_docs = len(docs)
        self._vocab = {w: i for i, w in enumerate(sorted(df_count))}
        self._idf = {w: math.log(n_docs / (c + 1)) for w, c in df_count.items()}
        self._q_vectors = [self._tfidf_vector(words) for words in docs]

        # 兜底 BM25（只依赖 schema CSV）
        self._fallback.fit(train_data)
        self._is_fitted = True

    def _rank_candidates(self, candidates, evidence_scores, bm25_order, bm25_scores):
        """按融合模式排序候选表，返回 (有序列表, 排序分数表)。

        - evidence_first（P1-B 原版，保留追溯）：(证据分降序, BM25 分降序, 表名升序)，
          输出分数 = 证据分（零票表为 BM25 分）。
        - rank_equal（P2-B 预注册 RRF 等权排名融合）：score(T) =
          Σ_{l∈{证据榜,BM25榜}} 1/(60+rank_l(T))，缺失榜单贡献 0；候选 =
          证据表 ∪ BM25 top-k；并列按表名。排名融合避免不同量纲原始分数相加；
          BM25 榜单独立参与（未见表可凭 schema 信号进前列），非仅历史候选内部
          重排；输出分数 = RRF 分数（与排序一致）。
        """
        if self.fusion_mode == 'rank_equal':
            return _rrf_rank(candidates, evidence_scores, bm25_order)
        ordered = sorted(candidates,
                         key=lambda t: (-evidence_scores.get(t, 0.0),
                                        -bm25_scores.get(t, 0.0), t))
        scores = {t: (evidence_scores.get(t, 0.0) or bm25_scores.get(t, 0.0))
                  for t in candidates}
        return ordered, scores

    def _tfidf_vector(self, words: List[str]) -> np.ndarray:
        vec = np.zeros(len(self._vocab))
        count = defaultdict(int)
        for w in words:
            count[w] += 1
        for w, c in count.items():
            if w in self._vocab:
                tf = 1 + math.log(c)
                vec[self._vocab[w]] = tf * self._idf.get(w, 1.0)
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    # ---------------- retrieve ----------------

    def _retrieve(self, query: str, k: int = 5) -> List[RetrievalResult]:
        # 兜底 BM25 榜单：evidence_first 保持原深度 k（行为不变）；
        # rank_equal_v2 使用独立固定深度 BM25_RANK_DEPTH
        depth = BM25_RANK_DEPTH if self.fusion_mode == 'rank_equal' else k
        fallback_results = self._fallback.retrieve(query, k=depth)
        bm25_order = [r.table for r in fallback_results]
        bm25_score = {r.table: float(r.table_score) for r in fallback_results}

        evidence = {'examples': [], 'table_votes': {}, 'field_votes': {}}
        rank_scores: Dict[str, float] = {}

        if self._q_vectors:
            q_vec = self._tfidf_vector([w for w in self.tokenize(query) if len(w) > 1])
            sims = []
            for i, v in enumerate(self._q_vectors):
                s = float(np.dot(q_vec, v)) if q_vec.any() and v.any() else 0.0
                if s > 0:
                    sims.append((s, i))
            # 固定规则：相似度降序、样例序升序，取前 MAX_EXAMPLES
            sims.sort(key=lambda x: (-x[0], x[1]))
            sims = sims[:MAX_EXAMPLES]

            table_votes: Dict[str, float] = defaultdict(float)
            field_votes: Dict[Tuple[str, str], float] = defaultdict(float)
            for s, i in sims:
                evidence['examples'].append(
                    {'qid': self.train_qids[i], 'sim': round(s, 6),
                     'tables': self._gold_tables[i]})
                for t in self._gold_tables[i]:
                    table_votes[t] += s
                    for f in self._gold_fields.get(i, {}).get(t, []):
                        field_votes[(t, f)] += s
            evidence['table_votes'] = dict(table_votes)
            evidence['field_votes'] = {f"{t}.{c}": v for (t, c), v in field_votes.items()}

            # 排序：按 fusion_mode（构造参数，默认 evidence_first）
            candidates = set(table_votes) | set(bm25_order)
            ordered, rank_scores = self._rank_candidates(
                candidates, table_votes, bm25_order, bm25_score)
            evidence['fusion_mode'] = ('rank_equal_v2' if self.fusion_mode == 'rank_equal'
                                       else self.fusion_mode)
        else:
            ordered = bm25_order

        results = []
        for table in ordered[:k]:
            columns = self._columns_for(query, table, evidence['field_votes'],
                                        fallback_results)
            results.append(RetrievalResult(
                table=table,
                table_score=rank_scores.get(
                    table, evidence['table_votes'].get(
                        table, bm25_score.get(table, 0.0))),
                columns=columns,
                metadata={'method': 'ExampleRAG', 'evidence': evidence},
            ))
        return results

    def _columns_for(self, query: str, table: str, field_votes: Dict[str, float],
                     fallback_results) -> List[Tuple[str, float]]:
        """每表列预算 10：投票字段优先，其余按查询词-字段重合度补齐（同 BM25 字段打分）。"""
        schema_cols = self.field_df[self.field_df['table'] == table]
        query_terms = [t for t in self.tokenize(query) if len(t) > 1]
        query_term_set = set(query_terms)

        scored = []
        for _, row in schema_cols.iterrows():
            fname = str(row['field_name'])
            fdesc = str(row.get('field_name_desc', '') or '')
            vote = field_votes.get(f"{table}.{fname}", 0.0)
            overlap = len(query_term_set & set(
                t for t in self.tokenize(f"{fname} {fdesc}") if len(t) > 1))
            overlap_score = overlap / len(query_terms) if query_terms else 0.0
            if any(t in fname for t in query_term_set):
                overlap_score += 0.5
            scored.append((fname, vote, overlap_score))

        # 排序：投票降序 → 重合度降序 → 字段名升序（去重由 schema 行唯一保证）
        scored.sort(key=lambda x: (-x[1], -x[2], x[0]))
        return [(f"C:{table}.{f}", v + o) for f, v, o in scored[:COLUMN_BUDGET]]
