"""
Matched Schema 检索对照（P2-C）

目的：在**与 TE-RAG 当前增强索引完全相同的 Schema 文档、分词与索引评分**下，
提供**无历史监督**的检索对照——不建历史图、不做模式匹配、不做历史字段推荐、
不挖码表。fit 只构建 schema 索引（对训练日志不敏感）；表排序=该索引的 BM25
评分；列排序=与基线相同的查询词-字段重合度（每表预算 10）。

它不是新方法，也不复制 V2：复用 terag/index_builder.IndexBuilder 与其
BM25Retriever 的文档构建与评分；仅作为因素对照存在于 retrievers/。
索引文档变体（字段描述/字段权重）经 TERAGConfig 的 index 旋钮控制，
与 TE-RAG 各变体共用同一实现。
"""

from typing import Dict, List, Tuple

import pandas as pd

from retrievers.base_retriever import BaseRetriever, RetrievalResult, RetrieverConfig

COLUMN_BUDGET = 10  # 协议统一列预算
from retrievers.annotation_miner import W_ANN as W_ANN_D1  # C 类沿用同一加成


class MatchedSchemaRetriever(BaseRetriever):
    """Matched Schema 对照（见模块 docstring）"""

    def __init__(self, field_csv: str, table_csv: str,
                 config: RetrieverConfig = None):
        if config is None:
            config = RetrieverConfig(
                name='MatchedSchema',
                description='Schema 索引对照（无历史监督；同 TE-RAG 文档/分词/评分）')
        super().__init__(field_csv, table_csv, config)
        self._field_csv = field_csv
        self._table_csv = table_csv
        self._bm25 = None
        self._index = None
        self.use_history = False
        self.miner = None

    def fit(self, train_data: pd.DataFrame = None):
        """构建 schema 索引；use_history=True 时同时从**当前训练折**挖掘直接
        注解并注入索引（内存传递，B 类）+ 列排序加成；False 时与纯元数据对照
        逐位一致（postings 相同）。测试 SQL/gold 不进入本方法。"""
        from terag.config import TERAGConfig
        from terag.index_builder import IndexBuilder, BM25Retriever

        injected = (self.config.extra_params or {}).get('terag_config')
        cfg = injected or TERAGConfig()
        cfg = cfg.with_data_paths(field_csv=self._field_csv,
                                  table_csv=self._table_csv,
                                  artifacts_dir=cfg.output.artifacts_dir)

        self.use_history = bool((self.config.extra_params or {}).get('use_history'))
        self.use_transfer = bool((self.config.extra_params or {}).get('use_transfer'))
        self.miner = None
        self.transfer_result = None
        extra = None
        if self.use_history:
            if train_data is None or len(train_data) == 0:
                raise ValueError('use_history=True 需要当前训练折数据（B 类注解来源）')
            from retrievers.annotation_miner import DirectAnnotationMiner
            self.miner = DirectAnnotationMiner(self.field_df, self.table_df,
                                               dialect='mysql').mine(train_data)
            extra = self.miner.table_annotation_texts(
                cfg.index.field_weights.pattern)

        # C 类迁移注解（独立通道，可单独关闭恢复 A+B）
        self._transfer_texts = {}
        self._transfer_terms = {}
        if self.use_history and self.use_transfer:
            from retrievers.annotation_transfer import (
                SchemaInfo, transfer_annotations, transfer_index_texts,
                transfer_column_bonus_map)
            columns = {}
            for t, g in self.field_df.groupby('table'):
                columns[t] = g[['field_name', 'field_name_desc', 'field_type']] \
                    .rename(columns={'field_name_desc': 'field_desc'}) \
                    .to_dict('records')
            table_desc = {r['table']: r.get('table_desc', '')
                          for _, r in self.table_df.iterrows()}
            schema = SchemaInfo(columns=columns, table_desc=table_desc)
            self.transfer_result = transfer_annotations(
                self.miner.annotations, schema)
            self._transfer_texts = transfer_index_texts(
                self.transfer_result, cfg.index.field_weights.pattern)
            self._transfer_terms = transfer_column_bonus_map(self.transfer_result)
            merged = dict(extra or {})
            for t, texts in self._transfer_texts.items():
                merged.setdefault(t, [])
                merged[t] = list(merged.get(t, [])) + list(texts)
            extra = merged

        index = IndexBuilder(cfg).build(extra_table_texts=extra)
        self._bm25 = BM25Retriever(index)
        self._index = index
        self._is_fitted = True

    def _retrieve(self, query: str, k: int = 5) -> List[RetrievalResult]:
        results = self._bm25.search(query, k=k)
        out = []
        for doc_id, score in results:
            table = doc_id.replace('T:', '')
            out.append(RetrievalResult(
                table=table,
                table_score=float(score),
                columns=self._columns_for(query, table),
                metadata=self._metadata(table),
            ))
        return out

    def _metadata(self, table: str) -> Dict:
        meta = {'method': 'MatchedSchema',
                'use_history': self.use_history}
        if self.miner is not None:
            meta['annotation_summary'] = self.miner.summary()
        return meta

    def _columns_for(self, query: str, table: str) -> List[Tuple[str, float]]:
        """列打分：查询词-字段重合度 + 字段名子串加分（与基线一致）+
        B 类注解命中加成（use_history 时，w_ann=0.5 固定），预算 10。"""
        schema_cols = self.field_df[self.field_df['table'] == table]
        query_terms = [t for t in self.tokenize(query) if len(t) > 1]
        query_term_set = set(query_terms)
        scored = []
        for _, row in schema_cols.iterrows():
            fname = str(row['field_name'])
            fdesc = str(row.get('field_name_desc', '') or '')
            field_terms = set(t for t in self.tokenize(f"{fname} {fdesc}") if len(t) > 1)
            overlap = len(query_term_set & field_terms)
            s = overlap / len(query_terms) if query_terms else 0.0
            if any(t in fname for t in query_term_set):
                s += 0.5
            if self.miner is not None:
                s += self.miner.column_bonus(query_term_set, table, fname)
                for term in self._transfer_terms.get(
                        f"{table.lower()}.{fname.lower()}", []):
                    if term in query_term_set:
                        s += W_ANN_D1
            scored.append((fname, s))
        scored.sort(key=lambda x: (-x[1], x[0]))
        return [(f"C:{table}.{f}", s) for f, s in scored[:COLUMN_BUDGET]]
