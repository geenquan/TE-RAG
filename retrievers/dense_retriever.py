"""
Dense Schema Retrieval Baseline

真实加载预训练 Embedding 模型（transformers 直接调用），对表级复合
文档与查询做余弦相似度排序。无历史监督。

模型：paraphrase-multilingual-MiniLM-L12-v2（384 维，支持中文）。
加载方式：transformers.AutoModel + Mean Pooling + L2 归一化。
分批编码：固定 BATCH_SIZE=64，顺序一致。
文档构造：表文档=表名+表描述+前 TABLE_COL_LIMIT(30) 个列名；
列文档=列名+列描述。截断已登记。
"""


import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

from retrievers.base_retriever import BaseRetriever, RetrievalResult, RetrieverConfig

COLUMN_BUDGET = 10
BATCH_SIZE = 64
MAX_TOKENS = 256
TABLE_COL_LIMIT = 30  # 表文档中最多包含的列名数（CSV 顺序，非相关性排序）
MODEL_NAME = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'

_MODEL = None
_TOKENIZER = None


def _get_model():
    global _MODEL, _TOKENIZER
    if _MODEL is None:
        import torch
        from transformers import AutoTokenizer, AutoModel
        _TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)
        _MODEL = AutoModel.from_pretrained(MODEL_NAME)
        _MODEL.eval()
    return _TOKENIZER, _MODEL


def _embed(texts: List[str], batch_size: int = BATCH_SIZE) -> np.ndarray:
    """批量编码文本，Mean Pooling，返回 L2 归一化向量。

    顺序保证：输入顺序 = 输出顺序。空输入返回空数组。
    """
    import torch
    if not texts:
        return np.zeros((0, 384), dtype=np.float32)
    tok, model = _get_model()
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = tok(batch, return_tensors='pt', padding=True,
                     truncation=True, max_length=MAX_TOKENS)
        with torch.no_grad():
            outputs = model(**inputs)
            mask = inputs['attention_mask'].unsqueeze(-1).float()
            emb = (outputs.last_hidden_state * mask).sum(1) / mask.sum(1)
        all_embs.append(emb.numpy())
    result = np.concatenate(all_embs, axis=0)
    norms = np.linalg.norm(result, axis=1, keepdims=True)
    return result / (norms + 1e-10)


class DenseRetriever(BaseRetriever):
    """Dense Schema Retrieval（真实预训练 Embedding，非 TF-IDF）"""

    def __init__(self, field_csv: str, table_csv: str,
                 config: RetrieverConfig = None):
        if config is None:
            config = RetrieverConfig(
                name='Dense',
                description='Dense retrieval with pretrained multilingual embedding')
        super().__init__(field_csv, table_csv, config)
        self._table_names: List[str] = []
        self._table_docs: List[str] = []
        self._table_embs: np.ndarray = None
        self._col_keys: Dict[str, List[str]] = {}
        self._col_embs: Dict[str, np.ndarray] = {}

    def fit(self, train_data: pd.DataFrame = None):
        """构建表级与列级 Embedding 索引（纯 schema，不用历史）。

        重复调用安全：每次 fit 前清空所有状态，不累积。
        """
        # 清空所有状态（重复 fit 安全）
        self._table_names = []
        self._table_docs = []
        self._table_embs = None
        self._col_keys = {}
        self._col_embs = {}

        # 表级文档：表名 + 表描述 + 前 TABLE_COL_LIMIT 个列名
        # （列名按 CSV 行顺序截断，非相关性排序，已登记）
        for _, row in self.table_df.iterrows():
            tname = str(row['table'])
            tdesc = str(row.get('table_desc', '') or '')
            cols = self.field_df[self.field_df['table'] == tname]
            col_names = ' '.join(
                cols['field_name'].astype(str).tolist()[:TABLE_COL_LIMIT])
            doc = f"{tname} {tdesc} {col_names}"
            self._table_names.append(tname)
            self._table_docs.append(doc)

        self._table_embs = _embed(self._table_docs)

        # 列级文档：列名 + 列描述（该表全部列）
        for tname in self._table_names:
            cols = self.field_df[self.field_df['table'] == tname]
            docs, keys = [], []
            for _, c in cols.iterrows():
                cname = str(c['field_name'])
                cdesc = str(c.get('field_name_desc', '') or '')
                docs.append(f"{cname} {cdesc}")
                keys.append(cname)
            if docs:
                self._col_keys[tname] = keys
                self._col_embs[tname] = _embed(docs)

        self._is_fitted = True

    def _retrieve(self, query: str, k: int = 5) -> List[RetrievalResult]:
        q_emb = _embed([query])[0]
        scores = self._table_embs @ q_emb
        # 稳定排序：分数降序 + 表名升序（与 Gold 无关）
        order = sorted(range(len(scores)),
                       key=lambda i: (-scores[i], self._table_names[i]))[:k]

        results = []
        for idx in order:
            tname = self._table_names[idx]
            score = float(scores[idx])
            if tname in self._col_embs:
                col_scores = self._col_embs[tname] @ q_emb
                col_order = sorted(
                    range(len(col_scores)),
                    key=lambda i: (-col_scores[i], self._col_keys[tname][i])
                )[:COLUMN_BUDGET]
                columns = [
                    (f"C:{tname}.{self._col_keys[tname][i]}", float(col_scores[i]))
                    for i in col_order
                ]
            else:
                columns = []
            results.append(RetrievalResult(
                table=tname, table_score=score, columns=columns,
                metadata={'method': 'Dense', 'model': MODEL_NAME}))
        return results
