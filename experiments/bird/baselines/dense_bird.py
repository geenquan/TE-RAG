"""Dense baseline adapter for the BIRD paper-completion offline run (Offline agent).

Identity (CONTRACTS6.md): real MiniLM `paraphrase-multilingual-MiniLM-L12-v2`
at cached revision e8f8c211226b894fcb81acc59f3b34ba3efd5f42 — mean pooling
over the attention mask, L2 normalization, 256-token truncation, per
workspace `retrievers/dense_retriever.py`:
  - table document text = table name + table description + the first
    TABLE_COL_LIMIT(30) column names in schema (CSV) order;
  - table ranking = cosine(query, table_doc) with (-score, table name) order;
  - per-table column budget = 10.

sentence_transformers is BROKEN in the pinned env (CONTRACTS6), so this module
loads the SAME cached weights via transformers.AutoModel/AutoTokenizer with
math identical to SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
mean-pooling: tokenize (padding, truncation, max_length=256) ->
last_hidden_state masked mean -> L2 normalize. The snapshot path is pinned and
opened with local_files_only=True (offline, no downloads). Loading
equivalence is documented in adapter/OFFLINE_READY.json.

Column ranking: the BM25-baseline overlap scoring family (English tokens):
score = |query_terms ∩ field(name+desc) terms| / |query_terms| (+0.5 if any
query term appears inside the field name), selection mechanics mirrored from
dense_retriever.py (per-table sort by (-score, field name), top 10). Documented
in OFFLINE_READY.json.

Fit is per-db: table documents are encoded ONCE per database and cached under
<cache_root>/dense/<db_id>/<schema_sha256>.npz (+ .json sidecar), keyed by the
SHA-256 of the two schema CSVs; a second fit with the same schema loads the
cache and returns the same vectors (verified in tests).

torch.set_num_threads(1) is applied at model load (deterministic, single
thread per the run env).

Public API (mirrors adapter.baselines.run_baselines.run_baseline plus
cache_root):
    run_dense(cfg, db_id, train_rows, test_qids, questions, field_csv,
              table_csv, cache_root) -> list[record]

Record shape/budgets/error semantics: see adapter/baselines/tfidf_bird.py
(shared run_retriever_records helper). The runner stamps protocol/fold.
Dense runs on the question protocol ONLY (CONTRACTS6).
"""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from retrievers.base_retriever import BaseRetriever, RetrieverConfig, RetrievalResult

from experiments.bird.baselines.tfidf_bird import run_retriever_records

METHOD = 'Dense'
K_TABLES = 5
COLUMN_BUDGET = 10
BATCH_SIZE = 64           # dense_retriever.py fixed batch size
MAX_TOKENS = 256          # dense_retriever.py fixed truncation
TABLE_COL_LIMIT = 30      # dense_retriever.py: column names per table doc
EMB_DIM = 384             # MiniLM-L12 output dimension

MODEL_NAME = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
MODEL_REVISION = 'e8f8c211226b894fcb81acc59f3b34ba3efd5f42'
MODEL_SNAPSHOT = (
    '/Users/apple/.cache/huggingface/hub/models--sentence-transformers--'
    f'paraphrase-multilingual-MiniLM-L12-v2/snapshots/{MODEL_REVISION}')

IDENTITY = {
    'method': METHOD,
    'model': MODEL_NAME,
    'revision': MODEL_REVISION,
    'snapshot_path': MODEL_SNAPSHOT,
    'loading': 'transformers.AutoModel/AutoTokenizer local_files_only=True '
               '(sentence_transformers broken in pinned env); math identical '
               'to SentenceTransformer mean pooling + L2 normalize',
    'pooling': 'masked mean over attention_mask, L2 normalized',
    'max_tokens': MAX_TOKENS,
    'batch_size': BATCH_SIZE,
    'table_doc': 'table name + table_desc + first '
                 f'{TABLE_COL_LIMIT} column names (schema CSV order)',
    'column_ranking': 'BM25-baseline overlap family (English tokens), '
                      'per-table top-10 by (-score, field name)',
    'torch': 'torch.set_num_threads(1) at model load',
}

_MODEL = None
_TOKENIZER = None


def _get_model():
    """Singleton model load: pinned local snapshot, offline, single thread."""
    global _MODEL, _TOKENIZER
    if _MODEL is None:
        import torch
        from transformers import AutoTokenizer, AutoModel
        torch.set_num_threads(1)
        _TOKENIZER = AutoTokenizer.from_pretrained(MODEL_SNAPSHOT,
                                                   local_files_only=True)
        _MODEL = AutoModel.from_pretrained(MODEL_SNAPSHOT, local_files_only=True)
        _MODEL.eval()
    return _TOKENIZER, _MODEL


def embed(texts: List[str], batch_size: int = BATCH_SIZE) -> np.ndarray:
    """Encode texts: masked mean pooling + L2 normalize (dense_retriever math).

    Order guarantee: input order = output order. Empty input -> (0, 384).
    Deterministic: eval mode, no_grad, fixed batch order, single thread.
    """
    import torch
    if not texts:
        return np.zeros((0, EMB_DIM), dtype=np.float32)
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


def table_doc_text(table_name: str, table_desc: str,
                   column_names: List[str]) -> str:
    """dense_retriever.py identity: '{name} {desc} {first-30 column names}'."""
    col_names = ' '.join(str(c) for c in list(column_names)[:TABLE_COL_LIMIT])
    return f"{table_name} {table_desc} {col_names}"


def _schema_sha256(field_csv: str, table_csv: str) -> str:
    """Cache key: SHA-256 over both schema CSV file bytes."""
    h = hashlib.sha256()
    for path in (field_csv, table_csv):
        with Path(path).open('rb') as f:
            for block in iter(lambda: f.read(1048576), b''):
                h.update(block)
        h.update(b'\n\x00schema-file-boundary\x00\n')
    return h.hexdigest()


class DenseBirdRetriever(BaseRetriever):
    """Dense schema retrieval: real MiniLM table embeddings, per-db cache.

    Subclasses the frozen BaseRetriever for schema loading + English
    tokenization (column overlap scoring); no workspace file is modified.
    """

    def __init__(self, field_csv: str, table_csv: str, db_id: str,
                 cache_root):
        super().__init__(field_csv, table_csv,
                         RetrieverConfig(name=METHOD,
                                         use_chinese_tokenizer=False))
        self.db_id = str(db_id)
        self.cache_root = Path(cache_root)
        self.schema_sha256 = _schema_sha256(field_csv, table_csv)
        self._table_names: List[str] = []
        self._table_docs: List[str] = []
        self._table_embs: np.ndarray = None
        self._cache_path = None
        self._cache_loaded = False  # test hook: True when cache file reused

    # ---------------- fit (per-db, cache keyed by schema SHA) ----------------

    def fit(self, train_data: pd.DataFrame = None):
        """Build the per-db table embedding index (schema-only, no history).

        Encodes table documents ONCE per database; the result is cached under
        <cache_root>/dense/<db_id>/<schema_sha256>.npz and reloaded verbatim on
        any later fit with the same schema (train_data is accepted for
        interface parity and ignored — the Dense identity is unsupervised).
        """
        cache_dir = self.cache_root / 'dense' / self.db_id
        cache_dir.mkdir(parents=True, exist_ok=True)
        npz_path = cache_dir / f'{self.schema_sha256}.npz'
        sidecar_path = cache_dir / f'{self.schema_sha256}.json'
        self._cache_path = npz_path

        if npz_path.is_file() and sidecar_path.is_file():
            # Unicode-dtype string arrays load without pickle.
            saved = np.load(npz_path, allow_pickle=False)
            self._table_names = [str(x) for x in saved['table_names'].tolist()]
            self._table_docs = [str(x) for x in saved['table_docs'].tolist()]
            self._table_embs = saved['table_embs'].astype(np.float32)
            self._cache_loaded = True
            if len(self._table_names) != len(self._table_embs):
                raise RuntimeError('dense cache corruption: name/vector count mismatch')
            self._is_fitted = True
            return

        docs, names = [], []
        for _, row in self.table_df.iterrows():
            tname = str(row['table'])
            tdesc = str(row.get('table_desc', '') or '')
            cols = self.field_df[self.field_df['table'] == tname]
            docs.append(table_doc_text(tname, tdesc,
                                       cols['field_name'].astype(str).tolist()))
            names.append(tname)
        self._table_names = names
        self._table_docs = docs
        self._table_embs = embed(docs).astype(np.float32)

        # Write via an open file object (np.savez would otherwise append
        # '.npz' to a tmp name), then atomically move into place.
        tmp = npz_path.with_name(npz_path.name + '.tmp')
        with tmp.open('wb') as f:
            np.savez(f, table_names=np.array(names, dtype=np.str_),
                     table_docs=np.array(docs, dtype=np.str_),
                     table_embs=self._table_embs)
        tmp.replace(npz_path)
        sidecar_path.write_text(json.dumps(
            {'db_id': self.db_id, 'schema_sha256': self.schema_sha256,
             'n_tables': len(names), 'identity': IDENTITY}, ensure_ascii=False,
            indent=1, default=str) + '\n', encoding='utf-8')
        self._cache_loaded = False
        self._is_fitted = True

    def cache_info(self) -> Dict[str, Any]:
        """Cache status for tests/auditing."""
        return {'cache_path': str(self._cache_path) if self._cache_path else None,
                'schema_sha256': self.schema_sha256,
                'loaded_from_cache': self._cache_loaded,
                'n_tables': len(self._table_names)}

    # ---------------- retrieve ----------------

    def _columns_for(self, query_terms: List[str], table: str) -> List[tuple]:
        """BM25-baseline overlap scoring family; dense_retriever selection
        mechanics (per-table sort by (-score, field name), top 10)."""
        query_term_set = set(query_terms)
        scored = []
        for _, row in self.field_df[self.field_df['table'] == table].iterrows():
            field_name = str(row['field_name'])
            field_desc = str(row.get('field_name_desc', '') or '')
            field_terms = set(t for t in self.tokenize(
                f"{field_name} {field_desc}") if len(t) > 1)
            overlap = len(query_term_set & field_terms)
            col_score = overlap / len(query_terms) if query_terms else 0.0
            if any(t in field_name for t in query_term_set):
                col_score += 0.5
            scored.append((field_name, col_score))
        scored.sort(key=lambda x: (-x[1], x[0]))
        return [(f"C:{table}.{f}", s) for f, s in scored[:COLUMN_BUDGET]]

    def _retrieve(self, query: str, k: int = K_TABLES) -> List[RetrievalResult]:
        q_emb = embed([query])[0]
        scores = self._table_embs @ q_emb
        # Stable order exactly as dense_retriever.py: (-score, table name).
        order = sorted(range(len(scores)),
                       key=lambda i: (-scores[i], self._table_names[i]))[:k]
        query_terms = [t for t in self.tokenize(query) if len(t) > 1]
        results = []
        for idx in order:
            tname = self._table_names[idx]
            results.append(RetrievalResult(
                table=tname, table_score=float(scores[idx]),
                columns=self._columns_for(query_terms, tname),
                metadata={'method': METHOD, 'model': MODEL_NAME,
                          'revision': MODEL_REVISION}))
        return results


def build_fitted_retriever(cfg, field_csv: str, table_csv: str, db_id: str,
                           train_rows: pd.DataFrame, cache_root):
    """Construct and fit DenseBirdRetriever (per-db encode, cache-aware)."""
    retriever = DenseBirdRetriever(field_csv, table_csv, db_id, cache_root)
    retriever.fit(train_rows)
    return retriever


def run_dense(cfg, db_id: str, train_rows: pd.DataFrame,
              test_qids: List[str], questions, field_csv: str,
              table_csv: str, cache_root) -> List[Dict[str, Any]]:
    """Run the Dense baseline over all test questions of one database.

    Args:
        cfg: TERAGConfig (signature parity; Dense does not consume it).
        db_id: database id (also the per-db cache key namespace).
        train_rows: DataFrame (accepted for parity; Dense is unsupervised).
        test_qids: qids to evaluate, in order; every qid yields a record.
        questions: {qid: question_text} mapping (or DataFrame qid/question).
        field_csv, table_csv: per-db schema CSV paths.
        cache_root: directory under which dense/<db_id>/ caches live
                    (shared across folds: one encode per database).

    Returns:
        One record per test qid (same order); per-question errors keep the
        fixed denominator (status='error', empty outputs).
    """
    retriever = build_fitted_retriever(cfg, field_csv, table_csv, db_id,
                                       train_rows, cache_root)
    return run_retriever_records(retriever, METHOD, db_id, test_qids, questions)
