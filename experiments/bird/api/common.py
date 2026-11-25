"""Shared plumbing for the three brokered API methods.

Copied from workspace retrievers/linkalign_official_retriever.py (frozen
identity) with ONLY these adaptations, all documented in API_READY.json:
- _embed comes from experiments.bird.api.embed_local (same cached MiniLM weights and
  identical pooling math; local_files_only loading path);
- ask() routes through adapter.api.api_client.call_api (broker, evidence,
  cache) instead of a raw OpenAI client;
- model id deepseek-flash with thinking disabled (was deepseek-chat);
- the HTTP cache lives in api_evidence/http_cache keyed by
  (model, stage, sha256(request_body)) and is shared across all methods.
Everything else (chunking, DDL rendering, lexical features, parse rules,
budget projection) is byte-for-byte the upstream logic.
"""
import ast
import hashlib
import json
import re
import threading
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from experiments.bird.api.api_client import MethodBlocked, call_api
from experiments.bird.api.config import DENSE_CACHE_DIR
from experiments.bird.api.embed_local import _embed
from retrievers.base_retriever import BaseRetriever, RetrievalResult

MODEL_NAME = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'

# Stage -> max_tokens (identity of the reference implementations; the
# isolation stage emits a potentially long list of irrelevant fields).
STAGE_MAX_TOKENS = {
    'semantic_enhancement': 8192,
    'irrelevant_isolation': 32768,
    'schema_extraction': 8192,
    'binary_schema_linker': 8192,
    'schema_selection': 2000,
}


class ApiAdapterMixin:
    """Attrs consumed by call_api + the fatal latch shared per method."""
    api_method = 'unknown'
    _FATAL_ERRORS = {}

    def __init__(self, unit_label=''):
        self.api_unit = unit_label
        self.api_qid = ''
        self._fatal_reason = None

    def ask(self, stage, prompt):
        """One brokered LLM call at the stage's fixed max_tokens."""
        mt = STAGE_MAX_TOKENS[stage]
        try:
            res = call_api(self, stage, prompt, mt)
        except MethodBlocked as exc:
            type(self)._FATAL_ERRORS[self.api_method] = exc.reason
            self._fatal_reason = exc.reason
            raise
        self.calls.append({'stage': stage, 'req_id': res.req_id,
                           'response_id': res.response_id,
                           'response_model': res.response_model,
                           'usage': res.usage, 'cached': res.cached,
                           'cost_cny': res.cost_cny,
                           'attempts': res.attempts})
        return res.content

    def _check_fatal(self):
        reason = type(self)._FATAL_ERRORS.get(self.api_method)
        if reason:
            raise MethodBlocked(reason,
                                'method already stopped by a fatal error')


class SchemaEvidence:
    """Read-only schema index shared by the adapters; no query SQL/labels.

    Verbatim copy of the workspace class (chunking, DDL, TF-IDF, positions);
    only the embedding function is the local-snapshot loader with identical
    math.
    """
    CHUNK_COLUMNS = 24

    def __init__(self, field_df, table_df):
        self.source_counts = {'table_rows': len(table_df),
                              'column_rows': len(field_df)}
        self.names = table_df['table'].astype(str).drop_duplicates().tolist()
        self.table_desc = {str(t): ' / '.join(dict.fromkeys(
            rows['table_desc'].fillna('').astype(str)))
            for t, rows in table_df.groupby('table', sort=False)}
        self.columns, self.descriptions, self.types = {}, {}, {}
        self.column_keys, column_texts, self.parents = [], [], []
        self.table_texts, self.chunks = [], []
        grouped = {str(t): rows for t, rows in
                   field_df.groupby('table', sort=False)}
        for ti, t in enumerate(self.names):
            rows = grouped.get(t)
            metadata = {}
            if rows is not None:
                rows = rows[rows['field_name'].notna()].copy()
                if 'field_type' not in rows:
                    rows['field_type'] = ''
                values = rows[['field_name', 'field_name_desc',
                               'field_type']].fillna('').astype(str)
                for c, desc, typ in values.itertuples(index=False,
                                                      name=None):
                    observed = metadata.setdefault(c, ({}, {}))
                    observed[0][desc], observed[1][typ] = None, None
            cols = list(metadata)
            self.columns[t] = cols
            descriptions = [' / '.join(metadata[c][0]) for c in cols]
            types = [' / '.join(metadata[c][1]) for c in cols]
            for c, desc, typ in zip(cols, descriptions, types):
                self.descriptions[t, c], self.types[t, c] = desc, typ
                self.column_keys.append((t, c))
                self.parents.append(ti)
                column_texts.append(f'{t} {self.table_desc[t]} {c} {desc}')
            self.table_texts.append(
                f'{t} {self.table_desc[t]} ' +
                ' '.join(f'{c} {d}'
                         for c, d in zip(cols, descriptions)))
            for offset in range(0, max(len(cols), 1), self.CHUNK_COLUMNS):
                subset = cols[offset:offset + self.CHUNK_COLUMNS]
                self.chunks.append((t, subset, self.ddl({t: subset})))
        self.parents = np.array(self.parents)
        self.table_positions = {t: i for i, t in enumerate(self.names)}
        self.column_positions = {tc: i for i, tc in
                                 enumerate(self.column_keys)}
        self.vectorizer = TfidfVectorizer(analyzer='char',
                                          ngram_range=(2, 4),
                                          max_features=30000,
                                          dtype=np.float32)
        matrix = self.vectorizer.fit_transform(self.table_texts +
                                               column_texts)
        self.table_matrix = matrix[:len(self.names)]
        self.column_matrix = matrix[len(self.names):]
        self.table_lengths = np.log1p(
            [len(s) for s in self.table_texts]).astype(np.float32) / 10
        self.column_lengths = np.log1p(
            [len(self.descriptions.get(tc, '')) for tc in
             self.column_keys]).astype(np.float32) / 10
        self._score_cache = {}
        self.chunk_embeddings = None
        self._embedding_lock = threading.Lock()

    def ddl(self, shown):
        parts = []
        for t, cols in shown.items():
            lines = []
            for c in cols:
                desc = self.descriptions.get((t, c), '').replace('\n', ' ')
                typ = self.types.get((t, c), '')
                if ' / ' in typ:  # conflicting metadata: assert no type
                    desc += ' observed types: ' + typ
                    typ = ''
                lines.append(f'  `{c}` {typ} /* {desc} */')
            parts.append(f'/* {self.table_desc.get(t, "")} */\n'
                         f'CREATE TABLE `{t}` (\n' + ',\n'.join(lines) +
                         '\n);')
        return '\n'.join(parts)

    def build_dense(self, cache_dir=None):
        cache_dir = Path(cache_dir) if cache_dir else DENSE_CACHE_DIR
        digest = hashlib.sha256(
            (MODEL_NAME + '|masked_mean_l2|max_tokens=256|chunk_columns=24\n'
             + '\n'.join(c[2] for c in self.chunks)).encode()).hexdigest()
        path = cache_dir / f'linkalign_chunks_{digest}.npy'
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            self.chunk_embeddings = np.load(path)
        else:
            vectors = []
            for start in range(0, len(self.chunks), 64):
                vectors.append(
                    _embed([c[2] for c in self.chunks[start:start + 64]]))
            self.chunk_embeddings = np.concatenate(vectors)
            tmp = path.with_suffix('.tmp.npy')
            np.save(tmp, self.chunk_embeddings)
            tmp.replace(path)
        if self.chunk_embeddings.shape != (len(self.chunks), 384):
            raise ValueError('Schema embedding cache shape mismatch')
        self.dense_fingerprint = digest

    def retrieve_chunks(self, question, exclude=(), k=20):
        if self.chunk_embeddings is None:
            raise RuntimeError('Dense schema index not built')
        with self._embedding_lock:
            scores = self.chunk_embeddings @ _embed([question])[0]
        excluded = set(exclude)
        order = sorted((i for i in range(len(scores)) if i not in excluded),
                       key=lambda i: (-float(scores[i]), i))[:k]
        return [(i, float(scores[i])) for i in order]

    def shown_from_chunks(self, nodes):
        shown = {}
        for i, _ in nodes:
            t, cols, _ = self.chunks[i]
            target = shown.setdefault(t, [])
            target.extend(c for c in cols if c not in target)
        return shown

    def lexical_features(self, question):
        if question not in self._score_cache:
            q = self.vectorizer.transform([question])
            tables = (self.table_matrix @ q.T).toarray().ravel()
            columns = (self.column_matrix @ q.T).toarray().ravel()
            self._score_cache[question] = tables, columns
        return self._score_cache[question]


def parse_schema_links(raw, shown):
    """Parse upstream list output without eval; reject all unshown refs."""
    start, end = raw.rfind('['), raw.rfind(']')
    if start < 0 or end < start:
        raise ValueError('Upstream schema output has no complete list')
    text = raw[start:end + 1].replace('`', '')
    try:
        items = ast.literal_eval(text)
        if not isinstance(items, list):
            raise ValueError('Expected list')
    except (SyntaxError, ValueError):
        items = [f'{t}.{c}' for t, c in re.findall(
            r'[\'\"]?(\w+)[\'\"]?\s*\.\s*[\'\"]?(\w+)[\'\"]?', text)]
        if not items and text.strip() != '[]':
            raise ValueError('Unparseable schema list')
    allowed = {f'{t}.{c}'.lower(): (t, c)
               for t, cols in shown.items() for c in cols}
    accepted, illegal = [], []
    for item in items:
        key = str(item).replace('`', '').strip().lower()
        if key not in allowed:
            illegal.append(str(item))
        elif allowed[key] not in accepted:
            accepted.append(allowed[key])
    return accepted, illegal


class ReviewerBase(BaseRetriever, ApiAdapterMixin):
    """I/O boundary of the workspace ReviewerRetriever, broker-routed."""
    def __init__(self, field_csv, table_csv, config, schema=None,
                 cache_dir=None, unit_label=''):
        BaseRetriever.__init__(self, field_csv, table_csv, config)
        ApiAdapterMixin.__init__(self, unit_label)
        self.schema = schema or SchemaEvidence(self.field_df, self.table_df)
        self.cache_dir = Path(cache_dir or
                              (DENSE_CACHE_DIR / 'reviewer_api_cache'))
        self.calls = []
        self.last_evidence = {}

    def project(self, pairs, table_scores, k=5, selected_tables=()):
        grouped = {t: [] for t in selected_tables}
        for t, c in pairs:
            if c not in grouped.setdefault(t, []):
                grouped[t].append(c)
        names = sorted(grouped,
                       key=lambda t: (-table_scores.get(t, 0.0), t))[:k]
        return [RetrievalResult(
            t, table_scores.get(t, 0.0),
            [(f'C:{t}.{c}', table_scores.get(t, 0.0))
             for c in grouped[t][:10]],
            {'method': self.config.name,
             'score_source': 'candidate_score_not_llm_probability'})
            for t in names]
