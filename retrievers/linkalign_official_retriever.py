"""LinkAlign single-database Pipeline adaptation, not the old DenseDeepSeek.

The three upstream Pipeline prompts are used unchanged. Database identity is
known; semantic retrieval operates on table/column chunks instead of databases.
Models: multilingual MiniLM and configured DeepSeek replace upstream BGE/Qwen.
"""
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import threading
import time

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from retrievers.base_retriever import BaseRetriever, RetrieverConfig, RetrievalResult
from retrievers.dense_retriever import _embed, MODEL_NAME
from retrievers.linkalign_retriever import LinkAlignRetriever as ExistingClient, _is_fatal_api_error
from retrievers.upstream.linkalign_prompts import (
    QUERY_REWRITING_TEMPLATE, LOCATE_TEMPLATE,
    SCHEMA_LINKING_MANUAL_TEMPLATE, SCHEMA_LINKING_FEW_EXAMPLES)


class SchemaEvidence:
    """Read-only schema index shared by the two adapters; no query SQL/labels."""
    CHUNK_COLUMNS = 24

    def __init__(self, field_df, table_df):
        # Match the evaluator's canonical IDs, not duplicate CSV row numbers.
        self.source_counts = {'table_rows': len(table_df), 'column_rows': len(field_df)}
        self.names = table_df['table'].astype(str).drop_duplicates().tolist()
        self.table_desc = {str(t): ' / '.join(dict.fromkeys(rows['table_desc'].fillna('').astype(str)))
                           for t, rows in table_df.groupby('table', sort=False)}
        self.columns, self.descriptions, self.types = {}, {}, {}
        self.column_keys, column_texts, self.parents = [], [], []
        self.table_texts, self.chunks = [], []
        grouped = {str(t): rows for t, rows in field_df.groupby('table', sort=False)}
        for ti, t in enumerate(self.names):
            rows = grouped.get(t)
            metadata = {}
            if rows is not None:
                rows = rows[rows['field_name'].notna()].copy()
                if 'field_type' not in rows:
                    rows['field_type'] = ''
                values = rows[['field_name', 'field_name_desc', 'field_type']].fillna('').astype(str)
                for c, desc, typ in values.itertuples(index=False, name=None):
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
            self.table_texts.append(f'{t} {self.table_desc[t]} ' + ' '.join(f'{c} {d}' for c, d in zip(cols, descriptions)))
            for offset in range(0, max(len(cols), 1), self.CHUNK_COLUMNS):
                subset = cols[offset:offset + self.CHUNK_COLUMNS]
                self.chunks.append((t, subset, self.ddl({t: subset})))
        self.parents = np.array(self.parents)
        self.table_positions = {t: i for i, t in enumerate(self.names)}
        self.column_positions = {tc: i for i, tc in enumerate(self.column_keys)}
        self.vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 4),
                                         max_features=30000, dtype=np.float32)
        matrix = self.vectorizer.fit_transform(self.table_texts + column_texts)
        self.table_matrix = matrix[:len(self.names)]
        self.column_matrix = matrix[len(self.names):]
        self.table_lengths = np.log1p([len(s) for s in self.table_texts]).astype(np.float32) / 10
        self.column_lengths = np.log1p([len(self.descriptions.get(tc, '')) for tc in self.column_keys]).astype(np.float32) / 10
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
                if ' / ' in typ:  # conflicting source metadata: do not assert a type
                    desc += ' observed types: ' + typ
                    typ = ''
                lines.append(f'  `{c}` {typ} /* {desc} */')
            parts.append(f'/* {self.table_desc.get(t, "")} */\nCREATE TABLE `{t}` (\n' + ',\n'.join(lines) + '\n);')
        return '\n'.join(parts)

    def build_dense(self, cache_dir):
        digest = hashlib.sha256((MODEL_NAME + '|masked_mean_l2|max_tokens=256|chunk_columns=24\n' + '\n'.join(c[2] for c in self.chunks)).encode()).hexdigest()
        path = Path(cache_dir) / f'linkalign_chunks_{digest}.npy'
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            self.chunk_embeddings = np.load(path)
        else:
            vectors = []
            for start in range(0, len(self.chunks), 64):
                vectors.append(_embed([c[2] for c in self.chunks[start:start + 64]]))
                if start % 640 == 0:
                    print(f'LinkAlign schema embedding {start}/{len(self.chunks)}', flush=True)
            self.chunk_embeddings = np.concatenate(vectors)
            np.save(path, self.chunk_embeddings)
        if self.chunk_embeddings.shape != (len(self.chunks), 384):
            raise ValueError('Schema embedding cache shape mismatch')
        self.dense_fingerprint = digest

    def retrieve_chunks(self, question, exclude=(), k=20):
        if self.chunk_embeddings is None:
            raise RuntimeError('Dense schema index not built')
        # The shared fast tokenizer has mutable padding/truncation state.
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
    """Parse upstream list output without eval; reject all unshown references."""
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
    allowed = {f'{t}.{c}'.lower(): (t, c) for t, cols in shown.items() for c in cols}
    accepted, illegal = [], []
    for item in items:
        key = str(item).replace('`', '').strip().lower()
        if key not in allowed:
            illegal.append(str(item))
        elif allowed[key] not in accepted:
            accepted.append(allowed[key])
    return accepted, illegal


class ReviewerRetriever(BaseRetriever):
    """Small shared I/O boundary: existing config reader, durable API evidence."""
    _FATAL_ERRORS = {}
    _CACHE_LOCKS = {}
    def __init__(self, field_csv, table_csv, config=None, schema=None, cache_dir=None):
        super().__init__(field_csv, table_csv, config)
        self.schema = schema or SchemaEvidence(self.field_df, self.table_df)
        self.cache_dir = Path(cache_dir or 'results/reviewer_api_cache')
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._api = ExistingClient(field_csv, table_csv)
        self._fatal = False
        self.calls = []
        self.last_evidence = {}

    def ask(self, stage, prompt):
        # Isolation emits a potentially long list of irrelevant schema columns.
        # Its output limit is a fixed resource bound, not a tuned ranking value.
        request = {'model': 'deepseek-chat', 'temperature': 0.1,
                   'max_tokens': 32768 if stage == 'irrelevant_isolation' else 8192}
        digest = hashlib.sha256(json.dumps([request, prompt], ensure_ascii=False).encode()).hexdigest()
        path = self.cache_dir / f'{digest}.json'
        with self._CACHE_LOCKS.setdefault(digest, threading.Lock()):
            return self._ask_cached(stage, prompt, request, path)

    def _ask_cached(self, stage, prompt, request, path):
        if path.exists():
            ev = json.loads(path.read_text())
            self.calls.append(dict(ev, stage=stage, reused=True))
            return ev['response']
        if self._fatal or self.config.name in self._FATAL_ERRORS:
            raise RuntimeError('Method stopped after fatal API error')
        self._api._init_client()  # reads existing config.yaml; never copies credentials
        for attempt in range(2):
            t0 = time.time()
            try:
                response = self._api._client.chat.completions.create(
                    **request, messages=[{'role': 'user', 'content': prompt}])
            except Exception as e:
                fatal = _is_fatal_api_error(e)
                self.calls.append(dict(stage=stage, request=request, prompt=prompt,
                                       error=type(e).__name__, attempt=attempt + 1,
                                       seconds=time.time() - t0))
                if fatal:
                    self._fatal = True
                    self._FATAL_ERRORS[self.config.name] = type(e).__name__
                if fatal or attempt == 1:
                    raise RuntimeError(f'{stage}: {type(e).__name__}') from None
                time.sleep(2)
                continue
            # A successful paid response must never be retried because parsing
            # or local persistence failed.
            raw = response.choices[0].message.content
            ev = dict(stage=stage, request=request, prompt=prompt, prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                      response=raw, response_id=response.id, response_model=response.model,
                      usage=response.usage.model_dump() if response.usage else None,
                      finish_reason=response.choices[0].finish_reason,
                      attempts=attempt + 1, seconds=time.time() - t0, reused=False)
            self.calls.append(ev)
            path.write_text(json.dumps(ev, ensure_ascii=False))
            return raw

    def project(self, pairs, table_scores, k=5, selected_tables=()):
        grouped = {t: [] for t in selected_tables}
        for t, c in pairs:
            if c not in grouped.setdefault(t, []):
                grouped[t].append(c)
        names = sorted(grouped, key=lambda t: (-table_scores.get(t, 0.0), t))[:k]
        return [RetrievalResult(t, table_scores.get(t, 0.0),
                [(f'C:{t}.{c}', table_scores.get(t, 0.0)) for c in grouped[t][:10]],
                {'method': self.config.name, 'score_source': 'candidate_score_not_llm_probability'})
                for t in names]


class OfficialLinkAlignRetriever(ReviewerRetriever):
    """Upstream Pipeline: semantic enhancement -> isolation -> extraction."""
    def __init__(self, field_csv, table_csv, config=None, **kwargs):
        super().__init__(field_csv, table_csv, config or RetrieverConfig(name='LinkAlign-SingleDB-Adapted'), **kwargs)

    def fit(self, train_data=None):
        self.schema.build_dense(self.cache_dir)
        self._is_fitted = True

    def _retrieve(self, query, k=5):
        self.calls = []
        self.last_evidence = {'method': self.config.name, 'calls': self.calls}
        first = self.schema.retrieve_chunks(query)
        initial = self.schema.shown_from_chunks(first)
        rewritten = self.ask('semantic_enhancement', QUERY_REWRITING_TEMPLATE.format(
            question=query, context=self.schema.ddl(initial)))
        second = self.schema.retrieve_chunks(query + '\n' + rewritten,
                                            exclude=[i for i, _ in first])
        nodes = sorted(first + second, key=lambda x: (-x[1], x[0]))
        shown = self.schema.shown_from_chunks(nodes)
        self.last_evidence.update(rewritten_question=rewritten, retrieved_chunks=nodes,
                                  candidate_columns=shown)
        removed_raw = self.ask('irrelevant_isolation', LOCATE_TEMPLATE.format(
            question=query, context=self.schema.ddl(shown)))
        removed, invalid_removed = parse_schema_links(removed_raw, shown)
        removed = set(removed)
        retained = {t: [c for c in cols if (t, c) not in removed] for t, cols in shown.items()}
        retained = {t: cols for t, cols in retained.items() if cols}
        if not retained:
            self.last_evidence.update(shown_columns={}, removed_columns=sorted(removed),
                                      native_selection=[], illegal=[], invalid_removed=invalid_removed,
                                      parse_status='valid_empty', early_stop='all_candidates_pruned')
            return []
        raw = self.ask('schema_extraction', SCHEMA_LINKING_MANUAL_TEMPLATE.format(
            few_examples=SCHEMA_LINKING_FEW_EXAMPLES,
            context_str='[The Start of Database Schemas]\n' + self.schema.ddl(retained) + '\n[The End of Database Schemas]',
            question=query))
        selected, illegal = parse_schema_links(raw, retained)
        scores = {}
        for i, score in nodes:
            t = self.schema.chunks[i][0]
            scores[t] = max(scores.get(t, -1.0), score)
        self.last_evidence.update(shown_columns=retained, removed_columns=sorted(removed),
                                  native_selection=selected, illegal=illegal,
                                  invalid_removed=invalid_removed,
                                  parse_status=('partial_illegal' if illegal else 'ok') if selected else
                                               ('illegal_elements' if illegal else 'valid_empty'))
        return self.project(selected, scores, k)
