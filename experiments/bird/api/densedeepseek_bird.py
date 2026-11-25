"""DenseDeepSeek (BIRD, brokered): Dense top-20 tables x 10 cols -> LLM
selection.

Adapted copy of workspace retrievers/linkalign_retriever.py. ONLY these
changes vs the reference (all documented in API_READY.json):
- model id deepseek-flash (was deepseek-chat) with thinking disabled, routed
  through adapter.api.api_client.call_api (broker + evidence + HTTP cache);
- MiniLM embeddings via adapter.api.embed_local (same cached weights,
  identical pooling math);
- per-question evidence kept, request block updated to the new model id.
UNCHANGED: SCHEMA_SELECT_PROMPT text, temperature=0.1, max_tokens=2000,
candidate/table/column limits, prompt construction, parse/validation rules,
budget caps (5 tables x 10 columns).
"""
import hashlib
import json
import re
import time
from typing import Dict, List, Set, Tuple

import numpy as np

from experiments.bird.api.api_client import (ApiTransientError, MethodBlocked,
                                    call_api)
from experiments.bird.api.common import ApiAdapterMixin, STAGE_MAX_TOKENS
from experiments.bird.api.embed_local import _embed
from retrievers.base_retriever import BaseRetriever, RetrievalResult

EMBED_MODEL = ('sentence-transformers/'
               'paraphrase-multilingual-MiniLM-L12-v2')

COLUMN_BUDGET = 10
TABLE_BUDGET = 5
CANDIDATE_K = 20
CTX_COL_LIMIT = 10        # max columns shown per table in the prompt
CTX_TABLE_LIMIT = 20      # max candidate tables shown in the prompt

SCHEMA_SELECT_PROMPT = """You are a schema linking expert. Given a natural language question and candidate database schema elements, select the tables and columns needed to answer the question.

## Candidate Schema Elements
{schema_text}

## Question
{question}

## Task
Select the tables and columns from the above candidates that are relevant to answering this question. Only select from what is shown above.

Output your answer in this exact JSON format:
{{
  "tables": [
    {{
      "table": "table_name",
      "columns": ["col1", "col2"]
    }}
  ]
}}

Rules:
- At most {table_budget} tables.
- At most {col_budget} columns per table.
- Only select tables and columns explicitly shown above.
- Output only valid JSON, no other text."""


class DenseDeepSeekBird(BaseRetriever, ApiAdapterMixin):
    """Dense+DeepSeek schema selector on the brokered deepseek-flash API."""

    api_method = 'DenseDeepSeek'
    _FATAL_ERRORS = {}

    def __init__(self, field_csv: str, table_csv: str, config=None,
                 unit_label: str = ''):
        from retrievers.base_retriever import RetrieverConfig
        if config is None:
            config = RetrieverConfig(
                name='DenseDeepSeek',
                description='Dense+DeepSeek Schema Selector (custom, NOT '
                            'LinkAlign); deepseek-flash brokered')
        BaseRetriever.__init__(self, field_csv, table_csv, config)
        ApiAdapterMixin.__init__(self, unit_label)
        self._table_names: List[str] = []
        self._table_docs: List[str] = []
        self._table_embs: np.ndarray = None
        self._col_keys: Dict[str, List[str]] = {}
        self._col_descs: Dict[str, Dict[str, str]] = {}
        self._table_descs: Dict[str, str] = {}
        self._col_embs: Dict[str, np.ndarray] = {}
        self._parse_errors = 0
        self._illegal_items: List[dict] = []
        self._current_fold = None
        self._last_query_evidence: Dict = {}
        self.calls = []  # per-question call summaries (runner api block)

    # ------------------------------------------------------------------ fit
    def fit(self, train_data=None):
        """Build the dense index (same doc construction as DenseRetriever)."""
        self._table_names = []
        self._table_docs = []
        self._table_embs = None
        self._col_keys = {}
        self._col_descs = {}
        self._table_descs = {}
        self._col_embs = {}

        TABLE_COL_LIMIT = 30  # dense_retriever.TABLE_COL_LIMIT
        for _, row in self.table_df.iterrows():
            tname = str(row['table'])
            tdesc = str(row.get('table_desc', '') or '')
            cols = self.field_df[self.field_df['table'] == tname]
            col_names = ' '.join(
                cols['field_name'].astype(str).tolist()[:TABLE_COL_LIMIT])
            doc = f"{tname} {tdesc} {col_names}"
            self._table_names.append(tname)
            self._table_docs.append(doc)
            self._table_descs[tname] = tdesc
            keys, descs = [], {}
            for _, c in cols.iterrows():
                cname = str(c['field_name'])
                cdesc = str(c.get('field_name_desc', '') or '')
                keys.append(cname)
                descs[cname] = cdesc
            self._col_keys[tname] = keys
            self._col_descs[tname] = descs

        self._table_embs = _embed(self._table_docs)
        for tname in self._table_names:
            cols = self.field_df[self.field_df['table'] == tname]
            docs = [f"{str(c['field_name'])} "
                    f"{str(c.get('field_name_desc', '') or '')}"
                    for _, c in cols.iterrows()]
            if docs:
                self._col_embs[tname] = _embed(docs)
        self._is_fitted = True

    # -------------------------------------------------------- prompt build
    def _build_schema_text(self, candidate_indices: List[int],
                           q_emb: np.ndarray = None
                           ) -> Tuple[str, List[str], Dict[str, List[str]]]:
        parts = []
        shown_tables: List[str] = []
        shown_cols: Dict[str, List[str]] = {}
        for idx in candidate_indices[:CTX_TABLE_LIMIT]:
            tname = self._table_names[idx]
            tdesc = self._table_descs.get(tname, '')
            keys = self._col_keys.get(tname, [])
            descs = self._col_descs.get(tname, {})
            if q_emb is not None and keys and tname in self._col_embs:
                col_scores = self._col_embs[tname] @ q_emb
                col_order = sorted(
                    range(len(keys)),
                    key=lambda i: (-col_scores[i], keys[i]))[:CTX_COL_LIMIT]
                display_keys = [keys[i] for i in col_order]
            else:
                display_keys = keys[:CTX_COL_LIMIT]
            cols_str = '; '.join(
                f"{k}({descs.get(k, '')})" for k in display_keys)
            parts.append(f"Table: {tname}\nDescription: {tdesc}\n"
                         f"Columns: {cols_str}")
            shown_tables.append(tname)
            shown_cols[tname] = list(display_keys)
        return '\n\n'.join(parts), shown_tables, shown_cols

    # --------------------------------------------------------------- parse
    def _parse_response(self, response: str, shown_tables: Set[str],
                        shown_cols: Dict[str, Set[str]]
                        ) -> Tuple[List[Tuple[str, List[str]]], str]:
        self._illegal_items = []
        text = response.strip()
        m = re.search(r'```(?:json)?\s*(.*?)\s*```', text, re.S)
        if m:
            text = m.group(1)
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            self._parse_errors += 1
            return [], 'json_error'

        if not isinstance(data, dict):
            self._parse_errors += 1
            return [], 'json_error'
        tables = data.get('tables', [])
        if not isinstance(tables, list):
            self._parse_errors += 1
            return [], 'json_error'

        result = []
        seen_tables_in_result = set()
        for t in tables:
            if not isinstance(t, dict):
                self._illegal_items.append(
                    {'type': 'bad_table_entry', 'value': str(t)[:50]})
                continue
            tname_raw = t.get('table')
            tname = str(tname_raw) if tname_raw is not None else ''
            columns_raw = t.get('columns', [])
            if not isinstance(columns_raw, list):
                self._illegal_items.append(
                    {'type': 'bad_columns_type', 'table': tname,
                     'value_type': type(columns_raw).__name__})
                continue
            if tname not in shown_tables:
                self._illegal_items.append(
                    {'type': 'unshown_table', 'table': tname})
                continue
            if tname in seen_tables_in_result:
                self._illegal_items.append(
                    {'type': 'duplicate_table', 'table': tname})
                continue
            shown_for_this = shown_cols.get(tname, set())
            valid = []
            seen_cols = set()
            for c in columns_raw:
                cname = str(c)
                if cname not in shown_for_this:
                    self._illegal_items.append(
                        {'type': 'unshown_column', 'table': tname,
                         'column': cname})
                    continue
                if cname in seen_cols:
                    self._illegal_items.append(
                        {'type': 'duplicate_column', 'table': tname,
                         'column': cname})
                    continue
                seen_cols.add(cname)
                valid.append(cname)

            valid = valid[:COLUMN_BUDGET]
            if valid:
                result.append((tname, valid))
                seen_tables_in_result.add(tname)
            if len(result) >= TABLE_BUDGET:
                break

        if not result:
            if self._illegal_items:
                return [], 'illegal_elements'
            return [], 'valid_empty'
        return result, 'ok' if not self._illegal_items else 'partial_illegal'

    # -------------------------------------------------------------- engine
    def _retrieve(self, query: str, k: int = 5,
                  qid: str = '') -> List:
        t_start = time.time()
        qid = qid or self.api_qid
        from experiments.bird.api.api_client import get_model
        try:
            model = get_model()
        except MethodBlocked as exc:
            raise
        ev: Dict = {
            'qid': qid, 'fold': self._current_fold, 'question': query,
            'request_model': model,
            'request': {'model': model, 'temperature': 0.1,
                        'max_tokens': STAGE_MAX_TOKENS['schema_selection'],
                        'timeout_s': 60, 'sdk_max_retries': 0,
                        'app_max_retries': 1},
            'candidate_tables': [], 'shown_columns': {}, 'prompt': '',
            'prompt_sha256': '',
            'api': {'attempts': 0, 'retries': 0, 'http_time_s': [],
                    'resp_model': 'not_provided', 'resp_id': 'not_provided',
                    'usage': 'not_provided', 'raw_response': None,
                    'req_id': '', 'cached': None, 'cost_cny': None,
                    'error': None},
            'parse': {'status': None, 'illegal_items': []},
            'final_selection': [], 'status': None,
        }
        self._last_query_evidence = ev

        self._check_fatal()  # raises MethodBlocked if the method is stopped

        self._illegal_items = []
        q_emb = _embed([query])[0]
        scores = self._table_embs @ q_emb
        order = sorted(range(len(scores)),
                       key=lambda i: (-scores[i], self._table_names[i]))
        top_idx = order[:CANDIDATE_K]
        ev['candidate_tables'] = [[self._table_names[i], float(scores[i])]
                                  for i in top_idx]

        schema_text, shown_tables, shown_cols = self._build_schema_text(
            top_idx, q_emb=q_emb)
        ev['shown_columns'] = shown_cols
        prompt = SCHEMA_SELECT_PROMPT.format(
            schema_text=schema_text, question=query,
            table_budget=TABLE_BUDGET, col_budget=COLUMN_BUDGET)
        ev['prompt'] = prompt
        ev['prompt_sha256'] = hashlib.sha256(
            prompt.encode('utf-8')).hexdigest()

        self.api_qid = qid
        try:
            res = call_api(self, 'schema_selection', prompt,
                           STAGE_MAX_TOKENS['schema_selection'],
                           temperature=0.1)
        except MethodBlocked:
            ev['status'] = 'method_blocked'
            ev['wall_time_s'] = round(time.time() - t_start, 3)
            raise
        except ApiTransientError as e:
            ev['api']['error'] = str(e)[:300]
            ev['status'] = 'api_error'
            ev['wall_time_s'] = round(time.time() - t_start, 3)
            raise RuntimeError(
                f'[DenseDeepSeek] LLM API failure for query '
                f'"{query[:50]}...": {e}') from e

        ev['api'].update({
            'attempts': res.attempts,
            'retries': max(0, res.attempts - 1),
            'resp_model': res.response_model or 'not_provided',
            'resp_id': res.response_id or 'not_provided',
            'usage': res.usage or 'not_provided',
            'raw_response': res.content,
            'req_id': res.req_id, 'cached': res.cached,
            'cost_cny': res.cost_cny})
        self.calls = [{'stage': 'schema_selection',
                       'req_id': res.req_id,
                       'response_id': res.response_id,
                       'response_model': res.response_model,
                       'usage': res.usage, 'cached': res.cached,
                       'cost_cny': res.cost_cny,
                       'attempts': res.attempts}]

        selected, parse_status = self._parse_response(
            res.content, set(shown_tables), shown_cols)
        ev['parse'] = {'status': parse_status,
                       'illegal_items': self._illegal_items[:10]}
        ev['status'] = parse_status

        score_map = {self._table_names[i]: float(scores[i])
                     for i in top_idx}
        ev['final_selection'] = [
            {'table': t, 'columns': c, 'table_score': score_map.get(t, 0.0)}
            for t, c in selected[:k]]
        ev['wall_time_s'] = round(time.time() - t_start, 3)

        results = []
        for tname, cols in selected[:k]:
            ts = score_map.get(tname, 0.0)
            columns = [(f"C:{tname}.{c}", ts) for c in cols]
            results.append(RetrievalResult(
                table=tname, table_score=ts, columns=columns,
                metadata={
                    'method': 'DenseDeepSeek',
                    'qid': qid,
                    'prompt_version': 'expansion_v2',
                    'embedding': EMBED_MODEL,
                    'llm': 'deepseek-flash',
                    'llm_resp_model': res.response_model,
                    'api_status': 'ok',
                    'resp_id': res.response_id,
                    'usage': res.usage,
                    'cached': res.cached,
                    'req_id': res.req_id,
                    'parse_status': parse_status,
                    'illegal_items': self._illegal_items[:10],
                    'candidate_tables': len(shown_tables),
                    'shown_cols_per_table': {t: len(c) for t, c in
                                             shown_cols.items()},
                    'parse_errors': self._parse_errors,
                }))
        return results
