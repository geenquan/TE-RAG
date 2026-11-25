"""LinkAlign-SingleDB-Adapted (BIRD, brokered): upstream Pipeline
semantic enhancement -> no-repeat chunk retrieval -> irrelevant isolation ->
schema extraction.

Adapted copy of workspace retrievers/linkalign_official_retriever.py. ONLY
these changes vs the reference (documented in API_READY.json):
- the three LLM calls go through adapter.api.api_client.call_api (broker +
  evidence + HTTP cache) on deepseek-flash with thinking disabled (was
  deepseek-chat via the raw client);
- MiniLM embeddings via adapter.api.embed_local (same cached weights,
  identical pooling math); chunk .npy cache under api_evidence/dense_cache;
- OVERLAP-DROPPED SHOTS (frozen rule, api_evidence/OVERLAP_FINDINGS.json):
  SCHEMA_LINKING_FEW_EXAMPLES is replaced by an EMPTY string because both
  embedded shots are verbatim BIRD dev questions (movie_platform). The
  template itself and every other prompt are unchanged.
UNCHANGED: QUERY_REWRITING_TEMPLATE / LOCATE_TEMPLATE /
SCHEMA_LINKING_MANUAL_TEMPLATE text, 24-column chunks, retrieve-20 +
exclude + retrieve-20 mechanics, isolation 32768 / extraction 8192
max_tokens, parse rules (reject all unshown references), valid-empty early
stop when isolation prunes everything, budget projection (max retrieval
score ordering, 5 tables x 10 columns).
"""
from experiments.bird.api.api_client import ApiTransientError, MethodBlocked
from experiments.bird.api.common import (ReviewerBase, SchemaEvidence,
                                parse_schema_links)
from retrievers.base_retriever import RetrieverConfig
from retrievers.upstream.linkalign_prompts import (
    QUERY_REWRITING_TEMPLATE, LOCATE_TEMPLATE,
    SCHEMA_LINKING_MANUAL_TEMPLATE)

# FROZEN overlap handling (see adapter/api/overlap_check.py): both upstream
# shots are verbatim BIRD dev questions -> dropped. No other text changes.
SCHEMA_LINKING_FEW_EXAMPLES = ''


class LinkAlignBird(ReviewerBase):
    """Upstream Pipeline: semantic enhancement -> isolation -> extraction."""
    api_method = 'LinkAlign-SingleDB-Adapted'
    _FATAL_ERRORS = {}

    def __init__(self, field_csv, table_csv, config=None, unit_label='',
                 schema=None, cache_dir=None):
        super().__init__(field_csv, table_csv,
                         config or RetrieverConfig(
                             name='LinkAlign-SingleDB-Adapted'),
                         schema=schema, cache_dir=cache_dir,
                         unit_label=unit_label)

    def fit(self, train_data=None):
        self.schema.build_dense(self.cache_dir)
        self._is_fitted = True

    def _retrieve(self, query, k=5):
        self.calls = []
        self.last_evidence = {'method': self.config.name,
                              'calls': self.calls}
        self._check_fatal()
        first = self.schema.retrieve_chunks(query)
        initial = self.schema.shown_from_chunks(first)
        try:
            rewritten = self.ask('semantic_enhancement',
                                 QUERY_REWRITING_TEMPLATE.format(
                                     question=query,
                                     context=self.schema.ddl(initial)))
            second = self.schema.retrieve_chunks(
                query + '\n' + rewritten,
                exclude=[i for i, _ in first])
            nodes = sorted(first + second, key=lambda x: (-x[1], x[0]))
            shown = self.schema.shown_from_chunks(nodes)
            self.last_evidence.update(
                rewritten_question=rewritten, retrieved_chunks=nodes,
                candidate_columns=shown)
            removed_raw = self.ask('irrelevant_isolation',
                                   LOCATE_TEMPLATE.format(
                                       question=query,
                                       context=self.schema.ddl(shown)))
        except ApiTransientError as e:
            raise RuntimeError(
                f'LinkAlign pipeline API failure: {e}') from e
        removed, invalid_removed = parse_schema_links(removed_raw, shown)
        removed = set(removed)
        retained = {t: [c for c in cols if (t, c) not in removed]
                    for t, cols in shown.items()}
        retained = {t: cols for t, cols in retained.items() if cols}
        if not retained:
            self.last_evidence.update(
                shown_columns={}, removed_columns=sorted(removed),
                native_selection=[], illegal=[], invalid_removed=invalid_removed,
                parse_status='valid_empty',
                early_stop='all_candidates_pruned')
            return []
        raw = self.ask('schema_extraction',
                       SCHEMA_LINKING_MANUAL_TEMPLATE.format(
                           few_examples=SCHEMA_LINKING_FEW_EXAMPLES,
                           context_str='[The Start of Database Schemas]\n' +
                           self.schema.ddl(retained) +
                           '\n[The End of Database Schemas]',
                           question=query))
        selected, illegal = parse_schema_links(raw, retained)
        scores = {}
        for i, score in nodes:
            t = self.schema.chunks[i][0]
            scores[t] = max(scores.get(t, -1.0), score)
        self.last_evidence.update(
            shown_columns=retained, removed_columns=sorted(removed),
            native_selection=selected, illegal=illegal,
            invalid_removed=invalid_removed,
            parse_status=('partial_illegal' if illegal else 'ok')
            if selected else ('illegal_elements' if illegal
                             else 'valid_empty'),
            few_examples='dropped_per_overlap_findings')
        return self.project(selected, scores, k)
