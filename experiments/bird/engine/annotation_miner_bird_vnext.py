"""Vnext English B-miner (route M): audit-backed D4/D5 fixes only.

D4 (registry/DEFECT_LIST.md D4, severity H for B precision): base STOP_TERMS
are 12 Chinese words; English function/temporal words slip through as unknown
co-occurrence terms and confirm against arbitrary columns; the global lexicon
adds only raw snake_case field names (parts never match query tokens) and
table DESC tokens but not table NAMES.

D5 (severity M): co-occurrence credits a term against EVERY referenced column
of the question (join/WHERE partners included), systematically diluting
concentration for good terms on multi-column English questions.

Fixes (each marked '# VNEXT: D4'/'# VNEXT: D5' at the changed lines):
- frozen English stopword list applied wherever STOP_TERMS was checked;
- global lexicon enriched with table-name tokens and field-name token parts;
- co-occurrence reference columns restricted to the parser's clean
  projection_fields when non-empty (fallback: all referenced fields).
"""
from collections import defaultdict

import pandas as pd

from terag.sql_role_parser import SQLRoleParser, SQLGLOT_AVAILABLE
from terag.tokenization import tokenize_text

from .annotation_miner_bird import AnnotationMinerBird
from retrievers.annotation_miner import (_NUM_RE, _norm_key, TEMPLATE_TOPK,
                                         TERM_MIN_SUPPORT, TERM_CONCENTRATION,
                                         AMBIGUITY_WINDOW,
                                         STOP_TERMS as _BASE_STOP_TERMS)

try:
    import sqlglot
    from sqlglot import exp
except ImportError:  # pragma: no cover
    sqlglot = None

# VNEXT: D4 — frozen English function/temporal word list (audit-reproduced
# rejects: five/how/many/this/from/last/year/week/month/located/still + the
# standard English function-word classes). List is frozen; no per-db fitting.
ENGLISH_STOP_TERMS = frozenset({
    'about', 'above', 'after', 'again', 'all', 'also', 'am', 'an', 'and',
    'any', 'are', 'as', 'at', 'be', 'because', 'been', 'before', 'being',
    'below', 'between', 'both', 'but', 'by', 'can', 'did', 'do', 'does',
    'doing', 'down', 'during', 'each', 'few', 'for', 'from', 'further',
    'had', 'has', 'have', 'having', 'he', 'her', 'here', 'hers', 'him',
    'his', 'how', 'i', 'if', 'in', 'into', 'is', 'it', 'its', 'just', 'me',
    'more', 'most', 'my', 'no', 'nor', 'not', 'now', 'of', 'off', 'on',
    'once', 'only', 'or', 'other', 'our', 'out', 'over', 'own', 'same',
    'she', 'should', 'so', 'some', 'such', 'than', 'that', 'the', 'their',
    'theirs', 'them', 'then', 'there', 'these', 'they', 'this', 'those',
    'through', 'to', 'too', 'under', 'until', 'up', 'very', 'was', 'we',
    'were', 'what', 'when', 'where', 'which', 'while', 'who', 'whom', 'why',
    'will', 'with', 'you', 'your', 'yours',
    # temporal/quantifier words seen confirmed/rejected in the audit fixture
    'five', 'many', 'much', 'last', 'first', 'next', 'year', 'years',
    'week', 'weeks', 'month', 'months', 'day', 'days', 'date', 'located',
    'still', 'per', 'each', 'every', 'whose', 'order', 'sort', 'list',
    'show', 'find', 'give', 'name', 'names', 'number', 'count', 'total',
    'average', 'sum', 'max', 'min', 'top', 'bottom', 'highest', 'lowest',
    'largest', 'smallest', 'band', 'bands', 'tier', 'tiers',
})


class AnnotationMinerBirdVnext(AnnotationMinerBird):
    """AnnotationMinerBird with audit-backed English fixes (D4/D5)."""

    miner_id = 'annotation_miner_bird_vnext_v1(D4+D5)'

    def _is_stop(self, word: str) -> bool:
        """VNEXT: D4 — Chinese-era STOP_TERMS test extended to English stops."""
        return word in ENGLISH_STOP_TERMS or word in _BASE_STOP_TERMS

    def _global_lexicon(self) -> set:
        """VNEXT: D4 — lexicon enriched with table-name tokens and field-name
        token parts so genuine schema words are never treated as unknown terms."""
        words = super()._global_lexicon()
        for _, r in self.table_df.iterrows():
            name = str(r['table'])
            words.add(name.lower())
            words |= {w for w in tokenize_text(name, self.language) if len(w) > 1}
        for _, r in self.field_df.iterrows():
            for w in tokenize_text(str(r['field_name']), self.language):
                if len(w) > 1:
                    words.add(w)
        return words

    def _alias_terms(self, sql):
        """Base R1 with the English stop test (VNEXT: D4)."""
        out = []
        tree, _ = self._parse_sql(sql)
        if tree is None:
            return out
        for select in tree.find_all(exp.Select):
            tables = self._scope_sources(select)
            n_phys = len({t for t in tables.values()})
            for e in select.expressions:
                alias = e.alias
                if not alias:
                    continue
                cols = [c for c in e.find_all(exp.Column)
                        if not isinstance(c.this, exp.Star)]
                if len(cols) != 1:
                    continue
                t = self._resolve(cols[0], tables, n_phys)
                if t is None:
                    continue
                local = self._local_lexicon(t, cols[0].name)
                for w in tokenize_text(alias, self.language):
                    if len(w) > 1 and not self._is_stop(w) \
                            and not _NUM_RE.search(w) and w not in local:  # VNEXT: D4
                        out.append((w, (t, cols[0].name), alias))
        return out

    def mine(self, train_df: pd.DataFrame) -> 'AnnotationMinerBirdVnext':
        """Base mine() with D4 stop test + D5 projection-restricted co-occurrence."""
        assert SQLGLOT_AVAILABLE and sqlglot is not None, 'vnext miner needs sqlglot (AST)'
        parser = SQLRoleParser(dialect=self.dialect)
        glob_lex = self._global_lexicon()

        alias_pairs = defaultdict(lambda: defaultdict(set))
        alias_meta = defaultdict(lambda: defaultdict(set))
        cooc = defaultdict(lambda: defaultdict(set))
        templates = defaultdict(lambda: defaultdict(set))
        tmpl_bindings = defaultdict(lambda: defaultdict(list))
        pred_counts = defaultdict(set)
        role_sets = defaultdict(set)
        role_qids = defaultdict(set)
        entity_vals = defaultdict(set)

        for _, row in train_df.iterrows():
            qid = str(row.get('qid') or row.name)
            sql = str(row.get('sql', ''))
            question = str(row.get('question', ''))
            pr = parser.parse(sql, schema_columns=self.schema_columns)
            if not pr.ast_parsed:
                continue
            tables = pr.get_gold_tables()
            for t, c in pr.get_all_fields():
                for r_ in pr.field_roles[t][c].roles:
                    role_sets[(t, c)].add(r_)
                role_qids[(t, c)].add(qid)
            for term, target, alias_text in self._alias_terms(sql):
                alias_pairs[target][term].add(qid)
                alias_meta[target][term].add(alias_text)

            q_tokens = [w for w in tokenize_text(question, self.language) if len(w) > 1]
            unknown_terms = [w for w in q_tokens
                             if w not in glob_lex and not self._is_stop(w)  # VNEXT: D4
                             and not _NUM_RE.search(w)]
            # VNEXT: D5 — credit a term only against projection columns when the
            # clean projection is non-empty (fallback: every referenced column,
            # preserving base behavior for SELECT * style statements)
            ref_cols = list(pr.projection_fields) or [k for k in pr.get_all_fields()]
            for term in set(unknown_terms):
                for key in ref_cols:
                    cooc[term][_norm_key(*key)].add(qid)

            for (t, c), op, val, vclass in self._predicates(sql):
                pred_counts[(t, c, op, vclass, val)].add(qid)
                if vclass == 'ENTITY':
                    entity_vals[val].add(qid)

            norm_q = self._normalize_template(question)
            for t in tables:
                templates[t][norm_q].add(qid)
                binds = sorted({(_norm_key(tt, cc), rr)
                                for (tt, cc) in pr.get_all_fields()
                                for rr in pr.field_roles[tt][cc].roles})
                tmpl_bindings[t][norm_q].append({'qid': qid, 'bindings': binds})

        self._entity_values = sorted(entity_vals.keys())
        templates2 = defaultdict(lambda: defaultdict(set))
        parser2 = SQLRoleParser(dialect=self.dialect)
        for _, row in train_df.iterrows():
            qid = str(row.get('qid') or row.name)
            pr2 = parser2.parse(str(row.get('sql', '')),
                                schema_columns=self.schema_columns)
            if not pr2.ast_parsed:
                continue
            for t in pr2.get_gold_tables():
                templates2[t][self._normalize_template(str(row.get('question', '')))].add(qid)

        self.annotations = []
        self._annotation_ids = {}
        for (t, c), per_term in alias_pairs.items():
            for term, qids in per_term.items():
                local = self._local_lexicon(t, c)
                lex_overlap = term in local
                self._append({
                    'type': 'TERM', 'rule': 'select_alias',
                    'target': _norm_key(t, c), 'content': term,
                    'alias_texts': sorted(alias_meta[(t, c)][term]),
                    'source_qids': sorted(qids), 'support': len(qids),
                    'confidence': f'heuristic:support={len(qids)}',
                    'status': 'confirmed',
                    'lexicon_overlap': lex_overlap})
        for term, per_elem in cooc.items():
            counts = {k: len(v) for k, v in per_elem.items()}
            total = sum(counts.values())
            best = max(counts.values())
            top = [k for k, n in counts.items() if n >= best * (1 - AMBIGUITY_WINDOW)]
            if best < TERM_MIN_SUPPORT:
                status = 'rejected_single'
            elif best / total >= TERM_CONCENTRATION and len(top) == 1:
                status = 'confirmed'
            else:
                status = 'ambiguous'
            qids = sorted(set().union(*per_elem.values()))
            self._append({
                'type': 'TERM', 'rule': 'co_occurrence',
                'target': top[0] if len(top) == 1 and status == 'confirmed' else None,
                'candidates': sorted(counts.items(), key=lambda x: -x[1])[:5],
                'content': term, 'source_qids': qids, 'support': total,
                'confidence': f'heuristic:concentration={round(best / total, 3)}',
                'status': status})
        for (t, c, op, vclass, val), qids in pred_counts.items():
            self._append({
                'type': 'PREDICATE', 'rule': 'where_ast',
                'target': _norm_key(t, c), 'content': {'operator': op,
                                                       'value_class': vclass,
                                                       'value': val},
                'source_qids': sorted(qids), 'support': len(qids),
                'confidence': 'heuristic:ast_exact', 'status': 'confirmed'})
        for (t, c), roles in role_sets.items():
            self._append({
                'type': 'ROLE', 'rule': 'ast_roles',
                'target': _norm_key(t, c), 'content': sorted(roles),
                'source_qids': sorted(role_qids[(t, c)]),
                'support': len(role_qids[(t, c)]),
                'confidence': 'heuristic:ast_exact', 'status': 'confirmed'})
        for t, per_tmpl in templates2.items():
            ranked = sorted(per_tmpl.items(), key=lambda x: (-len(x[1]), x[0]))
            for text, qids in ranked[:TEMPLATE_TOPK]:
                binds = []
                for rec in tmpl_bindings[t].get(text, []):
                    binds.append(rec['bindings'])
                self._append({
                    'type': 'TEMPLATE', 'rule': 'slot_normalization',
                    'target': t, 'content': text,
                    'bindings': binds,
                    'source_qids': sorted(qids), 'support': len(qids),
                    'confidence': f'heuristic:support={len(qids)}',
                    'status': 'confirmed'})

        self._term_by_key = defaultdict(list)
        for a in self.annotations:
            if a['type'] == 'TERM' and a['status'] == 'confirmed' and a.get('target'):
                self._term_by_key[a['target']].append((a['content'], a['confidence']))
        self._templates_by_table = {
            t: [(text, len(qids)) for text, qids in
                 sorted(per_tmpl.items(), key=lambda x: (-len(x[1]), x[0]))[:TEMPLATE_TOPK]]
            for t, per_tmpl in templates2.items()}
        return self
