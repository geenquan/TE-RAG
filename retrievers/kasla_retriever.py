"""KaSLA adaptation retaining the exact upstream optimization primitives.

Changes: DeepSeek generative binary linker and fold-trained logistic schema-item
classifiers replace fine-tuned LLM and CodeS. No benchmark precomputed scores.
"""
import json

import numpy as np
from sklearn.linear_model import LogisticRegression

from datasets.gt_utils import parse_ground_truth_tables, parse_ground_truth_fields
from retrievers.base_retriever import RetrieverConfig
from retrievers.linkalign_official_retriever import ReviewerRetriever
from retrievers.upstream.kasla_utils import estimation_relevance, estimation_weight, knapsack_dp


def optimize(names, probabilities, binary_names, capacity):
    """Exact upstream objective/weights; deterministic probability ranking."""
    if not names or capacity <= 0:
        return [], {'capacity': int(capacity), 'weights': [], 'relevance': []}
    p = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1.0).tolist()
    values = estimation_relevance(names, p, set(binary_names))
    weights = [estimation_weight(i, values) for i in range(len(names))]
    selected, objective = knapsack_dp(weights, values, int(capacity))
    selected = sorted(selected, key=lambda i: (-values[i], -p[i], names[i]))
    return [names[i] for i in selected], {
        'capacity': int(capacity), 'weights': weights, 'relevance': values,
        'selected_indices': selected, 'objective': objective}


class KaSLAAdaptedRetriever(ReviewerRetriever):
    def __init__(self, field_csv, table_csv, config=None, **kwargs):
        super().__init__(field_csv, table_csv, config or RetrieverConfig(name='KaSLA-Adapted'), **kwargs)
        self.table_model = None
        self.column_model = None
        self.training = []

    def features(self, question):
        ts, cs = self.schema.lexical_features(question)
        # Query-dependent lexical matching, not reference frequency.
        table_x = np.column_stack([ts, ts * ts, self.schema.table_lengths])
        column_x = np.column_stack([cs, cs * cs, ts[self.schema.parents], self.schema.column_lengths])
        return table_x, column_x

    def fit(self, train_data=None):
        if train_data is None or train_data.empty:
            raise ValueError('KaSLA classifiers/capacity estimation require current training fold')
        self.training = []
        tx, ty, cx, cy = [], [], [], []
        table_lookup = {t.lower(): t for t in self.schema.names}
        field_lookup = {(t.lower(), c.lower()): (t, c) for t, c in self.schema.column_keys}
        for _, row in train_data.iterrows():
            question = str(row['question'])
            tables = {table_lookup[t.lower()] for t in parse_ground_truth_tables(row)
                      if t.lower() in table_lookup}
            fields = {field_lookup[t.lower(), c.lower()] for t, c in parse_ground_truth_fields(row)
                      if (t.lower(), c.lower()) in field_lookup}
            t_features, c_features = self.features(question)
            tx.append(t_features)
            ty.extend(int(t in tables) for t in self.schema.names)
            mask = np.isin(self.schema.parents, [self.schema.table_positions[t] for t in tables])
            if mask.any():
                cx.append(c_features[mask])
                cy.extend(int(self.schema.column_keys[i] in fields) for i in np.flatnonzero(mask))
            self.training.append({'qid': str(row['qid']), 'question': question,
                                  'tables': tables, 'fields': fields})
        if len(set(ty)) < 2 or len(set(cy)) < 2:
            raise ValueError('Both classes required for each fold-trained schema classifier')
        self.table_model = LogisticRegression(C=1.0, max_iter=300, random_state=42).fit(np.vstack(tx), ty)
        self.column_model = LogisticRegression(C=1.0, max_iter=300, random_state=42).fit(np.vstack(cx), cy)
        self.scorer_manifest = {
            'training_qids': [r['qid'] for r in self.training],
            'table_pairs': len(ty), 'table_positives': int(sum(ty)),
            'column_pairs': len(cy), 'column_positives': int(sum(cy)),
            'table_features': ['tfidf_cosine', 'squared_cosine', 'log_text_length/10'],
            'column_features': ['tfidf_cosine', 'squared_cosine', 'parent_table_cosine', 'log_description_length/10'],
            'table_coefficients': self.table_model.coef_.tolist(),
            'table_intercept': self.table_model.intercept_.tolist(),
            'column_coefficients': self.column_model.coef_.tolist(),
            'column_intercept': self.column_model.intercept_.tolist()}
        self.train_question_matrix = self.schema.vectorizer.transform([r['question'] for r in self.training])
        # Upstream uses training gold as the binary signal for tolerance estimation.
        for row in self.training:
            tf, cf = self.features(row['question'])
            tp = self.table_model.predict_proba(tf)[:, 1]
            cp = self.column_model.predict_proba(cf)[:, 1]
            row['table_gold_cost'] = self._gold_cost(self.schema.names, tp, row['tables'])
            row['column_gold_costs'] = []
            for t in row['tables']:
                positions = [self.schema.column_positions[t, c] for c in self.schema.columns[t]]
                names = [f'{t}.{c}' for c in self.schema.columns[t]]
                positives = {f'{tt}.{c}' for tt, c in row['fields'] if tt == t}
                if positives:
                    row['column_gold_costs'].append(self._gold_cost(names, cp[positions], positives))
        self._is_fitted = True

    @staticmethod
    def _gold_cost(names, probabilities, positives):
        relevance = estimation_relevance(names, np.clip(probabilities, 1e-6, 1).tolist(), positives)
        return sum(estimation_weight(i, relevance) for i, name in enumerate(names) if name in positives)

    def capacities(self, question):
        similarities = (self.train_question_matrix @ self.schema.vectorizer.transform([question]).T).toarray().ravel()
        order = sorted(range(len(similarities)), key=lambda i: (-similarities[i], self.training[i]['qid']))[:30]
        rows = [self.training[i] for i in order]
        table_cap = max([r['table_gold_cost'] for r in rows] or [0])
        column_cap = max([v for r in rows for v in r['column_gold_costs']] or [0])
        return int(table_cap), int(column_cap), [r['qid'] for r in rows]

    def _retrieve(self, query, k=5):
        self.calls = []
        self.last_evidence = {'method': self.config.name, 'calls': self.calls}
        tf, cf = self.features(query)
        tp = self.table_model.predict_proba(tf)[:, 1]
        cp = self.column_model.predict_proba(cf)[:, 1]
        torder = sorted(range(len(tp)), key=lambda i: (-tp[i], self.schema.names[i]))[:20]
        shown = {}
        for i in torder:
            t = self.schema.names[i]
            cols = self.schema.columns[t]
            shown[t] = sorted(cols, key=lambda c: (-cp[self.schema.column_positions[t, c]], c))[:20]
        # Upstream schema-only generative linker input prefix (prepare_inputs.py).
        prompt = ('[Schema-Linking task]\ndatabase schema:\n' + self.schema.ddl(shown) +
                  '\nQuestion:\n' + query + '\nList the relevant columns in each table:\n' +
                  'Output JSON: {"Relevant columns": {"table_name": ["column_name"]}}. '
                  'Only select columns explicitly present in this schema.')
        raw = self.ask('binary_schema_linker', prompt)
        try:
            clean = raw.strip()
            if clean.startswith('```'):
                clean = clean.split('\n', 1)[1].rsplit('```', 1)[0]
            binary = json.loads(clean)['Relevant columns']
            if not isinstance(binary, dict):
                raise ValueError('Relevant columns must be an object')
        except (ValueError, KeyError, TypeError) as e:
            raise ValueError('Invalid KaSLA binary linker output') from e
        accepted, illegal = {}, []
        for t, cols in binary.items():
            original_t = next((key for key in shown if key.lower() == str(t).lower()), None)
            if original_t is None or not isinstance(cols, list):
                illegal.append(t)
                continue
            available = {c.lower(): c for c in shown[original_t]}
            accepted[original_t] = list(dict.fromkeys(available[str(c).lower()] for c in cols if str(c).lower() in available))
            illegal.extend(f'{t}.{c}' for c in cols if str(c).lower() not in available)
        table_cap, column_cap, neighbors = self.capacities(query)
        native_tables, table_trace = optimize(self.schema.names, tp, accepted, table_cap)
        pairs, column_traces = [], {}
        for t in native_tables:
            names = [f'{t}.{c}' for c in self.schema.columns[t]]
            positions = [self.schema.column_positions[t, c] for c in self.schema.columns[t]]
            selected, trace = optimize(names, cp[positions],
                                       [f'{t}.{c}' for c in accepted.get(t, [])], column_cap)
            pairs.extend(tuple(name.rsplit('.', 1)) for name in selected)
            column_traces[t] = trace
        self.last_evidence.update(candidate_columns=shown, binary_selection=accepted,
                                  illegal=illegal, neighbors=neighbors,
                                  training_qids=[r['qid'] for r in self.training],
                                  table_optimization=table_trace, column_optimization=column_traces,
                                  native_tables=native_tables, native_selection=pairs)
        scores = dict(zip(self.schema.names, tp.tolist()))
        return self.project(pairs, scores, k, selected_tables=native_tables)
