"""Route-M engine fork: terag_bird_vnext_v1.

Design: VnextEngine(BirdEngine) with param-gated defect fixes. Every gate
defaults OFF; with all gates off the engine delegates each method to the base
implementation, so predictions are bit-identical to BirdEngine (enforced by
tests/test_search_vnext.py parity cases). Gates live in p['vnext'] (dict):

  dedupe_b_terms      D1: dedupe duplicate (target, content) B TERM annotations
                            in hit terms AND index texts
  c_index_provenance  D2: keep C index texts with their own weight even when
                            string-equal to a B text of the table (per-column
                            provenance instead of per-table string membership)
  no_code_expansion   D3: disable CodeMapper query expansion (train-side cache
                            is plain-tokenized; restores symmetric token sets
                            for graph/pattern Jaccards)
  direct_match_fix    D6: token-membership bonus + dedup denominator in
                            _compute_direct_match_score
  english_miner       D4+D5: fit-time vnext B-miner bundle (English stopwords,
                            enriched lexicon, projection-restricted
                            co-occurrence) + its own C asset namespace
  tie_break_degree    D9: secondary tie-break by train frequency (tables) /
                            column degree (fields) before lexical order
  graph_margin        channel-reliability gate: a table's TableGraph score
                            counts only if its best train similarity exceeds
                            theta_g + margin (0.0 = off)

Variant 'adaptive_cols' (D13, SEPARATE — always paired with a fixed 5x10
control in reporting): identical scoring to Full; per-table column list cut
at the first column whose raw score < vnext['adaptive_tau'] * that table's
top raw score (min 1, max 10, tables stay <= 5).

D11 (ledger misreport) is fixed unconditionally in the vnext ledger (reporting
only, no scoring effect): B_consumed counts confirmed annotations, and
B_consumed_texts counts actually-injected B index texts.
"""
import collections
import copy
import math
import time
from pathlib import Path

from .bird_engine import (BirdEngine, annotation_inputs, transform,
                          load_json, write_json_new, sha, textnorm)
from . import annotation_transfer_bird as transfer
from .annotation_transfer_bird import SchemaInfo, TransferConfig
from .study_defaults import DEFAULT, VARIANTS, digest, require, switches
from .annotation_miner_bird_vnext import AnnotationMinerBirdVnext
from .feature_extractor_vnext import FeatureExtractorVnext
from terag.tokenization import tokenize_text

VNEXT_DEFAULT = {
    'dedupe_b_terms': False, 'c_index_provenance': False,
    'no_code_expansion': False, 'direct_match_fix': False,
    'english_miner': False, 'tie_break_degree': False,
    'graph_margin': 0.0, 'adaptive_tau': 0.3, 'adaptive_table_tau': 0.0,
}
VNEXT_FLAG_KEYS = ['dedupe_b_terms', 'c_index_provenance', 'no_code_expansion',
                   'direct_match_fix', 'english_miner', 'tie_break_degree',
                   'graph_margin']
VNEXT_VARIANTS = VARIANTS + ['adaptive_cols']


def vnext_flags(p):
    """Merged vnext gate vector (defaults off; unknown keys rejected)."""
    raw = (p or {}).get('vnext') or {}
    for k in raw:
        require(k in VNEXT_DEFAULT, f'unknown vnext flag: {k}')
    return {**VNEXT_DEFAULT, **raw}


def _any_gate(vn):
    return bool(vn['dedupe_b_terms'] or vn['c_index_provenance'] or
                vn['no_code_expansion'] or vn['direct_match_fix'] or
                vn['english_miner'] or vn['tie_break_degree'] or
                float(vn['graph_margin']) > 0.)


class VnextEngine(BirdEngine):
    """BirdEngine + audit-backed, param-gated route-M fixes."""

    implementation_id = 'terag_bird_vnext_v1'

    _VNEXT_IDENTITY_SOURCES = [
        'engine_vnext/annotation_miner_bird.py',
        'engine_vnext/annotation_miner_bird_vnext.py',
        'engine_vnext/annotation_transfer_bird.py',
        'engine_vnext/annotation_inputs.py',
        'engine_vnext/feature_extractor_vnext.py',
        'engine_vnext/vnext_engine.py',
        'engine_vnext/bird_engine.py',
        'terag/tokenization.py',
        'terag/feature_extractor.py',
        'terag/graph_builder.py',
        'terag/pattern_miner.py',
        'terag/index_builder.py',
    ]

    # ------------------------------------------------------------- fit
    def __init__(self, cfg, db_id, train_rows, test_qids, gold_local, question_text,
                 cache_dir, heldout_tables=None, trace_dir=None):
        super().__init__(cfg, db_id, train_rows, test_qids, gold_local,
                         question_text, cache_dir, heldout_tables, trace_dir)
        # VNEXT: D4+D5 — fit-time vnext miner bundle (predict-time switchable;
        # base bundle stays intact for flag-off parity)
        self.vnext_miner = AnnotationMinerBirdVnext(
            self.base.field_df, self.base.table_df,
            dialect='sqlite', language=self.language)
        self.vnext_miner.mine(self.train)
        self.vnext_annotations = self.vnext_miner.annotations
        self.vnext_btexts = self.vnext_miner.table_annotation_texts(
            self.cfg.index.field_weights.pattern)
        self.vnext_transfers = {}
        self.vnext_default_transfer = self.selected_transfer_vnext(DEFAULT)
        # VNEXT: D6 — vnext extractor (identical construction to the view's;
        # swapped in only when direct_match_fix is on)
        index = self.index(DEFAULT, 'Full')[0]
        view_cfg = copy.deepcopy(self.cfg)
        _, terms, _ = annotation_inputs(self.annotations, self.btexts,
                                        self.default_transfer, .5, transfer)
        self.extractor_vnext = FeatureExtractorVnext(
            view_cfg, self.base.graph, index, self.base.patterns,
            self.base.train_data, self.base.field_df, self.base.table_df)
        self.extractor_vnext.annotation_terms = copy.deepcopy(terms)
        self._extractor_mode = (False, False)  # (direct_match_fix, no_code_expansion)
        self._code_mapper_saved = self.native.feature_extractor.code_mapper
        self._vnext_mapper_saved = self.extractor_vnext.code_mapper

    def _vnext_transfer_identity(self, cfg):
        """VNEXT: C identity namespace for the vnext miner bundle."""
        base = self._transfer_identity(cfg)
        base['vnext_miner'] = self.vnext_miner.miner_id
        base['namespace'] = 'vnext_miner_v1'
        base['source_sha256'] = {
            rel: (sha(Path(__file__).resolve().parents[3] / rel)
                  if (Path(__file__).resolve().parents[3] / rel).exists() else None)
            for rel in self._VNEXT_IDENTITY_SOURCES}
        return base

    def selected_transfer_vnext(self, p):
        """Frozen C asset for the vnext miner bundle (own cache namespace)."""
        key = (p['min_total_sim'], p['source_topk'])
        require(key == (.5, 3), 'Frozen C generation parameters cannot be changed')
        if key not in self.vnext_transfers:
            cfg = TransferConfig(min_total_sim=key[0], max_sources_per_target=key[1])
            identity = self._vnext_transfer_identity(cfg)
            path = self.cache_dir / 'C_cache' / (digest(identity) + '.json')
            if path.exists():
                saved = load_json(path)
                require(saved['identity'] == identity, 'vnext C cache identity mismatch')
                require(digest(saved['result']) == saved['result_sha256'],
                        'vnext C cache content corruption')
                result = saved['result']
            else:
                result = transfer.transfer_annotations(self.vnext_annotations,
                                                       self.schema_info, cfg)
                self._validate_transfer(result, annotations=self.vnext_annotations)
                write_json_new(path, {
                    'identity': identity, 'result': result,
                    'result_sha256': digest(result),
                    'source_test_queries_or_gold_used': False})
            self._validate_transfer(result, annotations=self.vnext_annotations)
            self.vnext_transfers[key] = result
        return self.vnext_transfers[key]

    # ------------------------------------------------------------- channels
    def channels(self, p, variant):
        active = switches(p, 'Full' if variant == 'adaptive_cols' else variant)
        vn = vnext_flags(p)
        gate_here = vn['english_miner'] or vn['dedupe_b_terms'] or vn['c_index_provenance']
        if not gate_here:
            return super().channels(p, variant)
        keys = ['b_index', 'c_index', 'b_text_norm', 'b_duplicate_penalty',
                'b_source_weight', 'c_admission_subset', 'c_confidence', 'c_evidence_mix']
        ck = digest({'p': {k: p[k] for k in keys}, 'a': active,
                     'vn': {k: vn[k] for k in ('dedupe_b_terms', 'c_index_provenance',
                                               'english_miner')}})
        if ck in self.caches['channels']:
            return self.caches['channels'][ck]
        annotations = self.vnext_annotations if vn['english_miner'] else self.annotations
        btexts = self.vnext_btexts if vn['english_miner'] else self.btexts
        result = (self.selected_transfer_vnext(p) if vn['english_miner']
                  else self.selected_transfer(p)) if active['C'] else None
        if result and p['c_admission_subset'] > .5:
            result = {**result, 'accepted': [r for r in result['accepted']
                                             if r['similarity']['total'] >= p['c_admission_subset']]}
        if vn['c_index_provenance']:
            # VNEXT: D2 — origin-tagged merge; C texts kept even when string-equal
            # to a B text of the table (own weight, per-column provenance)
            canonical = collections.OrderedDict()
            for table, items in btexts.items():
                canonical[table] = [(txt, w, 'B') for txt, w in items]
            if result:
                for table, items in transfer.transfer_index_texts(result, .5).items():
                    canonical.setdefault(table, []).extend(
                        (txt, w, 'C') for txt, w in items)
            bcterms = {}
            for a in annotations:
                if a['type'] == 'TERM' and a.get('status') == 'confirmed' and a.get('target'):
                    bcterms.setdefault(a['target'], []).append(str(a['content']))
            if result:
                for key, items in transfer.transfer_column_bonus_map(result).items():
                    values = bcterms.setdefault(key, [])
                    for term in items:
                        if term not in values:
                            values.append(term)
        else:
            canonical_raw, bcterms, _ = annotation_inputs(annotations, btexts,
                                                          result, .5, transfer)
            canonical = {t: [(txt, w, 'B' if txt in {s for s, _ in btexts.get(t, [])} else 'C')
                             for txt, w in items]
                         for t, items in canonical_raw.items()}
        bterms = collections.defaultdict(list)
        support = collections.defaultdict(set)
        for a in annotations:
            if a.get('status') == 'confirmed' and a.get('target'):
                if a['type'] == 'TERM':
                    bterms[a['target'].lower()].append(str(a['content']))  # VNEXT: D1 deduped below
                if isinstance(a.get('content'), str):
                    support[str(a['content'])].update(a.get('source_qids', []))
        texts = {}
        b_index = {}
        c_index = {}
        for table, items in canonical.items():
            bset = {s for s, w in btexts.get(table, [])}
            seen = set()
            for text, w, origin in items:
                if origin == 'B' and (not active['BIndex'] or p['b_index'] == 0):
                    continue
                if origin == 'C' and (not active['C'] or p['c_index'] == 0):
                    continue
                # VNEXT: D1 — always dedupe B index texts by normalized content
                if origin == 'B' and (vn['dedupe_b_terms'] or
                                      p['b_duplicate_penalty'] == 'dedupe'):
                    from .bird_engine import textnorm
                    if textnorm(text) in seen:
                        continue
                    seen.add(textnorm(text))
                weight = p['b_index'] if origin == 'B' else p['c_index']
                if origin == 'B':
                    from terag.tokenization import tokenize_text
                    length = max(1, len([t for t in tokenize_text(text, self.language) if len(t) > 1]))
                    if p['b_text_norm'] == 'length':
                        weight /= length
                    elif p['b_text_norm'] == 'sqrt':
                        weight /= __import__('math').sqrt(length)
                    if p['b_source_weight'] == 'support':
                        weight *= min(1., __import__('math').log1p(max(1, len(support[text]))) / __import__('math').log(4))
                texts.setdefault(table, []).append((text, weight))
                (b_index if origin == 'B' else c_index).setdefault(table, []).append((text, weight))
        confidence = {}
        for r in result['accepted'] if result else []:
            if r['source_type'] == 'TERM' and isinstance(r['adapted_content'], str):
                sim = r['similarity']
                conf = float(sim['total'])
                if p['c_confidence'] == 'components':
                    components = [float(sim.get('evidence', {}).get(k) or 0.)
                                  for k in ['name_j', 'desc_j', 'ctx_j']]
                    conf = sum(w * v for w, v in zip(p['c_evidence_mix'], components))
                key = (r['target_element'].lower(), r['adapted_content'])
                confidence[key] = max(confidence.get(key, 0.), conf)
        weighted = {}
        for key, items in bcterms.items():
            remain = list(bterms.get(key.lower(), []))
            values = []
            seen_pair = set()
            for text in items:
                # VNEXT: D1 — one hit term per (column, content)
                if vn['dedupe_b_terms'] and (key.lower(), text) in seen_pair:
                    continue
                seen_pair.add((key.lower(), text))
                origin = 'B' if text in remain else 'C'
                if origin == 'B':
                    remain.remove(text)
                if origin == 'B' and not active['BHit']:
                    continue
                if origin == 'C' and not active['C']:
                    continue
                conf = confidence.get((key.lower(), text), 0.)
                scale = conf if p['c_confidence'] != 'constant' else 1.
                values.append((text, .5 if origin == 'B' else .5 * scale, origin))
            weighted[key.lower()] = values
        ledger = {'B_consumed': sum(1 for a in annotations if a.get('status') == 'confirmed')
                  if active['B'] else 0,  # VNEXT: D11 — confirmed count, not all
                  'B_consumed_texts': sum(len(v) for v in b_index.values()),
                  'C_consumed': len(result['accepted']) if result else 0,
                  'B_index': b_index, 'C_index': c_index,
                  'text_count': sum(map(len, texts.values())),
                  'term_columns': len(weighted),
                  'source_train_only': True, 'active': active,
                  'B_priority_before_consumer_masks': True,
                  'B_generation_for_C_retained': not active['B'] and active['C'],
                  'generation_config': result['provenance']['config'] if result else None,
                  'vnext_flags': {k: vn[k] for k in VNEXT_FLAG_KEYS}}
        answer = (texts, weighted, ledger)
        self.caches['channels'][ck] = answer
        return answer

    def _validate_transfer(self, result, annotations=None):
        """Base post-conditions; VNEXT: annotations parameter for the vnext
        bundle's C provenance check (base signature unchanged)."""
        if annotations is None:
            return super()._validate_transfer(result)
        train_qids = set(self.train['qid'])
        targets = collections.Counter(r['target_element'].lower() for r in result['accepted'])
        tables = collections.Counter(r['target_element'].rsplit('.', 1)[0].lower()
                                     for r in result['accepted'])
        if max(targets.values(), default=0) > 3 or max(tables.values(), default=0) > 8:
            raise ValueError('C budget violation')
        direct_ids = {a['annotation_id'] for a in annotations}
        for r in result['accepted']:
            if not set(r['source_qids']) <= train_qids:
                raise ValueError('C provenance outside train')
            if r['source_element'].lower() == r['target_element'].lower():
                raise ValueError('Self transfer')
            if r['source_annotation_id'] not in direct_ids:
                raise ValueError('C source is not a direct B annotation')
        return result

    # ------------------------------------------------------------- extractor mode
    def _ensure_extractor_mode(self, d6, d3):
        mode = (bool(d6), bool(d3))
        if mode == self._extractor_mode:
            return
        self._extractor_mode = mode
        target = self.extractor_vnext if d6 else self.native.feature_extractor
        saved = self._vnext_mapper_saved if d6 else self._code_mapper_saved
        # VNEXT: D3 — code mapper disabled in this mode (train-side cache is
        # plain-tokenized; this restores symmetric Jaccard token sets)
        target.code_mapper = None if d3 else saved
        target._query_tokens_cache = {}
        self.extractor = target

    def raw_fields(self, q, table):
        d6, d3 = self._extractor_mode
        key = ((d6, d3), q, table) if (d6 or d3) else (q, table)
        if key not in self.caches['fields']:
            self.caches['fields'][key] = self.extractor.extract_field_features(
                self.query_text[q], table)
        return self.caches['fields'][key]

    # ------------------------------------------------------------- graph gate
    def graph_table(self, q, p):
        vn = vnext_flags(p)
        margin = float(vn['graph_margin'])
        if margin <= 0.:
            return super().graph_table(q, p)
        # VNEXT: channel-reliability gate — table scores count only when the
        # table's best train-question similarity clears theta_g + margin
        ck = (q, p['theta_g'], margin)
        cache = self.caches.setdefault('vnext_table_graph', {})
        if ck not in cache:
            tokens = self.extractor._get_query_tokens(self.query_text[q])
            scores = collections.defaultdict(float)
            best = collections.defaultdict(float)
            for idx, hist in self.extractor._train_tokens_cache.items():
                union = len(tokens | hist)
                sim = len(tokens & hist) / union if union else 0.
                if sim > p['theta_g']:
                    for table in self.extractor._train_gold_cache[idx][0]:
                        scores['T:' + table] += sim * 3.
                        best['T:' + table] = max(best['T:' + table], sim)
            gated = {n: v for n, v in scores.items()
                     if best[n] >= p['theta_g'] + margin}
            cache[ck] = dict(gated)
        return {n: v * p['table_graph_gamma'] / 3 / max(1, self.frequency[n[2:]]) ** p['table_graph_alpha']
                for n, v in cache[ck].items()}

    # ------------------------------------------------------------- predict
    def predict(self, q, p=DEFAULT, variant='Full', trace=False):
        vn = vnext_flags(p)
        if variant == 'adaptive_cols' or _any_gate(vn):
            return self._predict_vnext(q, p, variant, trace, vn)
        self._ensure_extractor_mode(False, False)  # restore base mode for parity
        return super().predict(q, p, variant, trace)

    def _predict_vnext(self, q, p, variant, trace, vn):
        from .bird_engine import match_strength, normalize
        from eval.full_gold import full_gold_metrics
        from eval.retrieval_metrics import normalize_column_key
        require(variant in VNEXT_VARIANTS, 'Unknown vnext variant')
        self._ensure_extractor_mode(vn['direct_match_fix'], vn['no_code_expansion'])
        base_variant = 'Full' if variant == 'adaptive_cols' else variant
        a = switches(p, base_variant)
        started = time.perf_counter()
        features, terms, ledger, index_key = self.table_features(q, p, base_variant)
        names = ['bm25_score', 'graph_score', 'pattern_score']
        tw = list(p['table_weights'])
        if not a['TableGraph']:
            tw[1] = 0.
        if not a['Pattern']:
            tw[2] = 0.
        scale = sum(tw) or 1.
        tw = [w / scale for w in tw]
        scores = {n: sum(w * v[k] for k, w in zip(names, tw)) for n, v in features.items()}
        if vn['tie_break_degree']:
            # VNEXT: D9 — degree/frequency secondary key before lexical order
            ordered = sorted(scores, key=lambda n: (-scores[n],
                                                    -self.frequency.get(n[2:], 0), n))
        else:
            ordered = sorted(scores, key=lambda n: (-scores[n], n))
        table_time = time.perf_counter() - started
        if trace:
            nob = copy.deepcopy(p)
            nob['b_index_enabled'] = False
            counter = self.bmscores(q, nob, base_variant)[0]
            augmented = self.bmscores(q, p, base_variant)[0]
        else:
            counter = {}
        query = self.query_text[q]
        query_terms = [w for w in self.extractor._tokenize(query) if len(w) > 1]
        tokens = set(query_terms)
        details = []
        ft = []
        raw_field_signature = []
        ann_count = 0
        c_count = 0
        start_fields = time.perf_counter()
        tables_kept = ordered[:5]
        if variant == 'adaptive_cols':
            # VNEXT: D13 — optional table-count control (0.0 = keep all five);
            # tables below tau_t * top table score are dropped (min 1, max 5)
            tt = float(vn.get('adaptive_table_tau', 0.))
            if tt > 0. and ordered:
                top_t = scores[ordered[0]]
                tables_kept = [n for n in ordered if scores[n] >= tt * top_t][:5] or ordered[:1]
        for node in tables_kept:
            table = node[2:]
            raw = self.raw_fields(q, table)
            fields = {f: dict(v) for f, v in raw.items()}
            for field, v in fields.items():
                key = f'{table}.{field}'.lower()
                bh = ch = 0.
                for term, weight, origin in terms.get(key, []):
                    strength = match_strength(term, query, tokens,
                                              p['b_match'] if origin == 'B' else p['c_match'],
                                              self.language)
                    if origin == 'B':
                        bh += weight * strength * p['b_hit_weight']
                    else:
                        ch += weight * strength * p['c_hit_weight']
                v['BHit'] = bh
                v['CHit'] = ch
                v['annotation_hit'] = bh + ch
                ann_count += bh + ch > 0
                c_count += ch > 0
                v['graph_propagation'] = self.field_graph(q, table, field, p) if a['FieldGraph'] else 0.
                if not a['Pattern']:
                    v['role_prior'] = 0.
                elif p['pattern_aggregation'] != 'max' or p['pattern_smoothing'] != 0:
                    v['role_prior'] = self.patterns_for(q, p).get(f'C:{table}.{str(field).lower()}', 0.) / .6
                if p['description_scale'] != 1.:
                    nk = (q, table, field)
                    if nk not in self.caches['name_only']:
                        self.caches['name_only'][nk] = self.extractor._compute_direct_match_score(
                            field, '', query_terms, tokens)
                    name = self.caches['name_only'][nk]
                    v['direct_match'] = name + p['description_scale'] * (v['direct_match'] - name)
                v['direct_match'] *= p['direct_scale']
            fg = transform({f: v['graph_propagation'] for f, v in fields.items()}, p['field_graph_norm'])
            pat = transform({f: v['role_prior'] for f, v in fields.items()}, p['field_pattern_norm'])
            for f in fields:
                fields[f]['graph_propagation'] = fg[f]
                fields[f]['role_prior'] = pat[f]
            field_names = ['direct_match', 'graph_propagation', 'role_prior', 'train_recommend']
            if p['field_normalization'] == 'max':
                for name in field_names:
                    maximum = max((v[name] for v in fields.values()), default=0.) or 1.
                    for v in fields.values():
                        v[name] /= maximum
            fw = list(p['field_weights'])
            if not a['FieldGraph']:
                fw[1] = 0.
            if not a['Pattern']:
                fw[2] = 0.
            raw_scores = {f: sum(w * v[k] for k, w in zip(field_names, fw)) + v['BHit'] + v['CHit']
                          for f, v in fields.items()}
            denominator = sum(fw) + (p['b_hit_weight'] if a['BHit'] else 0.) + (p['c_hit_weight'] if a['C'] else 0.)
            field_scores = {f: v / (denominator or 1.) for f, v in raw_scores.items()}
            if vn['tie_break_degree']:
                # VNEXT: D9 — column-degree secondary key
                fo = sorted(raw_scores, key=lambda f: (-raw_scores[f],
                                                       -self.column_degree.get(f'C:{table}.{str(f).lower()}', 0), f))
            else:
                fo = sorted(raw_scores, key=lambda f: (-raw_scores[f], f))
            if variant == 'adaptive_cols':
                # VNEXT: D13 — separate variant; relative-threshold column cut
                # (min 1, max 10; tables stay <=5). ALWAYS reported with the
                # fixed 5x10 control (same params, variant Full).
                tau = float(vn.get('adaptive_tau', .3))
                top = raw_scores[fo[0]] if fo else 0.
                cut = [f for f in fo if raw_scores[f] >= tau * top][:10] or fo[:1]
                columns_out = cut
            else:
                columns_out = fo[:10]
            raw_field_signature.append([table, sorted(raw_scores.items())])
            details.append({'table': table, 'table_score': float(scores[node]),
                            'columns': [{'key': normalize_column_key(table + '.' + f),
                                         'score': float(field_scores[f])} for f in columns_out]})
            if trace:
                ft.append({'table': table, 'features': fields, 'raw_total': raw_scores,
                           'normalized_total': field_scores, 'ordered_fields': fo,
                           'field_weights': fw,
                           'gold_column_ranks': {key: fo.index(key.rsplit('.', 1)[1]) + 1
                                                 if key.rsplit('.', 1)[1] in fo else None
                                                 for key in self.gold[q]['gold_columns']
                                                 if key.lower().startswith(table.lower() + '.')}})
        record = {'qid': q, 'fold': self.fold, 'setting': self.setting, 'variant': variant,
                  'status': 'ok',
                  'retrieved_tables': [x['table'] for x in details],
                  'retrieved_column_keys': [c['key'] for t in details for c in t['columns']],
                  'retrieved_columns_detail': details,
                  'active_TERM_fields': ann_count, 'active_C_TERM_fields': c_count,
                  # VNEXT: D13 — actual output budget disclosure (never hidden)
                  'budget_actual': {'tables': len(details),
                                    'columns_by_table': {t['table']: len(t['columns'])
                                                         for t in details}}}
        require(len(details) <= 5 and all(len(t['columns']) <= 10 for t in details),
                'Budget expanded')
        m = full_gold_metrics(record, self.gold)
        ranks = {n: i + 1 for i, n in enumerate(ordered)}
        gt = self.gold[q]['gold_tables']
        metric = {'qid': q, 'valid': self.gold[q]['gold_status'] == 'verified_complete',
                  'covered@5': float(m[1]),
                  **{f'tr@{k}': float(m[0][k]) for k in [1, 3, 5]},
                  **{n: float(m[i]) for i, n in enumerate(['colp', 'colrec', 'colf1', 'fgcs'], 2)},
                  'mrr': max((1 / ranks['T:' + t] for t in gt if 'T:' + t in ranks), default=0.),
                  'candidate_recall': sum('T:' + t in ranks for t in gt) / len(gt) if gt else 0.}
        tr = {'qid': q, 'active': a,
              'table_ordered_scores': [(n, scores[n]) for n in ordered],
              'all_field_scores_sha256': digest(raw_field_signature),
              'table_features': features, 'fields': ft,
              'B_consumed': ledger['B_consumed'], 'C_consumed': ledger['C_consumed'],
              'gold_table_ranks': {t: ranks.get('T:' + t) for t in gt},
              'b_index_marginal_BM25': {n: augmented.get(n, 0.) - counter.get(n, 0.)
                                        for n in ordered} if trace else {},
              'raw_augmented_BM25': augmented if trace else {},
              'BIndex_attribution': 'Augmented BM25 minus counterfactual without B injection; nonlinear marginal, not separate additive score.',
              'vnext_flags': {k: vn[k] for k in VNEXT_FLAG_KEYS},
              'adaptive_cols_cut': (variant == 'adaptive_cols'),
              'timing': {'table_features_warm_seconds': table_time,
                         'field_ranking_warm_seconds': time.perf_counter() - start_fields}}
        return record, metric, tr

    # ------------------------------------------------------------- ledger
    def ledger(self, p, variant):
        vn = vnext_flags(p)
        base = super().ledger(p, 'Full' if variant == 'adaptive_cols' else variant)
        if vn['english_miner'] or vn['dedupe_b_terms'] or vn['c_index_provenance']:
            forked = self.channels(p, variant)[2]
            base = {**base, 'B_consumed': forked['B_consumed'],
                    'B_consumed_texts': forked['B_consumed_texts'],
                    'C_consumed': forked['C_consumed']}
        # VNEXT: D11 — unconditional reporting fix (no scoring effect)
        base['B_consumed_confirmed'] = sum(
            1 for a in (self.vnext_annotations if vn['english_miner'] else self.annotations)
            if a.get('status') == 'confirmed')
        base['vnext_flags'] = {k: vn[k] for k in VNEXT_FLAG_KEYS}
        base['variant'] = variant
        base['implementation'] = self.implementation_id
        return base

    def fit_stats(self):
        stats = super().fit_stats()
        stats['vnext_miner'] = {
            'miner_id': self.vnext_miner.miner_id,
            'B_counts_by_type_status': self.vnext_miner.summary(),
            'B_texts_by_table': {t: len(v) for t, v in sorted(self.vnext_btexts.items())},
            'C_accepted_vnext': len(self.vnext_default_transfer['accepted']),
        }
        return stats
