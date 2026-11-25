import copy
def annotation_inputs(annotations, texts_b, transfer, weight, module):
    """Exactly the B/C merge in TERAGRetrieverV2.fit; preserve order and B priority."""
    terms = {}
    for a in annotations:
        if a['type'] == 'TERM' and a.get('status') == 'confirmed' and a.get('target'):
            terms.setdefault(a['target'], []).append(str(a['content']))
    texts_c = module.transfer_index_texts(transfer, weight) if transfer else {}
    merged = {k: list(v) for k, v in texts_b.items()}
    for table, items in texts_c.items():
        values = merged.setdefault(table, [])
        existing = {txt for txt, _ in values}
        for txt, w in items:
            if txt not in existing:
                values.append((txt,w))
                existing.add(txt)
    for key, items in (module.transfer_column_bonus_map(transfer) if transfer else {}).items():
        values = terms.setdefault(key, [])
        for term in items:
            if term not in values:
                values.append(term)
    return merged, terms, texts_c

def make_view(base, config, index, terms, transfer):
    """Reuse only graph/pattern/B; use standard extractor/ranker for all outputs."""
    from terag.feature_extractor import FeatureExtractor
    from terag.ranker import CombinedRanker
    view = copy.copy(base)
    view.config, view.index, view.transfer_result = config, index, transfer
    view._annotation_terms = copy.deepcopy(terms)
    view._diagnostics = None
    view.feature_extractor = FeatureExtractor(config, base.graph, index, base.patterns,
        base.train_data, base.field_df, base.table_df)
    view.feature_extractor.annotation_terms = copy.deepcopy(terms)
    view.ranker = CombinedRanker(config)
    return view
