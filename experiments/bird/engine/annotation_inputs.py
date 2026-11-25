"""Verbatim copy of experiment/replay_transfer_impact.py::annotation_inputs.

Unchanged except that callers pass the BIRD-adapted transfer module
(adapter.model.annotation_transfer_bird) so transfer_index_texts and
transfer_column_bonus_map come from the English/REAL-aware copy.
"""


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
