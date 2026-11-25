import copy
from collections import Counter,defaultdict
from experiments.enterprise.study_common import digest
def normalized_table(t):return str(t).strip("`").split(".")[-1].lower()
def validate_direct_sources(annotations, training_qids, heldout_targets):
    training_qids = set(training_qids)
    heldout_targets = set(map(normalized_table, heldout_targets))
    for annotation in annotations:
        if not set(annotation['source_qids']) <= training_qids:
            raise ValueError('B source outside training partition')
        target = annotation.get('target')
        if target and normalized_table(target.split('.')[0]) in heldout_targets:
            raise ValueError('B annotation on held-out target')

def semantic_record(record):
    result = copy.deepcopy(record)
    result.pop('annotation_id', None)
    # These top-five tied candidates are diagnostic only; rejected/ambiguous
    # records never enter the index, TERM lookup, or transfer source set.
    if result.get('status') != 'confirmed':
        result.pop('candidates', None)
    return digest(result)

def validate_b_identity(fresh, saved, texts, saved_texts):
    if Counter(map(semantic_record, fresh)) != Counter(map(semantic_record, saved)):
        raise ValueError('B semantic/provenance/role content differs from accepted B')
    consumed = lambda records: [a for a in records if a.get('status') == 'confirmed' and
                                a['type'] in ['TERM', 'TEMPLATE', 'PREDICATE']]
    if digest(consumed(fresh)) != digest(consumed(saved)):
        raise ValueError('Ordered consumed B records or transfer-source IDs differ')
    if digest(texts) != digest(saved_texts):
        raise ValueError('Actual B index input differs from accepted B')
    return {'semantic_multiset_equal': True, 'ordered_consumed_records_and_ids_equal': True,
            'B_index_inputs_equal': True, 'whole_diagnostic_bytes_equal': digest(fresh) == digest(saved),
            'ignored_difference_scope': 'ROLE ordinal IDs/order and nonconfirmed diagnostic tie lists only'}
