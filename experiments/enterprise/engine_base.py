"""Shared construction/index operations used by the current scoring model."""
import collections, copy, math, pickle
import numpy as np
from experiments.enterprise.study_common import *
from experiments.enterprise.runtime import inputs, Job, ASSETS
from experiments.enterprise.annotation_inputs import annotation_inputs, make_view
from retrievers import annotation_transfer as transfer
from terag.index_builder import BM25Retriever, IndexBuilder
from terag.tokenization import tokenize_text

def physical_table_features(features, physical_nodes):
    """A column Pattern must never consume a table candidate/output slot."""
    return {node: values for node, values in features.items() if node in physical_nodes}

class Engine:

    def __init__(self, setting, fold, mode, directory):
        self.meta, self.qa, self.gold = inputs()
        self.setting, self.fold, self.mode, self.directory = (setting, fold, mode, directory)
        if mode != 'outer':
            raise ValueError('Use current fixed outer configurations')
        self.spec = next((s for s in self.meta['splits'] if s['setting'] == setting and s['fold'] == fold))
        self.job = Job(self.meta, self.qa, self.gold, self.spec, directory / 'core')
        self.job.cfg.feature.field_weights['annotation_hit'] = 1.0
        self.caches = {'indices': collections.OrderedDict(), 'tables': collections.OrderedDict(), 'channels': {}, 'patterns': {}, 'graphs': {}, 'fields': {}, 'field_graph': {}, 'name_only': {}}
        self.query_text = dict(zip(self.qa.qid, self.qa.question))
        index_path = ASSETS / 'metadata_index.pkl'
        self.metadata = pickle.loads(index_path.read_bytes())
        require(sha(index_path) == load(ASSETS / 'metadata_index_identity.json')['sha256'], 'Metadata index identity drift')
        self.base_tf = collections.defaultdict(dict)
        for word, postings in self.metadata['inverted_index'].items():
            for node, tf, length in postings:
                self.base_tf[node][word] = tf
        self.native_transfers = {}
        self.index_signatures = {}
        self.default_transfer = self.job.get_transfer({'min_total_sim': 0.5, 'max_sources_per_target': 3})
        texts, terms, _ = annotation_inputs(self.job.annotations, self.job.btexts, self.default_transfer, 0.5, transfer)
        index = self.index(DEFAULT, 'Full')[0]
        cfg = copy.deepcopy(self.job.cfg)
        self.native = make_view(self.job.base, cfg, index, terms, self.default_transfer)
        self.extractor = self.native.feature_extractor
        self.field_groups = {t: g for t, g in self.job.base.field_df.groupby('table', sort=False)}
        self.frequency = collections.Counter((t for tables, cols in self.extractor._train_gold_cache.values() for t in set(tables)))
        self.column_degree = collections.Counter()
        for node in self.job.base.graph.nodes:
            if str(node).startswith('C:'):
                self.column_degree[node] = self.job.base.graph.degree(node)
        self.controls = {}
        self.native_index_identity = None

    def selected_transfer(self, p):
        require((p['min_total_sim'], p['source_topk']) == (0.5, 3), 'Accepted C generation parameters changed')
        return self.job.get_transfer({'min_total_sim': 0.5, 'max_sources_per_target': 3})

    def index(self, p, variant):
        texts, terms, ledger = self.channels(p, variant)
        key = digest({'texts': texts, 'k1': p['k1'], 'b': p['b']})
        if key not in self.caches['indices']:
            tf = {node: dict(values) for node, values in self.base_tf.items()}
            documents = dict(self.metadata['documents'])
            for table, items in texts.items():
                node = 'T:' + table
                if node not in documents:
                    continue
                for text, weight in items:
                    documents[node] += ' ' + str(text)
                    for word in tokenize_text(str(text), 'chinese'):
                        if len(word) > 1:
                            tf[node][word] = tf[node].get(word, 0.0) + weight
            norms = {node: math.sqrt(sum((x * x for x in values.values()))) for node, values in tf.items()}
            inverted = collections.defaultdict(list)
            for node in documents:
                for word, value in tf[node].items():
                    inverted[word].append((node, value, norms[node]))
            index = {'documents': documents, 'inverted_index': dict(inverted), 'document_norms': {n: v or 1.0 for n, v in norms.items()}, 'avgdl': float(np.mean([v or 1.0 for v in norms.values()])), 'N': len(documents), 'k1': p['k1'], 'b': p['b'], 'language': 'chinese'}
            self.caches['indices'][key] = index
            if len(self.caches['indices']) > 6:
                self.caches['indices'].popitem(last=False)
        self.caches['indices'].move_to_end(key)
        return (self.caches['indices'][key], terms, ledger, key)

    def raw_fields(self, q, table):
        key = (q, table)
        if key not in self.caches['fields']:
            self.caches['fields'][key] = self.extractor.extract_field_features(self.query_text[q], table)
        return self.caches['fields'][key]

    def native_index_parity(self):
        cfg = copy.deepcopy(self.job.cfg)
        texts = self.channels(DEFAULT, 'Full')[0]
        expected = IndexBuilder(cfg).build(extra_table_texts=texts)
        actual = self.index(DEFAULT, 'Full')[0]
        require(actual['documents'] == expected['documents'], 'Factorized index document text differs')
        require(actual['N'] == expected['N'], 'Factorized index document count differs')
        for term, postings in expected['inverted_index'].items():
            ours = actual['inverted_index'][term]
            require([x[0] for x in ours] == [x[0] for x in postings], 'Posting identity/order differs')
            require(np.allclose([x[1:] for x in ours], [x[1:] for x in postings], rtol=1e-12, atol=1e-12), 'Factorized weighted posting differs')
        require(set(actual['inverted_index']) == set(expected['inverted_index']), 'Factorized vocabulary differs')
        self.native_index_identity = {'document_texts_and_posting_membership_equal': True, 'tf_and_norms_equal_tolerance': 1e-12, 'reason_for_tolerance': 'Floating summation order only; final predictions separately exact-checked.', 'native_index_sha256': digest(expected), 'factorized_index_sha256': digest(actual)}
        return self.native_index_identity
