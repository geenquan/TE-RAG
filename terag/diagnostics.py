"""Explicit prospective evidence export; disabled unless supplied to fit().

Never serializes the YAML/config object, credentials, or test SQL. Files are
created exclusively so a diagnostic run cannot overwrite historical evidence.
"""
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def content_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     default=str).encode()).hexdigest()


class DiagnosticExport:
    def __init__(self, directory, *, run_id, protocol, fold):
        self.directory = Path(directory)
        self.run_id, self.protocol, self.fold = run_id, protocol, fold
        # A new directory is required, not a global artifacts/resume directory.
        self.directory.mkdir(parents=True, exist_ok=False)

    def write(self, filename, data):
        with (self.directory / filename).open('x') as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)
            f.write('\n')

    def record_fit(self, retriever, train_data, texts_b, texts_c, merged_texts):
        if 'qid' not in train_data or train_data['qid'].duplicated().any():
            raise ValueError('diagnostic fit requires unique training qids')
        training_qids = set(train_data['qid'].astype(str))
        annotations = retriever.miner.annotations if retriever.miner else []
        transfer = retriever.transfer_result or {'accepted': [], 'rejected': [], 'provenance': {}}
        for record in [*annotations, *transfer['accepted'], *transfer['rejected']]:
            if not set(record.get('source_qids', [])) <= training_qids:
                raise ValueError('annotation source outside supplied training fold')
        root = Path(__file__).resolve().parents[1]
        code = ['retrievers/annotation_transfer.py', 'retrievers/annotation_miner.py',
                'terag/diagnostics.py', 'terag/terag_retriever_v2.py',
                'terag/index_builder.py', 'terag/feature_extractor.py', 'terag/ranker.py']
        columns = [c for c in ('qid', 'question', 'sql', 'table', 'field') if c in train_data]
        self.write('fit.json', {
            'identity': 'new diagnostic computation; not recovered historical records',
            'run_id': self.run_id, 'protocol': self.protocol, 'fold': self.fold,
            'training_qids': sorted(training_qids),
            'training_input_sha256': content_digest(train_data[columns].to_dict('records')),
            'schema_sha256': {'fields': digest(retriever.config.data.field_csv),
                              'tables': digest(retriever.config.data.table_csv)},
            'code_sha256': {p: digest(root / p) for p in code},
            'effective_parameters': {
                'table_weights': retriever.ranker.table_ranker.get_weights(),
                'field_weights': retriever.ranker.field_ranker.get_weights(),
                'transfer': transfer['provenance'].get('config'),
                'use_history': retriever.config.annotations.use_history,
                'use_transfer': retriever.config.annotations.use_transfer,
                'use_annotation_index': retriever.config.annotations.use_annotation_index,
                'use_enhanced_index': retriever.config.ablation.use_enhanced_index,
                'load_artifact_weights': bool(retriever.asset_sources)},
            'timing_note': 'export adds overhead; not comparable to original saved timing'})
        self.write('annotations.json', {'direct': annotations, 'transfer': transfer})
        index = retriever.index
        injected = {}
        index_on = (retriever.config.ablation.use_enhanced_index and
                    retriever.config.annotations.use_annotation_index)
        for table, texts in (merged_texts or {}).items():
            doc = 'T:' + table
            injected[table] = {
                'B': texts_b.get(table, []), 'C': texts_c.get(table, []),
                'merged_candidates': texts,
                'actually_injected': texts if index_on and doc in index['documents'] else [],
                'deduplication': 'B-first; identical text once',
                'document_sha256': content_digest(index['documents'].get(doc)),
                'weighted_document_norm': index['document_norms'].get(doc)}
        self.write('index.json', {
            'injected_tables': injected,
            'weighted_postings_sha256': content_digest(index['inverted_index']),
            'column_terms': retriever._annotation_terms,
            'note': 'actual index summary; no reconstruction of historical postings'})

    def record_query(self, qid, trace):
        # Hash file names so external qid strings cannot escape this directory.
        filename = 'query_' + hashlib.sha256(str(qid).encode()).hexdigest() + '.json'
        self.write(filename, {'qid': qid, 'run_id': self.run_id,
                              'protocol': self.protocol, 'fold': self.fold, **trace})
