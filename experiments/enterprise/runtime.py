"""Fold-local construction using the current data and accepted C assets."""
import copy
import json
from collections import Counter
from dataclasses import asdict, is_dataclass
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from experiments.enterprise.study_common import load, require

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / 'data/enterprise'


def inputs():
    meta = load(ASSETS / 'protocol.json')
    rows = pd.read_csv(ROOT / 'source_dataset/processed_qa_data.csv')
    rows['qid'] = [f'pg{i:03d}' for i in range(len(rows))]
    gold = {g['qid']: g for g in (json.loads(s) for s in
            (ROOT / 'source_dataset/derived/full_gold_v2.jsonl').read_text().splitlines() if s.strip())}
    require(set(rows.qid) == set(gold), 'QA/Gold population mismatch')
    require(all(str(r.sql) == str(gold[r.qid]['sql']) for r in rows.itertuples()), 'QA/Gold SQL mismatch')
    return meta, rows, gold


def configuration(meta, directory):
    from terag.config import TERAGConfig
    cfg = TERAGConfig()

    def update(node, values):
        for key, value in values.items():
            if is_dataclass(getattr(node, key)):
                update(getattr(node, key), value)
            else:
                setattr(node, key, copy.deepcopy(value))

    update(cfg, meta['algorithm_parameters'])
    for section, values in meta['algorithm_parameters'].items():
        require(asdict(getattr(cfg, section)) == values, 'Base parameters differ: '+section)
    cfg.data.field_csv = str(ROOT / 'source_dataset/processed_field_schema.csv')
    cfg.data.table_csv = str(ROOT / 'source_dataset/processed_table_schema.csv')
    cfg.data.qa_csv = str(ROOT / 'source_dataset/processed_qa_data.csv')
    cfg.annotations.use_history = cfg.annotations.use_transfer = cfg.annotations.use_annotation_index = True
    cfg.evaluation.execute_sql = False
    cfg.output.artifacts_dir = str(directory)
    cfg.seed = 42
    return cfg


def subset_train(rows, gold, qids):
    train = rows[rows.qid.isin(qids)].reset_index(drop=True).copy()
    tables, fields = [], []
    for qid in train.qid:
        pairs = [c.split('.', 1) for c in gold[qid]['gold_columns']]
        tables.append(json.dumps(list(dict.fromkeys(t for t, c in pairs)), ensure_ascii=False))
        fields.append(json.dumps(pairs, ensure_ascii=False))
    train['gold_tables'], train['gold_fields'] = tables, fields
    return train


class Job:
    def __init__(self, meta, rows, gold, spec, directory):
        from terag.terag_retriever_v2 import TERAGRetrieverV2
        from terag.graph_builder import BipartiteGraphBuilder
        from terag.pattern_miner import PatternMiner
        from retrievers.annotation_miner import DirectAnnotationMiner
        from experiments.enterprise.identity import validate_b_identity, validate_direct_sources
        self.spec, self.gold = spec, gold
        directory.mkdir(parents=True, exist_ok=False)
        self.cfg = configuration(meta, directory)
        self.train = subset_train(rows, gold, spec['train'])
        self.base = TERAGRetrieverV2(self.cfg)
        self.base.train_data = self.train
        builder = BipartiteGraphBuilder(self.cfg)
        self.base.graph = builder.build(self.train)
        self.base.patterns = PatternMiner(self.cfg).mine(builder.get_element_to_queries(self.base.graph, self.train))
        self.miner = DirectAnnotationMiner(self.base.field_df, self.base.table_df, dialect='mysql')
        self.miner.mine(self.train)
        self.annotations = self.miner.annotations
        self.btexts = self.miner.table_annotation_texts(self.cfg.index.field_weights.pattern)
        accepted = ASSETS / 'accepted' / f"{spec['setting']}_fold{spec['fold']}"
        validate_direct_sources(self.annotations, spec['train'], spec['targets'])
        identity = validate_b_identity(self.annotations, load(accepted/'B.json'), self.btexts, load(accepted/'B_texts.json'))
        (directory/'B_identity.json').write_text(json.dumps(identity, indent=2))
        self.transfer = load(accepted/'C.json')
        self.base._is_fitted = True

    def get_transfer(self, params):
        require((params['min_total_sim'], params['max_sources_per_target']) == (.5, 3), 'Accepted C generation parameters changed')
        result = self.transfer
        targets = Counter(r['target_element'].lower() for r in result['accepted'])
        tables = Counter(r['target_element'].rsplit('.', 1)[0].lower() for r in result['accepted'])
        require(max(targets.values(), default=0) <= 3 and max(tables.values(), default=0) <= 8, 'C budget violation')
        ids = {a['annotation_id'] for a in self.annotations}
        for row in result['accepted']:
            require(set(row['source_qids']) <= set(self.spec['train']), 'C source outside training fold')
            require(row['source_annotation_id'] in ids, 'C source is not direct B')
            require(row['source_element'].lower() != row['target_element'].lower(), 'Self transfer')
        return result
