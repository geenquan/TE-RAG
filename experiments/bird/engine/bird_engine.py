"""BIRD (English/SQLite, per-database) adaptation of the accepted TE-RAG scoring engine.

Scoring core copied line-for-line from the frozen accepted identity
(research_runs/targeted_ablation_optimization_v2):
- workspace/reference_research/model.py: transform, textnorm, match_strength and
  Engine.channels / bmscores / graph_table / patterns_for / table_features /
  field_graph / predict;
- workspace/reference_research/legacy_engine.py: normalize, physical_table_features
  and Engine.index (base_tf augmentation + BM25 index dict), Engine.raw_fields;
- experiment/replay_transfer_impact.py: annotation_inputs, make_view (verbatim);
- workspace/reference_research/study_common.py: VARIANTS/DEFAULT/switches/
  numeric_identity + digest/require/sha -> adapter/model/study_defaults.py.

Only the data/language/dialect adaptations listed in adapter/model/MODEL_READY.json
deviate; every such line is marked '# BIRD-ADAPT:'. __init__ is replaced per
contract (BIRD construction): per-db CSVs via the Lead's make_bird_config,
workspace graph/pattern builders, B via AnnotationMinerBird (English, sqlite),
metadata index via workspace IndexBuilder with NO extra texts (B/C injection
happens in Engine.index exactly as accepted), C via annotation_transfer_bird
cached per content identity under cache_dir, gold in local (simple-name) view.
"""
import collections
import copy
import hashlib
import json
import math
import resource
import time
import unicodedata
from dataclasses import asdict
from pathlib import Path

import numpy as np

from terag.feature_extractor import FeatureExtractor
from terag.graph_builder import BipartiteGraphBuilder
from terag.index_builder import BM25Retriever, IndexBuilder
from terag.pattern_miner import PatternMiner
from terag.ranker import CombinedRanker
from terag.terag_retriever_v2 import TERAGRetrieverV2
from terag.tokenization import tokenize_text
from eval.full_gold import full_gold_metrics
from eval.retrieval_metrics import normalize_column_key
from datasets.gt_utils import load_json_list

from . import annotation_transfer_bird as transfer
from .annotation_inputs import annotation_inputs
from .annotation_miner_bird import AnnotationMinerBird
from .annotation_transfer_bird import SchemaInfo, TransferConfig
from .study_defaults import DEFAULT, VARIANTS, digest, numeric_identity, require, switches

_MODULE_DIR = Path(__file__).resolve().parent
_RUN_ROOT = Path(__file__).resolve().parents[3]
# Cache identity must track these adaptation sources and the frozen workspace
# engine dependencies (CONTRACTS.md engine contract).
_IDENTITY_SOURCES = [
    'experiments/bird/engine/annotation_miner_bird.py',
    'experiments/bird/engine/annotation_transfer_bird.py',
    'experiments/bird/engine/annotation_inputs.py',
    'experiments/bird/engine/bird_engine.py',
    'experiments/bird/engine/study_defaults.py',
    'terag/tokenization.py',
    'terag/feature_extractor.py',
    'terag/graph_builder.py',
    'terag/pattern_miner.py',
    'terag/index_builder.py',
]


def normalized_table(table):
    """Verbatim from experiment/ablation_reexperiment.py."""
    return str(table).strip('`').split('.')[-1].lower()


def sha(path):
    """Verbatim from study_common.py."""
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text())


def write_json_new(path, value):
    """Write-once JSON (mirrors study_common.write open('x') semantics)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)


def transform(values,mode,clip=1000000.):
    """Transform positive evidence only; absent evidence remains exactly zero."""
    out={k:min(max(0.,float(v)),clip) for k,v in values.items()}
    positive=[v for v in out.values() if v>0]
    if not positive or mode in ['none','raw']:return out
    if mode in ['max','minmax']:
        lo=min(positive) if mode=='minmax' else 0.;hi=max(positive)
        return {k:((v-lo)/(hi-lo) if hi>lo else 1.) if v>0 else 0. for k,v in out.items()}
    if mode=='zscore':
        mean=float(np.mean(positive));sd=float(np.std(positive))
        return {k:max(0.,(v-mean)/sd) if v>0 and sd>1e-12 else (1. if v>0 else 0.) for k,v in out.items()}
    if mode=='rank':
        unique=sorted(set(positive));ranks={v:(i+1)/len(unique) for i,v in enumerate(unique)}
        return {k:ranks[v] if v>0 else 0. for k,v in out.items()}
    raise ValueError('Unknown evidence normalization: '+mode)


def textnorm(s):return unicodedata.normalize('NFKC',str(s)).lower().strip()


def match_strength(term,query,tokens,mode,language='chinese'):  # BIRD-ADAPT: language parameter
    if mode=='exact':return float(term in tokens)
    term=textnorm(term);query=textnorm(query)
    if len(term)<2:return 0.
    if mode=='phrase':return float(term in query)
    if mode=='token_overlap':
        words={w for w in tokenize_text(term,language) if len(w)>1}  # BIRD-ADAPT: language
        overlap=len(words&tokens)/len(words) if words else 0.
        return overlap if overlap>=.5 else 0.
    raise ValueError('Unknown TERM match rule')


def normalize(values):
    total=sum(values)
    require(total>0 and all(x>=0 and math.isfinite(x) for x in values),'Invalid mixture weights')
    return [x/total for x in values]


def physical_table_features(features,physical_nodes):
    """A column Pattern must never consume a table candidate/output slot."""
    return {node:values for node,values in features.items() if node in physical_nodes}


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


class BirdEngine:
    """Accepted TE-RAG scoring engine adapted to one BIRD database (English, SQLite).

    Construction mirrors legacy_engine.Engine.__init__ +
    targeted_ablation_optimization_v2 workspace experiment/ablation_runtime.py
    Job.__init__ (fresh fold-local B/graph/patterns, frozen C, metadata index,
    FeatureExtractor view); scoring methods below are copied line-for-line from
    the reference Engine classes.
    """

    implementation_id = ('bird_v1(base=graph_casefold_ids_v1+pattern_consumer_casefold_v1;'
                         'adaptations=english_tokenization+sqlite_miner_dialect+real_numeric_family)')

    def __init__(self, cfg, db_id, train_rows, test_qids, gold_local, question_text,
                 cache_dir, heldout_tables=None, trace_dir=None):
        started = time.perf_counter()
        timings = {}
        self.cfg = cfg
        self.db_id = str(db_id)
        self.language = getattr(cfg.tokenizer, 'language', 'english')  # BIRD-ADAPT: language from config
        # ---- BIRD inputs (replaces powergrid inputs()/Job split bookkeeping) ----
        missing = {'qid', 'question', 'sql', 'gold_tables', 'gold_fields'} - set(train_rows.columns)
        require(not missing, 'BIRD train_rows missing columns: ' + str(sorted(missing)))
        require(train_rows['qid'].astype(str).is_unique, 'duplicate train qids')
        self.train = train_rows.reset_index(drop=True)
        self.train['qid'] = self.train['qid'].astype(str)
        self.test_qids = [str(q) for q in test_qids]
        self.gold = {str(k): v for k, v in gold_local.items()}
        self.query_text = {str(k): str(v) for k, v in question_text.items()}
        require(set(self.test_qids) <= set(self.query_text), 'question_text missing test qids')
        require(set(self.test_qids) <= set(self.gold), 'gold_local missing test qids')
        self.fold = None  # BIRD-ADAPT: no powergrid fold/setting; record shape keys kept (None)
        self.setting = None  # BIRD-ADAPT
        self.heldout_tables = set(heldout_tables or [])
        self.cache_dir = Path(cache_dir)
        self.trace_dir = Path(trace_dir) if trace_dir is not None else None
        # ---- 1) base retriever, graph, patterns (mirrors Job.__init__) ----
        self.base = TERAGRetrieverV2(self.cfg)
        self.base.train_data = self.train
        mark = time.perf_counter()
        builder = BipartiteGraphBuilder(self.cfg)
        self.base.graph = builder.build(self.train)
        timings['graph_seconds'] = time.perf_counter() - mark
        mark = time.perf_counter()
        self.base.patterns = PatternMiner(self.cfg).mine(
            builder.get_element_to_queries(self.base.graph, self.train))
        timings['patterns_seconds'] = time.perf_counter() - mark
        # ---- 2) B mining (English lexicon subclass; dialect already parameterized) ----
        mark = time.perf_counter()
        self.miner = AnnotationMinerBird(self.base.field_df, self.base.table_df,
                                         dialect='sqlite', language=self.language)
        self.miner.mine(self.train)
        self.annotations = self.miner.annotations
        self.btexts = self.miner.table_annotation_texts(self.cfg.index.field_weights.pattern)
        timings['B_mining_seconds'] = time.perf_counter() - mark
        self._check_heldout_isolation()  # BIRD-ADAPT: H-protocol isolation re-check at fit
        # ---- 3) metadata index (English; NO extra texts: B/C injection happens in engine.index) ----
        self.cfg.feature.field_weights['annotation_hit'] = 1.  # mirrors legacy_engine.__init__ (frozen default)
        mark = time.perf_counter()
        self.metadata = IndexBuilder(self.cfg).build()
        timings['metadata_index_seconds'] = time.perf_counter() - mark
        self.base_tf = collections.defaultdict(dict)
        for word, postings in self.metadata['inverted_index'].items():
            for node, tf, length in postings:
                self.base_tf[node][word] = tf
        # ---- 4) caches (legacy keys + model.py extra keys) ----
        self.caches = {'indices': collections.OrderedDict(), 'tables': collections.OrderedDict(),
                       'channels': {}, 'patterns': {}, 'graphs': {}, 'fields': {},
                       'field_graph': {}, 'name_only': {}, 'bm': collections.OrderedDict(),
                       'table_graph_raw': {}, 'pattern_variants': {}, 'separate_field_graph': {}}
        self.native_transfers = {}
        self.index_signatures = {}
        # ---- 5) C transfer via the BIRD-adapted module, content-addressed under cache_dir ----
        self.schema_info = SchemaInfo(
            {t: g[['field_name', 'field_name_desc', 'field_type']].rename(
                columns={'field_name_desc': 'field_desc'}).to_dict('records')
             for t, g in self.base.field_df.groupby('table')},
            {r['table']: r.get('table_desc', '') for _, r in self.base.table_df.iterrows()})
        mark = time.perf_counter()
        self.default_transfer = self.selected_transfer(DEFAULT)
        timings['C_transfer_seconds'] = time.perf_counter() - mark
        # ---- 6) native extractor view (mirrors legacy_engine.__init__) ----
        texts, terms, _ = annotation_inputs(self.annotations, self.btexts,
                                            self.default_transfer, .5, transfer)  # BIRD-ADAPT: transfer module
        index = self.index(DEFAULT, 'Full')[0]
        view_cfg = copy.deepcopy(self.cfg)
        self.native = make_view(self.base, view_cfg, index, terms, self.default_transfer)
        self.extractor = self.native.feature_extractor
        self.field_groups = {t: g for t, g in self.base.field_df.groupby('table', sort=False)}
        self.frequency = collections.Counter(t for tables, cols
                                             in self.extractor._train_gold_cache.values()
                                             for t in set(tables))
        self.column_degree = collections.Counter()
        for node in self.base.graph.nodes:
            if str(node).startswith('C:'):
                self.column_degree[node] = self.base.graph.degree(node)
        self.controls = {}
        timings['total_fit_seconds'] = time.perf_counter() - started
        self.fit_timings = timings

    # -------------------------------------------------- BIRD construction helpers

    def _check_heldout_isolation(self):
        """H protocol: no training evidence may reference heldout tables (CONTRACTS)."""
        if not self.heldout_tables:
            return
        heldout = set(map(normalized_table, self.heldout_tables))
        for row in self.train.itertuples():
            tables = [normalized_table(t) for t in load_json_list(getattr(row, 'gold_tables', None))]
            leak = sorted(set(tables) & heldout)
            require(not leak, f'train row {row.qid} references heldout tables {leak}')
        for a in self.annotations:
            target = a.get('target')
            if target and normalized_table(str(target).split('.')[0]) in heldout:
                raise RuntimeError('B annotation on held-out target: ' + str(target))
        for node in self.base.graph.nodes:
            name = str(node)
            if name.startswith(('T:', 'C:')):
                table = normalized_table(name[2:].rsplit('.', 1)[0])
                if table in heldout:
                    for nbr in self.base.graph.neighbors(node):
                        require(not str(nbr).startswith('Q_'),
                                'training edge reaches heldout element ' + name)

    def _transfer_identity(self, cfg):
        """Content identity of the C asset (task spec + CONTRACTS engine contract)."""
        pairs = sorted([str(r.qid), str(r.sql)]
                       for r in self.train[['qid', 'sql']].itertuples(index=False))
        sources = {}
        for rel in _IDENTITY_SOURCES:
            path = _RUN_ROOT / rel
            if path.exists():
                sources[rel] = sha(path)
        return {'db_id': self.db_id,
                'train_qid_sql': pairs,
                'field_schema_sha256': sha(self.cfg.data.field_csv),
                'table_schema_sha256': sha(self.cfg.data.table_csv),
                'transfer_config': asdict(cfg),
                'source_sha256': sources}

    def _validate_transfer(self, result):
        """Post-conditions of Job.get_transfer (targeted_ablation_optimization_v2)."""
        train_qids = set(self.train['qid'])
        targets = collections.Counter(r['target_element'].lower() for r in result['accepted'])
        tables = collections.Counter(r['target_element'].rsplit('.', 1)[0].lower()
                                     for r in result['accepted'])
        if max(targets.values(), default=0) > 3 or max(tables.values(), default=0) > 8:
            raise ValueError('C budget violation')
        direct_ids = {a['annotation_id'] for a in self.annotations}
        for r in result['accepted']:
            if not set(r['source_qids']) <= train_qids:
                raise ValueError('C provenance outside train')
            if r['source_element'].lower() == r['target_element'].lower():
                raise ValueError('Self transfer')
            if r['source_annotation_id'] not in direct_ids:
                raise ValueError('C source is not a direct B annotation')
        return result

    def selected_transfer(self, p):
        """Frozen C asset with content-addressed caching (replaces powergrid paths)."""
        key = (p['min_total_sim'], p['source_topk'])
        require(key == (.5, 3), 'Frozen C generation parameters cannot be changed')
        if key not in self.native_transfers:
            cfg = TransferConfig(min_total_sim=key[0], max_sources_per_target=key[1])
            identity = self._transfer_identity(cfg)
            path = self.cache_dir / 'C_cache' / (digest(identity) + '.json')
            if path.exists():
                saved = load_json(path)
                require(saved['identity'] == identity, 'C cache input identity mismatch')
                require(digest(saved['result']) == saved['result_sha256'],
                        'C cache content corruption')
                result = saved['result']
            else:
                result = transfer.transfer_annotations(self.annotations, self.schema_info, cfg)  # BIRD-ADAPT: module
                self._validate_transfer(result)
                write_json_new(path, {'identity': identity, 'result': result,
                                      'result_sha256': digest(result),
                                      'source_test_queries_or_gold_used': False})
            self._validate_transfer(result)
            self.native_transfers[key] = result
        return self.native_transfers[key]

    def index_parity_check(self, p, variant):
        """Port of legacy_engine.native_index_parity (powergrid metadata comparison removed)."""
        cfg = copy.deepcopy(self.cfg)
        texts = self.channels(p, variant)[0]
        expected = IndexBuilder(cfg).build(extra_table_texts=texts)
        actual = self.index(p, variant)[0]
        require(actual['documents'] == expected['documents'],
                'Factorized index document text differs')
        require(actual['N'] == expected['N'], 'Factorized index document count differs')
        for term, postings in expected['inverted_index'].items():
            ours = actual['inverted_index'][term]
            require([x[0] for x in ours] == [x[0] for x in postings],
                    'Posting identity/order differs')
            require(np.allclose([x[1:] for x in ours], [x[1:] for x in postings],
                                rtol=1e-12, atol=1e-12),
                    'Factorized weighted posting differs')
        require(set(actual['inverted_index']) == set(expected['inverted_index']),
                'Factorized vocabulary differs')
        return {'document_texts_and_posting_membership_equal': True,
                'tf_and_norms_equal_tolerance': 1e-12,
                'reason_for_tolerance': 'Floating summation order only; final predictions separately exact-checked.',
                'native_index_sha256': digest(expected), 'factorized_index_sha256': digest(actual)}

    def ledger(self, p, variant):
        """model.py Engine.ledger fields + contract-mandated BIRD additions."""
        index, terms, ledger, key = self.index(p, variant)
        if key not in self.index_signatures:
            self.index_signatures[key] = digest(index)
        heldout = set(map(normalized_table, self.heldout_tables))
        return {**ledger,
                'implementation': self.implementation_id, 'config': p, 'variant': variant,
                'index_key': key, 'index_sha256': self.index_signatures[key],
                'weighted_TERM_sha256': digest(terms), 'table_budget': 5, 'columns_per_table': 10,
                'BHit_semantics': p['b_match'],
                'C_policy': 'Frozen accepted asset; optional stricter post-admission subset only',
                # BIRD-ADAPT: contract ledger additions (no scoring effect)
                'db_id': self.db_id,
                'heldout_target_C': sum(normalized_table(r['target_element'].rsplit('.', 1)[0]) in heldout
                                        for r in self.default_transfer['accepted']),
                'B_texts_by_table': {t: [text for text, _ in v]
                                     for t, v in sorted(self.btexts.items())}}

    def fit_stats(self):
        """Module statistics for the fit (contract: B/C/graph/pattern counts, timing, RSS)."""
        heldout = set(map(normalized_table, self.heldout_tables))
        graph = self.base.graph
        return {'db_id': self.db_id,
                'train_size': int(len(self.train)), 'test_size': len(self.test_qids),
                'B_counts_by_type_status': self.miner.summary(),
                'B_confirmed_TERM': sum(1 for a in self.annotations
                                        if a['type'] == 'TERM' and a.get('status') == 'confirmed'),
                'C_accepted': len(self.default_transfer['accepted']),
                'C_heldout_target': sum(normalized_table(r['target_element'].rsplit('.', 1)[0]) in heldout
                                        for r in self.default_transfer['accepted']),
                'graph_nodes': graph.number_of_nodes(),
                'graph_edges': graph.number_of_edges(),
                'graph_query_nodes': sum(str(n).startswith('Q_') for n in graph.nodes),
                'graph_table_nodes': sum(str(n).startswith('T:') for n in graph.nodes),
                'graph_column_nodes': sum(str(n).startswith('C:') for n in graph.nodes),
                'pattern_elements': len(self.base.patterns),
                'pattern_total': sum(len(v) for v in self.base.patterns.values()),
                'pattern_transferred': sum(sum(1 for q in v if q.transferred)
                                           for v in self.base.patterns.values()),
                'B_texts_by_table': {t: len(v) for t, v in sorted(self.btexts.items())},
                'timing_seconds': dict(self.fit_timings),
                'rss_max_kb': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                'implementation': self.implementation_id}


    # -------------------------------------------------- verbatim from reference: legacy_engine.py Engine.index
    def index(self,p,variant):
        texts,terms,ledger=self.channels(p,variant)
        key=digest({'texts':texts,'k1':p['k1'],'b':p['b']})
        if key not in self.caches['indices']:
            tf={node:dict(values) for node,values in self.base_tf.items()}
            documents=dict(self.metadata['documents'])
            for table,items in texts.items():
                node='T:'+table
                if node not in documents:continue
                for text,weight in items:
                    documents[node]+=' '+str(text)
                    for word in tokenize_text(str(text),self.language):  # BIRD-ADAPT
                        if len(word)>1:tf[node][word]=tf[node].get(word,0.)+weight
            norms={node:math.sqrt(sum(x*x for x in values.values())) for node,values in tf.items()}
            inverted=collections.defaultdict(list)
            # First-seen physical document order matches the unique-doc fix.
            for node in documents:
                for word,value in tf[node].items():inverted[word].append((node,value,norms[node]))
            index={'documents':documents,'inverted_index':dict(inverted),'document_norms':{n:v or 1. for n,v in norms.items()},
                'avgdl':float(np.mean([v or 1. for v in norms.values()])),'N':len(documents),
                'k1':p['k1'],'b':p['b'],'language':self.language}  # BIRD-ADAPT
            self.caches['indices'][key]=index
            if len(self.caches['indices'])>6:self.caches['indices'].popitem(last=False)
        self.caches['indices'].move_to_end(key)
        return self.caches['indices'][key],terms,ledger,key

    # -------------------------------------------------- verbatim from reference: legacy_engine.py Engine.raw_fields
    def raw_fields(self,q,table):
        key=(q,table)
        if key not in self.caches['fields']:
            self.caches['fields'][key]=self.extractor.extract_field_features(self.query_text[q],table)
        return self.caches['fields'][key]

    # -------------------------------------------------- verbatim from reference: model.py Engine.channels
    def channels(self,p,variant):  # BIRD-ADAPT: attribute rebinding only (see MODEL_READY)
        active=switches(p,variant)
        keys=['b_index','c_index','b_text_norm','b_duplicate_penalty','b_source_weight','c_admission_subset','c_confidence','c_evidence_mix']
        ck=digest({'p':{k:p[k] for k in keys},'a':active})
        if ck in self.caches['channels']:return self.caches['channels'][ck]
        result=self.selected_transfer(p) if active['C'] else None
        if result and p['c_admission_subset']>.5:
            result={**result,'accepted':[r for r in result['accepted'] if r['similarity']['total']>=p['c_admission_subset']]}
        canonical,bcterms,ctexts=annotation_inputs(self.annotations,self.btexts,result,.5,transfer)  # BIRD-ADAPT
        bterms=collections.defaultdict(list);support=collections.defaultdict(set)
        for a in self.annotations:  # BIRD-ADAPT
            if a.get('status')=='confirmed' and a.get('target'):
                if a['type']=='TERM':bterms[a['target'].lower()].append(str(a['content']))
                if isinstance(a.get('content'),str):support[str(a['content'])].update(a.get('source_qids',[]))
        texts={};b_index={};c_index={}
        for table,items in canonical.items():
            bset={s for s,w in self.btexts.get(table,[])};seen=set()  # BIRD-ADAPT
            for text,w in items:
                origin='B' if text in bset else 'C'
                if origin=='B' and (not active['BIndex'] or p['b_index']==0):continue
                if origin=='C' and (not active['C'] or p['c_index']==0):continue
                if origin=='B' and p['b_duplicate_penalty']=='dedupe':
                    if textnorm(text) in seen:continue
                    seen.add(textnorm(text))
                weight=p['b_index'] if origin=='B' else p['c_index']
                if origin=='B':
                    length=max(1,len([t for t in tokenize_text(text,self.language) if len(t)>1]))  # BIRD-ADAPT
                    if p['b_text_norm']=='length':weight/=length
                    elif p['b_text_norm']=='sqrt':weight/=math.sqrt(length)
                    if p['b_source_weight']=='support':weight*=min(1.,math.log1p(max(1,len(support[text])))/math.log(4))
                texts.setdefault(table,[]).append((text,weight))
                (b_index if origin=='B' else c_index).setdefault(table,[]).append((text,weight))
        confidence={}
        for r in result['accepted'] if result else []:
            if r['source_type']=='TERM' and isinstance(r['adapted_content'],str):
                sim=r['similarity'];conf=float(sim['total'])
                if p['c_confidence']=='components':
                    components=[float(sim.get('evidence',{}).get(k) or 0.) for k in ['name_j','desc_j','ctx_j']]
                    conf=sum(w*v for w,v in zip(p['c_evidence_mix'],components))
                key=(r['target_element'].lower(),r['adapted_content']);confidence[key]=max(confidence.get(key,0.),conf)
        weighted={}
        for key,items in bcterms.items():
            remain=list(bterms.get(key.lower(),[]));values=[]
            for text in items:
                origin='B' if text in remain else 'C'
                if origin=='B':remain.remove(text)
                if origin=='B' and not active['BHit']:continue
                if origin=='C' and not active['C']:continue
                conf=confidence.get((key.lower(),text),0.)
                scale=conf if p['c_confidence']!='constant' else 1.
                values.append((text,.5 if origin=='B' else .5*scale,origin))
            weighted[key.lower()]=values
        ledger={'B_consumed':len(self.annotations) if active['B'] else 0,  # BIRD-ADAPT
            'C_consumed':len(result['accepted']) if result else 0,'B_index':b_index,'C_index':c_index,
            'text_count':sum(map(len,texts.values())),'term_columns':len(weighted),
            'source_train_only':True,'active':active,'B_priority_before_consumer_masks':True,
            'B_generation_for_C_retained':not active['B'] and active['C'],  # BIRD-ADAPT: contract ledger field
            'generation_config':result['provenance']['config'] if result else None}
        answer=(texts,weighted,ledger);self.caches['channels'][ck]=answer
        return answer

    # -------------------------------------------------- verbatim from reference: model.py Engine.bmscores
    def bmscores(self,q,p,variant):
        index,terms,ledger,key=self.index(p,variant)
        cache=self.caches.setdefault('bm',collections.OrderedDict());ck=(q,key)
        if ck not in cache:
            cache[ck]=dict(BM25Retriever(index).search(self.query_text[q],k=len(self.base.table_df)))  # BIRD-ADAPT
            if len(cache)>200:cache.popitem(last=False)
        cache.move_to_end(ck)
        return cache[ck],terms,ledger,key

    # -------------------------------------------------- verbatim from reference: model.py Engine.graph_table
    def graph_table(self,q,p):
        ck=(q,p['theta_g']);cache=self.caches.setdefault('table_graph_raw',{})
        if ck not in cache:
            tokens=self.extractor._get_query_tokens(self.query_text[q]);scores=collections.defaultdict(float)
            for idx,hist in self.extractor._train_tokens_cache.items():
                union=len(tokens|hist);sim=len(tokens&hist)/union if union else 0.
                if sim>p['theta_g']:
                    for table in self.extractor._train_gold_cache[idx][0]:scores['T:'+table]+=sim*3.
            cache[ck]=dict(scores)
        return {n:v*p['table_graph_gamma']/3/max(1,self.frequency[n[2:]])**p['table_graph_alpha'] for n,v in cache[ck].items()}

    # -------------------------------------------------- verbatim from reference: model.py Engine.patterns_for
    def patterns_for(self,q,p):
        ck=(q,p['pattern_aggregation'],p['pattern_smoothing']);cache=self.caches.setdefault('pattern_variants',{})
        if ck not in cache:
            if p['pattern_aggregation']=='max' and p['pattern_smoothing']==0:
                tokens=self.extractor._get_query_tokens(self.query_text[q])
                cache[ck]=self.extractor._compute_pattern_scores(self.query_text[q],tokens)
            else:
                tokens=self.extractor._get_query_tokens(self.query_text[q]);scores={}
                for node,patterns in self.base.patterns.items():  # BIRD-ADAPT
                    matches=[]
                    for pattern in patterns:
                        pt=set(self.extractor._tokenize(pattern.pattern_text))
                        sim=len(tokens&pt)/len(pt) if pt else 0.
                        support=max(1,pattern.count)
                        matches.append(sim*support/(support+p['pattern_smoothing']))
                    if matches:
                        value=(.6*max(matches) if p['pattern_aggregation']=='max' else .6*sum(sorted(matches,reverse=True)[:3])/min(3,len(matches)))
                        if value>0:scores[node]=value
                cache[ck]=scores
        return cache[ck]

    # -------------------------------------------------- verbatim from reference: model.py Engine.table_features
    def table_features(self,q,p,variant):
        a=switches(p,variant);bm,terms,ledger,key=self.bmscores(q,p,variant)
        graph=self.graph_table(q,p) if a['TableGraph'] else {}
        pattern=self.patterns_for(q,p) if a['Pattern'] else {}
        graph=transform(graph,p['table_graph_norm'],p['table_graph_clip'])
        pattern=transform(pattern,p['table_pattern_norm'])
        nodes=set(bm)|set(graph)|set(pattern)
        values=physical_table_features({n:{'bm25_score':bm.get(n,0.),'graph_score':graph.get(n,0.),'pattern_score':pattern.get(n,0.)} for n in sorted(nodes)},set(self.metadata['documents']))
        if p['table_normalization']=='max':
            for feature in ['bm25_score','graph_score','pattern_score']:
                maximum=max((v[feature] for v in values.values()),default=0.) or 1.
                for v in values.values():v[feature]/=maximum
        return values,terms,ledger,key

    # -------------------------------------------------- verbatim from reference: model.py Engine.field_graph
    def field_graph(self,q,table,field,p):
        ck=(q,table,field,p['field_graph_threshold'],p['field_graph_neighbors'],p['field_edge_power'])
        cache=self.caches.setdefault('separate_field_graph',{})
        if ck not in cache:
            node=f'C:{table}.{str(field).lower()}';votes=[];tokens=self.extractor._get_query_tokens(self.query_text[q])
            if self.base.graph.has_node(node):  # BIRD-ADAPT
                for idx,hist in self.extractor._train_tokens_cache.items():
                    union=len(tokens|hist);sim=len(tokens&hist)/union if union else 0.
                    if sim>p['field_graph_threshold'] and self.base.graph.has_edge('Q_'+str(idx),node):  # BIRD-ADAPT
                        votes.append((sim,idx,self.base.graph['Q_'+str(idx)][node].get('weight',1.)**p['field_edge_power']))  # BIRD-ADAPT
            top=p['field_graph_neighbors']
            if top:votes.sort(key=lambda v:(-v[0],v[1]))
            cache[ck]=sum(sim*w for sim,idx,w in (votes[:top] if top else votes))
        return min(p['field_graph_cap'],cache[ck]*p['field_graph_gamma'])/max(1,self.column_degree[f'C:{table}.{str(field).lower()}'])**p['field_graph_alpha']

    # -------------------------------------------------- verbatim from reference: model.py Engine.predict
    def predict(self,q,p=DEFAULT,variant='Full',trace=False):  # BIRD-ADAPT: Lead call signature defaults
        a=switches(p,variant);started=time.perf_counter()
        features,terms,ledger,index_key=self.table_features(q,p,variant)
        names=['bm25_score','graph_score','pattern_score'];tw=list(p['table_weights'])
        if not a['TableGraph']:tw[1]=0.
        if not a['Pattern']:tw[2]=0.
        scale=sum(tw) or 1.;tw=[w/scale for w in tw]
        scores={n:sum(w*v[k] for k,w in zip(names,tw)) for n,v in features.items()}
        ordered=sorted(scores,key=lambda n:(-scores[n],n));table_time=time.perf_counter()-started
        if trace:
            nob=copy.deepcopy(p);nob['b_index_enabled']=False
            counter=self.bmscores(q,nob,variant)[0]
            augmented=self.bmscores(q,p,variant)[0]
        else:counter={}
        query=self.query_text[q];query_terms=[w for w in self.extractor._tokenize(query) if len(w)>1];tokens=set(query_terms)
        details=[];ft=[];raw_field_signature=[];ann_count=0;c_count=0;start_fields=time.perf_counter()
        for node in ordered[:5]:
            table=node[2:];raw=self.raw_fields(q,table);fields={f:dict(v) for f,v in raw.items()}
            for field,v in fields.items():
                key=f'{table}.{field}'.lower();bh=ch=0.
                for term,weight,origin in terms.get(key,[]):
                    strength=match_strength(term,query,tokens,p['b_match'] if origin=='B' else p['c_match'],self.language)  # BIRD-ADAPT
                    if origin=='B':bh+=weight*strength*p['b_hit_weight']
                    else:ch+=weight*strength*p['c_hit_weight']
                v['BHit']=bh;v['CHit']=ch;v['annotation_hit']=bh+ch
                ann_count+=bh+ch>0;c_count+=ch>0
                v['graph_propagation']=self.field_graph(q,table,field,p) if a['FieldGraph'] else 0.
                if not a['Pattern']:v['role_prior']=0.
                elif p['pattern_aggregation']!='max' or p['pattern_smoothing']!=0:
                    v['role_prior']=self.patterns_for(q,p).get(f'C:{table}.{str(field).lower()}',0.)/.6
                if p['description_scale']!=1.:
                    nk=(q,table,field)
                    if nk not in self.caches['name_only']:self.caches['name_only'][nk]=self.extractor._compute_direct_match_score(field,'',query_terms,tokens)
                    name=self.caches['name_only'][nk];v['direct_match']=name+p['description_scale']*(v['direct_match']-name)
                v['direct_match']*=p['direct_scale']
            fg=transform({f:v['graph_propagation'] for f,v in fields.items()},p['field_graph_norm'])
            pat=transform({f:v['role_prior'] for f,v in fields.items()},p['field_pattern_norm'])
            for f in fields:fields[f]['graph_propagation']=fg[f];fields[f]['role_prior']=pat[f]
            field_names=['direct_match','graph_propagation','role_prior','train_recommend']
            if p['field_normalization']=='max':
                for name in field_names:
                    maximum=max((v[name] for v in fields.values()),default=0.) or 1.
                    for v in fields.values():v[name]/=maximum
            fw=list(p['field_weights'])
            if not a['FieldGraph']:fw[1]=0.
            if not a['Pattern']:fw[2]=0.
            raw_scores={f:sum(w*v[k] for k,w in zip(field_names,fw))+v['BHit']+v['CHit'] for f,v in fields.items()}
            denominator=sum(fw)+(p['b_hit_weight'] if a['BHit'] else 0.)+(p['c_hit_weight'] if a['C'] else 0.)
            # Uniform normalization is diagnostic only; raw scores determine rank.
            field_scores={f:v/(denominator or 1.) for f,v in raw_scores.items()}
            fo=sorted(raw_scores,key=lambda f:(-raw_scores[f],f))
            raw_field_signature.append([table,sorted(raw_scores.items())])
            details.append({'table':table,'table_score':float(scores[node]),'columns':[{'key':normalize_column_key(table+'.'+f),'score':float(field_scores[f])} for f in fo[:10]]})
            if trace:ft.append({'table':table,'features':fields,'raw_total':raw_scores,'normalized_total':field_scores,'ordered_fields':fo,'field_weights':fw,
                'gold_column_ranks':{key:fo.index(key.rsplit('.',1)[1])+1 if key.rsplit('.',1)[1] in fo else None for key in self.gold[q]['gold_columns'] if key.lower().startswith(table.lower()+'.')}})
        record={'qid':q,'fold':self.fold,'setting':self.setting,'variant':variant,'status':'ok',
            'retrieved_tables':[x['table'] for x in details],'retrieved_column_keys':[c['key'] for t in details for c in t['columns']],
            'retrieved_columns_detail':details,'active_TERM_fields':ann_count,'active_C_TERM_fields':c_count}
        require(len(details)<=5 and all(len(t['columns'])<=10 for t in details),'Budget expanded')
        m=full_gold_metrics(record,self.gold);ranks={n:i+1 for i,n in enumerate(ordered)};gt=self.gold[q]['gold_tables']
        metric={'qid':q,'valid':self.gold[q]['gold_status']=='verified_complete','covered@5':float(m[1]),
            **{f'tr@{k}':float(m[0][k]) for k in [1,3,5]},
            **{n:float(m[i]) for i,n in enumerate(['colp','colrec','colf1','fgcs'],2)},
            'mrr':max((1/ranks['T:'+t] for t in gt if 'T:'+t in ranks),default=0.),
            'candidate_recall':sum('T:'+t in ranks for t in gt)/len(gt) if gt else 0.}
        tr={'qid':q,'active':a,'table_ordered_scores':[(n,scores[n]) for n in ordered],
            'all_field_scores_sha256':digest(raw_field_signature),'table_features':features,'fields':ft,
            'B_consumed':ledger['B_consumed'],'C_consumed':ledger['C_consumed'],
            'gold_table_ranks':{t:ranks.get('T:'+t) for t in gt},
            'b_index_marginal_BM25':{n:augmented.get(n,0.)-counter.get(n,0.) for n in ordered} if trace else {},
            'raw_augmented_BM25':augmented if trace else {},
            'BIndex_attribution':'Augmented BM25 minus counterfactual without B injection; nonlinear marginal, not separate additive score.',
            'timing':{'table_features_warm_seconds':table_time,'field_ranking_warm_seconds':time.perf_counter()-start_fields}}
        return record,metric,tr
