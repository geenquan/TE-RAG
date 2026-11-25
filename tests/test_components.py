"""Behavior regressions for independent consumers, true scores and safe admission."""
import copy,unittest
from types import SimpleNamespace
from unittest.mock import patch
from experiments.enterprise.study_common import DEFAULT,switches
from experiments.enterprise.model import Engine,transform,match_strength
from experiments.enterprise.parameters import apply,validate,axes,choose

class Switches(unittest.TestCase):
    def test_bindex_only(self):
        a=switches(DEFAULT,'noBIndex');self.assertFalse(a['BIndex']);self.assertTrue(a['BHit']);self.assertTrue(a['C'])
    def test_bhit_only(self):
        a=switches(DEFAULT,'noBHit');self.assertTrue(a['BIndex']);self.assertFalse(a['BHit'])
    def test_b_off(self):
        a=switches(DEFAULT,'noB');self.assertFalse(a['BIndex']);self.assertFalse(a['BHit']);self.assertTrue(a['C'])
    def test_table_graph_only(self):
        a=switches(DEFAULT,'noTableGraph');self.assertFalse(a['TableGraph']);self.assertTrue(a['FieldGraph']);self.assertTrue(a['Pattern'])
    def test_field_graph_only(self):
        a=switches(DEFAULT,'noFieldGraph');self.assertTrue(a['TableGraph']);self.assertFalse(a['FieldGraph'])
    def test_both_graph(self):
        a=switches(DEFAULT,'noGraph');self.assertFalse(a['TableGraph']);self.assertFalse(a['FieldGraph']);self.assertTrue(a['Pattern'])
    def test_independent_config(self):
        p={**DEFAULT,'b_index_enabled':False,'table_graph_enabled':False};a=switches(p,'Full')
        self.assertTrue(a['BHit']);self.assertTrue(a['FieldGraph']);self.assertFalse(a['BIndex']);self.assertFalse(a['TableGraph'])
    def test_no_pattern_preserves_b(self):
        a=switches(DEFAULT,'noPattern');self.assertFalse(a['Pattern']);self.assertTrue(a['BIndex'])
    def test_no_c_preserves_b(self):self.assertFalse(switches(DEFAULT,'noC')['C']);self.assertTrue(switches(DEFAULT,'noC')['BHit'])

class Scoring(unittest.TestCase):
    def test_empty_evidence_stays_zero(self):
        for mode in ['none','minmax','rank','zscore','max']:
            self.assertEqual(transform({'cold':0.,'seen':3.},mode)['cold'],0.)
    def test_unmatched_patterns_never_add_candidates(self):
        e=Engine.__new__(Engine);e.caches={};e.query_text={'q':'query'}
        e.extractor=SimpleNamespace(_get_query_tokens=lambda text:{'query'},_tokenize=lambda text:text.split(),_compute_pattern_scores=lambda q,t:{})
        e.job=SimpleNamespace(base=SimpleNamespace(patterns={'T:unrelated':[SimpleNamespace(pattern_text='unrelated',count=1)]}))
        for mode,smoothing in [('max',0.),('max',1.),('top3',0.),('top3',1.)]:
            self.assertEqual(e.patterns_for('q',{**DEFAULT,'pattern_aggregation':mode,'pattern_smoothing':smoothing}),{})
    def test_positive_pattern_still_admitted(self):
        e=Engine.__new__(Engine);e.caches={};e.query_text={'q':'query'}
        e.extractor=SimpleNamespace(_get_query_tokens=lambda text:{'query'},_tokenize=lambda text:text.split())
        e.job=SimpleNamespace(base=SimpleNamespace(patterns={'T:related':[SimpleNamespace(pattern_text='query',count=1)]}))
        self.assertAlmostEqual(e.patterns_for('q',{**DEFAULT,'pattern_aggregation':'top3','pattern_smoothing':1.})['T:related'],.3)
    def test_float_aliases_are_not_extra_trials(self):
        from experiments.enterprise.study_common import numeric_identity
        self.assertEqual(numeric_identity({'w':.35}),numeric_identity({'w':.35000000000000003}))
        self.assertNotEqual(numeric_identity({'w':.35}),numeric_identity({'w':.351}))
    def test_clipping(self):self.assertEqual(transform({'x':20.},'none',3.)['x'],3.)
    def test_rank_ties(self):self.assertEqual(transform({'a':2.,'b':2.},'rank'),{'a':1.,'b':1.})
    def test_minmax(self):self.assertEqual(transform({'a':1.,'b':2.},'minmax'),{'a':0.,'b':1.})
    def test_phrase_is_new_rule(self):
        self.assertEqual(match_strength('应收电费','统计应收电费',{'应收','电费'},'exact'),0.)
        self.assertEqual(match_strength('应收电费','统计应收电费',{'应收','电费'},'phrase'),1.)
    def test_no_arbitrary_substring_in_exact(self):self.assertEqual(match_strength('电费','电费统计',{'电费统计'},'exact'),0.)
    def test_safe_threshold_not_relaxed(self):
        with self.assertRaises(RuntimeError):validate({**DEFAULT,'c_admission_subset':.4})
    def test_generation_not_changed(self):
        with self.assertRaises(RuntimeError):validate({**DEFAULT,'min_total_sim':.3})
    def test_unknown_key_rejected(self):
        with self.assertRaises(RuntimeError):validate({**DEFAULT,'madeup':1})
    def test_weights_reach_separate_simplex(self):
        p=apply(DEFAULT,'table_graph_weight',.1);self.assertAlmostEqual(p['table_weights'][1],.1)
        self.assertEqual(p['field_weights'],DEFAULT['field_weights'])
        q=apply(DEFAULT,'field_graph_weight',.4);self.assertEqual(q['table_weights'],DEFAULT['table_weights'])
    def test_inactive_axes_excluded(self):
        self.assertNotIn('b_index',axes('noBIndex'));self.assertIn('b_hit_weight',axes('noBIndex'))
        self.assertNotIn('field_graph_norm',axes('noFieldGraph'));self.assertIn('table_graph_norm',axes('noFieldGraph'))

class ConsumerData(unittest.TestCase):
    def make_engine(self):
        e=Engine.__new__(Engine);e.caches={'channels':{}}
        e.job=SimpleNamespace(annotations=[{'type':'TERM','status':'confirmed','target':'t.a','content':'业务术语','source_qids':['train1']}],
            btexts={'t':[('业务术语',.5)]})
        e.selected_transfer=lambda p:{'accepted':[],'provenance':{'config':{}}}
        return e
    def test_index_off_keeps_hit_data(self):
        e=self.make_engine();texts,terms,_=e.channels(DEFAULT,'noBIndex');self.assertFalse(texts);self.assertTrue(terms['t.a'])
    def test_hit_off_keeps_index_data(self):
        e=self.make_engine();texts,terms,_=e.channels(DEFAULT,'noBHit');self.assertTrue(texts);self.assertFalse(terms['t.a'])
    def test_both_off(self):
        e=self.make_engine();texts,terms,_=e.channels(DEFAULT,'noB');self.assertFalse(texts);self.assertFalse(terms['t.a'])
    def test_cache_isolation(self):
        e=self.make_engine();a=e.channels(DEFAULT,'noBIndex');b=e.channels(DEFAULT,'noBHit');self.assertNotEqual(a,b)
    def test_zero_weight_removes_postings(self):
        e=self.make_engine();texts,terms,_=e.channels({**DEFAULT,'b_index':0.},'Full');self.assertFalse(texts);self.assertTrue(terms)
    def test_bhit_switch_keeps_c_identity(self):
        e=self.make_engine();a=e.channels(DEFAULT,'Full')[2];b=e.channels(DEFAULT,'noBHit')[2]
        self.assertEqual(a['C_index'],b['C_index'])
    def test_source_inputs_not_mutated(self):
        e=self.make_engine();before=copy.deepcopy(e.job.annotations);e.channels(DEFAULT,'noB');self.assertEqual(before,e.job.annotations)

class PhysicalFieldIdentity(unittest.TestCase):
    def builder(self,fields=('ID',),parser=False):
        import pandas as pd
        from terag.graph_builder import BipartiteGraphBuilder
        from collections import defaultdict
        b=BipartiteGraphBuilder.__new__(BipartiteGraphBuilder)
        b.config=SimpleNamespace(ablation=SimpleNamespace(use_role_parser=parser,use_role_weight=parser),
            graph=SimpleNamespace(role_weights={'FROM':1.,'SELECT':1.2,'WHERE':2.}))
        b.field_df=pd.DataFrame([{'table':'t','field_name':f,'field_name_desc':'identifier'} for f in fields])
        b.table_df=pd.DataFrame([{'table':'t','table_desc':'table'}]);b.schema_columns={'t':set(fields)}
        b.role_stats=defaultdict(lambda:defaultdict(int));return b
    def train(self,fields):
        import json,pandas as pd
        return pd.DataFrame([{'question':'query','gold_tables':json.dumps(['t']),
            'gold_fields':json.dumps([['t',f] for f in fields]),'sql':'select id from t where ID > 0'}])
    def test_case_only_edge_is_retained(self):
        graph=self.builder().build(self.train(['id']))
        self.assertTrue(graph.has_edge('Q_0','C:t.id'));self.assertFalse(graph.has_node('C:t.ID'))
    def test_case_aliases_keep_max_role_not_sum(self):
        b=self.builder(('ID','id'),True)
        parsed=SimpleNamespace(success=True,field_roles={'t':{'ID':SimpleNamespace(roles={'WHERE'}),'id':SimpleNamespace(roles={'SELECT'})}},
            get_field_weight=lambda *a:1.2,find_field_info=lambda *a:SimpleNamespace(roles={'SELECT'}))
        b.parser=SimpleNamespace(parse=lambda *a,**kw:parsed)
        graph=b.build(self.train(['ID','id']))
        self.assertEqual(graph['Q_0']['C:t.id']['weight'],2.);self.assertEqual(graph.degree('C:t.id'),1)
    def test_absent_physical_field_not_invented(self):
        graph=self.builder().build(self.train(['ghost']))
        self.assertFalse(graph.has_edge('Q_0','C:t.ghost'));self.assertFalse(graph.has_node('C:t.ghost'))
    def test_field_features_share_case_insensitive_identity(self):
        import pandas as pd,networkx as nx
        from terag.config import TERAGConfig
        from terag.feature_extractor import FeatureExtractor
        cfg=TERAGConfig();cfg.ablation.use_graph_weight=True;cfg.ablation.use_template_mining=True
        graph=nx.Graph();graph.add_edge('Q_0','C:t.id',weight=2.)
        index={'documents':{},'inverted_index':{},'document_norms':{},'N':0,'avgdl':1.,'k1':1.5,'b':.75,'language':'chinese'}
        f=FeatureExtractor(cfg,graph,index,{'C:t.id':[SimpleNamespace(pattern_text='query')]},
            self.train(['id']),pd.DataFrame([{'table':'t','field_name':'ID','field_name_desc':'identifier','field_type':'int'}]),
            pd.DataFrame([{'table':'t','table_desc':'table'}]))
        v=f.extract_field_features('query','t')['ID']
        self.assertAlmostEqual(v['graph_propagation'],2.);self.assertAlmostEqual(v['role_prior'],1.)
        self.assertAlmostEqual(v['train_recommend'],1.5)

class CustomPatternIdentity(unittest.TestCase):
    def test_top3_consumer_uses_physical_field_identity(self):
        from tests.pattern_probe import probe
        self.assertAlmostEqual(probe('top3',0.),1.)
    def test_smoothed_consumer_uses_physical_field_identity(self):
        from tests.pattern_probe import probe
        self.assertAlmostEqual(probe('max',1.),1.)

if __name__=='__main__':unittest.main()
