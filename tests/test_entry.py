"""Regression checks for current-run boundaries and mechanism execution."""
import copy
import importlib.util
import tempfile
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('entry',ROOT/'run_experiment.py')
entry=importlib.util.module_from_spec(spec);spec.loader.exec_module(entry)


class EntryTests(unittest.TestCase):
    def test_api_wrapper_preserves_question_identity(self):
        from experiments.bird.baselines.tfidf_bird import run_retriever_records
        class Retriever:
            api_qid=''
            seen=[]
            def retrieve(self,query,k):
                self.seen.append(self.api_qid)
                return []
        retriever=Retriever()
        rows=run_retriever_records(retriever,'method','db',['a','b'],{'a':'one','b':'two'})
        self.assertEqual(retriever.seen,['a','b'])
        self.assertTrue(all(row['status']=='ok' for row in rows))

    def test_budget_refusal_stops_instead_of_becoming_a_scored_error(self):
        from experiments.bird.baselines.tfidf_bird import run_retriever_records
        class BudgetRefusal(Exception):pass
        class Retriever:
            def retrieve(self,*args,**kwargs):raise BudgetRefusal()
        with self.assertRaises(BudgetRefusal):
            run_retriever_records(Retriever(),'method','db',['a'],{'a':'one'},stop_exceptions=(BudgetRefusal,))

    def test_outputs_cannot_overwrite_inputs_or_archive(self):
        for path in [ROOT, ROOT/'data', ROOT/'_archive/new', ROOT/'results']:
            with self.assertRaises(ValueError): entry.output_path(path)

    def test_existing_output_is_refused(self):
        (ROOT/'results').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT/'results') as p:
            with self.assertRaises(ValueError):entry.output_path(p)

    def test_independent_bird_parameters_include_all_eight_ablations(self):
        for protocol in ['question','heldout']:
            methods=entry.bird_choices(protocol,'independent')
            self.assertEqual(len(methods),9)
            self.assertTrue(methods[0]['name'].startswith('optimized_full_'))
            self.assertEqual(len({m['name'] for m in methods}),9)

    def test_declared_field_gate_executes_before_metrics(self):
        from experiments.bird.engine.unified_entry import run_one
        record={'qid':'q','status':'ok','retrieved_tables':['t'],
                'retrieved_column_keys':['t.a','t.b'],
                'retrieved_columns_detail':[{'table':'t','table_score':1.,'columns':[{'key':'t.a','score':2.},{'key':'t.b','score':1.}]}]}
        class Engine:
            def predict(self,*a):return copy.deepcopy(record),{},{}
        params={'field_rerank':{'k_cap':10,'k_low':1,'table_gate':'table_score','table_gate_tau':2.}}
        gold={'gold_tables':['t'],'gold_columns':['t.b']}
        result,metric,evidence=run_one(Engine(),'q',params,'Full',gold)
        self.assertEqual(result['retrieved_column_keys'],['t.a'])
        self.assertEqual(metric['colrec'],0.)
        self.assertTrue(evidence['mechanisms']['field_rerank']['executed'])

    def test_declared_h_gate_is_routed(self):
        from experiments.bird.engine.unified_entry import run_one
        from unittest.mock import Mock
        engine=Mock()
        rec={'qid':'q','status':'ok','retrieved_tables':[],'retrieved_column_keys':[]}
        gate=Mock(return_value=(rec,{},{}))
        _,_,evidence=run_one(engine,'q',{'hg':{'mode':'linear'}},'Full',{'gold_tables':['t'],'gold_columns':['t.a']},predict_gated=gate)
        engine.predict.assert_not_called();gate.assert_called_once()
        self.assertTrue(evidence['mechanisms']['hg']['executed'])
