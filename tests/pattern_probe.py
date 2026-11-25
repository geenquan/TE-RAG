"""Counterexample for the actual custom Pattern consumer inside Engine.predict."""
import copy,json
from types import SimpleNamespace
from unittest.mock import patch
from experiments.enterprise.model import Engine
from experiments.enterprise.study_common import DEFAULT

def probe(aggregation='top3',smoothing=0.):
    e=Engine.__new__(Engine);e.setting='toy';e.fold=0;e.query_text={'q':'query'}
    e.gold={'q':{'gold_tables':['t'],'gold_columns':['t.id'],'gold_status':'verified_complete'}}
    e.extractor=SimpleNamespace(_tokenize=lambda text:['query']);e.caches={'name_only':{}}
    e.table_features=lambda *a:({'T:t':{'bm25_score':1.,'graph_score':0.,'pattern_score':0.}},
                              {},{'B_consumed':0,'C_consumed':0},'index')
    e.raw_fields=lambda *a:{'ID':{'direct_match':0.,'graph_propagation':0.,'role_prior':0.,'train_recommend':0.}}
    e.patterns_for=lambda *a:{'C:t.id':.6}
    e.field_graph=lambda *a:0.
    e.bmscores=lambda *a:({'T:t':1.},{},{},'index')
    p={**copy.deepcopy(DEFAULT),'pattern_aggregation':aggregation,'pattern_smoothing':smoothing}
    with patch('experiments.enterprise.model.full_gold_metrics',return_value=({1:1.,3:1.,5:1.},1.,1.,1.,1.,1.)):
        _,_,trace=e.predict('q',p,'Full',trace=True)
    return trace['fields'][0]['features']['ID']['role_prior']

if __name__=='__main__':
    result={'top3_uppercase_schema_pattern_score':probe(),'smoothed_uppercase_schema_pattern_score':probe('max',1.)}
    print(json.dumps(result),flush=True)
    if any(abs(v-1.)>1e-12 for v in result.values()):raise RuntimeError('Custom Pattern consumer lost physical field identity')
