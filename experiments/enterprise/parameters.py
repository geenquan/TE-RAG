"""Prospective equal-budget phased search and inner-only stable selection."""
import copy,random
from experiments.enterprise.study_common import *
TOTAL=144
SINGLE_CAP=96
JOINT_END=128
METRICS=['covered@5','tr@1','tr@3','tr@5','colp','colrec','colf1','fgcs','mrr','candidate_recall']
def axes(variant):
    a=switches(DEFAULT,variant)
    s={'k1':[.75,2.5,1.5],'b':[.25,1.,.75],
       'description_scale':[.5,2.,1.],'direct_scale':[.5,2.,1.],
       'field_normalization':['raw','max'],'table_normalization':['raw','max'],
       'field_weights':[[.70,.10,.05,.15],[.25,.10,.05,.60],[.40,.20,.15,.25]],
       'bm25_weight':[.2,.6,.3,.4,.5]}
    if a['BIndex']:s.update(b_index=[0.,2.,.5,.1,1.,.2,1.5,.3,.75],b_text_norm=['length','sqrt','none'],
        b_duplicate_penalty=['dedupe','none'],b_source_weight=['support','none'])
    if a['BHit']:s.update(b_hit_weight=[0.,3.,.5,.1,1.,.25,1.5,.75,2.],b_match=['phrase','token_overlap','exact'])
    if a['TableGraph']:s.update(table_graph_weight=[0.,.5,.1,.2,.3,.05,.15,.4],
        table_graph_alpha=[1.,.25,.5,.75,0.],table_graph_norm=['rank','minmax','zscore','none'],
        table_graph_clip=[.5,1.,3.,1000000.],table_graph_gamma=[1.,4.,2.,3.],theta_g=[.05,.35,.25,.15])
    if a['FieldGraph']:s.update(field_graph_weight=[0.,.5,.1,.3,.05,.2,.4],
        field_graph_alpha=[1.,.25,.5,.75,0.],field_graph_norm=['rank','minmax','zscore','none'],
        field_graph_neighbors=[1,5,3,10,0],field_graph_threshold=[.05,.35,.25,.15],
        field_graph_gamma=[.5,2.,1.],field_edge_power=[0.,.5,1.],field_graph_cap=[1.,3.,5.,10.])
    if a['Pattern']:s.update(table_pattern_weight=[0.,1.,.1,.5,.2,.75,.3,.4],
        field_pattern_weight=[0.,.5,.1,.3,.05,.2,.4],table_pattern_norm=['rank','max','none'],
        field_pattern_norm=['rank','max','none'],pattern_aggregation=['top3','max'],pattern_smoothing=[1.,3.,0.])
    if a['C']:s.update(c_index=[0.,1.,.25,.75,.5,1.5],c_hit_weight=[0.,2.,.5,.25,1.],
        c_match=['phrase','token_overlap','exact'],c_confidence=['similarity','components','constant'],
        c_admission_subset=[.6,.7,.5],c_evidence_mix=[[.5,.25,.25],[.2,.6,.2],[.3,.4,.3]])
    return s
def validate(p):
    require(set(p)==set(DEFAULT),'Unknown/missing parameter')
    for k in ['table_weights','field_weights','c_evidence_mix']:
        require(all(mathfinite(x) and x>=0 for x in p[k]) and abs(sum(p[k])-1)<1e-9,'Invalid simplex')
    require(p['min_total_sim']==.5 and p['source_topk']==3 and p['c_admission_subset']>=.5,'C safety changed')
    require(p['k1']>0 and 0<=p['b']<=1,'Invalid BM25')
    for k in ['b_index','b_hit_weight','c_index','c_hit_weight','table_graph_gamma','field_graph_gamma','field_graph_cap']:
        require(mathfinite(p[k]) and p[k]>=0,'Invalid nonnegative score parameter')
    for k in ['b_index_enabled','b_hit_enabled','table_graph_enabled','field_graph_enabled']:require(isinstance(p[k],bool),'Invalid switch')
    require(p['field_graph_neighbors']>=0 and isinstance(p['field_graph_neighbors'],int),'Invalid neighbor budget')
    switches(p,'Full')
def mathfinite(x):
    import math
    return isinstance(x,(float,int)) and math.isfinite(x)
def apply(p,key,value):
    p=copy.deepcopy(p)
    special={'bm25_weight':('table_weights',0),'table_graph_weight':('table_weights',1),
       'table_pattern_weight':('table_weights',2),'field_graph_weight':('field_weights',1),
       'field_pattern_weight':('field_weights',2)}
    if key in special:
        name,idx=special[key];weights=p[name];rest=sum(weights)-weights[idx]
        if rest<=1e-12:
            weights=copy.deepcopy(DEFAULT[name]);rest=sum(weights)-weights[idx]
        p[name]=[float(value) if i==idx else float((1-value)*w/rest) for i,w in enumerate(weights)]
    else:p[key]=copy.deepcopy(value)
    validate(p);return p
def single_schedule(variant,base):
    s=axes(variant);out=[{'id':'anchor','axis':'anchor','params':copy.deepcopy(base)}]
    for i in range(max(map(len,s.values()))):
        for k,values in s.items():
            if i<len(values):out.append({'id':k+'_'+str(i),'axis':k,'params':apply(base,k,values[i])})
    return out
def frontier(rows):
    keys=METRICS[:-1];out=[]
    for r in rows:
        x=r['summary']
        if not any(all(t['summary'][k]>=x[k]-1e-12 for k in keys) and any(t['summary'][k]>x[k]+1e-12 for k in keys) for t in rows):out.append(r['config_id'])
    return out
def choose(rows,gold):
    base=rows[0];qids=[r['qid'] for r in base['per_question_metrics'] if r['valid']]
    groups=sorted({gold[q]['original_table'].lower() for q in qids});adequate=len(qids)>=15 and len(groups)>=5
    halves={q:groups.index(gold[q]['original_table'].lower())%2 for q in qids};eligible=[]
    for r in rows[1:]:
        if not adequate:continue
        if not all(r['summary'][k]>=base['summary'][k]-1e-12 for k in ['covered@5','tr@5','colp','colrec']):continue
        if r['summary']['colf1']<=base['summary']['colf1']+1e-12:continue
        b={x['qid']:x for x in base['per_question_metrics']};rr={x['qid']:x for x in r['per_question_metrics']}
        stable=all(sum(rr[q]['colf1']-b[q]['colf1'] for q in qids if halves[q]==h)>=-1e-12 for h in [0,1])
        if stable:eligible.append(r)
    winner=max(eligible,key=lambda r:(r['summary']['colf1'],r['summary']['covered@5'],r['summary']['colrec'],r['summary']['colp'],r['summary']['fgcs'],r['summary']['mrr'],
        -sum(r['params'][k]!=base['params'][k] for k in DEFAULT))) if eligible else base
    return {'chosen':winner['config_id'],'params':winner['params'],'eligible':len(eligible),
        'sample_adequate':adequate,'valid_dev':len(qids),'target_clusters':len(groups),
        'two_target_group_halves_F1_nondecrease_required':True,'anchor_config_id':base['config_id'],'outer_used':False}
def random_candidates(variant,base,rows,seed,count,local=False):
    rng=random.Random(seed);space=axes(variant);seen={numeric_identity(r['params']) for r in rows};out=[]
    ranked=sorted(rows,key=lambda r:(r['summary']['colf1'],r['summary']['covered@5'],r['summary']['mrr']),reverse=True)
    winners={}
    for r in rows:
        if r['axis'] in space:
            if r['axis'] not in winners or r['summary']['colf1']>winners[r['axis']]['summary']['colf1']:winners[r['axis']]=r
    for attempt in range(50000):
        if len(out)>=count:break
        p=copy.deepcopy(ranked[rng.randrange(min(6,len(ranked)))]['params'] if local else base)
        selected=rng.sample(list(space),min(len(space),rng.randint(1,2) if local else rng.randint(2,5)))
        for key in selected:
            value=rng.choice(space[key])
            if not local and key in winners and rng.random()<.6:
                source=winners[key]['params'];special={'bm25_weight':('table_weights',0),'table_graph_weight':('table_weights',1),'table_pattern_weight':('table_weights',2),'field_graph_weight':('field_weights',1),'field_pattern_weight':('field_weights',2)}
                value=source[special[key][0]][special[key][1]] if key in special else source[key]
            p=apply(p,key,value)
        if numeric_identity(p) in seen:continue
        seen.add(numeric_identity(p));out.append({'id':('local_' if local else 'joint_')+str(len(out)),'axis':'local' if local else 'joint','params':p})
    require(len(out)==count,'Equal actual budget could not be completed')
    return out
