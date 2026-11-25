"""Separated consumers and bounded graph/pattern/annotation scoring interventions."""
import collections,copy,math,re,time,unicodedata
import numpy as np
from experiments.enterprise.engine_base import Engine as LegacyEngine,physical_table_features
from experiments.enterprise.study_common import *
from experiments.enterprise.annotation_inputs import annotation_inputs
from retrievers import annotation_transfer as transfer
from terag.tokenization import tokenize_text
from terag.index_builder import BM25Retriever
from eval.full_gold import full_gold_metrics
from eval.retrieval_metrics import normalize_column_key

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
def match_strength(term,query,tokens,mode):
    if mode=='exact':return float(term in tokens)
    term=textnorm(term);query=textnorm(query)
    if len(term)<2:return 0.
    if mode=='phrase':return float(term in query)
    if mode=='token_overlap':
        words={w for w in tokenize_text(term,'chinese') if len(w)>1}
        overlap=len(words&tokens)/len(words) if words else 0.
        return overlap if overlap>=.5 else 0.
    raise ValueError('Unknown TERM match rule')

class Engine(LegacyEngine):
    implementation_id="graph_casefold_ids_v1+pattern_consumer_casefold_v1"
    def channels(self,p,variant):
        active=switches(p,variant)
        keys=['b_index','c_index','b_text_norm','b_duplicate_penalty','b_source_weight','c_admission_subset','c_confidence','c_evidence_mix']
        ck=digest({'p':{k:p[k] for k in keys},'a':active})
        if ck in self.caches['channels']:return self.caches['channels'][ck]
        result=self.selected_transfer(p) if active['C'] else None
        if result and p['c_admission_subset']>.5:
            result={**result,'accepted':[r for r in result['accepted'] if r['similarity']['total']>=p['c_admission_subset']]}
        canonical,bcterms,ctexts=annotation_inputs(self.job.annotations,self.job.btexts,result,.5,transfer)
        bterms=collections.defaultdict(list);support=collections.defaultdict(set)
        for a in self.job.annotations:
            if a.get('status')=='confirmed' and a.get('target'):
                if a['type']=='TERM':bterms[a['target'].lower()].append(str(a['content']))
                if isinstance(a.get('content'),str):support[str(a['content'])].update(a.get('source_qids',[]))
        texts={};b_index={};c_index={}
        for table,items in canonical.items():
            bset={s for s,w in self.job.btexts.get(table,[])};seen=set()
            for text,w in items:
                origin='B' if text in bset else 'C'
                if origin=='B' and (not active['BIndex'] or p['b_index']==0):continue
                if origin=='C' and (not active['C'] or p['c_index']==0):continue
                if origin=='B' and p['b_duplicate_penalty']=='dedupe':
                    if textnorm(text) in seen:continue
                    seen.add(textnorm(text))
                weight=p['b_index'] if origin=='B' else p['c_index']
                if origin=='B':
                    length=max(1,len([t for t in tokenize_text(text,'chinese') if len(t)>1]))
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
        ledger={'B_consumed':len(self.job.annotations) if active['B'] else 0,
            'C_consumed':len(result['accepted']) if result else 0,'B_index':b_index,'C_index':c_index,
            'text_count':sum(map(len,texts.values())),'term_columns':len(weighted),
            'source_train_only':True,'active':active,'B_priority_before_consumer_masks':True,
            'generation_config':result['provenance']['config'] if result else None}
        answer=(texts,weighted,ledger);self.caches['channels'][ck]=answer
        return answer

    def bmscores(self,q,p,variant):
        index,terms,ledger,key=self.index(p,variant)
        cache=self.caches.setdefault('bm',collections.OrderedDict());ck=(q,key)
        if ck not in cache:
            cache[ck]=dict(BM25Retriever(index).search(self.query_text[q],k=len(self.job.base.table_df)))
            if len(cache)>200:cache.popitem(last=False)
        cache.move_to_end(ck)
        return cache[ck],terms,ledger,key

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

    def patterns_for(self,q,p):
        ck=(q,p['pattern_aggregation'],p['pattern_smoothing']);cache=self.caches.setdefault('pattern_variants',{})
        if ck not in cache:
            if p['pattern_aggregation']=='max' and p['pattern_smoothing']==0:
                tokens=self.extractor._get_query_tokens(self.query_text[q])
                cache[ck]=self.extractor._compute_pattern_scores(self.query_text[q],tokens)
            else:
                tokens=self.extractor._get_query_tokens(self.query_text[q]);scores={}
                for node,patterns in self.job.base.patterns.items():
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

    def field_graph(self,q,table,field,p):
        ck=(q,table,field,p['field_graph_threshold'],p['field_graph_neighbors'],p['field_edge_power'])
        cache=self.caches.setdefault('separate_field_graph',{})
        if ck not in cache:
            node=f'C:{table}.{str(field).lower()}';votes=[];tokens=self.extractor._get_query_tokens(self.query_text[q])
            if self.job.base.graph.has_node(node):
                for idx,hist in self.extractor._train_tokens_cache.items():
                    union=len(tokens|hist);sim=len(tokens&hist)/union if union else 0.
                    if sim>p['field_graph_threshold'] and self.job.base.graph.has_edge('Q_'+str(idx),node):
                        votes.append((sim,idx,self.job.base.graph['Q_'+str(idx)][node].get('weight',1.)**p['field_edge_power']))
            top=p['field_graph_neighbors']
            if top:votes.sort(key=lambda v:(-v[0],v[1]))
            cache[ck]=sum(sim*w for sim,idx,w in (votes[:top] if top else votes))
        return min(p['field_graph_cap'],cache[ck]*p['field_graph_gamma'])/max(1,self.column_degree[f'C:{table}.{str(field).lower()}'])**p['field_graph_alpha']

    def predict(self,q,p,variant,trace=False):
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
                    strength=match_strength(term,query,tokens,p['b_match'] if origin=='B' else p['c_match'])
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

    def check_default(self):
        from datasets.gt_utils import parse_ground_truth_fields
        columns=[str(n) for n in self.job.base.graph if str(n).startswith('C:')]
        require(all(n.rsplit('.',1)[1]==n.rsplit('.',1)[1].lower() for n in columns),'Noncanonical field graph node')
        raw_nodes={f'C:{r.table}.{r.field_name}' for r in self.job.base.field_df.itertuples()}
        recovered=missing=attempts=0
        for pos,(_,row) in enumerate(self.job.train.iterrows()):
            for table,field in parse_ground_truth_fields(row):
                original=f'C:{table}.{field}';canonical=f'C:{table}.{str(field).lower()}';attempts+=1
                if self.job.base.graph.has_node(canonical):
                    require(self.job.base.graph.has_edge('Q_'+str(pos),canonical),'Physical training field lost from graph')
                    recovered+=original not in raw_nodes
                else:missing+=1
        self.controls={'canonical_field_graph_nodes':len(columns),'native_index_parity':self.native_index_parity(),
            'training_edge_attempts':attempts,'case_only_training_edges_recovered':recovered,'missing_schema_not_fabricated':missing,
            'old_prediction_equality_not_claimed_after_case_bugfix':True,'C_generation_unchanged':True}
        return self.controls

    def ledger(self,p,variant):
        index,terms,ledger,key=self.index(p,variant)
        if key not in self.index_signatures:self.index_signatures[key]=digest(index)
        return {**ledger,'implementation':self.implementation_id,'config':p,'variant':variant,'index_key':key,'index_sha256':self.index_signatures[key],
            'weighted_TERM_sha256':digest(terms),'table_budget':5,'columns_per_table':10,
            'BHit_semantics':p['b_match'],'C_policy':'Frozen accepted asset; optional stricter post-admission subset only'}
