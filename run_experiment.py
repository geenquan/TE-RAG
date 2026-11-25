"""Run the current enterprise or BIRD experiments from this working tree."""
import argparse
import gzip
import json
import os
from pathlib import Path
import site
import sys

ROOT = Path(__file__).resolve().parent
# Use the existing verified numerical environment, independently of any archive.
# -S avoids the IDE SDK's known global NumPy ABI conflict.
DEPENDENCIES = ROOT / '.runtime/site-packages'
if DEPENDENCIES.is_dir():
    sys.path.insert(0, str(DEPENDENCIES))
sys.path.insert(0, str(ROOT))
# Optional model/API packages come after the verified numerical dependencies.
# Listing SDK package directories does not execute their .pth files.
for path in site.getsitepackages():
    if path not in sys.path:
        sys.path.append(path)
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(key, '2')
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')

BASELINES = ['BM25', 'MetadataMatched', 'ExampleRRF', 'TFIDF', 'Hybrid', 'CooccurRRF', 'Dense',
             'DenseDeepSeek', 'KaSLA-Adapted', 'LinkAlign-SingleDB-Adapted']
API_METHODS = BASELINES[-3:]


def load(path):
    return json.loads(Path(path).read_text())


def enterprise_choices(protocol, fold, track):
    choices = load(ROOT/'configs/enterprise.json')['choices']
    settings = {'question': 'question_v1', 'heldout': 'table_v2'}
    for c in choices:
        if protocol != 'all' and c['setting'] != settings[protocol]: continue
        if fold is not None and c['fold'] != fold: continue
        for variant in c['selection']:
            p = c['selection'][variant if track == 'independent' else 'Full']['params']
            yield c['setting'], c['fold'], variant, p


def bird_choices(protocol, track):
    cfg = load(ROOT/'configs/bird.json')
    selected = cfg['selected'][protocol]
    if track == 'core':
        from experiments.bird.engine.study_defaults import DEFAULT, VARIANTS
        from experiments.bird.engine.legacy_ablations import masked_params
        rows=[{'name': v, 'engine': 'bird', 'engine_variant': v, 'params': DEFAULT} for v in VARIANTS]
        extra=['noBC','noAnnIdx','noAnnHit'] if protocol=='question' else ['noBC']
        return rows+[{'name':v,'engine':'bird','engine_variant':'Full','params':masked_params(DEFAULT,v)} for v in extra]
    if track == 'selected': return [selected]
    key = 'fixed_ablation_members' if track == 'fixed' else 'optimized_ablations'
    if track == 'fixed':
        return [selected] + [m for m in cfg[key] if m.get('ablation_of') == selected['name']]
    optimized = next(m for m in cfg[key] if m.get('ablation_of') == selected['name'])
    return [optimized] + [m for m in cfg[key] if m.get('ablation_of') == optimized['name']]


def output_path(value):
    path = Path(value).resolve()
    allowed = (ROOT/'results').resolve()
    if not path.is_relative_to(allowed) or path == allowed:
        raise ValueError('--output must be a new subdirectory of results/')
    if path.exists(): raise ValueError('Output already exists; choose a new --output')
    return path


def signature(record):
    return record['retrieved_tables'], record['retrieved_column_keys']


def reference(dataset):
    path = ROOT/'data/reference'/f'{dataset}.jsonl.gz'
    with gzip.open(path, 'rt') as f:
        return {tuple(r['key']): r['prediction'] for r in map(json.loads, f)}


def summarize(records, dataset):
    from collections import defaultdict
    import numpy as np
    groups = defaultdict(list)
    for r in records:
        if r['valid']: groups[tuple(r['group'])].append(r['metrics'])
    return [{'group': list(k), 'n': len(v), **{m: float(np.mean([x[m] for x in v])) for m in v[0]}}
            for k, v in sorted(groups.items())]


def enterprise(args, output):
    from experiments.enterprise.model import Engine
    from experiments.enterprise.runtime import inputs, configuration, subset_train
    from experiments.baselines import run_enterprise
    from eval.full_gold import full_gold_metrics
    _, rows, gold = inputs()
    engines, predictions, evaluated, mismatches = {}, [], [], []
    expected = reference('enterprise') if args.verify else {}
    for setting, fold, variant, params in enterprise_choices(args.protocol, args.fold, args.track):
        if args.methods and variant not in args.methods: continue
        key = setting, fold
        if key not in engines:
            engines[key] = Engine(setting, fold, 'outer', output/setting/f'fold{fold}')
        engine = engines[key]
        qids = engine.spec['test'][:args.limit] if args.limit else engine.spec['test']
        for qid in qids:
            rec, met, _ = engine.predict(qid, params, variant)
            rec['track'] = args.track
            predictions.append(rec)
            evaluated.append({'group':[setting,args.track,variant], 'valid':met['valid'],
                              'metrics':{k:v for k,v in met.items() if k not in ('qid','valid')}})
            if args.verify:
                k = (setting,fold,args.track,variant,qid)
                if k not in expected or signature(rec) != signature(expected[k]): mismatches.append(list(k))
        print(f'{setting} fold{fold} {args.track}/{variant}: {len(qids)} questions',flush=True)
    # Baselines run through the same fixed partitions and full-SQL training supervision.
    for setting, fold in sorted({(s,f) for s,f,_,_ in enterprise_choices(args.protocol,args.fold,args.track)}):
        for method in (m for m in (args.methods or []) if m in BASELINES):
            meta, rows, gold = inputs()
            spec=next(s for s in meta['splits'] if s['setting']==setting and s['fold']==fold)
            cfg=configuration(meta, output/'baseline_cache')
            qids=spec['test'][:args.limit] if args.limit else spec['test']
            recs=run_enterprise(method,cfg,subset_train(rows,gold,spec['train']),qids,dict(zip(rows.qid,rows.question)))
            for rec in recs:
                rec.update(setting=setting,fold=fold,track='baseline')
                predictions.append(rec)
                metric=full_gold_metrics(rec,gold)
                evaluated.append({'group':[setting,'baseline',method], 'valid':gold[rec['qid']]['gold_status']=='verified_complete',
                    'metrics':dict(zip(['covered@5','colp','colrec','colf1','fgcs'],map(float,metric[1:])))})
    if not predictions: raise ValueError('No requested enterprise methods/configurations')
    return predictions,evaluated,mismatches


def bird(args, output):
    import pandas as pd
    from experiments.bird.config import make_bird_config
    from experiments.bird.engine.bird_engine import BirdEngine
    from experiments.bird.engine.vnext_engine import VnextEngine
    from experiments.bird.engine.unified_entry import run_one, final_metric
    from experiments.baselines import run_bird
    data = ROOT/'data/bird'; gold = load(data/'gold.json')
    expected = reference('bird') if args.verify else {}
    predictions,evaluated,mismatches=[],[],[]
    for protocol in (['question','heldout'] if args.protocol=='all' else [args.protocol]):
        name = 'q5_manifest.json' if protocol=='question' else 'h5_manifest.json'
        manifest=load(data/name)
        choices=bird_choices(protocol,args.track)
        methods=([m for m in choices if m['name'] in args.methods or
                  (m==choices[0] and 'Full' in args.methods)] if args.methods else choices)
        baseline_names=[m for m in args.methods or [] if m in BASELINES]
        for db, ds in sorted(manifest['per_db'].items()):
            if args.db and db!=args.db:continue
            for f,fs in sorted(ds['per_fold'].items()):
                fold=int(f)
                if not fs['applicable'] or args.fold is not None and args.fold!=fold:continue
                qids=fs['test_qids'][:args.limit] if args.limit else fs['test_qids']
                rows=pd.DataFrame([json.loads(s) for s in (data/'train'/f'{protocol}_f{fold}_{db}.jsonl').read_text().splitlines() if s.strip()])
                if set(rows.qid)!=set(fs['train_qids']):raise ValueError('BIRD training partition mismatch')
                local={q:{'gold_status':gold[q]['parse_status'],'gold_tables':gold[q]['gold_tables'],
                          'gold_columns':[f'{t}.{c}' for t,c in gold[q]['gold_columns']]} for q in qids}
                questions={q:gold[q]['question'] for q in qids}
                cache=output/'cache'/protocol/f'fold{fold}'/db
                cfg=make_bird_config(db,cache);engines={}
                def append(rec,metric,method,evidence=None):
                    rec.update(protocol=protocol,fold=fold,db_id=db,variant=method)
                    if evidence is not None:rec['execution_evidence']=evidence
                    predictions.append(rec)
                    evaluated.append({'group':[protocol,args.track,method],
                        'valid':local[rec['qid']]['gold_status']=='verified_complete','metrics':metric})
                    if args.verify:
                        key=(protocol,fold,db,method,rec['qid'])
                        if key not in expected or signature(rec)!=signature(expected[key]):mismatches.append(list(key))
                for m in methods:
                    kind=m.get('engine','bird')
                    if kind not in engines:
                        engines[kind]=(VnextEngine if kind=='vnext' else BirdEngine)(cfg,db,rows,qids,local,questions,
                            cache_dir=cache/kind,heldout_tables=fs.get('heldout_tables',[]) if protocol=='heldout' else None)
                    for q in qids:
                        rec,metric,ev=run_one(engines[kind],q,m['params'],m.get('engine_variant','Full'),local[q])
                        append(rec,metric,m['name'],ev)
                for method in baseline_names:
                    for rec in run_bird(method,cfg,db,rows,qids,questions):
                        append(rec,final_metric(rec,local[rec['qid']]),method)
                print(f'{protocol} fold{fold} {db}: {len(qids)} questions, {len(methods)+len(baseline_names)} methods',flush=True)
    if not predictions:raise ValueError('No requested BIRD methods/configurations')
    return predictions,evaluated,mismatches


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset',choices=['enterprise','bird'])
    parser.add_argument('--list',action='store_true')
    parser.add_argument('--check',action='store_true')
    parser.add_argument('--protocol',choices=['all','question','heldout'],default='all')
    parser.add_argument('--fold',type=int,choices=range(5))
    parser.add_argument('--track',choices=['selected','fixed','independent','core'],default=None)
    parser.add_argument('--methods',nargs='+')
    parser.add_argument('--db')
    parser.add_argument('--limit',type=int,help='Smoke-test question limit per fold/db; never a formal result.')
    parser.add_argument('--output')
    parser.add_argument('--verify',action='store_true',help='Compare ordered outputs with saved current predictions.')
    parser.add_argument('--allow-api',action='store_true',help='Explicitly enable a requested paid baseline.')
    args=parser.parse_args()
    args.track=args.track or ('fixed' if args.dataset=='enterprise' else 'selected')
    if args.dataset=='enterprise' and args.track not in ('fixed','independent'):
        parser.error('Enterprise supports fixed/independent tracks')
    if args.limit is not None and args.limit<1:parser.error('--limit must be positive')
    if args.list:
        if args.dataset=='enterprise':
            print(json.dumps([{'protocol':s,'fold':f,'variant':v} for s,f,v,p in enterprise_choices(args.protocol,args.fold,args.track)],indent=2))
        else:
            print(json.dumps({p:[m['name'] for m in bird_choices(p,args.track)] for p in ['question','heldout']},indent=2))
        print('Baselines: '+', '.join(BASELINES));return
    if args.check:
        from experiments.check import check
        print(json.dumps(check(args.dataset),ensure_ascii=False,indent=2));return
    allowed=set(BASELINES)
    if args.dataset=='enterprise':
        allowed.update(v for _,_,v,_ in enterprise_choices(args.protocol,args.fold,args.track))
    else:
        for protocol in ['question','heldout']:
            allowed.update(m['name'] for m in bird_choices(protocol,args.track))
        allowed.add('Full')
    unknown=set(args.methods or [])-allowed
    if unknown:parser.error('Unknown methods for this track: '+', '.join(sorted(unknown)))
    if any(m in API_METHODS+['Dense'] for m in args.methods or []) and args.protocol!='question':
        parser.error('Dense and API comparisons are available under --protocol question')
    if args.dataset=='enterprise' and args.verify and any(m in BASELINES for m in args.methods or []):
        parser.error('Enterprise --verify covers current Full/ablations; run baselines separately without --verify')
    if os.environ.get('PYTHONHASHSEED')!='0':parser.error('Set PYTHONHASHSEED=0 before running experiments')
    if not args.output:parser.error('--output results/<new-name> is required')
    if any(m in API_METHODS for m in args.methods or []) and not args.allow_api:parser.error('Paid methods require --allow-api')
    output=output_path(args.output)
    os.environ['TERAG_OUTPUT']=str(output)
    output.mkdir(parents=True)
    predictions, evaluated, mismatches=globals()[args.dataset](args,output)
    with gzip.open(output/'predictions.jsonl.gz','xt') as f:
        for r in predictions:f.write(json.dumps(r,ensure_ascii=False,default=str)+'\n')
    summary={'dataset':args.dataset,'track':args.track,'smoke':bool(args.limit),
             'records':len(predictions),'errors':sum(r.get('status')=='error' for r in predictions),
             'verification_requested':args.verify,'mismatches':mismatches,'summary':summarize(evaluated,args.dataset)}
    (output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    print(json.dumps({k:v for k,v in summary.items() if k!='summary'},ensure_ascii=False))
    if mismatches or summary['errors']:raise SystemExit(1)

if __name__=='__main__':main()
