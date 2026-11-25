"""Current baseline wiring; shared prediction format and fixed budgets."""
import importlib
import os
from pathlib import Path
from retrievers.base_retriever import RetrieverConfig


def run_enterprise(method, cfg, train, qids, questions):
    names = {
        'BM25': ('bm25_retriever', 'BM25Retriever'),
        'MetadataMatched': ('matched_schema_retriever', 'MatchedSchemaRetriever'),
        'ExampleRRF': ('example_rag_retriever', 'ExampleRAGRetriever'),
        'TFIDF': ('vector_retriever', 'VectorRetriever'),
        'Hybrid': ('hybrid_retriever', 'HybridRetriever'),
        'CooccurRRF': ('cooccurrence_retriever', 'CooccurrenceRetriever'),
        'Dense': ('dense_retriever', 'DenseRetriever'),
        'DenseDeepSeek': ('linkalign_retriever', 'LinkAlignRetriever'),
        'KaSLA-Adapted': ('kasla_retriever', 'KaSLAAdaptedRetriever'),
        'LinkAlign-SingleDB-Adapted': ('linkalign_official_retriever', 'OfficialLinkAlignRetriever'),
    }
    module, name = names[method]
    implementation = getattr(importlib.import_module('retrievers.'+module), name)
    config = RetrieverConfig(name=method, extra_params={'terag_config':cfg, 'fusion_mode':'rank_equal',
                              'use_history':False,'use_transfer':False})
    retriever = implementation(cfg.data.field_csv, cfg.data.table_csv, config)
    retriever.fit(train)
    from experiments.bird.baselines.tfidf_bird import run_retriever_records
    return run_retriever_records(retriever,method,'enterprise',qids,questions)


def run_bird(method, cfg, db, train, qids, questions):
    field, table = cfg.data.field_csv, cfg.data.table_csv
    if method in ('BM25','MetadataMatched','ExampleRRF'):
        from experiments.bird.baselines.run_baselines import run_baseline
        return run_baseline(method,cfg,db,train,qids,questions,field,table)
    modules = {'TFIDF':('tfidf_bird','run_tfidf'),'Hybrid':('hybrid_bird','run_hybrid'),
               'CooccurRRF':('cooccur_bird','run_cooccur_rrf'),'Dense':('dense_bird','run_dense')}
    if method in modules:
        module,name=modules[method]
        function=getattr(importlib.import_module('experiments.bird.baselines.'+module),name)
        kwargs={'cache_root':Path(os.environ['TERAG_OUTPUT'])/'dense_cache'} if method=='Dense' else {}
        return function(cfg,db,train,qids,questions,field,table,**kwargs)
    classes={'DenseDeepSeek':('densedeepseek_bird','DenseDeepSeekBird'),
             'KaSLA-Adapted':('kasla_bird','KaSLABird'),
             'LinkAlign-SingleDB-Adapted':('linkalign_bird','LinkAlignBird')}
    module,name=classes[method]
    cls=getattr(importlib.import_module('experiments.bird.api.'+module),name)
    retriever=cls(field,table,unit_label=db)
    retriever.fit(train)
    from experiments.bird.baselines.tfidf_bird import run_retriever_records
    from experiments.bird.api.budget_broker import BudgetExceeded
    from experiments.bird.api.api_client import MethodBlocked
    return run_retriever_records(retriever,method,db,qids,questions,
                                 stop_exceptions=(BudgetExceeded,MethodBlocked))
