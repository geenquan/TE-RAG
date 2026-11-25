"""Read-only checks for the current experiment inputs and imports."""
import hashlib
import importlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def check(dataset):
    if not (ROOT/'data/INPUT_MANIFEST.json').is_file():
        raise RuntimeError('Missing authorized inputs: restore data/INPUT_MANIFEST.json and the dataset assets described in PRIVATE_INPUTS.md')
    entries=json.loads((ROOT/'data/INPUT_MANIFEST.json').read_text())
    checked=0
    for row in entries:
        if row['dataset'] not in ('shared',dataset):continue
        path=ROOT/row['path']
        if not path.is_file():raise RuntimeError('Missing current input: '+row['path'])
        if hashlib.sha256(path.read_bytes()).hexdigest()!=row['sha256']:
            raise RuntimeError('Current input changed: '+row['path'])
        checked+=1
    for name in ['numpy','pandas','scipy','sqlglot','networkx','jieba','sklearn','yaml',
                 'experiments.enterprise.model','experiments.bird.engine.unified_entry',
                 'experiments.bird.engine.vnext_engine']:
        module=importlib.import_module(name)
        if '_archive' in str(getattr(module,'__file__','')):raise RuntimeError('Archived import: '+name)
    return {'dataset':dataset,'checked_inputs':checked,'current_imports':True,'archive_required':False}
