"""Shared TERAGConfig factory for the BIRD quick run (Lead-owned; agents import, never edit).

Builds the per-database TE-RAG config programmatically. Never reads the project
config.yaml (it contains credentials). Fixed resource values come from
PARAMETER_REGISTRY_FINAL.md (targeted_ablation_optimization_v2 fixed base config);
scoring weights are additionally applied per-variant by the engine's DEFAULT params,
which are exported separately and frozen by the Lead.
"""
import copy
from pathlib import Path

from terag.config import (TERAGConfig, DataConfig, IndexConfig, BM25Config,
                          FieldWeightsConfig, GraphConfig, FeatureConfig,
                          AnnotationsConfig, AblationConfig, OutputConfig,
                          TokenizerConfig, PatternMiningConfig, BirdDatasetConfig)

RUN_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = RUN_ROOT / 'workspace'
SCHEMA_ROOT = RUN_ROOT / 'data' / 'bird' / 'schema'


def make_bird_config(db_id: str, artifacts_dir=None, language: str = 'english',
                     dialect: str = 'sqlite') -> TERAGConfig:
    """Per-database TE-RAG config for BIRD (question-only, no evidence)."""
    schema_dir = SCHEMA_ROOT / db_id
    field_csv = str(schema_dir / 'field_schema.csv')
    table_csv = str(schema_dir / 'table_schema.csv')
    cfg = TERAGConfig(
        seed=42,
        data=DataConfig(dataset='bird', field_csv=field_csv, table_csv=table_csv, qa_csv=''),
        index=IndexConfig(
            bm25=BM25Config(k1=1.5, b=0.75),
            field_weights=FieldWeightsConfig(table_name=2.0, table_desc=1.5,
                                             field_name=1.0, field_desc=0.8, pattern=0.5),
            include_field_desc=True),
        graph=GraphConfig(),          # accepted AST role weights (defaults)
        pattern_mining=PatternMiningConfig(mode='generic'),  # English generic patterns
        feature=FeatureConfig(),      # registry fixed values; engine DEFAULT governs scoring
        annotations=AnnotationsConfig(use_history=True, use_transfer=True,
                                      use_annotation_index=True),
        ablation=AblationConfig(),    # all accepted channels on; per-variant switches live in the engine
        output=OutputConfig(artifacts_dir=str(artifacts_dir or (RUN_ROOT / 'cache' / db_id)),
                            save_graph=False, save_index=False, save_patterns=False,
                            save_role_stats=False),
        tokenizer=TokenizerConfig(language=language),
    )
    # bird block carries ONLY the dialect/language identity (sql_dialect property
    # needs it); raw BIRD loading stays with the Data agent, paths stay empty.
    cfg.data.bird = BirdDatasetConfig(sql_dialect=dialect, use_evidence=False,
                                      use_database_description=True,
                                      use_value_description=True)
    assert cfg.sql_dialect == dialect
    return cfg
