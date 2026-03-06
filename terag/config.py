"""
TE-RAG 配置管理类

提供统一的配置加载和访问接口
"""

import os
import yaml
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from pathlib import Path


@dataclass
class DataConfig:
    """数据配置"""
    field_csv: str = ""
    table_csv: str = ""
    qa_csv: str = ""
    train_ratio: float = 0.70
    dev_ratio: float = 0.10
    test_ratio: float = 0.20
    splits_dir: str = "splits"


@dataclass
class BM25Config:
    """BM25 配置"""
    k1: float = 1.5
    b: float = 0.75


@dataclass
class FieldWeightsConfig:
    """字段权重配置"""
    table_name: float = 2.0
    table_desc: float = 1.5
    field_name: float = 1.0
    field_desc: float = 0.8
    pattern: float = 0.5


@dataclass
class IndexConfig:
    """索引配置"""
    bm25: BM25Config = field(default_factory=BM25Config)
    field_weights: FieldWeightsConfig = field(default_factory=FieldWeightsConfig)


@dataclass
class GraphConfig:
    """图配置"""
    role_weights: Dict[str, float] = field(default_factory=lambda: {
        'SELECT': 1.2,
        'WHERE': 2.0,
        'JOIN': 1.8,
        'GROUP_BY': 1.5,
        'ORDER_BY': 1.3,
        'HAVING': 1.4,
        'FROM': 1.0
    })


@dataclass
class PatternMiningConfig:
    """模式挖掘配置"""
    entity_keywords: List[str] = field(default_factory=lambda: [
        "公司", "供电所", "单位", "用户", "客户", "部门", "区域"
    ])
    attribute_keywords: List[str] = field(default_factory=lambda: [
        "售电量", "电费", "欠费", "回收率", "户数", "金额", "数量"
    ])
    similarity_threshold: float = 0.3
    min_pattern_count: int = 1


@dataclass
class FeatureConfig:
    """特征配置"""
    table_weights: Dict[str, float] = field(default_factory=lambda: {
        'bm25_score': 0.35,
        'graph_score': 0.35,
        'pattern_score': 0.30
    })
    field_weights: Dict[str, float] = field(default_factory=lambda: {
        'direct_match': 0.40,
        'graph_propagation': 0.20,
        'role_prior': 0.15,
        'train_recommend': 0.25
    })
    graph_propagation_threshold: float = 0.15
    train_similarity_threshold: float = 0.30

    # 特征计算参数（移除硬编码）
    graph_propagation_weight: float = 3.0       # 图传播增强权重
    template_match_weight: float = 0.6          # 模板匹配权重
    direct_match_multiplier: float = 1.5        # 直接匹配倍数
    field_name_match_bonus: float = 0.4         # 字段名匹配加分
    field_graph_score_cap: float = 5.0          # 字段图传播分数上限
    train_recommend_multiplier: float = 1.5     # 训练推荐倍数
    train_recommend_cap: float = 3.0            # 训练推荐分数上限


@dataclass
class AblationConfig:
    """消融实验配置"""
    use_graph_weight: bool = True
    use_template_mining: bool = True
    use_pattern_generalization: bool = True
    use_enhanced_index: bool = True
    use_role_parser: bool = True


@dataclass
class EvaluationConfig:
    """评估配置"""
    table_k: List[int] = field(default_factory=lambda: [1, 3, 5, 10])
    column_k: List[int] = field(default_factory=lambda: [5, 10, 20])
    normalize_sql: bool = True
    execute_sql: bool = True


@dataclass
class OutputConfig:
    """输出配置"""
    artifacts_dir: str = "artifacts"
    results_dir: str = "results"
    save_graph: bool = True
    save_index: bool = True
    save_patterns: bool = True
    save_role_stats: bool = True


class TERAGConfig:
    """
    TE-RAG 统一配置管理类

    使用方式:
        # 从 YAML 文件加载
        config = TERAGConfig.from_yaml("config.yaml")

        # 访问配置
        print(config.seed)
        print(config.data.train_ratio)
        print(config.graph.role_weights['WHERE'])

        # 获取消融开关
        if config.ablation.use_graph_weight:
            ...
    """

    def __init__(
        self,
        version: str = "1.0.0",
        seed: int = 42,
        data: Optional[DataConfig] = None,
        index: Optional[IndexConfig] = None,
        graph: Optional[GraphConfig] = None,
        pattern_mining: Optional[PatternMiningConfig] = None,
        feature: Optional[FeatureConfig] = None,
        ablation: Optional[AblationConfig] = None,
        evaluation: Optional[EvaluationConfig] = None,
        output: Optional[OutputConfig] = None,
    ):
        self.version = version
        self.seed = seed
        self.data = data or DataConfig()
        self.index = index or IndexConfig()
        self.graph = graph or GraphConfig()
        self.pattern_mining = pattern_mining or PatternMiningConfig()
        self.feature = feature or FeatureConfig()
        self.ablation = ablation or AblationConfig()
        self.evaluation = evaluation or EvaluationConfig()
        self.output = output or OutputConfig()

        # 设置随机种子
        self._set_seed()

    def _set_seed(self):
        """设置随机种子"""
        import numpy as np
        import random

        np.random.seed(self.seed)
        random.seed(self.seed)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'TERAGConfig':
        """
        从 YAML 文件加载配置

        Args:
            yaml_path: YAML 配置文件路径

        Returns:
            TERAGConfig 实例
        """
        # 获取项目根目录
        root_dir = Path(yaml_path).parent

        with open(yaml_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)

        # 解析数据配置
        data_cfg = cfg.get('data', {})
        data = DataConfig(
            field_csv=_resolve_path(data_cfg.get('field_csv', ''), root_dir),
            table_csv=_resolve_path(data_cfg.get('table_csv', ''), root_dir),
            qa_csv=_resolve_path(data_cfg.get('qa_csv', ''), root_dir),
            train_ratio=data_cfg.get('train_ratio', 0.70),
            dev_ratio=data_cfg.get('dev_ratio', 0.10),
            test_ratio=data_cfg.get('test_ratio', 0.20),
            splits_dir=_resolve_path(data_cfg.get('splits_dir', 'splits'), root_dir),
        )

        # 解析索引配置
        index_cfg = cfg.get('index', {})
        bm25_cfg = index_cfg.get('bm25', {})
        field_weights_cfg = index_cfg.get('field_weights', {})

        index = IndexConfig(
            bm25=BM25Config(
                k1=bm25_cfg.get('k1', 1.5),
                b=bm25_cfg.get('b', 0.75)
            ),
            field_weights=FieldWeightsConfig(
                table_name=field_weights_cfg.get('table_name', 2.0),
                table_desc=field_weights_cfg.get('table_desc', 1.5),
                field_name=field_weights_cfg.get('field_name', 1.0),
                field_desc=field_weights_cfg.get('field_desc', 0.8),
                pattern=field_weights_cfg.get('pattern', 0.5)
            )
        )

        # 解析图配置
        graph_cfg = cfg.get('graph', {})
        graph = GraphConfig(
            role_weights=graph_cfg.get('role_weights', {
                'SELECT': 1.2, 'WHERE': 2.0, 'JOIN': 1.8,
                'GROUP_BY': 1.5, 'ORDER_BY': 1.3, 'HAVING': 1.4, 'FROM': 1.0
            })
        )

        # 解析模式挖掘配置
        pm_cfg = cfg.get('pattern_mining', {})
        pattern_mining = PatternMiningConfig(
            entity_keywords=pm_cfg.get('entity_keywords', []),
            attribute_keywords=pm_cfg.get('attribute_keywords', []),
            similarity_threshold=pm_cfg.get('similarity_threshold', 0.3),
            min_pattern_count=pm_cfg.get('min_pattern_count', 1)
        )

        # 解析特征配置
        feat_cfg = cfg.get('feature', {})
        feature = FeatureConfig(
            table_weights=feat_cfg.get('table_weights', {
                'bm25_score': 0.35,
                'graph_score': 0.35,
                'pattern_score': 0.30
            }),
            field_weights=feat_cfg.get('field_weights', {
                'direct_match': 0.40,
                'graph_propagation': 0.20,
                'role_prior': 0.15,
                'train_recommend': 0.25
            }),
            graph_propagation_threshold=feat_cfg.get('graph_propagation_threshold', 0.15),
            train_similarity_threshold=feat_cfg.get('train_similarity_threshold', 0.30),
            # 新增特征计算参数
            graph_propagation_weight=feat_cfg.get('graph_propagation_weight', 3.0),
            template_match_weight=feat_cfg.get('template_match_weight', 0.6),
            direct_match_multiplier=feat_cfg.get('direct_match_multiplier', 1.5),
            field_name_match_bonus=feat_cfg.get('field_name_match_bonus', 0.4),
            field_graph_score_cap=feat_cfg.get('field_graph_score_cap', 5.0),
            train_recommend_multiplier=feat_cfg.get('train_recommend_multiplier', 1.5),
            train_recommend_cap=feat_cfg.get('train_recommend_cap', 3.0)
        )

        # 解析消融配置
        abl_cfg = cfg.get('ablation', {})
        ablation = AblationConfig(
            use_graph_weight=abl_cfg.get('use_graph_weight', True),
            use_template_mining=abl_cfg.get('use_template_mining', True),
            use_pattern_generalization=abl_cfg.get('use_pattern_generalization', True),
            use_enhanced_index=abl_cfg.get('use_enhanced_index', True),
            use_role_parser=abl_cfg.get('use_role_parser', True)
        )

        # 解析评估配置
        eval_cfg = cfg.get('evaluation', {})
        evaluation = EvaluationConfig(
            table_k=eval_cfg.get('table_k', [1, 3, 5, 10]),
            column_k=eval_cfg.get('column_k', [5, 10, 20]),
            normalize_sql=eval_cfg.get('normalize_sql', True),
            execute_sql=eval_cfg.get('execute_sql', True)
        )

        # 解析输出配置
        out_cfg = cfg.get('output', {})
        output = OutputConfig(
            artifacts_dir=_resolve_path(out_cfg.get('artifacts_dir', 'artifacts'), root_dir),
            results_dir=_resolve_path(out_cfg.get('results_dir', 'results'), root_dir),
            save_graph=out_cfg.get('save_graph', True),
            save_index=out_cfg.get('save_index', True),
            save_patterns=out_cfg.get('save_patterns', True),
            save_role_stats=out_cfg.get('save_role_stats', True)
        )

        return cls(
            version=cfg.get('version', '1.0.0'),
            seed=cfg.get('seed', 42),
            data=data,
            index=index,
            graph=graph,
            pattern_mining=pattern_mining,
            feature=feature,
            ablation=ablation,
            evaluation=evaluation,
            output=output
        )

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'version': self.version,
            'seed': self.seed,
            'data': {
                'field_csv': self.data.field_csv,
                'table_csv': self.data.table_csv,
                'qa_csv': self.data.qa_csv,
                'train_ratio': self.data.train_ratio,
                'dev_ratio': self.data.dev_ratio,
                'test_ratio': self.data.test_ratio,
                'splits_dir': self.data.splits_dir,
            },
            'index': {
                'bm25': {
                    'k1': self.index.bm25.k1,
                    'b': self.index.bm25.b,
                },
                'field_weights': {
                    'table_name': self.index.field_weights.table_name,
                    'table_desc': self.index.field_weights.table_desc,
                    'field_name': self.index.field_weights.field_name,
                    'field_desc': self.index.field_weights.field_desc,
                    'pattern': self.index.field_weights.pattern,
                }
            },
            'graph': {
                'role_weights': self.graph.role_weights,
            },
            'pattern_mining': {
                'entity_keywords': self.pattern_mining.entity_keywords,
                'attribute_keywords': self.pattern_mining.attribute_keywords,
                'similarity_threshold': self.pattern_mining.similarity_threshold,
                'min_pattern_count': self.pattern_mining.min_pattern_count,
            },
            'feature': {
                'table_weights': self.feature.table_weights,
                'field_weights': self.feature.field_weights,
                'graph_propagation_threshold': self.feature.graph_propagation_threshold,
                'train_similarity_threshold': self.feature.train_similarity_threshold,
                # 新增特征计算参数
                'graph_propagation_weight': self.feature.graph_propagation_weight,
                'template_match_weight': self.feature.template_match_weight,
                'direct_match_multiplier': self.feature.direct_match_multiplier,
                'field_name_match_bonus': self.feature.field_name_match_bonus,
                'field_graph_score_cap': self.feature.field_graph_score_cap,
                'train_recommend_multiplier': self.feature.train_recommend_multiplier,
                'train_recommend_cap': self.feature.train_recommend_cap,
            },
            'ablation': {
                'use_graph_weight': self.ablation.use_graph_weight,
                'use_template_mining': self.ablation.use_template_mining,
                'use_pattern_generalization': self.ablation.use_pattern_generalization,
                'use_enhanced_index': self.ablation.use_enhanced_index,
                'use_role_parser': self.ablation.use_role_parser,
            },
            'evaluation': {
                'table_k': self.evaluation.table_k,
                'column_k': self.evaluation.column_k,
                'normalize_sql': self.evaluation.normalize_sql,
                'execute_sql': self.evaluation.execute_sql,
            },
            'output': {
                'artifacts_dir': self.output.artifacts_dir,
                'results_dir': self.output.results_dir,
                'save_graph': self.output.save_graph,
                'save_index': self.output.save_index,
                'save_patterns': self.output.save_patterns,
                'save_role_stats': self.output.save_role_stats,
            }
        }

    def save_yaml(self, yaml_path: str):
        """保存到 YAML 文件"""
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.to_dict(), f, allow_unicode=True, default_flow_style=False)

    def get_artifact_path(self, filename: str) -> str:
        """获取 artifacts 文件路径"""
        os.makedirs(self.output.artifacts_dir, exist_ok=True)
        return os.path.join(self.output.artifacts_dir, filename)

    def get_split_path(self, split_name: str) -> str:
        """获取数据划分文件路径"""
        os.makedirs(self.data.splits_dir, exist_ok=True)
        return os.path.join(self.data.splits_dir, f"{split_name}.jsonl")


def _resolve_path(path: str, root_dir: Path) -> str:
    """解析路径，支持相对路径和绝对路径"""
    if not path:
        return path
    if os.path.isabs(path):
        return path
    return str((root_dir / path).resolve())
