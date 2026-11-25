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
class BirdSplitConfig:
    """BIRD 官方单个 split（train/dev/test）路径配置"""
    json: str = ""
    tables_json: str = ""
    databases_dir: str = ""


@dataclass
class BirdSplitStrategyConfig:
    """BIRD within-database 划分配置"""
    strategy: str = "within_database"
    train_ratio: float = 0.70
    dev_ratio: float = 0.10
    test_ratio: float = 0.20
    min_queries_per_db: int = 5


@dataclass
class BirdSchemaConfig:
    """BIRD schema 转换配置"""
    prefix_table_with_db: bool = True
    include_pk_fk: bool = True


@dataclass
class BirdDebugConfig:
    """BIRD 调试配置"""
    max_queries: Optional[int] = None


@dataclass
class BirdDatasetConfig:
    """BIRD 数据集配置"""
    root_dir: str = ""
    sql_dialect: str = "sqlite"
    source_split: str = "dev"
    train: BirdSplitConfig = field(default_factory=BirdSplitConfig)
    dev: BirdSplitConfig = field(default_factory=BirdSplitConfig)
    test: BirdSplitConfig = field(default_factory=BirdSplitConfig)
    processed_dir: str = ""
    use_evidence: bool = True
    use_database_description: bool = True
    use_value_description: bool = True
    evaluation_mode: str = "per_database"
    split: BirdSplitStrategyConfig = field(default_factory=BirdSplitStrategyConfig)
    schema: BirdSchemaConfig = field(default_factory=BirdSchemaConfig)
    pattern_mining_mode: str = "generic"
    debug: BirdDebugConfig = field(default_factory=BirdDebugConfig)

    # 标准化输出文件路径（由 processed_dir 派生）
    @property
    def table_csv(self) -> str:
        return os.path.join(self.processed_dir, 'bird_table_schema.csv')

    @property
    def field_csv(self) -> str:
        return os.path.join(self.processed_dir, 'bird_field_schema.csv')

    @property
    def qa_csv(self) -> str:
        return os.path.join(self.processed_dir, 'bird_qa.csv')

    def get_source_split_config(self, split_name: str) -> BirdSplitConfig:
        """获取官方 split 的路径配置"""
        if split_name == 'train':
            return self.train
        if split_name == 'test':
            return self.test
        return self.dev


@dataclass
class DataConfig:
    """数据配置"""
    dataset: str = "powergrid"
    field_csv: str = ""
    table_csv: str = ""
    qa_csv: str = ""
    train_ratio: float = 0.70
    dev_ratio: float = 0.10
    test_ratio: float = 0.20
    splits_dir: str = "splits"
    bird: Optional[BirdDatasetConfig] = None


@dataclass
class TokenizerConfig:
    """分词器配置（按数据集选择分词语言: chinese / english）"""
    default: str = "chinese"
    datasets: Dict[str, str] = field(default_factory=lambda: {
        'powergrid': 'chinese',
        'bird': 'english',
    })
    # 当前激活数据集对应的分词语言（加载时解析）
    language: str = "chinese"

    def for_dataset(self, dataset: str) -> str:
        """获取指定数据集的分词语言"""
        return self.datasets.get(dataset, self.default)


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
    # P2-C 索引因素分解旋钮：字段描述是否进入增强文档（默认 True=现行行为）
    include_field_desc: bool = True


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
    # SQL 角色解析模式：'ast'（P2-A 验证）| 'regex'（历史回退，仅诊断对照）
    parse_mode: str = 'ast'


@dataclass
class PatternMiningConfig:
    """模式挖掘配置"""
    # 挖掘模式: keywords（中文业务词表）/ generic（从历史 Query-SQL 自动学习，适配英文）
    mode: str = "keywords"
    entity_keywords: List[str] = field(default_factory=lambda: [
        "公司", "供电所", "单位", "用户", "客户", "部门", "区域"
    ])
    attribute_keywords: List[str] = field(default_factory=lambda: [
        "售电量", "电费", "欠费", "回收率", "户数", "金额", "数量"
    ])
    # 聚合关键词（中文，keywords 模式）
    aggregation_keywords: Dict[str, str] = field(default_factory=lambda: {
        '多少': 'VALUE', '哪些': 'LIST', '总计': 'SUM', '合计': 'SUM',
        '平均': 'AVG', '最大': 'MAX', '最小': 'MIN', '数量': 'COUNT', '排名': 'RANK',
    })
    # 聚合关键词（英文，generic 模式）
    english_aggregation_keywords: Dict[str, str] = field(default_factory=lambda: {
        'how many': 'COUNT', 'how much': 'VALUE', 'number of': 'COUNT', 'count': 'COUNT',
        'average': 'AVG', 'mean': 'AVG', 'total': 'SUM', 'sum': 'SUM',
        'maximum': 'MAX', 'max': 'MAX', 'highest': 'MAX', 'largest': 'MAX', 'most': 'MAX',
        'minimum': 'MIN', 'min': 'MIN', 'lowest': 'MIN', 'smallest': 'MIN', 'least': 'MIN',
    })
    # 时间模式（中文，keywords 模式）: [(regex, tag), ...]
    time_patterns: List[Any] = field(default_factory=lambda: [
        [r'\d{4}年\d{1,2}月', 'YEAR_MONTH'], [r'\d{4}年', 'YEAR'], [r'\d{1,2}月', 'MONTH'],
        [r'今年', 'THIS_YEAR'], [r'去年', 'LAST_YEAR'], [r'上月', 'LAST_MONTH'],
        [r'本季度', 'THIS_QUARTER'],
    ])
    # 时间模式（英文，generic 模式）
    english_time_patterns: List[Any] = field(default_factory=lambda: [
        [r'\b(19|20)\d{2}\s*(-|to|through|and)\s*(19|20)?\d{2}\b', 'YEAR_RANGE'],
        [r'\b(19|20)\d{2}\b', 'YEAR'],
        [r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\b', 'MONTH'],
        [r'\bthis year\b', 'THIS_YEAR'], [r'\blast year\b', 'LAST_YEAR'],
        [r'\blast month\b', 'LAST_MONTH'], [r'\bthis month\b', 'THIS_MONTH'],
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
        'train_recommend': 0.25,
        # P2-D3 注解命中特征：特征值已按 w_ann=0.5/命中 计提（与 D1/D2 对照
        # 同一声明值），排序权重取 1.0 = 直通（非调参；0 会使 B/C 列通道失效）
        'annotation_hit': 1.0
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
class AnnotationsConfig:
    """B/C 注解通道开关（P2-D3 主线接线；缺省关闭=旧行为，支持消融）"""
    use_history: bool = False    # B：训练日志直接注解（D1）
    use_transfer: bool = False   # C：受约束迁移注解（D2，依赖 use_history）
    use_annotation_index: bool = True   # 消融：B/C 文本是否入索引（False=仅 Schema 描述）


@dataclass
class AblationConfig:
    """消融实验配置"""
    use_graph_weight: bool = True
    use_template_mining: bool = True
    use_pattern_generalization: bool = True
    use_enhanced_index: bool = True
    use_role_parser: bool = True
    # 角色权重统一消融：True=使用 alpha_r 角色权重；False=所有角色统一为 1.0
    # （保留 AST 解析和边集，仅将权重设为统一值——区别于 use_role_parser=False
    #  会跳过 AST 进入字符串回退）
    use_role_weight: bool = True


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
        tokenizer: Optional[TokenizerConfig] = None,
        annotations: Optional[AnnotationsConfig] = None,
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
        self.tokenizer = tokenizer or TokenizerConfig()
        self.annotations = annotations or AnnotationsConfig()

        # 设置随机种子
        self._set_seed()

    def _set_seed(self):
        """设置随机种子"""
        import numpy as np
        import random

        np.random.seed(self.seed)
        random.seed(self.seed)

    @classmethod
    def from_yaml(cls, yaml_path: str, dataset: Optional[str] = None) -> 'TERAGConfig':
        """
        从 YAML 文件加载配置

        Args:
            yaml_path: YAML 配置文件路径
            dataset: 数据集覆盖（powergrid / bird），None 时使用配置文件中的 data.dataset

        Returns:
            TERAGConfig 实例
        """
        # 获取项目根目录
        root_dir = Path(yaml_path).parent

        with open(yaml_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)

        # 解析数据配置
        data_cfg = cfg.get('data', {})
        dataset_name = dataset or data_cfg.get('dataset', 'powergrid')

        # 解析 BIRD 配置
        bird = cls._parse_bird_config(data_cfg.get('bird', {}), root_dir)

        # 根据激活数据集解析数据路径：
        # - powergrid: 使用原有 field_csv/table_csv/qa_csv（完全向后兼容）
        # - bird: 使用 bird.processed_dir 下的标准化文件
        if dataset_name == 'bird':
            field_csv = bird.field_csv
            table_csv = bird.table_csv
            qa_csv = bird.qa_csv
            splits_dir = bird.processed_dir
            train_ratio = bird.split.train_ratio
            dev_ratio = bird.split.dev_ratio
            test_ratio = bird.split.test_ratio
        else:
            pg_cfg = data_cfg.get('powergrid', {})
            field_csv = _resolve_path(
                data_cfg.get('field_csv', '') or pg_cfg.get('field_csv', ''), root_dir)
            table_csv = _resolve_path(
                data_cfg.get('table_csv', '') or pg_cfg.get('table_csv', ''), root_dir)
            qa_csv = _resolve_path(
                data_cfg.get('qa_csv', '') or pg_cfg.get('qa_csv', ''), root_dir)
            splits_dir = _resolve_path(data_cfg.get('splits_dir', 'splits'), root_dir)
            train_ratio = data_cfg.get('train_ratio', 0.70)
            dev_ratio = data_cfg.get('dev_ratio', 0.10)
            test_ratio = data_cfg.get('test_ratio', 0.20)

        data = DataConfig(
            dataset=dataset_name,
            field_csv=field_csv,
            table_csv=table_csv,
            qa_csv=qa_csv,
            train_ratio=train_ratio,
            dev_ratio=dev_ratio,
            test_ratio=test_ratio,
            splits_dir=splits_dir,
            bird=bird,
        )

        # 解析分词器配置（按激活数据集解析分词语言）
        tok_cfg = cfg.get('tokenizer', {})
        tokenizer = TokenizerConfig(
            default=tok_cfg.get('default', 'chinese'),
            datasets=tok_cfg.get('datasets', {
                'powergrid': 'chinese', 'bird': 'english'
            }),
        )
        tokenizer.language = tokenizer.for_dataset(dataset_name)

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
            mode=pm_cfg.get('mode', 'keywords'),
            entity_keywords=pm_cfg.get('entity_keywords', []),
            attribute_keywords=pm_cfg.get('attribute_keywords', []),
            aggregation_keywords=pm_cfg.get('aggregation_keywords', {
                '多少': 'VALUE', '哪些': 'LIST', '总计': 'SUM', '合计': 'SUM',
                '平均': 'AVG', '最大': 'MAX', '最小': 'MIN', '数量': 'COUNT', '排名': 'RANK',
            }),
            english_aggregation_keywords=pm_cfg.get('english_aggregation_keywords', {
                'how many': 'COUNT', 'how much': 'VALUE', 'number of': 'COUNT', 'count': 'COUNT',
                'average': 'AVG', 'mean': 'AVG', 'total': 'SUM', 'sum': 'SUM',
                'maximum': 'MAX', 'max': 'MAX', 'highest': 'MAX', 'largest': 'MAX', 'most': 'MAX',
                'minimum': 'MIN', 'min': 'MIN', 'lowest': 'MIN', 'smallest': 'MIN', 'least': 'MIN',
            }),
            time_patterns=pm_cfg.get('time_patterns', []),
            english_time_patterns=pm_cfg.get('english_time_patterns', []),
            similarity_threshold=pm_cfg.get('similarity_threshold', 0.3),
            min_pattern_count=pm_cfg.get('min_pattern_count', 1)
        )

        # dataset=bird 时使用 BIRD 专属模式挖掘模式（默认 generic）
        if dataset_name == 'bird':
            pattern_mining.mode = bird.pattern_mining_mode

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
            use_role_parser=abl_cfg.get('use_role_parser', True),
            use_role_weight=abl_cfg.get('use_role_weight', True)
        )

        # 解析注解通道配置
        ann_cfg = cfg.get('annotations', {})
        annotations = AnnotationsConfig(
            use_history=ann_cfg.get('use_history', False),
            use_transfer=ann_cfg.get('use_transfer', False),
            use_annotation_index=ann_cfg.get('use_annotation_index', True)
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
            annotations=annotations,
            evaluation=evaluation,
            output=output,
            tokenizer=tokenizer
        )

    @staticmethod
    def _parse_bird_config(bird_cfg: Dict[str, Any], root_dir: Path) -> BirdDatasetConfig:
        """解析 BIRD 数据集配置（相对路径基于 config.yaml 目录 / bird.root_dir 解析）"""
        if not bird_cfg:
            return BirdDatasetConfig()

        root = _resolve_path(bird_cfg.get('root_dir', ''), root_dir)

        def _parse_split(split_cfg: Dict[str, Any]) -> BirdSplitConfig:
            split_cfg = split_cfg or {}
            return BirdSplitConfig(
                json=_resolve_under_root(split_cfg.get('json', ''), root),
                tables_json=_resolve_under_root(split_cfg.get('tables_json', ''), root),
                databases_dir=_resolve_under_root(split_cfg.get('databases_dir', ''), root),
            )

        split_cfg = bird_cfg.get('split', {})
        schema_cfg = bird_cfg.get('schema', {})
        debug_cfg = bird_cfg.get('debug', {})
        pm_cfg = bird_cfg.get('pattern_mining', {})

        return BirdDatasetConfig(
            root_dir=root,
            sql_dialect=bird_cfg.get('sql_dialect', 'sqlite'),
            source_split=bird_cfg.get('source_split', 'dev'),
            train=_parse_split(bird_cfg.get('train', {})),
            dev=_parse_split(bird_cfg.get('dev', {})),
            test=_parse_split(bird_cfg.get('test', {})),
            processed_dir=_resolve_path(bird_cfg.get('processed_dir', ''), root_dir),
            use_evidence=bird_cfg.get('use_evidence', True),
            use_database_description=bird_cfg.get('use_database_description', True),
            use_value_description=bird_cfg.get('use_value_description', True),
            evaluation_mode=bird_cfg.get('evaluation_mode', 'per_database'),
            split=BirdSplitStrategyConfig(
                strategy=split_cfg.get('strategy', 'within_database'),
                train_ratio=split_cfg.get('train_ratio', 0.70),
                dev_ratio=split_cfg.get('dev_ratio', 0.10),
                test_ratio=split_cfg.get('test_ratio', 0.20),
                min_queries_per_db=split_cfg.get('min_queries_per_db', 5),
            ),
            schema=BirdSchemaConfig(
                prefix_table_with_db=schema_cfg.get('prefix_table_with_db', True),
                include_pk_fk=schema_cfg.get('include_pk_fk', True),
            ),
            pattern_mining_mode=pm_cfg.get('mode', 'generic') if isinstance(pm_cfg, dict) else 'generic',
            debug=BirdDebugConfig(
                max_queries=debug_cfg.get('max_queries', None),
            ),
        )

    def with_data_paths(self, field_csv: str, table_csv: str, qa_csv: str = None,
                        artifacts_dir: str = None) -> 'TERAGConfig':
        """
        返回替换数据路径后的配置浅拷贝（用于 BIRD per-database 子实验等场景）

        Args:
            field_csv: 字段 CSV 路径
            table_csv: 表 CSV 路径
            qa_csv: QA CSV 路径（可选）
            artifacts_dir: artifacts 输出目录（可选，用于隔离不同子实验的中间产物）

        Returns:
            新的 TERAGConfig 实例
        """
        import copy
        new_config = copy.copy(self)
        new_config.data = copy.copy(self.data)
        new_config.data.field_csv = field_csv
        new_config.data.table_csv = table_csv
        if qa_csv is not None:
            new_config.data.qa_csv = qa_csv
        if artifacts_dir is not None:
            new_config.output = copy.copy(self.output)
            new_config.output.artifacts_dir = artifacts_dir
        return new_config

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        bird_dict = None
        if self.data.bird is not None:
            b = self.data.bird
            bird_dict = {
                'root_dir': b.root_dir,
                'sql_dialect': b.sql_dialect,
                'source_split': b.source_split,
                'processed_dir': b.processed_dir,
                'use_evidence': b.use_evidence,
                'use_database_description': b.use_database_description,
                'use_value_description': b.use_value_description,
                'evaluation_mode': b.evaluation_mode,
                'pattern_mining_mode': b.pattern_mining_mode,
            }

        return {
            'version': self.version,
            'seed': self.seed,
            'data': {
                'dataset': self.data.dataset,
                'field_csv': self.data.field_csv,
                'table_csv': self.data.table_csv,
                'qa_csv': self.data.qa_csv,
                'train_ratio': self.data.train_ratio,
                'dev_ratio': self.data.dev_ratio,
                'test_ratio': self.data.test_ratio,
                'splits_dir': self.data.splits_dir,
                'bird': bird_dict,
            },
            'tokenizer': {
                'default': self.tokenizer.default,
                'datasets': self.tokenizer.datasets,
                'language': self.tokenizer.language,
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
                'mode': self.pattern_mining.mode,
                'entity_keywords': self.pattern_mining.entity_keywords,
                'attribute_keywords': self.pattern_mining.attribute_keywords,
                'aggregation_keywords': self.pattern_mining.aggregation_keywords,
                'english_aggregation_keywords': self.pattern_mining.english_aggregation_keywords,
                'time_patterns': self.pattern_mining.time_patterns,
                'english_time_patterns': self.pattern_mining.english_time_patterns,
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
            'annotations': {
                'use_history': self.annotations.use_history,
                'use_transfer': self.annotations.use_transfer,
                'use_annotation_index': self.annotations.use_annotation_index,
            },
            'ablation': {
                'use_graph_weight': self.ablation.use_graph_weight,
                'use_template_mining': self.ablation.use_template_mining,
                'use_pattern_generalization': self.ablation.use_pattern_generalization,
                'use_enhanced_index': self.ablation.use_enhanced_index,
                'use_role_parser': self.ablation.use_role_parser,
                'use_role_weight': getattr(self.ablation, 'use_role_weight', True),
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

    @property
    def sql_dialect(self) -> str:
        """当前数据集的 SQL 方言（PowerGrid=mysql，BIRD=sqlite）"""
        if self.data.dataset == 'bird' and self.data.bird is not None:
            return self.data.bird.sql_dialect
        return 'mysql'

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


def _resolve_under_root(path: str, root: str) -> str:
    """解析 BIRD 子路径：绝对路径直接返回，相对路径基于 BIRD root_dir 解析"""
    if not path or not root:
        return path
    if os.path.isabs(path):
        return path
    return str((Path(root) / path).resolve())
