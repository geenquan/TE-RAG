"""
受约束模板与领域注解迁移（P2-D2，原论文 Eq 5–6 的可实现版本）

Eq 5 element_similarity(source, target, annotations, schema)：
- template_overlap：源/目标**直接注解（B 类）**的模板与术语文本集合的 Jaccard。
  任一侧无直接注解 → 该分量 None（不把两个空集合算成完全相似），总分退化为
  SCHEMA_ONLY_WEIGHT × schema_similarity（仍可仅凭合法 schema 证据判断）。
- schema_similarity：列名 token Jaccard、列描述 token Jaccard、表业务上下文
  （表名+表描述 token）Jaccard 的加权平均（0.3/0.4/0.3），全部只用公开
  schema 元数据；列类型兼容（数值/字符串/日期三族）是必要门槛而非相似度来源。
- total = 0.5×template_overlap + 0.5×schema_similarity（两侧均有模板证据），
  或 SCHEMA_ONLY_WEIGHT(=0.5)×schema_similarity（缺模板证据，标记 degraded）。
  与论文 Eq 5 的对应：论文为 template_overlap + schema_similarity 未定义
  归一化；本实现把两项各自归一到 [0,1] 后等权相加——实现选择，非论文已验证
  结论。

Eq 6 adapt_template(annotation, source, target, evidence)：
按注解类型分别适配/拒绝（TERM/TEMPLATE/PREDICATE；ROLE 只作兼容证据，
不作为可迁移注解）。拒绝必须给原因；不存在"返回原文本即算适配"的路径。

transfer_annotations(direct_annotations, schema, config)：
一跳、不自迁移、目标=无直接注解证据的稀疏元素、全库 schema 可见但不使用
任何 dev/test 目标表标签；固定上限（源数/每目标注解数/每表注入文本数）；
与直接注解冲突时直接证据优先；注入文本按 (text) 去重（多源同文本只保留
相似度最高源，重复不增加词频）。全部参数见 TransferConfig（运行前固定）。
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from terag.tokenization import tokenize_text

METHOD_VERSION = 'transfer_v3_target_cap'


@dataclass
class TransferConfig:
    """迁移参数（运行前固定；不以结果调参）"""
    min_total_sim: float = 0.5          # 接受迁移的最低总分
    schema_only_min: float = 0.5        # 无模板证据时 schema 相似度单独门槛
    context_min: float = 0.3            # TERM 迁移的表业务上下文相似度门槛
    max_sources_per_target: int = 3     # 每目标候选源上限
    max_annotations_per_target: int = 3  # 每目标接受注解上限
    max_texts_per_table: int = 8        # 每表注入索引文本上限
    template_role_required = ('SELECT', 'WHERE')  # 模板必要绑定的角色


@dataclass
class SchemaInfo:
    """公开 schema 元数据（不含任何测试标签/业务裁决）"""
    columns: Dict[str, List[Dict]]  # table -> [{field_name, field_desc, field_type}]
    table_desc: Dict[str, str]

    def col_info(self, table: str, column: str) -> Optional[Dict]:
        for c in self.columns.get(table, []):
            if str(c['field_name']).lower() == column.lower():
                return c
        return None

    def tables(self) -> List[str]:
        return sorted(self.columns.keys())


_TOKEN_CACHE = {}


def _tokens(text: str) -> set:
    """分词缓存（迁移需对全库 14 万列重复分词，必须记忆化）。"""
    key = str(text or '')
    if key not in _TOKEN_CACHE:
        _TOKEN_CACHE[key] = {w for w in tokenize_text(key, 'chinese')
                             if len(w) > 1}
    return _TOKEN_CACHE[key]


def _type_family(t: str) -> str:
    t = str(t or '').lower()
    if any(k in t for k in ('int', 'double', 'float', 'decimal', 'numeric', 'bigint')):
        return 'numeric'
    if any(k in t for k in ('date', 'time')):
        return 'date'
    return 'string'


def _jaccard(a: set, b: set) -> Optional[float]:
    if not a and not b:
        return None  # 双空不算相似
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


# ------------------------------------------------------------------ Eq 5

def element_similarity(source: Tuple[str, str], target: Tuple[str, str],
                       annotations: List[Dict], schema: SchemaInfo) -> Dict:
    """原论文 Eq 5 的实现。source/target = (table, column)（列级元素）。

    Returns: {template_overlap, schema_similarity, total, degraded,
              evidence, non_computable}
    """
    return _similarity_from_texts(source, target,
                                  _direct_texts(source, annotations),
                                  _direct_texts(target, annotations), schema)


def _similarity_from_texts(source, target, src_texts, tgt_texts, schema,
                           col_lookup=None):
    """Shared exact semantics; cached and standalone paths differ only in lookup."""
    non_computable = []
    t_ov = _jaccard(src_texts, tgt_texts)
    if t_ov is None:
        t_ov = None
        non_computable.append('template_overlap: 双侧均无直接注解证据')
    elif not src_texts:
        t_ov = None
        non_computable.append('template_overlap: 源无直接注解')
    elif not tgt_texts:
        t_ov = None
        non_computable.append('template_overlap: 目标无直接注解（未见表常态）')

    lookup = col_lookup or schema.col_info
    s, c = lookup(*source), lookup(*target)
    name_j = _jaccard(_tokens(s['field_name']) if s else set(),
                      _tokens(c['field_name']) if c else set())
    desc_j = _jaccard(_tokens(s.get('field_desc')) if s else set(),
                      _tokens(c.get('field_desc')) if c else set())
    ctx_j = _jaccard(_tokens(schema.table_desc.get(source[0])),
                     _tokens(schema.table_desc.get(target[0])))
    parts = [name_j, desc_j, ctx_j]
    if any(p is None for p in parts):
        non_computable.append('schema_similarity: 存在空元数据分量（按 0 计）')
    schema_sim = sum((p if p is not None else 0.0) * w
                     for p, w in zip(parts, (0.3, 0.4, 0.3)))

    type_ok = bool(s and c and _type_family(s.get('field_type'))
                   == _type_family(c.get('field_type')))
    if t_ov is not None:
        total = 0.5 * t_ov + 0.5 * schema_sim
        degraded = False
    else:
        total = 0.5 * schema_sim
        degraded = True
    return {'template_overlap': t_ov, 'schema_similarity': round(schema_sim, 4),
            'total': round(total, 4), 'degraded': degraded,
            'type_compatible': type_ok,
            'evidence': {'name_j': name_j, 'desc_j': desc_j, 'ctx_j': ctx_j,
                         'src_n_texts': len(src_texts), 'tgt_n_texts': len(tgt_texts)},
            'non_computable': non_computable}


def _direct_texts(element: Tuple[str, str], annotations: List[Dict]) -> set:
    """元素的直接（B 类）注解文本：TERM 术语 + TEMPLATE 模板（表级模板
    归属其绑定列）。只统计 status=confirmed 且 target 匹配的注解。"""
    key = f"{element[0].lower()}.{element[1].lower()}"
    tkey = element[0].lower()
    texts = set()
    for a in annotations:
        if a.get('status') != 'confirmed':
            continue
        if a['type'] == 'TERM' and a.get('target') == key:
            texts.add(str(a['content']))
        elif a['type'] == 'TEMPLATE' and a.get('target', '').lower() == tkey:
            for bind_list in a.get('bindings') or []:
                if any(b[0] == key for b in bind_list):
                    texts.add(str(a['content']))
    return texts


# ------------------------------------------------------------------ Eq 6

def _direct_text_index(annotations):
    """Call-local equivalent of _direct_texts, including bound table templates."""
    result = {}
    for a in annotations:
        if a.get('status') != 'confirmed':
            continue
        if a['type'] == 'TERM':
            key = a.get('target', '')
            if key == key.lower():
                result.setdefault(key, set()).add(str(a['content']))
        elif a['type'] == 'TEMPLATE':
            table = a.get('target', '').lower()
            for bindings in a.get('bindings') or []:
                for key, _role in bindings:
                    if key == key.lower() and key.rsplit('.', 1)[0] == table:
                        result.setdefault(key, set()).add(str(a['content']))
    return result


class _SimilarityCache:
    """Per-transfer-call caches; no train-fold decision state is shared."""
    def __init__(self, annotations, schema):
        self.schema = schema
        self.texts = _direct_text_index(annotations)
        self.columns = {}

    def direct_texts(self, element):
        return self.texts.get(f'{element[0].lower()}.{element[1].lower()}', set())

    def col_info(self, table, column):
        key = (table, str(column).lower())
        if key not in self.columns:
            self.columns[key] = self.schema.col_info(table, column)
        return self.columns[key]

    def similarity(self, source, target):
        return _similarity_from_texts(source, target, self.direct_texts(source),
                                     self.direct_texts(target), self.schema,
                                     self.col_info)


def adapt_template(annotation: Dict, source: Tuple[str, str],
                   target: Tuple[str, str], evidence: Dict,
                   schema: SchemaInfo,
                   config: 'TransferConfig' = None) -> Dict:
    """按注解类型适配或拒绝（原论文 Eq 6）。

    P2-D3 修复：
    - TERM 已被目标字段元数据覆盖（术语出现在目标列名/描述或表名/描述中）
      → 拒绝（metadata_covered），不重复注入。
    - TEMPLATE 增加表业务上下文门槛（与 TERM 同规则），不能仅凭字段名
      可映射就跨业务表复制模板。
    - 显式接收并使用传入的 TransferConfig（不再读类级默认值）。

    Returns: {accepted, adapted_content, field_mapping, reason, slots}
    """
    cfg = config or TransferConfig()
    a_type = annotation['type']
    src_col = schema.col_info(*source)
    tgt_col = schema.col_info(*target)

    if not tgt_col:
        return _rej('目标字段不在 schema 中')
    if not evidence['type_compatible']:
        return _rej(f"类型不兼容（{_type_family(src_col.get('field_type'))} vs "
                    f"{_type_family(tgt_col.get('field_type'))}）")

    def _ctx_gate():
        ctx = evidence['evidence']['ctx_j']
        if ctx is None or ctx < cfg.context_min:
            return (f"表业务上下文不兼容（ctx_j={ctx} < {cfg.context_min}）"
                    f"——结构类似≠业务含义相同")
        return None

    def _metadata_covered(text: str) -> bool:
        tgt_local = (_tokens(tgt_col['field_name'])
                     | _tokens(tgt_col.get('field_desc') or '')
                     | _tokens(schema.table_desc.get(target[0]) or ''))
        return text in tgt_local

    if a_type == 'TERM':
        reason = _ctx_gate()
        if reason:
            return _rej(reason)
        term = str(annotation['content'])
        if _metadata_covered(term):
            return _rej(f"术语 '{term}' 已被目标字段/表元数据覆盖，"
                        f"不重复注入（metadata_covered）")
        return {'accepted': True, 'adapted_content': term,
                'field_mapping': {f"{source[0]}.{source[1]}":
                                   f"{target[0]}.{target[1]}"},
                'reason': f"TERM 迁移：ctx_j={evidence['evidence']['ctx_j']:.2f}, "
                          f"total={evidence['total']}",
                'slots': None}

    if a_type == 'TEMPLATE':
        reason = _ctx_gate()
        if reason:
            return _rej(reason)
        binds = annotation.get('bindings') or []
        if not binds:
            return _rej('模板无绑定结构（D1 记录缺失），无法验证必要字段映射')
        # 必要绑定（SELECT/WHERE 角色）逐一映射到目标表
        required = set()
        for bind_list in binds:
            for (el, role) in bind_list:
                if role in cfg.template_role_required:
                    required.add(el)
        if not required:
            return _rej('模板绑定不含 SELECT/WHERE 角色，无必要元素可映射')
        mapping = {}
        missing = []
        for el in required:
            el_t, el_c = el.split('.', 1)
            cand = schema.col_info(el_t, el_c)
            if cand is None:
                missing.append(el)
                continue
            # 目标表内寻找同类型 + 名称/描述最相似的列
            best, best_sim = None, -1.0
            for tc in schema.columns.get(target[0], []):
                sim = (_jaccard(_tokens(cand['field_name']) | _tokens(cand.get('field_desc')),
                                _tokens(tc['field_name']) | _tokens(tc.get('field_desc')))
                       or 0.0)
                if _type_family(tc.get('field_type')) != _type_family(cand.get('field_type')):
                    sim = 0.0
                if sim > best_sim:
                    best, best_sim = tc, sim
            if best is None or best_sim <= 0.0:
                missing.append(el)
            else:
                mapping[el] = f"{target[0]}.{best['field_name']}"
        if missing:
            return _rej(f"必要字段未映射到目标 schema: {sorted(missing)}")
        # 槽位已归一（#DATE/#NUM/#ORG），不复制源业务实体/固定约束
        return {'accepted': True,
                'adapted_content': str(annotation['content']),
                'field_mapping': mapping,
                'reason': f"TEMPLATE 迁移：{len(mapping)} 个必要字段已映射",
                'slots': {'note': '模板文本槽位已在 D1 归一化，含源实体常量的'
                                  '整句不经适配直接复制不存在'}}

    if a_type == 'PREDICATE':
        content = dict(annotation['content'])
        vc = content.get('value_class')
        op = content.get('operator')
        if vc == 'CODE':
            return _rej('源 CODE 码值缺少目标码表/语义证据，具体码值不迁移')
        if vc == 'UNKNOWN':
            return _rej('UNKNOWN 值类型不得转换为确定业务规则')
        if vc == 'NULL':
            adapted = {'operator': op, 'value_class': vc, 'value': None}
        elif vc in ('DATE', 'ENTITY'):
            adapted = {'operator': op, 'value_class': vc, 'value': None}
        elif vc == 'THRESHOLD':
            if _type_family(tgt_col.get('field_type')) != 'numeric':
                return _rej('THRESHOLD 目标列非数值类型')
            adapted = {'operator': op, 'value_class': vc, 'value': None}
        else:
            return _rej(f'未支持的值类型 {vc}')
        return {'accepted': True, 'adapted_content': adapted,
                'field_mapping': {f"{source[0]}.{source[1]}":
                                   f"{target[0]}.{target[1]}"},
                'reason': f"PREDICATE 迁移：{vc} 保留为槽位（不复制训练常量）",
                'slots': adapted}

    return _rej(f'注解类型 {a_type} 不可迁移（ROLE 仅作兼容证据）')


def _rej(reason: str) -> Dict:
    return {'accepted': False, 'adapted_content': None,
            'field_mapping': None, 'reason': reason, 'slots': None}


# ------------------------------------------------------------------ 主入口

def _stable_id(*parts) -> str:
    """稳定摘要 ID（不依赖 Python 随机化 hash）。"""
    import hashlib
    return 'c' + hashlib.sha1(
        '|'.join(str(p) for p in parts).encode('utf-8')).hexdigest()[:10]


def transfer_annotations(direct_annotations: List[Dict], schema: SchemaInfo,
                         config: TransferConfig = None) -> Dict:
    """一跳受约束迁移。返回 {accepted, rejected, provenance}。

    目标候选 = 无 confirmed TERM 直接证据的列（稀疏规则，仅用 schema+B，
    不接收 dev/test 目标表标签）；源 = 有 confirmed TERM 直接证据的列。

    P2-F 性能优化；M-03 修正后共享 helper 语义，不能再声称与旧正式接受集等价：
    - 直接注解一次性按 源列/源表 建倒排索引（不再每对扫描全部注解）；
    - TERM 与绑定 TEMPLATE 文本一次索引，目标无 TERM 不等于无 TEMPLATE；
    - col_info 查找结果按 (table,col) 就地缓存（调用级，
      不跨实例/折共享决策状态；_TOKEN_CACHE 为纯字符串→分词的确定性缓存）；
    - 拒绝记录边计算边按原因计数+限量采样（不先累积数百万字典）；
    - 预筛（token 无交集跳过）与精算公式等价：无交集 → 各 Jaccard=0 →
      且无共享直接文本时 total=0；有共享模板时不可跳过。
    - annotation_id 用稳定 sha1 摘要。
    """
    cfg = config or TransferConfig()
    similarity_cache = _SimilarityCache(direct_annotations, schema)

    # ---- 一次性索引：源集合 / 稀疏判定 / 注解倒排 / 直接文本 ----
    sources = set()
    for a in direct_annotations:
        if a.get('status') == 'confirmed' and a['type'] == 'TERM' \
                and a.get('target') and '.' in a['target']:
            sources.add(tuple(a['target'].split('.', 1)))
    has_term = {a['target'] for a in direct_annotations
                if a['type'] == 'TERM' and a.get('status') == 'confirmed'}
    anns_by_src_key = {}      # 't.c' -> [annotation, ...]（TERM/PREDICATE）
    templates_by_table = {}   # 't' -> [annotation, ...]
    direct_texts_by_target = {}
    for a in direct_annotations:
        if a.get('status') != 'confirmed' or not a.get('target'):
            continue
        if a['type'] in ('TERM', 'PREDICATE'):
            anns_by_src_key.setdefault(a['target'], []).append(a)
        elif a['type'] == 'TEMPLATE':
            templates_by_table.setdefault(
                str(a['target']).lower(), []).append(a)
        direct_texts_by_target.setdefault(a['target'], set()).add(
            str(a['content']))

    # ---- 每源元素预计算（直接注解文本 / 列 token / 上下文 token）----
    src_texts = {}
    src_col_tokens = {}
    src_ctx_tokens_map = {}
    for src in sources:
        src_texts[src] = similarity_cache.direct_texts(src)
        ci = schema.col_info(*src)
        if ci:
            src_col_tokens[src] = (_tokens(ci['field_name'])
                                   | _tokens(ci.get('field_desc') or ''))
        else:
            src_col_tokens[src] = set()
        src_ctx_tokens_map[src] = _tokens(schema.table_desc.get(src[0]))
    src_token_universe = set()
    for s in sources:
        src_token_universe |= src_col_tokens[s] | src_ctx_tokens_map[s]

    # ---- 拒绝记录：边算边计数+采样 ----
    reason_counts = {}
    reject_samples = []       # 每原因类别最多 REJECT_SAMPLE 条
    per_reason_kept = {}
    rejection_stages = {}
    rejected_annotation_types = {}

    def record_reject(rec, stage=None):
        category = stage or rec['status']
        rejection_stages[category] = rejection_stages.get(category, 0) + 1
        if rec.get('source_type') is not None:
            kind = rec['source_type']
            rejected_annotation_types[kind] = rejected_annotation_types.get(kind, 0) + 1
        st = rec['status']
        reason_counts[st] = reason_counts.get(st, 0) + 1
        if per_reason_kept.get(st, 0) < REJECT_SAMPLE:
            reject_samples.append(rec)
            per_reason_kept[st] = per_reason_kept.get(st, 0) + 1

    REJECT_SAMPLE = 200
    accepted = []
    accepted_per_target = {}  # physical table.column, across duplicate raw schema rows
    used_texts_by_table = {}
    n_targets_considered = 0

    # A shared direct template can qualify even without schema-token overlap.
    source_text_universe = set().union(*src_texts.values()) if src_texts else set()
    counts = {'sparse_targets': 0, 'prefilter_skipped_targets': 0,
              'candidate_pairs_scored': 0, 'similarity_passed_pairs': 0,
              'selected_source_pairs': 0, 'annotations_adapted': 0}
    adapted_types = {}
    accepted_types = {}

    for table in schema.tables():
        for col in schema.columns[table]:
            if len(used_texts_by_table.get(table, [])) >= cfg.max_texts_per_table:
                break
            col_name = str(col['field_name'])
            tgt_key = f"{table.lower()}.{col_name.lower()}"
            if tgt_key in has_term:
                continue
            if accepted_per_target.get(tgt_key, 0) >= cfg.max_annotations_per_target:
                continue
            counts['sparse_targets'] += 1
            target = (table, col_name)
            tgt_tokens = (_tokens(col_name)
                          | _tokens(str(col.get('field_desc') or ''))
                          | _tokens(schema.table_desc.get(table)))
            if not (tgt_tokens & src_token_universe) and not (
                    similarity_cache.direct_texts(target) & source_text_universe):
                counts['prefilter_skipped_targets'] += 1
                continue
            n_targets_considered += 1

            scored = []
            for src in sources:
                if src[0].lower() == table.lower() and \
                        src[1].lower() == col_name.lower():
                    continue
                counts['candidate_pairs_scored'] += 1
                ev = similarity_cache.similarity(src, target)
                if ev['degraded'] and ev['schema_similarity'] < cfg.schema_only_min:
                    record_reject(_record(
                        None, src, target, None, ev, 'rejected',
                        f"无模板证据且 schema 相似度 {ev['schema_similarity']} "
                        f"< {cfg.schema_only_min}"), stage='schema_only_gate')
                    continue
                if ev['total'] < cfg.min_total_sim:
                    record_reject(_record(
                        None, src, target, None, ev, 'rejected',
                        f"总分 {ev['total']} < {cfg.min_total_sim}"), stage='total_similarity_gate')
                    continue
                counts['similarity_passed_pairs'] += 1
                scored.append((ev['total'], src, ev))
            scored.sort(key=lambda x: (-x[0], x[1][0], x[1][1]))
            scored = scored[:cfg.max_sources_per_target]
            counts['selected_source_pairs'] += len(scored)

            n_accepted_this_target = 0
            seen_texts = set()
            for total, src, ev in scored:
                src_key = f"{src[0].lower()}.{src[1].lower()}"
                cand_anns = list(anns_by_src_key.get(src_key, [])) + \
                    list(templates_by_table.get(src[0].lower(), []))
                for a in cand_anns:
                    if a['type'] not in ('TERM', 'TEMPLATE', 'PREDICATE'):
                        continue
                    if (n_accepted_this_target >= cfg.max_annotations_per_target or
                            accepted_per_target.get(tgt_key, 0) >= cfg.max_annotations_per_target):
                        break
                    if len(used_texts_by_table.get(table, [])) >= cfg.max_texts_per_table:
                        break
                    counts['annotations_adapted'] += 1
                    adapted_types[a['type']] = adapted_types.get(a['type'], 0) + 1
                    res = adapt_template(a, src, target, ev, schema, cfg)
                    text = (res['adapted_content']
                            if isinstance(res['adapted_content'], str)
                            else None)
                    if res['accepted'] and text:
                        if (text in direct_texts_by_target.get(tgt_key, set()) or
                                text in similarity_cache.direct_texts(target)):
                            record_reject(_record(
                                a, src, target, None, ev, 'conflict_skip',
                                '与直接注解文本冲突（直接证据优先）'))
                            continue
                        if text in seen_texts or text in \
                                used_texts_by_table.get(table, set()):
                            record_reject(_record(
                                a, src, target, None, ev, 'dedup_skip',
                                '重复文本（多源同文本只保留最高相似度源）'))
                            continue
                        seen_texts.add(text)
                        used_texts_by_table.setdefault(table, set()).add(text)
                        accepted.append(_record(a, src, target, res, ev,
                                                'accepted', res['reason']))
                        accepted_types[a['type']] = accepted_types.get(a['type'], 0) + 1
                        n_accepted_this_target += 1
                        accepted_per_target[tgt_key] = accepted_per_target.get(tgt_key, 0) + 1
                    elif not res['accepted']:
                        record_reject(_record(a, src, target, res, ev,
                                              'adapt_rejected', res['reason']))
                if len(used_texts_by_table.get(table, [])) >= \
                        cfg.max_texts_per_table:
                    break

    n_rejected_total = sum(reason_counts.values())
    return {'accepted': accepted,
            'rejected': reject_samples,
            'provenance': {
                'method_version': METHOD_VERSION,
                'n_sources': len(sources),
                'decision_counts': counts,
                'adapted_annotation_types': adapted_types,
                'accepted_annotation_types': accepted_types,
                'counting_note': 'prefilter skips count targets; similarity gates count pairs; adaptation counts annotations; rejected samples are not totals',
                'n_targets_considered': n_targets_considered,
                'n_accepted': len(accepted),
                'n_rejected_total': n_rejected_total,
                'n_rejected_sampled': len(reject_samples),
                'rejected_reasons': reason_counts,  # historical status-level counts
                'rejection_stages': rejection_stages,
                'rejected_annotation_types': rejected_annotation_types,
                'untyped_pair_gate_rejections': (rejection_stages.get('schema_only_gate', 0)
                                                 + rejection_stages.get('total_similarity_gate', 0)),
                'config': cfg.__dict__,
                'one_hop': True,
                'source_qids': sorted({q for r in accepted
                                       for q in r['source_qids']})}}


def _record(src_ann, source, target, res, ev, status, reason) -> Dict:
    return {
        'annotation_id': _stable_id(source, target, status, reason,
                                    (src_ann or {}).get('annotation_id')),
        'source_element': f"{source[0]}.{source[1]}",
        'target_element': f"{target[0]}.{target[1]}",
        'source_annotation_id': (src_ann or {}).get('annotation_id'),
        'source_type': (src_ann or {}).get('type'),
        'source_qids': (src_ann or {}).get('source_qids', []),
        'original_content': (src_ann or {}).get('content'),
        'adapted_content': (res or {}).get('adapted_content'),
        'field_mapping': (res or {}).get('field_mapping'),
        'constraint_evidence': {
            'type_compatible': ev.get('type_compatible'),
            'components': dict(ev.get('evidence') or {}),
            'non_computable': list(ev.get('non_computable') or [])},
        'similarity': {'template_overlap': ev.get('template_overlap'),
                       'schema_similarity': ev.get('schema_similarity'),
                       'total': ev.get('total'), 'degraded': ev.get('degraded')},
        'status': status, 'reason': reason,
        'method_version': METHOD_VERSION,
    }


def _reason_counts(rejected: List[Dict]) -> Dict[str, int]:
    out = {}
    for r in rejected:
        key = r['status']
        out[key] = out.get(key, 0) + 1
    return out


# ------------------------------------------------------------------ 注入通道

def transfer_index_texts(transfer_result: Dict,
                         pattern_weight: float) -> Dict[str, List[Tuple[str, float]]]:
    """C 类注解 → 表级索引文本（accepted + status=accepted + 字符串内容）。"""
    from collections import defaultdict
    out = defaultdict(list)
    for r in transfer_result['accepted']:
        if r['status'] == 'accepted' and isinstance(r['adapted_content'], str):
            t = r['target_element'].split('.', 1)[0]
            out[t].append((r['adapted_content'], pattern_weight))
    return dict(out)


def transfer_column_bonus_map(transfer_result: Dict) -> Dict[str, List[str]]:
    """C 类 TERM → {target_key: [term, ...]}（列排序通道，w_ann 沿用 D1）。"""
    from collections import defaultdict
    out = defaultdict(list)
    for r in transfer_result['accepted']:
        if r['status'] == 'accepted' and r.get('source_type') == 'TERM' \
                and isinstance(r['adapted_content'], str):
            out[r['target_element'].lower()].append(r['adapted_content'])
    return dict(out)
