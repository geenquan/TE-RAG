"""Vnext feature-extractor subclass (route M): audit-backed D6 fix only.

D6 (registry/DEFECT_LIST.md, severity M/H): base _compute_direct_match_score
adds a 0.4 bonus per query term SUBSTRING-contained in the raw field name or
prose description ('the'/'of'/'and' all match any prose description), and
divides the overlap by len(query_terms) WITH duplicates. On BIRD every
description is prose, so the bonus is near-constant noise across columns and
the duplicated denominator compresses long questions.

Fix (marked '# VNEXT: D6'): bonus fires only on exact token membership in the
field's token set; denominator uses the deduplicated query-term count.
Parameters (multiplier, bonus) unchanged; behavior differs only where the
base was defect-reproduced.
"""
from terag.feature_extractor import FeatureExtractor


class FeatureExtractorVnext(FeatureExtractor):
    """FeatureExtractor with the D6 direct-match fix."""

    extractor_id = 'feature_extractor_vnext_v1(D6)'

    def _compute_direct_match_score(self, field_name, field_desc,
                                    query_terms, query_term_set) -> float:
        field_text = f"{field_name} {field_desc}"
        field_terms = set(t for t in self._tokenize(field_text) if len(t) > 1)

        overlap = len(query_term_set & field_terms)
        # VNEXT: D6 — deduplicated denominator (base divided by the list length
        # including duplicate query terms)
        s_direct = overlap / max(len(query_term_set), 1) * self.direct_match_multiplier

        # VNEXT: D6 — token-membership bonus instead of raw substring containment
        for term in query_term_set:
            if term in field_terms:
                s_direct += self.field_name_match_bonus

        return s_direct
