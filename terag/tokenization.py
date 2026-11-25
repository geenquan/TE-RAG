"""
分词工具

提供统一的中文 / 英文分词接口：
- chinese: jieba 分词（PowerGrid 中文场景）
- english: 轻量正则分词（BIRD 英文场景），lowercase + 词元化 + 保留数字 + 处理下划线/连字符/括号

不引入额外重量级 NLP 依赖。
"""

import re
from typing import List

# 英文分词：小写化后提取字母数字词元（下划线、连字符、括号、标点均作为分隔符，
# 从而 snake_case / kebab-case 字段名（如 expense_description、K-12）能被合理拆分）
_EN_WORD_RE = re.compile(r"[A-Za-z0-9]+")


def tokenize_english(text: str) -> List[str]:
    """
    英文分词

    Args:
        text: 输入文本

    Returns:
        小写词元列表，例如 "Free Meal Count (K-12)" -> ['free', 'meal', 'count', 'k', '12']
    """
    if not text:
        return []
    return _EN_WORD_RE.findall(str(text).lower())


def tokenize_chinese(text: str) -> List[str]:
    """中文分词（jieba）"""
    import jieba
    if not text:
        return []
    return list(jieba.cut(str(text)))


def tokenize_text(text: str, language: str = "chinese") -> List[str]:
    """
    按语言分词

    Args:
        text: 输入文本
        language: "chinese"（jieba）或 "english"（正则词元化）

    Returns:
        词元列表
    """
    if language == 'english':
        return tokenize_english(text)
    return tokenize_chinese(text)


def tokenize_set(text: str, language: str = "chinese", min_len: int = 1) -> set:
    """
    按语言分词并转为集合（可选过滤单字符词元）

    Args:
        text: 输入文本
        language: 分词语言
        min_len: 最小词元长度（沿用项目惯例 len(t) > 1 时传 min_len=2）

    Returns:
        词元集合
    """
    return {t for t in tokenize_text(text, language) if len(t) >= min_len}
