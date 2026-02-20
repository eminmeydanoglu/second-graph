"""Deterministic tokenization and lexical scoring."""

from collections import Counter
import re

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_WS_RE = re.compile(r"\s+")


def normalize_whitespace(text: str | None) -> str:
    """Collapse whitespace to single spaces."""
    if text is None:
        return ""
    return _WS_RE.sub(" ", text.replace("\n", " ").replace("\t", " ")).strip()


def tokenize(text: str | None) -> tuple[str, ...]:
    """Tokenize text deterministically (lowercase alnum tokens)."""
    normalized = normalize_whitespace(text).lower()
    return tuple(_TOKEN_RE.findall(normalized))


def keyword_overlap_score(query: str, candidate: str) -> float:
    """Compute deterministic F1-style token overlap in [0, 1]."""
    q_tokens = tokenize(query)
    c_tokens = tokenize(candidate)

    if not q_tokens or not c_tokens:
        return 0.0

    q_counter = Counter(q_tokens)
    c_counter = Counter(c_tokens)
    overlap = sum(
        min(q_counter[t], c_counter[t]) for t in (q_counter.keys() & c_counter.keys())
    )

    if overlap == 0:
        return 0.0

    precision = overlap / len(c_tokens)
    recall = overlap / len(q_tokens)
    denom = precision + recall
    if denom == 0:
        return 0.0

    score = 2.0 * precision * recall / denom
    if score < 0:
        return 0.0
    if score > 1:
        return 1.0
    return score
