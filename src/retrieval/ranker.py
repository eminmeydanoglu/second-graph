"""Deterministic ranking logic for recall candidates."""

from dataclasses import replace

from ..graph.schema import get_node_types
from .config import DEFAULT_RETRIEVAL_CONFIG, RetrievalConfig
from .tokenize import keyword_overlap_score, normalize_whitespace
from .types import ScoredNode

_UNKNOWN_NODE_TYPE = "Unknown"
_UNREACHABLE_HOP = 99


def clamp_01(value: float | int | None) -> float:
    """Clamp a numeric score into [0, 1]."""
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    if score < 0.0:
        return 0.0
    if score > 1.0:
        return 1.0
    return score


def canonical_node_type_from_labels(labels: list[str] | None) -> str:
    """Pick canonical schema node type from labels."""
    if not labels:
        return _UNKNOWN_NODE_TYPE

    valid_types = get_node_types()
    for node_type in valid_types:
        if node_type in labels:
            return node_type

    for label in labels:
        if label != "Entity":
            return label
    return _UNKNOWN_NODE_TYPE


def _display_name(raw: dict) -> str:
    for key in ("name", "id", "node_id"):
        value = raw.get(key)
        if isinstance(value, str) and value.strip():
            return normalize_whitespace(value)
    return "unknown"


def _summary(raw: dict) -> str | None:
    value = raw.get("summary")
    if not isinstance(value, str):
        return None
    cleaned = normalize_whitespace(value)
    return cleaned or None


def _node_sort_key(node: ScoredNode) -> tuple[float, float, float, int, str]:
    return (
        -node.final_score,
        -node.vector_score,
        -node.keyword_score,
        node.hop_distance,
        node.node_id,
    )


def sort_scored_nodes(nodes: list[ScoredNode]) -> list[ScoredNode]:
    """Sort nodes with deterministic tie-break rules."""
    return sorted(nodes, key=_node_sort_key)


def graph_bonus_for_hop(
    hop_distance: int,
    config: RetrievalConfig = DEFAULT_RETRIEVAL_CONFIG,
) -> float:
    """Graph locality score by hop distance."""
    if hop_distance == 0:
        return config.graph_bonus_anchor
    if hop_distance == 1:
        return config.graph_bonus_hop1
    if hop_distance == 2:
        return config.graph_bonus_hop2
    return 0.0


def build_scored_candidates(
    query: str,
    *,
    vector_hits: list[dict],
    lexical_hits: list[dict],
    config: RetrievalConfig = DEFAULT_RETRIEVAL_CONFIG,
) -> list[ScoredNode]:
    """Merge vector/lexical hits and compute initial fused scores."""
    merged: dict[str, dict] = {}

    for hit in vector_hits:
        node_id = str(hit.get("node_id") or hit.get("id") or "").strip()
        if not node_id:
            continue

        entry = merged.setdefault(
            node_id,
            {
                "node_id": node_id,
                "node_type": _UNKNOWN_NODE_TYPE,
                "name": node_id,
                "summary": None,
                "vector_score": 0.0,
            },
        )

        entry["vector_score"] = max(
            entry["vector_score"], clamp_01(hit.get("score", 0.0))
        )

        node_type = hit.get("node_type")
        if isinstance(node_type, str) and node_type.strip():
            entry["node_type"] = normalize_whitespace(node_type)

        entry["name"] = _display_name(hit)
        summary = _summary(hit)
        if summary:
            entry["summary"] = summary

    for hit in lexical_hits[: config.lexical_cap]:
        node_id = str(hit.get("id") or hit.get("node_id") or "").strip()
        if not node_id:
            continue

        entry = merged.setdefault(
            node_id,
            {
                "node_id": node_id,
                "node_type": _UNKNOWN_NODE_TYPE,
                "name": node_id,
                "summary": None,
                "vector_score": 0.0,
            },
        )

        if entry["node_type"] == _UNKNOWN_NODE_TYPE:
            raw_type = hit.get("node_type")
            if isinstance(raw_type, str) and raw_type.strip():
                entry["node_type"] = normalize_whitespace(raw_type)
            else:
                entry["node_type"] = canonical_node_type_from_labels(hit.get("_labels"))

        if entry["name"] == node_id:
            entry["name"] = _display_name(hit)

        summary = _summary(hit)
        if summary and not entry["summary"]:
            entry["summary"] = summary

    scored_nodes: list[ScoredNode] = []
    for payload in merged.values():
        keyword_text = payload["name"]
        if payload["summary"]:
            keyword_text = f"{keyword_text} {payload['summary']}"

        vector_score = clamp_01(payload["vector_score"])
        keyword_score = keyword_overlap_score(query, keyword_text)
        final_score = (
            config.vector_weight * vector_score + config.keyword_weight * keyword_score
        )

        scored_nodes.append(
            ScoredNode(
                node_id=payload["node_id"],
                node_type=payload["node_type"],
                name=payload["name"],
                summary=payload["summary"],
                vector_score=vector_score,
                keyword_score=keyword_score,
                graph_score=0.0,
                final_score=final_score,
                hop_distance=_UNREACHABLE_HOP,
            )
        )

    return sort_scored_nodes(scored_nodes)


def select_anchors(
    nodes: list[ScoredNode],
    *,
    anchor_count: int,
) -> list[ScoredNode]:
    """Select top anchors from ranked nodes."""
    if anchor_count <= 0:
        return []
    return nodes[:anchor_count]


def apply_graph_locality(
    nodes: list[ScoredNode],
    *,
    hop_by_node: dict[str, int],
    config: RetrievalConfig = DEFAULT_RETRIEVAL_CONFIG,
) -> list[ScoredNode]:
    """Recompute final score after graph expansion locality bonuses."""
    enriched: list[ScoredNode] = []

    for node in nodes:
        hop_distance = hop_by_node.get(node.node_id, _UNREACHABLE_HOP)
        graph_score = graph_bonus_for_hop(hop_distance, config)
        final_score = (
            config.vector_weight * node.vector_score
            + config.keyword_weight * node.keyword_score
            + config.graph_weight * graph_score
        )

        enriched.append(
            replace(
                node,
                graph_score=graph_score,
                final_score=final_score,
                hop_distance=hop_distance,
            )
        )

    return sort_scored_nodes(enriched)
