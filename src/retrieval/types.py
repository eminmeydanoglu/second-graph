"""Typed contracts for deterministic recall pipeline."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ScoredNode:
    node_id: str
    node_type: str
    name: str
    summary: str | None
    vector_score: float
    keyword_score: float
    graph_score: float
    final_score: float
    hop_distance: int


@dataclass(frozen=True)
class RankedConnection:
    from_id: str
    from_name: str
    relation: str
    to_id: str
    to_name: str
    direction: str
    support_hop: int
    rank_score: float


@dataclass(frozen=True)
class RelatedNote:
    note_id: str
    title: str
    reason: str
    score: float


@dataclass(frozen=True)
class TraversalResult:
    connections: tuple[RankedConnection, ...]
    notes: tuple[RelatedNote, ...]
    hop_by_node: dict[str, int]
    node_lookup: dict[str, dict[str, Any]]
