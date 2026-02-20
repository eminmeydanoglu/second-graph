"""Configuration for deterministic retrieval."""

from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievalConfig:
    """Constants controlling recall ranking and rendering."""

    vector_limit: int = 24
    lexical_cap: int = 24
    anchor_count: int = 6
    neighbor_cap: int = 20
    max_depth: int = 2

    vector_weight: float = 0.65
    keyword_weight: float = 0.25
    graph_weight: float = 0.10

    graph_bonus_anchor: float = 1.0
    graph_bonus_hop1: float = 0.6
    graph_bonus_hop2: float = 0.3

    push_max_tokens: int = 900
    pull_max_tokens: int = 1600

    nodes_budget_ratio: float = 0.45
    connections_budget_ratio: float = 0.35
    notes_budget_ratio: float = 0.20

    def max_tokens_for_mode(self, mode: str) -> int:
        """Return token budget for push/pull profile."""
        return self.push_max_tokens if mode == "push" else self.pull_max_tokens

    def clamp_depth(self, depth: int) -> int:
        """Clamp traversal depth to supported range."""
        if depth < 1:
            return 1
        if depth > self.max_depth:
            return self.max_depth
        return depth


DEFAULT_RETRIEVAL_CONFIG = RetrievalConfig()
