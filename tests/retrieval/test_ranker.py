import pytest

from src.retrieval.config import DEFAULT_RETRIEVAL_CONFIG
from src.retrieval.ranker import apply_graph_locality, sort_scored_nodes
from src.retrieval.types import ScoredNode


def _node(
    node_id: str,
    *,
    vector: float,
    keyword: float,
    final: float,
    hop: int,
) -> ScoredNode:
    return ScoredNode(
        node_id=node_id,
        node_type="Goal",
        name=node_id,
        summary=None,
        vector_score=vector,
        keyword_score=keyword,
        graph_score=0.0,
        final_score=final,
        hop_distance=hop,
    )


def test_sort_scored_nodes_tie_break_order():
    nodes = [
        _node("z", vector=0.7, keyword=0.3, final=0.5, hop=1),
        _node("a", vector=0.7, keyword=0.3, final=0.5, hop=1),
        _node("m", vector=0.8, keyword=0.2, final=0.5, hop=1),
    ]

    sorted_ids = [node.node_id for node in sort_scored_nodes(nodes)]
    assert sorted_ids == ["m", "a", "z"]


def test_apply_graph_locality_bonus_by_hop():
    base = [
        _node("goal:a", vector=0.5, keyword=0.5, final=0.0, hop=99),
        _node("goal:b", vector=0.5, keyword=0.5, final=0.0, hop=99),
    ]

    ranked = apply_graph_locality(base, hop_by_node={"goal:a": 0, "goal:b": 2})

    assert ranked[0].node_id == "goal:a"
    assert ranked[0].graph_score == DEFAULT_RETRIEVAL_CONFIG.graph_bonus_anchor
    assert ranked[1].graph_score == DEFAULT_RETRIEVAL_CONFIG.graph_bonus_hop2
    assert ranked[0].final_score == pytest.approx(0.55)
    assert ranked[1].final_score == pytest.approx(0.48)
