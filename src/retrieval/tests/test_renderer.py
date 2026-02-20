from src.retrieval.renderer import render_context, render_context_with_meta
from src.retrieval.types import ScoredNode


def test_renderer_zero_hit_contract():
    output = render_context([], [], [], max_tokens=200)

    assert "## Matched Nodes" in output
    assert "## Connections" in output
    assert "## Related Notes" in output
    assert output.count("- (none)") == 3


def test_renderer_truncation_marker():
    nodes = [
        ScoredNode(
            node_id=f"goal:{i}",
            node_type="Goal",
            name=f"Goal {i}",
            summary=None,
            vector_score=0.9,
            keyword_score=0.7,
            graph_score=1.0,
            final_score=0.8,
            hop_distance=0,
        )
        for i in range(12)
    ]

    output, truncated_sections = render_context_with_meta(
        nodes,
        [],
        [],
        max_tokens=40,
    )

    assert "- ...(truncated" in output
    assert "matched_nodes" in truncated_sections
