import pytest

from src.retrieval.traversal import expand_from_anchors_with_hops
from src.retrieval.types import ScoredNode


class _Storage:
    def get_neighbors(
        self,
        node_id: str,
        direction: str = "both",
        edge_types: list[str] | None = None,
        depth: int = 1,
    ) -> list[dict]:
        if node_id == "goal:a":
            duplicate = {
                "node": {
                    "id": "concept:b",
                    "name": "Concept B",
                    "_labels": ["Concept"],
                },
                "relation": "related_to",
                "direction": "out",
            }
            return [
                duplicate,
                duplicate,
                {
                    "node": {"id": "note:n1", "name": "Note N1", "_labels": ["Note"]},
                    "relation": "mentions",
                    "direction": "out",
                },
            ]
        return []


def test_traversal_dedupes_connections_and_extracts_notes():
    anchor = ScoredNode(
        node_id="goal:a",
        node_type="Goal",
        name="Goal A",
        summary=None,
        vector_score=0.9,
        keyword_score=0.7,
        graph_score=1.0,
        final_score=0.8,
        hop_distance=0,
    )

    result = expand_from_anchors_with_hops(
        [anchor],
        storage=_Storage(),
        depth=1,
        neighbor_cap=20,
    )

    assert len(result.connections) == 2
    assert result.hop_by_node["concept:b"] == 1

    notes = list(result.notes)
    assert len(notes) == 1
    assert notes[0].note_id == "note:n1"
    assert notes[0].reason == "neighbor"
    assert notes[0].score == pytest.approx(0.68)


def test_anchor_note_gets_anchor_reason():
    anchor = ScoredNode(
        node_id="note:anchor",
        node_type="Note",
        name="Anchor Note",
        summary=None,
        vector_score=0.7,
        keyword_score=0.6,
        graph_score=1.0,
        final_score=0.9,
        hop_distance=0,
    )

    result = expand_from_anchors_with_hops(
        [anchor],
        storage=_Storage(),
        depth=1,
        neighbor_cap=20,
    )

    anchor_note = [note for note in result.notes if note.note_id == "note:anchor"][0]
    assert anchor_note.reason == "anchor"
    assert anchor_note.score == pytest.approx(0.9)
