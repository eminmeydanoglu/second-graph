"""Tests for source_note-aware reconciliation in NoteSynchronizer.

Key scenario:
- Note A links to Note B (source_note: note:note_a)
- Agent adds Note A -> Note C as RELATED_TO (different relation, no source_note)
- Note A is updated, B link removed
- Expectation: B wikilink deleted, C edge PRESERVED (different relation type)
"""

from pathlib import Path
from unittest.mock import Mock

import pytest

from src.graph.sync import NoteSynchronizer
from src.graph.neo4j_storage import Neo4jStorage
from src.graph.schema import EdgeType, generate_node_id
from src.vector.embedder import Embedder


class TestSourceNoteReconciliation:
    @pytest.fixture
    def workspace(self, tmp_path):
        return tmp_path

    @pytest.fixture
    def storage(self):
        """Real Neo4j storage - these are integration tests."""
        try:
            s = Neo4jStorage(
                uri="bolt://localhost:7687", user="neo4j", password="obsidian"
            )
            s.get_stats()
            s.clear()
            yield s
            s.clear()
            s.close()
        except Exception:
            pytest.skip("Neo4j not available")

    @pytest.fixture
    def embedder(self):
        mock = Mock(spec=Embedder)
        mock.embed.return_value = [0.1] * 384
        return mock

    @pytest.fixture
    def synchronizer(self, storage, embedder):
        return NoteSynchronizer(storage, embedder)

    def test_non_wikilink_edge_preserved_after_sync(
        self, synchronizer, workspace, storage
    ):
        """Non-WIKILINK edges survive wikilink re-sync."""
        note_a = workspace / "Note A.md"
        note_a.write_text(
            """---
type: Note
---
# Note A

Link to [[Note B]].
"""
        )

        result = synchronizer.sync_note_from_file(note_a)
        assert result["success"]
        assert result["edges"]["added"] == 1

        node_a_id = result["node_id"]

        # Verify wikilink edge exists
        wikilink_edges = storage.get_edges_by_source_note(
            node_a_id, node_a_id, relation="WIKILINK"
        )
        assert len(wikilink_edges) == 1
        assert wikilink_edges[0]["target_name"] == "Note B"

        # Agent adds a RELATED_TO edge (not a wikilink)
        storage.add_node("Note", "note:note_c", "Note C")
        storage.add_edge(
            node_a_id,
            "note:note_c",
            "RELATED_TO",
            {"source_note": node_a_id},
        )

        # Verify RELATED_TO edge exists
        related_edges = storage.get_edges_by_source_note(
            node_a_id, node_a_id, relation="RELATED_TO"
        )
        assert len(related_edges) == 1

        # Re-sync with different wikilinks
        note_a.write_text(
            """---
type: Note
---
# Note A

Now links to [[Note D]] instead.
"""
        )

        result2 = synchronizer.sync_note_from_file(note_a)
        assert result2["success"]
        assert result2["edges"]["removed"] == 1  # B removed
        assert result2["edges"]["added"] == 1  # D added

        # RELATED_TO edge should survive (different relation type)
        related_edges_after = storage.get_edges_by_source_note(
            node_a_id, node_a_id, relation="RELATED_TO"
        )
        assert len(related_edges_after) == 1, "Non-wikilink edge should be preserved!"

        # Wikilink now points to D
        wikilink_edges_after = storage.get_edges_by_source_note(
            node_a_id, node_a_id, relation="WIKILINK"
        )
        assert len(wikilink_edges_after) == 1
        assert wikilink_edges_after[0]["target_name"] == "Note D"

    def test_sync_returns_source_note_id(self, synchronizer, workspace):
        """Sync result includes source_note (the node_id)."""
        note = workspace / "Source Check.md"
        note.write_text("# Source Check\n\nContent.")

        result = synchronizer.sync_note_from_file(note)

        assert "source_note" in result
        assert result["source_note"].startswith("note:")

    def test_wikilink_edges_from_other_notes_preserved(
        self, synchronizer, workspace, storage
    ):
        """WIKILINK edges from a different source_note are not touched."""
        note_a = workspace / "Note A.md"
        note_a.write_text("# Note A\n\n[[Shared Target]]")

        result_a = synchronizer.sync_note_from_file(note_a)
        node_a_id = result_a["node_id"]

        # Manually add a WIKILINK from note_a's target with a different source_note
        target_id = generate_node_id("Note", "Shared Target")
        storage.add_edge(
            target_id,
            node_a_id,
            "WIKILINK",
            source_note=target_id,  # owned by the target note
        )

        # Re-sync note A — should not touch the reverse wikilink
        note_a.write_text("# Note A\n\n[[Different Target]]")
        result_a2 = synchronizer.sync_note_from_file(note_a)

        # The reverse wikilink (owned by target) should survive
        reverse_edges = storage.get_edges_by_source_note(
            target_id, target_id, relation="WIKILINK"
        )
        assert len(reverse_edges) == 1

    def test_wikilink_uses_note_identity_not_other_type(
        self, synchronizer, workspace, storage
    ):
        """Wikilinks should resolve to Note identity, not first same-name non-note node."""
        storage.add_node("Concept", "concept:shared_name", "Shared Name", {})

        note = workspace / "Main.md"
        note.write_text("# Main\n\n[[Shared Name]]")

        result = synchronizer.sync_note_from_file(note)
        assert result["success"]

        neighbors = storage.get_neighbors(result["node_id"], direction="out")
        target_ids = {n["node"]["id"] for n in neighbors}

        assert "note:shared_name" in target_ids
        assert "concept:shared_name" not in target_ids
