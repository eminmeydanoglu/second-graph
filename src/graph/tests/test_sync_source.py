"""Tests for source-aware reconciliation in NoteSynchronizer.

Key scenario:
- Note A links to Note B (source: file)
- Agent adds Note A -> Note C (source: agent)
- Note A is updated, B link removed
- Expectation: B edge deleted, C edge PRESERVED
"""

from pathlib import Path
from unittest.mock import Mock

import pytest

from src.graph.sync import NoteSynchronizer
from src.graph.neo4j_storage import Neo4jStorage
from src.graph.schema import SourceType, generate_source_id, EdgeType
from src.vector.store import VectorStore
from src.vector.embedder import Embedder


class TestSourceAwareReconciliation:
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
    def vectors(self, tmp_path):
        return VectorStore(tmp_path / "vectors.db")

    @pytest.fixture
    def embedder(self):
        mock = Mock(spec=Embedder)
        mock.embed.return_value = [0.1] * 384
        return mock

    @pytest.fixture
    def synchronizer(self, storage, vectors, embedder):
        return NoteSynchronizer(storage, vectors, embedder)

    def test_agent_edge_preserved_after_file_sync(
        self, synchronizer, workspace, storage
    ):
        """Core test: Agent-created edges survive file re-sync."""
        note_a = workspace / "Note A.md"
        note_a.write_text(
            """---
type: Note
---
# Note A

Link to [[Note B]].
"""
        )

        result = synchronizer.sync_note(note_a)
        assert result["success"]
        assert result["edges"]["added"] == 1

        node_a_id = result["node_id"]
        file_source = result["source"]

        file_edges = storage.get_edges_by_source(node_a_id, file_source)
        assert len(file_edges) == 1
        assert file_edges[0]["target_name"] == "Note B"

        storage.add_node("Note", "note:note_c", "Note C")
        agent_source = generate_source_id(SourceType.AGENT)
        storage.add_edge(
            node_a_id,
            "note:note_c",
            EdgeType.WIKILINK.value,
            source=agent_source,
        )

        agent_edges = storage.get_edges_by_source(node_a_id, agent_source)
        assert len(agent_edges) == 1
        assert agent_edges[0]["target_name"] == "Note C"

        note_a.write_text(
            """---
type: Note
---
# Note A

Now links to [[Note D]] instead.
"""
        )

        result2 = synchronizer.sync_note(note_a)
        assert result2["success"]
        assert result2["edges"]["removed"] == 1  # B removed
        assert result2["edges"]["added"] == 1  # D added

        agent_edges_after = storage.get_edges_by_source(node_a_id, agent_source)
        assert len(agent_edges_after) == 1, "Agent edge should be preserved!"
        assert agent_edges_after[0]["target_name"] == "Note C"

        file_edges_after = storage.get_edges_by_source(node_a_id, file_source)
        assert len(file_edges_after) == 1
        assert file_edges_after[0]["target_name"] == "Note D"

    def test_multiple_sources_independent(self, synchronizer, workspace, storage):
        """Each source manages its own edges independently."""
        note = workspace / "Multi Source.md"
        note.write_text("# Multi Source\n\n[[Target A]]")

        result = synchronizer.sync_note(note)
        node_id = result["node_id"]
        file_source = result["source"]

        storage.add_node("Note", "note:agent_target", "Agent Target")
        storage.add_node("Note", "note:extraction_target", "Extraction Target")

        agent_source = generate_source_id(SourceType.AGENT)
        extraction_source = generate_source_id(SourceType.EXTRACTION, "v1:" + str(note))

        storage.add_edge(
            node_id, "note:agent_target", "RELATED_TO", source=agent_source
        )
        storage.add_edge(
            node_id, "note:extraction_target", "MENTIONS", source=extraction_source
        )

        deleted = storage.delete_edges_by_source(node_id, extraction_source)
        assert deleted == 1

        assert len(storage.get_edges_by_source(node_id, file_source)) == 1
        assert len(storage.get_edges_by_source(node_id, agent_source)) == 1
        assert len(storage.get_edges_by_source(node_id, extraction_source)) == 0

    def test_sync_returns_source_id(self, synchronizer, workspace):
        """Sync result includes the source ID used."""
        note = workspace / "Source Check.md"
        note.write_text("# Source Check\n\nContent.")

        result = synchronizer.sync_note(note)

        assert "source" in result
        assert result["source"].startswith("file:")
        assert str(note) in result["source"]
