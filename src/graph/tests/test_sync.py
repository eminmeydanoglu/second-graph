"""Tests for NoteSynchronizer."""

import shutil
from pathlib import Path
from unittest.mock import Mock, MagicMock

import pytest

from src.graph.sync import NoteSynchronizer
from src.graph.neo4j_storage import Neo4jStorage
from src.graph.tests.neo4j_test_config import get_test_neo4j_config, guard_test_uri
from src.vector.embedder import Embedder


class TestNoteSynchronizer:
    @pytest.fixture
    def workspace(self, tmp_path):
        """Create a temporary workspace with files."""
        return tmp_path

    @pytest.fixture
    def storage(self):
        """Mock Neo4j storage."""
        try:
            uri, user, password = get_test_neo4j_config()
            guard_test_uri(uri)
            s = Neo4jStorage(uri=uri, user=user, password=password)
            s.get_stats()
            s.clear(force=True)
            yield s
            s.clear(force=True)
            s.close()
        except Exception:
            mock = Mock(spec=Neo4jStorage)
            mock.get_node.return_value = None
            mock.get_neighbors.return_value = []
            mock.find_nodes.return_value = []
            mock.find_edges.return_value = []
            mock.get_edges_by_source_note.return_value = []
            mock.delete_edges_by_source_note.return_value = 0
            mock.find_note_by_vault_rel_path.return_value = None
            mock.find_note_candidates_by_stem.return_value = []
            mock.add_node.return_value = {"id": "mock"}
            mock.update_node.return_value = {"id": "mock"}
            mock.add_edge.return_value = {"success": True, "edge_id": "edge:mock"}
            mock.set_embedding.return_value = True
            yield mock

    @pytest.fixture
    def embedder(self):
        """Mock embedder to avoid loading models."""
        mock = Mock(spec=Embedder)
        mock.embed.return_value = [0.1] * 384
        return mock

    @pytest.fixture
    def synchronizer(self, storage, embedder):
        return NoteSynchronizer(storage, embedder)

    def test_sync_new_note(self, synchronizer, workspace, storage):
        """Test syncing a new file creates nodes and edges."""
        note_path = workspace / "New Idea.md"
        note_path.write_text(
            """---
type: Concept
tags: [idea, cool]
---
# New Idea

This is a link to [[Existing Concept]].
And a link to [[New Concept]].
"""
        )

        if isinstance(storage, Mock):
            storage.get_node.return_value = None
            storage.find_nodes.return_value = []

        result = synchronizer.sync_note_from_file(note_path)

        assert result["success"] is True
        assert result["action"] == "created"
        assert result["node_id"] == "note:new_idea"
        assert result["edges"]["added"] == 2  # Two wikilinks
        # Tag sync now handled as graph edges

        if not isinstance(storage, Mock):
            node = storage.get_node("note:new_idea")
            assert node is not None
            assert node["node"]["name"] == "New Idea"
            assert "idea" in node["node"]["tags"]

            neighbors = storage.get_neighbors("note:new_idea", direction="out")
            names = {n["node"]["name"] for n in neighbors}
            assert "Existing Concept" in names
            assert "New Concept" in names
            assert "idea" in names  # Tag node connected

    def test_sync_update_note(self, synchronizer, workspace, storage):
        """Test updating a note updates edges."""
        note_path = workspace / "Project Alpha.md"
        note_path.write_text(
            """---
type: Project
---
Links: [[Person A]], [[Person B]]
"""
        )

        synchronizer.sync_note_from_file(note_path)

        note_path.write_text(
            """---
type: Project
---
Links: [[Person A]], [[Person C]]
"""
        )

        result = synchronizer.sync_note_from_file(note_path)

        assert result["success"] is True
        if not isinstance(storage, Mock):
            assert result["action"] == "updated"
        else:
            assert result["action"] in {"created", "updated"}

        if not isinstance(storage, Mock):
            assert result["edges"]["kept"] == 1
            assert result["edges"]["added"] == 1
            assert result["edges"]["removed"] == 1

            neighbors = storage.get_neighbors("note:project_alpha", direction="out")
            names = {n["node"]["name"] for n in neighbors}
            assert "Person A" in names
            assert "Person C" in names
            assert "Person B" not in names

    def test_file_not_found(self, synchronizer):
        """Test error handling for missing file."""
        result = synchronizer.sync_note_from_file("/non/existent/file.md")
        assert result["success"] is False
        assert "File not found" in result["error"]

    def test_frontmatter_type_is_normalized(self, synchronizer, workspace, storage):
        """Lower/upper case frontmatter types normalize to canonical schema type."""
        note_path = workspace / "Case Type.md"
        note_path.write_text(
            """---
type: note
---
# Case Type
"""
        )

        result = synchronizer.sync_note_from_file(note_path)

        assert result["success"] is True
        assert result["node_id"].startswith("note:")

        if not isinstance(storage, Mock):
            node = storage.get_node(result["node_id"])
            assert node is not None
            assert "Note" in node["node"].get("_labels", [])

    def test_sync_embedding_uses_routing_text_not_full_content(
        self, synchronizer, workspace, embedder
    ):
        """Embedding text should come from routing metadata, not note body."""
        note_path = workspace / "Routing Text.md"
        note_path.write_text(
            """---
type: Note
summary: Compact summary
tags: [memory]
---
# Routing Text

This body should never appear inside embedding payload.
"""
        )

        result = synchronizer.sync_note_from_file(note_path)

        assert result["success"] is True

        embed_text = embedder.embed.call_args[0][0]
        assert embed_text == "Note: Routing Text. Compact summary. Tags: memory"
        assert "never appear" not in embed_text

    def test_sync_node_id_uses_relative_path(self, synchronizer, workspace):
        """Node ID should be derived from vault-relative file path."""
        nested = workspace / "Folder A" / "Nested Note.md"
        nested.parent.mkdir(parents=True, exist_ok=True)
        nested.write_text("# Nested Note")

        result = synchronizer.sync_note_from_file(nested, vault_root=workspace)

        assert result["success"] is True
        assert result["node_id"] == "note:folder_a/nested_note"

    def test_sync_wikilink_fragments_are_deduplicated(
        self, synchronizer, workspace, storage
    ):
        """[[Target]] and [[Target#Heading]] should resolve to same edge target."""
        note_path = workspace / "Fragments.md"
        note_path.write_text(
            """# Fragments

[[Target]]
[[Target#Heading]]
[[Target^block-id]]
"""
        )

        result = synchronizer.sync_note_from_file(note_path)

        assert result["success"] is True
        assert result["edges"]["added"] == 1

    def test_sync_handles_nested_frontmatter_structures(
        self, synchronizer, workspace, storage
    ):
        """Nested YAML structures should not crash Neo4j property writes."""
        note_path = workspace / "Nested Frontmatter.md"
        note_path.write_text(
            """---
matrix:
  - [1, 2]
  - [3, 4]
metadata:
  owner: emin
  labels: [a, b]
---
# Nested Frontmatter
"""
        )

        result = synchronizer.sync_note_from_file(note_path)

        assert result["success"] is True
