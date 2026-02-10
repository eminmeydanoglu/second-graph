"""Tests for NoteSynchronizer."""

import shutil
from pathlib import Path
from unittest.mock import Mock, MagicMock

import pytest

from src.graph.sync import NoteSynchronizer
from src.graph.neo4j_storage import Neo4jStorage
from src.vector.store import VectorStore
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
            s = Neo4jStorage(
                uri="bolt://localhost:7687", user="neo4j", password="obsidian"
            )
            s.get_stats()
            s.clear()
            yield s
            s.clear()
            s.close()
        except Exception:
            mock = Mock(spec=Neo4jStorage)
            mock.get_node.return_value = None
            mock.get_neighbors.return_value = []
            mock.find_nodes.return_value = []
            mock.find_edges.return_value = []
            yield mock

    @pytest.fixture
    def vectors(self, tmp_path):
        """Real vector store in tmp dir."""
        return VectorStore(tmp_path / "vectors.db")

    @pytest.fixture
    def embedder(self):
        """Mock embedder to avoid loading models."""
        mock = Mock(spec=Embedder)
        mock.embed.return_value = [0.1] * 384
        return mock

    @pytest.fixture
    def synchronizer(self, storage, vectors, embedder):
        return NoteSynchronizer(storage, vectors, embedder)

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

        result = synchronizer.sync_note(note_path)

        assert result["success"] is True
        assert result["action"] == "created"
        assert result["node_id"] == "concept:new_idea"
        assert result["edges"]["added"] == 2  # Two wikilinks

        if isinstance(storage, Neo4jStorage):
            node = storage.get_node("concept:new_idea")
            assert node is not None
            assert node["node"]["title"] == "New Idea"
            assert "idea" in node["node"]["tags"]

            neighbors = storage.get_neighbors("concept:new_idea", direction="out")
            assert len(neighbors) == 2
            names = {n["node"]["name"] for n in neighbors}
            assert "Existing Concept" in names
            assert "New Concept" in names

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

        synchronizer.sync_note(note_path)

        note_path.write_text(
            """---
type: Project
---
Links: [[Person A]], [[Person C]]
"""
        )

        result = synchronizer.sync_note(note_path)

        assert result["success"] is True
        assert result["action"] == "updated"

        if isinstance(storage, Neo4jStorage):
            assert result["edges"]["kept"] == 1
            assert result["edges"]["added"] == 1
            assert result["edges"]["removed"] == 1

            neighbors = storage.get_neighbors("project:project_alpha", direction="out")
            names = {n["node"]["name"] for n in neighbors}
            assert "Person A" in names
            assert "Person C" in names
            assert "Person B" not in names

    def test_file_not_found(self, synchronizer):
        """Test error handling for missing file."""
        result = synchronizer.sync_note("/non/existent/file.md")
        assert result["success"] is False
        assert "File not found" in result["error"]
