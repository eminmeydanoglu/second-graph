"""Tests for MCP server tools."""

from pathlib import Path
import subprocess
from unittest.mock import Mock, MagicMock

import pytest

from src.mcp import server as mcp_server
from src.graph.schema import SourceType, generate_source_id


class _FakeStorage:
    """Fake storage for testing MCP tool behavior."""

    def delete_node(self, node_id: str) -> dict:
        return {"deleted": False, "edges_removed": 0}

    def merge_nodes_simple(self, keep_id: str, merge_id: str) -> dict:
        return {"merged": False, "edges_transferred": 0, "error": "node not found"}


class _FakeVectors:
    """Fake vector store for testing."""

    def delete_entity(self, node_id: str) -> bool:
        return False


class TestMCPFailurePropagation:
    """Tests for MCP tool failure propagation."""

    def test_delete_node_propagates_failure(self, monkeypatch):
        """MCP delete_node should set success False when deletion fails."""
        monkeypatch.setattr(mcp_server, "storage", _FakeStorage())
        monkeypatch.setattr(mcp_server, "vectors", _FakeVectors())

        result = mcp_server.delete_node("goal:missing")

        assert result["success"] is False
        assert result["deleted"] is False

    def test_merge_nodes_propagates_failure(self, monkeypatch):
        """MCP merge_nodes should set success False when merge fails."""
        monkeypatch.setattr(mcp_server, "storage", _FakeStorage())
        monkeypatch.setattr(mcp_server, "vectors", _FakeVectors())

        result = mcp_server.merge_nodes("goal:keep", "goal:missing")

        assert result["success"] is False
        assert result["merged"] is False


class TestCLIEntrypoint:
    """Tests for CLI entrypoint."""

    def test_console_script_entrypoint_runs_help(self):
        """Installed console script should run without ModuleNotFoundError."""
        script = (
            Path(__file__).resolve().parents[3] / ".venv" / "bin" / "obsidian-brain"
        )
        if not script.exists():
            pytest.skip("Script not found - not installed in dev mode")

        completed = subprocess.run(
            [str(script), "--help"],
            capture_output=True,
            text=True,
            check=False,
        )

        assert completed.returncode == 0, completed.stderr
        assert "Obsidian Brain" in completed.stdout

    def test_mcp_server_entrypoint_exists(self):
        """MCP server entrypoint should exist."""
        script = (
            Path(__file__).resolve().parents[3] / ".venv" / "bin" / "obsidian-brain-mcp"
        )
        if not script.exists():
            pytest.skip("MCP script not found - not installed in dev mode")

        assert script.exists()


class _FakeSynchronizer:
    """Fake synchronizer for testing."""

    def sync_note(self, path):
        return {
            "success": True,
            "node_id": "note:test",
            "action": "created",
            "edges": {"added": 1, "removed": 0, "kept": 0},
            "source": f"file:{path}",
        }


class _FakeStorageWithNodes:
    """Fake storage that simulates nodes existing."""

    def get_node(self, node_id):
        return {
            "node": {"id": node_id, "name": "Test", "_labels": ["Note"]},
            "connections": [],
        }

    def add_edge(self, from_id, to_id, relation, properties=None, source=None):
        return {
            "success": True,
            "edge_id": "edge:test123",
            "from_id": from_id,
            "to_id": to_id,
            "from_name": "From",
            "to_name": "To",
            "relation": relation,
        }


class TestSyncNoteToolUnit:
    """Unit tests for sync_note MCP tool."""

    def test_sync_note_calls_synchronizer(self, monkeypatch):
        """sync_note should delegate to NoteSynchronizer."""
        fake_sync = _FakeSynchronizer()
        monkeypatch.setattr(mcp_server, "synchronizer", fake_sync)

        result = mcp_server.sync_note("/vault/Test.md")

        assert result["success"] is True
        assert result["node_id"] == "note:test"
        assert result["source"] == "file:/vault/Test.md"

    def test_sync_note_requires_init(self, monkeypatch):
        """sync_note should fail if server not initialized."""
        monkeypatch.setattr(mcp_server, "synchronizer", None)

        with pytest.raises(RuntimeError, match="not initialized"):
            mcp_server.sync_note("/vault/Test.md")


class TestAddAgentEdgeToolUnit:
    """Unit tests for add_agent_edge MCP tool."""

    def test_add_agent_edge_sets_source(self, monkeypatch):
        """add_agent_edge should set agent source on edge."""
        fake_storage = _FakeStorageWithNodes()
        monkeypatch.setattr(mcp_server, "storage", fake_storage)

        result = mcp_server.add_agent_edge(
            from_id="note:a",
            to_id="note:b",
            relation="RELATED_TO",
        )

        assert result["success"] is True
        assert result["source"] == "agent"

    def test_add_agent_edge_missing_source_node(self, monkeypatch):
        """add_agent_edge should fail if source node missing."""

        class StorageMissingNode:
            def get_node(self, node_id):
                if node_id == "note:missing":
                    return None
                return {"node": {"_labels": ["Note"]}, "connections": []}

        monkeypatch.setattr(mcp_server, "storage", StorageMissingNode())

        result = mcp_server.add_agent_edge(
            from_id="note:missing",
            to_id="note:b",
            relation="RELATED_TO",
        )

        assert result["success"] is False
        assert "not found" in result["error"]


class TestSyncNoteToolIntegration:
    """Integration tests for sync_note with real Neo4j."""

    @pytest.fixture
    def init_server(self, tmp_path):
        """Initialize MCP server with real storage."""
        try:
            mcp_server.init_server(
                neo4j_uri="bolt://localhost:7687",
                neo4j_user="neo4j",
                neo4j_password="obsidian",
                vector_db=str(tmp_path / "vectors.db"),
            )
            mcp_server.storage.clear()
            yield
            mcp_server.storage.clear()
            mcp_server.storage.close()
        except Exception as e:
            pytest.skip(f"Neo4j not available: {e}")

    def test_sync_note_integration(self, init_server, tmp_path):
        """Test sync_note with real storage."""
        note = tmp_path / "Integration Test.md"
        note.write_text(
            """---
type: Note
---
# Integration Test

Links to [[Target Note]].
"""
        )

        result = mcp_server.sync_note(str(note))

        assert result["success"] is True
        assert result["action"] == "created"
        assert result["edges"]["added"] == 1
        assert "source" in result
        assert result["source"].startswith("file:")

    def test_add_agent_edge_integration(self, init_server):
        """Test add_agent_edge with real storage."""
        # Create nodes first
        mcp_server.storage.add_node("Note", "note:int_a", "Int A", {})
        mcp_server.storage.add_node("Note", "note:int_b", "Int B", {})

        result = mcp_server.add_agent_edge(
            from_id="note:int_a",
            to_id="note:int_b",
            relation="RELATED_TO",
        )

        assert result["success"] is True
        assert result["source"] == "agent"

        # Verify edge has source
        edge = mcp_server.storage.get_edge(result["edge_id"])
        assert edge["edge"]["source"] == "agent"
