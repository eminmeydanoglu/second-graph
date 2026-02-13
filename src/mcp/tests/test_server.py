"""Tests for MCP server tools."""

from pathlib import Path
import subprocess

import pytest

from src.mcp import server as mcp_server
from src.graph.tests.neo4j_test_config import get_test_neo4j_config, guard_test_uri


class _FakeStorage:
    """Fake storage for testing MCP tool behavior."""

    def delete_node(self, node_id: str) -> dict:
        return {"deleted": False, "edges_removed": 0}

    def merge_nodes_simple(self, keep_id: str, merge_id: str) -> dict:
        return {"merged": False, "edges_transferred": 0, "error": "node not found"}

    def get_node(self, node_id: str) -> dict | None:
        """Return a fake node for validation tests."""
        if node_id.startswith("goal:"):
            return {
                "node": {"_labels": ["Goal"], "id": node_id, "name": "Fake"},
                "connections": [],
            }
        if node_id.startswith("person:"):
            return {
                "node": {"_labels": ["Person"], "id": node_id, "name": "Fake"},
                "connections": [],
            }
        return None

    def add_edge(self, from_id, to_id, relation, properties=None, source_note=None):
        return {
            "success": True,
            "edge_id": "edge:fake",
            "from_id": from_id,
            "to_id": to_id,
            "relation": relation,
            "from_name": "F",
            "to_name": "T",
        }


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


class TestStrictValidation:
    """Tests for strict schema validation in MCP add_edge."""

    def test_add_edge_rejects_unknown_edge_type(self, monkeypatch):
        """MCP add_edge should reject unknown relation types."""
        monkeypatch.setattr(mcp_server, "storage", _FakeStorage())

        result = mcp_server.add_edge("person:x", "goal:y", "FAKE_RELATION")

        assert result["success"] is False
        assert any("Unknown edge type" in e for e in result["errors"])

    def test_add_edge_rejects_invalid_constraint(self, monkeypatch):
        """MCP add_edge should reject invalid source/target type combos."""
        monkeypatch.setattr(mcp_server, "storage", _FakeStorage())

        # Person -> Goal via CONTRIBUTES_TO is invalid (Person not in valid sources)
        result = mcp_server.add_edge("person:x", "goal:y", "CONTRIBUTES_TO")

        assert result["success"] is False
        assert len(result["errors"]) > 0

    def test_add_edge_passes_valid_combo(self, monkeypatch):
        """MCP add_edge should allow valid combos."""
        monkeypatch.setattr(mcp_server, "storage", _FakeStorage())

        result = mcp_server.add_edge("person:x", "goal:y", "HAS_GOAL")

        assert result["success"] is True


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


class TestSyncNoteToolIntegration:
    """Integration tests for source_note with real Neo4j."""

    @pytest.fixture
    def init_server(self, tmp_path):
        """Initialize MCP server with real storage."""
        try:
            uri, user, password = get_test_neo4j_config()
            guard_test_uri(uri)
            mcp_server.init_server(
                neo4j_uri=uri,
                neo4j_user=user,
                neo4j_password=password,
                vector_db=str(tmp_path / "vectors.db"),
            )
            mcp_server.storage.clear(force=True)
            yield
            mcp_server.storage.clear(force=True)
            mcp_server.storage.close()
        except Exception as e:
            pytest.skip(f"Neo4j not available: {e}")

    def test_source_note_integration(self, init_server, tmp_path):
        """Test source_note with real storage."""
        note = tmp_path / "Integration Test.md"
        note.write_text(
            """---
type: Note
---
# Integration Test

Links to [[Target Note]].
"""
        )

        result = mcp_server.source_note(str(note))

        assert result["success"] is True
        assert result["action"] == "created"
        assert result["edges"]["added"] == 1
        assert "source_note" in result
        assert result["source_note"].startswith("note:")
