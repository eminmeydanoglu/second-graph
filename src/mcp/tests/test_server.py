"""Tests for MCP server tools."""

from pathlib import Path
import subprocess

import pytest

from src.mcp import server as mcp_server


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


class TestMCPFailurePropagation:
    """Tests for MCP tool failure propagation."""

    def test_delete_node_propagates_failure(self, monkeypatch):
        """MCP delete_node should set success False when deletion fails."""
        monkeypatch.setattr(mcp_server, "storage", _FakeStorage())

        result = mcp_server.delete_node("goal:missing")

        assert result["success"] is False
        assert result["deleted"] is False

    def test_merge_nodes_propagates_failure(self, monkeypatch):
        """MCP merge_nodes should set success False when merge fails."""
        monkeypatch.setattr(mcp_server, "storage", _FakeStorage())

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

    def test_add_edge_uses_canonical_type_from_multiple_labels(self, monkeypatch):
        """Validation should use canonical type even with extra labels present."""

        class _MultiLabelStorage(_FakeStorage):
            def get_node(self, node_id: str) -> dict | None:
                if node_id.startswith("person:"):
                    return {
                        "node": {
                            "_labels": ["Entity", "Person"],
                            "id": node_id,
                            "name": "P",
                        },
                        "connections": [],
                    }
                if node_id.startswith("goal:"):
                    return {
                        "node": {
                            "_labels": ["Entity", "Goal"],
                            "id": node_id,
                            "name": "G",
                        },
                        "connections": [],
                    }
                return None

        monkeypatch.setattr(mcp_server, "storage", _MultiLabelStorage())

        result = mcp_server.add_edge("person:x", "goal:y", "HAS_GOAL")
        assert result["success"] is True


class _FakeEmbedder:
    def embed(self, text: str) -> list[float]:
        return [0.1] * 384


class _LeakyStorage:
    """Storage stub that intentionally leaks internal vector fields."""

    def get_node(self, node_id: str) -> dict | None:
        return {
            "node": {
                "id": node_id,
                "name": "Leaky",
                "embedding": [0.1, 0.2],
                "_labels": ["Goal", "Entity"],
            },
            "connections": [
                {
                    "neighbor_id": "goal:n2",
                    "neighbor_name": "N2",
                    "neighbor_labels": ["Goal", "Entity"],
                }
            ],
        }

    def search_similar(self, query_embedding, node_types=None, limit: int = 10):
        return [
            {
                "node_id": "goal:leak",
                "node_type": "Goal",
                "name": "Leaky Result",
                "summary": "s",
                "embedding": [0.1, 0.2],
                "score": 0.99,
            }
        ]

    def get_stats(self):
        return {
            "nodes": 1,
            "relationships": 0,
            "by_label": {"Entity": 5, "Goal": 1},
        }


class TestPrivacyFiltering:
    """Ensure MCP never exposes internal embedding details."""

    def test_get_node_strips_embedding_and_entity_label(self, monkeypatch):
        monkeypatch.setattr(mcp_server, "storage", _LeakyStorage())

        result = mcp_server.get_node("goal:leak")

        assert result["success"] is True
        assert "embedding" not in result["node"]
        assert "Entity" not in result["node"].get("_labels", [])
        assert "Entity" not in result["connections"][0]["neighbor_labels"]

    def test_search_entities_strips_embedding(self, monkeypatch):
        monkeypatch.setattr(mcp_server, "storage", _LeakyStorage())
        monkeypatch.setattr(mcp_server, "embedder", _FakeEmbedder())

        result = mcp_server.search_entities("test")

        assert result["success"] is True
        assert result["count"] == 1
        assert "embedding" not in result["entities"][0]

    def test_get_stats_strips_entity_label(self, monkeypatch):
        monkeypatch.setattr(mcp_server, "storage", _LeakyStorage())

        result = mcp_server.get_stats()

        assert result["success"] is True
        assert "Entity" not in result["by_label"]


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

    def test_source_note_integration(self, init_server, tmp_path):
        """Test source_note (alias) with real storage."""
        note = tmp_path / "Integration Test.md"
        note.write_text(
            """---
type: Note
---
# Integration Test

Links to [[Target Note]].
"""
        )

        # source_note alias
        result = mcp_server.source_note(str(note))

        assert result["success"] is True
        assert result["action"] == "created"
        assert result["edges"]["added"] == 1
        assert "source_note" in result
        assert result["source_note"].startswith("note:")

    def test_sync_note_integration(self, init_server, tmp_path):
        """Test sync_note (main function) with real storage."""
        note = tmp_path / "Sync Test.md"
        note.write_text("# Sync Test")

        result = mcp_server.sync_note(str(note))

        assert result["success"] is True
        assert result["action"] == "created"
