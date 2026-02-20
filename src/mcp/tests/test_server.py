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


class _RecallStorage:
    def search_similar(self, query_embedding, node_types=None, limit: int = 10):
        return [
            {
                "node_id": "goal:deterministic_renderer",
                "node_type": "Goal",
                "name": "Deterministic Renderer",
                "summary": "Build stable recall output",
                "score": 0.91,
            },
            {
                "node_id": "note:renderer_plan",
                "node_type": "Note",
                "name": "Renderer Plan",
                "summary": "Implementation details",
                "score": 0.82,
            },
        ]

    def find_nodes(self, name, node_type=None, match_type: str = "contains"):
        return [
            {
                "id": "goal:deterministic_renderer",
                "name": "Deterministic Renderer",
                "summary": "Build stable recall output",
                "_labels": ["Goal"],
            },
            {
                "id": "note:renderer_plan",
                "name": "Renderer Plan",
                "summary": "Implementation details",
                "_labels": ["Note"],
            },
        ]

    def get_neighbors(
        self, node_id, direction: str = "both", edge_types=None, depth: int = 1
    ):
        if node_id == "goal:deterministic_renderer":
            return [
                {
                    "node": {
                        "id": "note:renderer_plan",
                        "name": "Renderer Plan",
                        "_labels": ["Note"],
                    },
                    "relation": "related_to",
                    "direction": "out",
                }
            ]
        return []


class TestRecallTool:
    def test_recall_returns_markdown_contract(self, monkeypatch):
        monkeypatch.setattr(mcp_server, "storage", _RecallStorage())
        monkeypatch.setattr(mcp_server, "embedder", _FakeEmbedder())

        result = mcp_server.recall("deterministic renderer", depth=1, limit=5)

        assert result["success"] is True
        assert "## Matched Nodes" in result["markdown"]
        assert "## Connections" in result["markdown"]
        assert "## Related Notes" in result["markdown"]
        assert result["retrieval"]["vector_hits"] == 2

    def test_recall_can_include_debug_payload(self, monkeypatch):
        monkeypatch.setattr(mcp_server, "storage", _RecallStorage())
        monkeypatch.setattr(mcp_server, "embedder", _FakeEmbedder())

        result = mcp_server.recall(
            "deterministic renderer",
            depth=1,
            limit=5,
            include_debug=True,
        )

        assert result["success"] is True
        assert "debug" in result
        assert "matched_nodes" in result["debug"]


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


class _RoutingStorage:
    def __init__(self):
        self.embeddings: list[tuple[str, list[float]]] = []
        self.updated: dict[str, dict] = {}

    def add_node(self, node_type: str, node_id: str, name: str, props: dict):
        labels = [node_type]
        node = {"id": node_id, "name": name, "_labels": labels, **props}
        self.updated[node_id] = node
        return node

    def update_node(self, node_id: str, properties: dict):
        node = self.updated.setdefault(node_id, {"id": node_id, "name": "Unknown"})
        node.update(properties)
        return node

    def get_node(self, node_id: str):
        node = self.updated.get(node_id)
        if not node:
            return None
        return {"node": node, "connections": []}

    def set_embedding(self, node_id: str, embedding: list[float]):
        self.embeddings.append((node_id, embedding))
        return True


class _CaptureEmbedder:
    def __init__(self):
        self.texts: list[str] = []

    def embed(self, text: str) -> list[float]:
        self.texts.append(text)
        return [0.2] * 384


class TestRoutingEmbeddings:
    def test_add_node_embeds_without_summary(self, monkeypatch):
        storage = _RoutingStorage()
        emb = _CaptureEmbedder()
        monkeypatch.setattr(mcp_server, "storage", storage)
        monkeypatch.setattr(mcp_server, "embedder", emb)

        result = mcp_server.add_node(
            "Person",
            "Girard",
            properties={"relationship": "influential"},
        )

        assert result["success"] is True
        assert len(storage.embeddings) == 1
        assert emb.texts[-1] == "Person: Girard. Relationship: influential"

    def test_update_node_reembeds_when_routing_field_changes(self, monkeypatch):
        storage = _RoutingStorage()
        storage.updated["concept:rl"] = {
            "id": "concept:rl",
            "name": "RL",
            "summary": "Learning from rewards",
            "domain": "AI",
            "_labels": ["Concept"],
        }
        emb = _CaptureEmbedder()
        monkeypatch.setattr(mcp_server, "storage", storage)
        monkeypatch.setattr(mcp_server, "embedder", emb)

        result = mcp_server.update_node("concept:rl", {"domain": "AI"})

        assert result["success"] is True
        assert len(storage.embeddings) == 1
        assert emb.texts[-1] == "Concept: RL. Learning from rewards. Domain: AI"

    def test_update_node_skips_reembed_for_non_routing_fields(self, monkeypatch):
        storage = _RoutingStorage()
        storage.updated["concept:rl"] = {
            "id": "concept:rl",
            "name": "RL",
            "summary": "Learning from rewards",
            "domain": "AI",
            "_labels": ["Concept"],
        }
        emb = _CaptureEmbedder()
        monkeypatch.setattr(mcp_server, "storage", storage)
        monkeypatch.setattr(mcp_server, "embedder", emb)

        result = mcp_server.update_node("concept:rl", {"updated_at": "2026-01-01"})

        assert result["success"] is True
        assert len(storage.embeddings) == 0


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
