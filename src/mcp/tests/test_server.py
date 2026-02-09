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
