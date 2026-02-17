"""Tests for VaultScanner."""

from pathlib import Path
from unittest.mock import Mock

import pytest

from src.graph.scanner import VaultScanner
from src.graph.sync import NoteSynchronizer


class TestVaultScanner:
    @pytest.fixture
    def synchronizer(self):
        return Mock(spec=NoteSynchronizer)

    @pytest.fixture
    def scanner(self, synchronizer):
        return VaultScanner(synchronizer)

    @pytest.fixture
    def vault_path(self, tmp_path):
        """Create a temp vault with some notes."""
        (tmp_path / "Note1.md").write_text("# Note 1")
        (tmp_path / "Note2.md").write_text("# Note 2")
        sub = tmp_path / "Subfolder"
        sub.mkdir()
        (sub / "Note3.md").write_text("# Note 3")
        (tmp_path / ".hidden").mkdir()
        (tmp_path / ".hidden" / "Hidden.md").write_text("# Hidden")
        return tmp_path

    def test_scan_vault_calls_sync_for_all_md_files(
        self, scanner, synchronizer, vault_path
    ):
        """Scanner should find all .md files and call sync_note_from_file."""
        synchronizer.sync_note_from_file.return_value = {
            "success": True,
            "action": "created",
        }

        stats = scanner.scan_vault(vault_path)

        assert stats["processed"] == 3
        assert stats["created"] == 3
        assert stats["errors"] == 0
        assert synchronizer.sync_note_from_file.call_count == 3

        # Check that hidden files were ignored
        calls = [c[0][0] for c in synchronizer.sync_note_from_file.call_args_list]
        paths = [str(p) for p in calls]
        assert any("Note1.md" in p for p in paths)
        assert any("Note2.md" in p for p in paths)
        assert any("Note3.md" in p for p in paths)
        assert not any("Hidden.md" in p for p in paths)

    def test_scan_vault_handles_errors(self, scanner, synchronizer, vault_path):
        """Scanner should count errors and continue."""

        def side_effect(path):
            if "Note2" in str(path):
                return {"success": False, "error": "Fail"}
            return {"success": True, "action": "updated"}

        synchronizer.sync_note_from_file.side_effect = side_effect

        stats = scanner.scan_vault(vault_path)

        assert stats["processed"] == 3
        assert stats["updated"] == 2
        assert stats["errors"] == 1
        assert len(stats["error_details"]) == 1
        assert "Fail" in stats["error_details"][0]

    def test_scan_vault_handles_exceptions(self, scanner, synchronizer, vault_path):
        """Scanner should catch exceptions and continue."""

        def side_effect(path):
            if "Note1" in str(path):
                raise ValueError("Crash")
            return {"success": True}

        synchronizer.sync_note_from_file.side_effect = side_effect

        stats = scanner.scan_vault(vault_path)

        assert stats["processed"] == 2  # Note1 crashes before result
        assert stats["errors"] == 1
        assert "Crash" in stats["error_details"][0]
