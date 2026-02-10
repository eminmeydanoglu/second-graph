"""Tests for NoteTracker extraction tracking system."""

import tempfile
from pathlib import Path

import pytest

from src.extraction.tracker import NoteTracker
from src.vector.store import VectorStore


@pytest.fixture
def tmp_db(tmp_path):
    """Create a temporary VectorStore."""
    db_path = tmp_path / "test_vectors.db"
    return VectorStore(db_path)


@pytest.fixture
def tracker(tmp_db):
    """Create a NoteTracker with temporary storage."""
    return NoteTracker(tmp_db)


@pytest.fixture
def sample_vault(tmp_path):
    """Create a temporary vault with some notes."""
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "note_a.md").write_text(
        "# Note A\nSome content about AI.", encoding="utf-8"
    )
    (vault / "note_b.md").write_text(
        "# Note B\nSome content about RL.", encoding="utf-8"
    )
    # Create excluded dir
    (vault / ".obsidian").mkdir()
    (vault / ".obsidian" / "config.md").write_text("config", encoding="utf-8")
    return vault


class TestHashContent:
    def test_deterministic(self):
        h1 = NoteTracker.hash_content("hello world")
        h2 = NoteTracker.hash_content("hello world")
        assert h1 == h2

    def test_different_content(self):
        h1 = NoteTracker.hash_content("hello")
        h2 = NoteTracker.hash_content("world")
        assert h1 != h2

    def test_sha256_length(self):
        h = NoteTracker.hash_content("test")
        assert len(h) == 64  # SHA256 hex digest is 64 chars


class TestMakeDiff:
    def test_no_change(self):
        diff = NoteTracker.make_diff("hello\n", "hello\n", "test.md")
        assert diff == ""

    def test_addition(self):
        diff = NoteTracker.make_diff("line1\n", "line1\nline2\n", "test.md")
        assert "+line2" in diff
        assert "a/test.md" in diff
        assert "b/test.md" in diff

    def test_removal(self):
        diff = NoteTracker.make_diff("line1\nline2\n", "line1\n", "test.md")
        assert "-line2" in diff

    def test_modification(self):
        diff = NoteTracker.make_diff("old content\n", "new content\n", "test.md")
        assert "-old content" in diff
        assert "+new content" in diff


class TestCheckNoteStatus:
    def test_new_note(self, tracker, tmp_path):
        note = tmp_path / "new.md"
        note.write_text("# New\nBrand new content.", encoding="utf-8")

        result = tracker.check_note_status(str(note))
        assert result["status"] == "new"
        assert "# New" in result["content"]

    def test_file_not_found(self, tracker):
        result = tracker.check_note_status("/nonexistent/path.md")
        assert result["status"] == "error"
        assert "not found" in result["error"].lower()

    def test_unchanged_after_mark(self, tracker, tmp_path):
        note = tmp_path / "note.md"
        note.write_text("content", encoding="utf-8")

        tracker.mark_extracted(str(note))
        result = tracker.check_note_status(str(note))

        assert result["status"] == "unchanged"
        assert "last_extracted_at" in result

    def test_changed_after_modify(self, tracker, tmp_path):
        note = tmp_path / "note.md"
        note.write_text("original content\n", encoding="utf-8")

        tracker.mark_extracted(str(note))
        note.write_text("modified content\n", encoding="utf-8")

        result = tracker.check_note_status(str(note))
        assert result["status"] == "changed"
        assert "diff" in result
        assert "content" in result
        assert "-original content" in result["diff"]
        assert "+modified content" in result["diff"]
        assert "last_extracted_at" in result


class TestMarkExtracted:
    def test_first_extraction(self, tracker, tmp_path):
        note = tmp_path / "note.md"
        note.write_text("first content", encoding="utf-8")

        result = tracker.mark_extracted(str(note))
        assert result["success"] is True
        assert "content_hash" in result
        assert "extracted_at" in result

    def test_file_not_found(self, tracker):
        result = tracker.mark_extracted("/nonexistent.md")
        assert result["success"] is False

    def test_stores_snapshot(self, tracker, tmp_path):
        note = tmp_path / "note.md"
        note.write_text("snapshot content", encoding="utf-8")

        tracker.mark_extracted(str(note))
        snapshot = tracker.vectors.get_note_snapshot(str(note))
        assert snapshot == "snapshot content"

    def test_updates_snapshot_on_change(self, tracker, tmp_path):
        note = tmp_path / "note.md"
        note.write_text("v1", encoding="utf-8")
        tracker.mark_extracted(str(note))

        note.write_text("v2", encoding="utf-8")
        tracker.mark_extracted(str(note))

        snapshot = tracker.vectors.get_note_snapshot(str(note))
        assert snapshot == "v2"

    def test_records_diff_on_change(self, tracker, tmp_path):
        note = tmp_path / "note.md"
        note.write_text("original\n", encoding="utf-8")
        tracker.mark_extracted(str(note))

        note.write_text("modified\n", encoding="utf-8")
        tracker.mark_extracted(str(note))

        diffs = tracker.vectors.get_extraction_diffs(str(note))
        assert len(diffs) == 2  # "new" + "changed"
        assert diffs[0]["status"] == "changed"
        assert diffs[1]["status"] == "new"

    def test_no_diff_when_unchanged(self, tracker, tmp_path):
        note = tmp_path / "note.md"
        note.write_text("same content", encoding="utf-8")

        tracker.mark_extracted(str(note))
        tracker.mark_extracted(str(note))  # same content

        diffs = tracker.vectors.get_extraction_diffs(str(note))
        assert len(diffs) == 1  # only the initial "new" record


class TestListPendingNotes:
    def test_all_new(self, tracker, sample_vault):
        result = tracker.list_pending_notes(str(sample_vault))
        assert result["success"] is True
        assert result["pending_count"] == 2
        assert result["unchanged_count"] == 0
        statuses = {p["status"] for p in result["pending"]}
        assert statuses == {"new"}

    def test_invalid_vault_path(self, tracker):
        result = tracker.list_pending_notes("/nonexistent/vault")
        assert result["success"] is False

    def test_after_extraction(self, tracker, sample_vault):
        # Extract all notes
        for p in result_paths(tracker.list_pending_notes(str(sample_vault))):
            tracker.mark_extracted(p)

        result = tracker.list_pending_notes(str(sample_vault))
        assert result["pending_count"] == 0
        assert result["unchanged_count"] == 2

    def test_detect_changed(self, tracker, sample_vault):
        # Extract all
        for p in result_paths(tracker.list_pending_notes(str(sample_vault))):
            tracker.mark_extracted(p)

        # Modify one
        note_a = sample_vault / "note_a.md"
        note_a.write_text("# Note A\nUpdated content.", encoding="utf-8")

        result = tracker.list_pending_notes(str(sample_vault))
        assert result["pending_count"] == 1
        assert result["unchanged_count"] == 1
        assert result["pending"][0]["status"] == "changed"

    def test_excludes_obsidian_dirs(self, tracker, sample_vault):
        result = tracker.list_pending_notes(str(sample_vault))
        paths = [p["path"] for p in result["pending"]]
        assert not any(".obsidian" in p for p in paths)


def result_paths(result: dict) -> list[str]:
    """Helper to extract paths from list_pending_notes result."""
    return [p["path"] for p in result["pending"]]
