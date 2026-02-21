import subprocess
from pathlib import Path

from src.extraction.tracker import NoteTracker
from src.vector.store import VectorStore

from experimental.extract_tui.core import (
    NoteRow,
    load_pending_rows,
    parse_agents,
    parse_models,
    run_extraction_once,
    safe_log_name,
    select_first_n,
    selected_count,
    toggle_select_all,
    verify_post_extract_status,
)


def test_parse_agents_filters_noise():
    raw = """
graph-agent (all)
random noise
plan (primary)
{\"junk\": true}
"""
    assert parse_agents(raw) == ["graph-agent", "plan"]


def test_parse_models_keeps_model_ids():
    raw = """
openai/gpt-5.3-codex
not-a-model
google/gemini-2.5-pro
"""
    assert parse_models(raw) == ["openai/gpt-5.3-codex", "google/gemini-2.5-pro"]


def test_select_first_n_clears_previous_and_selects():
    rows = [
        NoteRow(path="a", status="new", last_extracted_at=None, selected=True),
        NoteRow(path="b", status="changed", last_extracted_at=None, selected=True),
        NoteRow(path="c", status="new", last_extracted_at=None, selected=False),
    ]
    count = select_first_n(rows, 2)
    assert count == 2
    assert [row.selected for row in rows] == [True, True, False]


def test_toggle_select_all_roundtrip():
    rows = [
        NoteRow(path="a", status="new", last_extracted_at=None),
        NoteRow(path="b", status="changed", last_extracted_at=None),
    ]
    toggle_select_all(rows)
    assert selected_count(rows) == 2
    toggle_select_all(rows)
    assert selected_count(rows) == 0


def test_safe_log_name_sanitizes_paths():
    assert safe_log_name("Folder/Note Name.md", 3).startswith(
        "0003_Folder_Note_Name.md"
    )


def test_load_pending_rows_reports_new_and_changed(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    note_a = vault / "a.md"
    note_b = vault / "b.md"
    note_a.write_text("a1", encoding="utf-8")
    note_b.write_text("b1", encoding="utf-8")

    db = tmp_path / "vectors.db"
    tracker = NoteTracker(VectorStore(db))
    tracker.mark_extracted(str(note_b))
    note_b.write_text("b2", encoding="utf-8")

    summary, rows = load_pending_rows(vault, db)
    assert summary.pending == 2
    assert summary.new == 1
    assert summary.changed == 1
    statuses = sorted(row.status for row in rows)
    assert statuses == ["changed", "new"]


def test_run_extraction_once_writes_logs(tmp_path, monkeypatch):
    repo = tmp_path
    log_path = tmp_path / "logs" / "one.log"

    def fake_run(*_args, **_kwargs):
        return subprocess.CompletedProcess(
            args=["opencode"],
            returncode=0,
            stdout="ok out",
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = run_extraction_once(
        repo_root=repo,
        note_path="/tmp/note.md",
        agent="graph-agent",
        model="openai/gpt-5.3-codex",
        log_path=log_path,
    )
    assert result.success is True
    assert log_path.exists()
    log_text = log_path.read_text(encoding="utf-8")
    assert "opencode run --agent graph-agent" in log_text
    assert "ok out" in log_text


def test_verify_post_extract_status(tmp_path):
    db = tmp_path / "vectors.db"
    note = tmp_path / "note.md"
    note.write_text("hello", encoding="utf-8")

    tracker = NoteTracker(VectorStore(db))
    tracker.mark_extracted(str(note))

    assert verify_post_extract_status(str(note), Path(db)) == "unchanged"
