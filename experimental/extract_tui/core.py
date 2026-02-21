"""Core services for the experimental extraction TUI."""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from src.extraction.tracker import NoteTracker
from src.vector.store import VectorStore

DEFAULT_VAULT = Path.home() / "obsidian_emin" / "obsidian_vault"
DEFAULT_DB = Path("data/vectors.experimental.db")
DEFAULT_AGENT = "graph-agent"
DEFAULT_MODEL = "openai/gpt-5.3-codex"

_AGENT_LINE_RE = re.compile(r"^([a-z0-9][a-z0-9-]*)\s+\((?:primary|all)\)$")
_MODEL_LINE_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*/[a-z0-9][a-z0-9._:-]*$")


@dataclass(slots=True)
class VaultSummary:
    total: int
    pending: int
    unchanged: int
    new: int
    changed: int


@dataclass(slots=True)
class NoteRow:
    path: str
    status: str
    last_extracted_at: str | None
    selected: bool = False


@dataclass(slots=True)
class RunResult:
    note_path: str
    success: bool
    returncode: int
    log_path: Path
    post_status: str | None = None


def parse_agents(raw: str) -> list[str]:
    """Parse `opencode agent list` output into unique agent ids."""
    agents: list[str] = []
    for line in raw.splitlines():
        match = _AGENT_LINE_RE.match(line.strip())
        if not match:
            continue
        agent = match.group(1)
        if agent not in agents:
            agents.append(agent)
    return agents


def parse_models(raw: str) -> list[str]:
    """Parse `opencode models` output into model ids."""
    models: list[str] = []
    for line in raw.splitlines():
        text = line.strip()
        if _MODEL_LINE_RE.match(text):
            models.append(text)
    return models


def _run_capture(command: list[str], cwd: Path, timeout: int = 120) -> str:
    result = subprocess.run(
        command,
        cwd=str(cwd),
        text=True,
        capture_output=True,
        check=False,
        timeout=timeout,
    )
    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        raise RuntimeError(stderr or f"Command failed: {' '.join(command)}")
    return result.stdout


def discover_agents(repo_root: Path) -> list[str]:
    """Discover available opencode agents, with safe fallback."""
    try:
        stdout = _run_capture(["opencode", "agent", "list"], cwd=repo_root)
        agents = parse_agents(stdout)
    except Exception:
        agents = []

    if DEFAULT_AGENT not in agents:
        agents.insert(0, DEFAULT_AGENT)
    return agents or [DEFAULT_AGENT]


def discover_models(repo_root: Path) -> list[str]:
    """Discover available opencode models, with safe fallback."""
    try:
        stdout = _run_capture(["opencode", "models"], cwd=repo_root)
        models = parse_models(stdout)
    except Exception:
        models = []

    if not models:
        return [DEFAULT_MODEL]
    if DEFAULT_MODEL in models:
        return [DEFAULT_MODEL] + [m for m in models if m != DEFAULT_MODEL]
    return models


def load_pending_rows(
    vault_path: Path, db_path: Path
) -> tuple[VaultSummary, list[NoteRow]]:
    """Load vault extraction summary + pending rows."""
    tracker = NoteTracker(VectorStore(str(db_path)))
    result = tracker.list_pending_notes(str(vault_path))
    if not result.get("success"):
        error = result.get("error", "unknown error")
        raise RuntimeError(error)

    rows: list[NoteRow] = []
    new_count = 0
    changed_count = 0
    pending_items = result["pending"]

    for item in pending_items:
        path = item["path"]
        status = item["status"]
        stored = tracker.vectors.get_extraction_status(path)
        last_extracted = stored["extracted_at"] if stored else None
        rows.append(
            NoteRow(
                path=path,
                status=status,
                last_extracted_at=last_extracted,
                selected=False,
            )
        )
        if status == "new":
            new_count += 1
        elif status == "changed":
            changed_count += 1

    rows.sort(key=lambda r: (0 if r.status == "changed" else 1, r.path.lower()))

    unchanged_count = int(result["unchanged_count"])
    pending_count = int(result["pending_count"])
    summary = VaultSummary(
        total=pending_count + unchanged_count,
        pending=pending_count,
        unchanged=unchanged_count,
        new=new_count,
        changed=changed_count,
    )
    return summary, rows


def selected_count(rows: list[NoteRow]) -> int:
    return sum(1 for row in rows if row.selected)


def clear_selection(rows: list[NoteRow]) -> None:
    for row in rows:
        row.selected = False


def select_first_n(rows: list[NoteRow], n: int) -> int:
    """Select first N rows after clearing previous selection."""
    clear_selection(rows)
    if n <= 0:
        return 0

    count = 0
    for row in rows:
        if count >= n:
            break
        row.selected = True
        count += 1
    return count


def toggle_select_all(rows: list[NoteRow]) -> None:
    if not rows:
        return
    all_selected = all(row.selected for row in rows)
    for row in rows:
        row.selected = not all_selected


def build_extract_prompt(note_path: str) -> str:
    return (
        "Extract knowledge from this note. "
        "Follow the check -> extract -> mark workflow. "
        f"Note path: {note_path}"
    )


def make_run_log_dir(repo_root: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = repo_root / "data" / "extraction-logs" / f"tui_{timestamp}"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def safe_log_name(note_path: str, index: int) -> str:
    base = re.sub(r"[^A-Za-z0-9._-]+", "_", note_path).strip("_")
    if not base:
        base = "note"
    if len(base) > 180:
        base = base[:180]
    return f"{index:04d}_{base}.log"


def run_extraction_once(
    *,
    repo_root: Path,
    note_path: str,
    agent: str,
    model: str | None,
    log_path: Path,
    timeout_seconds: int = 900,
) -> RunResult:
    """Run one opencode extraction command and persist output."""
    command = ["opencode", "run", "--agent", agent]
    if model:
        command.extend(["--model", model])
    command.append(build_extract_prompt(note_path))

    timed_out = False
    try:
        result = subprocess.run(
            command,
            cwd=str(repo_root),
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        timeout_stdout = (
            exc.stdout.decode("utf-8", errors="replace")
            if isinstance(exc.stdout, bytes)
            else (exc.stdout or "")
        )
        timeout_stderr = (
            exc.stderr.decode("utf-8", errors="replace")
            if isinstance(exc.stderr, bytes)
            else (exc.stderr or "")
        )
        result = subprocess.CompletedProcess(
            args=command,
            returncode=124,
            stdout=timeout_stdout,
            stderr=timeout_stderr + f"\nTimed out after {timeout_seconds}s",
        )

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_lines = [f"$ {' '.join(command)}", "", result.stdout or "", result.stderr or ""]
    log_path.write_text("\n".join(log_lines), encoding="utf-8")

    return RunResult(
        note_path=note_path,
        success=(result.returncode == 0) and (not timed_out),
        returncode=result.returncode,
        log_path=log_path,
    )


def verify_post_extract_status(note_path: str, db_path: Path) -> str:
    """Return post-run tracker status for the note."""
    tracker = NoteTracker(VectorStore(str(db_path)))
    result = tracker.check_note_status(note_path)
    return str(result.get("status", "error"))


def short_time(iso_timestamp: str | None) -> str:
    if not iso_timestamp:
        return "never"
    return iso_timestamp.replace("T", " ").replace("+00:00", "Z")


def relative_path(path: str, root: Path) -> str:
    full = Path(path)
    try:
        return str(full.relative_to(root))
    except ValueError:
        return str(full)
