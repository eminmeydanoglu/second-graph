"""Note extraction tracking system.

Tracks which notes have been extracted, detects changes via SHA256 content
hashing, and generates unified diffs for changed notes. All state is stored
in SQLite (via VectorStore) — no Neo4j dependency for tracking.
"""

import difflib
import hashlib
from datetime import datetime, timezone
from pathlib import Path

from ..parser.vault import iter_notes
from ..vector.store import VectorStore


class NoteTracker:
    """Tracks extraction status of Obsidian notes.

    Uses SQLite-backed VectorStore for persistence:
    - note_extractions table: path, content_hash, extracted_at, content
    - extraction_diffs table: audit trail of changes between extractions
    """

    def __init__(self, vectors: VectorStore):
        self.vectors = vectors

    @staticmethod
    def hash_content(content: str) -> str:
        """SHA256 hash of content string."""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    @staticmethod
    def make_diff(old_content: str, new_content: str, path: str) -> str:
        """Generate unified diff between old and new content."""
        old_lines = old_content.splitlines(keepends=True)
        new_lines = new_content.splitlines(keepends=True)
        diff = difflib.unified_diff(
            old_lines, new_lines, fromfile=f"a/{path}", tofile=f"b/{path}"
        )
        return "".join(diff)

    def check_note_status(self, path: str) -> dict:
        """Check if a note needs extraction.

        Reads the file, hashes it, compares with stored hash.

        Returns:
            - status="new": {status, content} — first extraction
            - status="changed": {status, diff, content, last_extracted_at}
            - status="unchanged": {status, last_extracted_at}
            - status="error": {status, error} — file not found, etc.
        """
        file_path = Path(path)
        if not file_path.exists():
            return {"status": "error", "error": f"File not found: {path}"}

        try:
            content = file_path.read_text(encoding="utf-8")
        except Exception as e:
            return {"status": "error", "error": f"Failed to read file: {e}"}

        current_hash = self.hash_content(content)
        stored = self.vectors.get_extraction_status(path)

        if stored is None:
            return {"status": "new", "content": content}

        if stored["content_hash"] == current_hash:
            return {
                "status": "unchanged",
                "last_extracted_at": stored["extracted_at"],
            }

        # Changed — generate diff
        old_content = self.vectors.get_note_snapshot(path) or ""
        diff = self.make_diff(old_content, content, path)

        return {
            "status": "changed",
            "diff": diff,
            "content": content,
            "last_extracted_at": stored["extracted_at"],
        }

    def mark_extracted(self, path: str) -> dict:
        """Mark a note as successfully extracted.

        Reads the file, stamps hash + timestamp in SQLite,
        stores content snapshot, and records diff if changed.

        Returns:
            Dict with success status, hash, extracted_at.
        """
        file_path = Path(path)
        if not file_path.exists():
            return {"success": False, "error": f"File not found: {path}"}

        try:
            content = file_path.read_text(encoding="utf-8")
        except Exception as e:
            return {"success": False, "error": f"Failed to read file: {e}"}

        current_hash = self.hash_content(content)
        now = datetime.now(timezone.utc).isoformat()

        # Check previous state for diff recording
        stored = self.vectors.get_extraction_status(path)
        old_hash = stored["content_hash"] if stored else None

        if old_hash and old_hash != current_hash:
            old_content = self.vectors.get_note_snapshot(path) or ""
            diff = self.make_diff(old_content, content, path)
            self.vectors.store_extraction_diff(
                path=path,
                extracted_at=now,
                old_hash=old_hash,
                new_hash=current_hash,
                diff=diff,
                status="changed",
            )
        elif old_hash is None:
            # First extraction — record with empty diff
            self.vectors.store_extraction_diff(
                path=path,
                extracted_at=now,
                old_hash=None,
                new_hash=current_hash,
                diff="",
                status="new",
            )

        # Upsert extraction record
        self.vectors.upsert_extraction(path, current_hash, now, content)

        return {
            "success": True,
            "content_hash": current_hash,
            "extracted_at": now,
        }

    def list_pending_notes(self, vault_path: str) -> dict:
        """List notes in vault that need extraction (new or changed).

        Iterates all markdown files, hashes each, compares with stored state.

        Returns:
            Dict with pending list and unchanged_count.
        """
        vault = Path(vault_path)
        if not vault.is_dir():
            return {
                "success": False,
                "error": f"Vault path not found: {vault_path}",
            }

        pending: list[dict] = []
        unchanged_count = 0

        for note_path in iter_notes(vault):
            path_str = str(note_path)

            try:
                content = note_path.read_text(encoding="utf-8")
            except Exception:
                continue

            current_hash = self.hash_content(content)
            stored = self.vectors.get_extraction_status(path_str)

            if stored is None:
                pending.append({"path": path_str, "status": "new"})
            elif stored["content_hash"] != current_hash:
                pending.append({"path": path_str, "status": "changed"})
            else:
                unchanged_count += 1

        return {
            "success": True,
            "pending": pending,
            "pending_count": len(pending),
            "unchanged_count": unchanged_count,
        }
