"""Vault scanner to synchronize all markdown files with the graph."""

import logging
from pathlib import Path
from typing import Iterator

from .sync import NoteSynchronizer

log = logging.getLogger(__name__)


class VaultScanner:
    """Scans a vault directory and syncs all notes to the graph."""

    def __init__(self, synchronizer: NoteSynchronizer):
        self.synchronizer = synchronizer

    def scan_vault(self, vault_path: str | Path) -> dict:
        """Scan all markdown files in the vault and sync them.

        Args:
            vault_path: Root directory of the vault

        Returns:
            Stats about the scan (processed, created, updated, errors)
        """
        root = Path(vault_path)
        if not root.exists():
            return {"success": False, "error": f"Vault path not found: {root}"}

        stats = {
            "processed": 0,
            "created": 0,
            "updated": 0,
            "errors": 0,
            "error_details": [],
        }

        log.info(f"Starting vault scan at {root}")

        for note_path in self._find_markdown_files(root):
            try:
                result = self.synchronizer.sync_note_from_file(note_path)
                stats["processed"] += 1

                if result.get("success"):
                    action = result.get("action", "unknown")
                    if action == "created":
                        stats["created"] += 1
                    elif action == "updated":
                        stats["updated"] += 1
                else:
                    stats["errors"] += 1
                    stats["error_details"].append(
                        f"{note_path.name}: {result.get('error')}"
                    )

            except Exception as e:
                log.error(f"Failed to sync {note_path}: {e}")
                stats["errors"] += 1
                stats["error_details"].append(f"{note_path.name}: {str(e)}")

        log.info(f"Scan complete. Stats: {stats}")
        return stats

    def _find_markdown_files(self, root: Path) -> Iterator[Path]:
        """Recursively find all .md files, ignoring hidden/system dirs."""
        for path in root.rglob("*.md"):
            # Skip .git, .obsidian, etc.
            if any(part.startswith(".") for part in path.parts):
                continue
            yield path
