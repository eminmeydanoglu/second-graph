"""Vault traversal and indexing."""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from .markdown import ParsedNote, parse_note


# Folders to exclude from indexing
DEFAULT_EXCLUDED = {
    ".obsidian",
    ".git",
    ".trash",
    "Templates",
    ".smart-connections",
}


@dataclass
class VaultStats:
    """Statistics about the vault."""

    total_notes: int = 0
    total_wikilinks: int = 0
    total_tags: int = 0
    unique_tags: set = None

    def __post_init__(self):
        if self.unique_tags is None:
            self.unique_tags = set()

    def __str__(self) -> str:
        return (
            f"Vault Stats:\n"
            f"  Notes: {self.total_notes}\n"
            f"  Wikilinks: {self.total_wikilinks}\n"
            f"  Tags: {self.total_tags} ({len(self.unique_tags)} unique)"
        )


def iter_notes(
    vault_path: Path,
    excluded: set[str] | None = None,
) -> Iterator[Path]:
    """Iterate over all markdown files in a vault.

    Args:
        vault_path: Path to the Obsidian vault root
        excluded: Set of folder names to skip (defaults to DEFAULT_EXCLUDED)

    Yields:
        Path objects for each .md file
    """
    if excluded is None:
        excluded = DEFAULT_EXCLUDED

    for path in vault_path.rglob("*.md"):
        # Check if any parent folder is excluded
        if any(part in excluded for part in path.parts):
            continue
        yield path


def parse_vault(
    vault_path: Path,
    excluded: set[str] | None = None,
) -> Iterator[ParsedNote]:
    """Parse all notes in a vault.

    Args:
        vault_path: Path to the Obsidian vault root
        excluded: Set of folder names to skip

    Yields:
        ParsedNote objects for each note
    """
    vault_path = Path(vault_path)

    for path in iter_notes(vault_path, excluded):
        try:
            yield parse_note(path, vault_root=vault_path)
        except Exception as e:
            print(f"Warning: Failed to parse {path}: {e}")
            continue


def get_vault_stats(vault_path: Path, excluded: set[str] | None = None) -> VaultStats:
    """Get statistics about a vault.

    Args:
        vault_path: Path to the Obsidian vault root
        excluded: Set of folder names to skip

    Returns:
        VaultStats with counts
    """
    stats = VaultStats()

    for note in parse_vault(vault_path, excluded):
        stats.total_notes += 1
        stats.total_wikilinks += len(note.wikilinks)
        stats.total_tags += len(note.tags)
        stats.unique_tags.update(note.tags)

    return stats
