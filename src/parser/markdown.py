"""Markdown parser for Obsidian notes."""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ParsedNote:
    """Parsed representation of an Obsidian note."""

    path: Path
    title: str
    content: str
    frontmatter: dict[str, Any] = field(default_factory=dict)
    wikilinks: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)

    @property
    def folder(self) -> str:
        """Get the parent folder name."""
        return self.path.parent.name if self.path.parent.name else "root"

    @property
    def relative_path(self) -> str:
        """Get path relative to vault root."""
        return str(self.path)


# Regex patterns
FRONTMATTER_PATTERN = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
WIKILINK_PATTERN = re.compile(r"\[\[([^\]|]+)(?:\|[^\]]+)?\]\]")
TAG_PATTERN = re.compile(r"(?:^|\s)#([a-zA-Z0-9_/-]+)")
FRONTMATTER_TAG_PATTERN = re.compile(r"^\s*-?\s*['\"]?([a-zA-Z0-9_/-]+)['\"]?\s*$")


def parse_frontmatter(content: str) -> tuple[dict[str, Any], str]:
    """Extract YAML frontmatter from note content.

    Returns:
        Tuple of (frontmatter dict, remaining content)
    """
    match = FRONTMATTER_PATTERN.match(content)
    if not match:
        return {}, content

    try:
        frontmatter = yaml.safe_load(match.group(1)) or {}
    except yaml.YAMLError:
        frontmatter = {}

    remaining = content[match.end() :]
    return frontmatter, remaining


def extract_wikilinks(content: str) -> list[str]:
    """Extract all [[wikilinks]] from content.

    Handles:
    - [[Simple Link]]
    - [[Link|Display Text]]
    - [[Folder/Nested Link]]

    Returns:
        List of unique link targets (without display text)
    """
    matches = WIKILINK_PATTERN.findall(content)
    # Deduplicate while preserving order
    seen = set()
    unique = []
    for link in matches:
        if link not in seen:
            seen.add(link)
            unique.append(link)
    return unique


def extract_tags(content: str, frontmatter: dict[str, Any]) -> list[str]:
    """Extract all #tags from content and frontmatter.

    Returns:
        List of unique tags (without # prefix)
    """
    tags = set()

    # Tags from content
    for match in TAG_PATTERN.finditer(content):
        tags.add(match.group(1))

    # Tags from frontmatter
    fm_tags = frontmatter.get("tags", [])
    if isinstance(fm_tags, list):
        for tag in fm_tags:
            if isinstance(tag, str):
                tags.add(tag.strip().lstrip("#"))
    elif isinstance(fm_tags, str):
        tags.add(fm_tags.strip().lstrip("#"))

    return sorted(tags)


def extract_title(path: Path, frontmatter: dict[str, Any], content: str) -> str:
    """Extract note title from frontmatter, first heading, or filename.

    Priority:
    1. frontmatter.title
    2. First # heading
    3. Filename (without .md)
    """
    # From frontmatter
    if "title" in frontmatter:
        return str(frontmatter["title"])

    # From first heading
    heading_match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
    if heading_match:
        return heading_match.group(1).strip()

    # From filename
    return path.stem


def parse_note(path: Path, vault_root: Path | None = None) -> ParsedNote:
    """Parse an Obsidian markdown note.

    Args:
        path: Path to the .md file
        vault_root: Optional vault root for relative paths

    Returns:
        ParsedNote with extracted metadata
    """
    content = path.read_text(encoding="utf-8")
    frontmatter, body = parse_frontmatter(content)

    # Use relative path if vault_root provided
    if vault_root:
        rel_path = path.relative_to(vault_root)
    else:
        rel_path = path

    return ParsedNote(
        path=rel_path,
        title=extract_title(path, frontmatter, body),
        content=body,
        frontmatter=frontmatter,
        wikilinks=extract_wikilinks(
            content
        ),  # Check full content for links in frontmatter
        tags=extract_tags(body, frontmatter),
    )
