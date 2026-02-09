"""Tests for VaultGraph resolution logic."""

from pathlib import Path

import pytest

from src.graph.builder import VaultGraph
from src.parser.markdown import ParsedNote


@pytest.fixture
def graph():
    """Create a fresh VaultGraph instance."""
    return VaultGraph()


def create_note(
    title: str, wikilinks: list[str] | None = None, path: str | None = None
) -> ParsedNote:
    """Helper to create a ParsedNote with minimal required fields."""
    if path is None:
        path = f"{title}.md"
    return ParsedNote(
        path=Path(path),
        title=title,
        content=f"Content for {title}",
        wikilinks=wikilinks or [],
        tags=[],
        frontmatter={},
    )


def test_basic_resolution(graph):
    """Test resolving a placeholder to a real note."""
    # 1. Create Note A linking to Note B
    note_a = create_note("Note A", wikilinks=["Note B"])
    graph.add_note(note_a)

    # Verify placeholder created
    assert "Note A.md" in graph.graph
    placeholder_id = "note:Note B"
    assert placeholder_id in graph.graph
    assert graph.graph.nodes[placeholder_id]["placeholder"] is True
    assert graph.graph.has_edge("Note A.md", placeholder_id)

    # 2. Add Note B
    note_b = create_note("Note B")
    graph.add_note(note_b)

    # Verify Note B exists but edge still points to placeholder
    assert "Note B.md" in graph.graph
    assert graph.graph.has_edge("Note A.md", placeholder_id)
    assert not graph.graph.has_edge("Note A.md", "Note B.md")

    # 3. Resolve placeholders
    resolved_count = graph.resolve_placeholders()

    # Verify resolution
    assert resolved_count == 1
    assert placeholder_id not in graph.graph
    assert graph.graph.has_edge("Note A.md", "Note B.md")


def test_case_insensitivity(graph):
    """Test resolving a link with different casing."""
    # 1. Note A links to 'note b' (lowercase)
    note_a = create_note("Note A", wikilinks=["note b"])
    graph.add_note(note_a)

    # Verify lowercase placeholder
    placeholder_id = "note:note b"
    assert placeholder_id in graph.graph
    assert graph.graph.nodes[placeholder_id]["placeholder"] is True

    # 2. Add real Note B (Title Case)
    note_b = create_note("Note B")
    graph.add_note(note_b)

    # 3. Resolve
    resolved_count = graph.resolve_placeholders()

    # Verify resolution to Title Case note
    assert resolved_count == 1
    assert placeholder_id not in graph.graph
    assert graph.graph.has_edge("Note A.md", "Note B.md")


def test_unresolved_placeholder(graph):
    """Test that non-existent notes remain as placeholders."""
    # 1. Note A links to NonExistent
    note_a = create_note("Note A", wikilinks=["NonExistent"])
    graph.add_note(note_a)

    placeholder_id = "note:NonExistent"
    assert placeholder_id in graph.graph

    # 2. Resolve (should do nothing for this node)
    resolved_count = graph.resolve_placeholders()

    assert resolved_count == 0
    assert placeholder_id in graph.graph
    assert graph.graph.nodes[placeholder_id]["placeholder"] is True
    assert graph.graph.has_edge("Note A.md", placeholder_id)


def test_self_reference_handling(graph):
    """Test that self-referential links are ignored."""
    # 1. Note A links to itself
    note_a = create_note("Note A", wikilinks=["Note A"])
    graph.add_note(note_a)

    assert "Note A.md" in graph.graph
    assert not graph.graph.has_edge("Note A.md", "Note A.md")

    # Also verify no placeholder created
    assert "note:Note A" not in graph.graph


def test_circular_resolution(graph):
    """Test resolving two notes that link to each other (added in reverse order)."""
    # 1. Note A links to B (B doesn't exist yet)
    note_a = create_note("Note A", wikilinks=["Note B"])
    graph.add_note(note_a)

    # 2. Note B links to A (A exists)
    note_b = create_note("Note B", wikilinks=["Note A"])
    graph.add_note(note_b)

    # At this point:
    # A -> placeholder(B)
    # B -> A (resolved immediately because A exists)

    assert graph.graph.has_edge("Note B.md", "Note A.md")
    assert graph.graph.has_edge("Note A.md", "note:Note B")

    # 3. Resolve
    resolved_count = graph.resolve_placeholders()

    assert resolved_count == 1
    assert graph.graph.has_edge("Note A.md", "Note B.md")
    assert graph.graph.has_edge("Note B.md", "Note A.md")
