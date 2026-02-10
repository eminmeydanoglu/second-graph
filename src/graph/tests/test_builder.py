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
    note_a = create_note("Note A", wikilinks=["Note B"])
    graph.add_note(note_a)

    assert "Note A.md" in graph.graph
    placeholder_id = "note:Note B"
    assert placeholder_id in graph.graph
    assert graph.graph.nodes[placeholder_id]["placeholder"] is True
    assert graph.graph.has_edge("Note A.md", placeholder_id)

    note_b = create_note("Note B")
    graph.add_note(note_b)

    assert "Note B.md" in graph.graph
    assert graph.graph.has_edge("Note A.md", placeholder_id)
    assert not graph.graph.has_edge("Note A.md", "Note B.md")

    resolved_count = graph.resolve_placeholders()

    assert resolved_count == 1
    assert placeholder_id not in graph.graph
    assert graph.graph.has_edge("Note A.md", "Note B.md")


def test_case_insensitivity(graph):
    """Test resolving a link with different casing."""
    note_a = create_note("Note A", wikilinks=["note b"])
    graph.add_note(note_a)

    placeholder_id = "note:note b"
    assert placeholder_id in graph.graph
    assert graph.graph.nodes[placeholder_id]["placeholder"] is True

    note_b = create_note("Note B")
    graph.add_note(note_b)

    resolved_count = graph.resolve_placeholders()

    assert resolved_count == 1
    assert placeholder_id not in graph.graph
    assert graph.graph.has_edge("Note A.md", "Note B.md")


def test_unresolved_placeholder(graph):
    """Test that non-existent notes remain as placeholders."""
    note_a = create_note("Note A", wikilinks=["NonExistent"])
    graph.add_note(note_a)

    placeholder_id = "note:NonExistent"
    assert placeholder_id in graph.graph

    resolved_count = graph.resolve_placeholders()

    assert resolved_count == 0
    assert placeholder_id in graph.graph
    assert graph.graph.nodes[placeholder_id]["placeholder"] is True
    assert graph.graph.has_edge("Note A.md", placeholder_id)


def test_self_reference_handling(graph):
    """Test that self-referential links are ignored."""
    note_a = create_note("Note A", wikilinks=["Note A"])
    graph.add_note(note_a)

    assert "Note A.md" in graph.graph
    assert not graph.graph.has_edge("Note A.md", "Note A.md")

    assert "note:Note A" not in graph.graph


def test_circular_resolution(graph):
    """Test resolving two notes that link to each other (added in reverse order)."""
    note_a = create_note("Note A", wikilinks=["Note B"])
    graph.add_note(note_a)

    note_b = create_note("Note B", wikilinks=["Note A"])
    graph.add_note(note_b)

    assert graph.graph.has_edge("Note B.md", "Note A.md")
    assert graph.graph.has_edge("Note A.md", "note:Note B")

    resolved_count = graph.resolve_placeholders()

    assert resolved_count == 1
    assert graph.graph.has_edge("Note A.md", "Note B.md")
    assert graph.graph.has_edge("Note B.md", "Note A.md")
