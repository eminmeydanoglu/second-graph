"""Tests for embedding routing text builder."""

from src.graph.routing_text import build_routing_text, MAX_ROUTING_TEXT_CHARS


def test_note_with_summary():
    text = build_routing_text(
        "Note",
        {
            "name": "Mimetic Theory",
            "summary": "Core concepts by Girard",
            "tags": ["philosophy"],
        },
    )
    assert text == "Note: Mimetic Theory. Core concepts by Girard. Tags: philosophy"


def test_note_without_summary():
    text = build_routing_text(
        "Note",
        {
            "name": "Mimetic Theory",
            "tags": ["philosophy"],
        },
    )
    assert text == "Note: Mimetic Theory. Tags: philosophy"


def test_person_with_summary():
    text = build_routing_text(
        "Person",
        {
            "name": "Girard",
            "summary": "French thinker known for mimetic desire",
        },
    )
    assert text == "Person: Girard. French thinker known for mimetic desire"


def test_person_minimal():
    text = build_routing_text("Person", {"name": "Girard"})
    assert text == "Person: Girard"


def test_concept_with_domain():
    text = build_routing_text(
        "Concept",
        {"name": "RL", "summary": "Learning from rewards", "domain": "AI"},
    )
    assert text == "Concept: RL. Learning from rewards. Domain: AI"


def test_goal_full():
    text = build_routing_text(
        "Goal",
        {
            "name": "Build AGI",
            "summary": "Long-term objective",
            "status": "active",
            "horizon": "life",
        },
    )
    assert text == "Goal: Build AGI. Long-term objective. Status: active. Horizon: life"


def test_unknown_type_fallback():
    text = build_routing_text("Weird", {"name": "something"})
    assert text == "Weird: something"


def test_empty_props_note():
    text = build_routing_text("Note", {})
    assert text == "Note:"


def test_routing_text_is_truncated():
    long_summary = "x" * (MAX_ROUTING_TEXT_CHARS + 50)
    text = build_routing_text("Concept", {"name": "Long", "summary": long_summary})
    assert len(text) == MAX_ROUTING_TEXT_CHARS
    assert text.endswith("...")
