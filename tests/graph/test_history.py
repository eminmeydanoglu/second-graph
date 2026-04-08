"""Tests for graph snapshot history utilities."""

from pathlib import Path
import json
import pytest

from src.graph.history import SnapshotRepository, diff_snapshots, graph_hash


def test_snapshot_repository_create_and_load(tmp_path: Path):
    repo = SnapshotRepository(tmp_path / "graph-history")

    snapshot = {
        "nodes": [
            {
                "id": "person:emin",
                "labels": ["Person"],
                "properties": {"id": "person:emin", "name": "Emin"},
            }
        ],
        "edges": [],
    }

    meta = repo.create(snapshot, message="first")

    assert meta.snapshot_id
    assert meta.nodes == 1
    assert repo.get_head() == meta.snapshot_id

    loaded = repo.load(meta.snapshot_id)
    assert len(loaded["nodes"]) == 1
    assert loaded["nodes"][0]["id"] == "person:emin"

    rows = repo.list(limit=10)
    assert len(rows) == 1
    assert rows[0].snapshot_id == meta.snapshot_id


def test_graph_hash_stable_for_ordering():
    a = {
        "nodes": [
            {
                "id": "b",
                "labels": ["Concept"],
                "properties": {"id": "b", "name": "B"},
            },
            {
                "id": "a",
                "labels": ["Concept"],
                "properties": {"id": "a", "name": "A"},
            },
        ],
        "edges": [
            {
                "from_id": "a",
                "to_id": "b",
                "relation": "RELATED_TO",
                "properties": {"id": "edge:1"},
            }
        ],
    }

    b = {
        "nodes": list(reversed(a["nodes"])),
        "edges": list(reversed(a["edges"])),
    }

    assert graph_hash(a) == graph_hash(b)


def test_diff_snapshots_detects_changes():
    old = {
        "nodes": [
            {
                "id": "person:emin",
                "labels": ["Person"],
                "properties": {"id": "person:emin", "name": "Emin"},
            }
        ],
        "edges": [],
    }

    new = {
        "nodes": [
            {
                "id": "person:emin",
                "labels": ["Person"],
                "properties": {
                    "id": "person:emin",
                    "name": "Emin",
                    "summary": "AI researcher",
                },
            },
            {
                "id": "concept:rl",
                "labels": ["Concept"],
                "properties": {"id": "concept:rl", "name": "RL"},
            },
        ],
        "edges": [
            {
                "from_id": "person:emin",
                "to_id": "concept:rl",
                "relation": "INTERESTED_IN",
                "properties": {"id": "edge:rl"},
            }
        ],
    }

    diff = diff_snapshots(old, new)

    assert diff["summary"]["added_nodes"] == 1
    assert diff["summary"]["updated_nodes"] == 1
    assert diff["summary"]["added_edges"] == 1


def test_snapshot_ids_unique_even_for_same_graph(tmp_path: Path):
    repo = SnapshotRepository(tmp_path / "graph-history")
    snapshot = {
        "nodes": [
            {
                "id": "concept:x",
                "labels": ["Concept"],
                "properties": {"id": "concept:x", "name": "X"},
            }
        ],
        "edges": [],
    }

    s1 = repo.create(snapshot, message="same")
    s2 = repo.create(snapshot, message="same")

    assert s1.snapshot_id != s2.snapshot_id


def test_load_fails_on_hash_mismatch(tmp_path: Path):
    repo = SnapshotRepository(tmp_path / "graph-history")
    snapshot = {
        "nodes": [
            {
                "id": "concept:y",
                "labels": ["Concept"],
                "properties": {"id": "concept:y", "name": "Y"},
            }
        ],
        "edges": [],
    }

    meta = repo.create(snapshot, message="tamper-test")
    path = repo.resolve_snapshot_path(meta.snapshot_id)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["nodes"][0]["properties"]["name"] = "Tampered"
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    with pytest.raises(ValueError):
        repo.load(meta.snapshot_id)
