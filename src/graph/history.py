"""Graph snapshot repository and diff utilities."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any


def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")


def _canonical_dumps(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def normalize_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    nodes = snapshot.get("nodes", [])
    edges = snapshot.get("edges", [])

    normalized_nodes = sorted(
        [
            {
                "id": n.get("id"),
                "labels": sorted(n.get("labels", [])),
                "properties": n.get("properties", {}),
            }
            for n in nodes
            if n.get("id")
        ],
        key=lambda n: n["id"],
    )

    normalized_edges = sorted(
        [
            {
                "from_id": e.get("from_id"),
                "to_id": e.get("to_id"),
                "relation": e.get("relation"),
                "properties": e.get("properties", {}),
            }
            for e in edges
            if e.get("from_id") and e.get("to_id") and e.get("relation")
        ],
        key=lambda e: (
            e["from_id"],
            e["relation"],
            e["to_id"],
            e["properties"].get("id", ""),
        ),
    )

    return {
        "nodes": normalized_nodes,
        "edges": normalized_edges,
    }


def graph_hash(snapshot: dict[str, Any]) -> str:
    normalized = normalize_snapshot(snapshot)
    payload = _canonical_dumps(normalized).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass
class SnapshotMeta:
    snapshot_id: str
    created_at: str
    message: str
    graph_hash: str
    parent: str | None
    nodes: int
    edges: int
    file: str


class SnapshotRepository:
    def __init__(self, history_dir: str | Path = "data/graph-history"):
        self.root = Path(history_dir)
        self.snapshots_dir = self.root / "snapshots"
        self.index_path = self.root / "index.json"
        self.head_path = self.root / "HEAD"
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)

    def _read_index(self) -> list[dict[str, Any]]:
        if not self.index_path.exists():
            return []
        return json.loads(self.index_path.read_text(encoding="utf-8"))

    def _write_index(self, index: list[dict[str, Any]]) -> None:
        self.index_path.write_text(
            json.dumps(index, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    def get_head(self) -> str | None:
        if not self.head_path.exists():
            return None
        value = self.head_path.read_text(encoding="utf-8").strip()
        return value or None

    def set_head(self, snapshot_id: str) -> None:
        self.head_path.write_text(snapshot_id + "\n", encoding="utf-8")

    def create(self, snapshot: dict[str, Any], message: str = "") -> SnapshotMeta:
        normalized = normalize_snapshot(snapshot)
        digest = graph_hash(normalized)
        timestamp = _now_utc()
        snapshot_id = f"{timestamp}_{digest[:12]}"
        filename = f"{snapshot_id}.json"
        file_path = self.snapshots_dir / filename

        payload = {
            "version": 1,
            "snapshot_id": snapshot_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "message": message,
            "graph_hash": digest,
            "parent": self.get_head(),
            "nodes": normalized["nodes"],
            "edges": normalized["edges"],
        }

        file_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        meta = SnapshotMeta(
            snapshot_id=snapshot_id,
            created_at=payload["created_at"],
            message=message,
            graph_hash=digest,
            parent=payload["parent"],
            nodes=len(normalized["nodes"]),
            edges=len(normalized["edges"]),
            file=str(file_path),
        )

        index = self._read_index()
        index.append(meta.__dict__)
        self._write_index(index)
        self.set_head(snapshot_id)
        return meta

    def list(self, limit: int = 50) -> list[SnapshotMeta]:
        index = self._read_index()
        rows = index[-limit:] if limit > 0 else index
        rows = list(reversed(rows))
        return [SnapshotMeta(**row) for row in rows]

    def resolve_snapshot_path(self, ref: str) -> Path:
        candidate = Path(ref)
        if candidate.exists():
            return candidate

        direct = self.snapshots_dir / f"{ref}.json"
        if direct.exists():
            return direct

        for row in self._read_index():
            if row["snapshot_id"] == ref:
                return Path(row["file"])

        raise FileNotFoundError(f"Snapshot not found: {ref}")

    def load(self, ref: str) -> dict[str, Any]:
        path = self.resolve_snapshot_path(ref)
        data = json.loads(path.read_text(encoding="utf-8"))
        normalized_payload = {
            "nodes": data.get("nodes", []),
            "edges": data.get("edges", []),
        }
        expected_hash = data.get("graph_hash")
        if expected_hash:
            actual_hash = graph_hash(normalized_payload)
            if actual_hash != expected_hash:
                raise ValueError(
                    f"Snapshot hash mismatch for {ref}: expected {expected_hash}, got {actual_hash}"
                )

        return {
            "nodes": normalized_payload["nodes"],
            "edges": normalized_payload["edges"],
            "meta": {
                "snapshot_id": data.get("snapshot_id"),
                "created_at": data.get("created_at"),
                "message": data.get("message", ""),
                "graph_hash": data.get("graph_hash"),
                "parent": data.get("parent"),
            },
        }


def diff_snapshots(
    old_snapshot: dict[str, Any], new_snapshot: dict[str, Any]
) -> dict[str, Any]:
    old_norm = normalize_snapshot(old_snapshot)
    new_norm = normalize_snapshot(new_snapshot)

    old_nodes = {n["id"]: n for n in old_norm["nodes"]}
    new_nodes = {n["id"]: n for n in new_norm["nodes"]}

    old_node_ids = set(old_nodes)
    new_node_ids = set(new_nodes)

    added_nodes = sorted(new_node_ids - old_node_ids)
    removed_nodes = sorted(old_node_ids - new_node_ids)

    updated_nodes = []
    for node_id in sorted(old_node_ids & new_node_ids):
        old_props = old_nodes[node_id].get("properties", {})
        new_props = new_nodes[node_id].get("properties", {})
        if old_props != new_props or old_nodes[node_id].get("labels") != new_nodes[
            node_id
        ].get("labels"):
            updated_nodes.append(node_id)

    def edge_key(edge: dict[str, Any]) -> tuple[str, str, str, str]:
        props = edge.get("properties", {})
        edge_id = props.get("id")
        if edge_id:
            fingerprint = str(edge_id)
        else:
            fingerprint = hashlib.sha256(
                _canonical_dumps(props).encode("utf-8")
            ).hexdigest()
        return (
            edge["from_id"],
            edge["relation"],
            edge["to_id"],
            fingerprint,
        )

    old_edges = {edge_key(e): e for e in old_norm["edges"]}
    new_edges = {edge_key(e): e for e in new_norm["edges"]}

    old_edge_keys = set(old_edges)
    new_edge_keys = set(new_edges)

    added_edges = sorted(new_edge_keys - old_edge_keys)
    removed_edges = sorted(old_edge_keys - new_edge_keys)

    updated_edges = []
    for key in sorted(old_edge_keys & new_edge_keys):
        if old_edges[key].get("properties", {}) != new_edges[key].get("properties", {}):
            updated_edges.append(key)

    return {
        "nodes": {
            "added": added_nodes,
            "removed": removed_nodes,
            "updated": updated_nodes,
        },
        "edges": {
            "added": [
                {
                    "from_id": k[0],
                    "relation": k[1],
                    "to_id": k[2],
                    "edge_id": k[3] or None,
                }
                for k in added_edges
            ],
            "removed": [
                {
                    "from_id": k[0],
                    "relation": k[1],
                    "to_id": k[2],
                    "edge_id": k[3] or None,
                }
                for k in removed_edges
            ],
            "updated": [
                {
                    "from_id": k[0],
                    "relation": k[1],
                    "to_id": k[2],
                    "edge_id": k[3] or None,
                }
                for k in updated_edges
            ],
        },
        "summary": {
            "added_nodes": len(added_nodes),
            "removed_nodes": len(removed_nodes),
            "updated_nodes": len(updated_nodes),
            "added_edges": len(added_edges),
            "removed_edges": len(removed_edges),
            "updated_edges": len(updated_edges),
        },
    }
