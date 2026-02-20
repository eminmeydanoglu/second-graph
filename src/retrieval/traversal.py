"""Graph traversal and deterministic neighborhood expansion."""

from collections import deque
from typing import Protocol

from .config import DEFAULT_RETRIEVAL_CONFIG, RetrievalConfig
from .ranker import canonical_node_type_from_labels
from .tokenize import normalize_whitespace
from .types import RankedConnection, RelatedNote, ScoredNode, TraversalResult

_REASON_PRIORITY = {"anchor": 0, "neighbor": 1, "path": 2}


class NeighborStorage(Protocol):
    def get_neighbors(
        self,
        node_id: str,
        direction: str = "both",
        edge_types: list[str] | None = None,
        depth: int = 1,
    ) -> list[dict]: ...


def _node_name(node: dict | None, default: str) -> str:
    if not isinstance(node, dict):
        return default
    for key in ("name", "title", "id"):
        value = node.get(key)
        if isinstance(value, str) and value.strip():
            return normalize_whitespace(value)
    return default


def _node_type(node: dict | None) -> str:
    if not isinstance(node, dict):
        return "Unknown"
    node_type = node.get("node_type")
    if isinstance(node_type, str) and node_type.strip():
        return normalize_whitespace(node_type)
    return canonical_node_type_from_labels(node.get("_labels"))


def _is_better_connection(new: RankedConnection, old: RankedConnection) -> bool:
    if new.support_hop != old.support_hop:
        return new.support_hop < old.support_hop
    return new.rank_score > old.rank_score


def _merge_note(existing: RelatedNote | None, candidate: RelatedNote) -> RelatedNote:
    if existing is None:
        return candidate
    if candidate.score > existing.score:
        return candidate
    if candidate.score < existing.score:
        return existing

    candidate_rank = _REASON_PRIORITY.get(candidate.reason, 99)
    existing_rank = _REASON_PRIORITY.get(existing.reason, 99)
    if candidate_rank < existing_rank:
        return candidate
    if candidate_rank > existing_rank:
        return existing
    if candidate.note_id < existing.note_id:
        return candidate
    return existing


def _connection_sort_key(
    connection: RankedConnection,
) -> tuple[float, int, str, str, str, str]:
    return (
        -connection.rank_score,
        connection.support_hop,
        connection.from_id,
        connection.relation,
        connection.to_id,
        connection.direction,
    )


def _note_sort_key(note: RelatedNote) -> tuple[float, str, str]:
    return (-note.score, note.note_id, note.title)


def expand_from_anchors_with_hops(
    anchors: list[ScoredNode],
    *,
    storage: NeighborStorage,
    depth: int,
    neighbor_cap: int,
    config: RetrievalConfig = DEFAULT_RETRIEVAL_CONFIG,
) -> TraversalResult:
    """Expand anchors with bounded BFS and deterministic dedup/sorting."""
    bounded_depth = config.clamp_depth(depth)
    bounded_neighbor_cap = max(1, neighbor_cap)

    hop_by_node: dict[str, int] = {}
    node_lookup: dict[str, dict] = {}
    connection_map: dict[tuple[str, str, str, str], RankedConnection] = {}
    note_map: dict[str, RelatedNote] = {}

    for anchor in anchors:
        hop_by_node[anchor.node_id] = 0
        node_lookup[anchor.node_id] = {
            "name": anchor.name,
            "node_type": anchor.node_type,
            "summary": anchor.summary,
        }

        if anchor.node_type == "Note":
            note_map[anchor.node_id] = _merge_note(
                note_map.get(anchor.node_id),
                RelatedNote(
                    note_id=anchor.node_id,
                    title=anchor.name,
                    reason="anchor",
                    score=anchor.final_score,
                ),
            )

        seen_hops: dict[str, int] = {anchor.node_id: 0}
        queue: deque[tuple[str, int]] = deque([(anchor.node_id, 0)])

        while queue:
            current_id, current_hop = queue.popleft()
            if current_hop >= bounded_depth:
                continue

            try:
                raw_neighbors = storage.get_neighbors(current_id, direction="both")
            except Exception:
                continue

            cleaned_neighbors: list[dict] = []
            for raw in raw_neighbors:
                node = raw.get("node")
                if not isinstance(node, dict):
                    continue

                neighbor_id = str(node.get("id") or "").strip()
                relation = str(raw.get("relation") or "").strip()
                if not neighbor_id or not relation:
                    continue

                direction = str(raw.get("direction") or "out").strip().lower()
                if direction not in {"in", "out"}:
                    direction = "out"

                cleaned_neighbors.append(
                    {
                        "node": node,
                        "neighbor_id": neighbor_id,
                        "relation": relation.upper(),
                        "direction": direction,
                    }
                )

            cleaned_neighbors.sort(
                key=lambda item: (
                    item["neighbor_id"],
                    item["relation"],
                    item["direction"],
                )
            )

            for item in cleaned_neighbors[:bounded_neighbor_cap]:
                node = item["node"]
                neighbor_id = item["neighbor_id"]
                relation = item["relation"]
                direction = item["direction"]

                next_hop = current_hop + 1

                previous_hop = seen_hops.get(neighbor_id)
                if previous_hop is None or next_hop < previous_hop:
                    seen_hops[neighbor_id] = next_hop
                    if next_hop < bounded_depth:
                        queue.append((neighbor_id, next_hop))

                global_hop = hop_by_node.get(neighbor_id)
                if global_hop is None or next_hop < global_hop:
                    hop_by_node[neighbor_id] = next_hop

                current_meta = node_lookup.get(current_id, {})
                from_name = _node_name(current_meta, current_id)
                to_name = _node_name(node, neighbor_id)
                to_type = _node_type(node)

                node_lookup.setdefault(
                    neighbor_id,
                    {
                        "name": to_name,
                        "node_type": to_type,
                        "summary": node.get("summary"),
                    },
                )

                rank_score = anchor.final_score * (0.85**next_hop)
                connection = RankedConnection(
                    from_id=current_id,
                    from_name=from_name,
                    relation=relation,
                    to_id=neighbor_id,
                    to_name=to_name,
                    direction=direction,
                    support_hop=next_hop,
                    rank_score=rank_score,
                )

                key = (
                    connection.from_id,
                    connection.relation,
                    connection.to_id,
                    connection.direction,
                )
                existing = connection_map.get(key)
                if existing is None or _is_better_connection(connection, existing):
                    connection_map[key] = connection

                if to_type == "Note":
                    note_map[neighbor_id] = _merge_note(
                        note_map.get(neighbor_id),
                        RelatedNote(
                            note_id=neighbor_id,
                            title=to_name,
                            reason="neighbor",
                            score=rank_score,
                        ),
                    )

    connections = sorted(connection_map.values(), key=_connection_sort_key)
    notes = sorted(note_map.values(), key=_note_sort_key)

    return TraversalResult(
        connections=tuple(connections),
        notes=tuple(notes),
        hop_by_node=hop_by_node,
        node_lookup=node_lookup,
    )


def expand_from_anchors(
    anchors: list[ScoredNode],
    *,
    storage: NeighborStorage,
    depth: int,
    neighbor_cap: int,
    config: RetrievalConfig = DEFAULT_RETRIEVAL_CONFIG,
) -> tuple[list[RankedConnection], list[RelatedNote]]:
    """Compatibility wrapper returning only edges and related notes."""
    result = expand_from_anchors_with_hops(
        anchors,
        storage=storage,
        depth=depth,
        neighbor_cap=neighbor_cap,
        config=config,
    )
    return list(result.connections), list(result.notes)
