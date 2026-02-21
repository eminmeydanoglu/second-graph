"""Graph Manipulator MCP Server.

Exposes knowledge graph CRUD operations as MCP tools.
Used by the Graph Agent (subagent) for runtime memory manipulation.
"""

from typing import Any

from mcp.server.fastmcp import FastMCP

from ..graph.neo4j_storage import Neo4jStorage
from ..graph.routing_text import build_routing_text
from ..graph.schema import (
    validate_node_type,
    validate_edge,
    generate_node_id,
    get_node_types,
    get_edge_types,
)
from ..graph.sync import NoteSynchronizer
from ..vector.store import VectorStore
from ..vector.embedder import Embedder
from ..extraction.tracker import NoteTracker
from ..retrieval.pipeline import recall_structured

mcp = FastMCP(
    "Graph Manipulator",
    instructions="""
Knowledge graph CRUD operations.

Node Types: Note, Tag, Goal, Project, Belief, Value, Person, Concept, Source, Fear
Edge Types: CONTRIBUTES_TO, MOTIVATES, HAS_GOAL, BELIEVES, INTERESTED_IN, RELATED_TO, etc.

Use add_node/add_edge to create new knowledge.
Use find_node/get_neighbors for exploration.
Use update_node/delete_node for modifications.
""",
)

storage: Neo4jStorage | None = None
embedder: Embedder | None = None
synchronizer: NoteSynchronizer | None = None
tracker: NoteTracker | None = None

ROUTING_FIELDS = {
    "summary",
    "name",
    "tags",
    "domain",
    "status",
    "horizon",
    "role",
    "relationship",
    "priority",
    "author",
    "type",
    "intensity",
}

MENTIONS_EXCLUDED_RELATIONS = {"WIKILINK", "TAGGED_WITH", "MENTIONS"}
MENTIONS_EXCLUDED_NODE_TYPES = {"Note", "Tag"}


def init_server(
    neo4j_uri: str = "bolt://localhost:7687",
    neo4j_user: str = "neo4j",
    neo4j_password: str = "obsidian",
    vector_db: str = "data/vectors.db",
):
    """Initialize server with database connections."""
    global storage, embedder, synchronizer, tracker
    storage = Neo4jStorage(uri=neo4j_uri, user=neo4j_user, password=neo4j_password)
    storage.ensure_vector_index()
    embedder = Embedder()
    synchronizer = NoteSynchronizer(storage, embedder)

    # VectorStore still used for extraction tracking (note hashes, diffs)
    vectors = VectorStore(vector_db)
    tracker = NoteTracker(vectors)


def _require_storage() -> Neo4jStorage:
    """Get storage or raise error."""
    if storage is None:
        raise RuntimeError("Server not initialized. Call init_server() first.")
    return storage


def _require_embedder() -> Embedder:
    """Get embedder or raise error."""
    if embedder is None:
        raise RuntimeError("Server not initialized. Call init_server() first.")
    return embedder


def _strip_internal_fields(data: Any) -> Any:
    """Recursively remove internal vector fields from MCP responses."""
    if isinstance(data, dict):
        cleaned = {}
        for key, value in data.items():
            if key == "embedding":
                continue
            if key in {"_labels", "neighbor_labels"} and isinstance(value, list):
                cleaned[key] = [label for label in value if label != "Entity"]
                continue
            if key == "by_label" and isinstance(value, dict):
                cleaned[key] = {
                    k: _strip_internal_fields(v)
                    for k, v in value.items()
                    if k != "Entity"
                }
                continue
            cleaned[key] = _strip_internal_fields(value)
        return cleaned
    if isinstance(data, list):
        return [_strip_internal_fields(item) for item in data]
    return data


def _strip_internal_dict(data: dict) -> dict:
    """Typed wrapper for dict responses."""
    return _strip_internal_fields(data)


def _canonical_node_type_from_labels(labels: list[str]) -> str:
    """Pick canonical schema type from labels deterministically."""
    valid_types = get_node_types()
    for node_type in valid_types:
        if node_type in labels:
            return node_type
    return "Unknown"


def _relation_key(relation: str) -> str:
    return relation.strip().upper()


def _extract_source_note(properties: dict | None) -> str | None:
    if not properties:
        return None
    source_note = properties.get("source_note")
    if not isinstance(source_note, str):
        return None
    normalized = source_note.strip()
    return normalized if normalized else None


def _maybe_backfill_mentions(
    db: Neo4jStorage,
    *,
    relation: str,
    properties: dict | None,
    source_node: dict,
    target_node: dict,
) -> dict:
    """Create Note->Entity MENTIONS edges from semantic edge provenance."""
    relation_key = _relation_key(relation)
    if relation_key in MENTIONS_EXCLUDED_RELATIONS:
        return {"created": 0, "targets": [], "warnings": []}

    source_note = _extract_source_note(properties)
    if not source_note:
        return {"created": 0, "targets": [], "warnings": []}

    source_note_node = db.get_node(source_note)
    if not source_note_node:
        return {
            "created": 0,
            "targets": [],
            "warnings": [f"source_note_not_found:{source_note}"],
        }

    source_note_type = _canonical_node_type_from_labels(
        source_note_node["node"].get("_labels", [])
    )
    if source_note_type != "Note":
        return {
            "created": 0,
            "targets": [],
            "warnings": [f"source_note_not_note_type:{source_note_type}"],
        }

    endpoint_nodes = [source_node, target_node]
    mention_targets: list[str] = []
    warnings: list[str] = []

    for endpoint in endpoint_nodes:
        endpoint_props = endpoint["node"]
        endpoint_id = endpoint_props.get("id")
        if not isinstance(endpoint_id, str) or not endpoint_id:
            continue
        if endpoint_id == source_note:
            continue

        endpoint_type = _canonical_node_type_from_labels(
            endpoint_props.get("_labels", [])
        )
        if endpoint_type in MENTIONS_EXCLUDED_NODE_TYPES:
            continue

        mention_result = db.add_edge(
            source_note,
            endpoint_id,
            "MENTIONS",
            properties={"source_note": source_note},
        )
        if mention_result.get("success"):
            mention_targets.append(endpoint_id)
        else:
            warnings.append(
                f"mentions_add_failed:{source_note}->{endpoint_id}:{mention_result.get('error', 'unknown')}"
            )

    return {
        "created": len(set(mention_targets)),
        "targets": sorted(set(mention_targets)),
        "warnings": warnings,
    }


@mcp.tool()
def add_node(
    node_type: str,
    name: str,
    summary: str | None = None,
    properties: dict | None = None,
) -> dict:
    """Add a new node to the knowledge graph.

    Args:
        node_type: Node type (Goal, Project, Person, Concept, Belief, Value, Fear, etc.)
        name: Human-readable name for the node
        summary: Optional description (used for semantic search)
        properties: Optional additional properties (status, priority, etc.)

    Returns:
        Dict with success status and created node
    """
    if not validate_node_type(node_type):
        return {
            "success": False,
            "error": f"Invalid node type: {node_type}. Valid types: {get_node_types()}",
        }

    node_id = generate_node_id(node_type, name)
    props = properties.copy() if properties else {}
    if summary:
        props["summary"] = summary

    db = _require_storage()
    result = db.add_node(node_type, node_id, name, props)

    if not result:
        return {"success": False, "error": "Failed to create node"}

    emb = _require_embedder()
    text = build_routing_text(node_type, {"name": name, **props})
    embedding = emb.embed(text)
    db.set_embedding(node_id, embedding)

    return _strip_internal_dict({"success": True, "node_id": node_id, "node": result})


@mcp.tool()
def get_node(node_id: str) -> dict:
    """Get a node by its ID with all connections.

    Args:
        node_id: The node's unique identifier (e.g., "goal:build_ai")

    Returns:
        Dict with node data and its connections
    """
    db = _require_storage()
    result = db.get_node(node_id)
    if not result:
        return {"success": False, "error": f"Node not found: {node_id}"}
    return _strip_internal_dict({"success": True, **result})


@mcp.tool()
def find_node(
    name: str,
    node_type: str | None = None,
    match_type: str = "contains",
) -> dict:
    """Search for nodes by name.

    Args:
        name: Name to search for
        node_type: Optional type filter (Goal, Person, etc.)
        match_type: "exact", "contains", or "starts_with"

    Returns:
        Dict with matching nodes
    """
    db = _require_storage()
    results = db.find_nodes(name, node_type, match_type)
    return _strip_internal_dict(
        {"success": True, "count": len(results), "nodes": results}
    )


@mcp.tool()
def update_node(node_id: str, properties: dict) -> dict:
    """Update a node's properties.

    Args:
        node_id: The node's unique identifier
        properties: Properties to update (merged with existing)

    Returns:
        Dict with updated node
    """
    db = _require_storage()
    result = db.update_node(node_id, properties)
    if not result:
        return {"success": False, "error": f"Node not found: {node_id}"}

    if ROUTING_FIELDS.intersection(properties):
        node = db.get_node(node_id)
        if node:
            node_props = node["node"]
            node_type = _canonical_node_type_from_labels(node_props.get("_labels", []))
            text = build_routing_text(node_type, node_props)
            emb = _require_embedder()
            embedding = emb.embed(text)
            db.set_embedding(node_id, embedding)

    return _strip_internal_dict({"success": True, "node": result})


@mcp.tool()
def delete_node(node_id: str) -> dict:
    """Delete a node and all its connections.

    Args:
        node_id: The node's unique identifier

    Returns:
        Dict with deletion status
    """
    db = _require_storage()
    result = db.delete_node(node_id)

    success = result.get("deleted", False)
    return {"success": success, **result}


@mcp.tool()
def merge_nodes(keep_id: str, merge_id: str) -> dict:
    """Merge two nodes, keeping one and transferring relationships.

    Args:
        keep_id: ID of node to keep
        merge_id: ID of node to merge (will be deleted)

    Returns:
        Dict with merge status
    """
    db = _require_storage()

    result = db.merge_nodes_simple(keep_id, merge_id)

    success = result.get("merged", False)
    return {"success": success, **result}


@mcp.tool()
def add_edge(
    from_id: str,
    to_id: str,
    relation: str,
    properties: dict | None = None,
) -> dict:
    """Create a relationship between two nodes.

    Args:
        from_id: Source node ID
        to_id: Target node ID
        relation: Relationship type (CONTRIBUTES_TO, MOTIVATES, HAS_GOAL, etc.)
        properties: Optional properties (confidence, fact, source, etc.)

    Returns:
        Dict with edge details
    """
    db = _require_storage()

    source_node = db.get_node(from_id)
    target_node = db.get_node(to_id)

    if not source_node:
        return {"success": False, "error": f"Source node not found: {from_id}"}
    if not target_node:
        return {"success": False, "error": f"Target node not found: {to_id}"}

    source_type = _canonical_node_type_from_labels(
        source_node["node"].get("_labels", [])
    )
    target_type = _canonical_node_type_from_labels(
        target_node["node"].get("_labels", [])
    )

    validation = validate_edge(source_type, target_type, relation, strict=True)
    if not validation.valid:
        return {"success": False, "errors": validation.errors}

    result = db.add_edge(from_id, to_id, relation, properties)

    mentions_info = _maybe_backfill_mentions(
        db,
        relation=relation,
        properties=properties,
        source_node=source_node,
        target_node=target_node,
    )

    warnings = []
    if validation.warnings:
        warnings.extend(validation.warnings)
    warnings.extend(mentions_info["warnings"])

    result["mentions_added"] = mentions_info["created"]
    if mentions_info["targets"]:
        result["mention_targets"] = mentions_info["targets"]
    if warnings:
        result["warnings"] = warnings

    return result


@mcp.tool()
def get_edge(edge_id: str) -> dict:
    """Get an edge by its ID.

    Args:
        edge_id: The edge's unique identifier

    Returns:
        Dict with edge details
    """
    db = _require_storage()
    result = db.get_edge(edge_id)
    if not result:
        return {"success": False, "error": f"Edge not found: {edge_id}"}
    return {"success": True, **result}


@mcp.tool()
def delete_edge(edge_id: str) -> dict:
    """Delete an edge.

    Args:
        edge_id: The edge's unique identifier

    Returns:
        Dict with deletion status
    """
    db = _require_storage()
    success = db.delete_edge(edge_id)
    return {"success": success, "error": None if success else "Edge not found"}


@mcp.tool()
def invalidate_edge(edge_id: str, reason: str | None = None) -> dict:
    """Mark an edge as invalid (soft delete with timestamp).

    Use this instead of delete when you want to preserve history.

    Args:
        edge_id: The edge's unique identifier
        reason: Optional reason for invalidation

    Returns:
        Dict with updated edge
    """
    db = _require_storage()
    result = db.invalidate_edge(edge_id, reason)
    if not result:
        return {"success": False, "error": f"Edge not found: {edge_id}"}
    return {"success": True, "edge": result}


@mcp.tool()
def get_neighbors(
    node_id: str,
    direction: str = "both",
    edge_types: list[str] | None = None,
) -> dict:
    """Get neighbors of a node.

    Args:
        node_id: The node's unique identifier
        direction: "in", "out", or "both"
        edge_types: Optional list of relationship types to filter

    Returns:
        Dict with neighbor nodes and relationships
    """
    db = _require_storage()
    results = db.get_neighbors(node_id, direction, edge_types)
    return _strip_internal_dict(
        {"success": True, "count": len(results), "neighbors": results}
    )


@mcp.tool()
def find_path(from_id: str, to_id: str, max_depth: int = 4) -> dict:
    """Find shortest path between two nodes.

    Args:
        from_id: Start node ID
        to_id: End node ID
        max_depth: Maximum path length (default 4)

    Returns:
        Dict with path nodes and relationships
    """
    db = _require_storage()
    result = db.find_path(from_id, to_id, max_depth)
    if not result:
        return {"success": False, "error": "No path found between nodes"}
    return {"success": True, **result}


@mcp.tool()
def search_entities(
    query: str,
    node_types: list[str] | None = None,
    limit: int = 10,
) -> dict:
    """Semantic search for entities.

    Uses vector similarity to find relevant nodes.

    Args:
        query: Natural language search query
        node_types: Optional list of node types to filter
        limit: Maximum results (default 10)

    Returns:
        Dict with matching entities ranked by similarity
    """
    emb = _require_embedder()
    db = _require_storage()

    query_embedding = emb.embed(query)
    results = db.search_similar(query_embedding, node_types, limit)
    return _strip_internal_dict(
        {"success": True, "count": len(results), "entities": results}
    )


@mcp.tool()
def recall(
    query: str,
    depth: int = 1,
    limit: int = 10,
    mode: str = "pull",
    include_debug: bool = False,
) -> dict:
    """Deterministic graph recall with strict markdown contract.

    Args:
        query: Natural language query
        depth: Graph traversal depth (1-2)
        limit: Max matched nodes in the rendered output
        mode: "pull" or "push" token budget profile
        include_debug: Include structured payload alongside markdown

    Returns:
        Dict with markdown and retrieval metadata
    """
    db = _require_storage()
    emb = _require_embedder()

    normalized_mode = "push" if mode == "push" else "pull"
    try:
        result = recall_structured(
            query,
            mode=normalized_mode,
            depth=depth,
            limit=limit,
            storage=db,
            embedder=emb,
        )
    except Exception as exc:
        return {"success": False, "error": f"recall_failed: {exc}"}

    response = {
        "success": True,
        "markdown": result["markdown"],
        "retrieval": result["retrieval"],
    }

    if include_debug:
        response["debug"] = {
            "query": result["query"],
            "mode": result["mode"],
            "depth": result["depth"],
            "limit": result["limit"],
            "warnings": result.get("warnings", []),
            "matched_nodes": result.get("matched_nodes", []),
            "connections": result.get("connections", []),
            "related_notes": result.get("related_notes", []),
        }

    return _strip_internal_dict(response)


@mcp.tool()
def get_schema() -> dict:
    """Get available node and edge types.

    Returns:
        Dict with valid node_types and edge_types
    """
    return {
        "success": True,
        "node_types": get_node_types(),
        "edge_types": get_edge_types(),
    }


@mcp.tool()
def get_stats() -> dict:
    """Get graph statistics.

    Returns:
        Dict with node counts, edge counts, and breakdown by type
    """
    db = _require_storage()
    stats = db.get_stats()
    return _strip_internal_dict({"success": True, **stats})


def _require_synchronizer() -> NoteSynchronizer:
    """Get synchronizer or raise error."""
    if synchronizer is None:
        raise RuntimeError("Server not initialized. Call init_server() first.")
    return synchronizer


@mcp.tool()
def sync_note(path: str) -> dict:
    """Sync a markdown note file into the knowledge graph.

    Parses the note, creates/updates nodes, reconciles wikilink edges,
    syncs tags, and updates vector embeddings.

    Args:
        path: Absolute path to the markdown file

    Returns:
        Dict with sync results (node_id, action, edges changed, source_note)
    """
    sync = _require_synchronizer()
    return _strip_internal_dict(sync.sync_note_from_file(path))


# Alias for backward compatibility
@mcp.tool()
def source_note(path: str) -> dict:
    """Alias for sync_note. Use sync_note instead."""
    return sync_note(path)


def _require_tracker() -> NoteTracker:
    """Get tracker or raise error."""
    if tracker is None:
        raise RuntimeError("Server not initialized. Call init_server() first.")
    return tracker


@mcp.tool()
def check_note_status(path: str) -> dict:
    """Check if a note needs extraction.

    Reads the file, computes SHA256 hash, compares with stored hash
    from last extraction.

    Args:
        path: Absolute path to the markdown file

    Returns:
        - status="new": {status, content} — never extracted before
        - status="changed": {status, diff, content, last_extracted_at} — content changed
        - status="unchanged": {status, last_extracted_at} — skip, no changes
        - status="error": {status, error} — file not found or read error
    """
    t = _require_tracker()
    return t.check_note_status(path)


@mcp.tool()
def mark_extracted(path: str) -> dict:
    """Mark a note as successfully extracted.

    Call this after the graph agent finishes extracting a note.
    Updates the content hash, stores content snapshot for future diffs,
    and records the diff in the audit trail.

    Args:
        path: Absolute path to the markdown file

    Returns:
        Dict with success, content_hash, extracted_at
    """
    t = _require_tracker()
    return t.mark_extracted(path)


@mcp.tool()
def list_pending_notes(vault_path: str) -> dict:
    """List notes in vault that need extraction.

    Scans all markdown files in the vault, computes hashes,
    and compares with stored extraction state.

    Args:
        vault_path: Absolute path to the Obsidian vault root

    Returns:
        Dict with pending list [{path, status}], pending_count, unchanged_count
    """
    t = _require_tracker()
    return t.list_pending_notes(vault_path)


def main():
    """Run the MCP server (stdio transport)."""
    import asyncio
    import os

    init_server(
        neo4j_uri=os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
        neo4j_user=os.environ.get("NEO4J_USER", "neo4j"),
        neo4j_password=os.environ.get("NEO4J_PASSWORD", "obsidian"),
        vector_db=os.environ.get("VECTOR_DB", "data/vectors.db"),
    )

    asyncio.run(mcp.run_stdio_async())


if __name__ == "__main__":
    main()
