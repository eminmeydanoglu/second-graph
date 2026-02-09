"""Graph Manipulator MCP Server.

Exposes knowledge graph CRUD operations as MCP tools.
Used by the Graph Agent (subagent) for runtime memory manipulation.
"""

from mcp.server.fastmcp import FastMCP

from ..graph.neo4j_storage import Neo4jStorage
from ..graph.schema import (
    NodeType,
    EdgeType,
    validate_node_type,
    validate_edge,
    generate_node_id,
    get_node_types,
    get_edge_types,
)
from ..vector.store import VectorStore
from ..vector.embedder import Embedder

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

# Global instances - initialized by init_server()
storage: Neo4jStorage | None = None
vectors: VectorStore | None = None
embedder: Embedder | None = None


def init_server(
    neo4j_uri: str = "bolt://localhost:7687",
    neo4j_user: str = "neo4j",
    neo4j_password: str = "obsidian",
    vector_db: str = "data/vectors.db",
):
    """Initialize server with database connections."""
    global storage, vectors, embedder
    storage = Neo4jStorage(uri=neo4j_uri, user=neo4j_user, password=neo4j_password)
    vectors = VectorStore(vector_db)
    embedder = Embedder()


def _require_storage() -> Neo4jStorage:
    """Get storage or raise error."""
    if storage is None:
        raise RuntimeError("Server not initialized. Call init_server() first.")
    return storage


def _require_vectors() -> VectorStore:
    """Get vector store or raise error."""
    if vectors is None:
        raise RuntimeError("Server not initialized. Call init_server() first.")
    return vectors


def _require_embedder() -> Embedder:
    """Get embedder or raise error."""
    if embedder is None:
        raise RuntimeError("Server not initialized. Call init_server() first.")
    return embedder


# =============================================================================
# NODE TOOLS
# =============================================================================


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

    # Add to Neo4j
    db = _require_storage()
    result = db.add_node(node_type, node_id, name, props)

    if not result:
        return {"success": False, "error": "Failed to create node"}

    # Add embedding if summary provided
    if summary:
        emb = _require_embedder()
        vecs = _require_vectors()
        embedding = emb.embed(f"{name}: {summary}")
        vecs.add_entity(node_id, node_type, name, summary, embedding)

    return {"success": True, "node_id": node_id, "node": result}


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
    return {"success": True, **result}


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
    return {"success": True, "count": len(results), "nodes": results}


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

    # Update embedding if summary changed
    if "summary" in properties:
        node = db.get_node(node_id)
        if node:
            name = node["node"].get("name", "")
            node_type = node["node"].get("_labels", ["Unknown"])[0]
            summary = properties["summary"]
            emb = _require_embedder()
            vecs = _require_vectors()
            embedding = emb.embed(f"{name}: {summary}")
            vecs.add_entity(node_id, node_type, name, summary, embedding)

    return {"success": True, "node": result}


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

    # Also remove from vector store
    vecs = _require_vectors()
    vecs.delete_entity(node_id)

    # Propagate failure if node wasn't found/deleted
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

    # Use simple merge (doesn't require APOC)
    result = db.merge_nodes_simple(keep_id, merge_id)

    # Only remove from vector store if merge succeeded
    if result.get("merged", False):
        vecs = _require_vectors()
        vecs.delete_entity(merge_id)

    # Propagate success/failure from storage result
    success = result.get("merged", False)
    return {"success": success, **result}


# =============================================================================
# EDGE TOOLS
# =============================================================================


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
    # Validate edge (get source and target types for validation)
    db = _require_storage()

    source_node = db.get_node(from_id)
    target_node = db.get_node(to_id)

    if not source_node:
        return {"success": False, "error": f"Source node not found: {from_id}"}
    if not target_node:
        return {"success": False, "error": f"Target node not found: {to_id}"}

    source_type = source_node["node"].get("_labels", ["Unknown"])[0]
    target_type = target_node["node"].get("_labels", ["Unknown"])[0]

    validation = validate_edge(source_type, target_type, relation)
    if not validation.valid:
        return {"success": False, "errors": validation.errors}
    if validation.warnings:
        # Log warnings but continue
        pass

    result = db.add_edge(from_id, to_id, relation, properties)
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


# =============================================================================
# QUERY TOOLS
# =============================================================================


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
    return {"success": True, "count": len(results), "neighbors": results}


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
    vecs = _require_vectors()

    query_embedding = emb.embed(query)
    results = vecs.search_entities(query_embedding, node_types, limit)
    return {"success": True, "count": len(results), "entities": results}


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
    return {"success": True, **stats}


# =============================================================================
# ENTRY POINT
# =============================================================================


def main():
    """Run the MCP server (stdio transport)."""
    import asyncio

    asyncio.run(mcp.run_stdio_async())


if __name__ == "__main__":
    main()
