"""Neo4j graph storage with CRUD operations.

Provides both initial vault import and runtime CRUD for the knowledge graph.
"""

from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from neo4j import GraphDatabase

from .builder import VaultGraph


class Neo4jStorage:
    """Neo4j graph database wrapper with full CRUD support."""

    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = "obsidian",
    ):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def clear(self):
        """Clear all nodes and relationships."""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")

    # =========================================================================
    # INITIAL IMPORT (from VaultGraph)
    # =========================================================================

    def import_vault_graph(
        self, graph: VaultGraph, batch_size: int = 500
    ) -> dict[str, int]:
        """Import a VaultGraph into Neo4j.

        Returns:
            Dict with counts of imported nodes and relationships
        """
        stats = {"nodes": 0, "relationships": 0}

        with self.driver.session() as session:
            # Import nodes in batches
            nodes = list(graph.graph.nodes(data=True))
            for i in range(0, len(nodes), batch_size):
                batch = nodes[i : i + batch_size]
                self._import_node_batch(session, batch)
                stats["nodes"] += len(batch)

            # Import edges in batches
            edges = list(graph.graph.edges(data=True))
            for i in range(0, len(edges), batch_size):
                batch = edges[i : i + batch_size]
                self._import_edge_batch(session, batch)
                stats["relationships"] += len(batch)

        # Create indexes for performance
        self._create_indexes()

        return stats

    def _sanitize_label(self, label: str) -> str:
        """Sanitize label for Neo4j (no spaces, special chars)."""
        import re

        # Replace spaces and special chars with underscore
        sanitized = re.sub(r"[^a-zA-Z0-9_]", "_", label)
        # Ensure it starts with a letter
        if sanitized and not sanitized[0].isalpha():
            sanitized = "N_" + sanitized
        return sanitized or "Unknown"

    def _import_node_batch(self, session, nodes: list[tuple[str, dict]]):
        """Import a batch of nodes."""
        # Group by type for efficient label creation
        by_type: dict[str, list] = {}
        for node_id, data in nodes:
            node_type = self._sanitize_label(data.get("type", "Unknown"))
            if node_type not in by_type:
                by_type[node_type] = []
            by_type[node_type].append(
                {
                    "id": node_id,
                    "title": data.get("title", data.get("name", node_id)),
                    "name": data.get("name", data.get("title", "")),
                    "folder": data.get("folder", ""),
                    "placeholder": data.get("placeholder", False),
                    "original_type": data.get("type", "Unknown"),
                }
            )

        for node_type, node_list in by_type.items():
            query = f"""
            UNWIND $nodes AS node
            MERGE (n:{node_type} {{id: node.id}})
            SET n.title = node.title,
                n.name = node.name,
                n.folder = node.folder,
                n.placeholder = node.placeholder,
                n.original_type = node.original_type
            """
            session.run(query, nodes=node_list)

    def _import_edge_batch(self, session, edges: list[tuple[str, str, dict]]):
        """Import a batch of edges."""
        # Group by type
        by_type: dict[str, list] = {}
        for source, target, data in edges:
            raw_type = data.get("type", "RELATED_TO")
            edge_type = self._sanitize_label(raw_type).upper()
            if edge_type not in by_type:
                by_type[edge_type] = []
            by_type[edge_type].append(
                {
                    "source": source,
                    "target": target,
                    "confidence": data.get("confidence", 1.0),
                }
            )

        for edge_type, edge_list in by_type.items():
            query = f"""
            UNWIND $edges AS edge
            MATCH (a {{id: edge.source}})
            MATCH (b {{id: edge.target}})
            MERGE (a)-[r:{edge_type}]->(b)
            SET r.confidence = edge.confidence
            """
            session.run(query, edges=edge_list)

    def _create_indexes(self):
        """Create indexes for common queries."""
        with self.driver.session() as session:
            # Index on id for all node types
            for label in [
                "Note",
                "Tag",
                "Folder",
                "Person",
                "Concept",
                "Tool",
                "Project",
                "Organization",
                "Goal",
                "Value",
                "Belief",
                "Fear",
                "Source",
            ]:
                try:
                    session.run(f"CREATE INDEX IF NOT EXISTS FOR (n:{label}) ON (n.id)")
                    session.run(
                        f"CREATE INDEX IF NOT EXISTS FOR (n:{label}) ON (n.title)"
                    )
                    session.run(
                        f"CREATE INDEX IF NOT EXISTS FOR (n:{label}) ON (n.name)"
                    )
                except Exception:
                    pass  # Index might already exist

    # =========================================================================
    # NODE OPERATIONS (CRUD)
    # =========================================================================

    def add_node(
        self, node_type: str, node_id: str, name: str, properties: dict | None = None
    ) -> dict | None:
        """Create a new node.

        Args:
            node_type: The node label (Goal, Project, Person, etc.)
            node_id: Unique identifier (e.g., "goal:build_ai")
            name: Human-readable name
            properties: Additional properties (status, priority, etc.)

        Returns:
            Created node dict or None on failure
        """
        props = properties.copy() if properties else {}
        props["id"] = node_id
        props["name"] = name
        props["created_at"] = datetime.now().isoformat()

        sanitized_type = self._sanitize_label(node_type)

        with self.driver.session() as session:
            result = session.run(
                f"""
                CREATE (n:{sanitized_type} $props)
                RETURN n, labels(n) as labels
                """,
                props=props,
            )
            record = result.single()
            if record:
                node_dict = dict(record["n"])
                node_dict["_labels"] = record["labels"]
                return node_dict
            return None

    def get_node(self, node_id: str) -> dict | None:
        """Get a node by ID with its connections.

        Args:
            node_id: The node's unique identifier

        Returns:
            Dict with 'node' and 'connections' or None if not found
        """
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (n {id: $id})
                OPTIONAL MATCH (n)-[r]-(neighbor)
                RETURN n, labels(n) as labels,
                       collect(DISTINCT {
                           neighbor_id: neighbor.id,
                           neighbor_name: neighbor.name,
                           neighbor_labels: labels(neighbor),
                           relation: type(r),
                           direction: CASE WHEN startNode(r) = n THEN 'out' ELSE 'in' END
                       }) as connections
                """,
                id=node_id,
            )
            record = result.single()
            if not record or record["n"] is None:
                return None

            node_dict = dict(record["n"])
            node_dict["_labels"] = record["labels"]

            # Filter out null connections
            connections = [
                c for c in record["connections"] if c["neighbor_id"] is not None
            ]

            return {"node": node_dict, "connections": connections}

    def find_nodes(
        self, name: str, node_type: str | None = None, match_type: str = "contains"
    ) -> list[dict]:
        """Search for nodes by name.

        Args:
            name: Name to search for
            node_type: Optional type filter (Goal, Person, etc.)
            match_type: "exact", "contains", or "starts_with"

        Returns:
            List of matching node dicts
        """
        type_filter = f":{self._sanitize_label(node_type)}" if node_type else ""

        if match_type == "exact":
            where = "WHERE n.name = $name"
        elif match_type == "starts_with":
            where = "WHERE n.name STARTS WITH $name"
        else:  # contains
            where = "WHERE toLower(n.name) CONTAINS toLower($name)"

        with self.driver.session() as session:
            result = session.run(
                f"""
                MATCH (n{type_filter})
                {where}
                RETURN n, labels(n) as labels
                LIMIT 20
                """,
                name=name,
            )
            nodes = []
            for r in result:
                node_dict = dict(r["n"])
                node_dict["_labels"] = r["labels"]
                nodes.append(node_dict)
            return nodes

    def update_node(self, node_id: str, properties: dict) -> dict | None:
        """Update node properties.

        Args:
            node_id: The node's unique identifier
            properties: Properties to update (merged with existing)

        Returns:
            Updated node dict or None if not found
        """
        props = properties.copy()
        props["updated_at"] = datetime.now().isoformat()

        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (n {id: $id})
                SET n += $props
                RETURN n, labels(n) as labels
                """,
                id=node_id,
                props=props,
            )
            record = result.single()
            if record and record["n"]:
                node_dict = dict(record["n"])
                node_dict["_labels"] = record["labels"]
                return node_dict
            return None

    def delete_node(self, node_id: str) -> dict:
        """Delete a node and all its relationships.

        Args:
            node_id: The node's unique identifier

        Returns:
            Dict with deletion status and edge count
        """
        with self.driver.session() as session:
            # Get edge count first
            count_result = session.run(
                """
                MATCH (n {id: $id})-[r]-()
                RETURN count(r) as edge_count
                """,
                id=node_id,
            )
            count_record = count_result.single()
            edge_count = count_record["edge_count"] if count_record else 0

            # Delete node and relationships
            delete_result = session.run(
                """
                MATCH (n {id: $id})
                DETACH DELETE n
                RETURN count(*) as deleted
                """,
                id=node_id,
            )
            deleted = delete_result.single()["deleted"] > 0

            return {"deleted": deleted, "edges_removed": edge_count}

    def merge_nodes(self, keep_id: str, merge_id: str) -> dict:
        """Merge two nodes, transferring relationships from merge_id to keep_id.

        Args:
            keep_id: ID of node to keep
            merge_id: ID of node to merge (will be deleted)

        Returns:
            Dict with merge status and transferred edge count
        """
        with self.driver.session() as session:
            # Transfer incoming edges
            session.run(
                """
                MATCH (keep {id: $keep_id})
                MATCH (merge {id: $merge_id})
                MATCH (source)-[r]->(merge)
                WHERE source <> keep AND source <> merge
                WITH keep, source, type(r) as rel_type, properties(r) as rel_props
                CALL apoc.create.relationship(source, rel_type, rel_props, keep) YIELD rel
                RETURN count(rel)
                """,
                keep_id=keep_id,
                merge_id=merge_id,
            )

            # Transfer outgoing edges
            session.run(
                """
                MATCH (keep {id: $keep_id})
                MATCH (merge {id: $merge_id})
                MATCH (merge)-[r]->(target)
                WHERE target <> keep AND target <> merge
                WITH keep, target, type(r) as rel_type, properties(r) as rel_props
                CALL apoc.create.relationship(keep, rel_type, rel_props, target) YIELD rel
                RETURN count(rel)
                """,
                keep_id=keep_id,
                merge_id=merge_id,
            )

            # Count and delete merge node
            count_result = session.run(
                """
                MATCH (n {id: $id})-[r]-()
                RETURN count(r) as transferred
                """,
                id=merge_id,
            )
            transferred = count_result.single()["transferred"]

            session.run(
                """
                MATCH (n {id: $id})
                DETACH DELETE n
                """,
                id=merge_id,
            )

            return {"merged": True, "edges_transferred": transferred}

    def merge_nodes_simple(self, keep_id: str, merge_id: str) -> dict:
        """Merge two nodes without APOC (simpler, creates new relationship types).

        For use when APOC is not available.
        """
        with self.driver.session() as session:
            # Check if both nodes exist
            check_result = session.run(
                """
                OPTIONAL MATCH (keep {id: $keep_id})
                OPTIONAL MATCH (merge {id: $merge_id})
                RETURN keep IS NOT NULL as keep_exists, merge IS NOT NULL as merge_exists
                """,
                keep_id=keep_id,
                merge_id=merge_id,
            )
            check = check_result.single()
            if not check["keep_exists"]:
                return {
                    "merged": False,
                    "error": f"Keep node not found: {keep_id}",
                    "edges_transferred": 0,
                }
            if not check["merge_exists"]:
                return {
                    "merged": False,
                    "error": f"Merge node not found: {merge_id}",
                    "edges_transferred": 0,
                }

            # Get edges to transfer
            result = session.run(
                """
                MATCH (merge {id: $merge_id})
                OPTIONAL MATCH (source)-[r_in]->(merge)
                WHERE source.id <> $keep_id
                OPTIONAL MATCH (merge)-[r_out]->(target)
                WHERE target.id <> $keep_id
                RETURN 
                    collect(DISTINCT {source: source.id, type: type(r_in)}) as incoming,
                    collect(DISTINCT {target: target.id, type: type(r_out)}) as outgoing
                """,
                merge_id=merge_id,
                keep_id=keep_id,
            )
            record = result.single()
            incoming = [e for e in record["incoming"] if e["source"]]
            outgoing = [e for e in record["outgoing"] if e["target"]]

            # Create new edges to keep node
            for edge in incoming:
                session.run(
                    f"""
                    MATCH (source {{id: $source_id}})
                    MATCH (keep {{id: $keep_id}})
                    MERGE (source)-[:{edge["type"]}]->(keep)
                    """,
                    source_id=edge["source"],
                    keep_id=keep_id,
                )

            for edge in outgoing:
                session.run(
                    f"""
                    MATCH (keep {{id: $keep_id}})
                    MATCH (target {{id: $target_id}})
                    MERGE (keep)-[:{edge["type"]}]->(target)
                    """,
                    keep_id=keep_id,
                    target_id=edge["target"],
                )

            # Delete merge node
            session.run(
                """
                MATCH (n {id: $id})
                DETACH DELETE n
                """,
                id=merge_id,
            )

            return {"merged": True, "edges_transferred": len(incoming) + len(outgoing)}

    # =========================================================================
    # EDGE OPERATIONS (CRUD)
    # =========================================================================

    def add_edge(
        self,
        from_id: str,
        to_id: str,
        relation: str,
        properties: dict | None = None,
    ) -> dict:
        """Create an edge between two nodes.

        Args:
            from_id: Source node ID
            to_id: Target node ID
            relation: Relationship type (CONTRIBUTES_TO, MOTIVATES, etc.)
            properties: Additional edge properties (confidence, fact, etc.)

        Returns:
            Dict with success status and edge details
        """
        props = properties.copy() if properties else {}
        edge_id = f"edge:{uuid4().hex[:12]}"
        props["id"] = edge_id
        props["created_at"] = datetime.now().isoformat()

        sanitized_rel = self._sanitize_label(relation).upper()

        with self.driver.session() as session:
            result = session.run(
                f"""
                MATCH (a {{id: $from_id}})
                MATCH (b {{id: $to_id}})
                CREATE (a)-[r:{sanitized_rel} $props]->(b)
                RETURN r, a.name as from_name, b.name as to_name
                """,
                from_id=from_id,
                to_id=to_id,
                props=props,
            )
            record = result.single()
            if not record:
                return {"success": False, "error": "One or both nodes not found"}
            return {
                "success": True,
                "edge_id": edge_id,
                "from_id": from_id,
                "to_id": to_id,
                "from_name": record["from_name"],
                "to_name": record["to_name"],
                "relation": relation,
            }

    def get_edge(self, edge_id: str) -> dict | None:
        """Get an edge by its ID.

        Args:
            edge_id: The edge's unique identifier

        Returns:
            Dict with edge details or None if not found
        """
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (a)-[r {id: $id}]->(b)
                RETURN r, a.id as from_id, a.name as from_name,
                       b.id as to_id, b.name as to_name, type(r) as relation
                """,
                id=edge_id,
            )
            record = result.single()
            if not record:
                return None
            return {
                "edge": dict(record["r"]),
                "from_id": record["from_id"],
                "from_name": record["from_name"],
                "to_id": record["to_id"],
                "to_name": record["to_name"],
                "relation": record["relation"],
            }

    def find_edges(
        self,
        from_id: str | None = None,
        to_id: str | None = None,
        relation: str | None = None,
    ) -> list[dict]:
        """Find edges by criteria.

        Args:
            from_id: Filter by source node ID
            to_id: Filter by target node ID
            relation: Filter by relationship type

        Returns:
            List of matching edge dicts
        """
        conditions = []
        params = {}

        if from_id:
            conditions.append("a.id = $from_id")
            params["from_id"] = from_id
        if to_id:
            conditions.append("b.id = $to_id")
            params["to_id"] = to_id

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        rel_pattern = f":{self._sanitize_label(relation).upper()}" if relation else ""

        with self.driver.session() as session:
            result = session.run(
                f"""
                MATCH (a)-[r{rel_pattern}]->(b)
                {where_clause}
                RETURN r, a.id as from_id, a.name as from_name,
                       b.id as to_id, b.name as to_name, type(r) as relation
                LIMIT 50
                """,
                **params,
            )
            return [
                {
                    "edge": dict(r["r"]),
                    "from_id": r["from_id"],
                    "from_name": r["from_name"],
                    "to_id": r["to_id"],
                    "to_name": r["to_name"],
                    "relation": r["relation"],
                }
                for r in result
            ]

    def delete_edge(self, edge_id: str) -> bool:
        """Delete an edge by its ID.

        Args:
            edge_id: The edge's unique identifier

        Returns:
            True if deleted, False if not found
        """
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH ()-[r {id: $id}]->()
                DELETE r
                RETURN count(r) as deleted
                """,
                id=edge_id,
            )
            return result.single()["deleted"] > 0

    def invalidate_edge(self, edge_id: str, reason: str | None = None) -> dict | None:
        """Mark an edge as invalid (soft delete with timestamp).

        Args:
            edge_id: The edge's unique identifier
            reason: Optional reason for invalidation

        Returns:
            Updated edge dict or None if not found
        """
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH ()-[r {id: $id}]->()
                SET r.invalid_at = datetime(), r.invalidation_reason = $reason
                RETURN r
                """,
                id=edge_id,
                reason=reason,
            )
            record = result.single()
            return dict(record["r"]) if record else None

    # =========================================================================
    # QUERY OPERATIONS
    # =========================================================================

    def get_neighbors(
        self,
        node_id: str,
        direction: str = "both",
        edge_types: list[str] | None = None,
        depth: int = 1,
    ) -> list[dict]:
        """Get neighbors of a node.

        Args:
            node_id: The node's unique identifier
            direction: "in", "out", or "both"
            edge_types: Optional list of relationship types to filter
            depth: Currently only supports depth=1

        Returns:
            List of neighbor dicts with node and relationship info
        """
        type_filter = ""
        if edge_types:
            sanitized = [self._sanitize_label(t).upper() for t in edge_types]
            type_filter = ":" + "|".join(sanitized)

        if direction == "out":
            pattern = f"(n)-[r{type_filter}]->(neighbor)"
        elif direction == "in":
            pattern = f"(n)<-[r{type_filter}]-(neighbor)"
        else:
            pattern = f"(n)-[r{type_filter}]-(neighbor)"

        with self.driver.session() as session:
            result = session.run(
                f"""
                MATCH (n {{id: $id}})
                MATCH {pattern}
                RETURN neighbor, labels(neighbor) as labels, type(r) as relation,
                       CASE WHEN startNode(r) = n THEN 'out' ELSE 'in' END as direction
                """,
                id=node_id,
            )
            neighbors = []
            for r in result:
                node_dict = dict(r["neighbor"])
                node_dict["_labels"] = r["labels"]
                neighbors.append(
                    {
                        "node": node_dict,
                        "relation": r["relation"],
                        "direction": r["direction"],
                    }
                )
            return neighbors

    def find_path(self, from_id: str, to_id: str, max_depth: int = 4) -> dict | None:
        """Find shortest path between two nodes.

        Args:
            from_id: Start node ID
            to_id: End node ID
            max_depth: Maximum path length

        Returns:
            Dict with path info or None if no path exists
        """
        with self.driver.session() as session:
            result = session.run(
                f"""
                MATCH (start {{id: $from_id}}), (end {{id: $to_id}})
                MATCH path = shortestPath((start)-[*..{max_depth}]-(end))
                RETURN [n IN nodes(path) | n.id] as node_ids,
                       [n IN nodes(path) | n.name] as node_names,
                       [r IN relationships(path) | type(r)] as relations,
                       length(path) as path_length
                """,
                from_id=from_id,
                to_id=to_id,
            )
            record = result.single()
            if not record:
                return None
            return {
                "node_ids": record["node_ids"],
                "node_names": record["node_names"],
                "relations": record["relations"],
                "length": record["path_length"],
            }

    def get_subgraph(self, node_id: str, depth: int = 2, max_nodes: int = 50) -> dict:
        """Get a subgraph centered on a node.

        Args:
            node_id: Center node ID
            depth: How many hops to traverse
            max_nodes: Maximum nodes to return

        Returns:
            Dict with nodes and edges lists
        """
        with self.driver.session() as session:
            result = session.run(
                f"""
                MATCH (center {{id: $id}})
                CALL apoc.path.subgraphAll(center, {{
                    maxLevel: {depth},
                    limit: {max_nodes}
                }})
                YIELD nodes, relationships
                RETURN nodes, relationships
                """,
                id=node_id,
            )
            record = result.single()
            if not record:
                # Fallback without APOC
                return self._get_subgraph_simple(node_id, depth, max_nodes)

            nodes = [
                {"id": n["id"], "name": n.get("name"), "labels": list(n.labels)}
                for n in record["nodes"]
            ]
            edges = [
                {
                    "from": r.start_node["id"],
                    "to": r.end_node["id"],
                    "type": r.type,
                }
                for r in record["relationships"]
            ]
            return {"nodes": nodes, "edges": edges}

    def _get_subgraph_simple(
        self, node_id: str, depth: int = 2, max_nodes: int = 50
    ) -> dict:
        """Get subgraph without APOC."""
        with self.driver.session() as session:
            # First get all connected nodes
            nodes_result = session.run(
                f"""
                MATCH path = (center {{id: $id}})-[*0..{depth}]-(connected)
                WITH DISTINCT connected
                LIMIT {max_nodes}
                RETURN collect({{
                    id: connected.id,
                    name: connected.name,
                    labels: labels(connected)
                }}) as nodes
                """,
                id=node_id,
            )
            nodes_record = nodes_result.single()
            if not nodes_record or not nodes_record["nodes"]:
                return {"nodes": [], "edges": []}

            nodes = nodes_record["nodes"]
            node_ids = [n["id"] for n in nodes if n["id"]]

            if not node_ids:
                return {"nodes": [], "edges": []}

            # Then get edges between those nodes
            edges_result = session.run(
                """
                MATCH (a)-[r]-(b)
                WHERE a.id IN $node_ids AND b.id IN $node_ids
                RETURN collect(DISTINCT {
                    from_id: startNode(r).id,
                    to_id: endNode(r).id,
                    type: type(r)
                }) as edges
                """,
                node_ids=node_ids,
            )
            edges_record = edges_result.single()
            edges = edges_record["edges"] if edges_record else []

            return {
                "nodes": nodes,
                "edges": [e for e in edges if e["from_id"]],
            }

    # =========================================================================
    # STATISTICS
    # =========================================================================

    def get_stats(self) -> dict[str, Any]:
        """Get database statistics."""
        with self.driver.session() as session:
            node_count = session.run("MATCH (n) RETURN count(n) as count").single()[
                "count"
            ]
            rel_count = session.run(
                "MATCH ()-[r]->() RETURN count(r) as count"
            ).single()["count"]

            # Count by label
            labels = session.run("CALL db.labels() YIELD label RETURN label").data()
            label_counts = {}
            for row in labels:
                label = row["label"]
                count = session.run(
                    f"MATCH (n:{label}) RETURN count(n) as count"
                ).single()["count"]
                label_counts[label] = count

            return {
                "nodes": node_count,
                "relationships": rel_count,
                "by_label": label_counts,
            }
