"""Neo4j graph storage with CRUD operations and native vector search.

Provides both initial vault import and runtime CRUD for the knowledge graph.
Embeddings are stored as node properties and indexed via Neo4j vector index.
"""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any
from uuid import uuid4

from neo4j import GraphDatabase

# Internal labels/properties hidden from MCP responses
INTERNAL_LABELS = frozenset({"Entity"})
INTERNAL_PROPERTIES = frozenset({"embedding"})
DEFAULT_EMBEDDING_DIMENSIONS = 384


class Neo4jStorage:
    """Neo4j graph database wrapper with full CRUD and vector search support."""

    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = "obsidian",
        embedding_dimensions: int = DEFAULT_EMBEDDING_DIMENSIONS,
    ):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.embedding_dimensions = embedding_dimensions

    @staticmethod
    def _clean_node(node_dict: dict) -> dict:
        """Strip internal properties from node dict before returning to MCP."""
        cleaned = {k: v for k, v in node_dict.items() if k not in INTERNAL_PROPERTIES}
        if "_labels" in cleaned:
            cleaned["_labels"] = [
                l for l in cleaned["_labels"] if l not in INTERNAL_LABELS
            ]
        return cleaned

    @staticmethod
    def _clean_connections(connections: list[dict]) -> list[dict]:
        """Strip internal labels from connection neighbor_labels."""
        for c in connections:
            if c.get("neighbor_labels"):
                c["neighbor_labels"] = [
                    l for l in c["neighbor_labels"] if l not in INTERNAL_LABELS
                ]
        return connections

    def close(self):
        self.driver.close()

    def clear(self, force: bool = False):
        """Clear all nodes and relationships.

        Args:
            force: Must be True to execute destructive wipe.
        """
        if not force:
            raise RuntimeError("Refusing to clear database without force=True")
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")

    def _sanitize_label(self, label: str) -> str:
        """Sanitize label for Neo4j (no spaces, special chars)."""
        import re

        sanitized = re.sub(r"[^a-zA-Z0-9_]", "_", label)
        if sanitized and not sanitized[0].isalpha():
            sanitized = "N_" + sanitized
        return sanitized or "Unknown"

    def _create_indexes(self):
        """Create indexes and uniqueness constraints."""
        with self.driver.session() as session:
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
                    session.run(
                        f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:{label}) REQUIRE n.id IS UNIQUE"
                    )
                    session.run(
                        f"CREATE INDEX IF NOT EXISTS FOR (n:{label}) ON (n.name)"
                    )
                except Exception:
                    pass

            # Vector index for semantic search
            # Drop legacy index name if present
            try:
                session.run("DROP INDEX entity_embedding_idx IF EXISTS")
            except Exception:
                pass
            try:
                session.run(f"""
                    CREATE VECTOR INDEX entity_embeddings IF NOT EXISTS
                    FOR (n:Entity)
                    ON (n.embedding)
                    OPTIONS {{indexConfig: {{
                        `vector.dimensions`: {self.embedding_dimensions},
                        `vector.similarity_function`: 'cosine'
                    }}}}
                """)
            except Exception:
                pass

    def add_node(
        self, node_type: str, node_id: str, name: str, properties: dict | None = None
    ) -> dict | None:
        """Create or update a node (idempotent).

        Uses MERGE on id so duplicate calls are safe.
        On first create, sets created_at. On match, preserves it.

        Args:
            node_type: The node label (Goal, Project, Person, etc.)
            node_id: Unique identifier (e.g., "goal:build_ai")
            name: Human-readable name
            properties: Additional properties (status, priority, etc.)

        Returns:
            Node dict or None on failure
        """
        props = properties.copy() if properties else {}
        props["name"] = name.strip()

        sanitized_type = self._sanitize_label(node_type)
        now = datetime.now().isoformat()

        with self.driver.session() as session:
            result = session.run(
                f"""
                MERGE (n:{sanitized_type} {{id: $node_id}})
                ON CREATE SET n += $props, n.created_at = $now
                ON MATCH SET n += $props
                RETURN n, labels(n) as labels
                """,
                node_id=node_id,
                props=props,
                now=now,
            )
            record = result.single()
            if record:
                node_dict = dict(record["n"])
                node_dict["_labels"] = record["labels"]
                return self._clean_node(node_dict)
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
                           edge_id: r.id,
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

            connections = [
                c for c in record["connections"] if c["neighbor_id"] is not None
            ]
            self._clean_connections(connections)

            return {"node": self._clean_node(node_dict), "connections": connections}

    def list_nodes(self) -> list[dict]:
        """List all nodes with labels for bulk operations."""
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (n)
                RETURN n, labels(n) as labels
                ORDER BY n.id
                """
            )

            nodes: list[dict] = []
            for record in result:
                node_dict = dict(record["n"])
                node_dict["_labels"] = record["labels"]
                nodes.append(self._clean_node(node_dict))
            return nodes

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
                nodes.append(self._clean_node(node_dict))
            return nodes

    def find_note_by_vault_rel_path(self, rel_path: str) -> dict | None:
        """Find a Note node by vault-relative path.

        Args:
            rel_path: Vault-relative path, with or without .md suffix

        Returns:
            Matching note node dict or None
        """
        normalized = rel_path.replace("\\", "/").strip().lstrip("./")
        if normalized.lower().endswith(".md"):
            normalized = normalized[:-3]

        path_no_ext = normalized.lower()
        path_with_ext = f"{path_no_ext}.md"
        path_with_ext_suffix = f"/{path_with_ext}"

        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (n)
                WHERE (
                        n.id STARTS WITH 'note:'
                        OR n.path IS NOT NULL
                      )
                  AND (
                        toLower(coalesce(n.vault_rel_path, '')) = $path_no_ext
                        OR toLower(coalesce(n.vault_rel_path, '')) = $path_with_ext
                        OR (
                            n.path IS NOT NULL
                            AND toLower(replace(n.path, '\\\\', '/')) ENDS WITH $path_with_ext_suffix
                        )
                  )
                RETURN n, labels(n) as labels
                ORDER BY CASE
                    WHEN toLower(coalesce(n.vault_rel_path, '')) = $path_no_ext THEN 0
                    WHEN toLower(coalesce(n.vault_rel_path, '')) = $path_with_ext THEN 1
                    WHEN coalesce(n.placeholder, false) = false THEN 2
                    ELSE 3
                END
                LIMIT 1
                """,
                path_no_ext=path_no_ext,
                path_with_ext=path_with_ext,
                path_with_ext_suffix=path_with_ext_suffix,
            )
            record = result.single()
            if not record:
                return None

            node_dict = dict(record["n"])
            node_dict["_labels"] = record["labels"]
            return self._clean_node(node_dict)

    def find_note_candidates_by_stem(self, stem: str, limit: int = 20) -> list[dict]:
        """Find Note candidates by stem-like matching."""
        normalized_stem = stem.replace("\\", "/").split("/")[-1].strip().lower()
        if normalized_stem.endswith(".md"):
            normalized_stem = normalized_stem[:-3]
        stem_with_ext = f"{normalized_stem}.md"

        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (n)
                WHERE (
                        n.id STARTS WITH 'note:'
                        OR n.path IS NOT NULL
                      )
                  AND (
                        toLower(coalesce(n.stem, '')) = $stem
                        OR toLower(coalesce(n.name, '')) = $stem
                        OR (
                            n.path IS NOT NULL
                            AND toLower(split(replace(n.path, '\\\\', '/'), '/')[-1]) = $stem_with_ext
                        )
                  )
                RETURN n, labels(n) as labels
                ORDER BY coalesce(n.placeholder, false) ASC,
                         size(coalesce(n.vault_rel_path, '')) ASC,
                         size(coalesce(n.name, '')) ASC
                LIMIT $limit
                """,
                stem=normalized_stem,
                stem_with_ext=stem_with_ext,
                limit=limit,
            )

            nodes: list[dict] = []
            for r in result:
                node_dict = dict(r["n"])
                node_dict["_labels"] = r["labels"]
                nodes.append(self._clean_node(node_dict))

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

        if "name" in props and isinstance(props["name"], str):
            props["name"] = props["name"].strip()

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
                return self._clean_node(node_dict)
            return None

    def delete_node(self, node_id: str) -> dict:
        """Delete a node and all its relationships.

        Args:
            node_id: The node's unique identifier

        Returns:
            Dict with deletion status and edge count
        """
        with self.driver.session() as session:
            count_result = session.run(
                """
                MATCH (n {id: $id})-[r]-()
                RETURN count(r) as edge_count
                """,
                id=node_id,
            )
            count_record = count_result.single()
            edge_count = count_record["edge_count"] if count_record else 0

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

    def merge_nodes_simple(self, keep_id: str, merge_id: str) -> dict:
        """Merge two nodes without APOC (simpler, creates new relationship types).

        For use when APOC is not available.
        """
        with self.driver.session() as session:
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

            result = session.run(
                """
                MATCH (merge {id: $merge_id})
                OPTIONAL MATCH (source)-[r_in]->(merge)
                WHERE source.id <> $keep_id
                OPTIONAL MATCH (merge)-[r_out]->(target)
                WHERE target.id <> $keep_id
                RETURN 
                    collect(DISTINCT {source: source.id, type: type(r_in), props: properties(r_in)}) as incoming,
                    collect(DISTINCT {target: target.id, type: type(r_out), props: properties(r_out)}) as outgoing
                """,
                merge_id=merge_id,
                keep_id=keep_id,
            )
            record = result.single()
            incoming = [e for e in record["incoming"] if e["source"]]
            outgoing = [e for e in record["outgoing"] if e["target"]]

            for edge in incoming:
                rel_props = dict(edge.get("props") or {})
                rel_props.pop("id", None)
                rel_props.pop("created_at", None)
                session.run(
                    f"""
                    MATCH (source {{id: $source_id}})
                    MATCH (keep {{id: $keep_id}})
                    MERGE (source)-[r:{edge["type"]}]->(keep)
                    SET r += $rel_props
                    """,
                    source_id=edge["source"],
                    keep_id=keep_id,
                    rel_props=rel_props,
                )

            for edge in outgoing:
                rel_props = dict(edge.get("props") or {})
                rel_props.pop("id", None)
                rel_props.pop("created_at", None)
                session.run(
                    f"""
                    MATCH (keep {{id: $keep_id}})
                    MATCH (target {{id: $target_id}})
                    MERGE (keep)-[r:{edge["type"]}]->(target)
                    SET r += $rel_props
                    """,
                    keep_id=keep_id,
                    target_id=edge["target"],
                    rel_props=rel_props,
                )

            session.run(
                """
                MATCH (n {id: $id})
                DETACH DELETE n
                """,
                id=merge_id,
            )

            return {"merged": True, "edges_transferred": len(incoming) + len(outgoing)}

    def add_edge(
        self,
        from_id: str,
        to_id: str,
        relation: str,
        properties: dict | None = None,
        source_note: str | None = None,
    ) -> dict:
        """Create or update an edge between two nodes (idempotent).

        Uses MERGE on (from, to, relation_type) so duplicate calls
        update instead of creating parallel edges.

        Args:
            from_id: Source node ID
            to_id: Target node ID
            relation: Relationship type (CONTRIBUTES_TO, MOTIVATES, etc.)
            properties: Additional edge properties (confidence, fact, etc.)
            source_note: Node ID of the note this edge was extracted from

        Returns:
            Dict with success status and edge details
        """
        props = properties.copy() if properties else {}
        if source_note:
            props["source_note"] = source_note

        sanitized_rel = self._sanitize_label(relation).upper()
        now = datetime.now().isoformat()

        with self.driver.session() as session:
            result = session.run(
                f"""
                MATCH (a {{id: $from_id}})
                MATCH (b {{id: $to_id}})
                MERGE (a)-[r:{sanitized_rel}]->(b)
                ON CREATE SET r.id = $edge_id, r.created_at = $now, r += $props
                ON MATCH SET r.id = coalesce(r.id, $edge_id),
                             r.created_at = coalesce(r.created_at, $now),
                             r += $props
                RETURN r, a.name as from_name, b.name as to_name
                """,
                from_id=from_id,
                to_id=to_id,
                props=props,
                edge_id=f"edge:{uuid4().hex[:12]}",
                now=now,
            )
            record = result.single()
            if not record:
                return {"success": False, "error": "One or both nodes not found"}
            edge_props = dict(record["r"])
            return {
                "success": True,
                "edge_id": edge_props["id"],
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

    def get_edges_by_source_note(
        self,
        node_id: str,
        source_note: str,
        relation: str | None = None,
        direction: str = "out",
    ) -> list[dict]:
        """Get edges from a node filtered by source_note (provenance).

        Args:
            node_id: The node's unique identifier
            source_note: Node ID of the originating note (e.g., "note:foo")
            relation: Optional relation type filter (e.g., "WIKILINK")
            direction: "in", "out", or "both"

        Returns:
            List of edges with their properties and target info
        """
        rel_filter = f":{self._sanitize_label(relation).upper()}" if relation else ""

        if direction == "out":
            pattern = f"(n)-[r{rel_filter}]->(target)"
        elif direction == "in":
            pattern = f"(n)<-[r{rel_filter}]-(target)"
        else:
            pattern = f"(n)-[r{rel_filter}]-(target)"

        with self.driver.session() as session:
            result = session.run(
                f"""
                MATCH (n {{id: $node_id}})
                MATCH {pattern}
                WHERE r.source_note = $source_note
                RETURN r, target.id as target_id, target.name as target_name,
                       type(r) as relation
                """,
                node_id=node_id,
                source_note=source_note,
            )
            return [
                {
                    "edge": dict(r["r"]),
                    "target_id": r["target_id"],
                    "target_name": r["target_name"],
                    "relation": r["relation"],
                }
                for r in result
            ]

    def delete_edges_by_source_note(
        self,
        node_id: str,
        source_note: str,
        relation: str | None = None,
        direction: str = "out",
    ) -> int:
        """Delete edges from a node filtered by source_note.

        Args:
            node_id: The node's unique identifier
            source_note: Node ID of the originating note (e.g., "note:foo")
            relation: Optional relation type filter (e.g., "WIKILINK")
            direction: "in", "out", or "both"

        Returns:
            Number of edges deleted
        """
        rel_filter = f":{self._sanitize_label(relation).upper()}" if relation else ""

        if direction == "out":
            pattern = f"(n)-[r{rel_filter}]->()"
        elif direction == "in":
            pattern = f"(n)<-[r{rel_filter}]-()"
        else:
            pattern = f"(n)-[r{rel_filter}]-()"

        with self.driver.session() as session:
            result = session.run(
                f"""
                MATCH (n {{id: $node_id}})
                MATCH {pattern}
                WHERE r.source_note = $source_note
                DELETE r
                RETURN count(*) as deleted
                """,
                node_id=node_id,
                source_note=source_note,
            )
            record = result.single()
            return record["deleted"] if record else 0

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
                       r.id as edge_id,
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
                        "node": self._clean_node(node_dict),
                        "edge_id": r["edge_id"],
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

    def get_stats(self) -> dict[str, Any]:
        """Get database statistics."""
        with self.driver.session() as session:
            node_count = session.run("MATCH (n) RETURN count(n) as count").single()[
                "count"
            ]
            rel_count = session.run(
                "MATCH ()-[r]->() RETURN count(r) as count"
            ).single()["count"]

            labels = session.run("CALL db.labels() YIELD label RETURN label").data()
            label_counts = {}
            for row in labels:
                label = row["label"]
                if label in INTERNAL_LABELS:
                    continue
                count = session.run(
                    f"MATCH (n:{label}) RETURN count(n) as count"
                ).single()["count"]
                label_counts[label] = count

            return {
                "nodes": node_count,
                "relationships": rel_count,
                "by_label": label_counts,
            }

    def _serialize_value(self, value: Any) -> Any:
        """Convert Neo4j values into JSON-serializable values."""
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, list):
            return [self._serialize_value(v) for v in value]
        if isinstance(value, dict):
            return {k: self._serialize_value(v) for k, v in value.items()}
        return str(value)

    def export_snapshot(self) -> dict[str, Any]:
        """Export full graph as a JSON-serializable snapshot payload."""
        with self.driver.session() as session:
            node_rows = session.run(
                """
                MATCH (n)
                RETURN n, labels(n) as labels
                ORDER BY n.id
                """
            )

            nodes = []
            for row in node_rows:
                props = self._serialize_value(dict(row["n"]))
                node_id = props.get("id")
                if not node_id:
                    continue
                nodes.append(
                    {
                        "id": node_id,
                        "labels": sorted(row["labels"]),
                        "properties": props,
                    }
                )

            edge_rows = session.run(
                """
                MATCH (a)-[r]->(b)
                RETURN a.id as from_id, b.id as to_id, type(r) as relation, properties(r) as props
                ORDER BY from_id, relation, to_id, coalesce(props.id, '')
                """
            )

            edges = []
            for row in edge_rows:
                props = self._serialize_value(dict(row["props"]))
                if not props.get("id"):
                    edge_basis = {
                        "from_id": row["from_id"],
                        "to_id": row["to_id"],
                        "relation": row["relation"],
                        "properties": props,
                    }
                    digest = hashlib.sha256(
                        json.dumps(
                            edge_basis, sort_keys=True, separators=(",", ":")
                        ).encode("utf-8")
                    ).hexdigest()[:24]
                    props["id"] = f"edge:snap_{digest}"

                edges.append(
                    {
                        "from_id": row["from_id"],
                        "to_id": row["to_id"],
                        "relation": row["relation"],
                        "properties": props,
                    }
                )

        return {
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "stats": {"nodes": len(nodes), "edges": len(edges)},
            "nodes": nodes,
            "edges": edges,
        }

    def import_snapshot(
        self, snapshot: dict[str, Any], clear_first: bool = False
    ) -> dict[str, int]:
        """Import a snapshot payload back into Neo4j."""
        nodes = snapshot.get("nodes", [])
        edges = snapshot.get("edges", [])

        def _import_tx(tx) -> dict[str, int]:
            node_count = 0
            edge_count = 0

            if clear_first:
                tx.run("MATCH (n) DETACH DELETE n")

            for node in nodes:
                props = dict(node.get("properties", {}))
                node_id = props.get("id") or node.get("id")
                if not node_id:
                    continue

                labels = node.get("labels") or ["Concept"]
                sanitized_labels = [
                    self._sanitize_label(label) for label in labels if label
                ]
                if not sanitized_labels:
                    sanitized_labels = ["Concept"]
                label_spec = ":".join(dict.fromkeys(sanitized_labels))

                props.pop("title", None)
                if not props.get("name"):
                    props["name"] = node_id
                props["id"] = node_id

                tx.run(
                    """
                    MERGE (n {id: $node_id})
                    SET n += $props
                    """,
                    node_id=node_id,
                    props=props,
                )
                tx.run(
                    f"""
                    MATCH (n {{id: $node_id}})
                    SET n:{label_spec}
                    """,
                    node_id=node_id,
                )
                node_count += 1

            for edge in edges:
                from_id = edge.get("from_id")
                to_id = edge.get("to_id")
                relation = edge.get("relation")
                if not from_id or not to_id or not relation:
                    continue

                rel = self._sanitize_label(relation).upper()
                props = dict(edge.get("properties", {}))
                edge_id = props.get("id")

                if edge_id:
                    existing_edge = tx.run(
                        """
                        MATCH (x)-[r {id: $edge_id}]->(y)
                        RETURN x.id as from_id, y.id as to_id, type(r) as relation
                        LIMIT 1
                        """,
                        edge_id=edge_id,
                    ).single()
                    if existing_edge and (
                        existing_edge["from_id"] != from_id
                        or existing_edge["to_id"] != to_id
                        or existing_edge["relation"] != rel
                    ):
                        raise ValueError(
                            f"Edge id collision for {edge_id}: "
                            f"existing {existing_edge['from_id']}-[{existing_edge['relation']}]->{existing_edge['to_id']} "
                            f"vs snapshot {from_id}-[{rel}]->{to_id}"
                        )

                    result = tx.run(
                        f"""
                        MATCH (a {{id: $from_id}})
                        MATCH (b {{id: $to_id}})
                        MERGE (a)-[r:{rel} {{id: $edge_id}}]->(b)
                        SET r += $props
                        RETURN count(r) as written
                        """,
                        from_id=from_id,
                        to_id=to_id,
                        edge_id=edge_id,
                        props=props,
                    )
                else:
                    if clear_first:
                        result = tx.run(
                            f"""
                            MATCH (a {{id: $from_id}})
                            MATCH (b {{id: $to_id}})
                            CREATE (a)-[r:{rel}]->(b)
                            SET r += $props
                            RETURN count(r) as written
                            """,
                            from_id=from_id,
                            to_id=to_id,
                            props=props,
                        )
                    else:
                        result = tx.run(
                            f"""
                            MATCH (a {{id: $from_id}})
                            MATCH (b {{id: $to_id}})
                            MERGE (a)-[r:{rel}]->(b)
                            SET r += $props
                            RETURN count(r) as written
                            """,
                            from_id=from_id,
                            to_id=to_id,
                            props=props,
                        )

                record = result.single()
                if not record:
                    continue
                edge_count += int(record["written"])

            return {"nodes": node_count, "edges": edge_count}

        with self.driver.session() as session:
            result = session.execute_write(_import_tx)

        self._create_indexes()

        return result

    # =========================================================================
    # VECTOR / EMBEDDING OPERATIONS
    # =========================================================================

    def ensure_vector_index(self):
        """Create vector index if it doesn't exist and verify availability."""
        self._create_indexes()
        with self.driver.session() as session:
            record = session.run(
                """
                SHOW INDEXES YIELD name, type, state
                WHERE name = 'entity_embeddings'
                RETURN type, state
                """
            ).single()
            if not record or record["type"] != "VECTOR":
                raise RuntimeError(
                    "Neo4j vector index 'entity_embeddings' is not available"
                )

    def set_embedding(self, node_id: str, embedding: list[float]) -> bool:
        """Store embedding on a node and add Entity label for vector indexing.

        Args:
            node_id: The node's unique identifier
            embedding: Vector embedding (list of floats)

        Returns:
            True if node found and embedding set, False otherwise
        """
        if len(embedding) != self.embedding_dimensions:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self.embedding_dimensions}, got {len(embedding)}"
            )

        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (n {id: $id})
                SET n.embedding = $embedding, n:Entity
                RETURN n.id as id
                """,
                id=node_id,
                embedding=embedding,
            )
            return result.single() is not None

    def search_similar(
        self,
        query_embedding: list[float],
        node_types: list[str] | None = None,
        limit: int = 10,
    ) -> list[dict]:
        """Semantic search using Neo4j native vector index.

        Args:
            query_embedding: Query vector
            node_types: Optional label filter
            limit: Max results

        Returns:
            List of dicts with node_id, node_type, name, summary, score.
            Embedding is never included.
        """
        if len(query_embedding) != self.embedding_dimensions:
            raise ValueError(
                f"Query embedding dimension mismatch: expected {self.embedding_dimensions}, got {len(query_embedding)}"
            )
        limit = max(1, limit)

        with self.driver.session() as session:
            result = session.run(
                """
                CALL db.index.vector.queryNodes('entity_embeddings', $limit, $embedding)
                YIELD node, score
                RETURN node.id as node_id,
                       node.name as name,
                       node.summary as summary,
                       labels(node) as labels,
                       score
                """,
                embedding=query_embedding,
                limit=limit,
            )

            results = []
            for r in result:
                labels = [l for l in r["labels"] if l not in INTERNAL_LABELS]
                node_type = labels[0] if labels else "Unknown"

                if node_types and node_type not in node_types:
                    continue

                results.append(
                    {
                        "node_id": r["node_id"],
                        "node_type": node_type,
                        "name": r["name"],
                        "summary": r["summary"],
                        "score": r["score"],
                    }
                )
            return results
