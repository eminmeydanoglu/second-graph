"""Neo4j graph storage and import."""

from pathlib import Path
from typing import Any

from neo4j import GraphDatabase

from .builder import VaultGraph


class Neo4jStorage:
    """Neo4j graph database wrapper."""

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
            ]:
                try:
                    session.run(f"CREATE INDEX IF NOT EXISTS FOR (n:{label}) ON (n.id)")
                    session.run(
                        f"CREATE INDEX IF NOT EXISTS FOR (n:{label}) ON (n.title)"
                    )
                except Exception:
                    pass  # Index might already exist

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
