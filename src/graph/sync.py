"""Synchronization logic between Markdown files and Knowledge Graph."""

import logging
from pathlib import Path
from typing import Any

from ..parser.markdown import parse_note, ParsedNote
from .neo4j_storage import Neo4jStorage
from .schema import NodeType, EdgeType, generate_node_id
from ..vector.store import VectorStore
from ..vector.embedder import Embedder

log = logging.getLogger(__name__)


class NoteSynchronizer:
    """Synchronizes Obsidian notes with the Neo4j graph and Vector store.

    Performs 'intelligent sync':
    1. Parses the note file.
    2. Compares with existing graph state.
    3. Adds missing edges/nodes.
    4. Removes deleted edges.
    5. Updates node properties.
    6. Refreshes vector embeddings.
    """

    def __init__(
        self,
        storage: Neo4jStorage,
        vectors: VectorStore,
        embedder: Embedder,
    ):
        self.storage = storage
        self.vectors = vectors
        self.embedder = embedder

    def sync_note(self, file_path: str | Path) -> dict[str, Any]:
        """Sync a single note file to the graph.

        Args:
            file_path: Absolute path to the markdown file.

        Returns:
            Dict containing stats of changes applied.
        """
        path = Path(file_path)
        if not path.exists():
            return {"success": False, "error": f"File not found: {path}"}

        # 1. Parse Note
        try:
            note = parse_note(path)
        except Exception as e:
            return {"success": False, "error": f"Failed to parse note: {e}"}

        # 2. Identify/Create Main Node
        node_type = self._determine_node_type(note)
        node_id = generate_node_id(node_type, note.title)

        # Upsert node
        props = {
            "path": str(path),
            "title": note.title,
            "modified_at": path.stat().st_mtime,
            **note.frontmatter,
        }

        # Check if node exists to decide add vs update
        existing = self.storage.get_node(node_id)
        if existing:
            self.storage.update_node(node_id, props)
            action = "updated"
        else:
            self.storage.add_node(node_type, node_id, note.title, props)
            action = "created"

        # 3. Sync Edges (Wikilinks)
        edge_stats = self._sync_wikilinks(node_id, note.wikilinks)

        # 4. Sync Tags (as Edges or Properties)
        # For now, let's treat tags as properties 'tags',
        # but advanced usage might want Tag nodes.
        self.storage.update_node(node_id, {"tags": note.tags})

        # 5. Update Vector Embedding
        self._update_embedding(node_id, node_type, note)

        return {
            "success": True,
            "node_id": node_id,
            "action": action,
            "edges": edge_stats,
        }

    def _determine_node_type(self, note: ParsedNote) -> str:
        """Determine node type from frontmatter or default to Note."""
        if "type" in note.frontmatter:
            return note.frontmatter["type"]
        # Can add more heuristics here (e.g. folder based)
        return NodeType.NOTE.value

    def _sync_wikilinks(self, source_id: str, wikilinks: list[str]) -> dict:
        """Diff and sync outgoing wikilinks."""
        stats = {"added": 0, "removed": 0, "kept": 0}

        # Get current outgoing WIKILINK edges
        current_neighbors = self.storage.get_neighbors(
            source_id, direction="out", edge_types=[EdgeType.WIKILINK.value]
        )

        # Map target_id -> edge_id for existing links
        # Note: We need to know what the target ID *would* be for the wikilink text
        # This is tricky because we don't know the type of the target node if it doesn't exist.
        # Assumption: Wikilinks target other "Note" types or "Concept" types mostly.
        # For simple sync, we'll assume target is Note if not found.

        current_targets = {
            n["node"]["name"]: n["node"]["id"] for n in current_neighbors
        }

        # Determine desired state
        desired_links = set(wikilinks)

        # Diff
        existing_link_names = set(current_targets.keys())

        to_add = desired_links - existing_link_names
        to_remove = existing_link_names - desired_links
        to_keep = desired_links & existing_link_names

        stats["kept"] = len(to_keep)

        # Remove deleted links
        for target_name in to_remove:
            # We need to find the specific edge to delete
            # Since get_neighbors returns aggregated info, we might need a more specific query
            # or just iterate connections from get_node.
            # Optimized: Let's assume we can delete by pattern
            target_id = current_targets[target_name]
            # Use specific delete edge logic
            # Find edges between source and target with type WIKILINK
            edges = self.storage.find_edges(
                from_id=source_id, to_id=target_id, relation=EdgeType.WIKILINK.value
            )
            for edge in edges:
                self.storage.delete_edge(edge["edge"]["id"])
            stats["removed"] += 1

        # Add new links
        for target_name in to_add:
            # We need a target ID.
            # Strategy:
            # 1. Try to find a node with this name (fuzzy/exact).
            # 2. If not found, create a placeholder Note node.

            matches = self.storage.find_nodes(target_name, match_type="exact")
            if matches:
                target_id = matches[0]["id"]
            else:
                # Create placeholder
                target_type = NodeType.NOTE.value
                target_id = generate_node_id(target_type, target_name)
                # Only create if it really doesn't exist (find_nodes check might be loose)
                if not self.storage.get_node(target_id):
                    self.storage.add_node(
                        target_type, target_id, target_name, {"placeholder": True}
                    )

            # Create edge
            self.storage.add_edge(source_id, target_id, EdgeType.WIKILINK.value)
            stats["added"] += 1

        return stats

    def _update_embedding(self, node_id: str, node_type: str, note: ParsedNote):
        """Update vector store embedding."""
        # Create rich text representation
        text = f"# {note.title}\n\n"
        if note.tags:
            text += f"Tags: {', '.join(note.tags)}\n"
        text += f"{note.content}"

        embedding = self.embedder.embed(text)

        # Update/Add entity
        # We treat every Note as an Entity in the vector store for semantic search
        self.vectors.add_entity(
            node_id=node_id,
            node_type=node_type,
            name=note.title,
            summary=note.content[:1000],  # Store first 1k chars as summary
            embedding=embedding,
        )
