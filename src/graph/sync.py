"""Synchronization logic between Markdown files and Knowledge Graph."""

import logging
from pathlib import Path
from typing import Any

from ..parser.markdown import parse_note, ParsedNote
from .neo4j_storage import Neo4jStorage
from .routing_text import build_routing_text
from .schema import NodeType, EdgeType, generate_node_id
from ..vector.embedder import Embedder

log = logging.getLogger(__name__)


class NoteSynchronizer:
    """Synchronizes Obsidian notes with the Neo4j graph.

    Performs 'intelligent sync':
    1. Parses the note file.
    2. Compares with existing graph state.
    3. Adds missing edges/nodes.
    4. Removes deleted edges.
    5. Updates node properties.
    6. Stores embeddings in Neo4j (native vector index).
    """

    def __init__(
        self,
        storage: Neo4jStorage,
        embedder: Embedder,
    ):
        self.storage = storage
        self.embedder = embedder

    def sync_note_from_file(self, file_path: str | Path) -> dict[str, Any]:
        """Sync a single note file into the graph.

        Args:
            file_path: Absolute path to the markdown file.

        Returns:
            Dict containing stats of changes applied.
        """
        path = Path(file_path)
        if not path.exists():
            return {"success": False, "error": f"File not found: {path}"}

        try:
            note = parse_note(path)
        except Exception as e:
            return {"success": False, "error": f"Failed to parse note: {e}"}

        node_type = self._determine_node_type(note)
        node_id = generate_node_id(node_type, note.title)

        props = {
            "path": str(path),
            "title": note.title,
            "modified_at": path.stat().st_mtime,
            **note.frontmatter,
        }

        existing = self.storage.get_node(node_id)
        if existing:
            self.storage.update_node(node_id, props)
            action = "updated"
        else:
            self.storage.add_node(node_type, node_id, note.title, props)
            action = "created"

        edge_stats = self._sync_wikilinks(node_id, note.wikilinks)
        tag_stats = self._sync_tags(node_id, note.tags)

        # Still update tags property for easy access, but now we also have real graph edges
        self.storage.update_node(node_id, {"tags": note.tags})

        self._update_embedding(node_id, node_type, note)

        return {
            "success": True,
            "node_id": node_id,
            "action": action,
            "edges": edge_stats,
            "tags": tag_stats,
            "source_note": node_id,
        }

    # Backward compatibility alias
    source_note = sync_note_from_file

    def _determine_node_type(self, note: ParsedNote) -> str:
        """Determine canonical node type from frontmatter or default to Note."""
        if "type" in note.frontmatter:
            raw_type = str(note.frontmatter["type"])
            for node_type in NodeType:
                if node_type.value.lower() == raw_type.lower():
                    return node_type.value
        return NodeType.NOTE.value

    def _sync_wikilinks(self, node_id: str, wikilinks: list[str]) -> dict:
        """Diff and sync outgoing wikilinks for a note.

        Filters by relation=WIKILINK and source_note=node_id, so only
        wikilink edges owned by this note are affected.
        """
        stats = {"added": 0, "removed": 0, "kept": 0}

        current_edges = self.storage.get_edges_by_source_note(
            node_id, source_note=node_id, relation="WIKILINK", direction="out"
        )

        current_targets: dict[str, str] = {}
        for edge in current_edges:
            if edge.get("target_name"):
                current_targets[edge["target_name"]] = edge["edge"]["id"]

        desired_links = set(wikilinks)
        existing_link_names = set(current_targets.keys())

        to_add = desired_links - existing_link_names
        to_remove = existing_link_names - desired_links
        to_keep = desired_links & existing_link_names

        stats["kept"] = len(to_keep)

        for target_name in to_remove:
            edge_id = current_targets[target_name]
            self.storage.delete_edge(edge_id)
            stats["removed"] += 1

        for target_name in to_add:
            target_type = NodeType.NOTE.value
            target_id = generate_node_id(target_type, target_name)

            if not self.storage.get_node(target_id):
                matches = self.storage.find_nodes(
                    target_name,
                    node_type=NodeType.NOTE.value,
                    match_type="exact",
                )
                if matches:
                    target_id = matches[0]["id"]
                else:
                    self.storage.add_node(
                        target_type, target_id, target_name, {"placeholder": True}
                    )

            self.storage.add_edge(
                node_id, target_id, EdgeType.WIKILINK.value, source_note=node_id
            )
            stats["added"] += 1

        return stats

    def _sync_tags(self, node_id: str, tags: list[str]) -> dict:
        """Diff and sync tag relationships for a note.

        Ensures Tag nodes exist and TAGGED_WITH edges are current.
        """
        stats = {"added": 0, "removed": 0, "kept": 0}

        # Get current TAGGED_WITH edges from this node
        # Note: TAGGED_WITH edges don't necessarily need source_note property
        # because they are intrinsic to the note's content.
        current_edges = self.storage.get_neighbors(
            node_id, direction="out", edge_types=[EdgeType.TAGGED_WITH.value]
        )

        current_tags: dict[str, str] = {}
        for neighbor in current_edges:
            # neighbor structure: {node: {name: "tagname", ...}, edge_id: "...", ...}
            tag_name = neighbor["node"].get("name")
            if tag_name:
                current_tags[tag_name] = neighbor["edge_id"]

        desired_tags = set(tags)
        existing_tag_names = set(current_tags.keys())

        to_add = desired_tags - existing_tag_names
        to_remove = existing_tag_names - desired_tags
        to_keep = desired_tags & existing_tag_names

        stats["kept"] = len(to_keep)

        for tag_name in to_remove:
            edge_id = current_tags[tag_name]
            self.storage.delete_edge(edge_id)
            stats["removed"] += 1

        for tag_name in to_add:
            tag_id = generate_node_id(NodeType.TAG.value, tag_name)

            if not self.storage.get_node(tag_id):
                self.storage.add_node(NodeType.TAG.value, tag_id, tag_name)

            self.storage.add_edge(
                node_id,
                tag_id,
                EdgeType.TAGGED_WITH.value,
                {"confidence": 1.0},
            )
            stats["added"] += 1

        return stats

    def _update_embedding(self, node_id: str, node_type: str, note: ParsedNote):
        """Store embedding on the Neo4j node."""
        props = {
            **note.frontmatter,
            "name": note.title,
            "title": note.title,
            "tags": note.tags,
        }
        text = build_routing_text(node_type, props)

        embedding = self.embedder.embed(text)
        self.storage.set_embedding(node_id, embedding)
