"""NetworkX graph builder for Obsidian vault."""

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import networkx as nx

from ..parser.markdown import ParsedNote


@dataclass
class GraphStats:
    """Statistics about the graph."""

    nodes: int
    edges: int
    node_types: dict[str, int]
    edge_types: dict[str, int]

    def __str__(self) -> str:
        nodes_str = ", ".join(
            f"{k}: {v}" for k, v in sorted(self.node_types.items(), key=lambda x: -x[1])
        )
        edges_str = ", ".join(
            f"{k}: {v}" for k, v in sorted(self.edge_types.items(), key=lambda x: -x[1])
        )
        return (
            f"Graph Stats:\n"
            f"  Nodes: {self.nodes} ({nodes_str})\n"
            f"  Edges: {self.edges} ({edges_str})"
        )


class VaultGraph:
    """Knowledge graph for an Obsidian vault."""

    def __init__(self):
        self.graph = nx.DiGraph()
        self._note_paths: dict[str, Path] = {}  # title -> path mapping

    def add_note(self, note: ParsedNote) -> None:
        """Add a note and its relationships to the graph."""
        note_id = str(note.path)

        # Add note node
        self.graph.add_node(
            note_id,
            type="Note",
            title=note.title,
            folder=note.folder,
        )
        self._note_paths[note.title] = note.path

        # Add folder relationship
        if note.folder and note.folder != "root":
            folder_id = f"folder:{note.folder}"
            if not self.graph.has_node(folder_id):
                self.graph.add_node(folder_id, type="Folder", name=note.folder)
            self.graph.add_edge(
                note_id,
                folder_id,
                type="in_folder",
                confidence=1.0,
            )

        # Add wikilinks
        for link in note.wikilinks:
            # Normalize link (could be "Folder/Note" or just "Note")
            link_target = link.split("/")[-1] if "/" in link else link

            # Try to find existing note, otherwise create placeholder
            target_id = self._resolve_link(link_target) or f"note:{link_target}"

            if not self.graph.has_node(target_id):
                self.graph.add_node(
                    target_id,
                    type="Note",
                    title=link_target,
                    placeholder=True,  # Mark as unresolved
                )

            self.graph.add_edge(
                note_id,
                target_id,
                type="wikilink",
                confidence=1.0,
            )

        # Add tags
        for tag in note.tags:
            tag_id = f"tag:{tag}"
            if not self.graph.has_node(tag_id):
                self.graph.add_node(tag_id, type="Tag", name=tag)
            self.graph.add_edge(
                note_id,
                tag_id,
                type="tagged_with",
                confidence=1.0,
            )

    def _resolve_link(self, title: str) -> str | None:
        """Try to resolve a wikilink to an existing note path."""
        if title in self._note_paths:
            return str(self._note_paths[title])
        # Case-insensitive fallback
        for note_title, path in self._note_paths.items():
            if note_title.lower() == title.lower():
                return str(path)
        return None

    def resolve_placeholders(self) -> int:
        """Resolve placeholder nodes to actual notes.

        Returns:
            Number of placeholders resolved
        """
        resolved = 0
        placeholders = [
            (n, d) for n, d in self.graph.nodes(data=True) if d.get("placeholder")
        ]

        for node_id, data in placeholders:
            title = data.get("title", "")
            actual_path = self._resolve_link(title)

            if actual_path and actual_path != node_id:
                # Merge placeholder into actual node
                for pred in list(self.graph.predecessors(node_id)):
                    edge_data = self.graph.edges[pred, node_id]
                    self.graph.add_edge(pred, actual_path, **edge_data)
                for succ in list(self.graph.successors(node_id)):
                    edge_data = self.graph.edges[node_id, succ]
                    self.graph.add_edge(actual_path, succ, **edge_data)
                self.graph.remove_node(node_id)
                resolved += 1

        return resolved

    def get_stats(self) -> GraphStats:
        """Get statistics about the graph."""
        node_types = Counter(
            data.get("type", "Unknown") for _, data in self.graph.nodes(data=True)
        )
        edge_types = Counter(
            data.get("type", "unknown") for _, _, data in self.graph.edges(data=True)
        )
        return GraphStats(
            nodes=self.graph.number_of_nodes(),
            edges=self.graph.number_of_edges(),
            node_types=dict(node_types),
            edge_types=dict(edge_types),
        )

    def get_backlinks(self, note_path: str) -> list[str]:
        """Get all notes that link to a given note."""
        return [
            pred
            for pred in self.graph.predecessors(note_path)
            if self.graph.edges[pred, note_path].get("type") == "wikilink"
        ]

    def get_outlinks(self, note_path: str) -> list[str]:
        """Get all notes that a given note links to."""
        return [
            succ
            for succ in self.graph.successors(note_path)
            if self.graph.edges[note_path, succ].get("type") == "wikilink"
        ]

    def get_tags(self, note_path: str) -> list[str]:
        """Get all tags for a given note."""
        return [
            self.graph.nodes[succ].get("name", "")
            for succ in self.graph.successors(note_path)
            if self.graph.edges[note_path, succ].get("type") == "tagged_with"
        ]

    def save(self, path: Path) -> None:
        """Save graph to JSON file."""
        data = nx.node_link_data(self.graph)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: Path) -> "VaultGraph":
        """Load graph from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        vault_graph = cls()
        vault_graph.graph = nx.node_link_graph(data)

        # Rebuild note paths index
        for node_id, data in vault_graph.graph.nodes(data=True):
            if data.get("type") == "Note" and not data.get("placeholder"):
                title = data.get("title", "")
                vault_graph._note_paths[title] = Path(node_id)

        return vault_graph
