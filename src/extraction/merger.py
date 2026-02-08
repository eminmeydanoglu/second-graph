"""Merge extracted entities/relationships into the vault graph."""

from dataclasses import dataclass
from pathlib import Path

from .reader import ExtractionReader, ExtractedNote, Entity, Relationship
from ..graph.builder import VaultGraph


@dataclass
class MergeStats:
    """Statistics about the merge operation."""

    entities_added: int = 0
    entities_existing: int = 0
    entities_resolved_to_notes: int = 0  # Identity resolution hits
    relationships_added: int = 0
    note_links_added: int = 0
    extractions_processed: int = 0

    def __str__(self) -> str:
        return (
            f"Merge Stats:\n"
            f"  Extractions processed: {self.extractions_processed}\n"
            f"  Entities added: {self.entities_added} (existing: {self.entities_existing})\n"
            f"  Entities resolved to Notes: {self.entities_resolved_to_notes}\n"
            f"  Relationships added: {self.relationships_added}\n"
            f"  Note→Entity links: {self.note_links_added}"
        )


class ExtractionMerger:
    """Merges extracted entities and relationships into the vault graph."""

    def __init__(self, graph: VaultGraph):
        self.graph = graph
        self._entity_registry: dict[str, str] = {}  # canonical name -> node_id
        self._note_title_index: dict[str, str] = {}  # normalized title -> node_id
        self._build_note_index()

    def _normalize_name(self, name: str) -> str:
        """Normalize entity name for matching."""
        return name.lower().strip()

    def _build_note_index(self) -> None:
        """Build index of note titles for identity resolution."""
        for node_id, data in self.graph.graph.nodes(data=True):
            if data.get("type") == "Note":
                title = data.get("title", "")
                if title:
                    self._note_title_index[self._normalize_name(title)] = node_id

    def _find_matching_note(self, name: str) -> str | None:
        """Find a Note node that matches entity name (identity resolution)."""
        canonical = self._normalize_name(name)
        return self._note_title_index.get(canonical)

    def _get_entity_id(self, entity: Entity) -> str:
        """Get or create node ID for an entity.

        Identity Resolution: If a Note exists with the same title as the entity,
        use the Note's node_id instead of creating a separate entity node.
        """
        canonical = self._normalize_name(entity.name)

        # Already registered?
        if canonical in self._entity_registry:
            return self._entity_registry[canonical]

        # Check aliases
        for alias in entity.aliases:
            alias_norm = self._normalize_name(alias)
            if alias_norm in self._entity_registry:
                self._entity_registry[canonical] = self._entity_registry[alias_norm]
                return self._entity_registry[alias_norm]

        # IDENTITY RESOLUTION: Check if matching Note exists
        matching_note = self._find_matching_note(entity.name)
        if matching_note:
            self._entity_registry[canonical] = matching_note
            # Also register aliases pointing to the note
            for alias in entity.aliases:
                self._entity_registry[self._normalize_name(alias)] = matching_note
            return matching_note

        # No matching note - create new entity node
        node_id = f"entity:{entity.type.lower()}:{canonical}"
        self._entity_registry[canonical] = node_id

        # Register aliases
        for alias in entity.aliases:
            self._entity_registry[self._normalize_name(alias)] = node_id

        return node_id

    def _add_entity(self, entity: Entity) -> tuple[str, bool, bool]:
        """Add entity to graph. Returns (node_id, is_new, resolved_to_note).

        If entity resolves to an existing Note, enriches the Note with entity info.
        """
        node_id = self._get_entity_id(entity)

        if self.graph.graph.has_node(node_id):
            # Node exists - check if it's a Note we're enriching
            node_data = self.graph.graph.nodes[node_id]
            is_note = node_data.get("type") == "Note"

            if is_note:
                # Enrich Note with entity info (merge, not replace)
                existing_types = node_data.get("entity_types", [])
                if entity.type not in existing_types:
                    existing_types.append(entity.type)
                    self.graph.graph.nodes[node_id]["entity_types"] = existing_types

                existing_aliases = set(node_data.get("aliases", []))
                existing_aliases.update(entity.aliases)
                if existing_aliases:
                    self.graph.graph.nodes[node_id]["aliases"] = list(existing_aliases)

            return node_id, False, is_note

        # Add new node (no matching Note found)
        self.graph.graph.add_node(
            node_id,
            type=entity.type,
            name=entity.name,
            aliases=entity.aliases,
            source="extracted",
        )
        return node_id, True, False

    def _add_relationship(self, rel: Relationship) -> bool:
        """Add relationship between entities. Returns True if new edge added."""
        # Find source and target nodes
        source_norm = self._normalize_name(rel.source)
        target_norm = self._normalize_name(rel.target)

        source_id = self._entity_registry.get(source_norm)
        target_id = self._entity_registry.get(target_norm)

        if not source_id or not target_id:
            return False

        # Check if edge exists
        if self.graph.graph.has_edge(source_id, target_id):
            return False

        # Add edge
        self.graph.graph.add_edge(
            source_id,
            target_id,
            type=rel.relation,
            confidence=rel.confidence,
            source="extracted",
        )
        return True

    def _link_note_to_entities(self, note_path: str, entities: list[Entity]) -> int:
        """Link a note to its extracted entities. Returns count of links added."""
        if not self.graph.graph.has_node(note_path):
            return 0

        links_added = 0
        for entity in entities:
            entity_id = self._entity_registry.get(self._normalize_name(entity.name))
            if not entity_id:
                continue

            if not self.graph.graph.has_edge(note_path, entity_id):
                self.graph.graph.add_edge(
                    note_path,
                    entity_id,
                    type="mentions",
                    confidence=0.8,
                    source="extracted",
                )
                links_added += 1

        return links_added

    def merge_extraction(self, extraction: ExtractedNote, stats: MergeStats) -> None:
        """Merge a single extraction into the graph."""
        stats.extractions_processed += 1

        # Add all entities first
        for entity in extraction.entities:
            _, is_new, resolved_to_note = self._add_entity(entity)
            if is_new:
                stats.entities_added += 1
            elif resolved_to_note:
                stats.entities_resolved_to_notes += 1
            else:
                stats.entities_existing += 1

        # Add relationships
        for rel in extraction.relationships:
            if self._add_relationship(rel):
                stats.relationships_added += 1

        # Link note to entities if path available
        if extraction.has_path and extraction.note_path:
            stats.note_links_added += self._link_note_to_entities(
                extraction.note_path, extraction.entities
            )

    def merge_all(self, extractions: list[ExtractedNote]) -> MergeStats:
        """Merge all extractions into the graph."""
        stats = MergeStats()

        for extraction in extractions:
            self.merge_extraction(extraction, stats)

        return stats

    @classmethod
    def from_files(
        cls,
        graph_path: Path,
        extractions_path: Path,
    ) -> tuple[VaultGraph, MergeStats]:
        """Load graph, merge extractions, return updated graph and stats."""
        graph = VaultGraph.load(graph_path)
        reader = ExtractionReader(extractions_path)
        extractions = reader.read_all()

        merger = cls(graph)
        stats = merger.merge_all(extractions)

        return graph, stats
