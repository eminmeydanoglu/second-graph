"""JSONL extraction reader."""

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator


@dataclass
class Entity:
    """Extracted entity from a note."""

    name: str
    type: str
    aliases: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict) -> "Entity":
        return cls(
            name=d.get("name", ""),
            type=d.get("type", "Concept"),
            aliases=d.get("aliases", []),
        )


@dataclass
class Relationship:
    """Extracted relationship between entities."""

    source: str
    relation: str
    target: str
    confidence: float = 0.8

    @classmethod
    def from_dict(cls, d: dict) -> "Relationship":
        conf_map = {"high": 0.9, "medium": 0.7, "low": 0.5}
        conf = d.get("confidence", "medium")
        if isinstance(conf, str):
            conf = conf_map.get(conf, 0.7)
        return cls(
            source=d.get("source", ""),
            relation=d.get("relation", "related_to"),
            target=d.get("target", ""),
            confidence=conf,
        )


@dataclass
class ExtractedNote:
    """Extraction result for a single note."""

    note_path: str | None
    entities: list[Entity]
    relationships: list[Relationship]
    extracted_at: str | None = None

    @classmethod
    def from_dict(cls, d: dict) -> "ExtractedNote":
        # Handle different path field names
        note_path = d.get("note_path") or d.get("source_file")

        return cls(
            note_path=note_path,
            entities=[Entity.from_dict(e) for e in d.get("entities", [])],
            relationships=[
                Relationship.from_dict(r) for r in d.get("relationships", [])
            ],
            extracted_at=d.get("extracted_at"),
        )

    @property
    def has_path(self) -> bool:
        return self.note_path is not None and len(self.note_path) > 0


class ExtractionReader:
    """Reader for extraction JSONL files."""

    def __init__(self, path: Path):
        self.path = path

    def __iter__(self) -> Iterator[ExtractedNote]:
        """Iterate over all extractions."""
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    yield ExtractedNote.from_dict(d)
                except json.JSONDecodeError:
                    continue

    def read_all(self) -> list[ExtractedNote]:
        """Read all extractions into memory."""
        return list(self)

    def read_with_paths(self) -> list[ExtractedNote]:
        """Read only extractions that have note paths."""
        return [e for e in self if e.has_path]

    def get_stats(self) -> dict:
        """Get statistics about extractions."""
        all_extractions = self.read_all()

        entity_types = Counter(
            ent.type for ext in all_extractions for ent in ext.entities
        )
        relation_types = Counter(
            rel.relation for ext in all_extractions for rel in ext.relationships
        )

        return {
            "total_extractions": len(all_extractions),
            "with_path": sum(1 for e in all_extractions if e.has_path),
            "without_path": sum(1 for e in all_extractions if not e.has_path),
            "total_entities": sum(len(e.entities) for e in all_extractions),
            "total_relationships": sum(len(e.relationships) for e in all_extractions),
            "entity_types": dict(entity_types),
            "relation_types": dict(relation_types),
        }
