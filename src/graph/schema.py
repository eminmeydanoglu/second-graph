"""Graph schema definitions for the knowledge graph.

Defines valid node types, edge types, and constraints for the graph.
"""

from dataclasses import dataclass
from enum import Enum


class SourceType(Enum):
    """Provenance types for graph data."""

    FILE = "file"  # From physical .md files (wikilinks, tags, YAML)
    AGENT = "agent"  # Manual additions by agents during conversation
    EXTRACTION = "extraction"  # LLM-extracted semantic relations
    SYSTEM = "system"  # Initial setup or system-generated


def generate_source_id(source_type: SourceType, identifier: str = "") -> str:
    """Generate a canonical source ID.

    Format: {type}:{identifier} or just {type} for AGENT
    Examples:
        file:/vault/Projects/AI.md
        agent
        extraction:v1:/vault/Projects/AI.md
    """
    if source_type == SourceType.AGENT:
        return "agent"
    return f"{source_type.value}:{identifier}"


class NodeType(Enum):
    """Valid node types in the knowledge graph."""

    NOTE = "Note"
    TAG = "Tag"
    GOAL = "Goal"
    PROJECT = "Project"
    BELIEF = "Belief"
    VALUE = "Value"
    PERSON = "Person"
    CONCEPT = "Concept"
    SOURCE = "Source"
    FEAR = "Fear"
    FOLDER = "Folder"
    TOOL = "Tool"
    ORGANIZATION = "Organization"


class EdgeType(Enum):
    """Valid edge/relationship types in the knowledge graph."""

    WIKILINK = "wikilink"
    TAGGED_WITH = "tagged_with"

    CONTRIBUTES_TO = "CONTRIBUTES_TO"
    WORKS_ON = "WORKS_ON"
    MENTIONS = "MENTIONS"
    BELIEVES = "BELIEVES"
    VALUES = "VALUES"
    SUPPORTS = "SUPPORTS"
    CONTRADICTS = "CONTRADICTS"
    MOTIVATES = "MOTIVATES"
    HAS_GOAL = "HAS_GOAL"
    HAS_VALUE = "HAS_VALUE"
    HAS_BELIEF = "HAS_BELIEF"
    KNOWS = "KNOWS"
    INTERESTED_IN = "INTERESTED_IN"
    LEARNED_FROM = "LEARNED_FROM"
    RELATED_TO = "RELATED_TO"
    FEARS = "FEARS"
    AVOIDS = "AVOIDS"
    USES = "USES"
    CREATED_BY = "CREATED_BY"
    PART_OF = "PART_OF"


EDGE_CONSTRAINTS: dict[EdgeType, dict] = {
    EdgeType.CONTRIBUTES_TO: {
        "sources": [NodeType.PROJECT, NodeType.GOAL],
        "targets": [NodeType.GOAL],
    },
    EdgeType.MOTIVATES: {
        "sources": [NodeType.VALUE, NodeType.BELIEF],
        "targets": [NodeType.GOAL, NodeType.PROJECT],
    },
    EdgeType.HAS_GOAL: {
        "sources": [NodeType.PERSON],
        "targets": [NodeType.GOAL],
    },
    EdgeType.HAS_VALUE: {
        "sources": [NodeType.PERSON],
        "targets": [NodeType.VALUE],
    },
    EdgeType.HAS_BELIEF: {
        "sources": [NodeType.PERSON],
        "targets": [NodeType.BELIEF],
    },
    EdgeType.FEARS: {
        "sources": [NodeType.PERSON],
        "targets": [NodeType.FEAR, NodeType.CONCEPT],
    },
    EdgeType.AVOIDS: {
        "sources": [NodeType.PERSON],
        "targets": [NodeType.CONCEPT, NodeType.PROJECT],
    },
    EdgeType.WORKS_ON: {
        "sources": [NodeType.PERSON],
        "targets": [NodeType.PROJECT],
    },
    EdgeType.INTERESTED_IN: {
        "sources": [NodeType.PERSON],
        "targets": [NodeType.CONCEPT, NodeType.PROJECT, NodeType.TOOL],
    },
    EdgeType.USES: {
        "sources": [NodeType.PERSON, NodeType.PROJECT],
        "targets": [NodeType.TOOL],
    },
    EdgeType.LEARNED_FROM: {
        "sources": [NodeType.PERSON, NodeType.CONCEPT],
        "targets": [NodeType.SOURCE, NodeType.PERSON],
    },
    EdgeType.SUPPORTS: {
        "sources": [NodeType.BELIEF, NodeType.CONCEPT],
        "targets": [NodeType.BELIEF, NodeType.CONCEPT, NodeType.GOAL],
    },
    EdgeType.CONTRADICTS: {
        "sources": [NodeType.BELIEF, NodeType.CONCEPT],
        "targets": [NodeType.BELIEF, NodeType.CONCEPT],
    },
    EdgeType.RELATED_TO: {"sources": None, "targets": None},
    EdgeType.MENTIONS: {"sources": None, "targets": None},
    EdgeType.WIKILINK: {"sources": None, "targets": None},
    EdgeType.TAGGED_WITH: {"sources": None, "targets": [NodeType.TAG]},
}


@dataclass
class ValidationResult:
    """Result of a validation check."""

    valid: bool
    errors: list[str]
    warnings: list[str]

    def __bool__(self) -> bool:
        return self.valid


def validate_node_type(node_type: str) -> bool:
    """Check if a node type is valid."""
    return node_type in [t.value for t in NodeType]


def validate_edge_type(edge_type: str) -> bool:
    """Check if an edge type is valid."""
    return edge_type in [t.value for t in EdgeType]


def validate_edge(
    from_type: str, to_type: str, relation: str, strict: bool = False
) -> ValidationResult:
    """Validate an edge against schema constraints.

    Args:
        from_type: Source node type
        to_type: Target node type
        relation: Relationship type
        strict: If True, reject unknown types. If False, allow with warnings.

    Returns:
        ValidationResult with valid status and any errors/warnings
    """
    errors = []
    warnings = []

    if not validate_node_type(from_type):
        msg = f"Unknown source node type: {from_type}"
        if strict:
            errors.append(msg)
        else:
            warnings.append(msg)

    if not validate_node_type(to_type):
        msg = f"Unknown target node type: {to_type}"
        if strict:
            errors.append(msg)
        else:
            warnings.append(msg)

    if not validate_edge_type(relation):
        msg = f"Unknown edge type: {relation}"
        if strict:
            errors.append(msg)
        else:
            warnings.append(msg)
        return ValidationResult(
            valid=len(errors) == 0, errors=errors, warnings=warnings
        )

    try:
        edge_type = EdgeType(relation)
        constraints = EDGE_CONSTRAINTS.get(edge_type)

        if constraints:
            valid_sources = constraints.get("sources")
            valid_targets = constraints.get("targets")

            if valid_sources is not None:
                try:
                    source_type = NodeType(from_type)
                    if source_type not in valid_sources:
                        msg = f"Invalid source type {from_type} for edge {relation}"
                        if strict:
                            errors.append(msg)
                        else:
                            warnings.append(msg)
                except ValueError:
                    pass  # Unknown type, already warned above

            if valid_targets is not None:
                try:
                    target_type = NodeType(to_type)
                    if target_type not in valid_targets:
                        msg = f"Invalid target type {to_type} for edge {relation}"
                        if strict:
                            errors.append(msg)
                        else:
                            warnings.append(msg)
                except ValueError:
                    pass  # Unknown type, already warned above

    except ValueError:
        pass  # Unknown edge type, already warned above

    return ValidationResult(valid=len(errors) == 0, errors=errors, warnings=warnings)


def generate_node_id(node_type: str, name: str) -> str:
    """Generate a canonical node ID from type and name.

    Format: type:normalized_name
    Example: goal:build_autonomous_agents

    Args:
        node_type: The node type (will be lowercased)
        name: The human-readable name

    Returns:
        Canonical node ID
    """
    normalized = name.lower().strip()
    normalized = normalized.replace(" ", "_").replace("-", "_")
    while "__" in normalized:
        normalized = normalized.replace("__", "_")
    normalized = normalized.strip("_")

    return f"{node_type.lower()}:{normalized}"


def get_node_types() -> list[str]:
    """Get list of all valid node types."""
    return [t.value for t in NodeType]


def get_edge_types() -> list[str]:
    """Get list of all valid edge types."""
    return [t.value for t in EdgeType]
