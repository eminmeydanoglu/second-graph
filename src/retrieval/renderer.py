"""Deterministic markdown renderer for recall output."""

import math

from .config import DEFAULT_RETRIEVAL_CONFIG, RetrievalConfig
from .tokenize import normalize_whitespace
from .types import RankedConnection, RelatedNote, ScoredNode

_MAX_NAME_LEN = 120
_MAX_NOTE_NAME_LEN = 120
_MAX_SUMMARY_LEN = 180


def estimate_tokens(text: str) -> int:
    """Estimate tokens with 10% safety margin."""
    if not text:
        return 0
    return int(math.ceil((len(text) / 4.0) * 1.10))


def _sanitize_field(value: str | None, *, max_len: int) -> str:
    cleaned = normalize_whitespace(value)
    if len(cleaned) <= max_len:
        return cleaned
    return cleaned[: max_len - 3].rstrip() + "..."


def _format_scored_node(node: ScoredNode) -> str:
    node_id = _sanitize_field(node.node_id, max_len=160)
    node_type = _sanitize_field(node.node_type, max_len=60)
    name = _sanitize_field(node.name, max_len=_MAX_NAME_LEN)
    return (
        f"- [{node_id}] {node_type} | {name} | "
        f"score={node.final_score:.4f} | vector={node.vector_score:.4f} | keyword={node.keyword_score:.4f}"
    )


def _format_connection(connection: RankedConnection) -> str:
    from_id = _sanitize_field(connection.from_id, max_len=160)
    to_id = _sanitize_field(connection.to_id, max_len=160)
    from_name = _sanitize_field(connection.from_name, max_len=_MAX_NAME_LEN)
    to_name = _sanitize_field(connection.to_name, max_len=_MAX_NAME_LEN)
    relation = _sanitize_field(connection.relation.upper(), max_len=80)

    if connection.direction == "in":
        arrow = f"<-[{relation}]-"
    else:
        arrow = f"-[{relation}]->"

    return f"- {from_id} ({from_name}) {arrow} {to_id} ({to_name}) | hop={connection.support_hop}"


def _format_note(note: RelatedNote) -> str:
    note_id = _sanitize_field(note.note_id, max_len=160)
    name = _sanitize_field(note.name, max_len=_MAX_NOTE_NAME_LEN)
    reason = _sanitize_field(note.reason, max_len=32)
    return f"- {note_id} | {name} | reason={reason} | score={note.score:.4f}"


def _truncate_section(lines: list[str], token_budget: int) -> tuple[list[str], int]:
    if not lines:
        return ["- (none)"], 0

    if token_budget <= 0:
        return [f"- ...(truncated {len(lines)})"], len(lines)

    kept: list[str] = []
    used = 0
    index = 0

    while index < len(lines):
        line = lines[index]
        line_tokens = estimate_tokens(line + "\n")
        if used + line_tokens > token_budget:
            break
        kept.append(line)
        used += line_tokens
        index += 1

    truncated = len(lines) - len(kept)
    if truncated <= 0:
        return kept, 0

    marker = f"- ...(truncated {truncated})"
    marker_tokens = estimate_tokens(marker + "\n")
    while kept and used + marker_tokens > token_budget:
        kept.pop()
        truncated += 1
        marker = f"- ...(truncated {truncated})"
        marker_tokens = estimate_tokens(marker + "\n")

    if not kept and marker_tokens > token_budget:
        return [marker], truncated

    kept.append(marker)
    return kept, truncated


def render_context_with_meta(
    nodes: list[ScoredNode],
    edges: list[RankedConnection],
    notes: list[RelatedNote],
    *,
    max_tokens: int,
    config: RetrievalConfig = DEFAULT_RETRIEVAL_CONFIG,
) -> tuple[str, list[str]]:
    """Render markdown and list of truncated sections."""
    bounded_tokens = max(1, max_tokens)

    node_lines = [_format_scored_node(node) for node in nodes]
    edge_lines = [_format_connection(edge) for edge in edges]
    note_lines = [_format_note(note) for note in notes]

    nodes_budget = int(bounded_tokens * config.nodes_budget_ratio)
    connections_budget = int(bounded_tokens * config.connections_budget_ratio)
    notes_budget = max(
        1,
        bounded_tokens - nodes_budget - connections_budget,
    )

    rendered_nodes, nodes_truncated = _truncate_section(node_lines, nodes_budget)
    rendered_edges, edges_truncated = _truncate_section(edge_lines, connections_budget)
    rendered_notes, notes_truncated = _truncate_section(note_lines, notes_budget)

    truncated_sections: list[str] = []
    if nodes_truncated:
        truncated_sections.append("matched_nodes")
    if edges_truncated:
        truncated_sections.append("connections")
    if notes_truncated:
        truncated_sections.append("related_notes")

    parts = [
        "## Matched Nodes",
        *rendered_nodes,
        "",
        "## Connections",
        *rendered_edges,
        "",
        "## Related Notes",
        *rendered_notes,
    ]
    return "\n".join(parts), truncated_sections


def render_context(
    nodes: list[ScoredNode],
    edges: list[RankedConnection],
    notes: list[RelatedNote],
    *,
    max_tokens: int,
    config: RetrievalConfig = DEFAULT_RETRIEVAL_CONFIG,
) -> str:
    """Render deterministic markdown with strict section contract."""
    markdown, _ = render_context_with_meta(
        nodes,
        edges,
        notes,
        max_tokens=max_tokens,
        config=config,
    )
    return markdown
