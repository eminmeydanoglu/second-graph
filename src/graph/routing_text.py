"""Routing text builder for semantic entity embeddings."""

from typing import Any

MAX_ROUTING_TEXT_CHARS = 500


def _string(value: Any) -> str:
    """Normalize any value to a stripped string."""
    if value is None:
        return ""
    return str(value).strip()


def _normalize_tags(tags: Any) -> list[str]:
    """Normalize tags into a list of cleaned strings."""
    if tags is None:
        return []
    if isinstance(tags, str):
        cleaned = tags.strip().lstrip("#")
        return [cleaned] if cleaned else []
    if not isinstance(tags, list):
        return []

    normalized: list[str] = []
    for tag in tags:
        cleaned = _string(tag).lstrip("#")
        if cleaned:
            normalized.append(cleaned)
    return normalized


def _compose(parts: list[str]) -> str:
    """Join parts and keep result within routing-text size limit."""
    filtered = [part.strip() for part in parts if part and part.strip()]
    if not filtered:
        text = ""
    else:
        text = filtered[0]
        for part in filtered[1:]:
            if text.endswith((".", "!", "?", ":")):
                text = f"{text} {part}"
            else:
                text = f"{text}. {part}"

    if len(text) <= MAX_ROUTING_TEXT_CHARS:
        return text
    return text[: MAX_ROUTING_TEXT_CHARS - 3].rstrip() + "..."


def build_routing_text(node_type: str, props: dict[str, Any] | None) -> str:
    """Build short, type-aware text used for semantic routing embeddings."""
    data = props or {}

    canonical_type = _string(node_type) or "Unknown"
    name = _string(data.get("name"))
    summary = _string(data.get("summary"))
    tags = _normalize_tags(data.get("tags"))

    if canonical_type == "Note":
        title = _string(data.get("title")) or name
        parts = [f"Note: {title}"]
        if summary:
            parts.append(summary)
        if tags:
            parts.append(f"Tags: {', '.join(tags)}")
        return _compose(parts)

    if canonical_type == "Person":
        parts = [f"Person: {name}"]
        if summary:
            parts.append(summary)
        role = _string(data.get("role"))
        if role:
            parts.append(f"Role: {role}")
        relationship = _string(data.get("relationship"))
        if relationship:
            parts.append(f"Relationship: {relationship}")
        return _compose(parts)

    if canonical_type == "Concept":
        parts = [f"Concept: {name}"]
        if summary:
            parts.append(summary)
        domain = _string(data.get("domain"))
        if domain:
            parts.append(f"Domain: {domain}")
        return _compose(parts)

    if canonical_type == "Goal":
        parts = [f"Goal: {name}"]
        if summary:
            parts.append(summary)
        status = _string(data.get("status"))
        if status:
            parts.append(f"Status: {status}")
        horizon = _string(data.get("horizon"))
        if horizon:
            parts.append(f"Horizon: {horizon}")
        return _compose(parts)

    if canonical_type == "Project":
        parts = [f"Project: {name}"]
        if summary:
            parts.append(summary)
        status = _string(data.get("status"))
        if status:
            parts.append(f"Status: {status}")
        return _compose(parts)

    if canonical_type == "Belief":
        parts = [f"Belief: {name}"]
        if summary:
            parts.append(summary)
        return _compose(parts)

    if canonical_type == "Value":
        parts = [f"Value: {name}"]
        priority = _string(data.get("priority"))
        if priority:
            parts.append(f"Priority: {priority}")
        return _compose(parts)

    if canonical_type == "Source":
        parts = [f"Source: {name}"]
        if summary:
            parts.append(summary)
        author = _string(data.get("author"))
        if author:
            parts.append(f"Author: {author}")
        source_type = _string(data.get("type"))
        if source_type:
            parts.append(f"Type: {source_type}")
        return _compose(parts)

    if canonical_type == "Fear":
        parts = [f"Fear: {name}"]
        intensity = _string(data.get("intensity"))
        if intensity:
            parts.append(f"Intensity: {intensity}")
        return _compose(parts)

    if canonical_type == "Tool":
        parts = [f"Tool: {name}"]
        if summary:
            parts.append(summary)
        return _compose(parts)

    if canonical_type == "Organization":
        parts = [f"Organization: {name}"]
        if summary:
            parts.append(summary)
        return _compose(parts)

    return _compose([f"{canonical_type}: {name}"])
