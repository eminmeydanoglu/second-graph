"""Deterministic retrieval and rendering pipeline."""

from typing import Any

__all__ = ["recall_markdown", "recall_structured"]


def recall_structured(*args: Any, **kwargs: Any) -> dict:
    from .pipeline import recall_structured as _recall_structured

    return _recall_structured(*args, **kwargs)


def recall_markdown(*args: Any, **kwargs: Any) -> str:
    from .pipeline import recall_markdown as _recall_markdown

    return _recall_markdown(*args, **kwargs)
