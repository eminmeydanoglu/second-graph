"""Deterministic recall pipeline orchestration."""

from dataclasses import asdict
import os
from typing import Protocol

from ..graph.neo4j_storage import Neo4jStorage
from ..vector.embedder import Embedder
from .config import DEFAULT_RETRIEVAL_CONFIG, RetrievalConfig
from .ranker import apply_graph_locality, build_scored_candidates, select_anchors
from .renderer import render_context_with_meta
from .tokenize import normalize_whitespace
from .traversal import expand_from_anchors_with_hops
from .types import TraversalResult


class RecallStorage(Protocol):
    def search_similar(
        self,
        query_embedding: list[float],
        node_types: list[str] | None = None,
        limit: int = 10,
    ) -> list[dict]: ...

    def find_nodes(
        self,
        name: str,
        node_type: str | None = None,
        match_type: str = "contains",
    ) -> list[dict]: ...

    def get_neighbors(
        self,
        node_id: str,
        direction: str = "both",
        edge_types: list[str] | None = None,
        depth: int = 1,
    ) -> list[dict]: ...

    def close(self) -> None: ...


class QueryEmbedder(Protocol):
    def embed(self, text: str) -> list[float]: ...


def _empty_traversal() -> TraversalResult:
    return TraversalResult(
        connections=tuple(), notes=tuple(), hop_by_node={}, node_lookup={}
    )


def _default_storage() -> Neo4jStorage:
    return Neo4jStorage(
        uri=os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
        user=os.environ.get("NEO4J_USER", "neo4j"),
        password=os.environ.get("NEO4J_PASSWORD", "obsidian"),
    )


def _default_embedder() -> Embedder:
    return Embedder()


def recall_structured(
    query: str,
    *,
    mode: str = "pull",
    depth: int = 1,
    limit: int = 10,
    storage: RecallStorage | None = None,
    embedder: QueryEmbedder | None = None,
    config: RetrievalConfig = DEFAULT_RETRIEVAL_CONFIG,
) -> dict:
    """Run deterministic recall and return structured payload."""
    normalized_query = normalize_whitespace(query)
    bounded_limit = max(0, limit)
    bounded_depth = config.clamp_depth(depth)
    max_tokens = config.max_tokens_for_mode(mode)

    owned_storage = storage is None
    owned_embedder = embedder is None

    db = storage or _default_storage()
    emb = embedder or _default_embedder()

    vector_hits: list[dict] = []
    lexical_hits: list[dict] = []
    anchors = []
    traversal = _empty_traversal()
    warnings: list[str] = []
    error_stage: str | None = None

    try:
        if normalized_query:
            try:
                query_embedding = emb.embed(normalized_query)
                vector_hits = db.search_similar(
                    query_embedding, limit=config.vector_limit
                )
            except Exception:
                warnings.append("vector_search_failed")
                error_stage = error_stage or "vector_search"

            try:
                lexical_hits = db.find_nodes(normalized_query, match_type="contains")
            except Exception:
                warnings.append("lexical_search_failed")
                error_stage = error_stage or "lexical_search"

        ranked_nodes = build_scored_candidates(
            normalized_query,
            vector_hits=vector_hits,
            lexical_hits=lexical_hits,
            config=config,
        )

        anchors = select_anchors(ranked_nodes, anchor_count=config.anchor_count)

        if anchors:
            try:
                traversal = expand_from_anchors_with_hops(
                    anchors,
                    storage=db,
                    depth=bounded_depth,
                    neighbor_cap=config.neighbor_cap,
                    config=config,
                )
            except Exception:
                warnings.append("graph_traversal_failed")
                error_stage = error_stage or "graph_traversal"

        ranked_with_graph = apply_graph_locality(
            ranked_nodes,
            hop_by_node=traversal.hop_by_node,
            config=config,
        )

        matched_nodes = ranked_with_graph[:bounded_limit]
        markdown, truncated_sections = render_context_with_meta(
            matched_nodes,
            list(traversal.connections),
            list(traversal.notes),
            max_tokens=max_tokens,
            config=config,
        )

        return {
            "success": True,
            "query": normalized_query,
            "mode": mode,
            "depth": bounded_depth,
            "limit": bounded_limit,
            "max_tokens": max_tokens,
            "retrieval": {
                "mode": mode,
                "vector_hits": len(vector_hits),
                "keyword_hits": len(lexical_hits),
                "anchor_count": len(anchors),
                "connection_count": len(traversal.connections),
                "note_count": len(traversal.notes),
                "truncated_sections": truncated_sections,
                "error_stage": error_stage,
            },
            "warnings": warnings,
            "matched_nodes": [asdict(node) for node in matched_nodes],
            "connections": [asdict(edge) for edge in traversal.connections],
            "related_notes": [asdict(note) for note in traversal.notes],
            "markdown": markdown,
        }
    finally:
        if owned_storage:
            db.close()
        if owned_embedder:
            # Embedder has no close hook; explicit delete drops model ref.
            del emb


def recall_markdown(
    query: str,
    *,
    mode: str = "pull",
    depth: int = 1,
    limit: int = 10,
    storage: RecallStorage | None = None,
    embedder: QueryEmbedder | None = None,
    config: RetrievalConfig = DEFAULT_RETRIEVAL_CONFIG,
) -> str:
    """Run deterministic recall and return markdown-only payload."""
    return recall_structured(
        query,
        mode=mode,
        depth=depth,
        limit=limit,
        storage=storage,
        embedder=embedder,
        config=config,
    )["markdown"]
