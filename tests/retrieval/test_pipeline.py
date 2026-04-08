from src.retrieval.pipeline import recall_structured


class _Embedder:
    def embed(self, text: str) -> list[float]:
        return [0.1] * 384


class _Storage:
    def search_similar(
        self,
        query_embedding: list[float],
        node_types: list[str] | None = None,
        limit: int = 10,
    ) -> list[dict]:
        return [
            {
                "node_id": "goal:rl",
                "node_type": "Goal",
                "name": "RL Agent",
                "summary": "Build deterministic recall",
                "score": 0.92,
            },
            {
                "node_id": "note:memory",
                "node_type": "Note",
                "name": "Memory Note",
                "summary": "retrieval notes",
                "score": 0.75,
            },
        ]

    def find_nodes(
        self,
        name: str,
        node_type: str | None = None,
        match_type: str = "contains",
    ) -> list[dict]:
        return [
            {
                "id": "goal:rl",
                "name": "RL Agent",
                "summary": "Build deterministic recall",
                "_labels": ["Goal"],
            },
            {
                "id": "note:memory",
                "name": "Memory Note",
                "summary": "retrieval notes",
                "_labels": ["Note"],
            },
        ]

    def get_neighbors(
        self,
        node_id: str,
        direction: str = "both",
        edge_types: list[str] | None = None,
        depth: int = 1,
    ) -> list[dict]:
        if node_id == "goal:rl":
            return [
                {
                    "node": {
                        "id": "note:memory",
                        "name": "Memory Note",
                        "_labels": ["Note"],
                    },
                    "relation": "related_to",
                    "direction": "out",
                }
            ]
        return []

    def close(self) -> None:
        return None


class _VectorFailStorage(_Storage):
    def search_similar(
        self,
        query_embedding: list[float],
        node_types: list[str] | None = None,
        limit: int = 10,
    ) -> list[dict]:
        raise RuntimeError("vector index unavailable")


def test_recall_structured_is_deterministic_for_same_input():
    storage = _Storage()
    embedder = _Embedder()

    first = recall_structured(
        "rl recall",
        storage=storage,
        embedder=embedder,
        depth=1,
        limit=5,
    )
    second = recall_structured(
        "rl recall",
        storage=storage,
        embedder=embedder,
        depth=1,
        limit=5,
    )

    assert first["markdown"] == second["markdown"]
    assert first["retrieval"]["vector_hits"] == 2
    assert first["retrieval"]["keyword_hits"] == 2


def test_recall_structured_falls_back_when_vector_search_fails():
    result = recall_structured(
        "rl recall",
        storage=_VectorFailStorage(),
        embedder=_Embedder(),
        depth=1,
        limit=5,
    )

    assert result["success"] is True
    assert result["retrieval"]["error_stage"] == "vector_search"
    assert "vector_search_failed" in result["warnings"]
    assert result["retrieval"]["keyword_hits"] == 2
