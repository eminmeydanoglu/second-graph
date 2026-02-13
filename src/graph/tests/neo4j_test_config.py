"""Shared Neo4j test configuration.

Uses a dedicated test Neo4j instance by default to avoid touching
the main local database.
"""

from __future__ import annotations

import os


def get_test_neo4j_config() -> tuple[str, str, str]:
    uri = os.getenv("TEST_NEO4J_URI", "bolt://localhost:17687")
    user = os.getenv("TEST_NEO4J_USER", "neo4j")
    password = os.getenv("TEST_NEO4J_PASSWORD", "obsidian")
    return uri, user, password


def guard_test_uri(uri: str) -> None:
    if uri.strip() == "bolt://localhost:7687":
        raise RuntimeError(
            "Refusing to run integration tests against main Neo4j (7687). "
            "Use TEST_NEO4J_URI=bolt://localhost:17687"
        )
