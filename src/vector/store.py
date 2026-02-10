"""SQLite-vec vector storage for notes and entities."""

import sqlite3
import struct
from pathlib import Path
from typing import Any

import sqlite_vec


def serialize_vector(vec: list[float]) -> bytes:
    """Serialize a vector to bytes for sqlite-vec."""
    return struct.pack(f"{len(vec)}f", *vec)


class VectorStore:
    """SQLite-vec based vector storage for notes and entities."""

    def __init__(self, db_path: Path | str, dimensions: int = 384):
        self.db_path = Path(db_path)
        self.dimensions = dimensions
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Initialize the database schema."""
        conn = sqlite3.connect(self.db_path)
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)

        # Notes table (existing)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS notes (
                id INTEGER PRIMARY KEY,
                path TEXT UNIQUE NOT NULL,
                title TEXT,
                content TEXT,
                chunk_index INTEGER DEFAULT 0
            )
        """)

        # Notes vector table (existing)
        conn.execute(f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS note_vectors USING vec0(
                embedding float[{self.dimensions}]
            )
        """)

        # Entities table (NEW)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS entities (
                id INTEGER PRIMARY KEY,
                node_id TEXT UNIQUE NOT NULL,
                node_type TEXT NOT NULL,
                name TEXT NOT NULL,
                summary TEXT
            )
        """)

        # Create index on node_id
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_entities_node_id ON entities(node_id)
        """)

        # Entity vectors table (NEW)
        conn.execute(f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS entity_vectors USING vec0(
                embedding float[{self.dimensions}]
            )
        """)

        # Extraction tracking: stores content snapshot at extraction time
        conn.execute("""
            CREATE TABLE IF NOT EXISTS note_extractions (
                id INTEGER PRIMARY KEY,
                path TEXT UNIQUE NOT NULL,
                content_hash TEXT NOT NULL,
                extracted_at TEXT NOT NULL,
                content TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_extractions_path ON note_extractions(path)
        """)

        # Extraction diffs: audit trail of what changed between extractions
        conn.execute("""
            CREATE TABLE IF NOT EXISTS extraction_diffs (
                id INTEGER PRIMARY KEY,
                path TEXT NOT NULL,
                extracted_at TEXT NOT NULL,
                old_hash TEXT,
                new_hash TEXT NOT NULL,
                diff TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'new'
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_diffs_path ON extraction_diffs(path)
        """)

        conn.commit()
        conn.close()

    def _get_conn(self) -> sqlite3.Connection:
        """Get a connection with sqlite-vec loaded."""
        conn = sqlite3.connect(self.db_path)
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        return conn

    # =========================================================================
    # NOTE OPERATIONS (existing)
    # =========================================================================

    def add(
        self,
        path: str,
        title: str,
        content: str,
        embedding: list[float],
        chunk_index: int = 0,
    ) -> int:
        """Add a note with its embedding.

        Returns:
            The row ID of the inserted note
        """
        conn = self._get_conn()
        cursor = conn.cursor()

        # Insert metadata
        cursor.execute(
            "INSERT OR REPLACE INTO notes (path, title, content, chunk_index) VALUES (?, ?, ?, ?)",
            (path, title, content, chunk_index),
        )
        row_id = cursor.lastrowid
        if row_id is None:
            conn.close()
            raise RuntimeError("Failed to insert note")

        # Insert vector (rowid must match)
        cursor.execute(
            "INSERT OR REPLACE INTO note_vectors (rowid, embedding) VALUES (?, ?)",
            (row_id, serialize_vector(embedding)),
        )

        conn.commit()
        conn.close()
        return row_id

    def search(
        self, query_embedding: list[float], limit: int = 10
    ) -> list[dict[str, Any]]:
        """Search for similar notes.

        Args:
            query_embedding: The query vector
            limit: Maximum number of results

        Returns:
            List of dicts with path, title, content, distance
        """
        conn = self._get_conn()
        cursor = conn.cursor()

        # Vector similarity search
        cursor.execute(
            """
            SELECT 
                n.path,
                n.title,
                n.content,
                n.chunk_index,
                v.distance
            FROM note_vectors v
            JOIN notes n ON n.id = v.rowid
            WHERE v.embedding MATCH ? AND k = ?
            ORDER BY v.distance
        """,
            (serialize_vector(query_embedding), limit),
        )

        results = []
        for row in cursor.fetchall():
            results.append(
                {
                    "path": row[0],
                    "title": row[1],
                    "content": row[2][:500] + "..." if len(row[2]) > 500 else row[2],
                    "chunk_index": row[3],
                    "distance": row[4],
                    "score": 1 - row[4],  # Convert distance to similarity score
                }
            )

        conn.close()
        return results

    def count(self) -> int:
        """Get the number of stored note vectors."""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM notes")
        count = cursor.fetchone()[0]
        conn.close()
        return count

    def clear(self):
        """Clear all note data."""
        conn = self._get_conn()
        conn.execute("DELETE FROM notes")
        conn.execute("DELETE FROM note_vectors")
        conn.commit()
        conn.close()

    # =========================================================================
    # ENTITY OPERATIONS (NEW)
    # =========================================================================

    def add_entity(
        self,
        node_id: str,
        node_type: str,
        name: str,
        summary: str | None,
        embedding: list[float],
    ) -> int:
        """Add or update an entity with its embedding.

        Args:
            node_id: Unique node identifier (e.g., "goal:build_ai")
            node_type: Node type (Goal, Person, etc.)
            name: Human-readable name
            summary: Entity description
            embedding: Vector embedding

        Returns:
            The row ID of the inserted entity
        """
        conn = self._get_conn()
        cursor = conn.cursor()

        # Check if entity exists
        cursor.execute("SELECT id FROM entities WHERE node_id = ?", (node_id,))
        existing = cursor.fetchone()

        if existing:
            # Update existing
            row_id = existing[0]
            cursor.execute(
                "UPDATE entities SET node_type = ?, name = ?, summary = ? WHERE id = ?",
                (node_type, name, summary, row_id),
            )
            # Update vector
            cursor.execute(
                "DELETE FROM entity_vectors WHERE rowid = ?",
                (row_id,),
            )
            cursor.execute(
                "INSERT INTO entity_vectors (rowid, embedding) VALUES (?, ?)",
                (row_id, serialize_vector(embedding)),
            )
        else:
            # Insert new
            cursor.execute(
                "INSERT INTO entities (node_id, node_type, name, summary) VALUES (?, ?, ?, ?)",
                (node_id, node_type, name, summary),
            )
            row_id = cursor.lastrowid
            if row_id is None:
                conn.close()
                raise RuntimeError("Failed to insert entity")
            cursor.execute(
                "INSERT INTO entity_vectors (rowid, embedding) VALUES (?, ?)",
                (row_id, serialize_vector(embedding)),
            )

        conn.commit()
        conn.close()
        return row_id

    def search_entities(
        self,
        query_embedding: list[float],
        node_types: list[str] | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Search for similar entities using vector similarity.

        Args:
            query_embedding: The query vector
            node_types: Optional list of node types to filter
            limit: Maximum number of results

        Returns:
            List of entity dicts with scores
        """
        conn = self._get_conn()
        cursor = conn.cursor()

        # Build query with optional type filter
        if node_types:
            placeholders = ",".join("?" * len(node_types))
            query = f"""
                SELECT 
                    e.node_id, e.node_type, e.name, e.summary, v.distance
                FROM entity_vectors v
                JOIN entities e ON e.id = v.rowid
                WHERE v.embedding MATCH ? AND k = ?
                  AND e.node_type IN ({placeholders})
                ORDER BY v.distance
            """
            params: tuple = (serialize_vector(query_embedding), limit, *node_types)
        else:
            query = """
                SELECT 
                    e.node_id, e.node_type, e.name, e.summary, v.distance
                FROM entity_vectors v
                JOIN entities e ON e.id = v.rowid
                WHERE v.embedding MATCH ? AND k = ?
                ORDER BY v.distance
            """
            params = (serialize_vector(query_embedding), limit)

        cursor.execute(query, params)

        results = []
        for row in cursor.fetchall():
            results.append(
                {
                    "node_id": row[0],
                    "node_type": row[1],
                    "name": row[2],
                    "summary": row[3],
                    "distance": row[4],
                    "score": 1 - row[4],  # Convert distance to similarity
                }
            )

        conn.close()
        return results

    def get_entity(self, node_id: str) -> dict | None:
        """Get an entity by node_id.

        Args:
            node_id: The entity's unique identifier

        Returns:
            Entity dict or None if not found
        """
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT node_id, node_type, name, summary FROM entities WHERE node_id = ?",
            (node_id,),
        )
        row = cursor.fetchone()
        conn.close()

        if not row:
            return None

        return {
            "node_id": row[0],
            "node_type": row[1],
            "name": row[2],
            "summary": row[3],
        }

    def delete_entity(self, node_id: str) -> bool:
        """Delete an entity by node_id.

        Args:
            node_id: The entity's unique identifier

        Returns:
            True if deleted, False if not found
        """
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM entities WHERE node_id = ?", (node_id,))
        row = cursor.fetchone()

        if row:
            entity_id = row[0]
            cursor.execute("DELETE FROM entities WHERE id = ?", (entity_id,))
            cursor.execute("DELETE FROM entity_vectors WHERE rowid = ?", (entity_id,))
            conn.commit()
            conn.close()
            return True

        conn.close()
        return False

    def entity_count(self) -> int:
        """Get the number of stored entities."""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM entities")
        count = cursor.fetchone()[0]
        conn.close()
        return count

    def clear_entities(self):
        """Clear all entity data."""
        conn = self._get_conn()
        conn.execute("DELETE FROM entities")
        conn.execute("DELETE FROM entity_vectors")
        conn.commit()
        conn.close()

    # =========================================================================
    # EXTRACTION TRACKING OPERATIONS
    # =========================================================================

    def get_extraction_status(self, path: str) -> dict | None:
        """Get stored extraction status for a note path.

        Returns:
            Dict with content_hash, extracted_at, or None if never extracted.
        """
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT content_hash, extracted_at FROM note_extractions WHERE path = ?",
            (path,),
        )
        row = cursor.fetchone()
        conn.close()
        if not row:
            return None
        return {"content_hash": row[0], "extracted_at": row[1]}

    def get_note_snapshot(self, path: str) -> str | None:
        """Get the stored content snapshot from last extraction."""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT content FROM note_extractions WHERE path = ?",
            (path,),
        )
        row = cursor.fetchone()
        conn.close()
        return row[0] if row else None

    def upsert_extraction(
        self, path: str, content_hash: str, extracted_at: str, content: str
    ) -> None:
        """Insert or update extraction record for a note."""
        conn = self._get_conn()
        conn.execute(
            """
            INSERT INTO note_extractions (path, content_hash, extracted_at, content)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(path) DO UPDATE SET
                content_hash = excluded.content_hash,
                extracted_at = excluded.extracted_at,
                content = excluded.content
            """,
            (path, content_hash, extracted_at, content),
        )
        conn.commit()
        conn.close()

    def store_extraction_diff(
        self,
        path: str,
        extracted_at: str,
        old_hash: str | None,
        new_hash: str,
        diff: str,
        status: str,
    ) -> None:
        """Store an extraction diff record for audit trail."""
        conn = self._get_conn()
        conn.execute(
            """
            INSERT INTO extraction_diffs (path, extracted_at, old_hash, new_hash, diff, status)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (path, extracted_at, old_hash, new_hash, diff, status),
        )
        conn.commit()
        conn.close()

    def get_extraction_diffs(self, path: str, limit: int = 10) -> list[dict]:
        """Get extraction diff history for a note."""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT extracted_at, old_hash, new_hash, diff, status
            FROM extraction_diffs
            WHERE path = ?
            ORDER BY extracted_at DESC
            LIMIT ?
            """,
            (path, limit),
        )
        results = [
            {
                "extracted_at": row[0],
                "old_hash": row[1],
                "new_hash": row[2],
                "diff": row[3],
                "status": row[4],
            }
            for row in cursor.fetchall()
        ]
        conn.close()
        return results

    def clear_extractions(self):
        """Clear all extraction tracking data."""
        conn = self._get_conn()
        conn.execute("DELETE FROM note_extractions")
        conn.execute("DELETE FROM extraction_diffs")
        conn.commit()
        conn.close()
