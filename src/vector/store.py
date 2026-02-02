"""SQLite-vec vector storage."""

import sqlite3
import struct
from pathlib import Path
from typing import Any

import sqlite_vec


def serialize_vector(vec: list[float]) -> bytes:
    """Serialize a vector to bytes for sqlite-vec."""
    return struct.pack(f"{len(vec)}f", *vec)


def deserialize_vector(data: bytes, dimensions: int) -> list[float]:
    """Deserialize bytes back to a vector."""
    return list(struct.unpack(f"{dimensions}f", data))


class VectorStore:
    """SQLite-vec based vector storage."""

    def __init__(self, db_path: Path, dimensions: int = 384):
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

        # Create metadata table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS notes (
                id INTEGER PRIMARY KEY,
                path TEXT UNIQUE NOT NULL,
                title TEXT,
                content TEXT,
                chunk_index INTEGER DEFAULT 0
            )
        """)

        # Create virtual table for vectors
        conn.execute(f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS note_vectors USING vec0(
                embedding float[{self.dimensions}]
            )
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
        """Get the number of stored vectors."""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM notes")
        count = cursor.fetchone()[0]
        conn.close()
        return count

    def clear(self):
        """Clear all data."""
        conn = self._get_conn()
        conn.execute("DELETE FROM notes")
        conn.execute("DELETE FROM note_vectors")
        conn.commit()
        conn.close()
