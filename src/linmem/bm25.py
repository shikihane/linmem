"""BM25 index backed by SQLite FTS5."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import List, Optional, Tuple


class BM25Index:
    """SQLite FTS5-based BM25 full-text search index."""

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._init_tables()

    def _init_tables(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS chunks (
                hash_id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                timestamp REAL,
                metadata TEXT
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                content,
                content_rowid='rowid',
                tokenize='trigram'
            );
        """)
        # trigger: keep FTS in sync
        self._conn.executescript("""
            CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
                INSERT INTO chunks_fts(rowid, content)
                VALUES (new.rowid, new.content);
            END;
            CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
                DELETE FROM chunks_fts WHERE rowid = old.rowid;
            END;
        """)
        self._conn.commit()

    # --- public API ---

    def insert(self, hash_id: str, text: str, timestamp: Optional[float] = None, metadata: Optional[str] = None) -> bool:
        """Insert a chunk. Returns False if hash_id already exists.

        Args:
            hash_id: Unique identifier for the chunk
            text: Text content
            timestamp: Unix timestamp (seconds since epoch)
            metadata: JSON string with additional metadata
        """
        try:
            self._conn.execute(
                "INSERT INTO chunks (hash_id, content, timestamp, metadata) VALUES (?, ?, ?, ?)",
                (hash_id, text, timestamp, metadata),
            )
            self._conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False

    def search(self, query: str, top_k: int = 100, start_time: Optional[float] = None, end_time: Optional[float] = None) -> List[Tuple[str, float]]:
        """BM25 search with optional temporal filtering.

        Args:
            query: Search query
            top_k: Maximum number of results
            start_time: Filter results after this timestamp (inclusive)
            end_time: Filter results before this timestamp (inclusive)

        Returns:
            List of (hash_id, bm25_score) tuples
        """
        # Build WHERE clause for temporal filtering
        where_clauses = ["chunks_fts MATCH ?"]
        params = [query]

        if start_time is not None:
            where_clauses.append("c.timestamp >= ?")
            params.append(start_time)

        if end_time is not None:
            where_clauses.append("c.timestamp <= ?")
            params.append(end_time)

        where_clause = " AND ".join(where_clauses)
        params.append(top_k)

        rows = self._conn.execute(
            f"""
            SELECT c.hash_id, f.rank
            FROM chunks_fts f
            JOIN chunks c ON c.rowid = f.rowid
            WHERE {where_clause}
            ORDER BY f.rank
            LIMIT ?
            """,
            params,
        ).fetchall()
        # FTS5 rank is negative (lower = better), negate for positive scores
        return [(hid, -score) for hid, score in rows]

    def delete(self, hash_id: str) -> bool:
        """Delete a chunk by hash_id."""
        cur = self._conn.execute("DELETE FROM chunks WHERE hash_id = ?", (hash_id,))
        self._conn.commit()
        return cur.rowcount > 0

    def count(self) -> int:
        """Return total number of indexed chunks."""
        row = self._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()
        return row[0] if row else 0

    def has(self, hash_id: str) -> bool:
        """Check if a chunk exists."""
        row = self._conn.execute(
            "SELECT 1 FROM chunks WHERE hash_id = ?", (hash_id,)
        ).fetchone()
        return row is not None

    def close(self) -> None:
        self._conn.close()
