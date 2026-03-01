"""Tests for linmem.bm25 — SQLite FTS5 BM25 index."""

from linmem.bm25 import BM25Index


class TestBM25Index:
    def test_insert_and_count(self, tmp_path):
        idx = BM25Index(tmp_path / "bm25.db")
        assert idx.count() == 0
        assert idx.insert("h1", "hello world") is True
        assert idx.count() == 1
        idx.close()

    def test_insert_duplicate_returns_false(self, tmp_path):
        idx = BM25Index(tmp_path / "bm25.db")
        idx.insert("h1", "hello world")
        assert idx.insert("h1", "hello world") is False
        assert idx.count() == 1
        idx.close()

    def test_has(self, tmp_path):
        idx = BM25Index(tmp_path / "bm25.db")
        idx.insert("h1", "hello world")
        assert idx.has("h1") is True
        assert idx.has("h2") is False
        idx.close()

    def test_search(self, tmp_path):
        idx = BM25Index(tmp_path / "bm25.db")
        idx.insert("h1", "the quick brown fox")
        idx.insert("h2", "the lazy dog sleeps")
        idx.insert("h3", "fox jumps over the dog")

        results = idx.search("fox")
        hash_ids = [r[0] for r in results]
        assert "h1" in hash_ids
        assert "h3" in hash_ids
        # scores should be positive
        for _, score in results:
            assert score > 0
        idx.close()

    def test_delete(self, tmp_path):
        idx = BM25Index(tmp_path / "bm25.db")
        idx.insert("h1", "hello world")
        assert idx.delete("h1") is True
        assert idx.count() == 0
        assert idx.has("h1") is False
        idx.close()

    def test_delete_nonexistent(self, tmp_path):
        idx = BM25Index(tmp_path / "bm25.db")
        assert idx.delete("missing") is False
        idx.close()

    def test_dedup(self, tmp_path):
        """Insert same hash_id twice, second should fail."""
        idx = BM25Index(tmp_path / "bm25.db")
        idx.insert("h1", "content a")
        idx.insert("h1", "content b")  # duplicate hash_id
        assert idx.count() == 1
        idx.close()
