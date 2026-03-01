"""Tests for temporal (timestamp) functionality."""

from __future__ import annotations

import json
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from linmem.bm25 import BM25Index
from linmem.config import LinmemConfig
from linmem.graph import TriGraph
from linmem.ingest import ingest_jsonl, parse_timestamp
from linmem.retriever import Retriever


class TestTimestampParsing:
    """Test timestamp parsing utilities."""

    def test_parse_iso8601_with_z(self):
        ts = parse_timestamp("2024-03-01T10:30:00Z")
        assert ts is not None
        dt = datetime.fromtimestamp(ts)
        assert dt.year == 2024
        assert dt.month == 3
        assert dt.day == 1

    def test_parse_iso8601_with_offset(self):
        ts = parse_timestamp("2024-03-01T10:30:00+08:00")
        assert ts is not None

    def test_parse_invalid_timestamp(self):
        assert parse_timestamp("invalid") is None
        assert parse_timestamp("") is None
        assert parse_timestamp(None) is None


class TestBM25Temporal:
    """Test BM25 index with timestamp support."""

    def test_insert_with_timestamp(self, tmp_path):
        db = BM25Index(tmp_path / "test.db")
        timestamp = datetime(2024, 3, 1, 10, 0).timestamp()

        success = db.insert("id1", "test content", timestamp=timestamp)
        assert success
        assert db.count() == 1

    def test_insert_with_metadata(self, tmp_path):
        db = BM25Index(tmp_path / "test.db")
        metadata = json.dumps({"type": "tool_use", "tool": "Read"})

        success = db.insert("id1", "test content", metadata=metadata)
        assert success

    def test_search_with_time_filter(self, tmp_path):
        db = BM25Index(tmp_path / "test.db")

        # Insert records with different timestamps
        ts1 = datetime(2024, 3, 1, 10, 0).timestamp()
        ts2 = datetime(2024, 3, 1, 11, 0).timestamp()
        ts3 = datetime(2024, 3, 1, 12, 0).timestamp()

        db.insert("id1", "test content one", timestamp=ts1)
        db.insert("id2", "test content two", timestamp=ts2)
        db.insert("id3", "test content three", timestamp=ts3)

        # Search without time filter
        results = db.search("test")
        assert len(results) == 3

        # Search with start_time filter
        results = db.search("test", start_time=ts2)
        assert len(results) == 2
        assert all(hid in ["id2", "id3"] for hid, _ in results)

        # Search with end_time filter
        results = db.search("test", end_time=ts2)
        assert len(results) == 2
        assert all(hid in ["id1", "id2"] for hid, _ in results)

        # Search with both filters
        results = db.search("test", start_time=ts2, end_time=ts2)
        assert len(results) == 1
        assert results[0][0] == "id2"


class TestTriGraphTemporal:
    """Test TriGraph with timestamp metadata."""

    def test_add_paragraph_with_timestamp(self, tmp_path):
        graph = TriGraph(tmp_path)
        timestamp = datetime(2024, 3, 1, 10, 0).timestamp()

        graph.add_paragraph(
            "id1",
            "test content",
            ["entity1"],
            timestamp=timestamp
        )

        meta = graph.get_paragraph_metadata("id1")
        assert meta is not None
        assert meta["timestamp"] == timestamp

    def test_add_paragraph_with_metadata(self, tmp_path):
        graph = TriGraph(tmp_path)
        metadata = {"type": "tool_use", "tool": "Read"}

        graph.add_paragraph(
            "id1",
            "test content",
            ["entity1"],
            metadata=metadata
        )

        meta = graph.get_paragraph_metadata("id1")
        assert meta is not None
        assert meta["metadata"] == metadata

    def test_save_and_load_metadata(self, tmp_path):
        graph = TriGraph(tmp_path)
        timestamp = datetime(2024, 3, 1, 10, 0).timestamp()
        metadata = {"type": "tool_use"}

        graph.add_paragraph(
            "id1",
            "test content",
            ["entity1"],
            timestamp=timestamp,
            metadata=metadata
        )
        graph.save()

        # Load in new instance
        graph2 = TriGraph(tmp_path)
        meta = graph2.get_paragraph_metadata("id1")
        assert meta["timestamp"] == timestamp
        assert meta["metadata"] == metadata


class TestJSONLIngestion:
    """Test JSONL file ingestion."""

    def test_ingest_jsonl_basic(self, tmp_path):
        jsonl_file = tmp_path / "test.jsonl"
        jsonl_file.write_text(
            '{"timestamp": "2024-03-01T10:00:00Z", "type": "tool_use", "content": "test content 1"}\n'
            '{"timestamp": "2024-03-01T11:00:00Z", "type": "tool_result", "content": "test content 2"}\n',
            encoding="utf-8"
        )

        cfg = LinmemConfig(data_dir=str(tmp_path / ".linmem"))
        chunks = ingest_jsonl(jsonl_file, cfg)

        assert len(chunks) == 2
        assert all(len(chunk) == 4 for chunk in chunks)  # (hash_id, text, timestamp, metadata)

        # Check first chunk
        hash_id, text, timestamp, metadata = chunks[0]
        assert text == "test content 1"
        assert timestamp is not None
        assert metadata["type"] == "tool_use"

    def test_ingest_jsonl_no_timestamp(self, tmp_path):
        jsonl_file = tmp_path / "test.jsonl"
        jsonl_file.write_text(
            '{"type": "message", "content": "test without timestamp"}\n',
            encoding="utf-8"
        )

        cfg = LinmemConfig(data_dir=str(tmp_path / ".linmem"))
        chunks = ingest_jsonl(jsonl_file, cfg)

        assert len(chunks) == 1
        hash_id, text, timestamp, metadata = chunks[0]
        assert text == "test without timestamp"
        assert timestamp is None

    def test_ingest_jsonl_skip_invalid(self, tmp_path):
        jsonl_file = tmp_path / "test.jsonl"
        jsonl_file.write_text(
            '{"content": "valid line"}\n'
            'invalid json line\n'
            '{"no_content": "skip this"}\n'
            '{"content": "another valid line"}\n',
            encoding="utf-8"
        )

        cfg = LinmemConfig(data_dir=str(tmp_path / ".linmem"))
        chunks = ingest_jsonl(jsonl_file, cfg)

        assert len(chunks) == 2


class TestRetrieverTemporal:
    """Test Retriever with temporal queries."""

    def test_index_chunks_with_timestamp(self, tmp_path):
        cfg = LinmemConfig(data_dir=str(tmp_path / ".linmem"))
        retriever = Retriever(cfg)

        timestamp = datetime(2024, 3, 1, 10, 0).timestamp()
        metadata = {"type": "tool_use"}

        chunks = [
            ("id1", "test content", timestamp, metadata)
        ]

        count = retriever.index_chunks(chunks)
        assert count == 1

    def test_search_with_temporal_filter(self, tmp_path):
        cfg = LinmemConfig(data_dir=str(tmp_path / ".linmem"))
        retriever = Retriever(cfg)

        # Index chunks with different timestamps
        ts1 = datetime(2024, 3, 1, 10, 0).timestamp()
        ts2 = datetime(2024, 3, 1, 11, 0).timestamp()
        ts3 = datetime(2024, 3, 1, 12, 0).timestamp()

        chunks = [
            ("id1", "张明在星辰科技工作", ts1, {"type": "message"}),
            ("id2", "李薇负责市场部", ts2, {"type": "message"}),
            ("id3", "王强是技术总监", ts3, {"type": "message"}),
        ]

        retriever.index_chunks(chunks)

        # Search without time filter
        results = retriever.search("星辰科技")
        assert len(results) > 0

        # Search with time filter
        results = retriever.search("星辰科技", start_time=ts2)
        # Should not include id1 (before ts2)
        result_ids = [r["hash_id"] for r in results]
        assert "id1" not in result_ids

    def test_search_results_include_metadata(self, tmp_path):
        cfg = LinmemConfig(data_dir=str(tmp_path / ".linmem"))
        retriever = Retriever(cfg)

        timestamp = datetime(2024, 3, 1, 10, 0).timestamp()
        metadata = {"type": "tool_use", "tool": "Read"}

        chunks = [
            ("id1", "test content with metadata", timestamp, metadata)
        ]

        retriever.index_chunks(chunks)
        results = retriever.search("test")

        assert len(results) > 0
        result = results[0]
        assert "timestamp" in result
        assert result["timestamp"] == timestamp
        assert "metadata" in result
        assert result["metadata"]["type"] == "tool_use"


class TestBackwardCompatibility:
    """Test backward compatibility with old chunk format."""

    def test_index_chunks_old_format(self, tmp_path):
        cfg = LinmemConfig(data_dir=str(tmp_path / ".linmem"))
        retriever = Retriever(cfg)

        # Old format: (hash_id, text)
        chunks = [
            ("id1", "test content one"),
            ("id2", "test content two"),
        ]

        count = retriever.index_chunks(chunks)
        assert count == 2

        results = retriever.search("test")
        assert len(results) == 2

    def test_index_chunks_mixed_format(self, tmp_path):
        cfg = LinmemConfig(data_dir=str(tmp_path / ".linmem"))
        retriever = Retriever(cfg)

        timestamp = datetime(2024, 3, 1, 10, 0).timestamp()

        # Mixed format
        chunks = [
            ("id1", "old format chunk"),
            ("id2", "new format chunk", timestamp, {"type": "test"}),
        ]

        count = retriever.index_chunks(chunks)
        assert count == 2
