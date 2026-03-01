"""Integration tests — require ML models (spaCy, sentence-transformers).

All tests marked @pytest.mark.slow.
Run with: pytest tests/ -m slow
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def ensure_spacy_model():
    """Download zh_core_web_sm if not already installed."""
    import spacy
    try:
        spacy.load("zh_core_web_sm")
    except OSError:
        subprocess.check_call(
            [sys.executable, "-m", "spacy", "download", "zh_core_web_sm"]
        )


@pytest.fixture()
def index_dir(tmp_path):
    """Return a fresh temp directory for index data."""
    return tmp_path / "linmem_idx"


@pytest.mark.slow
class TestNERExtract:
    def test_ner_extract(self, ensure_spacy_model):
        import spacy
        nlp = spacy.load("zh_core_web_sm")
        doc = nlp("张明是星辰科技的创始人，公司位于深圳南山区。")
        entities = [ent.text for ent in doc.ents]
        # Should extract at least some named entities
        assert len(entities) > 0
        # The specific entities depend on the model, but common ones:
        entity_texts = " ".join(entities)
        assert any(
            name in entity_texts
            for name in ["张明", "星辰科技", "深圳", "南山"]
        ), f"Expected some known entities, got: {entities}"


@pytest.mark.slow
class TestIndexAndSearch:
    def test_full_pipeline(self, ensure_spacy_model, index_dir):
        from linmem.config import LinmemConfig
        from linmem.ingest import ingest_directory
        from linmem.retriever import Retriever

        cfg = LinmemConfig(data_dir=str(index_dir), language="zh")

        # Ingest fixtures
        chunks = ingest_directory(FIXTURES_DIR, cfg)
        assert len(chunks) > 0, "Should produce chunks from fixture docs"

        # Index
        retriever = Retriever(cfg)
        new_count = retriever.index_chunks(chunks)
        assert new_count > 0

        # Search
        results = retriever.search("星辰科技", top_k=3)
        assert len(results) > 0, "Should find results for 星辰科技"
        # Results should contain relevant text
        texts = " ".join(r["text"] for r in results)
        assert "星辰科技" in texts

        # Scores should be positive
        for r in results:
            assert r["score"] >= 0

        retriever.bm25.close()


@pytest.mark.slow
class TestIncrementalIndex:
    def test_incremental(self, ensure_spacy_model, index_dir):
        from linmem.config import LinmemConfig
        from linmem.ingest import chunk_text
        from linmem.retriever import Retriever
        from linmem.utils import text_hash

        cfg = LinmemConfig(data_dir=str(index_dir), language="zh")
        retriever = Retriever(cfg)

        # Index initial chunk
        text1 = "星辰科技是一家人工智能公司。"
        h1 = text_hash(text1)
        count1 = retriever.index_chunks([(h1, text1)])
        assert count1 == 1

        status1 = retriever.status()
        initial_chunks = status1["chunks"]

        # Index additional chunk
        text2 = "李薇是星辰科技的首席技术官。"
        h2 = text_hash(text2)
        count2 = retriever.index_chunks([(h2, text2)])
        assert count2 == 1

        status2 = retriever.status()
        assert status2["chunks"] == initial_chunks + 1

        # Re-index same chunks should not increase count
        count3 = retriever.index_chunks([(h1, text1), (h2, text2)])
        assert count3 == 0

        retriever.bm25.close()


@pytest.mark.slow
class TestCLIStatus:
    def test_cli_status_no_index(self, tmp_path):
        """linmem status with no existing index should not crash."""
        import shutil
        linmem_cmd = shutil.which("linmem") or shutil.which("linmem.exe")
        assert linmem_cmd is not None, "linmem CLI not found on PATH"

        result = subprocess.run(
            [linmem_cmd, "--data-dir", str(tmp_path / "nonexistent"),
             "status"],
            capture_output=True, text=True, encoding="utf-8",
        )
        # Should exit cleanly (0) with a message about no index
        assert result.returncode == 0
        assert "No index found" in result.stdout
