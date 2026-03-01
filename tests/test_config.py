"""Tests for linmem.config — save/load roundtrip, properties."""

from pathlib import Path

from linmem.config import LinmemConfig


class TestConfigProperties:
    def test_ner_model_zh(self):
        cfg = LinmemConfig(language="zh")
        assert cfg.ner_model == "zh_core_web_sm"

    def test_ner_model_en(self):
        cfg = LinmemConfig(language="en")
        assert cfg.ner_model == "en_core_web_sm"

    def test_embedding_model_zh(self):
        cfg = LinmemConfig(language="zh")
        assert cfg.embedding_model == "BAAI/bge-small-zh-v1.5"

    def test_embedding_model_en(self):
        cfg = LinmemConfig(language="en")
        assert cfg.embedding_model == "sentence-transformers/all-mpnet-base-v2"

    def test_index_dir(self):
        cfg = LinmemConfig(data_dir="myindex")
        assert cfg.index_dir == Path("myindex")


class TestConfigSaveLoad:
    def test_roundtrip(self, tmp_path):
        cfg = LinmemConfig(
            data_dir=str(tmp_path / "idx"),
            language="en",
            bm25_top_k=50,
            chunk_size=256,
        )
        save_path = tmp_path / "config.json"
        cfg.save(save_path)

        loaded = LinmemConfig.load(save_path)
        assert loaded.language == "en"
        assert loaded.bm25_top_k == 50
        assert loaded.chunk_size == 256

    def test_roundtrip_defaults(self, tmp_path):
        cfg = LinmemConfig(data_dir=str(tmp_path / "idx"))
        save_path = tmp_path / "config.json"
        cfg.save(save_path)

        loaded = LinmemConfig.load(save_path)
        assert loaded.language == cfg.language
        assert loaded.chunk_overlap == cfg.chunk_overlap
        assert loaded.damping == cfg.damping
