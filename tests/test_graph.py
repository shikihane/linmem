"""Tests for linmem.graph — TriGraph with mock entities."""

import numpy as np

from linmem.graph import TriGraph


class TestTriGraph:
    def test_add_paragraph(self, tmp_path):
        g = TriGraph(tmp_path / "graph")
        g.add_paragraph("p1", "张三在星辰科技工作。他负责AI项目。", ["张三", "星辰科技"], language="zh")
        stats = g.stats()
        assert stats["paragraphs"] == 1
        assert stats["entities"] == 2
        assert stats["sentences"] >= 1

    def test_add_duplicate_paragraph(self, tmp_path):
        g = TriGraph(tmp_path / "graph")
        g.add_paragraph("p1", "测试文本。", ["实体A"], language="zh")
        g.add_paragraph("p1", "测试文本。", ["实体A"], language="zh")
        assert g.stats()["paragraphs"] == 1

    def test_entity_activation(self, tmp_path):
        g = TriGraph(tmp_path / "graph")
        g.add_paragraph("p1", "张三在星辰科技工作。", ["张三", "星辰科技"], language="zh")
        g.add_paragraph("p2", "李四也在星辰科技。他认识张三。", ["李四", "星辰科技", "张三"], language="zh")

        activation = g.activate_entities(["星辰科技"])
        assert activation.shape[0] == len(g._entities)
        # "星辰科技" should have activation >= 1.0
        idx = g._entity_idx["星辰科技"]
        assert activation[idx] >= 1.0

    def test_entity_activation_unknown_entity(self, tmp_path):
        g = TriGraph(tmp_path / "graph")
        g.add_paragraph("p1", "Hello world.", ["Hello"], language="en")
        activation = g.activate_entities(["不存在的实体"])
        assert activation.sum() == 0.0

    def test_entity_activation_empty_graph(self, tmp_path):
        g = TriGraph(tmp_path / "graph")
        activation = g.activate_entities(["any"])
        assert len(activation) == 0

    def test_retrieve(self, tmp_path):
        g = TriGraph(tmp_path / "graph")
        g.add_paragraph("p1", "张三在星辰科技工作。", ["张三", "星辰科技"], language="zh")
        g.add_paragraph("p2", "李四在另一家公司。", ["李四"], language="zh")

        results = g.retrieve(["星辰科技"], top_k=5)
        assert len(results) > 0
        # p1 should rank higher than p2 (or p2 may not appear)
        hash_ids = [r[0] for r in results]
        assert "p1" in hash_ids

    def test_retrieve_empty_graph(self, tmp_path):
        g = TriGraph(tmp_path / "graph")
        results = g.retrieve(["anything"])
        assert results == []

    def test_stats(self, tmp_path):
        g = TriGraph(tmp_path / "graph")
        stats = g.stats()
        assert stats["entities"] == 0
        assert stats["paragraphs"] == 0

    def test_has_paragraph(self, tmp_path):
        g = TriGraph(tmp_path / "graph")
        g.add_paragraph("p1", "测试。", ["测试"], language="zh")
        assert g.has_paragraph("p1") is True
        assert g.has_paragraph("p2") is False

    def test_save_load_roundtrip(self, tmp_path):
        g = TriGraph(tmp_path / "graph")
        g.add_paragraph("p1", "张三在星辰科技。", ["张三", "星辰科技"], language="zh")
        g.add_paragraph("p2", "李四也在星辰科技。", ["李四", "星辰科技"], language="zh", prev_hash_id="p1")
        g.save()

        g2 = TriGraph(tmp_path / "graph")
        assert g2.stats() == g.stats()
        assert g2.has_paragraph("p1")
        assert g2.has_paragraph("p2")
        assert len(g2._para_adj) == len(g._para_adj)

    def test_dedup_across_paragraphs(self, tmp_path):
        """Same entity appearing in multiple paragraphs should be one node."""
        g = TriGraph(tmp_path / "graph")
        g.add_paragraph("p1", "星辰科技成立了。", ["星辰科技"], language="zh")
        g.add_paragraph("p2", "星辰科技很好。", ["星辰科技"], language="zh")
        assert g.stats()["entities"] == 1

    def test_get_sentence_texts(self, tmp_path):
        g = TriGraph(tmp_path / "graph")
        g.add_paragraph("p1", "第一句。第二句。", ["实体"], language="zh")
        texts = g.get_sentence_texts()
        assert len(texts) >= 2
        assert any("第一句" in t for t in texts)

    def test_paragraph_adjacency(self, tmp_path):
        g = TriGraph(tmp_path / "graph")
        g.add_paragraph("p1", "First.", ["A"], language="en")
        g.add_paragraph("p2", "Second.", ["B"], language="en", prev_hash_id="p1")
        assert len(g._para_adj) == 1
