"""Tests for linmem.utils — pure logic, no model dependencies."""

from linmem.utils import text_hash, split_sentences, normalize_answer, detect_device


class TestTextHash:
    def test_deterministic(self):
        assert text_hash("hello") == text_hash("hello")

    def test_different_inputs(self):
        assert text_hash("a") != text_hash("b")

    def test_returns_hex_string(self):
        h = text_hash("test")
        assert isinstance(h, str)
        assert len(h) == 64  # SHA-256 hex digest


class TestSplitSentences:
    def test_chinese_basic(self):
        sents = split_sentences("你好。世界！测试？", language="zh")
        assert sents == ["你好。", "世界！", "测试？"]

    def test_chinese_semicolon(self):
        sents = split_sentences("第一句；第二句。", language="zh")
        assert sents == ["第一句；", "第二句。"]

    def test_chinese_newline(self):
        sents = split_sentences("第一段\n第二段", language="zh")
        assert sents == ["第一段", "第二段"]

    def test_english_basic(self):
        sents = split_sentences("Hello world. How are you? Fine!", language="en")
        assert sents == ["Hello world.", "How are you?", "Fine!"]

    def test_empty_string(self):
        assert split_sentences("", language="zh") == []
        assert split_sentences("", language="en") == []

    def test_single_sentence_no_terminal(self):
        sents = split_sentences("no punctuation", language="en")
        assert sents == ["no punctuation"]


class TestNormalizeAnswer:
    def test_chinese_removes_punctuation(self):
        result = normalize_answer("你好，世界！", language="zh")
        assert result == "你好世界"

    def test_chinese_removes_whitespace(self):
        result = normalize_answer("你 好 世 界", language="zh")
        assert result == "你好世界"

    def test_english_removes_articles(self):
        result = normalize_answer("The quick brown fox", language="en")
        assert result == "quick brown fox"

    def test_english_removes_punctuation(self):
        result = normalize_answer("Hello, world!", language="en")
        assert result == "hello world"

    def test_english_lowercases(self):
        result = normalize_answer("HELLO", language="en")
        assert result == "hello"

    def test_chinese_lowercases(self):
        result = normalize_answer("ABC你好", language="zh")
        assert result == "abc你好"


class TestDetectDevice:
    def test_returns_valid_device(self):
        device = detect_device()
        assert device in ("cuda", "mps", "cpu")
