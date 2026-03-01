"""Tests for linmem.ingest — chunk_text with various inputs."""

from linmem.ingest import chunk_text


class TestChunkText:
    def test_short_text_single_chunk(self):
        text = "This is short."
        chunks = chunk_text(text, chunk_size=512)
        assert len(chunks) == 1
        assert chunks[0] == "This is short."

    def test_paragraph_merging(self):
        text = "Para one.\n\nPara two.\n\nPara three."
        chunks = chunk_text(text, chunk_size=100)
        # All three paragraphs fit in one chunk
        assert len(chunks) == 1
        assert "Para one." in chunks[0]
        assert "Para three." in chunks[0]

    def test_paragraph_splitting(self):
        # Two paragraphs that together exceed chunk_size
        p1 = "A" * 60
        p2 = "B" * 60
        text = f"{p1}\n\n{p2}"
        chunks = chunk_text(text, chunk_size=100)
        assert len(chunks) == 2

    def test_long_paragraph_fixed_split(self):
        # Single paragraph longer than chunk_size
        text = "X" * 300
        chunks = chunk_text(text, chunk_size=100, chunk_overlap=20)
        assert len(chunks) > 1
        # First chunk should be exactly chunk_size
        assert len(chunks[0]) == 100

    def test_overlap(self):
        text = "X" * 200
        chunks = chunk_text(text, chunk_size=100, chunk_overlap=20)
        # With overlap of 20, step is 80
        # Chunks at: 0-100, 80-180, 160-200
        assert len(chunks) == 3

    def test_empty_input(self):
        assert chunk_text("") == []
        assert chunk_text("   ") == []
        assert chunk_text("\n\n\n") == []

    def test_preserves_content(self):
        text = "Hello world.\n\nGoodbye world."
        chunks = chunk_text(text, chunk_size=512)
        combined = " ".join(chunks)
        assert "Hello world." in combined
        assert "Goodbye world." in combined
