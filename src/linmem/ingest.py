"""Document ingestion: read files, chunk, deduplicate."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple

from .config import LinmemConfig
from .utils import text_hash

logger = logging.getLogger(__name__)

# supported extensions
_TEXT_EXTS = {".txt", ".md", ".markdown"}
_PDF_EXTS = {".pdf"}


def discover_files(directory: Path) -> List[Path]:
    """Recursively find supported document files."""
    files = []
    for p in sorted(directory.rglob("*")):
        if p.is_file() and p.suffix.lower() in (_TEXT_EXTS | _PDF_EXTS):
            files.append(p)
    return files


def read_file(path: Path) -> str:
    """Read a file's text content."""
    ext = path.suffix.lower()
    if ext in _TEXT_EXTS:
        return path.read_text(encoding="utf-8", errors="replace")
    if ext in _PDF_EXTS:
        return _read_pdf(path)
    return ""


def _read_pdf(path: Path) -> str:
    """Read PDF using pymupdf (fitz)."""
    try:
        import fitz
    except ImportError:
        raise RuntimeError(
            f"pymupdf required to read PDF files ({path.name}). "
            "Install with: uv pip install linmem[pdf]"
        )
    doc = fitz.open(str(path))
    pages = [page.get_text() for page in doc]
    doc.close()
    return "\n\n".join(pages)


def chunk_text(
    text: str,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> List[str]:
    """Split text into chunks.

    Strategy: split by double-newline (paragraphs) first, then merge small
    paragraphs up to chunk_size. If a single paragraph exceeds chunk_size,
    split by fixed length with overlap.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: List[str] = []
    buffer = ""

    for para in paragraphs:
        if len(para) > chunk_size:
            # flush buffer
            if buffer:
                chunks.append(buffer)
                buffer = ""
            # split long paragraph by fixed length
            for i in range(0, len(para), chunk_size - chunk_overlap):
                chunk = para[i : i + chunk_size]
                if chunk.strip():
                    chunks.append(chunk.strip())
        elif len(buffer) + len(para) + 2 > chunk_size:
            if buffer:
                chunks.append(buffer)
            buffer = para
        else:
            buffer = f"{buffer}\n\n{para}".strip() if buffer else para

    if buffer:
        chunks.append(buffer)

    return chunks


def ingest_directory(
    directory: Path,
    config: LinmemConfig,
) -> List[Tuple[str, str]]:
    """Ingest all documents in a directory.

    Returns list of (hash_id, text) chunks ready for indexing.
    """
    files = discover_files(directory)
    logger.info("Found %d files in %s", len(files), directory)

    all_chunks: List[Tuple[str, str]] = []
    seen_hashes: set[str] = set()

    for fpath in files:
        logger.info("Reading %s", fpath)
        text = read_file(fpath)
        if not text.strip():
            continue

        chunks = chunk_text(text, config.chunk_size, config.chunk_overlap)
        for chunk in chunks:
            h = text_hash(chunk)
            if h not in seen_hashes:
                seen_hashes.add(h)
                all_chunks.append((h, chunk))

    logger.info("Produced %d unique chunks", len(all_chunks))
    return all_chunks
