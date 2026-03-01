"""Document ingestion: read files, chunk, deduplicate."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .config import LinmemConfig
from .utils import text_hash

logger = logging.getLogger(__name__)

# supported extensions
_TEXT_EXTS = {".txt", ".md", ".markdown"}
_PDF_EXTS = {".pdf"}
_JSONL_EXTS = {".jsonl"}


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


def parse_timestamp(ts_str: Optional[str]) -> Optional[float]:
    """Parse ISO 8601 timestamp string to Unix timestamp.

    Args:
        ts_str: ISO 8601 timestamp string (e.g., "2024-03-01T10:30:00Z")

    Returns:
        Unix timestamp (seconds since epoch) or None if parsing fails
    """
    if not ts_str:
        return None
    try:
        # Try parsing with timezone
        dt = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
        return dt.timestamp()
    except (ValueError, AttributeError):
        return None


def ingest_jsonl(
    jsonl_path: Path,
    config: LinmemConfig,
) -> List[Tuple[str, str, Optional[float], Optional[Dict]]]:
    """Ingest JSONL file (e.g., Claude Code execution logs).

    Each line should be a JSON object with at least a 'content' field.
    Optional fields: 'timestamp', 'type', 'tool', 'file', etc.

    Returns list of (hash_id, text, timestamp, metadata) tuples.
    """
    logger.info("Reading JSONL file: %s", jsonl_path)

    chunks: List[Tuple[str, str, Optional[float], Optional[Dict]]] = []
    seen_hashes: set[str] = set()

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning("Line %d: Invalid JSON: %s", line_num, e)
                continue

            # Extract content
            content = record.get('content', '')
            if not content or not isinstance(content, str):
                logger.debug("Line %d: No content field, skipping", line_num)
                continue

            # Parse timestamp
            timestamp = parse_timestamp(record.get('timestamp'))

            # Extract metadata (everything except content)
            metadata = {k: v for k, v in record.items() if k != 'content'}

            # Generate hash from content
            h = text_hash(content)
            if h in seen_hashes:
                continue
            seen_hashes.add(h)

            chunks.append((h, content, timestamp, metadata))

    logger.info("Produced %d unique chunks from JSONL", len(chunks))
    return chunks
