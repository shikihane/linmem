# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

linmem is a local memory search CLI tool combining BM25 full-text search (SQLite FTS5) with LinearRAG graph-based ranking (Tri-Graph + PageRank). Primarily targets Chinese documents.

- Language: Python 3.10+, build system: hatchling
- Entry point: `linmem` CLI via `src/linmem/cli.py:main`
- License: GPL-3.0-or-later

## Commands

```bash
# Install (dev mode)
pip install -e ".[dev]"

# Tests
pytest tests/ -m "not slow"    # Fast tests only (no ML model deps), 50 tests
pytest tests/ -m slow           # Integration tests (require spaCy + embedding models), 4 tests
pytest tests/                   # All tests
pytest tests/test_bm25.py -k "test_insert"  # Single test

# CLI
linmem index <directory>        # Index documents
linmem search "<query>"         # Search
linmem ask "<question>"         # Search + LLM answer (requires LLM config)
linmem status                   # Index statistics
```

**CLI caveat**: `--data-dir` is a global option and must come *before* the subcommand (e.g., `linmem --data-dir ./myindex index ./docs/`).

## Architecture

```
文档 → [Ingest] chunk_text → [Index] BM25 + NER + Embedding + TriGraph → [Retrieve] BM25预筛 → LinearRAG两阶段排序 → [可选] LLM生成
```

### Module Dependency Graph

```
cli.py
├→ config.py          # Dataclass config, persisted as config.json
├→ ingest.py          # File discovery + paragraph-based chunking
│  └→ utils.py
└→ retriever.py       # Orchestrates the full search pipeline
   ├→ bm25.py         # SQLite FTS5 with trigram tokenizer
   ├→ ner.py          # spaCy NER with JSON cache
   ├→ embedding.py    # sentence-transformers, persisted as .npz
   ├→ graph.py        # 3-layer graph (entity/sentence/paragraph) + PageRank
   │  └→ utils.py
   └→ llm.py          # Optional OpenAI-compatible LLM call
```

### Indexing Pipeline

`ingest.discover_files()` → `read_file()` → `chunk_text()` produces `(hash_id, text)` tuples where hash_id = SHA-256. Then `retriever.index_chunks()` fans out to four stores in parallel: BM25 insert, NER extraction (batch), embedding computation (batch), and TriGraph node/edge construction.

### Search Pipeline (Two-Stage)

1. **BM25 pre-filter**: Top 100 candidates via SQLite FTS5
2. **LinearRAG re-ranking**: Entity activation (3 iterations with semantic bridging via query-sentence cosine similarity) → Personalized PageRank over tri-graph → Paragraph scores
3. **Score merging**: `final = graph_score * 0.95 + bm25_score * 0.05` (controlled by `passage_ratio`)

### Storage Layout (default `.linmem/`)

| File | Format | Contents |
|------|--------|----------|
| `bm25.db` | SQLite (WAL, FTS5 trigram) | Chunk text + full-text index |
| `ner_cache.json` | JSON | `{hash_id: [entities]}` |
| `embeddings/embeddings.npz` | NumPy compressed | Dense vectors (N × 384 for BGE) |
| `embeddings/embedding_ids.json` | JSON | Positional hash_id mapping |
| `trigraph.json` | JSON | Graph nodes, edges, sentence texts |
| `chunks.json` | JSON | `{hash_id: text}` for result display |
| `config.json` | JSON | Full config dump |

## Key Design Decisions

- **Trigram tokenizer** for FTS5: `unicode61` cannot segment Chinese; `trigram` works at character level for CJK.
- **Lazy model loading**: spaCy and sentence-transformers models load only on first use.
- **Hash-based dedup**: All stores use SHA-256 of chunk text as ID; re-indexing skips existing chunks.
- **Language-dependent models**: Config `language` field ("zh"/"en") determines NER model (`zh_core_web_sm` vs `en_core_web_sm`) and embedding model (`BAAI/bge-small-zh-v1.5` vs `all-mpnet-base-v2`).

## ML Model Dependencies

| Model | Purpose | Size |
|-------|---------|------|
| `zh_core_web_sm` | spaCy Chinese NER | ~75 MB |
| `BAAI/bge-small-zh-v1.5` | Sentence embeddings (Chinese) | ~184 MB |

In proxy environments, set `HF_ENDPOINT=https://hf-mirror.com` for HuggingFace model downloads.

## Test Fixtures

Test corpus: 5 Chinese documents about fictional company "星辰科技" in `tests/fixtures/`. Entities cross-reference across documents (张明, 李薇, 星语, 星图, etc.) to exercise graph retrieval.

## Known Pitfalls

- BM25 FTS5 delete trigger must use `DELETE FROM chunks_fts WHERE rowid = old.rowid` (not external content table syntax).
- Windows path assertions in tests must compare `Path` objects, not strings.
