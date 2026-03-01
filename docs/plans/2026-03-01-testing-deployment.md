# Testing & Deployment Implementation Plan

> **For Claude:** Execute this plan task-by-task.

**Goal:** Add unit tests, prepare test data, fix model defaults for CPU, and verify CLI end-to-end.

**Architecture:** Two-layer testing: pure-logic unit tests (no model deps) + integration tests (with models, marked slow). Small Chinese test corpus for E2E validation.

**Tech Stack:** pytest, pytest markers, spaCy zh_core_web_sm, BAAI/bge-small-zh-v1.5

**Verification Commands:**
- Test (fast): `pytest tests/ -m "not slow"`
- Test (all): `pytest tests/`
- Syntax: `python -m py_compile src/linmem/*.py`

---

### Task 1: Fix model defaults for CPU-friendly operation

**Files:**
- Modify: `src/linmem/config.py:51` — change `en_core_web_trf` to `en_core_web_sm`

**Step 1:** In config.py, change the English NER model:
```python
# Before
return "zh_core_web_sm" if self.language == "zh" else "en_core_web_trf"
# After
return "zh_core_web_sm" if self.language == "zh" else "en_core_web_sm"
```

**Step 2:** Commit
```bash
git add src/linmem/config.py
git commit -m "fix: use en_core_web_sm for CPU-friendly English NER"
```

---

### Task 2: Add pytest to dev dependencies + test config

**Files:**
- Modify: `pyproject.toml` — add `[project.optional-dependencies] dev`

**Step 1:** Add to pyproject.toml:
```toml
[project.optional-dependencies]
pdf = ["pymupdf"]
llm = ["openai"]
dev = ["pytest>=7.0"]
```

**Step 2:** Add pytest config:
```toml
[tool.pytest.ini_options]
markers = ["slow: requires ML models"]
testpaths = ["tests"]
```

**Step 3:** Install dev deps:
```bash
uv pip install -e ".[dev]"
```

**Step 4:** Commit

---

### Task 3: Unit tests — pure logic (no model deps)

**Files:**
- Create: `tests/test_utils.py`
- Create: `tests/test_config.py`
- Create: `tests/test_bm25.py`
- Create: `tests/test_ingest.py`
- Create: `tests/test_graph.py`

**test_utils.py** — test text_hash, split_sentences (zh/en), normalize_answer (zh/en), detect_device

**test_config.py** — test save/load roundtrip, ner_model/embedding_model properties for zh/en

**test_bm25.py** — test insert/search/delete/count/has/dedup (uses temp SQLite in tmp_path)

**test_ingest.py** — test chunk_text with various inputs: short text, long paragraph splitting, overlap, empty input

**test_graph.py** — test TriGraph: add_paragraph, entity activation, retrieve with mock entities, stats, save/load roundtrip, dedup (uses tmp_path)

**Step:** Write tests, run `pytest tests/ -m "not slow"`, verify all pass. Commit.

---

### Task 4: Create test corpus

**Files:**
- Create: `tests/fixtures/doc1.txt` through `tests/fixtures/doc5.txt`

5 short Chinese documents (~100-200 chars each) with overlapping entities across documents to test multi-hop retrieval. Topics: a fictional company "星辰科技" with people, products, and events spanning multiple docs.

**Step:** Create files. Commit.

---

### Task 5: Integration tests (marked slow)

**Depends on:** Task 2, Task 3, Task 4

**Files:**
- Create: `tests/test_integration.py`

**Tests (all marked `@pytest.mark.slow`):**
1. Download spaCy model if missing: `python -m spacy download zh_core_web_sm`
2. `test_ner_extract` — NER on a Chinese sentence, verify entities returned
3. `test_index_and_search` — full pipeline: ingest fixtures → index → search → verify results
4. `test_incremental_index` — index, add new doc, re-index, verify count increases
5. `test_cli_status` — subprocess call `linmem status`

**Step:** Write tests, run `pytest tests/ -m slow`, verify pass. Commit.

---

### Task 6: Download models and verify CLI E2E

**Step 1:** Download models:
```bash
python -m spacy download zh_core_web_sm
```
(sentence-transformers model auto-downloads on first use)

**Step 2:** E2E test:
```bash
linmem index tests/fixtures/
linmem search "星辰科技"
linmem status
```

**Step 3:** Verify output is reasonable. Commit any fixes.

---

## Verification Checklist

- [ ] All fast tests pass: `pytest tests/ -m "not slow"`
- [ ] All slow tests pass: `pytest tests/ -m slow`
- [ ] CLI E2E works: `linmem index` / `linmem search` / `linmem status`
- [ ] No model larger than ~100MB downloaded (CPU-friendly)

## Rollback Plan

If verification fails:
1. `git stash` current changes
2. Review failing tests
3. Fix or escalate
