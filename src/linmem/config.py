"""Configuration for linmem."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


@dataclass
class LinmemConfig:
    """All configuration for a linmem index."""

    # --- paths ---
    data_dir: str = ".linmem"

    # --- language ---
    language: str = "zh"  # "zh" or "en"

    # --- BM25 ---
    bm25_top_k: int = 100

    # --- NER ---
    ner_batch_size: int = 64

    # --- Embedding ---
    embedding_batch_size: int = 64

    # --- Graph / LinearRAG hyperparameters ---
    max_iterations: int = 3
    iteration_threshold: float = 0.4
    passage_ratio: float = 0.05  # λ — DPR weight (small, entity signal dominates)
    damping: float = 0.85
    retrieval_top_k: int = 5

    # --- Chunking ---
    chunk_size: int = 512
    chunk_overlap: int = 64

    # --- LLM ---
    llm_base_url: Optional[str] = None
    llm_api_key: Optional[str] = None
    llm_model: str = "gpt-4o-mini"
    llm_max_retries: int = 3

    # --- derived ---

    @property
    def ner_model(self) -> str:
        return "zh_core_web_sm" if self.language == "zh" else "en_core_web_sm"

    @property
    def embedding_model(self) -> str:
        if self.language == "zh":
            return "BAAI/bge-small-zh-v1.5"
        return "sentence-transformers/all-mpnet-base-v2"

    @property
    def index_dir(self) -> Path:
        return Path(self.data_dir)

    # --- persistence ---

    def save(self, path: Optional[Path] = None) -> None:
        p = path or self.index_dir / "config.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "LinmemConfig":
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
