"""Embedding storage using sentence-transformers + numpy."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .utils import detect_device

logger = logging.getLogger(__name__)


class EmbeddingStore:
    """Compute, store, and search dense embeddings.

    Embeddings are persisted as .npz (vectors) + .json (id mapping).
    Hash-based dedup: already-embedded chunks are skipped.
    """

    def __init__(self, model_name: str, store_dir: Path) -> None:
        self._model_name = model_name
        self._store_dir = store_dir
        self._store_dir.mkdir(parents=True, exist_ok=True)

        self._model = None
        self._ids: List[str] = []
        self._vectors: Optional[np.ndarray] = None
        self._id_set: set[str] = set()
        self._load()

    # --- lazy model ---

    def _ensure_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            device = detect_device()
            logger.info("Loading embedding model %s on %s", self._model_name, device)
            self._model = SentenceTransformer(self._model_name, device=device)
        return self._model

    # --- persistence ---

    @property
    def _vec_path(self) -> Path:
        return self._store_dir / "embeddings.npz"

    @property
    def _ids_path(self) -> Path:
        return self._store_dir / "embedding_ids.json"

    def _load(self) -> None:
        import json
        if self._ids_path.exists() and self._vec_path.exists():
            self._ids = json.loads(self._ids_path.read_text(encoding="utf-8"))
            data = np.load(str(self._vec_path))
            self._vectors = data["vectors"]
            self._id_set = set(self._ids)

    def _save(self) -> None:
        import json
        self._ids_path.write_text(
            json.dumps(self._ids, ensure_ascii=False), encoding="utf-8"
        )
        if self._vectors is not None:
            np.savez_compressed(str(self._vec_path), vectors=self._vectors)

    # --- public API ---

    def add(self, items: List[Tuple[str, str]], batch_size: int = 64) -> int:
        """Add (hash_id, text) pairs. Returns number of newly added items."""
        new_items = [(hid, txt) for hid, txt in items if hid not in self._id_set]
        if not new_items:
            return 0

        model = self._ensure_model()
        texts = [txt for _, txt in new_items]
        new_vecs = model.encode(texts, batch_size=batch_size, show_progress_bar=False)
        new_vecs = np.array(new_vecs, dtype=np.float32)

        if self._vectors is not None:
            self._vectors = np.vstack([self._vectors, new_vecs])
        else:
            self._vectors = new_vecs

        for hid, _ in new_items:
            self._ids.append(hid)
            self._id_set.add(hid)

        self._save()
        return len(new_items)

    def get_vector(self, hash_id: str) -> Optional[np.ndarray]:
        """Get embedding vector for a hash_id."""
        if hash_id not in self._id_set or self._vectors is None:
            return None
        idx = self._ids.index(hash_id)
        return self._vectors[idx]

    def get_vectors(self, hash_ids: List[str]) -> np.ndarray:
        """Get embedding vectors for multiple hash_ids. Returns (N, dim) array."""
        indices = [self._ids.index(hid) for hid in hash_ids if hid in self._id_set]
        if not indices or self._vectors is None:
            return np.empty((0, 0), dtype=np.float32)
        return self._vectors[indices]

    def encode_query(self, query: str) -> np.ndarray:
        """Encode a query string into an embedding vector."""
        model = self._ensure_model()
        return np.array(model.encode(query, show_progress_bar=False), dtype=np.float32)

    def has(self, hash_id: str) -> bool:
        return hash_id in self._id_set

    def count(self) -> int:
        return len(self._ids)

    def delete(self, hash_id: str) -> None:
        """Remove an embedding by hash_id."""
        if hash_id not in self._id_set:
            return
        idx = self._ids.index(hash_id)
        self._ids.pop(idx)
        self._id_set.discard(hash_id)
        if self._vectors is not None:
            self._vectors = np.delete(self._vectors, idx, axis=0)
            if len(self._ids) == 0:
                self._vectors = None
        self._save()
