"""Named Entity Recognition module using spaCy."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Set

import spacy
from spacy.language import Language

logger = logging.getLogger(__name__)


class NERExtractor:
    """Extract named entities from text using spaCy.

    Maintains a persistent cache so already-processed chunks are skipped
    on incremental re-indexing.
    """

    def __init__(self, model_name: str, cache_path: Path) -> None:
        self._model_name = model_name
        self._cache_path = cache_path
        self._nlp: Language | None = None
        self._cache: Dict[str, List[str]] = {}
        self._load_cache()

    # --- lazy model loading ---

    def _ensure_model(self) -> Language:
        if self._nlp is None:
            try:
                self._nlp = spacy.load(self._model_name)
            except OSError:
                logger.info("Downloading spaCy model %s ...", self._model_name)
                spacy.cli.download(self._model_name)
                self._nlp = spacy.load(self._model_name)
        return self._nlp

    # --- cache ---

    def _load_cache(self) -> None:
        if self._cache_path.exists():
            self._cache = json.loads(self._cache_path.read_text(encoding="utf-8"))

    def _save_cache(self) -> None:
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._cache_path.write_text(
            json.dumps(self._cache, ensure_ascii=False), encoding="utf-8"
        )

    # --- extraction ---

    def extract(self, hash_id: str, text: str) -> List[str]:
        """Return deduplicated entity strings for a chunk."""
        if hash_id in self._cache:
            return self._cache[hash_id]

        nlp = self._ensure_model()
        doc = nlp(text)
        entities = list({ent.text.strip() for ent in doc.ents if ent.text.strip()})
        self._cache[hash_id] = entities
        return entities

    def extract_batch(
        self, items: List[tuple[str, str]], batch_size: int = 64
    ) -> Dict[str, List[str]]:
        """Extract entities for multiple (hash_id, text) pairs.

        Skips already-cached items. Returns mapping hash_id -> entities.
        """
        to_process = [(hid, txt) for hid, txt in items if hid not in self._cache]
        if not to_process:
            return {hid: self._cache[hid] for hid, _ in items}

        nlp = self._ensure_model()
        texts = [txt for _, txt in to_process]
        ids = [hid for hid, _ in to_process]

        for i, doc in enumerate(nlp.pipe(texts, batch_size=batch_size)):
            entities = list({ent.text.strip() for ent in doc.ents if ent.text.strip()})
            self._cache[ids[i]] = entities

        self._save_cache()
        return {hid: self._cache[hid] for hid, _ in items}

    def get_all_entities(self) -> Set[str]:
        """Return the set of all unique entity strings across all cached chunks."""
        all_ents: Set[str] = set()
        for ents in self._cache.values():
            all_ents.update(ents)
        return all_ents

    def delete(self, hash_id: str) -> None:
        """Remove a chunk from the NER cache."""
        self._cache.pop(hash_id, None)
        self._save_cache()
