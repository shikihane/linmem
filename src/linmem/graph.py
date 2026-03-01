"""Tri-Graph construction and LinearRAG retrieval core.

Implements the LinearRAG (ICLR 2026) algorithm:
- Three-layer graph: Entity (Ve), Sentence (Vs), Paragraph (Vp)
- Two sparse matrices: Mention (sentence <-> entity), Contain (paragraph <-> entity)
- Two-stage retrieval:
  1. Entity activation via sparse matrix multiplication + threshold pruning
  2. Personalized PageRank via igraph
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import igraph as ig
import numpy as np

from .utils import split_sentences

logger = logging.getLogger(__name__)


class TriGraph:
    """Tri-Graph: entities / sentences / paragraphs with sparse adjacency."""

    def __init__(self, store_dir: Path) -> None:
        self._store_dir = store_dir
        self._store_dir.mkdir(parents=True, exist_ok=True)

        # node registries
        self._entities: List[str] = []       # entity text -> index
        self._entity_idx: Dict[str, int] = {}
        self._sentences: List[str] = []      # sentence hash -> index
        self._sent_idx: Dict[str, int] = {}
        self._paragraphs: List[str] = []     # paragraph hash_id -> index
        self._para_idx: Dict[str, int] = {}

        # edges (use sets to avoid duplicates)
        self._mention_edges: Set[Tuple[int, int]] = set()  # (sent_idx, ent_idx)
        self._contain_edges: Set[Tuple[int, int]] = set()  # (para_idx, ent_idx)
        # paragraph adjacency (sequential)
        self._para_adj: Set[Tuple[int, int]] = set()

        # sentence text storage for embedding lookup
        self._sent_texts: Dict[str, str] = {}  # sent_key -> text

        self._load()

    # --- node registration ---

    def _get_or_add_entity(self, name: str) -> int:
        if name not in self._entity_idx:
            idx = len(self._entities)
            self._entities.append(name)
            self._entity_idx[name] = idx
        return self._entity_idx[name]

    def _get_or_add_sentence(self, key: str) -> int:
        if key not in self._sent_idx:
            idx = len(self._sentences)
            self._sentences.append(key)
            self._sent_idx[key] = idx
        return self._sent_idx[key]

    def _get_or_add_paragraph(self, hash_id: str) -> int:
        if hash_id not in self._para_idx:
            idx = len(self._paragraphs)
            self._paragraphs.append(hash_id)
            self._para_idx[hash_id] = idx
        return self._para_idx[hash_id]

    # --- graph building ---

    def add_paragraph(
        self,
        hash_id: str,
        text: str,
        entities: List[str],
        language: str = "zh",
        prev_hash_id: Optional[str] = None,
    ) -> None:
        """Add a paragraph (chunk) to the Tri-Graph.

        Args:
            hash_id: unique id for this paragraph
            text: paragraph text
            entities: NER entities found in this paragraph
            language: for sentence splitting
            prev_hash_id: previous paragraph hash_id (for adjacency edges)
        """
        if hash_id in self._para_idx:
            return  # already indexed

        para_idx = self._get_or_add_paragraph(hash_id)

        # adjacency edge to previous paragraph
        if prev_hash_id and prev_hash_id in self._para_idx:
            prev_idx = self._para_idx[prev_hash_id]
            self._para_adj.add((prev_idx, para_idx))

        # split into sentences
        sents = split_sentences(text, language)
        sent_indices = []
        for i, sent in enumerate(sents):
            sent_key = f"{hash_id}:{i}"
            sent_idx = self._get_or_add_sentence(sent_key)
            sent_indices.append(sent_idx)
            self._sent_texts[sent_key] = sent

        # entity nodes + edges
        for ent_name in entities:
            ent_idx = self._get_or_add_entity(ent_name)
            # contain edge: paragraph <-> entity
            self._contain_edges.add((para_idx, ent_idx))
            # mention edges: find which sentences mention this entity
            for i, sent in enumerate(sents):
                if ent_name in sent:
                    sent_key = f"{hash_id}:{i}"
                    s_idx = self._sent_idx[sent_key]
                    self._mention_edges.add((s_idx, ent_idx))

    # --- sparse matrices ---

    def _build_mention_matrix(self) -> np.ndarray:
        """Build Mention matrix M: (n_sentences, n_entities), binary."""
        n_s = len(self._sentences)
        n_e = len(self._entities)
        if n_s == 0 or n_e == 0:
            return np.zeros((n_s, n_e), dtype=np.float32)
        M = np.zeros((n_s, n_e), dtype=np.float32)
        for s_idx, e_idx in self._mention_edges:
            M[s_idx, e_idx] = 1.0
        return M

    def _build_contain_matrix(self) -> np.ndarray:
        """Build Contain matrix C: (n_paragraphs, n_entities), binary."""
        n_p = len(self._paragraphs)
        n_e = len(self._entities)
        if n_p == 0 or n_e == 0:
            return np.zeros((n_p, n_e), dtype=np.float32)
        C = np.zeros((n_p, n_e), dtype=np.float32)
        for p_idx, e_idx in self._contain_edges:
            C[p_idx, e_idx] = 1.0
        return C

    # --- retrieval: entity activation ---

    def activate_entities(
        self,
        query_entities: List[str],
        max_iterations: int = 3,
        threshold: float = 0.4,
        sigma_q: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Stage 1: Entity activation via semantic bridging.

        a_q^t = MAX(diag(sigma_q) * M * a_q^{t-1}, a_q^{t-1})
        where sigma_q is the query-sentence similarity vector (n_sentences,).

        If sigma_q is None, falls back to structural propagation only.
        Returns activation vector over entities.
        """
        n_e = len(self._entities)
        if n_e == 0:
            return np.zeros(0, dtype=np.float32)

        # initial activation: 1.0 for query entities, 0.0 otherwise
        a = np.zeros(n_e, dtype=np.float32)
        for ent in query_entities:
            if ent in self._entity_idx:
                a[self._entity_idx[ent]] = 1.0

        if a.sum() == 0:
            return a

        M = self._build_mention_matrix()  # (n_s, n_e)

        for _ in range(max_iterations):
            # sentence activation = M @ a
            s_act = M @ a  # (n_s,)
            # apply sigma_q: weight by query-sentence semantic similarity
            if sigma_q is not None and len(sigma_q) == len(s_act):
                s_act = s_act * sigma_q
            # normalize sentence activation
            s_max = s_act.max()
            if s_max > 0:
                s_act = s_act / s_max
            # back to entities: M^T @ s_act
            a_new = M.T @ s_act  # (n_e,)
            # normalize
            a_max = a_new.max()
            if a_max > 0:
                a_new = a_new / a_max
            # threshold pruning
            a_new[a_new < threshold] = 0.0
            # monotonic: MAX(new, old)
            a = np.maximum(a, a_new)

        return a

    # --- retrieval: Personalized PageRank ---

    def _build_igraph(self) -> Tuple[ig.Graph, int, int, int]:
        """Build an igraph Graph with all three node layers."""
        n_e = len(self._entities)
        n_s = len(self._sentences)
        n_p = len(self._paragraphs)
        total = n_e + n_s + n_p

        g = ig.Graph(n=total, directed=False)
        # node types: 0..n_e-1 = entity, n_e..n_e+n_s-1 = sentence, n_e+n_s.. = paragraph
        offset_s = n_e
        offset_p = n_e + n_s

        edges = []
        # mention edges: sentence <-> entity
        for s_idx, e_idx in self._mention_edges:
            edges.append((offset_s + s_idx, e_idx))
        # contain edges: paragraph <-> entity
        for p_idx, e_idx in self._contain_edges:
            edges.append((offset_p + p_idx, e_idx))
        # paragraph adjacency
        for p1, p2 in self._para_adj:
            edges.append((offset_p + p1, offset_p + p2))

        if edges:
            g.add_edges(edges)
        return g, n_e, offset_s, offset_p

    def retrieve(
        self,
        query_entities: List[str],
        max_iterations: int = 3,
        iteration_threshold: float = 0.4,
        damping: float = 0.85,
        top_k: int = 5,
        sigma_q: Optional[np.ndarray] = None,
    ) -> List[Tuple[str, float]]:
        """Full LinearRAG retrieval: entity activation + PageRank.

        Args:
            sigma_q: query-sentence similarity vector (n_sentences,).
                     If provided, used as semantic bridge in entity activation.

        Returns list of (paragraph_hash_id, score) sorted by score descending.
        """
        if not self._paragraphs:
            return []

        # Stage 1: entity activation (with semantic bridge if sigma_q provided)
        entity_activation = self.activate_entities(
            query_entities, max_iterations, iteration_threshold, sigma_q=sigma_q
        )

        # Stage 2: Personalized PageRank
        g, n_e, offset_s, offset_p = self._build_igraph()
        n_total = g.vcount()

        # build reset vector: activated entities as seeds
        reset = np.zeros(n_total, dtype=np.float64)
        for i, val in enumerate(entity_activation):
            if val > 0:
                reset[i] = val

        # if no entities activated, return empty
        if reset.sum() == 0:
            return []

        reset = reset / reset.sum()  # normalize to probability

        pr = g.personalized_pagerank(
            damping=damping,
            reset=reset.tolist(),
            directed=False,
        )

        # extract paragraph scores
        para_scores = []
        for i, hash_id in enumerate(self._paragraphs):
            score = pr[offset_p + i]
            para_scores.append((hash_id, score))

        para_scores.sort(key=lambda x: x[1], reverse=True)
        return para_scores[:top_k]

    # --- persistence ---

    @property
    def _graph_path(self) -> Path:
        return self._store_dir / "trigraph.json"

    def _load(self) -> None:
        if not self._graph_path.exists():
            return
        data = json.loads(self._graph_path.read_text(encoding="utf-8"))
        self._entities = data.get("entities", [])
        self._entity_idx = {e: i for i, e in enumerate(self._entities)}
        self._sentences = data.get("sentences", [])
        self._sent_idx = {s: i for i, s in enumerate(self._sentences)}
        self._paragraphs = data.get("paragraphs", [])
        self._para_idx = {p: i for i, p in enumerate(self._paragraphs)}
        self._mention_edges = {tuple(e) for e in data.get("mention_edges", [])}
        self._contain_edges = {tuple(e) for e in data.get("contain_edges", [])}
        self._para_adj = {tuple(e) for e in data.get("para_adj", [])}
        self._sent_texts = data.get("sent_texts", {})

    def save(self) -> None:
        data = {
            "entities": self._entities,
            "sentences": self._sentences,
            "paragraphs": self._paragraphs,
            "mention_edges": list(self._mention_edges),
            "contain_edges": list(self._contain_edges),
            "para_adj": list(self._para_adj),
            "sent_texts": self._sent_texts,
        }
        self._graph_path.write_text(
            json.dumps(data, ensure_ascii=False), encoding="utf-8"
        )

    # --- stats ---

    def stats(self) -> Dict[str, int]:
        return {
            "entities": len(self._entities),
            "sentences": len(self._sentences),
            "paragraphs": len(self._paragraphs),
            "mention_edges": len(self._mention_edges),
            "contain_edges": len(self._contain_edges),
        }

    def has_paragraph(self, hash_id: str) -> bool:
        return hash_id in self._para_idx

    def get_sentence_texts(self) -> List[str]:
        """Return sentence texts in index order, for computing sigma_q."""
        return [self._sent_texts.get(key, "") for key in self._sentences]

    @property
    def n_sentences(self) -> int:
        return len(self._sentences)
