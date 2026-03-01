"""Retrieval orchestration: BM25 pre-filter -> LinearRAG graph ranking."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .bm25 import BM25Index
from .config import LinmemConfig
from .embedding import EmbeddingStore
from .graph import TriGraph
from .ner import NERExtractor

logger = logging.getLogger(__name__)


class Retriever:
    """Orchestrates BM25 -> LinearRAG two-stage retrieval."""

    def __init__(self, config: LinmemConfig) -> None:
        self.cfg = config
        idx = config.index_dir

        self.bm25 = BM25Index(idx / "bm25.db")
        self.ner = NERExtractor(config.ner_model, idx / "ner_cache.json")
        self.embeddings = EmbeddingStore(config.embedding_model, idx / "embeddings")
        self.graph = TriGraph(idx / "graph")

        # chunk text store (hash_id -> text) for returning results
        self._text_store_path = idx / "chunks.json"
        self._texts: Dict[str, str] = {}
        self._load_texts()

    def _load_texts(self) -> None:
        import json
        if self._text_store_path.exists():
            self._texts = json.loads(
                self._text_store_path.read_text(encoding="utf-8")
            )

    def _save_texts(self) -> None:
        import json
        self._text_store_path.parent.mkdir(parents=True, exist_ok=True)
        self._text_store_path.write_text(
            json.dumps(self._texts, ensure_ascii=False), encoding="utf-8"
        )

    # --- indexing ---

    def index_chunks(self, chunks: List[Tuple[str, str, Optional[float], Optional[Dict]]] | List[Tuple[str, str]]) -> int:
        """Index a list of chunks. Returns count of new chunks.

        Args:
            chunks: List of tuples, either:
                - (hash_id, text) for backward compatibility
                - (hash_id, text, timestamp, metadata) for temporal indexing
        """
        new_count = 0
        prev_hash = None

        # normalize input format
        normalized_chunks = []
        for chunk in chunks:
            if len(chunk) == 2:
                hid, txt = chunk
                timestamp, metadata = None, None
            elif len(chunk) == 4:
                hid, txt, timestamp, metadata = chunk
            else:
                raise ValueError(f"Invalid chunk format: {chunk}")
            normalized_chunks.append((hid, txt, timestamp, metadata))

        # filter already-indexed
        new_chunks = [(hid, txt, ts, meta) for hid, txt, ts, meta in normalized_chunks if hid not in self._texts]
        if not new_chunks:
            return 0

        # BM25
        for hid, txt, timestamp, metadata in new_chunks:
            import json
            metadata_json = json.dumps(metadata) if metadata else None
            self.bm25.insert(hid, txt, timestamp=timestamp, metadata=metadata_json)

        # NER (batch) - only need (hid, txt) for NER
        ner_input = [(hid, txt) for hid, txt, _, _ in new_chunks]
        ner_results = self.ner.extract_batch(
            ner_input, batch_size=self.cfg.ner_batch_size
        )

        # Embeddings (batch)
        self.embeddings.add(ner_input, batch_size=self.cfg.embedding_batch_size)

        # Graph
        for hid, txt, timestamp, metadata in new_chunks:
            entities = ner_results.get(hid, [])
            self.graph.add_paragraph(
                hid, txt, entities,
                language=self.cfg.language,
                prev_hash_id=prev_hash,
                timestamp=timestamp,
                metadata=metadata,
            )
            self._texts[hid] = txt
            prev_hash = hid
            new_count += 1

        self.graph.save()
        self._save_texts()
        return new_count

    # --- search ---

    def _compute_sigma_q(self, query: str) -> np.ndarray:
        """Compute query-sentence similarity vector (sigma_q).

        Returns cosine similarity between query embedding and each sentence
        embedding in the graph, as a vector of shape (n_sentences,).
        """
        n_s = self.graph.n_sentences
        if n_s == 0:
            return np.zeros(0, dtype=np.float32)

        query_vec = self.embeddings.encode_query(query)  # (dim,)
        sent_texts = self.graph.get_sentence_texts()

        # encode sentences (use the embedding model directly)
        model = self.embeddings._ensure_model()
        sent_vecs = model.encode(sent_texts, batch_size=self.cfg.embedding_batch_size, show_progress_bar=False)
        sent_vecs = np.array(sent_vecs, dtype=np.float32)

        # cosine similarity
        q_norm = query_vec / (np.linalg.norm(query_vec) + 1e-9)
        s_norms = sent_vecs / (np.linalg.norm(sent_vecs, axis=1, keepdims=True) + 1e-9)
        sigma = s_norms @ q_norm  # (n_s,)

        # clamp to [0, 1]
        sigma = np.clip(sigma, 0.0, 1.0)
        return sigma

    def search(self, query: str, top_k: int | None = None, start_time: Optional[float] = None, end_time: Optional[float] = None) -> List[Dict]:
        """BM25 pre-filter -> LinearRAG graph-enhanced ranking with optional temporal filtering.

        Args:
            query: Search query
            top_k: Number of results to return
            start_time: Filter results after this timestamp (inclusive)
            end_time: Filter results before this timestamp (inclusive)

        Returns list of dicts: {hash_id, text, score, timestamp, metadata}
        """
        top_k = top_k or self.cfg.retrieval_top_k

        # Step 1: BM25 pre-filter with temporal filtering
        bm25_results = self.bm25.search(query, top_k=self.cfg.bm25_top_k, start_time=start_time, end_time=end_time)
        if not bm25_results:
            return []

        bm25_ids = {hid for hid, _ in bm25_results}

        # Step 2: NER on query (don't pollute cache with query entries)
        nlp = self.ner._ensure_model()
        doc = nlp(query)
        query_entities = list({ent.text.strip() for ent in doc.ents if ent.text.strip()})

        # Step 3: Compute sigma_q (query-sentence semantic similarity)
        sigma_q = self._compute_sigma_q(query)

        # Step 4: LinearRAG graph retrieval with semantic bridge
        graph_results = self.graph.retrieve(
            query_entities=query_entities,
            max_iterations=self.cfg.max_iterations,
            iteration_threshold=self.cfg.iteration_threshold,
            damping=self.cfg.damping,
            top_k=top_k * 3,
            sigma_q=sigma_q,
        )

        # Step 5: Filter graph results by time if needed
        if start_time is not None or end_time is not None:
            filtered_graph_results = []
            for hid, score in graph_results:
                meta = self.graph.get_paragraph_metadata(hid)
                if meta and meta.get('timestamp') is not None:
                    ts = meta['timestamp']
                    if (start_time is None or ts >= start_time) and (end_time is None or ts <= end_time):
                        filtered_graph_results.append((hid, score))
                elif start_time is None and end_time is None:
                    # No timestamp, include if no time filter
                    filtered_graph_results.append((hid, score))
            graph_results = filtered_graph_results

        # Step 6: Merge scores
        # Combine BM25 and graph scores with passage_ratio weighting
        bm25_scores = {hid: score for hid, score in bm25_results}
        graph_scores = {hid: score for hid, score in graph_results}

        # normalize BM25 scores
        bm25_max = max(bm25_scores.values()) if bm25_scores else 1.0
        if bm25_max > 0:
            bm25_scores = {k: v / bm25_max for k, v in bm25_scores.items()}

        # normalize graph scores
        graph_max = max(graph_scores.values()) if graph_scores else 1.0
        if graph_max > 0:
            graph_scores = {k: v / graph_max for k, v in graph_scores.items()}

        # merge: graph_score * (1 - λ) + bm25_score * λ
        lam = self.cfg.passage_ratio
        all_ids = bm25_ids | set(graph_scores.keys())
        merged = []
        for hid in all_ids:
            gs = graph_scores.get(hid, 0.0)
            bs = bm25_scores.get(hid, 0.0)
            score = gs * (1 - lam) + bs * lam
            merged.append((hid, score))

        merged.sort(key=lambda x: x[1], reverse=True)

        results = []
        for hid, score in merged[:top_k]:
            result = {
                "hash_id": hid,
                "text": self._texts.get(hid, ""),
                "score": round(score, 6),
            }
            # Add timestamp and metadata if available
            meta = self.graph.get_paragraph_metadata(hid)
            if meta:
                if meta.get('timestamp') is not None:
                    result['timestamp'] = meta['timestamp']
                if meta.get('metadata'):
                    result['metadata'] = meta['metadata']
            results.append(result)
        return results

    # --- status ---

    def status(self) -> Dict:
        graph_stats = self.graph.stats()
        return {
            "chunks": len(self._texts),
            "bm25_indexed": self.bm25.count(),
            "embeddings": self.embeddings.count(),
            **graph_stats,
        }
