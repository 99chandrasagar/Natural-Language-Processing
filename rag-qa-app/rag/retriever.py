"""
rag/retriever.py
────────────────
Embeds text chunks and indexes them in FAISS for fast similarity search.

Embedding model: sentence-transformers/all-MiniLM-L6-v2
  - 384-dimensional dense vectors
  - Fast, lightweight, runs locally (no API key needed)
  - Great for semantic similarity tasks

Index type: FAISS IndexFlatIP (inner product / cosine similarity)
  - Exact nearest-neighbour search
  - Good for < 100k chunks; swap for IVF or HNSW at larger scale
"""

import logging
import numpy as np
from typing import List

log = logging.getLogger(__name__)


class Retriever:
    """
    Encapsulates embedding + FAISS index for one document session.

    Usage:
        retriever = Retriever(chunks)          # builds index
        results   = retriever.retrieve(query)  # returns top-k chunks
    """

    def __init__(self, chunks: List[str], model_name: str = "all-MiniLM-L6-v2"):
        self.chunks = chunks
        self._model_name = model_name
        self._model = self._load_model(model_name)
        self._index = self._build_index(chunks)

    # ── Public API ─────────────────────────────────────────────────────────────

    def retrieve(self, query: str, top_k: int = 4) -> List[str]:
        """
        Return the top-k most semantically similar chunks for a given query.

        Steps:
          1. Embed the query
          2. Cosine-search FAISS index
          3. Return matching chunk strings
        """
        query_vec = self._embed([query])  # shape (1, dim)
        query_vec = self._normalize(query_vec)

        k = min(top_k, len(self.chunks))
        distances, indices = self._index.search(query_vec, k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            log.debug("Retrieved chunk %d (score=%.4f): %s…", idx, dist, self.chunks[idx][:60])
            results.append(self.chunks[idx])

        return results

    def count(self) -> int:
        """Number of chunks in the index."""
        return len(self.chunks)

    # ── Internal ───────────────────────────────────────────────────────────────

    def _load_model(self, model_name: str):
        """Load a sentence-transformer model for embedding."""
        try:
            from sentence_transformers import SentenceTransformer
            log.info("Loading embedding model: %s", model_name)
            return SentenceTransformer(model_name)
        except ImportError:
            raise ImportError("Run: pip install sentence-transformers")

    def _embed(self, texts: List[str]) -> np.ndarray:
        """Embed a list of texts → float32 numpy array."""
        embeddings = self._model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
            batch_size=64,
        )
        return embeddings.astype(np.float32)

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        """L2-normalize rows so inner product == cosine similarity."""
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)  # avoid division by zero
        return vectors / norms

    def _build_index(self, chunks: List[str]) -> "faiss.IndexFlatIP":
        """
        Embed all chunks and build a FAISS flat index.

        IndexFlatIP uses inner product — combined with L2 normalization
        this gives exact cosine similarity search.
        """
        try:
            import faiss
        except ImportError:
            raise ImportError("Run: pip install faiss-cpu")

        log.info("Embedding %d chunks…", len(chunks))
        embeddings = self._embed(chunks)
        embeddings = self._normalize(embeddings)

        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)

        log.info("FAISS index built: %d vectors, dim=%d", index.ntotal, dim)
        return index
