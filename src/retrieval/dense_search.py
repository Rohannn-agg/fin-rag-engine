import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

try:
    import faiss
except ImportError:
    faiss = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

from src.config import EMBEDDING_MODEL_NAME, INDEX_DIR

logger = logging.getLogger(__name__)


class DenseSearchEngine:
    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        self.model_name = model_name
        self.model = None
        self.index = None
        self.embeddings = None
        self.chunks: List[Dict[str, Any]] = []

    def load_model(self) -> None:
        if self.model is None and SentenceTransformer is not None:
            logger.info(f"Loading SentenceTransformer: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)

    def encode(self, texts: List[str]) -> np.ndarray:
        self.load_model()
        if self.model is not None:
            vectors = self.model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
            return np.array(vectors, dtype=np.float32)
        else:
            # Deterministic hash embedding fallback if model is unavailable
            dim = 384
            vecs = []
            for t in texts:
                np.random.seed(abs(hash(t)) % (2**32))
                v = np.random.randn(dim).astype(np.float32)
                v = v / (np.linalg.norm(v) + 1e-9)
                vecs.append(v)
            return np.array(vecs, dtype=np.float32)

    def build_index(self, chunks: List[Dict[str, Any]]) -> None:
        self.chunks = chunks
        texts = [c["text"] for c in chunks]
        self.embeddings = self.encode(texts)

        dim = self.embeddings.shape[1]
        if faiss is not None:
            self.index = faiss.IndexFlatIP(dim)
            self.index.add(self.embeddings)
            logger.info(f"FAISS index built with {len(chunks)} vectors of dim {dim}")
        else:
            logger.warning("FAISS not found, using NumPy cosine similarity fallback.")

    def search(self, query: str, top_k: int = 20) -> List[Tuple[Dict[str, Any], float]]:
        if not self.chunks or self.embeddings is None:
            return []

        q_vec = self.encode([query])

        if self.index is not None:
            scores, indices = self.index.search(q_vec, min(top_k, len(self.chunks)))
            results = []
            for idx, score in zip(indices[0], scores[0]):
                if idx >= 0:
                    results.append((self.chunks[idx], float(score)))
            return results
        else:
            # NumPy cosine similarity
            sims = np.dot(self.embeddings, q_vec.T).flatten()
            top_indices = np.argsort(sims)[::-1][:top_k]
            return [(self.chunks[i], float(sims[i])) for i in top_indices]

