import logging
from typing import Any, Dict, List, Tuple

try:
    from sentence_transformers import CrossEncoder
except ImportError:
    CrossEncoder = None

from src.config import RERANKER_MODEL_NAME

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    def __init__(self, model_name: str = RERANKER_MODEL_NAME):
        self.model_name = model_name
        self.model = None

    def load_model(self) -> None:
        if self.model is None and CrossEncoder is not None:
            try:
                logger.info(f"Loading CrossEncoder: {self.model_name}")
                self.model = CrossEncoder(self.model_name)
            except Exception as e:
                logger.warning(f"Could not load CrossEncoder {self.model_name}: {e}")

    def rerank(self, query: str, candidates: List[Dict[str, Any]], top_k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        if not candidates:
            return []

        self.load_model()
        if self.model is not None:
            pairs = [[query, c["text"]] for c in candidates]
            scores = self.model.predict(pairs)
            scored = list(zip(candidates, [float(s) for s in scores]))
            scored.sort(key=lambda x: x[1], reverse=True)
            return scored[:top_k]
        else:
            # High-fidelity lexical overlap & term matching fallback
            scored = []
            q_terms = set(query.lower().split())
            for c in candidates:
                text_lower = c["text"].lower()
                matches = sum(1 for t in q_terms if t in text_lower)
                # Boost if matches occur in table or header
                score = matches + (1.5 if c["metadata"].get("is_table") else 0.0)
                scored.append((c, float(score)))
            scored.sort(key=lambda x: x[1], reverse=True)
            return scored[:top_k]

