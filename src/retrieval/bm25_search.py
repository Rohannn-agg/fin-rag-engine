import re
from typing import Any, Dict, List, Tuple

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None

from src.config import BM25_B, BM25_K1


def financial_tokenizer(text: str) -> List[str]:
    """Tokenizes text preserving financial abbreviations, tickers, metrics, and percentages."""
    text = text.lower()
    # Normalize currencies and percentages
    text = re.sub(r"\$(\d+)", r" \1 dollar ", text)
    text = re.sub(r"(\d+)%", r" \1 percent ", text)
    # Extract alphanumeric tokens, compound terms, and numbers
    tokens = re.findall(r"\b[a-z0-9\.\-\_]{2,}\b", text)
    return tokens


class BM25SearchEngine:
    def __init__(self, k1: float = BM25_K1, b: float = BM25_B):
        self.k1 = k1
        self.b = b
        self.bm25: Any = None
        self.chunks: List[Dict[str, Any]] = []

    def fit(self, chunks: List[Dict[str, Any]]) -> None:
        self.chunks = chunks
        corpus = [financial_tokenizer(c["text"]) for c in chunks]
        if BM25Okapi is not None:
            self.bm25 = BM25Okapi(corpus, k1=self.k1, b=self.b)
        else:
            self.bm25 = SimpleBM25(corpus, k1=self.k1, b=self.b)

    def search(self, query: str, top_k: int = 20) -> List[Tuple[Dict[str, Any], float]]:
        if not self.chunks or not self.bm25:
            return []

        tokens = financial_tokenizer(query)
        if not tokens:
            return []

        scores = self.bm25.get_scores(tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        results = []
        for i in top_indices:
            if scores[i] > 0.0:
                results.append((self.chunks[i], float(scores[i])))
        return results


class SimpleBM25:
    """Fallback BM25 implementation in pure Python if rank_bm25 is not installed."""
    def __init__(self, corpus: List[List[str]], k1: float = 1.5, b: float = 0.75):
        import math
        self.k1 = k1
        self.b = b
        self.corpus_size = len(corpus)
        self.doc_lengths = [len(doc) for doc in corpus]
        self.avg_dl = sum(self.doc_lengths) / max(1, self.corpus_size)
        self.doc_freqs: Dict[str, int] = {}
        self.corpus = corpus

        for doc in corpus:
            for term in set(doc):
                self.doc_freqs[term] = self.doc_freqs.get(term, 0) + 1

        self.idf: Dict[str, float] = {}
        for term, freq in self.doc_freqs.items():
            self.idf[term] = math.log(1 + (self.corpus_size - freq + 0.5) / (freq + 0.5))

    def get_scores(self, query_tokens: List[str]) -> List[float]:
        scores = [0.0] * self.corpus_size
        for term in query_tokens:
            if term not in self.idf:
                continue
            idf = self.idf[term]
            for doc_idx, doc in enumerate(self.corpus):
                tf = doc.count(term)
                if tf > 0:
                    score = idf * (tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * (self.doc_lengths[doc_idx] / self.avg_dl)))
                    scores[doc_idx] += score
        return scores

