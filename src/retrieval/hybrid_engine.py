import time
from typing import Any, Dict, List, Optional, Tuple

from src.config import RRF_K, TOP_K_RERANK, TOP_K_RETRIEVAL
from src.retrieval.bm25_search import BM25SearchEngine
from src.retrieval.dense_search import DenseSearchEngine
from src.retrieval.filter import MetadataFilter
from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.rrf import reciprocal_rank_fusion


class HybridRetrievalEngine:
    def __init__(self):
        self.bm25_engine = BM25SearchEngine()
        self.dense_engine = DenseSearchEngine()
        self.reranker = CrossEncoderReranker()
        self.all_chunks: List[Dict[str, Any]] = []

    def index(self, chunks: List[Dict[str, Any]]) -> None:
        self.all_chunks = chunks
        self.bm25_engine.fit(chunks)
        self.dense_engine.build_index(chunks)

    def retrieve(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        mode: str = "hybrid_rerank",
        top_k: int = TOP_K_RERANK
    ) -> Dict[str, Any]:
        """
        Modes supported for ablation study:
        - 'bm25_only': BM25 sparse search
        - 'dense_only': FAISS dense embedding search
        - 'hybrid': BM25 + Dense merged with RRF
        - 'hybrid_rerank': Hybrid + Cross-Encoder reranking
        """
        start_time = time.perf_counter()

        # Step 1: Metadata Filtering
        active_filters = filters or MetadataFilter.extract_filters(query)
        filtered_chunks = MetadataFilter.filter_chunks(self.all_chunks, active_filters)

        # Build sub-indices if filtered pool is a subset
        if len(filtered_chunks) < len(self.all_chunks):
            sub_bm25 = BM25SearchEngine()
            sub_bm25.fit(filtered_chunks)
            bm25_searcher = sub_bm25

            sub_dense = DenseSearchEngine()
            sub_dense.build_index(filtered_chunks)
            dense_searcher = sub_dense
        else:
            bm25_searcher = self.bm25_engine
            dense_searcher = self.dense_engine

        retrieval_start = time.perf_counter()

        if mode == "bm25_only":
            candidates = bm25_searcher.search(query, top_k=top_k)
            retrieval_ms = (time.perf_counter() - retrieval_start) * 1000
            rerank_ms = 0.0
            final_results = candidates

        elif mode == "dense_only":
            candidates = dense_searcher.search(query, top_k=top_k)
            retrieval_ms = (time.perf_counter() - retrieval_start) * 1000
            rerank_ms = 0.0
            final_results = candidates

        elif mode == "hybrid":
            bm25_cands = bm25_searcher.search(query, top_k=TOP_K_RETRIEVAL)
            dense_cands = dense_searcher.search(query, top_k=TOP_K_RETRIEVAL)
            rrf_cands = reciprocal_rank_fusion(bm25_cands, dense_cands, rrf_k=RRF_K, top_n=top_k)
            retrieval_ms = (time.perf_counter() - retrieval_start) * 1000
            rerank_ms = 0.0
            final_results = rrf_cands

        else:  # hybrid_rerank (Default)
            bm25_cands = bm25_searcher.search(query, top_k=TOP_K_RETRIEVAL)
            dense_cands = dense_searcher.search(query, top_k=TOP_K_RETRIEVAL)
            rrf_cands = reciprocal_rank_fusion(bm25_cands, dense_cands, rrf_k=RRF_K, top_n=TOP_K_RETRIEVAL)
            retrieval_ms = (time.perf_counter() - retrieval_start) * 1000

            rerank_start = time.perf_counter()
            top_chunks = [c for c, _ in rrf_cands]
            reranked = self.reranker.rerank(query, top_chunks, top_k=top_k)
            rerank_ms = (time.perf_counter() - rerank_start) * 1000
            final_results = reranked

        total_ms = (time.perf_counter() - start_time) * 1000

        return {
            "query": query,
            "mode": mode,
            "applied_filters": active_filters,
            "results": [{"chunk": c, "score": round(score, 4)} for c, score in final_results],
            "latency": {
                "retrieval_ms": round(retrieval_ms, 2),
                "rerank_ms": round(rerank_ms, 2),
                "total_ms": round(total_ms, 2)
            }
        }

