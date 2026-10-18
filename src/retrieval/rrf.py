from typing import Any, Dict, List, Tuple


def reciprocal_rank_fusion(
    bm25_results: List[Tuple[Dict[str, Any], float]],
    dense_results: List[Tuple[Dict[str, Any], float]],
    rrf_k: int = 60,
    top_n: int = 20
) -> List[Tuple[Dict[str, Any], float]]:
    """Merges ranked lists from BM25 and Dense search using Reciprocal Rank Fusion."""
    scores: Dict[str, float] = {}
    chunk_map: Dict[str, Dict[str, Any]] = {}

    for rank, (chunk, _) in enumerate(bm25_results):
        cid = chunk["chunk_id"]
        chunk_map[cid] = chunk
        scores[cid] = scores.get(cid, 0.0) + (1.0 / (rrf_k + (rank + 1)))

    for rank, (chunk, _) in enumerate(dense_results):
        cid = chunk["chunk_id"]
        chunk_map[cid] = chunk
        scores[cid] = scores.get(cid, 0.0) + (1.0 / (rrf_k + (rank + 1)))

    sorted_cids = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    results = []
    for cid, score in sorted_cids[:top_n]:
        results.append((chunk_map[cid], score))

    return results

