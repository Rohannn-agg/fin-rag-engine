| Configuration | Recall@1 | Recall@3 | Recall@5 | MRR | NDCG@5 | Faithfulness | Correctness (F1) | Retrieval (ms) | Rerank (ms) | Total Latency (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BM25 Sparse Only | 0.450 | 0.725 | 0.775 | 0.588 | 0.302 | 100.0% | 0.239 | 0.1ms | 0.0ms | 0.2ms |
| Dense FAISS Only | 0.425 | 0.700 | 0.800 | 0.560 | 0.323 | 100.0% | 0.228 | 0.0ms | 0.0ms | 0.1ms |
| Hybrid (BM25 + Dense RRF) | 0.500 | 0.825 | 0.925 | 0.665 | 0.372 | 100.0% | 0.238 | 0.1ms | 0.0ms | 0.2ms |
| Hybrid + Cross-Encoder Rerank | 0.575 | 0.900 | 0.925 | 0.735 | 0.411 | 100.0% | 0.243 | 0.1ms | 0.0ms | 0.2ms |