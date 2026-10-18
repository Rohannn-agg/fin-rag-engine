import json
import logging
import time
from typing import Any, Dict, List
import numpy as np

from src.config import EVAL_DIR
from src.evaluation.metrics import (
    compute_answer_correctness,
    compute_faithfulness,
    compute_mrr,
    compute_ndcg_at_k,
    compute_recall_at_k
)
from src.evaluation.test_dataset import get_evaluation_dataset
from src.generation.generator import FinancialAnswerGenerator
from src.retrieval.hybrid_engine import HybridRetrievalEngine

logger = logging.getLogger(__name__)


class BenchmarkRunner:
    def __init__(self, retrieval_engine: HybridRetrievalEngine):
        self.engine = retrieval_engine
        self.generator = FinancialAnswerGenerator()
        self.dataset = get_evaluation_dataset()

    def run_ablation_benchmark(self) -> Dict[str, Any]:
        """Runs evaluation over all 4 retrieval configurations and computes metrics."""
        modes = ["bm25_only", "dense_only", "hybrid", "hybrid_rerank"]
        benchmark_results = {}

        for mode in modes:
            logger.info(f"Running benchmark ablation for mode: {mode}...")
            recalls_1, recalls_3, recalls_5 = [], [], []
            mrrs, ndcgs = [], []
            faithfulness_scores = []
            correctness_scores = []
            retrieval_latencies = []
            rerank_latencies = []
            gen_latencies = []

            for item in self.dataset:
                q = item["question"]
                target_snippet = item["gold_passage_snippet"]
                target_sec = item["target_section"]
                is_neg = item["is_negative"]
                gt = item["ground_truth_answer"]

                # 1. Retrieval
                retrieval_res = self.engine.retrieve(q, mode=mode, top_k=5)
                chunks = retrieval_res["results"]
                raw_chunks = [c["chunk"] for c in chunks]

                retrieval_latencies.append(retrieval_res["latency"]["retrieval_ms"])
                rerank_latencies.append(retrieval_res["latency"]["rerank_ms"])

                # Metrics for non-negative queries
                if not is_neg:
                    recalls_1.append(compute_recall_at_k(chunks, target_snippet, target_sec, k=1))
                    recalls_3.append(compute_recall_at_k(chunks, target_snippet, target_sec, k=3))
                    recalls_5.append(compute_recall_at_k(chunks, target_snippet, target_sec, k=5))
                    mrrs.append(compute_mrr(chunks, target_snippet, target_sec))
                    ndcgs.append(compute_ndcg_at_k(chunks, target_snippet, target_sec, k=5))

                # 2. Generation
                gen_res = self.generator.generate_answer(q, raw_chunks)
                gen_latencies.append(gen_res["generation_ms"])

                faithfulness_scores.append(compute_faithfulness(gen_res["answer"], raw_chunks, is_neg))
                correctness_scores.append(compute_answer_correctness(gen_res["answer"], gt))

            benchmark_results[mode] = {
                "recall@1": round(float(np.mean(recalls_1)), 3),
                "recall@3": round(float(np.mean(recalls_3)), 3),
                "recall@5": round(float(np.mean(recalls_5)), 3),
                "mrr": round(float(np.mean(mrrs)), 3),
                "ndcg@5": round(float(np.mean(ndcgs)), 3),
                "faithfulness": round(float(np.mean(faithfulness_scores)), 3),
                "answer_correctness_f1": round(float(np.mean(correctness_scores)), 3),
                "latency_retrieval_ms": round(float(np.mean(retrieval_latencies)), 1),
                "latency_rerank_ms": round(float(np.mean(rerank_latencies)), 1),
                "latency_generation_ms": round(float(np.mean(gen_latencies)), 1),
                "latency_total_ms": round(float(np.mean(retrieval_latencies) + np.mean(rerank_latencies) + np.mean(gen_latencies)), 1)
            }

        # Save JSON
        save_file = EVAL_DIR / "benchmark_results.json"
        save_file.write_text(json.dumps(benchmark_results, indent=2), encoding="utf-8")

        # Format Markdown table
        md_table = self.generate_markdown_table(benchmark_results)
        (EVAL_DIR / "benchmark_table.md").write_text(md_table, encoding="utf-8")

        return {
            "results": benchmark_results,
            "markdown_table": md_table
        }

    def generate_markdown_table(self, results: Dict[str, Any]) -> str:
        headers = [
            "Configuration", "Recall@1", "Recall@3", "Recall@5", "MRR", "NDCG@5",
            "Faithfulness", "Correctness (F1)", "Retrieval (ms)", "Rerank (ms)", "Total Latency (ms)"
        ]
        rows = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |"
        ]

        mode_labels = {
            "bm25_only": "BM25 Sparse Only",
            "dense_only": "Dense FAISS Only",
            "hybrid": "Hybrid (BM25 + Dense RRF)",
            "hybrid_rerank": "Hybrid + Cross-Encoder Rerank"
        }

        for mode, data in results.items():
            row = [
                mode_labels.get(mode, mode),
                f"{data['recall@1']:.3f}",
                f"{data['recall@3']:.3f}",
                f"{data['recall@5']:.3f}",
                f"{data['mrr']:.3f}",
                f"{data['ndcg@5']:.3f}",
                f"{data['faithfulness'] * 100:.1f}%",
                f"{data['answer_correctness_f1']:.3f}",
                f"{data['latency_retrieval_ms']}ms",
                f"{data['latency_rerank_ms']}ms",
                f"{data['latency_total_ms']}ms"
            ]
            rows.append("| " + " | ".join(row) + " |")

        return "\n".join(rows)

