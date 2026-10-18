import json
import logging
import sys
import time
from pathlib import Path

from src.chunking.comparison import evaluate_chunking_comparison
from src.chunking.fixed_chunker import FixedSizeChunker
from src.chunking.section_chunker import SectionAwareChunker
from src.config import CHUNKS_DIR, EVAL_DIR, PARSED_DIR
from src.evaluation.benchmark import BenchmarkRunner
from src.ingestion.sec_data_loader import ensure_sample_filings_ingested
from src.retrieval.hybrid_engine import HybridRetrievalEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("PipelineRunner")


def run_full_pipeline():
    logger.info("=" * 70)
    logger.info("STAGE 1: Ingestion & Document Parsing (12 Public Companies)")
    logger.info("=" * 70)

    engine = HybridRetrievalEngine()
    chunking_comparison = ensure_sample_filings_ingested(engine)

    logger.info("=" * 70)
    logger.info("STAGE 2: Chunking Strategy Comparison (Ablation)")
    logger.info("=" * 70)
    print(json.dumps(chunking_comparison, indent=2))

    logger.info("=" * 70)
    logger.info("STAGE 3: Running Comprehensive Retrieval & Generation Ablation")
    logger.info("=" * 70)

    runner = BenchmarkRunner(engine)
    benchmark_data = runner.run_ablation_benchmark()

    print("\n" + "=" * 70)
    print("FINAL BENCHMARK RESULTS TABLE (FOR README)")
    print("=" * 70)
    print(benchmark_data["markdown_table"])
    print("=" * 70 + "\n")

    logger.info("Pipeline completed successfully! All artifacts saved to data/evaluation/.")
    return benchmark_data


if __name__ == "__main__":
    run_full_pipeline()

