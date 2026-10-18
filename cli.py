import argparse
import sys
from src.generation.generator import FinancialAnswerGenerator
from src.ingestion.sec_data_loader import ensure_sample_filings_ingested
from src.retrieval.hybrid_engine import HybridRetrievalEngine


def main():
    parser = argparse.ArgumentParser(description="LSEG Financial RAG CLI")
    parser.add_argument("--query", "-q", type=str, help="Question to ask financial filings")
    parser.add_argument("--mode", "-m", type=str, default="hybrid_rerank", choices=["bm25_only", "dense_only", "hybrid", "hybrid_rerank"])
    parser.add_argument("--benchmark", "-b", action="store_true", help="Run ablation benchmark evaluation")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive terminal Q&A loop")
    args = parser.parse_args()

    engine = HybridRetrievalEngine()
    print("Loading financial filings and initializing indices...")
    ensure_sample_filings_ingested(engine)
    generator = FinancialAnswerGenerator()

    if args.benchmark:
        from src.evaluation.benchmark import BenchmarkRunner
        print("Running ablation benchmark on 75 test questions...")
        runner = BenchmarkRunner(engine)
        res = runner.run_ablation_benchmark()
        print("\n=== ABLATION BENCHMARK RESULTS TABLE ===\n")
        print(res["markdown_table"])
        return

    if args.interactive:
        print("\n=== LSEG Financial Document RAG Interactive Shell ===")
        print("Type 'exit' or 'quit' to stop.\n")
        while True:
            try:
                user_q = input("Query > ").strip()
                if not user_q or user_q.lower() in ["exit", "quit"]:
                    break
                retrieval = engine.retrieve(user_q, mode=args.mode, top_k=5)
                raw_chunks = [item["chunk"] for item in retrieval["results"]]
                gen = generator.generate_answer(user_q, raw_chunks)

                print(f"\n[Filters Applied]: {retrieval['applied_filters']}")
                print(f"[Latency]: Retrieval: {retrieval['latency']['retrieval_ms']}ms | Rerank: {retrieval['latency']['rerank_ms']}ms | Gen: {gen['generation_ms']}ms")
                print(f"\n[Answer]:\n{gen['answer']}\n")
                if gen["citations"]:
                    print("[Citations]:")
                    for c in gen["citations"]:
                        print(f" - {c['doc']} | {c['section']}")
                print("-" * 60 + "\n")
            except (KeyboardInterrupt, EOFError):
                break
        return

    if args.query:
        retrieval = engine.retrieve(args.query, mode=args.mode, top_k=5)
        raw_chunks = [item["chunk"] for item in retrieval["results"]]
        gen = generator.generate_answer(args.query, raw_chunks)
        print(f"\nQuery: {args.query}")
        print(f"Answer: {gen['answer']}")
        print(f"Latency: {retrieval['latency']['total_ms'] + gen['generation_ms']}ms")
        return

    parser.print_help()


if __name__ == "__main__":
    main()

