import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.config import EVAL_DIR, PARSED_DIR
from src.evaluation.benchmark import BenchmarkRunner
from src.generation.generator import FinancialAnswerGenerator
from src.retrieval.filter import MetadataFilter
from src.retrieval.hybrid_engine import HybridRetrievalEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

from contextlib import asynccontextmanager
from src.ingestion.sec_data_loader import ensure_sample_filings_ingested

# Global engine instances
retrieval_engine = HybridRetrievalEngine()
generator = FinancialAnswerGenerator()
benchmark_runner = BenchmarkRunner(retrieval_engine)

# Eagerly initialize indexed chunks
ensure_sample_filings_ingested(retrieval_engine)

@asynccontextmanager
async def lifespan(app: FastAPI):
    if len(retrieval_engine.all_chunks) == 0:
        ensure_sample_filings_ingested(retrieval_engine)
    yield

app = FastAPI(
    title="Financial Document Q&A RAG Engine",
    description="High-precision SEC 10-K/10-Q & Earnings Call RAG system with BM25, FAISS, RRF, and Cross-Encoder Reranking.",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    query: str = Field(..., example="What was Apple's R&D spend in 2023?")
    mode: str = Field("hybrid_rerank", description="bm25_only, dense_only, hybrid, hybrid_rerank")
    company: Optional[str] = Field(None, example="AAPL")
    fiscal_year: Optional[int] = Field(None, example=2023)
    doc_type: Optional[str] = Field(None, example="10-K")
    top_k: int = Field(5, ge=1, le=20)


class QueryResponse(BaseModel):
    query: str
    answer: str
    citations: List[Dict[str, Any]]
    retrieved_chunks: List[Dict[str, Any]]
    applied_filters: Dict[str, Any]
    latency: Dict[str, float]





@app.get("/health")
def health():
    return {
        "status": "healthy",
        "indexed_chunks": len(retrieval_engine.all_chunks),
        "embedding_model": retrieval_engine.dense_engine.model_name
    }


@app.post("/query", response_model=QueryResponse)
def query_financial_docs(req: QueryRequest):
    filters = {}
    if req.company:
        filters["ticker"] = req.company.upper()
    if req.fiscal_year:
        filters["fiscal_year"] = req.fiscal_year
    if req.doc_type:
        filters["doc_type"] = req.doc_type

    # 1. Retrieve
    retrieval_res = retrieval_engine.retrieve(
        query=req.query,
        filters=filters or None,
        mode=req.mode,
        top_k=req.top_k
    )

    raw_chunks = [item["chunk"] for item in retrieval_res["results"]]

    # 2. Generate
    gen_res = generator.generate_answer(req.query, raw_chunks)

    latency = {
        **retrieval_res["latency"],
        "generation_ms": gen_res["generation_ms"],
        "total_e2e_ms": round(retrieval_res["latency"]["total_ms"] + gen_res["generation_ms"], 2)
    }

    return QueryResponse(
        query=req.query,
        answer=gen_res["answer"],
        citations=gen_res["citations"],
        retrieved_chunks=retrieval_res["results"],
        applied_filters=retrieval_res["applied_filters"],
        latency=latency
    )


@app.get("/query/stream")
async def query_stream(query: str = Query(...), mode: str = Query("hybrid_rerank")):
    """Server-Sent Events (SSE) streaming endpoint."""
    retrieval_res = retrieval_engine.retrieve(query=query, mode=mode, top_k=5)
    raw_chunks = [item["chunk"] for item in retrieval_res["results"]]
    return StreamingResponse(
        generator.stream_answer(query, raw_chunks),
        media_type="text/event-stream"
    )


@app.post("/evaluate")
@app.get("/evaluate")
def run_evaluation():
    """Triggers benchmark ablation study across all 4 modes."""
    results = benchmark_runner.run_ablation_benchmark()
    return results


@app.get("/", response_class=HTMLResponse)
def serve_ui():
    ui_path = Path(__file__).parent / "static" / "index.html"
    if ui_path.exists():
        return ui_path.read_text(encoding="utf-8")
    return "<h1>Financial RAG Engine is Running. Access /docs for OpenAPI specifications.</h1>"
