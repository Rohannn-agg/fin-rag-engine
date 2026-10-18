import json
import pytest
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["indexed_chunks"] > 0


def test_query_grounded_answer_with_citations():
    payload = {
        "query": "What was Apple's R&D spend in fiscal 2023?",
        "mode": "hybrid_rerank",
        "top_k": 5
    }
    response = client.post("/query", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "29,915" in data["answer"]
    assert len(data["citations"]) > 0
    assert data["applied_filters"].get("ticker") == "AAPL"
    assert "latency" in data
    assert data["latency"]["total_e2e_ms"] > 0


def test_query_negative_hallucination_guard():
    payload = {
        "query": "What was Apple's revenue from commercial space rocket launches in 2023?",
        "mode": "hybrid_rerank",
        "top_k": 5
    }
    response = client.post("/query", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "Information not found in provided financial filings" in data["answer"]


def test_streaming_sse_endpoint():
    response = client.get("/query/stream?query=What+was+NVIDIA+gross+margin+in+fiscal+2024")
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]
    lines = [line if isinstance(line, str) else line.decode("utf-8") for line in response.iter_lines() if line]
    assert len(lines) > 0
    assert any("data:" in line for line in lines)


def test_evaluate_endpoint():
    response = client.get("/evaluate")
    assert response.status_code == 200
    data = response.json()
    assert "results" in data
    assert "hybrid_rerank" in data["results"]
    assert data["results"]["hybrid_rerank"]["recall@5"] >= 0.85
