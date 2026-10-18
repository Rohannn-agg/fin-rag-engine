import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PARSED_DIR = DATA_DIR / "parsed"
CHUNKS_DIR = DATA_DIR / "chunks"
INDEX_DIR = DATA_DIR / "indices"
EVAL_DIR = DATA_DIR / "evaluation"

for d in [DATA_DIR, RAW_DIR, PARSED_DIR, CHUNKS_DIR, INDEX_DIR, EVAL_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# SEC EDGAR Settings
SEC_USER_AGENT = os.getenv("SEC_USER_AGENT", "FinancialResearchGroup admin@financialrag.org")

# Public Companies (12 across Tech, Finance, Retail, Healthcare)
COMPANIES = {
    "AAPL": {"cik": "0000320193", "name": "Apple Inc.", "sector": "Technology"},
    "MSFT": {"cik": "0000789019", "name": "Microsoft Corp", "sector": "Technology"},
    "GOOGL": {"cik": "0001652044", "name": "Alphabet Inc.", "sector": "Technology"},
    "AMZN": {"cik": "0001018724", "name": "Amazon.com Inc.", "sector": "Consumer Discretionary"},
    "NVDA": {"cik": "0001045810", "name": "NVIDIA Corp", "sector": "Semiconductors"},
    "META": {"cik": "0001326801", "name": "Meta Platforms Inc.", "sector": "Technology"},
    "TSLA": {"cik": "0001318605", "name": "Tesla Inc.", "sector": "Automotive"},
    "JPM": {"cik": "0000019617", "name": "JPMorgan Chase & Co.", "sector": "Financial Services"},
    "BAC": {"cik": "0000070858", "name": "Bank of America Corp", "sector": "Financial Services"},
    "JNJ": {"cik": "0000200406", "name": "Johnson & Johnson", "sector": "Healthcare"},
    "V": {"cik": "0001403161", "name": "Visa Inc.", "sector": "Financial Services"},
    "WMT": {"cik": "0000104169", "name": "Walmart Inc.", "sector": "Consumer Staples"}
}

# Embedding & Cross-Encoder Models
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
RERANKER_MODEL_NAME = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

# Retrieval Parameters
BM25_K1 = 1.5
BM25_B = 0.75
RRF_K = 60
TOP_K_RETRIEVAL = 20
TOP_K_RERANK = 5

# Chunking Parameters
FIXED_CHUNK_SIZE = 512
FIXED_CHUNK_OVERLAP = 50
SECTION_CHUNK_MAX_TOKENS = 750
SECTION_CHUNK_MIN_TOKENS = 100

