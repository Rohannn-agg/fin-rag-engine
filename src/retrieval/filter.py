import re
from typing import Any, Dict, List, Optional, Tuple
from src.config import COMPANIES

COMPANY_ALIASES = {
    "apple": "AAPL", "aapl": "AAPL", "iphone": "AAPL",
    "microsoft": "MSFT", "msft": "MSFT", "azure": "MSFT", "windows": "MSFT",
    "google": "GOOGL", "alphabet": "GOOGL", "googl": "GOOGL", "goog": "GOOGL",
    "amazon": "AMZN", "amzn": "AMZN", "aws": "AMZN",
    "nvidia": "NVDA", "nvda": "NVDA",
    "meta": "META", "facebook": "META",
    "tesla": "TSLA", "tsla": "TSLA",
    "jpmorgan": "JPM", "jpm": "JPM", "chase": "JPM",
    "bank of america": "BAC", "bofa": "BAC", "bac": "BAC",
    "johnson & johnson": "JNJ", "jnj": "JNJ",
    "visa": "V",
    "walmart": "WMT", "wmt": "WMT"
}


class MetadataFilter:
    """Extracts metadata constraints from natural language questions and filters candidates."""

    @staticmethod
    def extract_filters(query: str) -> Dict[str, Any]:
        filters = {}
        query_lower = query.lower()

        # Company / Ticker extraction
        for alias, ticker in COMPANY_ALIASES.items():
            pattern = r"\b" + re.escape(alias) + r"\b"
            if re.search(pattern, query_lower):
                filters["ticker"] = ticker
                break

        # Fiscal Year extraction
        year_match = re.search(r"\b(20[12][0-9]|FY\s?20[12][0-9]|FY\s?[12][0-9])\b", query, flags=re.IGNORECASE)
        if year_match:
            raw_yr = year_match.group(1).upper().replace("FY", "").strip()
            if len(raw_yr) == 2:
                filters["fiscal_year"] = 2000 + int(raw_yr)
            else:
                filters["fiscal_year"] = int(raw_yr)

        # Document Type extraction
        if re.search(r"\b(10-?k|annual report)\b", query_lower):
            filters["doc_type"] = "10-K"
        elif re.search(r"\b(10-?q|quarterly report)\b", query_lower):
            filters["doc_type"] = "10-Q"
        elif re.search(r"\b(transcript|earnings call|call|qa session)\b", query_lower):
            filters["doc_type"] = "Transcript"

        return filters

    @staticmethod
    def filter_chunks(chunks: List[Dict[str, Any]], filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not filters:
            return chunks

        filtered = []
        for c in chunks:
            meta = c.get("metadata", {})
            match = True

            if "ticker" in filters and meta.get("ticker") != filters["ticker"]:
                match = False
            if "fiscal_year" in filters and meta.get("fiscal_year") != filters["fiscal_year"]:
                match = False
            if "doc_type" in filters and meta.get("doc_type") != filters["doc_type"]:
                match = False

            if match:
                filtered.append(c)

        # Fallback to unfiltered if overly strict filtering leaves nothing
        return filtered if filtered else chunks

