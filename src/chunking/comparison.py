import json
import logging
from typing import Any, Dict, List
import numpy as np

logger = logging.getLogger(__name__)


def evaluate_chunking_comparison(fixed_chunks: List[Dict[str, Any]], section_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compares Fixed-Size vs Section-Aware chunking on structural coherence and table preservation."""

    def stats_for(chunks: List[Dict[str, Any]], name: str) -> Dict[str, Any]:
        lengths = [len(c["text"].split()) for c in chunks]
        tables = [c for c in chunks if c["metadata"].get("is_table", False)]

        # Check for severed tables: table syntax markers missing balanced rows or headers
        severed_tables = 0
        for c in chunks:
            text = c["text"]
            if "| --- |" in text:
                lines = [line.strip() for line in text.split("\n") if line.strip().startswith("|")]
                if len(lines) < 3:  # Header + separator + at least one data row
                    severed_tables += 1

        chunks_with_headers = sum(1 for c in chunks if "[" in c["text"] and "Section:" in c["text"])

        return {
            "strategy": name,
            "total_chunks": len(chunks),
            "avg_tokens": round(float(np.mean(lengths)), 1) if lengths else 0,
            "std_tokens": round(float(np.std(lengths)), 1) if lengths else 0,
            "min_tokens": int(np.min(lengths)) if lengths else 0,
            "max_tokens": int(np.max(lengths)) if lengths else 0,
            "table_chunks_count": len(tables),
            "severed_tables_count": severed_tables,
            "table_sever_rate_pct": round(severed_tables / max(1, len(tables)) * 100, 2),
            "header_attachment_rate_pct": round(chunks_with_headers / max(1, len(chunks)) * 100, 2)
        }

    return {
        "fixed_size_baseline": stats_for(fixed_chunks, "Fixed-Size (512 tokens / 50 overlap)"),
        "section_aware": stats_for(section_chunks, "Section-Aware + Atomic Tables")
    }

