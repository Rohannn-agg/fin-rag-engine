import math
import re
from typing import Any, Dict, List, Tuple
import numpy as np


def compute_recall_at_k(retrieved_chunks: List[Dict[str, Any]], target_snippet: str, target_section: str, k: int) -> float:
    """Returns 1.0 if any of top-k chunks contains the target snippet or section, else 0.0."""
    top_k = retrieved_chunks[:k]
    snippet_clean = target_snippet.lower().strip()
    sec_clean = target_section.lower().strip()

    for item in top_k:
        chunk = item.get("chunk", item)
        text = chunk.get("text", "").lower()
        sec = chunk.get("metadata", {}).get("section_title", "").lower()

        if snippet_clean and snippet_clean in text:
            return 1.0
        if sec_clean and sec_clean in sec:
            return 1.0
    return 0.0


def compute_mrr(retrieved_chunks: List[Dict[str, Any]], target_snippet: str, target_section: str) -> float:
    """Computes Reciprocal Rank (1/rank) of the first relevant chunk."""
    snippet_clean = target_snippet.lower().strip()
    sec_clean = target_section.lower().strip()

    for rank, item in enumerate(retrieved_chunks, start=1):
        chunk = item.get("chunk", item)
        text = chunk.get("text", "").lower()
        sec = chunk.get("metadata", {}).get("section_title", "").lower()

        if (snippet_clean and snippet_clean in text) or (sec_clean and sec_clean in sec):
            return 1.0 / rank
    return 0.0


def compute_ndcg_at_k(retrieved_chunks: List[Dict[str, Any]], target_snippet: str, target_section: str, k: int) -> float:
    """Computes Normalized Discounted Cumulative Gain at rank K."""
    top_k = retrieved_chunks[:k]
    snippet_clean = target_snippet.lower().strip()
    sec_clean = target_section.lower().strip()

    dcg = 0.0
    for rank, item in enumerate(top_k, start=1):
        chunk = item.get("chunk", item)
        text = chunk.get("text", "").lower()
        sec = chunk.get("metadata", {}).get("section_title", "").lower()

        rel = 0.0
        if snippet_clean and snippet_clean in text:
            rel = 2.0  # High relevance for exact snippet match
        elif sec_clean and sec_clean in sec:
            rel = 1.0  # Partial relevance for section match

        if rel > 0:
            dcg += (2**rel - 1) / math.log2(rank + 1)

    # Ideal DCG for 1 highly relevant document at rank 1
    idcg = (2**2.0 - 1) / math.log2(1 + 1)
    return min(1.0, dcg / idcg) if idcg > 0 else 0.0


def compute_faithfulness(generated_answer: str, retrieved_chunks: List[Dict[str, Any]], is_negative: bool) -> float:
    """
    Checks if generated answer is strictly grounded in retrieved chunks.
    For negative queries, requires 'information not found'.
    For positive queries, verifies numbers and entities in answer exist in context.
    """
    ans_lower = generated_answer.lower()
    if is_negative:
        return 1.0 if "information not found" in ans_lower else 0.0

    if "information not found" in ans_lower:
        return 1.0

    # Extract numeric figures (e.g. $383,285, 29.9, 72.7%)
    numbers = re.findall(r"\b\d+(?:[\,\.]\d+)?%?\b", generated_answer)
    if not numbers:
        return 1.0

    context_text = " ".join([c.get("chunk", c).get("text", "") for c in retrieved_chunks])

    supported = sum(1 for n in numbers if n in context_text)
    return round(supported / len(numbers), 2)


def compute_answer_correctness(generated_answer: str, ground_truth: str) -> float:
    """Computes token-level precision/recall F1 score against ground-truth answer."""
    def tokenize(s: str):
        return set(re.findall(r"\b\w+\b", s.lower())) - {"the", "a", "an", "is", "was", "in", "of", "and"}

    pred_tokens = tokenize(generated_answer)
    gold_tokens = tokenize(ground_truth)

    if not gold_tokens:
        return 1.0 if not pred_tokens else 0.0

    common = pred_tokens.intersection(gold_tokens)
    if not common:
        return 0.0

    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gold_tokens)
    f1 = 2 * (precision * recall) / (precision + recall)
    return round(f1, 3)

