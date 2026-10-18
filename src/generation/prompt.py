from typing import Any, Dict, List

STRICT_FINANCIAL_PROMPT_TEMPLATE = """You are a senior financial analyst and SEC compliance auditor.
Your task is to answer questions strictly and exclusively using the provided context chunks below.

RULES:
1. Grounding: Answer ONLY based on the facts directly stated in the context chunks. Do NOT make outside assumptions or extrapolate numbers.
2. Missing Information: If the provided context does not contain sufficient factual evidence to answer the question, you MUST output:
   "Information not found in provided financial filings."
3. Citations: Every single factual claim, revenue figure, margin, and percentage must be immediately followed by its bracketed citation, formatted exactly as:
   [Doc: <document_id>, Section: <section_title>, Chunk: <chunk_id>]
4. Tables: When citing financial tables, preserve exact monetary units (e.g. in millions, in billions) and reporting periods.

CONTEXT CHUNKS:
----------------------------------------
{context_blocks}
----------------------------------------

USER QUESTION:
{question}

DETAILED GROUNDED ANSWER (with inline citations):"""


def format_context_blocks(chunks: List[Dict[str, Any]]) -> str:
    blocks = []
    for c in chunks:
        cid = c.get("chunk_id", "chunk")
        did = c.get("document_id", "doc")
        meta = c.get("metadata", {})
        sec = meta.get("section_title", "General")
        text = c.get("text", "")
        block = f"[CITATION KEY -> Doc: {did}, Section: {sec}, Chunk: {cid}]\n{text}"
        blocks.append(block)
    return "\n\n".join(blocks)

