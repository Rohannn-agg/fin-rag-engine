import json
import os
import re
import time
from typing import Any, AsyncGenerator, Dict, List, Optional
import requests

from src.generation.prompt import STRICT_FINANCIAL_PROMPT_TEMPLATE, format_context_blocks


class FinancialAnswerGenerator:
    """Generates grounded financial answers with citations and supports streaming."""

    def __init__(self, model_name: str = "gemini-flash"):
        self.model_name = model_name
        self.api_key = os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY")

    def generate_answer(self, question: str, retrieved_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        start_time = time.perf_counter()

        if not retrieved_chunks:
            return {
                "answer": "Information not found in provided financial filings.",
                "citations": [],
                "generation_ms": round((time.perf_counter() - start_time) * 1000, 2),
                "is_found": False
            }

        context_str = format_context_blocks(retrieved_chunks)
        prompt = STRICT_FINANCIAL_PROMPT_TEMPLATE.format(context_blocks=context_str, question=question)

        # 1. Check if Gemini / OpenAI API key is set
        if os.getenv("GEMINI_API_KEY"):
            answer = self._call_gemini(prompt)
        elif os.getenv("OPENAI_API_KEY"):
            answer = self._call_openai(prompt)
        else:
            answer = self._extractive_grounded_answer(question, retrieved_chunks)

        gen_ms = (time.perf_counter() - start_time) * 1000

        # Extract citations
        citation_matches = re.findall(r"\[Doc:\s*([^,]+),\s*Section:\s*([^,]+),\s*Chunk:\s*([^\]]+)\]", answer)
        citations = [{"doc": m[0].strip(), "section": m[1].strip(), "chunk": m[2].strip()} for m in citation_matches]

        is_found = "information not found" not in answer.lower()

        return {
            "answer": answer.strip(),
            "citations": citations,
            "generation_ms": round(gen_ms, 2),
            "is_found": is_found
        }

    async def stream_answer(self, question: str, retrieved_chunks: List[Dict[str, Any]]) -> AsyncGenerator[str, None]:
        """Streams answer tokens via Server-Sent Events (SSE)."""
        result = self.generate_answer(question, retrieved_chunks)
        full_text = result["answer"]

        # Stream words/tokens smoothly with simulated latency for realism
        import asyncio
        words = full_text.split(" ")
        for i, word in enumerate(words):
            chunk_data = json.dumps({"token": word + (" " if i < len(words) - 1 else ""), "done": False})
            yield f"data: {chunk_data}\n\n"
            await asyncio.sleep(0.015)

        done_data = json.dumps({
            "token": "",
            "done": True,
            "citations": result["citations"],
            "generation_ms": result["generation_ms"]
        })
        yield f"data: {done_data}\n\n"

    def _extractive_grounded_answer(self, question: str, chunks: List[Dict[str, Any]]) -> str:
        """Deterministic, grounded extractive answer synthesizer that generates citations."""
        # Expand common financial abbreviations
        normalized_q = re.sub(r"\br&d\b", "research and development", question, flags=re.IGNORECASE)
        normalized_q = re.sub(r"\bcapex\b", "capital expenditures", normalized_q, flags=re.IGNORECASE)

        stop_words = {
            "what", "was", "the", "in", "for", "and", "how", "much", "is", "of", "did",
            "were", "does", "from", "with", "during", "ended", "at", "to", "by", "or",
            "company", "fiscal", "year", "total", "net", "reported", "reporting", "spend",
            "revenue", "revenues", "sales", "income"
        }
        all_q_words = re.findall(r"\b[a-zA-Z0-9\.\%\$\-]{2,}\b", normalized_q.lower())
        q_tokens = set(all_q_words) - stop_words
        distinctive_tokens = {w for w in q_tokens if len(w) >= 3 and not w.isdigit()}

        # Verify that distinctive question terms exist in the retrieved context
        context_corpus = " ".join([c.get("text", "").lower() for c in chunks])
        if distinctive_tokens:
            missing_distinctive = [w for w in distinctive_tokens if w not in context_corpus]
            # If more than 50% of the distinctive subject terms are completely missing from the context, fail closed
            if len(missing_distinctive) > 0 and (len(missing_distinctive) / len(distinctive_tokens) >= 0.5):
                return "Information not found in provided financial filings."

        candidate_facts = []
        for c in chunks:
            text = c.get("text", "")
            cid = c.get("chunk_id", "chunk")
            did = c.get("document_id", "doc")
            sec = c.get("metadata", {}).get("section_title", "General")

            for line in text.split("\n"):
                line_str = line.strip()
                if not line_str or len(line_str) < 10:
                    continue
                overlap = sum(1 for t in q_tokens if t in line_str.lower())
                if overlap >= 2:
                    candidate_facts.append((overlap, line_str, did, sec, cid))

        if not candidate_facts:
            return "Information not found in provided financial filings."

        candidate_facts.sort(key=lambda x: x[0], reverse=True)
        top_facts = candidate_facts[:3]

        answer_parts = []
        for _, fact, did, sec, cid in top_facts:
            answer_parts.append(f"{fact} [Doc: {did}, Section: {sec}, Chunk: {cid}]")

        return " ".join(answer_parts)

    def _call_gemini(self, prompt: str) -> str:
        key = os.getenv("GEMINI_API_KEY")
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={key}"
        payload = {"contents": [{"parts": [{"text": prompt}]}]}
        try:
            resp = requests.post(url, json=payload, timeout=20)
            if resp.status_code == 200:
                data = resp.json()
                return data["candidates"][0]["content"]["parts"][0]["text"]
        except Exception:
            pass
        return "Information not found in provided financial filings."

    def _call_openai(self, prompt: str) -> str:
        key = os.getenv("OPENAI_API_KEY")
        url = "https://api.openai.com/v1/chat/completions"
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        payload = {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0
        }
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=20)
            if resp.status_code == 200:
                return resp.json()["choices"][0]["message"]["content"]
        except Exception:
            pass
        return "Information not found in provided financial filings."
