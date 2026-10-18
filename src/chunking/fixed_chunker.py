from typing import Any, Dict, List
from src.config import FIXED_CHUNK_OVERLAP, FIXED_CHUNK_SIZE


class FixedSizeChunker:
    """Baseline chunking: splits text into fixed token/word windows with sliding overlap."""

    def __init__(self, chunk_size: int = FIXED_CHUNK_SIZE, overlap: int = FIXED_CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_document(self, parsed_doc: Dict[str, Any]) -> List[Dict[str, Any]]:
        full_text = parsed_doc.get("full_text", "")
        metadata = parsed_doc.get("metadata", {})
        doc_id = parsed_doc.get("document_id", "doc")

        words = full_text.split()
        chunks = []
        step = max(1, self.chunk_size - self.overlap)

        chunk_idx = 0
        for i in range(0, len(words), step):
            chunk_words = words[i : i + self.chunk_size]
            if not chunk_words:
                continue

            text = " ".join(chunk_words)
            chunks.append({
                "chunk_id": f"{doc_id}_fixed_{chunk_idx}",
                "document_id": doc_id,
                "text": text,
                "metadata": {
                    **metadata,
                    "chunking_strategy": "fixed",
                    "chunk_index": chunk_idx,
                    "is_table": "| --- |" in text,
                    "section_title": "Unknown (Fixed Split)"
                }
            })
            chunk_idx += 1

        return chunks

