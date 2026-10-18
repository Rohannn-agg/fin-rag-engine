import re
from typing import Any, Dict, List
from src.config import SECTION_CHUNK_MAX_TOKENS, SECTION_CHUNK_MIN_TOKENS


class SectionAwareChunker:
    """Section-aware chunker that preserves SEC Item headers and atomic financial tables."""

    def __init__(self, max_tokens: int = SECTION_CHUNK_MAX_TOKENS, min_tokens: int = SECTION_CHUNK_MIN_TOKENS):
        self.max_tokens = max_tokens
        self.min_tokens = min_tokens

    def _extract_blocks(self, section_text: str) -> List[Dict[str, Any]]:
        """Separates text into paragraphs and atomic markdown tables."""
        blocks = []
        lines = section_text.split("\n")
        in_table = False
        current_table = []
        current_para = []

        for line in lines:
            if line.strip().startswith("|") and ("|" in line.strip()[1:]):
                if not in_table:
                    if current_para:
                        text = "\n".join(current_para).strip()
                        if text:
                            blocks.append({"type": "text", "content": text})
                        current_para = []
                    in_table = True
                current_table.append(line)
            else:
                if in_table:
                    table_str = "\n".join(current_table).strip()
                    if table_str:
                        blocks.append({"type": "table", "content": table_str})
                    current_table = []
                    in_table = False
                current_para.append(line)

        if in_table and current_table:
            blocks.append({"type": "table", "content": "\n".join(current_table).strip()})
        elif current_para:
            text = "\n".join(current_para).strip()
            if text:
                blocks.append({"type": "text", "content": text})

        return blocks

    def chunk_document(self, parsed_doc: Dict[str, Any]) -> List[Dict[str, Any]]:
        doc_id = parsed_doc.get("document_id", "doc")
        metadata = parsed_doc.get("metadata", {})
        sections = parsed_doc.get("sections", [])

        ticker = metadata.get("ticker", "UNKNOWN")
        doc_type = metadata.get("doc_type", "Filing")
        year = metadata.get("fiscal_year", "N/A")

        chunks = []
        chunk_idx = 0

        for sec in sections:
            sec_title = sec.get("title", "General")
            sec_id = sec.get("section_id", "sec")
            blocks = self._extract_blocks(sec.get("content", ""))

            current_chunk_blocks = []
            current_tokens = 0

            header_prefix = f"[{ticker} | {doc_type} FY{year} | Section: {sec_title}]\n"
            header_tokens = len(header_prefix.split())

            for b in blocks:
                block_tokens = len(b["content"].split())

                # If an atomic table fits within max_tokens, treat it as a single chunk
                if b["type"] == "table":
                    if current_chunk_blocks:
                        combined_text = "\n\n".join([blk["content"] for blk in current_chunk_blocks])
                        chunks.append({
                            "chunk_id": f"{doc_id}_sec_{chunk_idx}",
                            "document_id": doc_id,
                            "text": f"{header_prefix}{combined_text}",
                            "metadata": {
                                **metadata,
                                "chunking_strategy": "section_aware",
                                "chunk_index": chunk_idx,
                                "section_id": sec_id,
                                "section_title": sec_title,
                                "is_table": any(blk["type"] == "table" for blk in current_chunk_blocks)
                            }
                        })
                        chunk_idx += 1
                        current_chunk_blocks = []
                        current_tokens = 0

                    # Output atomic table chunk
                    chunks.append({
                        "chunk_id": f"{doc_id}_sec_{chunk_idx}",
                        "document_id": doc_id,
                        "text": f"{header_prefix}Table:\n{b['content']}",
                        "metadata": {
                            **metadata,
                            "chunking_strategy": "section_aware",
                            "chunk_index": chunk_idx,
                            "section_id": sec_id,
                            "section_title": sec_title,
                            "is_table": True
                        }
                    })
                    chunk_idx += 1
                    continue

                # Standard text block accumulation
                if current_tokens + block_tokens + header_tokens > self.max_tokens and current_tokens >= self.min_tokens:
                    combined_text = "\n\n".join([blk["content"] for blk in current_chunk_blocks])
                    chunks.append({
                        "chunk_id": f"{doc_id}_sec_{chunk_idx}",
                        "document_id": doc_id,
                        "text": f"{header_prefix}{combined_text}",
                        "metadata": {
                            **metadata,
                            "chunking_strategy": "section_aware",
                            "chunk_index": chunk_idx,
                            "section_id": sec_id,
                            "section_title": sec_title,
                            "is_table": False
                        }
                    })
                    chunk_idx += 1
                    current_chunk_blocks = [b]
                    current_tokens = block_tokens
                else:
                    current_chunk_blocks.append(b)
                    current_tokens += block_tokens

            if current_chunk_blocks:
                combined_text = "\n\n".join([blk["content"] for blk in current_chunk_blocks])
                chunks.append({
                    "chunk_id": f"{doc_id}_sec_{chunk_idx}",
                    "document_id": doc_id,
                    "text": f"{header_prefix}{combined_text}",
                    "metadata": {
                        **metadata,
                        "chunking_strategy": "section_aware",
                        "chunk_index": chunk_idx,
                        "section_id": sec_id,
                        "section_title": sec_title,
                        "is_table": any(blk["type"] == "table" for blk in current_chunk_blocks)
                    }
                })
                chunk_idx += 1

        return chunks

