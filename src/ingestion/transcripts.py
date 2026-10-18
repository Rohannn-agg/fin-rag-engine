import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List
from src.config import PARSED_DIR

logger = logging.getLogger(__name__)


class EarningsTranscriptParser:
    """Parses earnings call transcripts separating Prepared Executive Remarks and Analyst Q&A."""

    def __init__(self, output_dir: Path = PARSED_DIR):
        self.output_dir = output_dir

    def parse_transcript(self, raw_text: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Separates Executives Presentation vs Analyst Question and Answer session."""
        qa_split_patterns = [
            r"QUESTION\s+AND\s+ANSWER\s+SECTION",
            r"QUESTIONS\s+AND\s+ANSWERS",
            r"Q&A\s+SESSION",
            r"ANALYST\s+QUESTIONS"
        ]

        sections = []
        qa_index = -1
        for pattern in qa_split_patterns:
            match = re.search(pattern, raw_text, flags=re.IGNORECASE)
            if match:
                qa_index = match.start()
                break

        if qa_index != -1:
            executive_text = raw_text[:qa_index].strip()
            qa_text = raw_text[qa_index:].strip()

            sections.append({
                "section_id": "Section_ExecRemarks",
                "title": "Executive Remarks & Presentation",
                "content": executive_text,
                "tables_count": 0
            })
            sections.append({
                "section_id": "Section_AnalystQA",
                "title": "Analyst Q&A Session",
                "content": qa_text,
                "tables_count": 0
            })
        else:
            sections.append({
                "section_id": "Section_Transcript",
                "title": "Earnings Call Transcript",
                "content": raw_text.strip(),
                "tables_count": 0
            })

        doc_id = f"{metadata.get('ticker')}_Transcript_{metadata.get('quarter', 'Q3')}_{metadata.get('fiscal_year', 2023)}"
        parsed_doc = {
            "document_id": doc_id,
            "metadata": metadata,
            "sections": sections,
            "full_text": raw_text
        }

        save_path = self.output_dir / f"{doc_id}.json"
        save_path.write_text(json.dumps(parsed_doc, indent=2), encoding="utf-8")
        return parsed_doc

