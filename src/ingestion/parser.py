import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional
from bs4 import BeautifulSoup

from src.config import PARSED_DIR

logger = logging.getLogger(__name__)

# Standard SEC 10-K / 10-Q Item regex patterns
ITEM_PATTERNS = [
    (r"ITEM\s+1\b[\.\:\s]+BUSINESS", "Item 1. Business"),
    (r"ITEM\s+1A\b[\.\:\s]+RISK\s+FACTORS", "Item 1A. Risk Factors"),
    (r"ITEM\s+1B\b[\.\:\s]+UNRESOLVED\s+STAFF\s+COMMENTS", "Item 1B. Unresolved Staff Comments"),
    (r"ITEM\s+2\b[\.\:\s]+PROPERTIES", "Item 2. Properties"),
    (r"ITEM\s+3\b[\.\:\s]+LEGAL\s+PROCEEDINGS", "Item 3. Legal Proceedings"),
    (r"ITEM\s+5\b[\.\:\s]+MARKET\s+FOR\s+REGISTRANT", "Item 5. Market for Common Equity"),
    (r"ITEM\s+7\b[\.\:\s]+MANAGEMENT'?S?\s+DISCUSSION\s+AND\s+ANALYSIS", "Item 7. MD&A"),
    (r"ITEM\s+7A\b[\.\:\s]+QUANTITATIVE\s+AND\s+QUALITATIVE", "Item 7A. Market Risk"),
    (r"ITEM\s+8\b[\.\:\s]+FINANCIAL\s+STATEMENTS", "Item 8. Financial Statements"),
    (r"ITEM\s+9\b[\.\:\s]+CONTROLS\s+AND\s+PROCEDURES", "Item 9. Controls and Procedures")
]


class FinancialDocumentParser:
    """Parses raw HTML/text filings into structured sections with markdown tables."""

    def __init__(self, output_dir: Path = PARSED_DIR):
        self.output_dir = output_dir

    def html_table_to_markdown(self, table_soup: BeautifulSoup) -> str:
        """Converts an HTML table into a clean markdown table string."""
        rows = table_soup.find_all("tr")
        if not rows:
            return ""

        table_data = []
        for row in rows:
            cols = [re.sub(r"\s+", " ", cell.get_text(strip=True)) for cell in row.find_all(["td", "th"])]
            # Filter empty rows
            if any(cols):
                table_data.append(cols)

        if not table_data:
            return ""

        max_cols = max(len(r) for r in table_data)
        normalized = [r + [""] * (max_cols - len(r)) for r in table_data]

        header = normalized[0]
        md_lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * max_cols) + " |"]
        for row in normalized[1:]:
            md_lines.append("| " + " | ".join(row) + " |")

        return "\n" + "\n".join(md_lines) + "\n"

    def parse_html_document(self, html_content: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Extracts text, preserving tables as markdown, and detects section headers."""
        soup = BeautifulSoup(html_content, "lxml")

        # Strip scripts and styles
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()

        # Convert tables to markdown placeholders
        tables = []
        for idx, table in enumerate(soup.find_all("table")):
            md_table = self.html_table_to_markdown(table)
            placeholder = f"[[FINANCIAL_TABLE_{idx}]]"
            table.replace_with(placeholder)
            tables.append((placeholder, md_table))

        raw_text = soup.get_text(separator="\n")
        # Clean redundant whitespaces
        cleaned_text = re.sub(r"\n{3,}", "\n\n", raw_text)

        # Restore tables
        for placeholder, md_table in tables:
            cleaned_text = cleaned_text.replace(placeholder, md_table)

        sections = self.split_by_sec_items(cleaned_text, metadata)

        parsed_doc = {
            "document_id": f"{metadata.get('ticker')}_{metadata.get('doc_type')}_{metadata.get('fiscal_year')}",
            "metadata": metadata,
            "sections": sections,
            "full_text": cleaned_text
        }

        save_file = self.output_dir / f"{parsed_doc['document_id']}.json"
        save_file.write_text(json.dumps(parsed_doc, indent=2), encoding="utf-8")
        return parsed_doc

    def split_by_sec_items(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Segments text into standard SEC filing items."""
        matches = []
        for pattern, section_title in ITEM_PATTERNS:
            for match in re.finditer(pattern, text, flags=re.IGNORECASE):
                matches.append((match.start(), section_title))

        matches.sort(key=lambda x: x[0])

        if not matches:
            return [{
                "section_id": "Section_Full",
                "title": "General Disclosures",
                "content": text.strip(),
                "tables_count": text.count("| --- |")
            }]

        sections = []
        # Pre-item section (prospectus / forward-looking statements)
        if matches[0][0] > 0:
            pre_content = text[:matches[0][0]].strip()
            if len(pre_content) > 100:
                sections.append({
                    "section_id": "Section_Intro",
                    "title": "Introductory Information",
                    "content": pre_content,
                    "tables_count": pre_content.count("| --- |")
                })

        for i in range(len(matches)):
            start_pos = matches[i][0]
            end_pos = matches[i+1][0] if i + 1 < len(matches) else len(text)
            title = matches[i][1]
            content = text[start_pos:end_pos].strip()

            sections.append({
                "section_id": f"Section_{i+1}",
                "title": title,
                "content": content,
                "tables_count": content.count("| --- |")
            })

        return sections

