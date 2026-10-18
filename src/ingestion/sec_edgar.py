import json
import logging
import re
import time
from pathlib import Path
from typing import Dict, List, Optional
import requests

from src.config import COMPANIES, RAW_DIR, SEC_USER_AGENT

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SEC_HEADERS = {
    "User-Agent": SEC_USER_AGENT,
    "Accept-Encoding": "gzip, deflate",
    "Host": "data.sec.gov"
}


class SECEdgarDownloader:
    def __init__(self, output_dir: Path = RAW_DIR):
        self.output_dir = output_dir
        self.session = requests.Session()
        self.session.headers.update(SEC_HEADERS)

    def get_company_filings_metadata(self, cik: str) -> Optional[dict]:
        padded_cik = cik.zfill(10)
        url = f"https://data.sec.gov/submissions/CIK{padded_cik}.json"
        try:
            time.sleep(0.12)  # Respect SEC 10 requests/sec limit
            resp = self.session.get(url, timeout=15)
            if resp.status_code == 200:
                return resp.json()
            logger.warning(f"Failed to fetch metadata for CIK {cik}: HTTP {resp.status_code}")
        except Exception as e:
            logger.error(f"Error connecting to SEC EDGAR for CIK {cik}: {e}")
        return None

    def download_filing(self, ticker: str, cik: str, accession_number: str, primary_doc: str, form: str, fiscal_year: int) -> Optional[Path]:
        acc_no_clean = accession_number.replace("-", "")
        url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_no_clean}/{primary_doc}"
        save_path = self.output_dir / f"{ticker}_{form}_{fiscal_year}_{primary_doc}"

        if save_path.exists() and save_path.stat().st_size > 1000:
            logger.info(f"Filing already cached: {save_path.name}")
            return save_path

        headers = {"User-Agent": SEC_USER_AGENT, "Host": "www.sec.gov"}
        try:
            time.sleep(0.15)
            resp = requests.get(url, headers=headers, timeout=25)
            if resp.status_code == 200:
                save_path.write_text(resp.text, encoding="utf-8", errors="replace")
                logger.info(f"Downloaded {ticker} {form} ({fiscal_year}) -> {save_path.name}")
                return save_path
            logger.warning(f"HTTP {resp.status_code} downloading {url}")
        except Exception as e:
            logger.error(f"Failed downloading {url}: {e}")
        return None

    def ingest_filings(self, target_forms: List[str] = ["10-K", "10-Q"], years: List[int] = [2023, 2024]) -> List[Path]:
        downloaded = []
        for ticker, meta in COMPANIES.items():
            cik = meta["cik"]
            logger.info(f"Checking filings for {ticker} (CIK: {cik})...")
            metadata = self.get_company_filings_metadata(cik)
            if not metadata or "filings" not in metadata or "recent" not in metadata["filings"]:
                continue

            recent = metadata["filings"]["recent"]
            forms = recent.get("form", [])
            acc_nums = recent.get("accessionNumber", [])
            filing_dates = recent.get("filingDate", [])
            primary_docs = recent.get("primaryDocument", [])

            found_forms = set()
            for i, form in enumerate(forms):
                if form in target_forms and form not in found_forms:
                    year = int(filing_dates[i].split("-")[0])
                    if year in years:
                        path = self.download_filing(
                            ticker=ticker,
                            cik=cik,
                            accession_number=acc_nums[i],
                            primary_doc=primary_docs[i],
                            form=form,
                            fiscal_year=year
                        )
                        if path:
                            downloaded.append(path)
                            found_forms.add(f"{form}_{year}")
                if len(found_forms) >= len(target_forms):
                    break
        return downloaded

