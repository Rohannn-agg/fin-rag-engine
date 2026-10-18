import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from src.chunking.comparison import evaluate_chunking_comparison
from src.chunking.fixed_chunker import FixedSizeChunker
from src.chunking.section_chunker import SectionAwareChunker
from src.config import CHUNKS_DIR, EVAL_DIR, PARSED_DIR
from src.ingestion.parser import FinancialDocumentParser
from src.ingestion.transcripts import EarningsTranscriptParser
from src.retrieval.hybrid_engine import HybridRetrievalEngine

logger = logging.getLogger(__name__)

# Complete real SEC 10-K & Transcript excerpts for 12 public companies
CORE_FINANCIAL_DOCS = [
    {
        "ticker": "AAPL",
        "company": "Apple Inc.",
        "fiscal_year": 2023,
        "doc_type": "10-K",
        "raw_html": """
        <h1>ITEM 1. BUSINESS</h1>
        <p>Apple Inc. designs, manufactures, and markets smartphones, personal computers, tablets, wearables, and accessories, and sells a variety of related services. Products include iPhone, Mac, iPad, and Wearables, Home and Accessories. Services include Advertising, AppleCare, Cloud Services, Digital Content, and Payment Services.</p>

        <h1>ITEM 1A. RISK FACTORS</h1>
        <p>Global economic conditions, geopolitical conflicts, and trade disputes could adversely affect the Company's business. In particular, substantially all of the Company's manufacturing is performed by outsourced partners located primarily in Asia, including mainland China, India, and Vietnam. Any disruption at these manufacturing facilities could severely limit supply.</p>

        <h1>ITEM 7. MANAGEMENT'S DISCUSSION AND ANALYSIS OF FINANCIAL CONDITION AND RESULTS OF OPERATIONS</h1>
        <p>Total net sales decreased 3% or $11.0 billion during 2023 compared to 2022. Products net sales decreased due to lower sales of Mac and iPad. Services net sales increased 9% driven by growth in advertising, App Store, and cloud services.</p>
        <p>Research and development expense was $29,915 million in 2023, compared to $26,251 million in 2022, representing an increase of 14% to support next-generation silicon, AI technologies, and visionOS platform development.</p>

        <h1>ITEM 8. FINANCIAL STATEMENTS AND SUPPLEMENTARY DATA</h1>
        <p>Consolidated Statements of Operations (in millions):</p>
        <table>
            <tr><th>Net Sales by Category</th><th>2023</th><th>2022</th><th>2021</th></tr>
            <tr><td>iPhone</td><td>$ 200,583</td><td>$ 205,489</td><td>$ 191,973</td></tr>
            <tr><td>Mac</td><td>$ 29,357</td><td>$ 40,177</td><td>$ 35,190</td></tr>
            <tr><td>iPad</td><td>$ 28,300</td><td>$ 29,292</td><td>$ 31,862</td></tr>
            <tr><td>Wearables, Home and Accessories</td><td>$ 39,845</td><td>$ 41,241</td><td>$ 38,367</td></tr>
            <tr><td>Services</td><td>$ 85,200</td><td>$ 78,129</td><td>$ 68,425</td></tr>
            <tr><td>Total net sales</td><td>$ 383,285</td><td>$ 394,328</td><td>$ 365,817</td></tr>
            <tr><td>Total cost of sales</td><td>$ 214,137</td><td>$ 223,546</td><td>$ 212,981</td></tr>
            <tr><td>Gross margin</td><td>$ 169,148</td><td>$ 170,782</td><td>$ 152,836</td></tr>
            <tr><td>Operating income</td><td>$ 114,301</td><td>$ 119,437</td><td>$ 108,949</td></tr>
        </table>
        """
    },
    {
        "ticker": "AAPL",
        "company": "Apple Inc.",
        "fiscal_year": 2023,
        "doc_type": "Transcript",
        "quarter": "Q3",
        "raw_text": """
        EXECUTIVE REMARKS & PRESENTATION:
        Tim Cook (CEO): Good afternoon, everyone. Today Apple is reporting revenue of $81.8 billion for the June quarter. We hit an all-time revenue record in India, growing strong double digits, driven by robust iPhone adoption and the opening of our first two retail stores in Mumbai and Delhi. We continue to invest heavily in artificial intelligence and machine learning, which are foundational technologies woven into virtually every product we build.

        Luca Maestri (CFO): Our Services revenue set an all-time record of $21.2 billion, up 8% year-over-year. We reached over 1 billion paid subscriptions across the services on our platform.

        QUESTION AND ANSWER SECTION:
        Shannon Cross (Analyst): Tim, can you discuss the trajectory of AI spending and silicon development?
        Tim Cook: We view AI and ML as fundamental core technologies, and we have been doing research across a wide range of AI technologies, including generative AI, for years.
        """
    },
    {
        "ticker": "MSFT",
        "company": "Microsoft Corp",
        "fiscal_year": 2023,
        "doc_type": "10-K",
        "raw_html": """
        <h1>ITEM 1. BUSINESS</h1>
        <p>Microsoft develops and supports software, services, devices, and solutions. Our three operating segments are Productivity and Business Processes, Intelligent Cloud, and More Personal Computing.</p>

        <h1>ITEM 7. MANAGEMENT'S DISCUSSION AND ANALYSIS</h1>
        <p>Revenue increased $13.7 billion or 7% driven by growth in Intelligent Cloud and Productivity and Business Processes. Operating income increased 6% to $88,523 million.</p>

        <h1>ITEM 8. FINANCIAL STATEMENTS</h1>
        <table>
            <tr><th>Segment Revenue (in millions)</th><th>2023</th><th>2022</th></tr>
            <tr><td>Productivity and Business Processes</td><td>$ 69,274</td><td>$ 63,364</td></tr>
            <tr><td>Intelligent Cloud</td><td>$ 87,907</td><td>$ 75,251</td></tr>
            <tr><td>More Personal Computing</td><td>$ 54,734</td><td>$ 59,579</td></tr>
            <tr><td>Total Revenue</td><td>$ 211,915</td><td>$ 198,194</td></tr>
            <tr><td>Operating income</td><td>$ 88,523</td><td>$ 83,383</td></tr>
            <tr><td>Net income</td><td>$ 72,361</td><td>$ 72,738</td></tr>
        </table>
        """
    },
    {
        "ticker": "MSFT",
        "company": "Microsoft Corp",
        "fiscal_year": 2023,
        "doc_type": "Transcript",
        "quarter": "Q4",
        "raw_text": """
        EXECUTIVE REMARKS & PRESENTATION:
        Satya Nadella (CEO): We had a tremendous close to the fiscal year. Microsoft Cloud annual revenue surpassed $110 billion, up 27% in constant currency. Azure OpenAI Service now has more than 11,00 organizations using the service, up from 4,500 just last quarter. Organizations are choosing Azure because of our leading infrastructure and enterprise-grade security.

        QUESTION AND ANSWER SECTION:
        Mark Murphy (Analyst): Satya, what is the impact of Copilot monetization?
        Satya Nadella: Copilot is being rapidly adopted across GitHub, Microsoft 365, and Power Platform. GitHub Copilot has over 1 million paying users.
        """
    },
    {
        "ticker": "NVDA",
        "company": "NVIDIA Corp",
        "fiscal_year": 2024,
        "doc_type": "10-K",
        "raw_html": """
        <h1>ITEM 1. BUSINESS</h1>
        <p>NVIDIA pioneered accelerated computing to help solve the most challenging computational problems. We specialize in GPUs, DPUs, and full-stack software platforms powering AI supercomputing.</p>

        <h1>ITEM 7. MANAGEMENT'S DISCUSSION AND ANALYSIS</h1>
        <p>Fiscal year 2024 was a transformative year. Total revenue increased 126% to $60,922 million, driven by exceptional growth in our Data Center platform following the enterprise inflection point for generative AI.</p>
        <p>Gross margin expanded significantly to 72.7% from 56.9% in fiscal 2023, reflecting higher Data Center architecture sales.</p>

        <h1>ITEM 8. FINANCIAL STATEMENTS</h1>
        <table>
            <tr><th>Revenue by Platform (in millions)</th><th>2024</th><th>2023</th></tr>
            <tr><td>Data Center</td><td>$ 47,525</td><td>$ 15,013</td></tr>
            <tr><td>Gaming</td><td>$ 10,447</td><td>$ 9,067</td></tr>
            <tr><td>Professional Visualization</td><td>$ 1,553</td><td>$ 1,544</td></tr>
            <tr><td>Automotive</td><td>$ 1,091</td><td>$ 903</td></tr>
            <tr><td>OEM and Other</td><td>$ 306</td><td>$ 447</td></tr>
            <tr><td>Total revenue</td><td>$ 60,922</td><td>$ 26,974</td></tr>
            <tr><td>Gross margin</td><td>72.7%</td><td>56.9%</td></tr>
            <tr><td>Operating income</td><td>$ 32,972</td><td>$ 4,224</td></tr>
        </table>
        """
    },
    {
        "ticker": "NVDA",
        "company": "NVIDIA Corp",
        "fiscal_year": 2024,
        "doc_type": "Transcript",
        "quarter": "Q4",
        "raw_text": """
        EXECUTIVE REMARKS & PRESENTATION:
        Jensen Huang (CEO): Accelerated computing and generative AI have reached the tipping point. Demand is surging worldwide across companies, industries and nations. Our Data Center platform is powered by increasingly diverse drivers — demand for data processing, training and inference from large cloud service providers and GPU specialized providers, as well as from enterprise software and consumer internet companies.

        QUESTION AND ANSWER SECTION:
        Toshiya Hari (Analyst): Jensen, how do you see Blackwell transition impacting supply?
        Jensen Huang: The demand for Blackwell is incredible. Supply will be constrained through the coming fiscal year as customer demand exceeds available capacity.
        """
    },
    {
        "ticker": "GOOGL",
        "company": "Alphabet Inc.",
        "fiscal_year": 2023,
        "doc_type": "10-K",
        "raw_html": """
        <h1>ITEM 1. BUSINESS</h1>
        <p>Alphabet is a collection of businesses — the largest of which is Google. Google segments include Google Services and Google Cloud.</p>

        <h1>ITEM 7. MANAGEMENT'S DISCUSSION AND ANALYSIS</h1>
        <p>Google Cloud demonstrated strong operating leverage, achieving operating income of $1,716 million in 2023 compared to an operating loss of $(1,864) million in 2022.</p>

        <h1>ITEM 8. FINANCIAL STATEMENTS</h1>
        <table>
            <tr><th>Revenues and Operating Income (in millions)</th><th>2023</th><th>2022</th></tr>
            <tr><td>Google Search & other</td><td>$ 175,032</td><td>$ 162,450</td></tr>
            <tr><td>YouTube ads</td><td>$ 31,510</td><td>$ 29,242</td></tr>
            <tr><td>Google Network</td><td>$ 31,315</td><td>$ 32,780</td></tr>
            <tr><td>Google advertising</td><td>$ 237,857</td><td>$ 224,472</td></tr>
            <tr><td>Google Services total</td><td>$ 272,504</td><td>$ 253,528</td></tr>
            <tr><td>Google Cloud</td><td>$ 33,088</td><td>$ 26,280</td></tr>
            <tr><td>Other Bets</td><td>$ 1,527</td><td>$ 1,068</td></tr>
            <tr><td>Total revenues</td><td>$ 307,394</td><td>$ 282,836</td></tr>
            <tr><td>Google Cloud operating income (loss)</td><td>$ 1,716</td><td>$ (1,864)</td></tr>
        </table>
        """
    },
    {
        "ticker": "AMZN",
        "company": "Amazon.com Inc.",
        "fiscal_year": 2023,
        "doc_type": "10-K",
        "raw_html": """
        <h1>ITEM 7. MANAGEMENT'S DISCUSSION AND ANALYSIS</h1>
        <p>Net sales increased 12% to $574,785 million in 2023. AWS segment sales were $90,764 million in 2023, compared to $80,096 million in 2022.</p>
        <p>North America operating income was $14,877 million in 2023, compared to an operating loss of $(2,847) million in 2022.</p>

        <h1>ITEM 8. FINANCIAL STATEMENTS</h1>
        <table>
            <tr><th>Segment Information (in millions)</th><th>2023</th><th>2022</th></tr>
            <tr><td>North America sales</td><td>$ 352,828</td><td>$ 315,880</td></tr>
            <tr><td>International sales</td><td>$ 131,193</td><td>$ 118,007</td></tr>
            <tr><td>AWS sales</td><td>$ 90,764</td><td>$ 80,096</td></tr>
            <tr><td>Total net sales</td><td>$ 574,785</td><td>$ 513,983</td></tr>
            <tr><td>Operating income</td><td>$ 36,852</td><td>$ 12,248</td></tr>
        </table>
        """
    },
    {
        "ticker": "META",
        "company": "Meta Platforms Inc.",
        "fiscal_year": 2023,
        "doc_type": "10-K",
        "raw_html": """
        <h1>ITEM 7. MANAGEMENT'S DISCUSSION AND ANALYSIS</h1>
        <p>In 2023, ad impressions delivered across our Family of Apps increased by 28% year-over-year. Reality Labs expenses were $17.97 billion, resulting in an operating loss of $(16,120) million.</p>

        <h1>ITEM 8. FINANCIAL STATEMENTS</h1>
        <table>
            <tr><th>Consolidated Financial Highlights (in millions)</th><th>2023</th><th>2022</th></tr>
            <tr><td>Family of Apps revenue</td><td>$ 133,015</td><td>$ 114,450</td></tr>
            <tr><td>Reality Labs revenue</td><td>$ 1,896</td><td>$ 2,159</td></tr>
            <tr><td>Total revenue</td><td>$ 134,902</td><td>$ 116,609</td></tr>
            <tr><td>Reality Labs operating loss</td><td>$ (16,120)</td><td>$ (13,717)</td></tr>
            <tr><td>Net income</td><td>$ 39,098</td><td>$ 23,200</td></tr>
        </table>
        """
    },
    {
        "ticker": "TSLA",
        "company": "Tesla Inc.",
        "fiscal_year": 2023,
        "doc_type": "10-K",
        "raw_html": """
        <h1>ITEM 7. MD&A</h1>
        <p>Automotive regulatory credits accounted for $1,790 million in 2023 compared to $1,607 million in 2022. Total automotive gross margin was 18.2% in 2023.</p>

        <h1>ITEM 8. FINANCIAL STATEMENTS</h1>
        <table>
            <tr><th>Revenues (in millions)</th><th>2023</th><th>2022</th></tr>
            <tr><td>Total automotive revenues</td><td>$ 82,419</td><td>$ 71,462</td></tr>
            <tr><td>Automotive regulatory credits</td><td>$ 1,790</td><td>$ 1,607</td></tr>
            <tr><td>Energy generation and storage</td><td>$ 6,035</td><td>$ 3,909</td></tr>
            <tr><td>Services and other</td><td>$ 8,319</td><td>$ 6,091</td></tr>
            <tr><td>Total revenues</td><td>$ 96,773</td><td>$ 81,462</td></tr>
        </table>
        """
    },
    {
        "ticker": "JPM",
        "company": "JPMorgan Chase & Co.",
        "fiscal_year": 2023,
        "doc_type": "10-K",
        "raw_html": """
        <h1>ITEM 8. FINANCIAL STATEMENTS</h1>
        <table>
            <tr><th>Consolidated Financial Results (in millions)</th><th>2023</th><th>2022</th></tr>
            <tr><td>Net interest income</td><td>$ 89,267</td><td>$ 66,710</td></tr>
            <tr><td>Noninterest revenue</td><td>$ 68,834</td><td>$ 61,913</td></tr>
            <tr><td>Total net revenue - managed</td><td>$ 162,368</td><td>$ 132,279</td></tr>
            <tr><td>Total assets</td><td>$ 3,875,393</td><td>$ 3,665,743</td></tr>
            <tr><td>Net income</td><td>$ 49,552</td><td>$ 37,676</td></tr>
        </table>
        """
    },
    {
        "ticker": "BAC",
        "company": "Bank of America Corp",
        "fiscal_year": 2023,
        "doc_type": "10-K",
        "raw_html": """
        <h1>ITEM 8. FINANCIAL STATEMENTS</h1>
        <table>
            <tr><th>Financial Highlights (in millions)</th><th>2023</th><th>2022</th></tr>
            <tr><td>Total revenue, net of interest expense</td><td>$ 98,581</td><td>$ 94,950</td></tr>
            <tr><td>Net income</td><td>$ 26,515</td><td>$ 27,528</td></tr>
        </table>
        """
    },
    {
        "ticker": "JNJ",
        "company": "Johnson & Johnson",
        "fiscal_year": 2023,
        "doc_type": "10-K",
        "raw_html": """
        <h1>ITEM 8. FINANCIAL STATEMENTS</h1>
        <table>
            <tr><th>Sales by Segment (in millions)</th><th>2023</th><th>2022</th></tr>
            <tr><td>Innovative Medicine</td><td>$ 54,759</td><td>$ 52,563</td></tr>
            <tr><td>MedTech</td><td>$ 30,400</td><td>$ 27,427</td></tr>
            <tr><td>Total sales</td><td>$ 85,159</td><td>$ 79,990</td></tr>
        </table>
        """
    },
    {
        "ticker": "V",
        "company": "Visa Inc.",
        "fiscal_year": 2023,
        "doc_type": "10-K",
        "raw_html": """
        <h1>ITEM 7. MD&A</h1>
        <p>During fiscal 2023, payments volume grew 9% on a constant-dollar basis, and cross-border volume excluding intra-Europe transactions grew 20%.</p>
        <h1>ITEM 8. FINANCIAL STATEMENTS</h1>
        <table>
            <tr><th>Operating Results (in millions)</th><th>2023</th><th>2022</th></tr>
            <tr><td>Net revenues</td><td>$ 32,653</td><td>$ 29,310</td></tr>
            <tr><td>Operating income</td><td>$ 21,957</td><td>$ 19,813</td></tr>
        </table>
        """
    },
    {
        "ticker": "WMT",
        "company": "Walmart Inc.",
        "fiscal_year": 2024,
        "doc_type": "10-K",
        "raw_html": """
        <h1>ITEM 7. MD&A</h1>
        <p>Global eCommerce sales increased 23% in fiscal 2024, surpassing $100 billion. Walmart U.S. comp sales grew 5.6%.</p>
        <h1>ITEM 8. FINANCIAL STATEMENTS</h1>
        <table>
            <tr><th>Revenues (in millions)</th><th>2024</th><th>2023</th></tr>
            <tr><td>Net sales</td><td>$ 642,637</td><td>$ 605,881</td></tr>
            <tr><td>Membership and other income</td><td>$ 5,488</td><td>$ 5,408</td></tr>
            <tr><td>Total revenues</td><td>$ 648,125</td><td>$ 611,289</td></tr>
            <tr><td>Operating income</td><td>$ 27,011</td><td>$ 20,428</td></tr>
        </table>
        """
    }
]


def ensure_sample_filings_ingested(retrieval_engine: HybridRetrievalEngine) -> Dict[str, Any]:
    """Ingests, parses, chunks, compares, and indexes financial documents."""
    parser = FinancialDocumentParser()
    transcript_parser = EarningsTranscriptParser()
    fixed_chunker = FixedSizeChunker()
    section_chunker = SectionAwareChunker()

    all_fixed_chunks = []
    all_section_chunks = []

    for doc_meta in CORE_FINANCIAL_DOCS:
        meta = {
            "ticker": doc_meta["ticker"],
            "company": doc_meta["company"],
            "fiscal_year": doc_meta["fiscal_year"],
            "doc_type": doc_meta["doc_type"]
        }

        if doc_meta["doc_type"] == "Transcript":
            parsed = transcript_parser.parse_transcript(doc_meta["raw_text"], meta)
        else:
            parsed = parser.parse_html_document(doc_meta["raw_html"], meta)

        fixed = fixed_chunker.chunk_document(parsed)
        section = section_chunker.chunk_document(parsed)

        all_fixed_chunks.extend(fixed)
        all_section_chunks.extend(section)

    # Save chunks to disk
    (CHUNKS_DIR / "fixed_chunks.json").write_text(json.dumps(all_fixed_chunks, indent=2), encoding="utf-8")
    (CHUNKS_DIR / "section_chunks.json").write_text(json.dumps(all_section_chunks, indent=2), encoding="utf-8")

    # Run Chunking Comparison Evaluation
    comparison_stats = evaluate_chunking_comparison(all_fixed_chunks, all_section_chunks)
    (EVAL_DIR / "chunking_comparison.json").write_text(json.dumps(comparison_stats, indent=2), encoding="utf-8")

    # Index section-aware chunks into hybrid search engine
    retrieval_engine.index(all_section_chunks)
    logger.info(f"Ingested {len(CORE_FINANCIAL_DOCS)} filings, indexed {len(all_section_chunks)} section-aware chunks.")

    return comparison_stats

