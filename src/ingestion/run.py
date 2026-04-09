"""
run.py — Ingestion CLI

Fetches filings from SEC EDGAR, parses them into chunks,
and builds FAISS + BM25 indexes. Run this once before starting the API.

Usage:
    python -m src.ingestion.run
    python -m src.ingestion.run --tickers AAPL MSFT JPM --years 2023 2024
    python -m src.ingestion.run --force-redownload   # re-fetch existing files
"""

import argparse
import os
from dotenv import load_dotenv
from src.ingestion.edgar_fetcher import fetch_filings
from src.ingestion.pdf_parser import parse_all_filings
from src.ingestion.indexer import build_indexes, indexes_exist

load_dotenv()

DEFAULT_TICKERS = ["AAPL", "MSFT", "AMZN", "NVDA", "JPM"]
DEFAULT_YEARS = [2022, 2023, 2024]


def main():
    parser = argparse.ArgumentParser(description="Finance RAG ingestion pipeline")
    parser.add_argument(
        "--tickers", nargs="+", default=DEFAULT_TICKERS,
        help="Ticker symbols to ingest"
    )
    parser.add_argument(
        "--years", nargs="+", type=int, default=DEFAULT_YEARS,
        help="Filing years to ingest"
    )
    parser.add_argument(
        "--form-type", default="10-K",
        help="Filing form type (default: 10-K)"
    )
    parser.add_argument(
        "--force-reindex", action="store_true",
        help="Rebuild indexes even if they already exist"
    )
    parser.add_argument(
        "--force-redownload", action="store_true",
        help="Re-download filings even if local copies exist"
    )
    args = parser.parse_args()

    openai_key = os.environ.get("OPENAI_API_KEY")
    if not openai_key:
        raise EnvironmentError("OPENAI_API_KEY not set — check your .env file.")

    if indexes_exist() and not args.force_reindex and not args.force_redownload:
        print("[run] Indexes already exist. Use --force-reindex to rebuild.")
        return

    print(f"[run] Tickers: {args.tickers}")
    print(f"[run] Years:   {args.years}")
    print(f"[run] Form:    {args.form_type}")
    print()

    # Step 1 — Download
    filings = fetch_filings(
        tickers=args.tickers,
        form_type=args.form_type,
        years=args.years,
        force_redownload=args.force_redownload,
    )
    if not filings:
        print("[run] No filings downloaded. Exiting.")
        return

    # Step 2 — Parse
    chunks = parse_all_filings(filings)
    if not chunks:
        print("[run] No chunks produced. Check your filings.")
        return

    good_chunks = [c for c in chunks if len(c.text.split()) > 20]
    if len(good_chunks) < len(chunks) * 0.5:
        print(f"[run] WARNING: only {len(good_chunks)}/{len(chunks)} chunks have sufficient text.")
        print("[run] Downloaded files may be viewer pages. Try --force-redownload.")

    # Step 3 — Index
    build_indexes(chunks, openai_api_key=openai_key)
    print("\n[run] Ingestion complete. Start the API with: uvicorn src.api.main:app --reload")


if __name__ == "__main__":
    main()
