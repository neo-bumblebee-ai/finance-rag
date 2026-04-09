"""
edgar_fetcher.py

Downloads 10-K and 10-Q filings from SEC EDGAR for a given list of tickers.
Uses the public EDGAR full-text search API — no authentication required.
Respects EDGAR's rate limit guidance (max 10 req/sec, we stay well below).
"""

import time
import requests
from pathlib import Path
from bs4 import BeautifulSoup

HEADERS = {"User-Agent": "finance-rag-portfolio contact@example.com"}
EDGAR_BASE = "https://data.sec.gov"
SUPPORTED_FORMS = {"10-K", "10-Q"}


def get_cik(ticker: str) -> str:
    """Resolve a ticker symbol to a zero-padded 10-digit CIK."""
    url = "https://www.sec.gov/files/company_tickers.json"
    resp = requests.get(url, headers=HEADERS, timeout=10)
    resp.raise_for_status()
    for entry in resp.json().values():
        if entry["ticker"].upper() == ticker.upper():
            return str(entry["cik_str"]).zfill(10)
    raise ValueError(f"Ticker '{ticker}' not found in EDGAR company list.")


def get_filing_urls(
    cik: str,
    form_type: str = "10-K",
    years: list[int] | None = None,
) -> list[dict]:
    """Return filing metadata dicts for a given CIK and form type."""
    if form_type not in SUPPORTED_FORMS:
        raise ValueError(f"form_type must be one of {SUPPORTED_FORMS}")

    url = f"{EDGAR_BASE}/submissions/CIK{cik}.json"
    resp = requests.get(url, headers=HEADERS, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    recent = data.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    dates = recent.get("filingDate", [])
    accessions = recent.get("accessionNumber", [])

    results = []
    for form, date, accession in zip(forms, dates, accessions):
        if form != form_type:
            continue
        if years and int(date[:4]) not in years:
            continue
        acc_clean = accession.replace("-", "")
        results.append({
            "accession_number": accession,
            "filing_date": date,
            "filing_index_url": (
                f"https://www.sec.gov/Archives/edgar/data/"
                f"{int(cik)}/{acc_clean}/{accession}-index.htm"
            ),
            "cik": cik,
            "form_type": form_type,
        })
    return results


def get_primary_document_url(filing: dict) -> str | None:
    """
    Find the URL of the primary HTM document by scraping the EDGAR
    filing index page. The index table lists all documents in a filing
    with their type in the 4th column.
    """
    index_url = filing["filing_index_url"]
    resp = requests.get(index_url, headers=HEADERS, timeout=10)
    if resp.status_code != 200:
        return None

    soup = BeautifulSoup(resp.text, "html.parser")

    # Primary pass: find row where Type column matches form_type exactly
    for row in soup.find_all("tr"):
        cells = row.find_all("td")
        if len(cells) >= 4:
            doc_type = cells[3].get_text(strip=True)
            if doc_type == filing["form_type"]:
                link = row.find("a", href=True)
                if link:
                    href = link["href"]
                    if href.endswith((".htm", ".html")):
                        if href.startswith("/"):
                            return f"https://www.sec.gov{href}"
                        return href

    # Fallback: first non-index HTM link in the table
    for row in soup.find_all("tr"):
        link = row.find("a", href=True)
        if link:
            href = link["href"]
            if href.endswith((".htm", ".html")) and "index" not in href.lower():
                if href.startswith("/"):
                    return f"https://www.sec.gov{href}"
                return href

    return None


def download_filing(url: str, output_path: Path) -> Path:
    """Download a filing document to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(url, headers=HEADERS, timeout=60)
    resp.raise_for_status()
    output_path.write_bytes(resp.content)
    return output_path


def fetch_filings(
    tickers: list[str],
    form_type: str = "10-K",
    years: list[int] | None = None,
    output_dir: Path = Path("data/filings"),
) -> list[dict]:
    """
    Download filings for a list of tickers and return metadata + local paths.

    Usage:
        results = fetch_filings(
            tickers=["AAPL", "MSFT", "JPM"],
            form_type="10-K",
            years=[2023, 2024],
        )
    """
    downloaded = []
    for ticker in tickers:
        print(f"[edgar] Resolving CIK for {ticker}...")
        try:
            cik = get_cik(ticker)
        except ValueError as e:
            print(f"[edgar] WARNING: {e}")
            continue

        print(f"[edgar] Fetching {form_type} filings for {ticker}...")
        filings = get_filing_urls(cik, form_type=form_type, years=years)
        if not filings:
            print(f"[edgar] No {form_type} filings found for {ticker} in {years}")
            continue

        for filing in filings:
            doc_url = get_primary_document_url(filing)
            if not doc_url:
                print(f"[edgar] Could not find primary doc for {filing['accession_number']}")
                continue

            filename = f"{ticker}_{filing['form_type']}_{filing['filing_date']}.htm"
            local_path = output_dir / ticker / filename

            if local_path.exists():
                print(f"[edgar] Already exists: {local_path.name}")
            else:
                print(f"[edgar] Downloading {filename}...")
                try:
                    download_filing(doc_url, local_path)
                except Exception as e:
                    print(f"[edgar] Download failed: {e}")
                    continue
                time.sleep(0.5)  # respect EDGAR rate limit

            downloaded.append({
                **filing,
                "ticker": ticker,
                "document_url": doc_url,
                "local_path": str(local_path),
            })

    print(f"[edgar] Done — {len(downloaded)} filing(s) ready.")
    return downloaded


if __name__ == "__main__":
    results = fetch_filings(
        tickers=["AAPL", "MSFT", "AMZN", "NVDA", "JPM"],
        form_type="10-K",
        years=[2023, 2024],
    )
    for r in results:
        print(r["local_path"])