"""
edgar_fetcher.py

Downloads 10-K and 10-Q filings from SEC EDGAR.
Uses the submissions API to get filing metadata, then the filing index
to locate the primary document — handling both traditional HTM and
modern iXBRL viewer filings correctly.
"""

import time
import requests
from pathlib import Path
from bs4 import BeautifulSoup

HEADERS = {"User-Agent": "finance-rag-portfolio contact@example.com"}
EDGAR_BASE = "https://data.sec.gov"
SEC_BASE = "https://www.sec.gov"
SUPPORTED_FORMS = {"10-K", "10-Q"}


def get_cik(ticker: str) -> str:
    """Resolve a ticker symbol to a zero-padded 10-digit CIK."""
    url = f"{SEC_BASE}/files/company_tickers.json"
    resp = requests.get(url, headers=HEADERS, timeout=10)
    resp.raise_for_status()
    for entry in resp.json().values():
        if entry["ticker"].upper() == ticker.upper():
            return str(entry["cik_str"]).zfill(10)
    raise ValueError(f"Ticker '{ticker}' not found in EDGAR company list.")


def get_filing_metadata(
    cik: str,
    form_type: str = "10-K",
    years: list[int] | None = None,
) -> list[dict]:
    """Return filing metadata for a given CIK and form type."""
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
            "acc_clean": acc_clean,
            "filing_date": date,
            "cik": cik,
            "cik_int": int(cik),
            "form_type": form_type,
        })
    return results


def get_primary_document_url(filing: dict) -> str | None:
    """
    Find the actual 10-K document URL from the EDGAR filing index.

    Strategy:
    1. Fetch the filing index JSON from data.sec.gov
    2. Look for a document whose type matches the form (e.g. '10-K')
       and whose name ends in .htm
    3. Fall back to the largest .htm file if no type match found
    """
    cik_int = filing["cik_int"]
    acc_clean = filing["acc_clean"]
    accession = filing["accession_number"]

    # EDGAR filing index JSON
    index_url = f"{EDGAR_BASE}/Archives/edgar/data/{cik_int}/{acc_clean}/{accession}-index.json"
    resp = requests.get(index_url, headers=HEADERS, timeout=10)

    if resp.status_code == 200:
        try:
            data = resp.json()
            # The index JSON has a 'directory' key with 'item' list
            items = data.get("directory", {}).get("item", [])

            # Pass 1: exact form type match on .htm files
            for item in items:
                name = item.get("name", "")
                item_type = item.get("type", "")
                if item_type == filing["form_type"] and name.lower().endswith((".htm", ".html")):
                    return f"{SEC_BASE}/Archives/edgar/data/{cik_int}/{acc_clean}/{name}"

            # Pass 2: largest .htm file that isn't the index itself
            htm_items = [
                i for i in items
                if i.get("name", "").lower().endswith((".htm", ".html"))
                and "index" not in i.get("name", "").lower()
            ]
            if htm_items:
                # Sort by size descending — the real 10-K is always the biggest file
                htm_items.sort(key=lambda i: int(i.get("size", 0)), reverse=True)
                name = htm_items[0]["name"]
                return f"{SEC_BASE}/Archives/edgar/data/{cik_int}/{acc_clean}/{name}"

        except Exception as e:
            print(f"[edgar] JSON index parse error: {e}")

    # Final fallback: scrape the HTM index page
    index_htm = f"{SEC_BASE}/Archives/edgar/data/{cik_int}/{acc_clean}/{accession}-index.htm"
    resp = requests.get(index_htm, headers=HEADERS, timeout=10)
    if resp.status_code != 200:
        return None

    soup = BeautifulSoup(resp.text, "html.parser")
    best_link = None
    best_size = 0

    for row in soup.find_all("tr"):
        cells = row.find_all("td")
        link = row.find("a", href=True)
        if not link:
            continue
        href = link["href"]
        if not href.lower().endswith((".htm", ".html")):
            continue
        if "index" in href.lower():
            continue

        # Try to get size from the last cell
        try:
            size = int(cells[-1].get_text(strip=True).replace(",", "")) if cells else 0
        except ValueError:
            size = 0

        # Prefer type match
        if len(cells) >= 4:
            doc_type = cells[3].get_text(strip=True)
            if doc_type == filing["form_type"]:
                full_url = f"{SEC_BASE}{href}" if href.startswith("/") else href
                return full_url

        # Otherwise track largest
        if size > best_size:
            best_size = size
            best_link = f"{SEC_BASE}{href}" if href.startswith("/") else href

    return best_link


def download_filing(url: str, output_path: Path) -> Path:
    """Download a filing document to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(url, headers=HEADERS, timeout=120)
    resp.raise_for_status()
    output_path.write_bytes(resp.content)
    return output_path


def fetch_filings(
    tickers: list[str],
    form_type: str = "10-K",
    years: list[int] | None = None,
    output_dir: Path = Path("data/filings"),
    force_redownload: bool = False,
) -> list[dict]:
    """
    Download filings for a list of tickers and return metadata + local paths.
    Set force_redownload=True to re-fetch already downloaded files.
    """
    downloaded = []
    for ticker in tickers:
        print(f"[edgar] Resolving CIK for {ticker}...")
        try:
            cik = get_cik(ticker)
        except ValueError as e:
            print(f"[edgar] WARNING: {e}")
            continue

        print(f"[edgar] Fetching {form_type} filings for {ticker} (CIK {int(cik)})...")
        filings = get_filing_metadata(cik, form_type=form_type, years=years)
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

            if local_path.exists() and not force_redownload:
                print(f"[edgar] Already exists: {local_path.name}")
            else:
                print(f"[edgar] Downloading {filename} from {doc_url}...")
                try:
                    download_filing(doc_url, local_path)
                    size_kb = local_path.stat().st_size // 1024
                    print(f"[edgar] Saved {filename} ({size_kb} KB)")
                except Exception as e:
                    print(f"[edgar] Download failed: {e}")
                    continue
                time.sleep(0.5)

            downloaded.append({
                **{k: v for k, v in filing.items() if k != "acc_clean"},
                "ticker": ticker,
                "document_url": doc_url,
                "local_path": str(local_path),
            })

    print(f"[edgar] Done - {len(downloaded)} filing(s) ready.")
    return downloaded


if __name__ == "__main__":
    results = fetch_filings(
        tickers=["AAPL", "MSFT", "AMZN", "NVDA", "JPM"],
        form_type="10-K",
        years=[2022, 2023, 2024],
        force_redownload=True,
    )
    for r in results:
        print(r["local_path"])
