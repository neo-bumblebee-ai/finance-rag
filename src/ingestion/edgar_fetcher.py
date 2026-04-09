"""
edgar_fetcher.py

Downloads 10-K and 10-Q filings from SEC EDGAR.
Handles the modern EDGAR filing structure where the primary document
is an inline XBRL (iXBRL) HTM file, NOT the XBRL viewer shell page.
"""

import time
import requests
from pathlib import Path
from bs4 import BeautifulSoup

HEADERS = {"User-Agent": "finance-rag-portfolio contact@example.com"}
EDGAR_BASE = "https://data.sec.gov"
SEC_BASE = "https://www.sec.gov"
SUPPORTED_FORMS = {"10-K", "10-Q"}

# These patterns indicate a viewer/shell page, not the real filing
VIEWER_PATTERNS = ["ixviewer", "xbrl-viewer", "viewer.htm", "ixv-"]


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
    url = f"{EDGAR_BASE}/submissions/CIK{cik}.json"
    resp = requests.get(url, headers=HEADERS, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    recent = data.get("filings", {}).get("recent", {})
    results = []
    for form, date, accession in zip(
        recent.get("form", []),
        recent.get("filingDate", []),
        recent.get("accessionNumber", []),
    ):
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


def _is_viewer_page(name: str) -> bool:
    """Return True if the filename looks like an XBRL viewer shell."""
    name_lower = name.lower()
    return any(p in name_lower for p in VIEWER_PATTERNS)


def get_primary_document_url(filing: dict) -> str | None:
    """
    Find the actual 10-K HTM document URL from the EDGAR filing index.

    Modern EDGAR filings have this structure in the index:
      - viewer.htm or R*.htm  -> XBRL viewer shell  (skip these)
      - aapl-20240928.htm     -> the real inline XBRL filing  (want this)

    Strategy: fetch the filing index JSON, find the HTM file that:
      1. Is NOT a viewer page
      2. Has the correct form type label OR is the largest HTM file
    """
    cik_int = filing["cik_int"]
    acc_clean = filing["acc_clean"]
    accession = filing["accession_number"]

    index_url = (
        f"{EDGAR_BASE}/Archives/edgar/data/{cik_int}/{acc_clean}/{accession}-index.json"
    )
    resp = requests.get(index_url, headers=HEADERS, timeout=10)

    candidates = []

    if resp.status_code == 200:
        try:
            items = resp.json().get("directory", {}).get("item", [])
            for item in items:
                name = item.get("name", "")
                item_type = item.get("type", "")
                size = int(item.get("size", 0) or 0)

                if not name.lower().endswith((".htm", ".html")):
                    continue
                if _is_viewer_page(name):
                    continue
                if "index" in name.lower():
                    continue

                url = f"{SEC_BASE}/Archives/edgar/data/{cik_int}/{acc_clean}/{name}"

                # Exact type match is top priority
                if item_type == filing["form_type"]:
                    return url

                candidates.append((size, url))

        except Exception as e:
            print(f"[edgar] JSON parse error for {accession}: {e}")

    # If no exact type match, return the largest non-viewer HTM
    if candidates:
        candidates.sort(reverse=True)
        return candidates[0][1]

    # Final fallback: scrape the HTM index page
    index_htm = (
        f"{SEC_BASE}/Archives/edgar/data/{cik_int}/{acc_clean}/{accession}-index.htm"
    )
    resp = requests.get(index_htm, headers=HEADERS, timeout=10)
    if resp.status_code != 200:
        return None

    soup = BeautifulSoup(resp.text, "html.parser")
    best = (0, None)
    for row in soup.find_all("tr"):
        cells = row.find_all("td")
        link = row.find("a", href=True)
        if not link:
            continue
        href = link["href"]
        if not href.lower().endswith((".htm", ".html")):
            continue
        if "index" in href.lower() or _is_viewer_page(href):
            continue

        # Prefer exact type match
        if len(cells) >= 4:
            if cells[3].get_text(strip=True) == filing["form_type"]:
                full = f"{SEC_BASE}{href}" if href.startswith("/") else href
                return full

        # Track largest by size column
        try:
            size = int(cells[-1].get_text(strip=True).replace(",", ""))
        except (ValueError, IndexError):
            size = 0
        if size > best[0]:
            full = f"{SEC_BASE}{href}" if href.startswith("/") else href
            best = (size, full)

    return best[1]


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
    """Download filings for a list of tickers. Returns metadata + local paths."""
    downloaded = []
    for ticker in tickers:
        print(f"[edgar] Resolving CIK for {ticker}...")
        try:
            cik = get_cik(ticker)
        except ValueError as e:
            print(f"[edgar] WARNING: {e}")
            continue

        print(f"[edgar] Fetching {form_type} for {ticker} (CIK {int(cik)})...")
        filings = get_filing_metadata(cik, form_type=form_type, years=years)
        if not filings:
            print(f"[edgar] No {form_type} filings found for {ticker} in {years}")
            continue

        for filing in filings:
            doc_url = get_primary_document_url(filing)
            if not doc_url:
                print(f"[edgar] No doc found for {filing['accession_number']}")
                continue

            filename = f"{ticker}_{filing['form_type']}_{filing['filing_date']}.htm"
            local_path = output_dir / ticker / filename

            if local_path.exists() and not force_redownload:
                print(f"[edgar] Already exists: {local_path.name}")
            else:
                print(f"[edgar] Downloading {filename}...")
                print(f"[edgar]   URL: {doc_url}")
                try:
                    download_filing(doc_url, local_path)
                    size_kb = local_path.stat().st_size // 1024
                    print(f"[edgar]   Saved: {size_kb} KB")
                except Exception as e:
                    print(f"[edgar] Download failed: {e}")
                    continue
                time.sleep(0.5)

            downloaded.append({
                **{k: v for k, v in filing.items() if k not in ("acc_clean",)},
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
