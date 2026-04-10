"""
pdf_parser.py

Parses SEC filing documents (HTM/HTML or PDF) into text chunks.
Handles both traditional HTML filings and modern iXBRL inline filings.
Preserves approximate page numbers on every chunk for citation enforcement.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf
from bs4 import BeautifulSoup


@dataclass
class Chunk:
    """A single text chunk with full provenance for citations."""
    text: str
    ticker: str
    form_type: str
    filing_date: str
    page_number: int
    chunk_index: int
    source_file: str
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "ticker": self.ticker,
            "form_type": self.form_type,
            "filing_date": self.filing_date,
            "page_number": self.page_number,
            "chunk_index": self.chunk_index,
            "source_file": self.source_file,
            **self.metadata,
        }


def _clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
    return text.strip()


def _split_into_chunks(
    text: str, chunk_size: int = 512, overlap: int = 64
) -> list[str]:
    """Word-based overlapping split."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        chunk = " ".join(words[start : start + chunk_size])
        if len(chunk.strip()) > 50:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def _extract_text_from_html(html: str) -> str:
    """
    Extract readable text from HTML or iXBRL filing.
    iXBRL filings use ix: namespace tags wrapping the actual text content.
    We parse with lxml when available for better namespace handling,
    falling back to html.parser.
    """
    # Try lxml first (better iXBRL handling), fall back to html.parser
    for parser in ["lxml", "html.parser"]:
        try:
            soup = BeautifulSoup(html, parser)
            break
        except Exception:
            continue

    # Remove noise tags
    for tag in soup(["script", "style", "meta", "link", "head"]):
        tag.decompose()

    # iXBRL documents wrap content in ix:nonNumeric and ix:nonFraction tags
    # BeautifulSoup treats these as regular tags - get_text() still works
    # but we need to make sure we're not just getting the outer shell
    text = soup.get_text(separator=" ", strip=True)
    text = _clean_text(text)
    return text


def parse_pdf(
    file_path: Path,
    ticker: str,
    form_type: str,
    filing_date: str,
    chunk_size: int = 512,
    overlap: int = 64,
) -> list[Chunk]:
    """Parse a PDF filing into chunks preserving page numbers."""
    doc = pymupdf.open(str(file_path))
    all_chunks: list[Chunk] = []
    chunk_index = 0

    for page_num, page in enumerate(doc, start=1):
        clean = _clean_text(page.get_text("text"))
        if not clean:
            continue
        for chunk_text in _split_into_chunks(clean, chunk_size, overlap):
            all_chunks.append(
                Chunk(
                    text=chunk_text,
                    ticker=ticker,
                    form_type=form_type,
                    filing_date=filing_date,
                    page_number=page_num,
                    chunk_index=chunk_index,
                    source_file=str(file_path),
                )
            )
            chunk_index += 1

    doc.close()
    print(f"[parser] {file_path.name}: {len(doc)} pages -> {len(all_chunks)} chunks")
    return all_chunks


def parse_htm(
    file_path: Path,
    ticker: str,
    form_type: str,
    filing_date: str,
    chunk_size: int = 512,
    overlap: int = 64,
) -> list[Chunk]:
    """
    Parse an HTM/HTML SEC filing (including iXBRL) into chunks.
    Approximates page numbers by segmenting every ~3000 words.
    """
    html = file_path.read_text(encoding="utf-8", errors="replace")
    full_text = _extract_text_from_html(html)

    if len(full_text.split()) < 100:
        print(f"[parser] WARNING: {file_path.name} extracted very little text ({len(full_text.split())} words). File may be a redirect or viewer page.")

    words = full_text.split()
    page_size_words = 3000
    all_chunks: list[Chunk] = []
    chunk_index = 0
    page_num = 0

    for page_start in range(0, len(words), page_size_words):
        page_num += 1
        page_text = " ".join(words[page_start : page_start + page_size_words])
        for chunk_text in _split_into_chunks(page_text, chunk_size, overlap):
            all_chunks.append(
                Chunk(
                    text=chunk_text,
                    ticker=ticker,
                    form_type=form_type,
                    filing_date=filing_date,
                    page_number=page_num,
                    chunk_index=chunk_index,
                    source_file=str(file_path),
                )
            )
            chunk_index += 1

    print(f"[parser] {file_path.name}: ~{page_num} pages -> {len(all_chunks)} chunks")
    return all_chunks


def parse_filing(
    file_path: Path,
    ticker: str,
    form_type: str,
    filing_date: str,
    chunk_size: int = 512,
    overlap: int = 64,
) -> list[Chunk]:
    """Route to the right parser based on file extension."""
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        return parse_pdf(file_path, ticker, form_type, filing_date, chunk_size, overlap)
    elif suffix in {".htm", ".html"}:
        return parse_htm(file_path, ticker, form_type, filing_date, chunk_size, overlap)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")


def parse_all_filings(
    filings: list[dict],
    chunk_size: int = 512,
    overlap: int = 64,
) -> list[Chunk]:
    """Parse a batch of filing dicts into a flat chunk list."""
    all_chunks: list[Chunk] = []
    for filing in filings:
        path = Path(filing["local_path"])
        if not path.exists():
            print(f"[parser] WARNING: not found: {path}")
            continue
        chunks = parse_filing(
            file_path=path,
            ticker=filing["ticker"],
            form_type=filing["form_type"],
            filing_date=filing["filing_date"],
            chunk_size=chunk_size,
            overlap=overlap,
        )
        all_chunks.extend(chunks)
    print(f"[parser] Total: {len(all_chunks)} chunks from {len(filings)} filings")
    return all_chunks
