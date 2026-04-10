"""
tests/test_ingestion.py

Unit tests for ingestion logic - chunking, metadata fields, and index helpers.
No API keys needed. No file I/O beyond tmp files.
"""

from src.ingestion.pdf_parser import _clean_text, _split_into_chunks, Chunk


# -- Text cleaning -------------------------------------------------------------

def test_clean_text_collapses_whitespace():
    assert _clean_text("hello   world\n\t foo") == "hello world foo"


def test_clean_text_strips_control_chars():
    assert "\x00" not in _clean_text("clean\x00text")


def test_clean_text_strips_leading_trailing():
    assert _clean_text("  padded  ") == "padded"


def test_clean_text_empty_string():
    assert _clean_text("") == ""


# -- Chunking ------------------------------------------------------------------

def test_split_respects_chunk_size():
    words = ["word"] * 600
    text = " ".join(words)
    chunks = _split_into_chunks(text, chunk_size=100, overlap=10)
    for chunk in chunks:
        assert len(chunk.split()) <= 100


def test_split_produces_overlap():
    words = [f"w{i}" for i in range(200)]
    text = " ".join(words)
    chunks = _split_into_chunks(text, chunk_size=100, overlap=20)
    # The first word of chunk 2 should appear near the end of chunk 1
    chunk1_words = set(chunks[0].split())
    chunk2_words = set(chunks[1].split())
    overlap_words = chunk1_words & chunk2_words
    assert len(overlap_words) > 0


def test_split_skips_tiny_chunks():
    # A very short text shouldn't produce a chunk (< 50 chars threshold)
    chunks = _split_into_chunks("hi", chunk_size=512, overlap=64)
    assert chunks == []


def test_split_returns_list():
    chunks = _split_into_chunks("This is a normal sentence with enough words.", chunk_size=5, overlap=1)
    assert isinstance(chunks, list)


# -- Chunk dataclass -----------------------------------------------------------

def test_chunk_to_dict_has_required_keys():
    chunk = Chunk(
        text="Apple discloses supply chain risk.",
        ticker="AAPL",
        form_type="10-K",
        filing_date="2023-11-03",
        page_number=14,
        chunk_index=0,
        source_file="AAPL_10-K_2023-11-03.htm",
    )
    d = chunk.to_dict()
    for key in ["text", "ticker", "form_type", "filing_date", "page_number", "chunk_index", "source_file"]:
        assert key in d


def test_chunk_to_dict_values_correct():
    chunk = Chunk(
        text="Test text.",
        ticker="JPM",
        form_type="10-Q",
        filing_date="2024-05-01",
        page_number=3,
        chunk_index=7,
        source_file="JPM_10-Q_2024-05-01.htm",
    )
    d = chunk.to_dict()
    assert d["ticker"] == "JPM"
    assert d["page_number"] == 3
    assert d["chunk_index"] == 7


def test_chunk_metadata_merged_into_dict():
    chunk = Chunk(
        text="Revenue grew.",
        ticker="MSFT",
        form_type="10-K",
        filing_date="2023-07-27",
        page_number=5,
        chunk_index=1,
        source_file="MSFT_10-K_2023-07-27.htm",
        metadata={"custom_field": "custom_value"},
    )
    d = chunk.to_dict()
    assert d["custom_field"] == "custom_value"


# -- Index helpers -------------------------------------------------------------

def test_indexes_exist_returns_false_when_missing(tmp_path, monkeypatch):
    import src.ingestion.indexer as indexer
    monkeypatch.setattr(indexer, "INDEX_DIR", tmp_path)
    monkeypatch.setattr(indexer, "FAISS_INDEX_PATH", tmp_path / "faiss.index")
    monkeypatch.setattr(indexer, "BM25_INDEX_PATH", tmp_path / "bm25.pkl")
    monkeypatch.setattr(indexer, "METADATA_PATH", tmp_path / "metadata.json")
    assert indexer.indexes_exist() is False


def test_indexes_exist_returns_true_when_present(tmp_path, monkeypatch):
    import src.ingestion.indexer as indexer
    monkeypatch.setattr(indexer, "INDEX_DIR", tmp_path)
    faiss_path = tmp_path / "faiss.index"
    bm25_path = tmp_path / "bm25.pkl"
    meta_path = tmp_path / "metadata.json"
    monkeypatch.setattr(indexer, "FAISS_INDEX_PATH", faiss_path)
    monkeypatch.setattr(indexer, "BM25_INDEX_PATH", bm25_path)
    monkeypatch.setattr(indexer, "METADATA_PATH", meta_path)

    faiss_path.write_bytes(b"dummy")
    bm25_path.write_bytes(b"dummy")
    meta_path.write_text("[]")

    assert indexer.indexes_exist() is True
