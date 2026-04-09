.PHONY: install run ingest eval test lint clean

install:
	pip install -e ".[dev]"

ingest:
	python -m src.ingestion.run

run:
	uvicorn src.api.main:app --reload --port 8000

eval:
	python eval/run_ragas.py

test:
	pytest tests/ -v

lint:
	ruff check src/ eval/ tests/

clean:
	rm -rf data/index/
	find . -type d -name __pycache__ -exec rm -rf {} +
