.PHONY: install dev test validate

install:
	python3 -m venv .venv
	.venv/bin/pip install -r requirements.txt

dev:
	.venv/bin/uvicorn backend.main:app --reload --port 8000

test:
	.venv/bin/pytest -q

validate:
	.venv/bin/python scripts/validate_content.py

