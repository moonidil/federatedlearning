PYTHON := .venv/bin/python

.PHONY: format lint test run-server run-client
format:
	.venv/bin/black src tests
	.venv/bin/isort src tests

lint:
	.venv/bin/ruff check src tests

test:
	.venv/bin/pytest

run-server:
	PYTHONPATH=src $(PYTHON) -m flsys.server.main --rounds 5

run-client:
	CLIENT_ID=$${CLIENT_ID:-0} PYTHONPATH=src $(PYTHON) -m flsys.client.main --client-id $$CLIENT_ID