PYTHON := .venv/bin/python

.PHONY: format lint test run-server run-client
format:
\t.venv/bin/black src tests
\t.venv/bin/isort src tests

lint:
\t.venv/bin/ruff check src tests

test:
\t.venv/bin/pytest

run-server:
\t$(PYTHON) -m src.flsys.server.main --rounds 5

run-client:
\tCLIENT_ID=$${CLIENT_ID:-0} $(PYTHON) -m src.flsys.client.main --client-id $$CLIENT_ID

