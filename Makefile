.PHONY: help install run test test-verbose lint clean freeze

# Makefile for the Monte Carlo project.
# Run `make` or `make help` to see the available targets.
#
# NOTE: Makefile recipe lines must be indented with a TAB, never spaces.
# If you edit this file and get "missing separator", that is why.

PYTHON  := python3
VENV    := .venv
BIN     := $(VENV)/bin
SCRIPT  := monte_carlo.py

help:
	@echo "Available targets:"
	@echo "  make install       Create the venv and install dependencies"
	@echo "  make run           Run the simulation and print the results"
	@echo "  make test          Run the test suite quietly"
	@echo "  make test-verbose  Run the test suite with one line per test"
	@echo "  make lint          Byte-compile every file to catch syntax errors"
	@echo "  make freeze        Write exact installed versions to requirements.lock"
	@echo "  make clean         Remove the venv and all Python caches"

install:
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -r requirements.txt
	@echo ""
	@echo "Done. Activate with:  source $(BIN)/activate"

run:
	$(BIN)/python $(SCRIPT)

test:
	$(BIN)/pytest -q

test-verbose:
	$(BIN)/pytest -v

lint:
	$(BIN)/python -m compileall -q . && echo "No syntax errors."

freeze:
	$(BIN)/pip freeze > requirements.lock
	@echo "Wrote requirements.lock"

clean:
	rm -rf $(VENV) .pytest_cache requirements.lock
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	@echo "Cleaned."
