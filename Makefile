UV ?= uv
PYTHON ?= .venv/bin/python
.DEFAULT_GOAL := check

.PHONY: setup setup-llada build check check-python

.venv/bin/python:
	$(UV) venv --python 3.11 .venv

check-python: $(PYTHON)
	$(PYTHON) -c 'import sys; sys.exit("Select a virtualenv interpreter with PYTHON=...") if sys.prefix == sys.base_prefix else None'

setup: check-python
	$(UV) pip install --python $(PYTHON) -r requirements-dev.txt
	env -u CONDA_PREFIX VIRTUAL_ENV="$$($(PYTHON) -c 'import sys; print(sys.prefix)')" $(PYTHON) -m maturin develop --uv --release --manifest-path src/rust_dfa/Cargo.toml

setup-llada: check-python
	$(UV) pip install --python $(PYTHON) -r requirements-llada.txt pytest ruff
	env -u CONDA_PREFIX VIRTUAL_ENV="$$($(PYTHON) -c 'import sys; print(sys.prefix)')" $(PYTHON) -m maturin develop --uv --release --manifest-path src/rust_dfa/Cargo.toml

build: check-python
	env -u CONDA_PREFIX $(PYTHON) -m maturin build --release --manifest-path src/rust_dfa/Cargo.toml -i $(PYTHON)

check: check-python
	$(PYTHON) -m ruff check src tests
	$(PYTHON) -m ruff format --check src tests
	$(PYTHON) -m pytest
	cargo fmt --manifest-path src/rust_dfa/Cargo.toml --check
	env -u CONDA_PREFIX PYO3_PYTHON="$(abspath $(PYTHON))" cargo clippy --manifest-path src/rust_dfa/Cargo.toml -- -D warnings
	git diff --check
