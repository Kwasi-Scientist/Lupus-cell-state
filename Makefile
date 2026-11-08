SHELL := /bin/bash
PYTHON ?= .venv/bin/python
PYTEST ?= .venv/bin/pytest
CONFIG ?= config.yaml

.PHONY: setup fetch subsample qc scdrs ifn figures report all test clean

setup:
	uv venv --python 3.12 .venv
	uv pip install -r requirements.txt

fetch:
	$(PYTHON) run_phase1.py --step fetch --config $(CONFIG)

subsample:
	$(PYTHON) run_phase1.py --step subsample --config $(CONFIG)

qc:
	$(PYTHON) run_phase1.py --step qc --config $(CONFIG)

scdrs:
	$(PYTHON) run_phase1.py --step scdrs --config $(CONFIG)

ifn:
	$(PYTHON) run_phase1.py --step ifn --config $(CONFIG)

figures:
	$(PYTHON) run_phase1.py --step figures --config $(CONFIG)

report:
	$(PYTHON) run_phase1.py --step report --config $(CONFIG)

all:
	$(PYTHON) run_phase1.py --step all --config $(CONFIG)

test:
	$(PYTEST) -q

clean:
	rm -rf outputs/ __pycache__/ .pytest_cache/ src/lupus_p1/__pycache__/ tests/__pycache__/
