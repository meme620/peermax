PY := .venv/bin/python

.PHONY: env doctor imports smoke lint spec gate selftest status run-all report test

env:
	uv venv --python 3.13 .venv
	uv pip install --python .venv/bin/python -r requirements.txt

doctor:
	$(PY) -m peerlab doctor

imports:
	$(PY) -m peerlab smoke --imports --clean

smoke:
	$(PY) -m peerlab smoke

spec:
	$(PY) -m peerlab spec

lint:
	$(PY) -m peerlab lint

gate:
	$(PY) -m peerlab gate

contracts:
	$(PY) -m peerlab data-contracts

selftest:
	$(PY) -m pytest tests/test_peerlab_*.py tests/test_script_utils.py tests/test_gate_*.py tests/test_data_contracts.py -q

status:
	$(PY) -m peerlab status

run-all:
	$(PY) -m peerlab run-all

report:
	$(PY) -m peerlab report

test:
	$(PY) -m pytest
