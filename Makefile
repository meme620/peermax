PY := .venv/bin/python

.PHONY: env doctor imports smoke lint selftest status run-all report test

env:
	uv venv --python 3.13 .venv
	uv pip install --python .venv/bin/python -r requirements.txt

doctor:
	$(PY) -m peerlab doctor

imports:
	$(PY) -m peerlab smoke --imports --clean

smoke:
	$(PY) -m peerlab smoke

lint:
	$(PY) -m peerlab lint

selftest:
	$(PY) -m pytest peerlab/tests -q

status:
	$(PY) -m peerlab status

run-all:
	$(PY) -m peerlab run-all

report:
	$(PY) -m peerlab report

test:
	$(PY) -m pytest
