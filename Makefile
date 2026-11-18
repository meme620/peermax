PY := .venv/bin/python

.PHONY: env preflight imports smoke selftest status run-all report test

env:
	uv venv --python 3.13 .venv
	uv pip install --python .venv/bin/python -r requirements.txt

preflight:
	$(PY) tools/preflight.py --level all

imports:
	$(PY) tools/smoke_import.py

smoke:
	$(PY) -m peerlab smoke

selftest:
	$(PY) -m pytest tools peerlab/tests -q

status:
	$(PY) -m peerlab status

run-all:
	$(PY) -m peerlab run-all

report:
	$(PY) -m peerlab report

test:
	$(PY) -m pytest
