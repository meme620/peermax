PY := .venv/bin/python

.PHONY: env preflight smoke selftest test

env:
	uv venv --python 3.13 .venv
	uv pip install --python .venv/bin/python -r requirements.txt

preflight:
	$(PY) tools/preflight.py --level all

smoke:
	$(PY) tools/smoke_import.py

selftest:
	$(PY) -m pytest tools -q

test:
	$(PY) -m pytest
