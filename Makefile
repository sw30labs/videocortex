VENV := .venv
PY   := $(VENV)/bin/python

.PHONY: help venv install install-predict doctor test test-fast sample run deck clean

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN{FS=":.*?## "};{printf "  \033[36m%-16s\033[0m %s\n",$$1,$$2}'

venv: ## create a 3.11 virtualenv
	uv venv --python 3.11 $(VENV)

install: venv ## renderer only, no torch
	uv pip install --python $(PY) -e '.[dev]'

install-predict: venv ## + tribev2 and torch (large)
	uv pip install --python $(PY) -e '.[predict,dev]'

doctor: ## preflight this machine
	$(PY) -m videocortex doctor

test: ## full suite
	$(PY) -m pytest

test-fast: ## skip the tests that rasterise surfaces
	$(PY) -m pytest -m "not slow"

sample: ## build and render the synthetic example (no model needed)
	$(PY) examples/make_sample.py
	$(PY) -m videocortex draw examples/sample_run/predictions.npy --max-frames 6

run: ## venv + tests + synthetic sample
	./setup_and_run.sh

deck: ## loopback command deck (http://127.0.0.1:8730)
	$(PY) -m videocortex serve

clean:
	rm -rf runs .videocortex-cache .pytest_cache **/__pycache__ *.egg-info
