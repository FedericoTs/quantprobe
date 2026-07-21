# evo-compress Makefile.
#
# On Windows without GNU make, use the equivalent commands in the README
# (PowerShell). The venv python path is selected automatically per OS.

ifeq ($(OS),Windows_NT)
  VENVPY = .venv\Scripts\python.exe
else
  VENVPY = .venv/bin/python
endif

PY ?= python
DOMAIN ?= time-series
SEED ?= 0

.PHONY: setup data smoke evolve report test clean help

help:
	@echo "targets: setup data smoke evolve report test clean"
	@echo "  make setup            create .venv and install pinned deps"
	@echo "  make data             generate/fetch the default corpus ($(DOMAIN))"
	@echo "  make smoke            quick end-to-end spike (small budget)"
	@echo "  make evolve           full spike run (larger budget)"
	@echo "  make report           print the latest leaderboard"
	@echo "  make test             run the pytest suite"
	@echo "  make clean            remove caches, corpora, and results artifacts"

setup:
	$(PY) -m venv .venv
	$(VENVPY) -m pip install --upgrade pip
	$(VENVPY) -m pip install -r requirements.txt
	@echo "setup complete. activate with: .venv\\Scripts\\activate (Windows) or source .venv/bin/activate"

data:
	$(VENVPY) data/fetch_data.py --domain $(DOMAIN)

smoke:
	$(VENVPY) -m experiments.run_spike --domain $(DOMAIN) --engine ga \
		--population 40 --generations 20 --seed $(SEED) --objective max_ratio

evolve:
	$(VENVPY) -m experiments.run_spike --domain $(DOMAIN) --engine ga \
		--population 80 --generations 40 --seed $(SEED) --objective max_ratio

report:
	$(VENVPY) -c "import json,sys; d=json.load(open('results/results.json')); \
		print('domain:', d['meta'].get('domain')); \
		[print(f\"{r['name']:<22}{r['ratio']:>8.4f}  dec={r['decode_MBps']:>7.2f} MB/s  rt={r['roundtrip_ok']}\") for r in d['leaderboard']]"

test:
	$(VENVPY) -m pytest -q

clean:
	$(PY) -c "import shutil,glob,os; [shutil.rmtree(p,ignore_errors=True) for p in glob.glob('**/__pycache__',recursive=True)+['.pytest_cache','data/corpora']]; [os.remove(p) for p in glob.glob('results/*.png')+glob.glob('results/*.json') if os.path.exists(p)]"
