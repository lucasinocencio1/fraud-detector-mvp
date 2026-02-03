export PYTHONPATH := $(shell pwd)
export DISABLE_PANDERA_IMPORT_WARNING := 1
PYTHON := python3

.PHONY: setup generate ingest data features train_sup evaluate test serve retrain

setup:
	$(PYTHON) -m venv .venv && . .venv/bin/activate && pip install -U pip && pip install -r requirements.txt

generate:
	$(PYTHON) -m src.data.gen_sample -n $(or $(ROWS),250000)

ingest:
	$(PYTHON) -m src.data.cli ingest $(or $(BATCH),data/sample_transactions.csv)

data:
	$(PYTHON) -m src.data.cli split

features: data
	$(PYTHON) -m src.data.cli features

train_sup: features
	$(PYTHON) src/models/train_supervised.py

evaluate:
	$(PYTHON) src/models/evaluate.py

test:
	pytest -q --maxfail=1 --disable-warnings

serve:
	uvicorn src.server.app:app --reload

retrain:
	$(PYTHON) -m src.data.cli full-run --batch-path $(or $(BATCH),data/sample_transactions.csv)
	$(PYTHON) src/models/train_supervised.py
