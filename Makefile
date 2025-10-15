export PYTHONPATH := $(shell pwd)
PYTHON := python3

.PHONY: setup data update features train_sup train_unsup evaluate test serve retrain

setup:
	$(PYTHON) -m venv .venv && . .venv/bin/activate && pip install -U pip && pip install -r requirements.txt

data:
	$(PYTHON) src/data/synth_data.py && $(PYTHON) src/data/make_dataset.py

update:
	$(PYTHON) src/data/update_data.py && $(PYTHON) src/data/make_dataset.py

features:
	$(PYTHON) src/data/feature_build.py

train_sup:
	$(PYTHON) src/models/train_supervised.py

evaluate:
	$(PYTHON) src/models/evaluate.py

test:
	pytest -q --maxfail=1 --disable-warnings

serve:
	uvicorn src.serve.api:app --reload

retrain:
	$(PYTHON) src/data/update_data.py && $(PYTHON) src/data/make_dataset.py && $(PYTHON) src/data/feature_build.py && $(PYTHON) src/models/train_supervised.py
