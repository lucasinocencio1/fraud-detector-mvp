export PYTHONPATH := $(shell pwd)
PYTHON := python3

.PHONY: setup data update features train_sup train_unsup evaluate test serve retrain

setup:
\t$(PYTHON) -m venv .venv && . .venv/bin/activate && pip install -U pip && pip install -r requirements.txt

data:
\t$(PYTHON) src/data/synth_data.py && $(PYTHON) src/data/make_dataset.py

update:
\t$(PYTHON) src/data/update_data.py && $(PYTHON) src/data/make_dataset.py

features:
\t$(PYTHON) src/data/feature_build.py

train_sup:
\t$(PYTHON) src/models/train_supervised.py

evaluate:
\t$(PYTHON) src/models/evaluate.py

test:
\tpytest -q --maxfail=1 --disable-warnings

serve:
\tuvicorn src.serve.api:app --reload

retrain:
\t$(PYTHON) src/data/update_data.py && $(PYTHON) src/data/make_dataset.py && $(PYTHON) src/data/feature_build.py && $(PYTHON) src/models/train_supervised.py
