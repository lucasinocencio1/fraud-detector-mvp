
    export PYTHONPATH := $(shell pwd)
    PYTHON := python3
    
    .PHONY: setup data features train_sup train_unsup test serve

    setup:
	python3 -m venv .venv && . .venv/bin/activate && pip install -U pip && pip install -r requirements.txt

    data:
	python3 src/data/synth_data.py && python src/data/make_dataset.py

    features:
	python3 src/data/feature_build.py

    train_sup:
	python3 src/models/train_supervised.py

    train_unsup:
	python3 src/models/train_unsupervised.py

    test:
	pytest -q --maxfail=1 --disable-warnings

    serve:
	uvicorn src.serve.api:app --reload
