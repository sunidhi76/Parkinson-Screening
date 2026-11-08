PY ?= python
.PHONY: setup data train evaluate leakage serve test lint docker notebook all
setup:      ## create venv and install pinned requirements
	$(PY) -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt
data:       ## download UCI voice data (add MIRROR=1 for the third-party GitHub fallback)
	$(PY) data/download_data.py $(if $(MIRROR),--mirror,)
leakage:    ## HEADLINE: random vs subject-wise vs LOSO on voice data
	$(PY) -m src.leakage_experiment
train:      ## voice EDA + comparison + calibration + export, SHAP, drawing pipeline (synthetic smoke if no real drawings)
	$(PY) -m src.voice.train && $(PY) -m src.voice.explain && $(PY) -m src.robustness && $(PY) -m src.drawing.train
evaluate:   ## re-run all experiments (alias of leakage + train)
	$(MAKE) leakage && $(MAKE) train
serve:      ## API + web demo at http://localhost:8000
	$(PY) -m uvicorn api.main:app --host 0.0.0.0 --port 8000
test:
	$(PY) -m pytest -q
lint:
	$(PY) -m ruff check src api tests data
notebook:
	$(PY) -m jupyter nbconvert --to notebook --execute notebooks/01_leakage_demo.ipynb --inplace
docker:
	docker compose up --build
all: data leakage train test
