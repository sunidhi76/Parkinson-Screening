# Parkinson's Screening

Leakage-aware evaluation of Parkinson's screening models, with a voice module, a spiral-drawing module, a FastAPI backend and a browser demo.

> Screening support, not diagnosis. Nothing in this repo is a medical device or clinical advice.

## Headline result: subject-level data leakage

The UCI Parkinson's voice dataset has 195 recordings from 32 subject codes (24 PD, 8 healthy), about 6 recordings per person. A random split puts the same person in both train and test (23 to 26 of 32 subjects per fold), so models partly learn to recognise voices. Splitting by subject gives a much more honest estimate.

| Model | Random split AUC | Subject-wise AUC | Random split bal. acc. | Subject-wise bal. acc. |
|---|---|---|---|---|
| Logistic regression | 0.894 | 0.762 | 0.770 | 0.652 |
| SVM (RBF) | 0.926 | 0.699 | 0.812 | 0.640 |
| Random forest | 0.952 | 0.778 | 0.833 | 0.627 |
| XGBoost | 0.957 | 0.786 | 0.872 | 0.681 |

Mean over 10 repeats of 5-fold CV. Leave-one-subject-out gives similar numbers (AUC 0.65 to 0.79).

![leakage chart](experiments/results/voice/leakage_chart.png)

Best subject-wise model (XGBoost), with 95% CIs from bootstrapping subjects: AUC 0.795 [0.631, 0.941], balanced accuracy 0.702 [0.566, 0.849], sensitivity 0.905, specificity 0.500. With only 8 healthy subjects, the intervals are wide.

## What's in the project

**Voice module** (real data, UCI)
- Subject-wise splits everywhere (StratifiedGroupKFold, leave-one-subject-out), each verified by tests.
- Four models (logistic regression, SVM, random forest, XGBoost) in sklearn Pipelines with class weighting. Preprocessing is fit inside each fold.
- Balanced accuracy, ROC-AUC, PR-AUC, sensitivity, specificity, Brier score and ECE, all with subject-level bootstrap CIs.
- Nested Platt and isotonic calibration, SHAP explanations, and noise-robustness curves.

**Drawing module** (pipeline only, no real results yet)
- Loaders for user-supplied HandPD and NewHandPD folders, with source-prefixed subject IDs.
- Handcrafted spiral features and a small CNN (with and without augmentation).
- A dataset-shortcut check, cross-dataset evaluation in both directions, noise robustness, and Grad-CAM with a stroke-versus-border check.
- A synthetic spiral generator runs the whole pipeline as a smoke test when no real data is present. Its outputs are stamped as synthetic.

**Voice and drawing are not fused.** They come from different people, so a combined score would be fabricated. There is no combined endpoint.

## Quick start

Python 3.12 was used. The pinned numpy 2.x and pandas 3 need 3.11 or later.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python data/download_data.py       # add --mirror if UCI is unreachable
python -m src.leakage_experiment   # headline result
python -m src.voice.train
python -m src.voice.explain
python -m src.robustness
python -m src.drawing.train        # synthetic smoke test unless real drawings are present

python -m uvicorn api.main:app --port 8000   # demo at http://localhost:8000
python -m pytest -q
```

`make` shortcuts: `make setup data leakage train serve test lint docker`.
Docker: `docker compose up --build` (mounts `./models`, so train first).

## Using real drawing data

HandPD and NewHandPD are not redistributed here. Request them from their owners (UNESP, Brazil), check the licences, then place spiral images like this:

```
data/drawings/handpd/{healthy,parkinson}/*.png
data/drawings/newhandpd/{healthy,parkinson}/*.png
data/drawings/subjects.csv        # filename,subject_id,source
```

`subjects.csv` matters when one person contributed several images. Without it, each image counts as its own subject and splits can leak people. Re-run `python -m src.drawing.train` and the results go to `experiments/results/drawing/`.

## API

| Endpoint | Purpose |
|---|---|
| `GET /health` | Status, loaded models, model provenance |
| `POST /predict/voice-features` | JSON with `row` (a pasted UCI row) or `features` (name to value) |
| `POST /predict/drawing` | Multipart image upload. Returns probability, risk band, retest flag and input-quality checks |

Uploads are processed in memory and never stored. Browser audio is analysed client-side, and only seven numbers are sent to the server.

## Repository layout

```
src/          splits, evaluate, calibration, robustness, leakage_experiment, synthetic
  voice/      load, models, train, explain
  drawing/    load, features, models, train, source_check, explain
api/          FastAPI app
web/          browser demo
tests/        split disjointness, loaders, features, API
notebooks/    01_leakage_demo.ipynb
experiments/results/   generated tables and figures
data/         download script, DATA_CARD.md
```

## Limitations

- **Small voice data.** 32 subjects (8 healthy) means wide intervals, poor specificity (0.35 to 0.50 at the 0.5 threshold), and unreliable calibration. There is no external validation set. NeuroVoz and mPower are the natural next step.
- **Confounders.** UCI has no age or sex metadata, and two of the top SHAP features (Fo, Fhi) are absolute pitch measures that depend on both.
- **Voice features can't be reproduced from a recording.** The 22 UCI features come from specialised software. The browser microphone demo approximates seven of them, and its output is not comparable to the training data.
- **The drawing model is trained on synthetic spirals only.** Its scores are meaningless for real or hand-drawn input, and a mouse-drawn spiral can easily score high. The feature extractor was tuned on synthetic images and is untested on real scans.
- **Model selection.** The default API model was chosen using the same CV that produced the reported numbers, which is mildly optimistic.

## Data and citation

Voice data: UCI Parkinson's dataset (CC BY 4.0). See `data/DATA_CARD.md` for subject counts and confounders.

- Little MA, McSharry PE, Roberts SJ, Costello DAE, Moroz IM. *Exploiting nonlinear recurrence and fractal scaling properties for voice disorder detection.* BioMedical Engineering OnLine, 2007.
- Little MA, McSharry PE, Hunter EJ, Ramig LO. *Suitability of dysphonia measurements for telemonitoring of Parkinson's disease.* IEEE Transactions on Biomedical Engineering, 2009.
