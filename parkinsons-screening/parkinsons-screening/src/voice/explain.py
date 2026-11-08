"""SHAP explainability for the voice model + per-feature physiological / artifact notes."""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.config import RESULTS_DIR, SEED
from src.voice.load import FEATURES, get_xyg, load_voice
from src.voice.models import build_model

OUT = RESULTS_DIR / "voice"

NOTES = {
    "MDVP:Fo(Hz)": ("pitch level", "Mean f0 depends strongly on sex and age, which UCI does not provide: confounder-prone."),
    "MDVP:Fhi(Hz)": ("pitch level", "Max f0: sex/age-dependent and sensitive to pitch-tracking errors."),
    "MDVP:Flo(Hz)": ("pitch level", "Min f0: sex/age-dependent and sensitive to pitch-tracking errors."),
    "MDVP:Jitter(%)": ("f0 perturbation", "Physiologically plausible (vocal-fold irregularity) but very sensitive to mic/noise/pitch tracking."),
    "MDVP:Jitter(Abs)": ("f0 perturbation", "Absolute jitter scales with pitch period, so it mixes dysphonia with pitch level."),
    "MDVP:RAP": ("f0 perturbation", "Short-term jitter; plausible but recording-sensitive."),
    "MDVP:PPQ": ("f0 perturbation", "Jitter variant; plausible but recording-sensitive."),
    "Jitter:DDP": ("f0 perturbation", "Mathematically 3x RAP: redundant with RAP."),
    "MDVP:Shimmer": ("amplitude perturbation", "Plausible (amplitude instability) but affected by mic gain/distance."),
    "MDVP:Shimmer(dB)": ("amplitude perturbation", "Shimmer in dB; same caveats."),
    "Shimmer:APQ3": ("amplitude perturbation", "Shimmer variant; same caveats."),
    "Shimmer:APQ5": ("amplitude perturbation", "Shimmer variant; same caveats."),
    "MDVP:APQ": ("amplitude perturbation", "Shimmer variant; same caveats."),
    "Shimmer:DDA": ("amplitude perturbation", "Mathematically 3x APQ3: redundant."),
    "NHR": ("noise ratio", "Noise-to-harmonics: reflects breathiness but also room/microphone noise (artifact-prone)."),
    "HNR": ("noise ratio", "Harmonics-to-noise: reflects voice quality but also recording noise (artifact-prone)."),
    "RPDE": ("nonlinear dynamics", "Recurrence period density entropy: plausibly captures irregular phonation; less interpretable."),
    "DFA": ("nonlinear dynamics", "Fractal scaling of the signal; hard to interpret clinically."),
    "spread1": ("nonlinear f0 variation", "Nonlinear f0-variation measure: plausibly dysphonia-related; strongly correlated with PPE."),
    "spread2": ("nonlinear f0 variation", "Nonlinear f0-variation measure; plausible, less interpretable."),
    "D2": ("nonlinear dynamics", "Correlation dimension; hard to interpret clinically."),
    "PPE": ("nonlinear f0 variation", "Pitch period entropy: designed to be robust to pitch-tracking errors and to normal f0 variation."),
}


def explain(model_name: str = "xgboost", top_k: int = 10) -> pd.DataFrame:
    """SHAP on a model fit to ALL recordings. This explains the deployed model, it is not an evaluation."""
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_voice(); X, y, _ = get_xyg(df)
    pipe = build_model(model_name, SEED).fit(X, y)
    Xs = pipe[:-1].transform(X)
    clf = pipe[-1]
    est = clf.model_ if hasattr(clf, "model_") else clf
    sv = shap.TreeExplainer(est).shap_values(Xs)
    sv = sv[1] if isinstance(sv, list) else (sv[..., 1] if getattr(sv, "ndim", 2) == 3 else sv)
    imp = np.abs(sv).mean(axis=0)
    order = np.argsort(-imp)
    shap.summary_plot(sv, Xs, feature_names=FEATURES, show=False, max_display=12)
    plt.title(f"SHAP summary ({model_name}, fit on all recordings)", fontsize=9); plt.tight_layout()
    plt.savefig(OUT / "voice_shap_summary.png", dpi=140, bbox_inches="tight"); plt.close()
    rows = [{"rank": r + 1, "feature": FEATURES[i], "mean_abs_shap": float(imp[i]), "family": NOTES[FEATURES[i]][0],
             "note": NOTES[FEATURES[i]][1]} for r, i in enumerate(order[:top_k])]
    res = pd.DataFrame(rows); res.to_csv(OUT / "voice_shap_top_features.csv", index=False)
    fam = res.head(5)["family"].value_counts().to_dict()
    (OUT / "voice_shap_summary.json").write_text(json.dumps({"top5_families": fam}, indent=2))
    print(res.head(8).to_string(index=False))
    return res


if __name__ == "__main__":
    explain()
