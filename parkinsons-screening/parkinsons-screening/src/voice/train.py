"""Voice experiments: EDA, subject-wise model comparison, calibration, final artifact export."""
from __future__ import annotations

import json

import joblib
import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd

from src.calibration import nested_calibrated_oof, plot_reliability
from src.config import CV_REPEATS, DISCLAIMER, MODELS_DIR, RESULTS_DIR, SEED
from src.evaluate import (
    METRICS,
    cluster_bootstrap_ci,
    fmt_ci,
    per_fold_table,
    point_metrics,
    repeated_oof,
    summarize_repeats,
)
from src.splits import assert_disjoint_groups, make_splits
from src.voice.load import BASIC_FEATURES, FEATURES, get_xyg, load_voice, subject_summary
from src.voice.models import MODEL_NAMES, factory

OUT = RESULTS_DIR / "voice"


def eda() -> dict:
    """Print and save subject/class counts (never assume them)."""
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_voice(); s = subject_summary(df)
    s["subjects_listed"] = sorted(df["subject"].unique().tolist())
    s["note"] = ("UCI describes 31 people (23 PD); the file contains the number of distinct subject codes "
                 "reported here. Counts are measured, not assumed.")
    print(json.dumps({k: v for k, v in s.items() if k != "subjects_listed"}, indent=2))
    (OUT / "eda_summary.json").write_text(json.dumps(s, indent=2))
    return s


def compare_models(n_repeats: int = CV_REPEATS) -> pd.DataFrame:
    """Subject-wise only: repeated StratifiedGroupKFold (mean±std over repeats, pooled-OOF metrics with
    subject-level bootstrap 95% CIs) and Leave-One-Subject-Out. Also a 7-feature model for the browser demo."""
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_voice(); X, y, g = get_xyg(df)
    Xb, _, _ = get_xyg(df, BASIC_FEATURES)
    seeds = [SEED + i for i in range(n_repeats)]
    rows, fold_rows = [], []
    configs = [(n, FEATURES, X) for n in MODEL_NAMES] + [(f"{n}__basic7", BASIC_FEATURES, Xb) for n in ("logreg", "xgboost")]
    for label, feats, XX in configs:
        mname = label.split("__")[0]
        for kind, sd in (("group_cv", seeds), ("logo", [SEED])):
            k = "group" if kind == "group_cv" else "logo"
            for s in sd:
                assert_disjoint_groups(g, make_splits(k, y, g, s))
            P = repeated_oof(factory(mname), XX, y, g, k, sd)
            summ = summarize_repeats(y, P)
            ci = cluster_bootstrap_ci(y, P.mean(axis=0), g)
            row = {"model": label, "scheme": kind, "n_features": len(feats), "n_repeats": len(sd)}
            for m in METRICS:
                row[f"{m}_mean"], row[f"{m}_std"] = summ[m]
                row[f"{m}_ci"] = fmt_ci(ci.loc[m])
            rows.append(row)
            if kind == "group_cv":
                sp = make_splits("group", y, g, seeds[0])
                pf = per_fold_table(y, P[0], sp); pf.insert(0, "model", label); fold_rows.append(pf)
            print(f"{label:22s} {kind:8s} balAcc {row['balanced_accuracy_ci']:>24s}  AUC {row['roc_auc_ci']}")
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "voice_model_comparison.csv", index=False)
    pd.concat(fold_rows).to_csv(OUT / "voice_per_fold.csv", index=False)
    return res


def calibration_experiment(n_repeats: int = 3) -> pd.DataFrame:
    """Raw vs Platt vs isotonic, nested and subject-wise. ECE/Brier mean±std over repeats."""
    df = load_voice(); X, y, g = get_xyg(df)
    rows, curves_for = [], {}
    for name in MODEL_NAMES:
        acc = {"raw": [], "platt": [], "isotonic": []}; first = {}
        for r in range(n_repeats):
            sp = make_splits("group", y, g, SEED + r)
            for method in ("platt", "isotonic"):
                raw, cal = nested_calibrated_oof(factory(name), X, y, g, sp, method, seed=SEED + r)
                acc[method].append(point_metrics(y, cal))
                if method == "platt":
                    acc["raw"].append(point_metrics(y, raw))
                if r == 0:
                    first["raw"] = raw; first[method] = cal
        for variant, lst in acc.items():
            d = pd.DataFrame(lst)
            ci = cluster_bootstrap_ci(y, first[variant], g, n_boot=500)
            rows.append({"model": name, "variant": variant,
                         "ece_mean": d.ece.mean(), "ece_std": d.ece.std(ddof=1),
                         "brier_mean": d.brier.mean(), "brier_std": d.brier.std(ddof=1),
                         "roc_auc_mean": d.roc_auc.mean(), "balanced_accuracy_mean": d.balanced_accuracy.mean(),
                         "ece_ci_repeat0": fmt_ci(ci.loc["ece"]), "brier_ci_repeat0": fmt_ci(ci.loc["brier"])})
        curves_for[name] = first
        print(name, {v: round(float(np.mean([m['ece'] for m in l])), 3) for v, l in acc.items()})
    res = pd.DataFrame(rows); res.to_csv(OUT / "voice_calibration.csv", index=False)
    best = res[res.variant == "raw"].sort_values("roc_auc_mean", ascending=False).iloc[0]["model"]
    plot_reliability({f"{best} {v}": (y, curves_for[best][v]) for v in ("raw", "platt", "isotonic")},
                     OUT / "voice_reliability.png", f"Voice reliability ({best}, subject-wise, nested calibration)")
    return res


def export_models() -> dict:
    """Fit on ALL data and save artifacts for the API. (Final fit is for deployment, not for evaluation.)"""
    MODELS_DIR.mkdir(exist_ok=True)
    df = load_voice(); X, y, g = get_xyg(df); Xb, _, _ = get_xyg(df, BASIC_FEATURES)
    cmp_ = pd.read_csv(OUT / "voice_model_comparison.csv")
    gc = cmp_[(cmp_.scheme == "group_cv") & (~cmp_.model.str.contains("__"))].sort_values("roc_auc_mean", ascending=False)
    default = str(gc.iloc[0]["model"])
    meta = {"disclaimer": DISCLAIMER, "default_model": default,
            "default_model_choice": "highest subject-wise-CV ROC-AUC (selected on the same CV; mildly optimistic)",
            "n_subjects": int(len(set(g))), "n_recordings": int(len(y)),
            "feature_order": FEATURES, "basic_features": BASIC_FEATURES,
            "feature_ranges": {f: {"min": float(X[:, i].min()), "p1": float(np.percentile(X[:, i], 1)),
                                   "p99": float(np.percentile(X[:, i], 99)), "max": float(X[:, i].max())}
                               for i, f in enumerate(FEATURES)}}
    for n in MODEL_NAMES:
        joblib.dump(factory(n)().fit(X, y), MODELS_DIR / f"voice_{n}.joblib")
    joblib.dump(factory(default)().fit(Xb, y), MODELS_DIR / "voice_basic7.joblib")
    (MODELS_DIR / "voice_meta.json").write_text(json.dumps(meta, indent=2))
    print("exported voice models; default =", default)
    return meta


if __name__ == "__main__":
    eda(); compare_models(); calibration_experiment(); export_models()
