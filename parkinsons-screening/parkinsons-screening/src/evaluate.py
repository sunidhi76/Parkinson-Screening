"""Metrics, subject-level (cluster) bootstrap CIs, and out-of-fold prediction helpers."""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from src.calibration import expected_calibration_error
from src.config import N_BOOT, SEED

METRICS = ["balanced_accuracy", "roc_auc", "pr_auc", "sensitivity", "specificity", "brier", "ece"]


def point_metrics(y: np.ndarray, p: np.ndarray, thr: float = 0.5) -> dict[str, float]:
    """All headline metrics. AUCs are NaN if only one class is present."""
    y, p = np.asarray(y).astype(int), np.asarray(p, float)
    pred = (p >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    two = len(set(y.tolist())) == 2
    return {
        "balanced_accuracy": float(np.nanmean([sens, spec])),
        "roc_auc": float(roc_auc_score(y, p)) if two else float("nan"),
        "pr_auc": float(average_precision_score(y, p)) if two else float("nan"),
        "sensitivity": sens, "specificity": spec,
        "brier": float(brier_score_loss(y, p)), "ece": expected_calibration_error(y, p),
    }


def cluster_bootstrap_ci(y: np.ndarray, p: np.ndarray, groups: np.ndarray, n_boot: int = N_BOOT,
                         seed: int = SEED, thr: float = 0.5) -> pd.DataFrame:
    """95% percentile CIs resampling SUBJECTS (not recordings) with replacement, since recordings from one
    subject are not independent. Returns a DataFrame indexed by metric with point, lo, hi."""
    y, p, g = np.asarray(y), np.asarray(p), np.asarray(groups)
    rng = np.random.default_rng(seed)
    uniq = np.unique(g)
    idx_by = {u: np.where(g == u)[0] for u in uniq}
    draws = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by[u] for u in pick])
        if len(set(y[idx].tolist())) < 2:
            continue
        draws.append(point_metrics(y[idx], p[idx], thr))
    d = pd.DataFrame(draws)
    pt = point_metrics(y, p, thr)
    rows = {m: {"point": pt[m], "lo": float(np.nanpercentile(d[m], 2.5)), "hi": float(np.nanpercentile(d[m], 97.5))}
            for m in METRICS}
    out = pd.DataFrame(rows).T
    out.attrs["n_valid_boot"] = len(d)
    return out


def fmt_ci(row: pd.Series, nd: int = 3) -> str:
    return f"{row['point']:.{nd}f} [{row['lo']:.{nd}f}, {row['hi']:.{nd}f}]"


def oof_predict(make_model: Callable, X: np.ndarray, y: np.ndarray, splits: Sequence) -> np.ndarray:
    """Out-of-fold P(class 1). A fresh model (hence fresh preprocessing) is fit per fold."""
    p = np.full(len(y), np.nan)
    for tr, te in splits:
        m = make_model().fit(X[tr], y[tr])
        p[te] = m.predict_proba(X[te])[:, 1]
    assert not np.isnan(p).any(), "some samples were never predicted"
    return p


def repeated_oof(make_model: Callable, X: np.ndarray, y: np.ndarray, groups: np.ndarray, kind: str,
                 seeds: Sequence[int], n_splits: int = 5) -> np.ndarray:
    """Array [n_repeats, n]: one OOF prediction vector per CV repetition."""
    from src.splits import make_splits
    return np.vstack([oof_predict(make_model, X, y, make_splits(kind, y, groups, s, n_splits)) for s in seeds])


def per_fold_table(y: np.ndarray, p: np.ndarray, splits: Sequence) -> pd.DataFrame:
    """Metrics per outer fold (only meaningful when folds contain both classes)."""
    rows = []
    for i, (_, te) in enumerate(splits):
        r = point_metrics(y[te], p[te]); r["fold"] = i; r["n_test"] = len(te); rows.append(r)
    return pd.DataFrame(rows)


def summarize_repeats(y: np.ndarray, P: np.ndarray) -> dict[str, tuple[float, float]]:
    """mean and std over CV repetitions of the pooled-OOF metric."""
    ms = pd.DataFrame([point_metrics(y, P[i]) for i in range(len(P))])
    return {m: (float(ms[m].mean()), float(ms[m].std(ddof=1)) if len(ms) > 1 else 0.0) for m in METRICS}
