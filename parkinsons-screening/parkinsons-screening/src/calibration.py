"""Calibration: ECE, reliability plot, and NESTED subject-wise calibration (isotonic / Platt)."""
from __future__ import annotations

from typing import Callable

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from src.splits import group_kfold


def expected_calibration_error(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    """Equal-width-bin ECE (weighted mean |accuracy(bin) - confidence(bin)|)."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            ece += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(ece)


def reliability_table(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> list[dict]:
    bins = np.minimum((np.asarray(p) * n_bins).astype(int), n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = bins == b
        if m.any():
            rows.append({"bin": b, "n": int(m.sum()), "mean_pred": float(np.mean(np.asarray(p)[m])),
                         "frac_pos": float(np.mean(np.asarray(y)[m]))})
    return rows


class _Platt:
    def fit(self, p: np.ndarray, y: np.ndarray) -> "_Platt":
        z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))[:, None]
        self.lr = LogisticRegression(C=1e6, max_iter=1000).fit(z, y)
        return self

    def predict(self, p: np.ndarray) -> np.ndarray:
        z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))[:, None]
        return self.lr.predict_proba(z)[:, 1]


def make_calibrator(method: str):
    if method == "isotonic":
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)

        class _Iso:
            def fit(self, p, y):
                iso.fit(p, y)
                return self

            def predict(self, p):
                return iso.predict(p)
        return _Iso()
    if method == "platt":
        return _Platt()
    raise ValueError(method)


def nested_calibrated_oof(make_model: Callable, X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                          outer_splits: list, method: str, inner_folds: int = 4, seed: int = 42
                          ) -> tuple[np.ndarray, np.ndarray]:
    """Out-of-fold (raw, calibrated) probabilities. The calibrator is fit ONLY on inner subject-wise
    out-of-fold predictions made within each outer training set, so test subjects never influence it."""
    raw, cal = np.full(len(y), np.nan), np.full(len(y), np.nan)
    for tr, te in outer_splits:
        inner = group_kfold(y[tr], groups[tr], n_splits=inner_folds, seed=seed)
        p_inner = np.full(len(tr), np.nan)
        for itr, ite in inner:
            m = make_model().fit(X[tr][itr], y[tr][itr])
            p_inner[ite] = m.predict_proba(X[tr][ite])[:, 1]
        calibrator = make_calibrator(method).fit(p_inner, y[tr])
        full = make_model().fit(X[tr], y[tr])
        p_te = full.predict_proba(X[te])[:, 1]
        raw[te] = p_te
        cal[te] = calibrator.predict(p_te)
    return raw, cal


def plot_reliability(curves: dict[str, tuple[np.ndarray, np.ndarray]], path, title: str, stamp: str | None = None,
                     n_bins: int = 10) -> None:
    """curves: label -> (y, p). Saves a reliability diagram."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.5, 5.2))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
    for label, (y, p) in curves.items():
        rt = reliability_table(y, p, n_bins)
        ax.plot([r["mean_pred"] for r in rt], [r["frac_pos"] for r in rt], "o-", label=label)
    ax.set_xlabel("mean predicted probability")
    ax.set_ylabel("observed fraction positive")
    ax.set_title(title + (f"\n{stamp}" if stamp else ""), fontsize=9, color="crimson" if stamp else "black")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
