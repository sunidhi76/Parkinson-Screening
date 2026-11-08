"""Robustness: Gaussian noise on voice features and on drawings at several levels (subject-wise CV)."""
from __future__ import annotations

from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import RESULTS_DIR, SEED
from src.evaluate import point_metrics
from src.splits import make_splits

VOICE_LEVELS = [0.0, 0.05, 0.1, 0.25, 0.5, 1.0]  # noise SD as a fraction of each training-fold feature SD
IMAGE_LEVELS = [0.0, 0.05, 0.1, 0.2, 0.4]          # pixel-noise SD on images scaled to [0, 1]


def voice_noise_curve(model_factories: dict[str, Callable], X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                      levels: list[float] = VOICE_LEVELS, n_repeats: int = 3, n_noise: int = 3) -> pd.DataFrame:
    """Test features get N(0, (level*SD_train)^2) noise; models are fit on clean training folds."""
    rows = []
    for name, fac in model_factories.items():
        for lv in levels:
            bal, auc = [], []
            for r in range(n_repeats):
                P = np.zeros((n_noise, len(y)))
                for tr, te in make_splits("group", y, groups, SEED + r):
                    m = fac().fit(X[tr], y[tr]); sd = X[tr].std(axis=0)
                    for k in range(n_noise):
                        rng = np.random.default_rng(SEED + 1000 * r + k)
                        P[k, te] = m.predict_proba(X[te] + rng.normal(0, 1, X[te].shape) * sd * lv)[:, 1]
                for k in range(n_noise):
                    mm = point_metrics(y, P[k]); bal.append(mm["balanced_accuracy"]); auc.append(mm["roc_auc"])
            rows.append({"model": name, "noise_level": lv, "balanced_accuracy": np.mean(bal), "roc_auc": np.mean(auc),
                         "balanced_accuracy_std": np.std(bal), "roc_auc_std": np.std(auc)})
    return pd.DataFrame(rows)


def image_noise_curve(model_factories: dict[str, Callable], imgs: np.ndarray, y: np.ndarray, groups: np.ndarray,
                      levels: list[float] = IMAGE_LEVELS, n_splits: int = 5) -> pd.DataFrame:
    """Noise added to TEST images only (clipped to [0,1]); features are re-extracted from the noisy image."""
    rows = []
    sp = make_splits("group", y, groups, SEED, n_splits)
    for name, fac in model_factories.items():
        models = [(te, fac().fit(imgs[tr], y[tr])) for tr, te in sp]
        for lv in levels:
            p = np.zeros(len(y)); rng = np.random.default_rng(SEED)
            for te, m in models:
                noisy = np.clip(imgs[te] + rng.normal(0, lv, imgs[te].shape), 0, 1).astype(np.float32)
                p[te] = m.predict_proba(noisy)[:, 1]
            mm = point_metrics(y, p)
            rows.append({"model": name, "noise_level": lv, "balanced_accuracy": mm["balanced_accuracy"], "roc_auc": mm["roc_auc"]})
    return pd.DataFrame(rows)


def plot_noise_curves(df: pd.DataFrame, path, title: str, xlabel: str, stamp: str | None = None) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
    for ax, col in zip(axes, ["balanced_accuracy", "roc_auc"]):
        for name, d in df.groupby("model"):
            ax.plot(d["noise_level"], d[col], "o-", label=name)
        ax.axhline(0.5, color="gray", ls=":", lw=1); ax.set_xlabel(xlabel); ax.set_ylabel(col); ax.set_ylim(0.3, 1.0)
    axes[0].legend(fontsize=7)
    fig.suptitle(title + (f"\n{stamp}" if stamp else ""), fontsize=9, color="crimson" if stamp else "black")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def run_voice() -> pd.DataFrame:
    from src.voice.load import get_xyg, load_voice
    from src.voice.models import MODEL_NAMES, factory
    out = RESULTS_DIR / "voice"; out.mkdir(parents=True, exist_ok=True)
    X, y, g = get_xyg(load_voice())
    df = voice_noise_curve({n: factory(n) for n in MODEL_NAMES}, X, y, g)
    df.to_csv(out / "voice_robustness.csv", index=False)
    plot_noise_curves(df, out / "voice_robustness.png", "Voice: performance vs feature noise (subject-wise CV)",
                      "noise SD / feature SD")
    return df


if __name__ == "__main__":
    print(run_voice().round(3).to_string(index=False))
