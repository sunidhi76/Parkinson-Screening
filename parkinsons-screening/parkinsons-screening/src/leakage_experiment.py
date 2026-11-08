"""HEADLINE EXPERIMENT: data leakage on UCI voice data. Recording-level random split vs subject-wise
StratifiedGroupKFold vs Leave-One-Subject-Out. Saves a table and a bar chart."""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import CV_REPEATS, RESULTS_DIR, SEED
from src.evaluate import cluster_bootstrap_ci, repeated_oof, summarize_repeats
from src.splits import assert_cv_partition, assert_disjoint_groups, make_splits
from src.voice.load import get_xyg, load_voice, subject_summary
from src.voice.models import MODEL_NAMES, factory

SCHEMES = {"random": "Random (recording-level)", "group": "GroupKFold (subject-wise)",
           "logo": "Leave-One-Subject-Out"}


def run(n_repeats: int = CV_REPEATS, models: list[str] | None = None) -> pd.DataFrame:
    models = models or MODEL_NAMES
    out = RESULTS_DIR / "voice"; out.mkdir(parents=True, exist_ok=True)
    df = load_voice(); X, y, g = get_xyg(df)
    # verify the split guarantees before using them
    for s in range(n_repeats):
        sp = make_splits("group", y, g, SEED + s); assert_disjoint_groups(g, sp); assert_cv_partition(len(y), sp)
    sp = make_splits("logo", y, g); assert_disjoint_groups(g, sp); assert_cv_partition(len(y), sp)
    leaked = [len(set(g[tr]) & set(g[te])) for tr, te in make_splits("random", y, g, SEED)]
    print("Subjects present in BOTH train and test per random fold:", leaked)

    rows = []
    for name in models:
        for kind in SCHEMES:
            seeds = [SEED + i for i in range(n_repeats)] if kind != "logo" else [SEED]
            P = repeated_oof(factory(name), X, y, g, kind, seeds)
            s = summarize_repeats(y, P)
            ci = cluster_bootstrap_ci(y, P.mean(axis=0), g)
            rows.append({
                "model": name, "scheme": kind, "n_repeats": len(seeds),
                "balanced_accuracy_mean": s["balanced_accuracy"][0], "balanced_accuracy_std": s["balanced_accuracy"][1],
                "bal_acc_ci_lo": ci.loc["balanced_accuracy", "lo"], "bal_acc_ci_hi": ci.loc["balanced_accuracy", "hi"],
                "roc_auc_mean": s["roc_auc"][0], "roc_auc_std": s["roc_auc"][1],
                "auc_ci_lo": ci.loc["roc_auc", "lo"], "auc_ci_hi": ci.loc["roc_auc", "hi"],
                "sensitivity_mean": s["sensitivity"][0], "specificity_mean": s["specificity"][0],
            })
            print(f"{name:14s} {kind:7s} balAcc={s['balanced_accuracy'][0]:.3f}±{s['balanced_accuracy'][1]:.3f} "
                  f"AUC={s['roc_auc'][0]:.3f}±{s['roc_auc'][1]:.3f}")
    res = pd.DataFrame(rows)
    res.to_csv(out / "leakage_results.csv", index=False)

    gap = res.pivot(index="model", columns="scheme", values=["balanced_accuracy_mean", "roc_auc_mean"])
    summary = {
        "voice_data": subject_summary(df),
        "leaked_subjects_per_random_fold": leaked,
        "gap_random_minus_group": {
            m: {"balanced_accuracy": float(gap.loc[m, ("balanced_accuracy_mean", "random")] - gap.loc[m, ("balanced_accuracy_mean", "group")]),
                "roc_auc": float(gap.loc[m, ("roc_auc_mean", "random")] - gap.loc[m, ("roc_auc_mean", "group")])}
            for m in models},
    }
    (out / "leakage_summary.json").write_text(json.dumps(summary, indent=2))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    w = 0.26
    for ax, (col, lo, hi, title) in zip(axes, [("balanced_accuracy_mean", "bal_acc_ci_lo", "bal_acc_ci_hi", "Balanced accuracy"),
                                               ("roc_auc_mean", "auc_ci_lo", "auc_ci_hi", "ROC-AUC")]):
        for i, kind in enumerate(SCHEMES):
            sub = res[res.scheme == kind].set_index("model").loc[models]
            xs = np.arange(len(models)) + (i - 1) * w
            err = np.vstack([sub[col] - sub[lo], sub[hi] - sub[col]]).clip(min=0)
            ax.bar(xs, sub[col], w, yerr=err, capsize=2, label=SCHEMES[kind])
        ax.axhline(0.5, color="gray", ls=":", lw=1)
        ax.set_xticks(range(len(models))); ax.set_xticklabels(models, fontsize=8)
        ax.set_ylim(0.3, 1.02); ax.set_title(title)
    axes[0].legend(fontsize=7, loc="lower left")
    fig.suptitle("UCI voice: same models, different splits (error bars: 95% subject-level bootstrap CI)", fontsize=10)
    fig.tight_layout(); fig.savefig(out / "leakage_chart.png", dpi=140); plt.close(fig)
    return res


if __name__ == "__main__":
    run()
