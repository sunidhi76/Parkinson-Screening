"""Drawing experiments 5-9 (source shortcut, pooled subject-wise CV, cross-dataset, robustness, Grad-CAM).

If no real drawings are present, everything runs on the SYNTHETIC generator as a SMOKE TEST and every output is
stamped 'SYNTHETIC: pipeline validation only, not scientific results.'"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd

from src.config import DATA_DIR, DISCLAIMER, DRAWINGS_DIR, MODELS_DIR, RESULTS_DIR, SEED, SYNTHETIC_STAMP
from src.drawing.explain import attention_check, gradcam, plot_gradcam
from src.drawing.features import FEATURE_NAMES, features_matrix
from src.drawing.load import INSTRUCTIONS, build_metadata, load_images
from src.drawing.models import CNNModel, HandcraftedModel
from src.drawing.source_check import source_check
from src.evaluate import cluster_bootstrap_ci, fmt_ci, point_metrics, summarize_repeats
from src.robustness import image_noise_curve, plot_noise_curves
from src.splits import assert_disjoint_groups, leave_one_dataset_out, make_splits
from src.synthetic import generate_dataset


class Ctx:
    """Holds data + output location + synthetic flag."""

    def __init__(self, quick: bool = False, force_synthetic: bool = False) -> None:
        meta = pd.DataFrame() if force_synthetic else build_metadata()
        self.synthetic = len(meta) == 0
        if self.synthetic:
            print(INSTRUCTIONS); print(f"\n>>> {SYNTHETIC_STAMP}  Running SMOKE TEST on synthetic spirals.\n")
            meta = generate_dataset(DATA_DIR / "drawings_synthetic")
        self.meta = meta.reset_index(drop=True)
        self.imgs, self.infos = load_images(self.meta)
        self.y = self.meta["label"].to_numpy(int); self.g = self.meta["subject_id"].to_numpy()
        self.src = self.meta["source_dataset"].to_numpy()
        self.out = RESULTS_DIR / ("drawing_synthetic_smoke" if self.synthetic else "drawing"); self.out.mkdir(parents=True, exist_ok=True)
        self.quick = quick
        self.epochs = 6 if quick else 20
        self.repeats = 1 if quick else 3
        self.stamp = SYNTHETIC_STAMP if self.synthetic else None
        if self.synthetic:
            (self.out / "SYNTHETIC_STAMP.txt").write_text(SYNTHETIC_STAMP + "\n")

    def models(self, which: list[str] | None = None) -> dict:
        allm = {"handcrafted_rf": lambda: HandcraftedModel("rf"), "handcrafted_logreg": lambda: HandcraftedModel("logreg"),
                "cnn_aug": lambda: CNNModel(True, self.epochs), "cnn_noaug": lambda: CNNModel(False, self.epochs)}
        return {k: v for k, v in allm.items() if not which or k in which}

    def save(self, df: pd.DataFrame, name: str) -> None:
        d = df.copy()
        if self.synthetic:
            d["note"] = SYNTHETIC_STAMP
        d.to_csv(self.out / name, index=False)


def _oof(fac, imgs, y, splits) -> np.ndarray:
    p = np.full(len(y), np.nan)
    for tr, te in splits:
        p[te] = fac().fit(imgs[tr], y[tr]).predict_proba(imgs[te])[:, 1]
    return p


def _row(model: str, scope: str, y, P, g, n_boot: int = 500) -> dict:
    """P: [n_repeats, n] OOF predictions for this scope."""
    s = summarize_repeats(y, P); ci = cluster_bootstrap_ci(y, P.mean(axis=0), g, n_boot=n_boot)
    r = {"model": model, "scope": scope, "n_images": len(y), "n_subjects": len(set(g)),
         "n_pd_subjects": len(set(g[y == 1])), "n_healthy_subjects": len(set(g[y == 0]))}
    for m in ("balanced_accuracy", "roc_auc", "pr_auc", "sensitivity", "specificity", "brier", "ece"):
        r[f"{m}_mean±std"] = f"{s[m][0]:.3f} ± {s[m][1]:.3f}"
    r["balanced_accuracy_ci95"] = fmt_ci(ci.loc["balanced_accuracy"]); r["roc_auc_ci95"] = fmt_ci(ci.loc["roc_auc"])
    return r


def exp5_source(c: Ctx) -> pd.DataFrame:
    res = source_check(c.meta, c.imgs); c.save(res, "5_source_check.csv")
    c.save(res.attrs["class_balance_by_source"].reset_index(), "5_class_balance_by_source.csv")
    print("\n[5] Source-shortcut check (subject-wise CV):\n", res.round(3).to_string(index=False))
    return res


def exp6_pooled(c: Ctx) -> pd.DataFrame:
    rows, P_store = [], {}
    seeds = [SEED + i for i in range(c.repeats)]
    for name, fac in c.models().items():
        P = []
        for s in seeds:
            sp = make_splits("group", c.y, c.g, s); assert_disjoint_groups(c.g, sp)
            P.append(_oof(fac, c.imgs, c.y, sp))
        P = np.vstack(P); P_store[name] = P
        rows.append(_row(name, "pooled", c.y, P, c.g))
        for s_name in sorted(set(c.src)):
            m = c.src == s_name
            if len(set(c.y[m])) == 2:
                rows.append(_row(name, s_name, c.y[m], P[:, m], c.g[m]))
        print(f"[6] {name:20s} pooled AUC {rows[-len(set(c.src)) - 1]['roc_auc_ci95']}")
    res = pd.DataFrame(rows); c.save(res, "6_pooled_subject_wise_cv.csv")
    return res


def exp7_cross(c: Ctx) -> pd.DataFrame:
    sources = sorted(set(c.src))
    if len(sources) < 2:
        print("[7] skipped: need two datasets."); return pd.DataFrame()
    rows = []
    for name, fac in c.models(["handcrafted_rf", "cnn_aug", "cnn_noaug"]).items():
        for held, tr, te in leave_one_dataset_out(c.src):
            assert not (set(c.g[tr]) & set(c.g[te])), "subject overlap across datasets!"
            p = fac().fit(c.imgs[tr], c.y[tr]).predict_proba(c.imgs[te])[:, 1]
            cross = point_metrics(c.y[te], p); ci = cluster_bootstrap_ci(c.y[te], p, c.g[te], n_boot=500)
            # within-dataset baseline: subject-wise CV on the held-out dataset only
            mw = np.where(c.src == held)[0]; Pw = []
            for s in [SEED + i for i in range(c.repeats)]:
                sp = [(mw[a], mw[b]) for a, b in make_splits("group", c.y[mw], c.g[mw], s)]; assert_disjoint_groups(c.g, sp)
                Pw.append(_oof(fac, c.imgs, c.y, sp)[mw])
            within = summarize_repeats(c.y[mw], np.vstack(Pw))
            rows.append({"model": name, "train_on": "+".join(s for s in sources if s != held), "test_on": held,
                         "within_dataset_auc": within["roc_auc"][0], "cross_dataset_auc": cross["roc_auc"],
                         "cross_auc_ci95": fmt_ci(ci.loc["roc_auc"]), "auc_drop": within["roc_auc"][0] - cross["roc_auc"],
                         "within_dataset_bal_acc": within["balanced_accuracy"][0], "cross_dataset_bal_acc": cross["balanced_accuracy"],
                         "cross_bal_acc_ci95": fmt_ci(ci.loc["balanced_accuracy"]),
                         "bal_acc_drop": within["balanced_accuracy"][0] - cross["balanced_accuracy"]})
            print(f"[7] {name:15s} {rows[-1]['train_on']}->{held}: AUC within {within['roc_auc'][0]:.3f} cross {cross['roc_auc']:.3f}")
    res = pd.DataFrame(rows); c.save(res, "7_cross_dataset.csv"); return res


def exp8_robust(c: Ctx) -> pd.DataFrame:
    res = image_noise_curve(c.models(["handcrafted_rf", "cnn_aug", "cnn_noaug"]), c.imgs, c.y, c.g)
    c.save(res, "8_image_robustness.csv")
    plot_noise_curves(res, c.out / "8_image_robustness.png", "Drawings: performance vs pixel noise (subject-wise CV)",
                      "pixel-noise SD (image range 0-1)", c.stamp)
    print("[8]\n", res.pivot(index="noise_level", columns="model", values="roc_auc").round(3).to_string()); return res


def exp9_gradcam(c: Ctx) -> pd.DataFrame:
    sp = make_splits("group", c.y, c.g, SEED); tr, te = sp[0]
    m = CNNModel(True, c.epochs).fit(c.imgs[tr], c.y[tr])
    cams = gradcam(m, c.imgs[te]); p = m.predict_proba(c.imgs[te])[:, 1]
    plot_gradcam(c.imgs[te], cams, c.y[te], p, c.out / "9_gradcam_examples.png", c.stamp)
    att = attention_check(cams, c.imgs[te]); att["source"] = c.src[te]
    summ = att.groupby("source").mean(numeric_only=True).reset_index()
    summ["verdict"] = np.where((summ.stroke_lift > 1.0) & (summ.border_lift < 1.0), "attention concentrated on stroke",
                               "CHECK: attention not clearly on stroke (possible border/paper artifact)")
    c.save(summ, "9_gradcam_attention_check.csv"); print("[9]\n", summ.round(3).to_string(index=False)); return summ


def exp_signals(c: Ctx) -> None:
    """OPTIONAL: within-person image + NewHandPD pen-signal experiment. Skipped when no signal files exist."""
    sig_dir = DRAWINGS_DIR / "newhandpd" / "signals"
    if c.synthetic or not sig_dir.is_dir():
        print("[opt] NewHandPD pen-signal experiment skipped (no data/drawings/newhandpd/signals/)."); return
    feats, idx = [], []
    for i, p in enumerate(c.meta["image_path"]):
        f = sig_dir / (Path(p).stem + ".csv")
        if c.src[i] == "newhandpd" and f.exists():
            a = pd.read_csv(f).select_dtypes("number").to_numpy(float)
            feats.append(np.r_[a.mean(0), a.std(0), a.min(0), a.max(0), np.abs(np.diff(a, axis=0)).mean(0)]); idx.append(i)
    if len(idx) < 20 or len({len(x) for x in feats}) != 1:
        print("[opt] signals found but unusable (too few or inconsistent columns); skipped."); return
    idx = np.array(idx); S = np.vstack(feats); I = features_matrix(c.imgs[idx]); y, g = c.y[idx], c.g[idx]
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from src.evaluate import oof_predict
    fac = lambda: make_pipeline(SimpleImputer(), StandardScaler(), RandomForestClassifier(300, class_weight="balanced_subsample", random_state=SEED, n_jobs=1))
    rows = []
    for name, X in (("image_features", I), ("pen_signal_features", S), ("image+signal", np.c_[I, S])):
        sp = make_splits("group", y, g, SEED); assert_disjoint_groups(g, sp)
        P = oof_predict(fac, X, y, sp); r = point_metrics(y, P); r.update(model=name, n=len(y)); rows.append(r)
    c.save(pd.DataFrame(rows), "opt_newhandpd_image_plus_signal.csv")


def export_model(c: Ctx) -> None:
    """Fit the handcrafted model on ALL images and save it with provenance (API refuses to hide synthetic origin)."""
    MODELS_DIR.mkdir(exist_ok=True)
    m = HandcraftedModel("rf").fit(c.imgs, c.y)
    joblib.dump(m, MODELS_DIR / "drawing_handcrafted.joblib")
    meta = {"trained_on": "synthetic" if c.synthetic else "real", "n_images": int(len(c.y)), "n_subjects": int(len(set(c.g))),
            "sources": sorted(set(c.src)), "feature_names": FEATURE_NAMES,
            "warning": SYNTHETIC_STAMP + " This model was trained on synthetic spirals only; its outputs are meaningless for real drawings."
            if c.synthetic else "Trained on pooled real data; see README for subject-wise and cross-dataset performance.",
            "disclaimer": DISCLAIMER}
    (MODELS_DIR / "drawing_meta.json").write_text(json.dumps(meta, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--quick", action="store_true"); ap.add_argument("--synthetic", action="store_true")
    a = ap.parse_args(); c = Ctx(a.quick, a.synthetic)
    print(f"images={len(c.y)} subjects={len(set(c.g))} per source: {pd.Series(c.src).value_counts().to_dict()}")
    exp5_source(c); exp6_pooled(c); exp7_cross(c); exp8_robust(c); exp9_gradcam(c); exp_signals(c); export_model(c)
    print("done ->", c.out)


if __name__ == "__main__":
    main()
