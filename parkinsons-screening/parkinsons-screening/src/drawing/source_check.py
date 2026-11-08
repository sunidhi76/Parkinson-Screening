"""Dataset-shortcut check: can a classifier tell WHICH DATASET an image came from?

If trivially-computable cues (image size, background tone, ink colour) identify the source, a model trained on
pooled data can exploit them instead of disease signal, and cross-dataset performance will collapse. We therefore
also ask whether those same trivial cues predict the LABEL (subject-wise)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.config import SEED
from src.drawing.features import features_matrix
from src.splits import assert_disjoint_groups, group_kfold


def raw_views(meta: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """(trivial metadata features, 16x16 RAW grayscale thumbnails) from the ORIGINAL files (no normalisation)."""
    trivial, thumbs = [], []
    for p in meta["image_path"]:
        with Image.open(p) as im:
            rgb = np.asarray(im.convert("RGB"), dtype=np.float32) / 255
            g = np.asarray(im.convert("L"), dtype=np.float32) / 255
            h, w = g.shape
            ink = g < np.median(g) - 0.1 if np.median(g) > 0.5 else g > np.median(g) + 0.1
            trivial.append([w, h, w / h, g.mean(), np.median(g), g.std(), *rgb.reshape(-1, 3).mean(0),
                            float(ink.mean())])
            thumbs.append(np.asarray(im.convert("L").resize((16, 16), Image.BILINEAR), dtype=np.float32).ravel() / 255)
    return np.array(trivial), np.array(thumbs)


def _group_cv_scores(X: np.ndarray, target: np.ndarray, groups: np.ndarray, seeds=(0, 1, 2)) -> tuple[float, float]:
    """Subject-wise CV (balanced accuracy, AUC when binary) for predicting `target`."""
    t = pd.factorize(target)[0]
    accs, aucs = [], []
    for s in seeds:
        sp = group_kfold(t, groups, 5, SEED + s); assert_disjoint_groups(groups, sp)
        pred = np.zeros(len(t), int); prob = np.zeros(len(t))
        for tr, te in sp:
            m = make_pipeline(StandardScaler(), RandomForestClassifier(200, class_weight="balanced_subsample",
                                                                     random_state=SEED, n_jobs=1))
            X_tr = np.nan_to_num(X[tr], nan=0.0); X_te = np.nan_to_num(X[te], nan=0.0)
            m.fit(X_tr, t[tr]); pred[te] = m.predict(X_te)
            if len(set(t)) == 2:
                prob[te] = m.predict_proba(X_te)[:, 1]
        accs.append(balanced_accuracy_score(t, pred))
        if len(set(t)) == 2:
            aucs.append(roc_auc_score(t, prob))
    return float(np.mean(accs)), float(np.mean(aucs)) if aucs else float("nan")


def source_check(meta: pd.DataFrame, imgs: np.ndarray) -> pd.DataFrame:
    """Predict source_dataset (and, for comparison, label) from three views. Chance balanced accuracy = 1/n_sources."""
    groups = meta["subject_id"].to_numpy(); src = meta["source_dataset"].to_numpy(); lab = meta["label"].to_numpy()
    trivial, thumbs = raw_views(meta)
    views = {"trivial_metadata (size, bg tone, colour)": trivial, "raw_thumbnail_16x16": thumbs,
             "handcrafted_spiral_features": features_matrix(imgs)}
    rows = []
    for name, X in views.items():
        a, _ = _group_cv_scores(X, src, groups)
        la, lauc = _group_cv_scores(X, lab, groups)
        rows.append({"view": name, "predict_SOURCE_balanced_acc": a, "chance_source": 1 / len(set(src)),
                     "predict_LABEL_balanced_acc": la, "predict_LABEL_auc": lauc})
    res = pd.DataFrame(rows)
    prev = meta.groupby("source_dataset")["label"].agg(["mean", "size"]).rename(columns={"mean": "frac_parkinson", "size": "n_images"})
    res.attrs["class_balance_by_source"] = prev
    return res
