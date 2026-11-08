"""UCI Parkinson's voice dataset loader. Features are pre-computed; there is no raw audio."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import VOICE_FILE

TARGET = "status"
SUBJECT_RE = re.compile(r"_(S\d+)_", re.IGNORECASE)
FEATURES: list[str] = [
    "MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)", "MDVP:Jitter(%)", "MDVP:Jitter(Abs)", "MDVP:RAP",
    "MDVP:PPQ", "Jitter:DDP", "MDVP:Shimmer", "MDVP:Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5",
    "MDVP:APQ", "Shimmer:DDA", "NHR", "HNR", "RPDE", "DFA", "spread1", "spread2", "D2", "PPE",
]
# Seven features a browser can approximate crudely from audio (used ONLY by the unvalidated demo model).
BASIC_FEATURES: list[str] = [
    "MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)", "MDVP:Jitter(%)", "MDVP:Shimmer", "NHR", "HNR",
]


def parse_subject(name: str) -> str:
    """'phon_R01_S01_1' -> 'S01'. Raises ValueError if the pattern is absent."""
    m = SUBJECT_RE.search(str(name))
    if not m:
        raise ValueError(f"cannot parse subject id from {name!r}")
    return m.group(1).upper()


def load_voice(path: str | Path = VOICE_FILE) -> pd.DataFrame:
    """Load parkinsons.data with a ``subject`` column. Raises FileNotFoundError with instructions."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found. Run `python data/download_data.py` (see README for the manual download link)."
        )
    df = pd.read_csv(p)
    missing = [c for c in FEATURES + [TARGET, "name"] if c not in df.columns]
    if missing:
        raise ValueError(f"unexpected file format, missing columns: {missing}")
    df["subject"] = df["name"].map(parse_subject)
    return df


def get_xyg(df: pd.DataFrame, features: list[str] | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (X, y, groups) with groups = subject id."""
    cols = features or FEATURES
    return df[cols].to_numpy(float), df[TARGET].to_numpy(int), df["subject"].to_numpy()


def subject_summary(df: pd.DataFrame) -> dict:
    """Subject-level counts per class, recordings per subject and any label inconsistency."""
    g = df.groupby("subject")[TARGET].agg(["nunique", "first", "size"])
    return {
        "n_recordings": int(len(df)),
        "n_subjects": int(len(g)),
        "subjects_per_class": {int(k): int(v) for k, v in g["first"].value_counts().sort_index().items()},
        "recordings_per_class": {int(k): int(v) for k, v in df[TARGET].value_counts().sort_index().items()},
        "recordings_per_subject_min_max": [int(g["size"].min()), int(g["size"].max())],
        "subjects_with_mixed_labels": int((g["nunique"] > 1).sum()),
    }
