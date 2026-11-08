"""Synthetic spiral generator for SMOKE-TESTING the drawing pipeline only.

SYNTHETIC: pipeline validation only, not scientific results. Nothing produced here may appear in reported tables.

healthy   = smooth Archimedean spiral + very small noise.
parkinson = same spiral + 4-6 Hz radial tremor + amplitude jitter + micro-irregularities.
Two fake 'sources' (synA, synB) differ in image size, ink width and background tone, so the source-shortcut
check and cross-dataset code paths have something to detect. Each subject contributes several images.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

SOURCES = {
    "synA": {"size": 200, "width": 2, "bg": 255, "ink": (20, 20, 20)},
    "synB": {"size": 240, "width": 3, "bg": 244, "ink": (30, 30, 80)},
}


def spiral_xy(label: int, rng: np.random.Generator, style: dict, fs: float = 100.0) -> np.ndarray:
    """Return an (n, 2) array of pen positions in [-1, 1] units for one drawing."""
    turns = style["turns"]
    rev_per_s = style["rev_per_s"]
    duration = turns / rev_per_s
    t = np.arange(0, duration, 1 / fs)
    theta = 2 * np.pi * rev_per_s * t + style["phase"]
    r = 0.08 + 0.92 * (t / duration)                      # Archimedean: r grows linearly with angle
    if label == 1:
        f = style["tremor_hz"]                            # 4-6 Hz
        env = 1 + 0.6 * np.convolve(rng.normal(size=len(t)), np.ones(25) / 25, mode="same") * 5  # amplitude jitter
        r = r + style["tremor_amp"] * np.clip(env, 0.2, 2.0) * np.sin(2 * np.pi * f * t + rng.uniform(0, 6.28))
        r = r + rng.normal(0, style["micro"], len(t))     # micro-irregularities
    else:
        r = r + rng.normal(0, 0.0015, len(t))
    return np.c_[r * np.cos(theta), r * np.sin(theta)]


def render(xy: np.ndarray, source: str, rng: np.random.Generator, style: dict) -> Image.Image:
    cfg = SOURCES[source]
    s = cfg["size"]
    img = Image.new("RGB", (s, s), (cfg["bg"],) * 3)
    d = ImageDraw.Draw(img)
    scale = s * 0.42 * style["scale"]
    c = np.array([s / 2 + style["dx"] * s, s / 2 + style["dy"] * s])
    pts = [tuple(p) for p in (xy * scale + c)]
    d.line(pts, fill=cfg["ink"], width=cfg["width"], joint="curve")
    arr = np.asarray(img).astype(np.float32) + rng.normal(0, 2.0, (s, s, 3))   # mild scanner noise
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def generate_dataset(root: str | Path, n_subjects_per_class: int = 14, imgs_per_subject: int = 2,
                     seed: int = 42) -> pd.DataFrame:
    """Write PNGs under root/<source>/<label>/ and return the metadata table (same schema as real data)."""
    root = Path(root)
    rng = np.random.default_rng(seed)
    rows = []
    for source in SOURCES:
        for label, lname in ((0, "healthy"), (1, "parkinson")):
            (root / source / lname).mkdir(parents=True, exist_ok=True)
            for k in range(n_subjects_per_class):
                sid = f"{source}_S{label}{k:03d}"
                style = {"turns": rng.uniform(3.0, 4.0), "rev_per_s": rng.uniform(0.5, 0.75),
                         "phase": rng.uniform(0, 6.28), "scale": rng.uniform(0.85, 1.0),
                         "dx": rng.uniform(-0.03, 0.03), "dy": rng.uniform(-0.03, 0.03),
                         "tremor_hz": rng.uniform(4, 6), "tremor_amp": rng.uniform(0.018, 0.04),
                         "micro": rng.uniform(0.004, 0.009)}
                for j in range(imgs_per_subject):
                    xy = spiral_xy(label, rng, style)
                    p = root / source / lname / f"{sid}_{j}.png"
                    render(xy, source, rng, style).save(p)
                    rows.append({"image_path": str(p), "source_dataset": source, "subject_id": sid, "label": label,
                                 "subject_id_source": "synthetic"})
    return pd.DataFrame(rows)
