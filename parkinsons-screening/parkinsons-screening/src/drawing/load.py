"""Drawing data: user-placed HandPD / NewHandPD folders -> one metadata table + normalised image arrays."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from src.config import DRAWINGS_DIR

EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
SOURCES = {"handpd": "HandPD", "newhandpd": "NewHandPD"}
CLASSES = {"healthy": 0, "parkinson": 1}
WORK_SIZE = 128

INSTRUCTIONS = """\
No real drawing data found. To use real data, request the datasets from their owners (they are not
redistributable here and cannot be auto-downloaded reliably):

  HandPD     (Pereira et al.)  and  NewHandPD  - UNESP, Brazil. Search for "HandPD dataset UNESP"
             (the RECOD / Department of Computing, Sao Paulo State University, Bauru, project pages).
  Read each dataset's licence / terms of use before use.

Place SPIRAL images (not meanders) like this:
  data/drawings/handpd/healthy/*.png      data/drawings/handpd/parkinson/*.png
  data/drawings/newhandpd/healthy/*.png   data/drawings/newhandpd/parkinson/*.png

Several images may come from one person. Provide data/drawings/subjects.csv with columns
  filename,subject_id,source[,label]     (filename = image file name, source = handpd|newhandpd)
Without it every image is treated as its OWN subject, which can LEAK people across train/test.
Optional NewHandPD pen signals: data/drawings/newhandpd/signals/<image_stem>.csv (numeric columns).
"""


def build_metadata(root: str | Path = DRAWINGS_DIR, mapping_csv: str | Path | None = None) -> pd.DataFrame:
    """Columns: image_path, source_dataset, subject_id (prefixed by source), label, subject_id_source."""
    root = Path(root)
    mapping_csv = Path(mapping_csv) if mapping_csv else root / "subjects.csv"
    mapping = None
    if mapping_csv.exists():
        mapping = pd.read_csv(mapping_csv)
        need = {"filename", "subject_id", "source"}
        if not need <= set(mapping.columns):
            raise ValueError(f"{mapping_csv} must have columns {sorted(need)}; found {list(mapping.columns)}")
        mapping["source"] = mapping["source"].astype(str).str.lower()
        mapping = mapping.set_index(["source", "filename"])
    rows = []
    for src in SOURCES:
        for cname, lab in CLASSES.items():
            d = root / src / cname
            if not d.is_dir():
                continue
            for p in sorted(d.iterdir()):
                if p.suffix.lower() not in EXTS:
                    continue
                if mapping is not None and (src, p.name) in mapping.index:
                    sid = f"{src}_{mapping.loc[(src, p.name), 'subject_id']}"; how = "mapping_csv"
                else:
                    sid = f"{src}_{p.stem}"; how = "filename_fallback_UNVERIFIED"
                rows.append({"image_path": str(p), "source_dataset": src, "subject_id": sid, "label": lab,
                             "subject_id_source": how})
    meta = pd.DataFrame(rows, columns=["image_path", "source_dataset", "subject_id", "label", "subject_id_source"])
    if len(meta) and (meta.subject_id_source == "filename_fallback_UNVERIFIED").any():
        print("WARNING: some images have no subject mapping; each is treated as its own subject. "
              "If several images come from one person, subject-wise splits may still leak. Add subjects.csv.")
    if len(meta):
        assert meta.groupby("subject_id")["label"].nunique().max() == 1, "a subject has conflicting labels"
        assert meta.groupby("subject_id")["source_dataset"].nunique().max() == 1, "a subject spans datasets"
    return meta


def otsu(x: np.ndarray) -> float:
    """Otsu threshold for a float array in [0, 1]."""
    hist, edges = np.histogram(x.ravel(), bins=256, range=(0, 1))
    hist = hist.astype(float); w0 = np.cumsum(hist); w1 = w0[-1] - w0
    c = (edges[:-1] + edges[1:]) / 2; m0 = np.cumsum(hist * c) / np.maximum(w0, 1e-9)
    m1 = (np.sum(hist * c) - np.cumsum(hist * c)) / np.maximum(w1, 1e-9)
    return float(c[np.argmax(w0 * w1 * (m0 - m1) ** 2)])


def preprocess_image(img: Image.Image, size: int = WORK_SIZE) -> tuple[np.ndarray, dict]:
    """-> (ink image float32 [0,1] with stroke=1/background=0, cropped to the stroke and square, quality dict)."""
    if img.mode in ("RGBA", "LA"):
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255)); bg.alpha_composite(img.convert("RGBA")); img = bg
    g = np.asarray(img.convert("L"), dtype=np.float32) / 255.0
    h, w = g.shape
    lo, hi = np.percentile(g, 0.5), np.percentile(g, 99.5)
    contrast = float(hi - lo)
    ink = 1 - g if np.median(g) > 0.5 else g.copy()            # normalise polarity: stroke bright
    bgv = np.median(ink)
    ink = np.clip((ink - bgv) / max(ink.max() - bgv, 1e-6), 0, 1)
    thr = max(otsu(ink), 0.15)
    mask = ink > thr
    from scipy import ndimage as ndi
    lab, n = ndi.label(mask)
    if n:
        sizes = ndi.sum(mask, lab, range(1, n + 1)); keep = np.isin(lab, 1 + np.where(sizes >= max(5, 0.002 * mask.sum()))[0])
        mask_k = keep
    else:
        mask_k = mask
    n_comp = int(ndi.label(mask_k)[1]) if mask_k.any() else 0
    info = {"orig_height": h, "orig_width": w, "contrast": contrast, "ink_fraction": float(mask_k.mean()),
            "n_components": n_comp, "blank": not bool(mask_k.any()),
            "touches_border": bool(mask_k[0].any() or mask_k[-1].any() or mask_k[:, 0].any() or mask_k[:, -1].any())}
    if info["blank"]:
        return np.zeros((size, size), np.float32), info
    ys, xs = np.where(mask_k)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    side = int(max(y1 - y0, x1 - x0) * 1.08) + 2
    cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
    canvas = np.zeros((side, side), np.float32)
    ty0, tx0 = side // 2 - (cy - max(cy - side // 2, 0)), side // 2 - (cx - max(cx - side // 2, 0))
    sy0, sx0 = max(cy - side // 2, 0), max(cx - side // 2, 0)
    sub = ink[sy0:sy0 + side, sx0:sx0 + side] * mask_k[sy0:sy0 + side, sx0:sx0 + side]
    canvas[ty0:ty0 + sub.shape[0], tx0:tx0 + sub.shape[1]] = sub[: side - ty0, : side - tx0]
    out = np.asarray(Image.fromarray((canvas * 255).astype(np.uint8)).resize((size, size), Image.BILINEAR),
                     dtype=np.float32) / 255.0
    return out, info


def load_images(meta: pd.DataFrame, size: int = WORK_SIZE) -> tuple[np.ndarray, list[dict]]:
    """Load every image in the metadata table. Images are read, normalised in memory and never copied."""
    arrs, infos = [], []
    for p in meta["image_path"]:
        with Image.open(p) as im:
            a, i = preprocess_image(im, size)
        arrs.append(a); infos.append(i)
    return np.stack(arrs).astype(np.float32), infos
