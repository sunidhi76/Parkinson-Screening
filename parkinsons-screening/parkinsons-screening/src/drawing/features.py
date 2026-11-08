"""Handcrafted spiral features computed from a normalised ink image (stroke=1, background=0).

IMPORTANT: a static image has no time axis. 'Tremor frequency' here is a SPATIAL frequency in cycles per
revolution of the radial deviation from a smooth spiral, not Hz. Features are heuristic and were developed
and smoke-tested on synthetic spirals; they have not been validated on real HandPD/NewHandPD images.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

N_BINS = 180
FEATURE_NAMES = [
    "ink_fraction", "stroke_width", "n_components", "largest_comp_frac", "n_turns", "centerline_ok",
    "radial_dev_rms", "radial_dev_mad", "tremor_dom_freq", "tremor_band_frac", "low_band_frac", "spec_peakiness",
    "spacing_cv", "curv_rough", "curv_jerk", "dev_zero_cross_rate", "length_excess",
]


def _nan_features() -> dict[str, float]:
    return {k: float("nan") for k in FEATURE_NAMES}


def _crossings(r: np.ndarray, gap: float = 2.0) -> list[float]:
    """Cluster sorted radii into strokes (gap > `gap` pixels starts a new stroke) -> stroke centre radii."""
    if len(r) == 0:
        return []
    r = np.sort(r); cuts = np.where(np.diff(r) > gap)[0] + 1
    return [float(c.mean()) for c in np.split(r, cuts)]


def extract_features(img: np.ndarray) -> dict[str, float]:
    """img: (H, W) float in [0,1], stroke bright. Returns a dict keyed by FEATURE_NAMES (NaN on failure)."""
    f = _nan_features()
    mask = img > max(0.12, 0.2 * float(img.max())) if img.max() > 0 else np.zeros_like(img, bool)
    if mask.sum() < 20:
        f.update(ink_fraction=float(mask.mean()), centerline_ok=0.0)
        return f
    lab, n = ndi.label(mask)
    sizes = np.bincount(lab.ravel())[1:]
    perim = (mask & ~ndi.binary_erosion(mask)).sum()
    f.update(ink_fraction=float(mask.mean()), stroke_width=float(2 * mask.sum() / max(perim, 1)),
             n_components=float(n), largest_comp_frac=float(sizes.max() / sizes.sum()), centerline_ok=0.0)
    ys, xs = np.nonzero(mask)
    cy, cx = ys.mean(), xs.mean()
    dy, dx = ys - cy, xs - cx
    r = np.hypot(dx, dy); th = np.mod(np.arctan2(dy, dx), 2 * np.pi)
    rmax = r.max(); rmin = 0.06 * rmax
    keep = r > rmin
    b = np.minimum((th[keep] / (2 * np.pi) * N_BINS).astype(int), N_BINS - 1)
    rr = r[keep]
    per_bin = [_crossings(rr[b == i]) for i in range(N_BINS)]
    counts = np.array([len(c) for c in per_bin])
    nz = counts[counts > 0]
    if len(nz) < N_BINS // 2:
        return f
    # A spiral with a fractional number of turns has one extra crossing (its outermost) in some angular bins.
    # Use the K innermost crossings, with K a low quantile of the per-bin counts, so turns stay aligned.
    K = int(np.percentile(nz, 15))
    good = np.where(counts >= K)[0]
    f["n_turns"] = float(K)
    if K < 2 or len(good) < 0.6 * N_BINS:
        return f
    R = np.full((N_BINS, K), np.nan)
    for i in good:
        R[i] = per_bin[i][:K]
    # spacing regularity between consecutive turns
    sp = np.diff(R, axis=1)
    f["spacing_cv"] = float(np.nanstd(sp) / max(np.nanmean(sp), 1e-6))
    # radial deviation: high-pass each turn's radius-vs-angle series (circular moving average, 15 bins)
    dev_all, curv_rough, curv_jerk, zc, lexcess, spectra = [], [], [], [], [], []
    w = 15; ker = np.ones(w) / w
    dth = 2 * np.pi / N_BINS
    for k in range(K):
        s = R[:, k].copy()
        ok = ~np.isnan(s)
        if ok.sum() < 0.6 * N_BINS:
            continue
        s = np.interp(np.arange(N_BINS), np.where(ok)[0], s[ok])
        pad = np.r_[s[-w:], s, s[:w]]
        smooth = np.convolve(pad, ker, mode="same")[w:-w]
        e = s - smooth
        med, mad = np.median(e), np.median(np.abs(e - np.median(e))) + 1e-6
        e = np.clip(e - med, -6 * 1.4826 * mad - 1e-6, 6 * 1.4826 * mad + 1e-6)   # tame the start-angle jump
        dev_all.append(e / s.mean())
        spectra.append(np.abs(np.fft.rfft(e - e.mean())) ** 2)
        zc.append(np.mean(np.diff(np.sign(e - np.median(e))) != 0))
        d1 = (np.roll(smooth, -1) - np.roll(smooth, 1)) / (2 * dth)
        d2 = (np.roll(s, -1) - 2 * s + np.roll(s, 1)) / dth ** 2
        kappa = (s ** 2 + 2 * d1 ** 2 - s * d2) / np.power(s ** 2 + d1 ** 2, 1.5)
        kappa_s = np.convolve(np.r_[kappa[-w:], kappa, kappa[:w]], ker, mode="same")[w:-w]
        ra = kappa - kappa_s
        curv_rough.append(np.median(np.abs(ra)) / max(abs(np.median(kappa)), 1e-6))
        curv_jerk.append(np.median(np.abs(np.diff(kappa))) / max(abs(np.median(kappa)), 1e-6))
        L = np.sum(np.hypot(np.diff(s), s[:-1] * dth)); Ls = np.sum(np.hypot(np.diff(smooth), smooth[:-1] * dth))
        lexcess.append(L / max(Ls, 1e-6) - 1)
    if not dev_all:
        return f
    dev = np.concatenate(dev_all)
    spec = np.mean(spectra, axis=0); freqs = np.arange(len(spec))            # cycles per revolution
    tot = spec[2:].sum() + 1e-12
    band = (freqs >= 3) & (freqs <= 20)
    f.update(centerline_ok=1.0, radial_dev_rms=float(np.sqrt(np.mean(dev ** 2))),
             radial_dev_mad=float(np.median(np.abs(dev - np.median(dev)))),
             tremor_dom_freq=float(freqs[2:][np.argmax(spec[2:])]), tremor_band_frac=float(spec[band].sum() / tot),
             low_band_frac=float(spec[2:3].sum() / tot), spec_peakiness=float(spec[2:].max() / (np.median(spec[2:]) + 1e-12)),
             curv_rough=float(np.mean(curv_rough)), curv_jerk=float(np.mean(curv_jerk)),
             dev_zero_cross_rate=float(np.mean(zc)), length_excess=float(np.mean(lexcess)))
    return f


def features_matrix(imgs: np.ndarray) -> np.ndarray:
    """(N, H, W) -> (N, n_features) in FEATURE_NAMES order (NaN where extraction failed)."""
    rows = []
    for im in imgs:
        d = extract_features(im)
        rows.append([d[k] for k in FEATURE_NAMES])
    return np.array(rows, dtype=float)
