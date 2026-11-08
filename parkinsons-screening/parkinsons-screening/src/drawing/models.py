"""Drawing classifiers. Both take NORMALISED INK IMAGES (N,H,W) in [0,1] so noise/robustness tests act end-to-end.

* HandcraftedModel: spiral features -> impute -> scale -> class-weighted classifier (sklearn Pipeline).
* CNNModel: small CNN (optional augmentation) trained from scratch with a class-weighted loss.
"""
from __future__ import annotations

import math

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import IMG_SIZE, SEED
from src.drawing.features import features_matrix


class HandcraftedModel:
    """Image -> handcrafted features -> classifier. ``kind``: 'rf' (default) or 'logreg'."""

    def __init__(self, kind: str = "rf", seed: int = SEED) -> None:
        self.kind, self.seed = kind, seed

    def _clf(self):
        if self.kind == "logreg":
            return LogisticRegression(C=0.5, class_weight="balanced", max_iter=2000, random_state=self.seed)
        return RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample", min_samples_leaf=2,
                                      random_state=self.seed, n_jobs=1)

    def fit(self, imgs: np.ndarray, y: np.ndarray) -> "HandcraftedModel":
        self.pipe_ = Pipeline([("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                               ("scale", StandardScaler()), ("clf", self._clf())])
        self.pipe_.fit(features_matrix(imgs), y)
        return self

    def predict_proba(self, imgs: np.ndarray) -> np.ndarray:
        return self.pipe_.predict_proba(features_matrix(imgs))

    def predict_proba_features(self, F: np.ndarray) -> np.ndarray:
        return self.pipe_.predict_proba(F)


def _torch():
    try:
        import torch
        return torch
    except ImportError as e:  # pragma: no cover
        raise ImportError("PyTorch is required for the CNN (pip install torch).") from e


def downsample(imgs: np.ndarray, size: int = IMG_SIZE) -> np.ndarray:
    """Area-average (N,H,W) -> (N,size,size) with H divisible by size (128 -> 64)."""
    n, h, w = imgs.shape
    if h == size:
        return imgs.astype(np.float32)
    f = h // size
    return imgs[:, : f * size, : f * size].reshape(n, size, f, size, f).mean(axis=(2, 4)).astype(np.float32)


def augment_batch(x, gen):
    """Random affine (rot ±15°, scale 0.9-1.1, shift ±6%), contrast, additive noise. x: (B,1,H,W) tensor."""
    torch = _torch()
    b = x.shape[0]
    ang = (torch.rand(b, generator=gen) * 2 - 1) * math.radians(15)
    sc = 0.9 + 0.2 * torch.rand(b, generator=gen)
    tx, ty = (torch.rand(b, generator=gen) * 2 - 1) * 0.12, (torch.rand(b, generator=gen) * 2 - 1) * 0.12
    cos, sin = torch.cos(ang) / sc, torch.sin(ang) / sc
    theta = torch.stack([torch.stack([cos, -sin, tx], 1), torch.stack([sin, cos, ty], 1)], 1)
    grid = torch.nn.functional.affine_grid(theta, x.shape, align_corners=False)
    out = torch.nn.functional.grid_sample(x, grid, padding_mode="zeros", align_corners=False)
    out = out * (0.8 + 0.4 * torch.rand(b, 1, 1, 1, generator=gen)) + 0.03 * torch.randn(out.shape, generator=gen)
    return out.clamp(0, 1)


def build_net():
    torch = _torch(); nn = torch.nn

    class Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv1 = nn.Sequential(nn.Conv2d(1, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU(), nn.MaxPool2d(2))
            self.conv2 = nn.Sequential(nn.Conv2d(8, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2))
            self.conv3 = nn.Sequential(nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU())   # Grad-CAM layer (16x16)
            self.head = nn.Sequential(nn.MaxPool2d(2), nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
                                      nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(32, 1))

        def forward(self, x):
            return self.head(self.conv3(self.conv2(self.conv1(x))))

    return Net()


class CNNModel:
    """Small CNN. ``augment`` toggles training-time augmentation. Class imbalance handled with pos_weight."""

    def __init__(self, augment: bool = True, epochs: int = 20, lr: float = 2e-3, batch: int = 32,
                 seed: int = SEED) -> None:
        self.augment, self.epochs, self.lr, self.batch, self.seed = augment, epochs, lr, batch, seed

    def fit(self, imgs: np.ndarray, y: np.ndarray) -> "CNNModel":
        torch = _torch(); torch.set_num_threads(1); torch.manual_seed(self.seed)
        gen = torch.Generator().manual_seed(self.seed)
        x = torch.from_numpy(downsample(imgs))[:, None]; yt = torch.from_numpy(np.asarray(y, np.float32))
        self.net_ = build_net()
        pw = torch.tensor(float((yt == 0).sum() / max((yt == 1).sum(), 1)))
        lossf = torch.nn.BCEWithLogitsLoss(pos_weight=pw)
        opt = torch.optim.Adam(self.net_.parameters(), lr=self.lr, weight_decay=1e-4)
        n = len(x)
        for _ in range(self.epochs):
            self.net_.train(); perm = torch.randperm(n, generator=gen)
            for i in range(0, n, self.batch):
                idx = perm[i:i + self.batch]
                if len(idx) < 2:
                    continue
                xb = augment_batch(x[idx], gen) if self.augment else x[idx]
                opt.zero_grad(); lossf(self.net_(xb).squeeze(1), yt[idx]).backward(); opt.step()
        return self

    def predict_proba(self, imgs: np.ndarray) -> np.ndarray:
        torch = _torch(); torch.set_num_threads(1)
        self.net_.eval()
        with torch.no_grad():
            p = torch.sigmoid(self.net_(torch.from_numpy(downsample(imgs))[:, None]).squeeze(1)).numpy()
        return np.c_[1 - p, p]
