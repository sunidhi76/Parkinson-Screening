"""Grad-CAM for the drawing CNN, plus a check that attention is on the STROKE rather than borders/paper."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from src.drawing.models import CNNModel, downsample


def gradcam(model: CNNModel, imgs: np.ndarray) -> np.ndarray:
    """Grad-CAM on the conv3 block (16x16) of the CNN for the predicted class, upsampled to 64x64, in [0,1]."""
    import torch
    net = model.net_.eval(); x = torch.from_numpy(downsample(imgs))[:, None].requires_grad_(False)
    cams = []
    for i in range(len(x)):
        acts = {}
        h = net.conv3.register_forward_hook(lambda m, a, o: acts.__setitem__("a", o))
        out = net(x[i:i + 1]); h.remove()
        sign = 1.0 if float(out.detach()) >= 0 else -1.0      # explain the PREDICTED class (PD if logit>=0, else healthy)
        grads = torch.autograd.grad(sign * out.sum(), acts["a"])[0]
        w = grads.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((w * acts["a"]).sum(1, keepdim=True))
        cam = torch.nn.functional.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        cam = cam - cam.min(); cam = cam / (cam.max() + 1e-9)
        cams.append(cam.detach().numpy())
    return np.stack(cams)


def attention_check(cams: np.ndarray, imgs: np.ndarray, border_frac: float = 0.08, dilate_px: int = 3) -> pd.DataFrame:
    """Per image: fraction of Grad-CAM mass on (dilated) stroke / on the outer border frame / on other 'paper',
    each with a 'lift' = mass fraction / area fraction (lift > 1 means attention is concentrated there)."""
    ims = downsample(imgs); n = ims.shape[-1]; b = max(1, int(border_frac * n))
    frame = np.zeros((n, n), bool); frame[:b] = frame[-b:] = True; frame[:, :b] = frame[:, -b:] = True
    rows = []
    for cam, im in zip(cams, ims):
        stroke = ndi.binary_dilation(im > 0.2, iterations=dilate_px)
        paper = ~stroke & ~frame
        tot = cam.sum() + 1e-9
        r = {}
        for name, m in (("stroke", stroke), ("border", frame & ~stroke), ("paper", paper)):
            area = m.mean() + 1e-9
            r[f"{name}_mass"] = float(cam[m].sum() / tot); r[f"{name}_lift"] = float(cam[m].sum() / tot / area)
        rows.append(r)
    return pd.DataFrame(rows)


def plot_gradcam(imgs: np.ndarray, cams: np.ndarray, labels: np.ndarray, probs: np.ndarray, path,
                 stamp: str | None = None, n_show: int = 8) -> None:
    ims = downsample(imgs); k = min(n_show, len(ims)); idx = np.linspace(0, len(ims) - 1, k).astype(int)
    fig, axes = plt.subplots(2, k, figsize=(1.7 * k, 3.9))
    axes = np.atleast_2d(axes).reshape(2, k)
    for j, i in enumerate(idx):
        axes[0, j].imshow(ims[i], cmap="gray_r"); axes[0, j].set_title(f"y={int(labels[i])} p={probs[i]:.2f}", fontsize=7)
        axes[1, j].imshow(ims[i], cmap="gray_r"); axes[1, j].imshow(cams[i], cmap="jet", alpha=0.5)
        for a in axes[:, j]:
            a.axis("off")
    fig.suptitle("Grad-CAM (held-out images)" + (f"\n{stamp}" if stamp else ""), fontsize=9,
                 color="crimson" if stamp else "black")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
