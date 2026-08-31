"""Donut heatmaps of cyclic angle distributions, ported from ``Experiments - TownCentre.ipynb``.

The original notebook (cells 10-18) draws the prediction distribution of a model as a
donut-shaped cyclic histogram (paper Fig. 1) - softmax models "stick" to the bin centres while
Biternion models produce a continuous ring. The maths here is a faithful port (same defaults:
3600 bins, gaussian window 41, Spectral_r, zero at the top, counts scaled by ``n/400``);
``donut_heatmap`` is vectorised instead of looping over pixels.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"


def gaussfilter(n: int, sigma: float = 0.3, norm=np.sum) -> np.ndarray:
    x = np.arange(-(n - 1) / 2, (n + 1) / 2)
    x /= np.max(x)
    y = 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-(x**2) / (2 * sigma**2))
    if norm is not None:
        y /= norm(y)
    return y


def cyclic_filter(a: np.ndarray, f: np.ndarray) -> np.ndarray:
    """'same' correlation with cyclic (wrap) padding."""
    a = np.pad(a, pad_width=len(f) // 2, mode="wrap")
    return np.correlate(a, f, mode="valid")


def mkheatmap_deg(preds: np.ndarray, nbins: int = 360) -> np.ndarray:
    """Cyclic histogram of angle predictions in degrees (unnormalised counts)."""
    preds = np.mod(np.asarray(preds, dtype=np.float64) + 3600.0, 360.0)
    indices = (preds / (360.0 / nbins)).astype(int) % nbins
    return np.bincount(indices, minlength=nbins).astype(np.float64)


def donut_heatmap(
    hm: np.ndarray,
    bg: tuple[int, int] = (201, 201),
    R: float = 50,
    zero_rad: float = np.deg2rad(-90),
    colormap=None,
    aapow: float | None = 40,
) -> np.ndarray:
    """Render the cyclic distribution ``hm`` (values ~[0,1]) as an RGBA donut image."""
    colormap = colormap or mpl.cm.Spectral_r
    h, w = bg
    y, x = np.mgrid[0:h, 0:w]
    cx, cy = w / 2.0, h / 2.0
    l = np.hypot(x - cx, y - cy)
    lc = (l - (w / 2.0 - R / 2.0)) / (R / 2.0)  # radial position within the band, in [-1, 1]
    inside = (lc > -1) & (lc < 1)
    angle = np.mod(np.rad2deg(np.arctan2(-(y - cy), x - cx) - zero_rad) + 360.0, 360.0)
    image = np.zeros((h, w, 4))
    values = np.clip(hm[(angle[inside] * len(hm) / 360.0).astype(int) % len(hm)], 0.0, 1.0)
    image[inside] = colormap(values)
    if aapow is not None:
        image[inside, 3] = 1 - (np.exp(lc[inside] ** aapow) - 1) / (np.e - 1)
    return image


def donut_from_angles(angles: np.ndarray, nbins: int = 3600, smooth: int = 41, per400: bool = True) -> np.ndarray:
    """Notebook-default pipeline: histogram -> cyclic gaussian smoothing -> scale by n/400."""
    hm = cyclic_filter(mkheatmap_deg(angles, nbins=nbins), gaussfilter(smooth))
    if per400:
        hm = hm / (len(angles) / 400.0)
    return hm


def predict_angles(checkpoint: str | Path, manifest: str | Path, split: str, device_name: str | None = None,
                   batch_size: int | None = None, num_workers: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Return (predicted_deg, target_deg) for every record of ``split``."""
    from .experiments import config_from_checkpoint, evaluation_target_kind, with_overrides
    from .losses import bit2deg
    from .models import build_model
    from .train import _eval_dataset, model_config

    data = torch.load(checkpoint, map_location="cpu")
    config = config_from_checkpoint(data["experiment"])
    if batch_size is not None:
        config = with_overrides(config, None, batch_size, None)
    if config.target_kind == "classification" or config.target_kind == "pose_rad":
        raise ValueError("donut plots need a single-angle experiment")
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    class_to_idx = data.get("class_to_idx", {})
    dataset = _eval_dataset(config, manifest, split, class_to_idx)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False, num_workers=num_workers)
    model = build_model(model_config(config, class_to_idx)).to(device)
    model.load_state_dict(data["model_state_dict"])
    model.eval()
    preds: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    kind = evaluation_target_kind(config)
    with torch.no_grad():
        for images, batch_targets in loader:
            outputs = model(images.to(device))
            if config.model_head == "biternion":
                preds.append(bit2deg(outputs).reshape(-1).cpu())
            else:
                out = outputs.reshape(-1)
                preds.append((torch.rad2deg(out) if config.target_kind == "angle_rad" else out).cpu())
            if kind == "biternion":
                targets.append(bit2deg(batch_targets).reshape(-1))
            elif kind == "angle_rad":
                targets.append(torch.rad2deg(batch_targets).reshape(-1))
            else:
                targets.append(batch_targets.reshape(-1))
    return torch.cat(preds).numpy() % 360.0, torch.cat(targets).numpy() % 360.0


def render_donut_figure(
    checkpoints: list[str | Path],
    manifest: str | Path,
    split: str,
    output: str | Path,
    labels: list[str] | None = None,
    nbins: int = 3600,
    smooth: int = 41,
    size: int = 201,
    ring: int = 50,
    device_name: str | None = None,
    num_workers: int = 0,
    dpi: int = 150,
) -> dict[str, Any]:
    """Ground-truth donut first (leftmost), then one donut per checkpoint's predictions; saved as JPEG."""
    labels = labels or [Path(c).parent.name or Path(c).stem for c in checkpoints]
    panels: list[tuple[str, np.ndarray, int]] = []
    target_deg: np.ndarray | None = None
    for label, checkpoint in zip(labels, checkpoints):
        pred_deg, target_deg = predict_angles(checkpoint, manifest, split, device_name, num_workers=num_workers)
        panels.append((label, pred_deg, len(pred_deg)))
    assert target_deg is not None
    panels.insert(0, ("ground truth", target_deg, len(target_deg)))

    fig, axes = plt.subplots(1, len(panels), figsize=(3.1 * len(panels), 3.6), facecolor=SURFACE, squeeze=False)
    for ax, (label, angles, n) in zip(axes[0], panels):
        ax.imshow(donut_heatmap(donut_from_angles(angles, nbins, smooth), bg=(size, size), R=ring))
        ax.set_title(label, fontsize=10, color=TEXT_PRIMARY)
        ax.text(0.5, -0.04, f"n={n:,}", transform=ax.transAxes, ha="center", fontsize=8, color=TEXT_SECONDARY)
        # notebook orientation: 0 deg at the bottom, counter-clockwise (90 right, 180 top, 270 left)
        for deg, (tx, ty, ha, va) in {0: (0.5, 0.04, "center", "top"), 90: (0.985, 0.5, "left", "center"),
                                      180: (0.5, 0.985, "center", "bottom"), 270: (0.015, 0.5, "right", "center")}.items():
            ax.text(tx, ty, f"{deg}°", transform=ax.transAxes, ha=ha, va=va, fontsize=7, color=TEXT_SECONDARY)
        ax.axis("off")
    fig.suptitle(f"Prediction distribution — split {split!r} · 0° bottom, CCW (notebook orientation)",
                 x=0.01, ha="left", fontsize=9, color=TEXT_SECONDARY, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, facecolor=SURFACE, format="jpg", pil_kwargs={"quality": 92})
    plt.close(fig)
    return {"output": str(output), "panels": [(label, n) for label, _, n in panels]}
