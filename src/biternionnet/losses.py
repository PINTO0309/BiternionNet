from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn


def deg2bit(angles_deg: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    """Convert degrees to normalized biternion vectors [cos(theta), sin(theta)]."""
    if isinstance(angles_deg, torch.Tensor):
        radians = torch.deg2rad(angles_deg)
        return torch.stack((torch.cos(radians), torch.sin(radians)), dim=-1)
    radians = np.deg2rad(angles_deg)
    return np.stack((np.cos(radians), np.sin(radians)), axis=-1)


def bit2deg(vectors: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    """Convert biternion vectors back to degrees in [0, 360)."""
    if isinstance(vectors, torch.Tensor):
        degrees = torch.rad2deg(torch.atan2(vectors[..., 1], vectors[..., 0]))
        return torch.remainder(degrees + 360.0, 360.0)
    degrees = np.rad2deg(np.arctan2(vectors[..., 1], vectors[..., 0]))
    return np.mod(degrees + 360.0, 360.0)


def angle_difference_deg(pred: np.ndarray | torch.Tensor, target: np.ndarray | torch.Tensor):
    """Smallest absolute angular difference in degrees."""
    if isinstance(pred, torch.Tensor) or isinstance(target, torch.Tensor):
        pred_t = pred if isinstance(pred, torch.Tensor) else torch.as_tensor(pred)
        target_t = target if isinstance(target, torch.Tensor) else torch.as_tensor(target)
        delta = torch.deg2rad(target_t - pred_t)
        return torch.rad2deg(torch.abs(torch.atan2(torch.sin(delta), torch.cos(delta))))
    delta = np.deg2rad(target - pred)
    return np.rad2deg(np.abs(np.arctan2(np.sin(delta), np.cos(delta))))


def cyclic_mae_deg(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return angle_difference_deg(pred, target).mean()


def normalize_biternion(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x / torch.clamp(torch.linalg.vector_norm(x, dim=-1, keepdim=True), min=eps)


def quantize_labels(values: np.ndarray, borders: np.ndarray) -> np.ndarray:
    """Quantize circular degree labels using notebook-compatible border semantics."""
    q = np.empty(values.shape, dtype=np.int64)
    for i in range(len(borders) - 1):
        left = borders[i]
        right = borders[i + 1]
        if left < right:
            mask = (left <= values) & (values < right)
        else:
            mask = (left <= values) | (values < right)
        q[mask] = i
    return q


class CosineLoss(nn.Module):
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = normalize_biternion(pred)
        target = normalize_biternion(target)
        return (1.0 - (pred * target).sum(dim=-1)).mean()


class VonMisesLoss(nn.Module):
    def __init__(self, kappa: float, radians: bool = True) -> None:
        super().__init__()
        self.kappa = kappa
        self.radians = radians

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        delta = pred - target
        if not self.radians:
            delta = torch.deg2rad(delta)
        constant = math.exp(2.0 * self.kappa)
        return (constant - torch.exp(self.kappa * (1.0 + torch.cos(delta)))).mean()


class VonMisesBiternionLoss(nn.Module):
    def __init__(self, kappa: float) -> None:
        super().__init__()
        self.kappa = kappa

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = normalize_biternion(pred)
        target = normalize_biternion(target)
        cos_angles = (pred * target).sum(dim=-1)
        return (1.0 - torch.exp(self.kappa * (cos_angles - 1.0))).mean()


class ModuloMAELoss(nn.Module):
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.remainder(torch.abs(pred - target), 360.0).mean()


def parabola_vertex(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Vertex (x, y) of the parabola through three points (notebook helper)."""
    x1, x2, x3 = (float(v) for v in x)
    y1, y2, y3 = (float(v) for v in y)
    denom = (x1 - x2) * (x1 - x3) * (x2 - x3)
    a = (x3 * (y2 - y1) + x2 * (y1 - y3) + x1 * (y3 - y2)) / denom
    b = (x3 * x3 * (y1 - y2) + x2 * x2 * (y3 - y1) + x1 * x1 * (y2 - y3)) / denom
    c = (x2 * x3 * (x2 - x3) * y1 + x3 * x1 * (x3 - x1) * y2 + x1 * x2 * (x1 - x2) * y3) / denom
    if a == 0.0:
        return x2, y2
    return -b / (2.0 * a), c - b * b / (4.0 * a)


def probs2deg_centre(probs: np.ndarray, centres: np.ndarray) -> np.ndarray:
    """Class probabilities -> angle in degrees using the most likely class centre."""
    return np.asarray(centres, dtype=np.float64)[np.argmax(probs, axis=-1)]


def prob2deg_quadint(prob: np.ndarray, centres: np.ndarray) -> float:
    """Quadratic interpolation between the best class and its two cyclic neighbours (paper Sec. 5)."""
    centres = np.asarray(centres, dtype=np.float64)
    prob = np.asarray(prob, dtype=np.float64)
    i = int(np.argmax(prob))
    n = len(centres)
    if i == 0:
        xs = [centres[-1] - 360.0, centres[0], centres[1]]
        ys = prob[[-1, 0, 1]]
    elif i == n - 1:
        xs = [centres[-2], centres[-1], centres[0] + 360.0]
        ys = prob[[-2, -1, 0]]
    else:
        xs = centres[[i - 1, i, i + 1]]
        ys = prob[[i - 1, i, i + 1]]
    return float(np.mod(parabola_vertex(np.asarray(xs), ys)[0], 360.0))


def probs2deg_quadint(probs: np.ndarray, centres: np.ndarray) -> np.ndarray:
    return np.array([prob2deg_quadint(p, centres) for p in np.asarray(probs)], dtype=np.float64)
