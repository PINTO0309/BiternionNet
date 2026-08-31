"""HRFFA ViT-L iBUG68 diagnostics for synthetic-image QA.

The crop and preprocessing contract is adapted from HRFFA at commit
1155c7f7b3f07c649c64f45516750f86ca0e7015 (MIT). The ViT-L weights are a
separate local asset subject to their upstream derived-work terms.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort

from .ort_policy import (
    ProviderSpec,
    build_provider_plan,
    require_batch_one,
    validate_model_batch_axis,
)

INPUT_SIZE = 320
CROP_PAD = 0.05
VISIBILITY_CONFIDENCE = 0.80
MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
VISIBILITY_NAMES = ("outside", "occluded", "visible")
IBUG68_GROUPS = {
    "jaw": tuple(range(0, 17)),
    "subject_right_brow": tuple(range(17, 22)),
    "subject_left_brow": tuple(range(22, 27)),
    "nose": tuple(range(27, 36)),
    "subject_right_eye": tuple(range(36, 42)),
    "subject_left_eye": tuple(range(42, 48)),
    "outer_mouth": tuple(range(48, 60)),
    "inner_mouth": tuple(range(60, 68)),
}


@dataclass(frozen=True)
class LandmarkResult:
    """One head's predictions in crop-normalized and source-image coordinates."""

    points_normalized: np.ndarray
    points_xy: np.ndarray
    visibility_logits: np.ndarray
    visibility_probabilities: np.ndarray
    visibility: np.ndarray
    crop_transform: np.ndarray
    crop_box_xyxy: tuple[float, float, float, float]
    crop_image_coverage_ratio: float


def crop_transform(
    head_bbox: list[float] | tuple[float, float, float, float],
    *,
    out_size: int = INPUT_SIZE,
    pad: float = CROP_PAD,
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """Return HRFFA's image-to-crop transform and its square source-image box."""
    if out_size <= 0:
        raise ValueError("HRFFA output size must be positive")
    if pad < 0:
        raise ValueError("HRFFA crop padding must be non-negative")
    x1, y1, x2, y2 = map(float, head_bbox)
    width, height = x2 - x1, y2 - y1
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid head bbox for HRFFA: {head_bbox}")
    centre_x, centre_y = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    side = max(width, height) * (1.0 + 2.0 * pad)
    scale = out_size / side
    half = out_size / 2.0
    transform = np.asarray(
        [
            [scale, 0.0, half - scale * centre_x],
            [0.0, scale, half - scale * centre_y],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    crop_box = (
        centre_x - side / 2.0,
        centre_y - side / 2.0,
        centre_x + side / 2.0,
        centre_y + side / 2.0,
    )
    return transform, crop_box


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    values = np.exp(shifted)
    return values / np.sum(values, axis=-1, keepdims=True)


def _crop_coverage(
    crop_box: tuple[float, float, float, float], image_width: int, image_height: int
) -> float:
    x1, y1, x2, y2 = crop_box
    side_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    intersection = max(0.0, min(x2, image_width) - max(x1, 0.0)) * max(
        0.0, min(y2, image_height) - max(y1, 0.0)
    )
    return intersection / side_area if side_area else 0.0


class HRFFAViTL:
    """Run the static-batch HRFFA ViT-L iBUG68 ONNX graph on a DEIM head box."""

    def __init__(
        self,
        model_path: Path,
        providers: list[ProviderSpec] | None = None,
    ) -> None:
        options = ort.SessionOptions()
        options.log_severity_level = 4
        self.session = ort.InferenceSession(
            str(model_path),
            sess_options=options,
            providers=(
                providers
                if providers is not None
                else build_provider_plan(model_path).providers
            ),
        )
        model_input = self.session.get_inputs()[0]
        shape = model_input.shape
        validate_model_batch_axis(shape, model_name="HRFFA ViT-L", allow_dynamic=False)
        if (
            len(shape) != 4
            or shape[1] != 3
            or not isinstance(shape[2], int)
            or shape[2] != shape[3]
        ):
            raise ValueError(f"HRFFA input must be [N,3,S,S] with fixed square S, got {shape}")
        if int(shape[2]) != INPUT_SIZE:
            raise ValueError(f"HRFFA ViT-L QA requires {INPUT_SIZE}x{INPUT_SIZE} input, got {shape}")
        output_names = {output.name for output in self.session.get_outputs()}
        if not {"points", "vis_logits"}.issubset(output_names):
            raise ValueError(
                "HRFFA model must provide `points` and `vis_logits`, "
                f"got {sorted(output_names)}"
            )
        self.input_name = model_input.name
        self.input_size = int(shape[2])
        self.crop_pad = CROP_PAD
        self.execution_providers = self.session.get_providers()

    def preprocess(
        self, image_bgr: np.ndarray, head_bbox: list[float]
    ) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float, float]]:
        transform, crop_box = crop_transform(
            head_bbox, out_size=self.input_size, pad=self.crop_pad
        )
        crop = cv2.warpPerspective(
            image_bgr,
            transform,
            (self.input_size, self.input_size),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32)
        tensor = ((rgb.transpose(2, 0, 1) / 255.0) - MEAN) / STD
        return tensor[None].astype(np.float32), transform, crop_box

    def infer(self, image_bgr: np.ndarray, head_bbox: list[float]) -> LandmarkResult:
        tensor, transform, crop_box = self.preprocess(image_bgr, head_bbox)
        require_batch_one(tensor, model_name="HRFFA ViT-L")
        points, visibility_logits = self.session.run(
            ["points", "vis_logits"], {self.input_name: tensor}
        )
        if points.shape != (1, 68, 2) or visibility_logits.shape != (1, 68, 3):
            raise ValueError(
                "unexpected HRFFA outputs: "
                f"points={points.shape}, vis_logits={visibility_logits.shape}"
            )
        normalized = points[0].astype(np.float64)
        logits = visibility_logits[0].astype(np.float64)
        if not np.isfinite(normalized).all() or not np.isfinite(logits).all():
            raise ValueError("HRFFA returned non-finite landmarks or visibility logits")
        crop_xy = normalized * self.input_size
        homogeneous = np.concatenate(
            [crop_xy, np.ones((len(crop_xy), 1), dtype=np.float64)], axis=1
        )
        image_xy = (np.linalg.inv(transform) @ homogeneous.T).T[:, :2]
        probabilities = _softmax(logits)
        image_height, image_width = image_bgr.shape[:2]
        return LandmarkResult(
            points_normalized=normalized.astype(np.float32),
            points_xy=image_xy.astype(np.float32),
            visibility_logits=logits.astype(np.float32),
            visibility_probabilities=probabilities.astype(np.float32),
            visibility=np.argmax(logits, axis=1).astype(np.int64),
            crop_transform=transform,
            crop_box_xyxy=crop_box,
            crop_image_coverage_ratio=_crop_coverage(crop_box, image_width, image_height),
        )


def landmark_annotation(
    result: LandmarkResult,
    image_shape: tuple[int, ...],
    *,
    confidence_threshold: float = VISIBILITY_CONFIDENCE,
) -> dict[str, Any]:
    """Build calibration-oriented diagnostics without imposing an acceptance gate."""
    points = result.points_normalized.astype(np.float64)
    points_xy = result.points_xy.astype(np.float64)
    probabilities = result.visibility_probabilities.astype(np.float64)
    visibility = result.visibility.astype(np.int64)
    image_height, image_width = image_shape[:2]
    visible_probability = probabilities[:, 2]
    high_confidence_visible = visible_probability >= confidence_threshold
    within_crop = np.logical_and(points >= 0.0, points <= 1.0).all(axis=1)
    within_image = (
        (points_xy[:, 0] >= 0.0)
        & (points_xy[:, 0] < image_width)
        & (points_xy[:, 1] >= 0.0)
        & (points_xy[:, 1] < image_height)
    )
    groups: dict[str, dict[str, Any]] = {}
    for name, indices in IBUG68_GROUPS.items():
        selected = np.asarray(indices, dtype=np.int64)
        groups[name] = {
            "mean_visible_probability": round(float(np.mean(visible_probability[selected])), 5),
            "high_confidence_visible_count": int(np.sum(high_confidence_visible[selected])),
            "point_count": len(indices),
        }
    right_eye = groups["subject_right_eye"]["mean_visible_probability"]
    left_eye = groups["subject_left_eye"]["mean_visible_probability"]
    counts = {name: int(np.sum(visibility == index)) for index, name in enumerate(VISIBILITY_NAMES)}
    eye_centres = (
        np.mean(points[np.asarray(IBUG68_GROUPS["subject_right_eye"])], axis=0),
        np.mean(points[np.asarray(IBUG68_GROUPS["subject_left_eye"])], axis=0),
    )
    return {
        "landmark_status": "ok",
        "hrffa_points_xy": np.round(points_xy, 3).tolist(),
        "hrffa_points_normalized": np.round(points, 6).tolist(),
        "hrffa_visibility": visibility.tolist(),
        "hrffa_visibility_probabilities": np.round(probabilities, 5).tolist(),
        "hrffa_visibility_counts": counts,
        "hrffa_high_confidence_visible_count": int(np.sum(high_confidence_visible)),
        "hrffa_visibility_confidence_threshold": float(confidence_threshold),
        "hrffa_mean_visible_probability": round(float(np.mean(visible_probability)), 5),
        "hrffa_landmark_groups": groups,
        "hrffa_subject_left_minus_right_eye_visibility": round(float(left_eye - right_eye), 5),
        "hrffa_nose_tip_x_offset": round(float(points[30, 0] - 0.5), 6),
        "hrffa_inter_eye_distance": round(float(np.linalg.norm(eye_centres[1] - eye_centres[0])), 6),
        "hrffa_point_span_xy": np.round(np.ptp(points, axis=0), 6).tolist(),
        "hrffa_points_within_crop_ratio": round(float(np.mean(within_crop)), 5),
        "hrffa_points_within_image_ratio": round(float(np.mean(within_image)), 5),
        "hrffa_crop_pad": CROP_PAD,
        "hrffa_crop_box_xyxy": [round(float(value), 3) for value in result.crop_box_xyxy],
        "hrffa_crop_image_coverage_ratio": round(float(result.crop_image_coverage_ratio), 5),
        "hrffa_diagnostic_only": True,
    }
