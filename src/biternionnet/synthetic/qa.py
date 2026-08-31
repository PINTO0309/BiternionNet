"""Machine QA, camera-elevation classification, and hash-bound human review."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError

from .detector import CLASS_BODY, CLASS_HEAD, DIR8_CLASSES, Deimv2Detector
from .generate import (
    DEIM_CROP_MARGIN,
    DIR8_BY_CENTRE,
    ORIENTATION_BY_CENTRE,
    PipelineError,
    circular_error_deg,
    load_config,
    load_state,
    read_plan,
    sha256_file,
    validate_evaluation_protocol,
    wrap180,
    write_jsonl,
)
from .landmarks import HRFFAViTL, landmark_annotation
from .ort_policy import OnnxProviderPlan, build_provider_plan
from .pose import SixDRepNet360

REVIEW_COLUMNS = [
    "custom_id",
    "filename",
    "abs_pan_bin",
    "intent_pan_deg",
    "estimated_pan_deg",
    "direction",
    "landmark_status",
    "hrffa_high_confidence_visible_count",
    "hrffa_core_face_high_conf_visible_count",
    "landmark_alignment",
    "photorealism",
    "intent_match",
    "framing",
    "head_neck_shoulders_integrity",
    "camera_elevation_class",
    "crop_margin",
    "notes",
    "reviewed_sha256",
]
REVIEW_INSTRUCTIONS_NAME = "human_review_instructions.md"
PASS_FAIL = {"pass", "fail"}
INTENT_VALUES = {"match", "off-by-one-bin", "wrong"}
ELEVATION_VALUES = {"high_angle_match", "eye_level_or_low_angle", "unresolved"}
LANDMARK_ALIGNMENT_VALUES = {"match", "mismatch", "unresolved"}
QA_POLICY_SCHEMA_VERSION = 1
QA_POLICY_KEYS = {
    "schema_version",
    "pan_tolerance_deg",
    "enforce_head_height_ratio",
    "deim_direction_max_bin_distance",
    "deim_crop_margin",
}
DIRECTION_INDEX = {
    name: index for index, (_, name) in enumerate(sorted(DIR8_BY_CENTRE.items()))
}


def load_qa_policy(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        policy = yaml.safe_load(stream)
    if not isinstance(policy, dict) or set(policy) != QA_POLICY_KEYS:
        raise PipelineError(f"QA policy must define exactly {sorted(QA_POLICY_KEYS)}")
    if int(policy["schema_version"]) != QA_POLICY_SCHEMA_VERSION:
        raise PipelineError("unsupported QA policy schema version")
    tolerance = float(policy["pan_tolerance_deg"])
    if not 0.0 < tolerance <= 90.0:
        raise PipelineError("QA pan tolerance must be in (0, 90]")
    if not isinstance(policy["enforce_head_height_ratio"], bool):
        raise PipelineError("QA enforce_head_height_ratio must be boolean")
    direction_distance = policy["deim_direction_max_bin_distance"]
    if not isinstance(direction_distance, int) or not 0 <= direction_distance < 4:
        raise PipelineError(
            "QA DEIM direction bin distance must be an integer in [0, 3]"
        )
    crop_margin = float(policy["deim_crop_margin"])
    if not math.isclose(crop_margin, DEIM_CROP_MARGIN, abs_tol=1e-12):
        raise PipelineError(
            f"QA DEIM crop margin must be fixed to {DEIM_CROP_MARGIN:.2f}"
        )
    return {
        "schema_version": QA_POLICY_SCHEMA_VERSION,
        "pan_tolerance_deg": tolerance,
        "enforce_head_height_ratio": policy["enforce_head_height_ratio"],
        "deim_direction_max_bin_distance": direction_distance,
        "deim_crop_margin": DEIM_CROP_MARGIN,
    }


def effective_qa_config(
    config: dict[str, Any], policy_path: Path | None
) -> tuple[dict[str, Any], dict[str, Any]]:
    effective = deepcopy(config)
    if policy_path is None:
        policy = {
            "schema_version": QA_POLICY_SCHEMA_VERSION,
            "pan_tolerance_deg": float(config["qa"]["pan_tolerance_deg"]),
            "enforce_head_height_ratio": bool(
                config["qa"].get("enforce_head_height_ratio", True)
            ),
            "deim_direction_max_bin_distance": int(
                config["qa"].get("deim_direction_max_bin_distance", 1)
            ),
            "deim_crop_margin": float(
                config["qa"].get("deim_crop_margin", DEIM_CROP_MARGIN)
            ),
        }
        source = {
            **policy,
            "source": "generation_config",
            "path": None,
            "sha256": None,
        }
    else:
        policy_path = policy_path.resolve()
        if not policy_path.exists():
            raise PipelineError(f"QA policy not found: {policy_path}")
        policy = load_qa_policy(policy_path)
        source = {
            **policy,
            "source": "qa_policy_override",
            "path": str(policy_path),
            "sha256": sha256_file(policy_path),
        }
    effective["qa"]["pan_tolerance_deg"] = policy["pan_tolerance_deg"]
    effective["qa"]["enforce_head_height_ratio"] = policy["enforce_head_height_ratio"]
    effective["qa"]["deim_direction_max_bin_distance"] = policy[
        "deim_direction_max_bin_distance"
    ]
    effective["qa"]["deim_crop_margin"] = policy["deim_crop_margin"]
    return effective, source


def config_from_recorded_qa_policy(
    config: dict[str, Any], qa_report: dict[str, Any]
) -> dict[str, Any]:
    recorded = qa_report.get("qa_policy")
    if not isinstance(recorded, dict):
        raise PipelineError("QA report does not bind an effective QA policy")
    source = recorded.get("source")
    if source == "generation_config":
        effective, verified = effective_qa_config(config, None)
    elif source == "qa_policy_override":
        path_value = recorded.get("path")
        if not isinstance(path_value, str) or not path_value:
            raise PipelineError("QA policy override path is missing")
        effective, verified = effective_qa_config(config, Path(path_value))
    else:
        raise PipelineError("QA report has an unsupported QA policy source")
    if verified != recorded:
        raise PipelineError("QA policy changed after automatic QA")
    return effective


def _iou(left: list[float], right: list[float]) -> float:
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_left = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    area_right = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = area_left + area_right - intersection
    return intersection / union if union else 0.0


def direction_bin_distance(
    direction: str | None, expected_direction: str | None
) -> int | None:
    if direction not in DIRECTION_INDEX or expected_direction not in DIRECTION_INDEX:
        return None
    difference = abs(DIRECTION_INDEX[direction] - DIRECTION_INDEX[expected_direction])
    return min(difference, len(DIRECTION_INDEX) - difference)


def direction_consistent(
    direction: str | None,
    expected_direction: str | None,
    *,
    max_bin_distance: int = 1,
) -> bool:
    distance = direction_bin_distance(direction, expected_direction)
    return distance is not None and distance <= max_bin_distance


def _detection_annotation(
    image: np.ndarray, detections: list[list[float]]
) -> dict[str, Any]:
    height, width = image.shape[:2]
    heads = [row for row in detections if int(row[0]) == CLASS_HEAD]
    bodies = [row for row in detections if int(row[0]) == CLASS_BODY]
    if not heads:
        return {
            "detector_status": "no_head",
            "head_count": 0,
            "body_count": len(bodies),
            "direction": None,
        }
    head = max(heads, key=lambda row: row[5])
    box = [float(value) for value in head[1:5]]
    head_width, head_height = box[2] - box[0], box[3] - box[1]
    directions = [row for row in detections if int(row[0]) in DIR8_CLASSES]
    direction = (
        max(directions, key=lambda row: _iou(box, row[1:5])) if directions else None
    )
    body = max(bodies, key=lambda row: row[5]) if bodies else None
    return {
        "detector_status": "ok",
        "head_count": len(heads),
        "body_count": len(bodies),
        "head_box_xyxy": [round(value, 2) for value in box],
        "head_score": float(head[5]),
        "body_box_xyxy": [round(float(value), 2) for value in body[1:5]]
        if body
        else None,
        "body_score": float(body[5]) if body else None,
        "head_height_ratio": round(head_height / height, 5),
        "margin_left_head_ratio": round(box[0] / head_width, 5) if head_width else 0.0,
        "margin_right_head_ratio": round((width - box[2]) / head_width, 5)
        if head_width
        else 0.0,
        "margin_top_head_ratio": round(box[1] / head_height, 5) if head_height else 0.0,
        "margin_bottom_head_ratio": round((height - box[3]) / head_height, 5)
        if head_height
        else 0.0,
        "direction": DIR8_CLASSES[int(direction[0])] if direction else None,
        "direction_score": float(direction[5]) if direction else None,
    }


def quality_reasons(
    row: dict[str, Any], config: dict[str, Any]
) -> tuple[list[str], bool]:
    qa = config["qa"]
    reasons: list[str] = []
    if not row.get("exists"):
        reasons.append("image_missing")
    elif not row.get("image_valid"):
        reasons.append("invalid_image")
    elif not row.get("dimension_match"):
        reasons.append("wrong_dimensions")
    if qa["reject_duplicates"] and row.get("duplicate_of"):
        reasons.append("duplicate_image")
    detector_status = row.get("detector_status")
    detector_complete = detector_status in {"ok", "no_head"}
    if detector_status == "no_head":
        reasons.append("head_not_detected")
    elif detector_status == "ok":
        if qa["require_single_head"] and row.get("head_count") != 1:
            reasons.append("head_count_not_one")
        if qa.get("enforce_head_height_ratio", True):
            limits = qa["head_height_ratio"]
            ratio = row.get("head_height_ratio")
            if ratio is None or float(ratio) < float(limits["min"]):
                reasons.append("head_too_small")
            elif float(ratio) > float(limits["max"]):
                reasons.append("head_too_large")
        margin_limit = float(qa["margin_min_head_ratio"])
        margins = [
            row.get("margin_left_head_ratio"),
            row.get("margin_right_head_ratio"),
            row.get("margin_top_head_ratio"),
            row.get("margin_bottom_head_ratio"),
        ]
        if any(value is None or float(value) < margin_limit for value in margins):
            reasons.append("insufficient_margin")
        # Body detection is used only to require neck/shoulder/upper-torso context.
        # Lower-body anatomy is outside this head-crop QA contract.
        if qa["require_body_detection"] and not row.get("body_count"):
            reasons.append("body_not_detected")
        if not row.get("direction_consistent"):
            reasons.append("direction_conflict")
    return reasons, detector_complete


def calibrate_pitch(
    rows: list[dict[str, Any]], config: dict[str, Any]
) -> dict[str, Any]:
    eligible = [
        row
        for row in rows
        # SixDRepNet360's Euler pitch folds near a side profile (observed around
        # 147..171 degrees at abs pan 90), so only the stable front/three-quarter
        # range may calibrate camera-relative elevation.
        if int(row["abs_pan_bin"]) <= 60
        and row.get("pose_status") == "ok"
        and row.get("quality_gate_pass")
    ]
    elevation = config["qa"]["elevation"]
    minimum = 6
    if len(eligible) < minimum:
        return {
            "valid": False,
            "reason": f"need at least {minimum} stable front/three-quarter records, got {len(eligible)}",
            "sample_count": len(eligible),
            "maximum_abs_pan_deg": 60,
        }
    residuals = np.asarray(
        [float(row["pitch_residual_deg"]) for row in eligible], dtype=np.float64
    )
    bias = float(np.median(residuals))
    centred = np.abs(residuals - bias)
    q = float(np.quantile(centred, float(elevation["quantile"]), method="linear"))
    data_threshold = q + float(elevation["buffer_deg"])
    maximum = float(elevation["maximum_threshold_deg"])
    threshold = max(float(elevation["minimum_threshold_deg"]), data_threshold)
    pitch_by_elevation: dict[str, float] = {}
    for requested in sorted({float(row["camera_elevation"]) for row in eligible}):
        group = [
            float(row["sixd_pitch_deg"])
            for row in eligible
            if float(row["camera_elevation"]) == requested
        ]
        pitch_by_elevation[f"{requested:g}"] = float(np.median(group))
    trend_valid: bool | None = None
    if len(pitch_by_elevation) >= 2:
        ordered = sorted(
            (float(key), value) for key, value in pitch_by_elevation.items()
        )
        trend_valid = ordered[-1][1] < ordered[0][1]
    valid = data_threshold <= maximum and trend_valid is not False
    return {
        "valid": valid,
        "reason": (
            None
            if valid
            else (
                "requested camera elevation did not shift median pitch in the negative direction"
                if trend_valid is False
                else f"data threshold {data_threshold:.3f} exceeds hard maximum {maximum:.3f}"
            )
        ),
        "sample_count": len(eligible),
        "maximum_abs_pan_deg": 60,
        "bias_deg": bias,
        "centred_q_deg": q,
        "data_threshold_deg": data_threshold,
        "threshold_deg": threshold if valid else None,
        "hard_maximum_deg": maximum,
        "median_pitch_by_camera_elevation_deg": pitch_by_elevation,
        "negative_pitch_trend_valid": trend_valid,
        "method": "median_bias_then_q90_abs_residual_plus_5deg",
    }


def rear_reliability_policy(
    qa_rows: dict[str, dict[str, Any]],
    review_rows: list[dict[str, str]],
    config: dict[str, Any],
    *,
    stage: str,
) -> dict[str, Any]:
    tolerance = float(config["qa"]["pan_tolerance_deg"])
    agreement_min = float(config["qa"]["rear_sixd_min_agreement"])
    conflict_max = float(config["qa"]["rear_deim_max_conflict"])
    grouped: dict[str, list[dict[str, Any]]] = {}
    for review in review_rows:
        qa = qa_rows[review["custom_id"]]
        if int(qa["abs_pan_bin"]) <= 90:
            continue
        grouped.setdefault(str(qa["expected_direction"]), []).append(qa)
    sectors: dict[str, dict[str, Any]] = {}
    for sector, rows in sorted(grouped.items()):
        count = len(rows)
        agreement = (
            sum(
                row.get("pose_status") == "ok"
                and float(row.get("pan_error_deg", 999.0)) <= tolerance
                for row in rows
            )
            / count
        )
        conflicts = (
            sum(not bool(row.get("direction_consistent")) for row in rows) / count
        )
        sectors[sector] = {
            "reviewed": count,
            "sixd_within_tolerance_ratio": agreement,
            "deim_conflict_ratio": conflicts,
            "sixd_allowed": stage == "pilot"
            and agreement >= agreement_min
            and conflicts <= conflict_max,
        }
    return {
        "schema_version": 1,
        "source_stage": stage,
        "pan_tolerance_deg": tolerance,
        "minimum_sixd_agreement": agreement_min,
        "maximum_deim_conflict": conflict_max,
        "sectors": sectors,
        "fallback": "intent_rear_with_deim_and_human_intent_review",
    }


def classify_elevation(
    row: dict[str, Any], calibration: dict[str, Any], config: dict[str, Any]
) -> tuple[str, bool]:
    if (
        int(row.get("abs_pan_bin", 999))
        > int(calibration.get("maximum_abs_pan_deg", 60))
        or row.get("pose_status") != "ok"
        or not calibration.get("valid")
    ):
        return "unresolved", False
    residual_error = abs(
        float(row["pitch_residual_deg"]) - float(calibration["bias_deg"])
    )
    requested_camera = float(row["camera_elevation"])
    pitch = float(row["sixd_pitch_deg"])
    ratio = float(config["qa"]["elevation"]["minimum_negative_camera_ratio"])
    high = (
        residual_error <= float(calibration["threshold_deg"])
        and pitch <= -ratio * requested_camera
    )
    return ("high_angle_match", True) if high else ("eye_level_or_low_angle", False)


def summarize_landmarks(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize HRFFA signals for later human calibration, without accepting/rejecting rows."""
    status_counts = Counter(str(row.get("landmark_status", "missing")) for row in rows)
    usable = [row for row in rows if row.get("landmark_status") == "ok"]

    def metrics(group: list[dict[str, Any]]) -> dict[str, Any]:
        if not group:
            return {"count": 0}
        return {
            "count": len(group),
            "median_high_confidence_visible_count": float(
                np.median([row["hrffa_high_confidence_visible_count"] for row in group])
            ),
            "median_core_face_high_conf_visible_count": float(
                np.median(
                    [row["hrffa_core_face_high_conf_visible_count"] for row in group]
                )
            ),
            "median_mean_visible_probability": float(
                np.median([row["hrffa_mean_visible_probability"] for row in group])
            ),
            "median_points_within_crop_ratio": float(
                np.median([row["hrffa_points_within_crop_ratio"] for row in group])
            ),
            "median_left_minus_right_eye_visibility": float(
                np.median(
                    [
                        row["hrffa_subject_left_minus_right_eye_visibility"]
                        for row in group
                    ]
                )
            ),
            "median_nose_tip_x_offset": float(
                np.median([row["hrffa_nose_tip_x_offset"] for row in group])
            ),
        }

    sectors = {
        direction: metrics(
            [row for row in usable if row["expected_direction"] == direction]
        )
        for direction in sorted({str(row["expected_direction"]) for row in rows})
    }
    return {
        "mode": "diagnostic_only",
        "hard_gate_active": False,
        "promotion_requirement": (
            "calibrate against human-reviewed Pilot by direction/pan sector before enabling gates"
        ),
        "status_counts": dict(status_counts),
        "overall": metrics(usable),
        "by_expected_direction": sectors,
    }


def landmark_human_calibration(
    qa_rows: dict[str, dict[str, Any]],
    review_rows: list[dict[str, str]],
    *,
    stage: str,
) -> dict[str, Any]:
    """Join diagnostics to explicit human landmark-overlay review without choosing thresholds."""
    by_label: dict[str, list[dict[str, Any]]] = {
        value: [] for value in sorted(LANDMARK_ALIGNMENT_VALUES)
    }
    by_direction: dict[str, Counter[str]] = {}
    for review in review_rows:
        qa = qa_rows[review["custom_id"]]
        label = review["landmark_alignment"]
        by_label[label].append(qa)
        by_direction.setdefault(str(qa["expected_direction"]), Counter())[label] += 1
    return {
        "schema_version": 1,
        "source_stage": stage,
        "mode": "diagnostic_only",
        "hard_gate_active": False,
        "review_definition": (
            "match means the overlay follows the actual visible facial structure; "
            "unresolved is expected for genuine rear views"
        ),
        "metrics_by_human_alignment": {
            label: summarize_landmarks(group)["overall"]
            for label, group in by_label.items()
        },
        "review_counts_by_expected_direction": {
            direction: dict(counts)
            for direction, counts in sorted(by_direction.items())
        },
        "promotion_requirement": (
            "define and preregister sector-specific thresholds from human-reviewed Pilot, then "
            "measure false accepts and false rejects before enabling a hard gate"
        ),
    }


def run_auto_qa(
    run_dir: Path,
    *,
    detector_model: Path | None,
    pose_model: Path | None,
    landmark_model: Path | None = None,
    calibration_path: Path | None = None,
    rear_policy_path: Path | None = None,
    qa_policy_path: Path | None = None,
    cpu: bool = False,
) -> dict[str, Any]:
    state = load_state(run_dir)
    generation_config = load_config(Path(state["config_path"]))
    config, qa_policy = effective_qa_config(generation_config, qa_policy_path)
    plan = list(read_plan(run_dir, state).values())
    detector = None
    detector_provider_plan: OnnxProviderPlan | None = None
    detector_sha256 = None
    if detector_model is not None:
        if not detector_model.exists():
            raise PipelineError(f"DEIM model not found: {detector_model}")
        detector_sha256 = sha256_file(detector_model)
        if detector_sha256 != config["models"]["deimv2"]["sha256"]:
            raise PipelineError(
                "DEIM model SHA-256 does not match the configured QA asset"
            )
        detector_provider_plan = build_provider_plan(
            detector_model,
            model_sha256=detector_sha256,
            force_cpu=cpu,
            allow_tensorrt=False,
        )
        detector = Deimv2Detector(
            detector_model,
            providers=detector_provider_plan.providers,
            score_threshold=float(config["qa"]["detector_score_threshold"]),
        )
    pose = None
    pose_provider_plan: OnnxProviderPlan | None = None
    pose_sha256 = None
    if pose_model is not None:
        if not pose_model.exists():
            raise PipelineError(f"SixD model not found: {pose_model}")
        pose_sha256 = sha256_file(pose_model)
        if pose_sha256 != config["models"]["sixdrepnet360"]["sha256"]:
            raise PipelineError(
                "SixD model SHA-256 does not match the configured QA asset"
            )
        pose_provider_plan = build_provider_plan(
            pose_model, model_sha256=pose_sha256, force_cpu=cpu
        )
        pose = SixDRepNet360(pose_model, providers=pose_provider_plan.providers)
    landmarks = None
    landmark_provider_plan: OnnxProviderPlan | None = None
    landmark_sha256 = None
    if landmark_model is not None:
        if not landmark_model.exists():
            raise PipelineError(f"HRFFA model not found: {landmark_model}")
        landmark_sha256 = sha256_file(landmark_model)
        if landmark_sha256 != config["models"]["hrffa_vitl_ibug68"]["sha256"]:
            raise PipelineError(
                "HRFFA model SHA-256 does not match the configured QA asset"
            )
        landmark_provider_plan = build_provider_plan(
            landmark_model, model_sha256=landmark_sha256, force_cpu=cpu
        )
        landmarks = HRFFAViTL(
            landmark_model, providers=landmark_provider_plan.providers
        )
    images: dict[str, np.ndarray] = {}
    rows: list[dict[str, Any]] = []
    for record in plan:
        path = run_dir / "images" / record["filename"]
        row: dict[str, Any] = {
            "custom_id": record["custom_id"],
            "filename": record["filename"],
            "abs_pan_bin": record["abs_pan_bin"],
            "intent_pan_deg": record["intent_pan_deg"],
            "camera_elevation": record["camera_elevation"],
            "expected_direction": record["expected_direction"],
            "exists": path.exists(),
            "image_valid": False,
            "dimension_match": False,
            "actual_size": None,
            "sha256": None,
            "duplicate_of": None,
        }
        if path.exists():
            try:
                with Image.open(path) as image:
                    image.verify()
                with Image.open(path) as image:
                    row["image_valid"] = image.format == "JPEG"
                    row["actual_size"] = f"{image.width}x{image.height}"
                row["dimension_match"] = row["actual_size"] == record["size"]
                row["sha256"] = sha256_file(path)
                image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
                if image_bgr is not None:
                    images[record["custom_id"]] = image_bgr
            except (OSError, UnidentifiedImageError):
                pass
        rows.append(row)
    if detector is not None:
        present = [record for record in plan if record["custom_id"] in images]
        for record in present:
            found = detector.infer(images[record["custom_id"]])
            rows[record["serial"] - 1].update(
                _detection_annotation(images[record["custom_id"]], found)
            )
    else:
        for row in rows:
            row["detector_status"] = "skipped"
    hashes: dict[str, str] = {}
    for row in rows:
        digest = row.get("sha256")
        row["duplicate_of"] = hashes.get(digest) if digest else None
        if digest:
            hashes.setdefault(digest, row["custom_id"])
        row["direction_bin_distance"] = direction_bin_distance(
            row.get("direction"), row.get("expected_direction")
        )
        row["deim_direction_max_bin_distance"] = int(
            config["qa"]["deim_direction_max_bin_distance"]
        )
        row["direction_consistent"] = direction_consistent(
            row.get("direction"),
            row.get("expected_direction"),
            max_bin_distance=row["deim_direction_max_bin_distance"],
        )
        row["qa_pan_tolerance_deg"] = float(config["qa"]["pan_tolerance_deg"])
        row["head_height_ratio_gate_active"] = bool(
            config["qa"].get("enforce_head_height_ratio", True)
        )
        reasons, detector_complete = quality_reasons(row, config)
        row["quality_gate_reasons"] = reasons
        row["quality_gate_complete"] = detector_complete
        row["quality_gate_pass"] = detector_complete and not reasons
        row["pose_status"] = "skipped"
        row["landmark_status"] = "skipped"
        if pose is not None and row.get("head_box_xyxy") and row["custom_id"] in images:
            yaw, pitch, roll = pose.infer(
                images[row["custom_id"]], row["head_box_xyxy"]
            )
            if all(math.isfinite(value) for value in (yaw, pitch, roll)):
                estimated = (-yaw) % 360.0
                row.update(
                    {
                        "pose_status": "ok",
                        "sixd_yaw_deg": round(yaw, 4),
                        "sixd_pitch_deg": round(pitch, 4),
                        "sixd_roll_deg": round(roll, 4),
                        "estimated_pan_deg": round(estimated, 4),
                        "pan_error_deg": round(
                            circular_error_deg(estimated, row["intent_pan_deg"]), 4
                        ),
                        "pitch_expected_deg": -float(row["camera_elevation"]),
                        "pitch_residual_deg": round(
                            wrap180(pitch + float(row["camera_elevation"])), 4
                        ),
                    }
                )
            else:
                row["pose_status"] = "invalid"
        if (
            landmarks is not None
            and row.get("head_box_xyxy")
            and row["custom_id"] in images
        ):
            try:
                result = landmarks.infer(images[row["custom_id"]], row["head_box_xyxy"])
                row.update(landmark_annotation(result, images[row["custom_id"]].shape))
                groups = row["hrffa_landmark_groups"]
                row["hrffa_core_face_high_conf_visible_count"] = sum(
                    int(groups[name]["high_confidence_visible_count"])
                    for name in (
                        "subject_right_eye",
                        "subject_left_eye",
                        "nose",
                        "outer_mouth",
                        "inner_mouth",
                    )
                )
            except ValueError as exc:
                row["landmark_status"] = "invalid"
                row["landmark_error"] = str(exc)
    if state["stage"] == "validation":
        calibration = calibrate_pitch(rows, config)
        calibration.update(
            {
                "stage": "validation",
                "run_id": state["local_batch_id"],
                "created_at": datetime.now(UTC).isoformat(),
                "qa_policy": qa_policy,
            }
        )
        calibration_file = run_dir / "pitch_calibration.json"
        calibration_file.write_text(
            json.dumps(calibration, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    else:
        if calibration_path is None or not calibration_path.exists():
            raise PipelineError(
                "non-validation QA requires --calibration from approved Validation"
            )
        calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
        if not calibration.get("valid"):
            raise PipelineError("pitch calibration is not valid")
        if state.get("parent_batch_dir"):
            parent_approval = json.loads(
                (Path(state["parent_batch_dir"]) / "approval.json").read_text(
                    encoding="utf-8"
                )
            )
            if parent_approval.get("pitch_calibration_sha256") != sha256_file(
                calibration_path
            ):
                raise PipelineError(
                    "QA calibration does not match the approved parent calibration"
                )
    rear_policy: dict[str, Any] | None = None
    if state["stage"] in {"floor_120", "uniform_200"}:
        if rear_policy_path is None or not rear_policy_path.exists():
            raise PipelineError(
                "quota-fill QA requires --rear-policy from approved Pilot"
            )
        rear_policy = json.loads(rear_policy_path.read_text(encoding="utf-8"))
        if rear_policy.get("source_stage") != "pilot":
            raise PipelineError("rear policy must come from Pilot")
        parent_approval = json.loads(
            (Path(state["parent_batch_dir"]) / "approval.json").read_text(
                encoding="utf-8"
            )
        )
        if parent_approval.get("rear_label_policy_sha256") != sha256_file(
            rear_policy_path
        ):
            raise PipelineError("rear policy does not match approved Pilot")
    tolerance = float(config["qa"]["pan_tolerance_deg"])
    for row in rows:
        elevation_class, counts = classify_elevation(row, calibration, config)
        row["camera_elevation_class_auto"] = elevation_class
        row["counts_toward_high_angle_quota_auto"] = counts
        if int(row["abs_pan_bin"]) <= 90:
            pose_pan_pass = (
                row.get("pose_status") == "ok"
                and float(row["pan_error_deg"]) <= tolerance
            )
            row["label_source_auto"] = "sixdrepnet360" if pose_pan_pass else None
            row["angle_deg_auto"] = (
                row.get("estimated_pan_deg") if pose_pan_pass else None
            )
            row["pan_quality_pass_auto"] = bool(
                row["quality_gate_pass"] and pose_pan_pass
            )
        else:
            sector_policy = (
                (rear_policy or {})
                .get("sectors", {})
                .get(str(row["expected_direction"]), {})
            )
            use_sixd = bool(
                sector_policy.get("sixd_allowed")
                and row.get("pose_status") == "ok"
                and float(row.get("pan_error_deg", 999.0)) <= tolerance
            )
            row["label_source_auto"] = (
                "sixdrepnet360_rear_pilot_validated" if use_sixd else "intent_rear"
            )
            row["angle_deg_auto"] = (
                row.get("estimated_pan_deg") if use_sixd else row["intent_pan_deg"]
            )
            row["pan_quality_pass_auto"] = bool(
                row["quality_gate_pass"] and row["direction_consistent"]
            )
    write_jsonl(run_dir / "auto_qa.jsonl", rows)
    summary = {
        "stage": state["stage"],
        "total": len(rows),
        "quality_pass": sum(bool(row["quality_gate_pass"]) for row in rows),
        "pan_quality_pass_auto": sum(
            bool(row["pan_quality_pass_auto"]) for row in rows
        ),
        "elevation_counts_auto": dict(
            Counter(row["camera_elevation_class_auto"] for row in rows)
        ),
        "qa_policy": qa_policy,
        "calibration": calibration,
        "calibration_path": (
            str((run_dir / "pitch_calibration.json").resolve())
            if state["stage"] == "validation"
            else str(calibration_path.resolve())
        ),
        "calibration_sha256": (
            sha256_file(run_dir / "pitch_calibration.json")
            if state["stage"] == "validation"
            else sha256_file(calibration_path)
        ),
        "detector_model": str(detector_model) if detector_model else None,
        "detector_sha256": detector_sha256,
        "pose_model": str(pose_model) if pose_model else None,
        "pose_sha256": pose_sha256,
        "landmark_model": str(landmark_model) if landmark_model else None,
        "landmark_sha256": landmark_sha256,
        "landmark_diagnostics": summarize_landmarks(rows),
        "onnx_execution": {
            "invariant": "every session.run input has batch size 1",
            "detector": (
                {
                    **detector_provider_plan.report(),
                    "actual_session_providers": detector.execution_providers,
                }
                if detector_provider_plan and detector
                else None
            ),
            "pose": (
                {
                    **pose_provider_plan.report(),
                    "actual_session_providers": pose.execution_providers,
                }
                if pose_provider_plan and pose
                else None
            ),
            "landmark": (
                {
                    **landmark_provider_plan.report(),
                    "actual_session_providers": landmarks.execution_providers,
                }
                if landmark_provider_plan and landmarks
                else None
            ),
        },
        "rear_policy": str(rear_policy_path.resolve()) if rear_policy_path else None,
        "rear_policy_sha256": sha256_file(rear_policy_path)
        if rear_policy_path
        else None,
    }
    (run_dir / "qa_report.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def _contact_sheet(run_dir: Path, rows: list[dict[str, Any]], output: Path) -> None:
    thumb_width, thumb_height = 180, 180
    columns = 5
    rows_count = math.ceil(len(rows) / columns)
    canvas = Image.new(
        "RGB", (columns * thumb_width, rows_count * (thumb_height + 45)), "white"
    )
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for index, row in enumerate(rows):
        path = run_dir / "images" / row["filename"]
        if not path.exists():
            continue
        with Image.open(path) as image:
            image = image.convert("RGB")
            image.thumbnail((thumb_width, thumb_height))
            x = (index % columns) * thumb_width + (thumb_width - image.width) // 2
            y = (index // columns) * (thumb_height + 45)
            canvas.paste(image, (x, y))
        label = (
            f"{row['abs_pan_bin']:03d} {row.get('direction') or '-'}\n"
            f"pan={row.get('estimated_pan_deg')} p={row.get('sixd_pitch_deg')}"
        )
        draw.multiline_text(
            ((index % columns) * thumb_width + 2, y + thumb_height + 2),
            label,
            fill="black",
            font=font,
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, "JPEG", quality=92)


def _landmark_contact_sheet(
    run_dir: Path, rows: list[dict[str, Any]], output: Path
) -> None:
    """Render HRFFA landmarks separately so the photorealism sheet stays unobstructed."""
    thumb_width, thumb_height = 240, 240
    columns = 4
    row_height = thumb_height + 72
    rows_count = math.ceil(len(rows) / columns)
    canvas = Image.new("RGB", (columns * thumb_width, rows_count * row_height), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    colours = {0: (220, 40, 40), 1: (255, 160, 30), 2: (20, 190, 70)}
    for index, row in enumerate(rows):
        path = run_dir / "images" / row["filename"]
        if not path.exists():
            continue
        with Image.open(path) as source:
            annotated = source.convert("RGB")
        overlay = ImageDraw.Draw(annotated)
        head_box = row.get("head_box_xyxy")
        if head_box:
            overlay.rectangle(tuple(head_box), outline=(40, 100, 240), width=3)
        crop_box = row.get("hrffa_crop_box_xyxy")
        if crop_box:
            overlay.rectangle(tuple(crop_box), outline=(20, 210, 210), width=2)
        points = row.get("hrffa_points_xy", [])
        visibility = row.get("hrffa_visibility", [])
        radius = (
            max(
                2,
                int(
                    round(
                        max(head_box[2] - head_box[0], head_box[3] - head_box[1]) / 100
                    )
                ),
            )
            if head_box
            else 2
        )
        for point, state in zip(points, visibility, strict=False):
            x, y = map(float, point)
            overlay.ellipse(
                (x - radius, y - radius, x + radius, y + radius),
                fill=colours.get(int(state), (255, 255, 255)),
            )
        annotated.thumbnail((thumb_width, thumb_height))
        x = (index % columns) * thumb_width + (thumb_width - annotated.width) // 2
        y = (index // columns) * row_height
        canvas.paste(annotated, (x, y))
        counts = row.get("hrffa_visibility_counts", {})
        label = (
            f"{row['abs_pan_bin']:03d} {row.get('direction') or '-'} "
            f"HRFFA={row.get('landmark_status', '-')}\n"
            f"vis={counts.get('visible', '-')} occ={counts.get('occluded', '-')} "
            f"out={counts.get('outside', '-')} core80="
            f"{row.get('hrffa_core_face_high_conf_visible_count', '-')}\n"
            "blue=DEIM cyan=HRFFA crop\n"
            "green/orange/red=visible/occluded/outside"
        )
        draw.multiline_text(
            ((index % columns) * thumb_width + 2, y + thumb_height + 2),
            label,
            fill="black",
            font=font,
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, "JPEG", quality=92)


def prepare_human_review(run_dir: Path) -> dict[str, Any]:
    qa_path = run_dir / "auto_qa.jsonl"
    if not qa_path.exists():
        raise PipelineError("run auto QA before preparing human review")
    rows = [
        json.loads(line) for line in qa_path.read_text(encoding="utf-8").splitlines()
    ]
    review_path = run_dir / "human_review.csv"
    if review_path.exists():
        raise PipelineError(f"refusing to overwrite existing review: {review_path}")
    with review_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=REVIEW_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "custom_id": row["custom_id"],
                    "filename": row["filename"],
                    "abs_pan_bin": row["abs_pan_bin"],
                    "intent_pan_deg": row["intent_pan_deg"],
                    "estimated_pan_deg": row.get("estimated_pan_deg", ""),
                    "direction": row.get("direction", ""),
                    "landmark_status": row.get("landmark_status", ""),
                    "hrffa_high_confidence_visible_count": row.get(
                        "hrffa_high_confidence_visible_count", ""
                    ),
                    "hrffa_core_face_high_conf_visible_count": row.get(
                        "hrffa_core_face_high_conf_visible_count", ""
                    ),
                    "camera_elevation_class": row["camera_elevation_class_auto"],
                    "reviewed_sha256": row.get("sha256", ""),
                }
            )
    instructions_path = run_dir / REVIEW_INSTRUCTIONS_NAME
    instructions_path.write_text(
        """# Human review instructions

- Review the head, neck, both shoulders, and visible upper torso as the target region.
- Set `head_neck_shoulders_integrity` to `fail` for anatomical breakage in that target region.
- Lower-body breakage alone is acceptable. Do not fail `photorealism`, `framing`, or
  `head_neck_shoulders_integrity` solely because legs, feet, or other lower-body regions are malformed.
- `framing` still requires a complete head and TownCentre-like head margins; a distant full-body composition
  can fail because the head is too small, independently of lower-body quality.
- Record all review fields before approval. Do not change `reviewed_sha256`.
""",
        encoding="utf-8",
    )
    contact_path = run_dir / "review_contact_sheet.jpg"
    _contact_sheet(run_dir, rows, contact_path)
    landmark_contact_path = run_dir / "landmark_contact_sheet.jpg"
    _landmark_contact_sheet(run_dir, rows, landmark_contact_path)
    sign_path = run_dir / "sign_calibration.json"
    sign_rows = []
    for pan_sector, direction in DIR8_BY_CENTRE.items():
        sixd_yaw = wrap180(-float(pan_sector))
        sign_rows.append(
            {
                "deim_direction": direction,
                "pan_sector_deg": pan_sector,
                "visual_orientation": ORIENTATION_BY_CENTRE[pan_sector],
                "expected_sixd_yaw_deg": sixd_yaw,
                "conversion": "pan_deg = (-sixd_yaw_deg) mod 360",
            }
        )
    sign_payload = {
        "status": "requires_human_approval",
        "run_id": load_state(run_dir)["local_batch_id"],
        "rows": sign_rows,
    }
    sign_path.write_text(
        json.dumps(sign_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "review": str(review_path),
        "instructions": str(instructions_path),
        "contact_sheet": str(contact_path),
        "landmark_contact_sheet": str(landmark_contact_path),
        "sign_calibration": str(sign_path),
        "records": len(rows),
    }


def approve_human_review(
    run_dir: Path,
    *,
    reviewer: str,
    sign_calibration_approved: bool,
    evaluation_protocol: Path,
    usage_report: Path,
    account_verified_snapshot: str | None,
) -> dict[str, Any]:
    if not reviewer.strip():
        raise PipelineError("reviewer must be non-empty")
    if not sign_calibration_approved:
        raise PipelineError("explicit sign calibration approval is required")
    state = load_state(run_dir)
    config = load_config(Path(state["config_path"]))
    qa_report_path = run_dir / "qa_report.json"
    if not qa_report_path.exists():
        raise PipelineError("run auto QA before approval")
    qa_report = json.loads(qa_report_path.read_text(encoding="utf-8"))
    config = config_from_recorded_qa_policy(config, qa_report)
    expected_model_hashes = {
        "detector_sha256": config["models"]["deimv2"]["sha256"],
        "pose_sha256": config["models"]["sixdrepnet360"]["sha256"],
        "landmark_sha256": config["models"]["hrffa_vitl_ibug68"]["sha256"],
    }
    for report_key, expected_hash in expected_model_hashes.items():
        if qa_report.get(report_key) != expected_hash:
            raise PipelineError(f"QA report does not bind the configured {report_key}")
    crop_margin = float(config["qa"]["deim_crop_margin"])
    if not math.isclose(crop_margin, DEIM_CROP_MARGIN, abs_tol=1e-12):
        raise PipelineError("effective QA policy has an invalid DEIM crop margin")
    if not evaluation_protocol.exists():
        raise PipelineError(f"evaluation protocol not found: {evaluation_protocol}")
    validate_evaluation_protocol(evaluation_protocol)
    if not usage_report.exists():
        raise PipelineError(f"usage report not found: {usage_report}")
    usage = json.loads(usage_report.read_text(encoding="utf-8"))
    if usage.get("local_batch_id") != state["local_batch_id"]:
        raise PipelineError("usage report belongs to a different run")
    sign_path = run_dir / "sign_calibration.json"
    if not sign_path.exists():
        raise PipelineError("prepare human review and sign calibration before approval")
    qa_rows = {
        row["custom_id"]: row
        for row in (
            json.loads(line)
            for line in (run_dir / "auto_qa.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        )
    }
    review_path = run_dir / "human_review.csv"
    with review_path.open(encoding="utf-8", newline="") as stream:
        review_rows = list(csv.DictReader(stream))
    if len(review_rows) != len(qa_rows) or {
        row["custom_id"] for row in review_rows
    } != set(qa_rows):
        raise PipelineError("human review rows do not exactly match auto QA")
    if state["stage"] in {"validation", "pilot"}:
        rear_policy = rear_reliability_policy(
            qa_rows, review_rows, config, stage=state["stage"]
        )
    else:
        qa_report = json.loads((run_dir / "qa_report.json").read_text(encoding="utf-8"))
        policy_value = qa_report.get("rear_policy")
        if not policy_value:
            raise PipelineError("approved Pilot rear policy is missing from QA report")
        rear_policy = json.loads(Path(policy_value).read_text(encoding="utf-8"))
    approved_rows: list[dict[str, Any]] = []
    for row in review_rows:
        custom_id = row["custom_id"]
        qa = qa_rows[custom_id]
        integrity_fields = (
            "photorealism",
            "framing",
            "head_neck_shoulders_integrity",
        )
        if any(row[field] not in PASS_FAIL for field in integrity_fields):
            raise PipelineError(f"incomplete pass/fail review for {custom_id}")
        if row["intent_match"] not in INTENT_VALUES:
            raise PipelineError(f"incomplete intent review for {custom_id}")
        if row["camera_elevation_class"] not in ELEVATION_VALUES:
            raise PipelineError(f"invalid elevation class for {custom_id}")
        if row["landmark_alignment"] not in LANDMARK_ALIGNMENT_VALUES:
            raise PipelineError(f"invalid landmark alignment review for {custom_id}")
        image_path = run_dir / "images" / row["filename"]
        if not image_path.exists() or row["reviewed_sha256"] != sha256_file(image_path):
            raise PipelineError(f"reviewed image changed for {custom_id}")
        human_pass = all(row[field] == "pass" for field in integrity_fields)
        intent_pass = row["intent_match"] in {"match", "off-by-one-bin"}
        pan_quality = bool(qa["pan_quality_pass_auto"] and human_pass and intent_pass)
        elevation = row["camera_elevation_class"]
        angle = qa.get("angle_deg_auto") if pan_quality else None
        label_source = qa.get("label_source_auto") if pan_quality else None
        if pan_quality and state["stage"] == "pilot" and int(qa["abs_pan_bin"]) > 90:
            sector = rear_policy["sectors"].get(str(qa["expected_direction"]), {})
            if (
                sector.get("sixd_allowed")
                and qa.get("pose_status") == "ok"
                and float(qa.get("pan_error_deg", 999.0))
                <= float(config["qa"]["pan_tolerance_deg"])
            ):
                angle = qa["estimated_pan_deg"]
                label_source = "sixdrepnet360_rear_pilot_validated"
        approved_rows.append(
            {
                **qa,
                "human_review": {key: row[key] for key in REVIEW_COLUMNS if key in row},
                "pan_quality_pass": pan_quality,
                "camera_elevation_class": elevation,
                "counts_toward_high_angle_quota": pan_quality
                and elevation == "high_angle_match",
                "angle_deg": angle,
                "label_source": label_source,
                "label_confidence": 1.0 if pan_quality else 0.0,
            }
        )
    if state["stage"] == "validation":
        if account_verified_snapshot != state["api_request"]["model"]:
            raise PipelineError(
                "Validation approval requires the configured snapshot to be explicitly account-verified"
            )
        matches = sum(
            row["human_review"]["intent_match"] == "match" for row in approved_rows
        )
        if matches < 15:
            raise PipelineError(
                f"Validation requires at least 15/19 exact intent matches, got {matches}"
            )
        calibration = json.loads(
            (run_dir / "pitch_calibration.json").read_text(encoding="utf-8")
        )
        if not calibration.get("valid"):
            raise PipelineError("Validation pitch calibration failed")
        calibration_path = run_dir / "pitch_calibration.json"
    else:
        calibration_path = Path(qa_report["calibration_path"])
        if qa_report.get("calibration_sha256") != sha256_file(calibration_path):
            raise PipelineError("QA pitch calibration changed before approval")
    write_jsonl(run_dir / "approved_annotations.jsonl", approved_rows)
    landmark_calibration = landmark_human_calibration(
        qa_rows, review_rows, stage=state["stage"]
    )
    landmark_calibration.update(
        {
            "created_at": datetime.now(UTC).isoformat(),
            "reviewer": reviewer.strip(),
        }
    )
    landmark_calibration_path = run_dir / "landmark_calibration.json"
    landmark_calibration_path.write_text(
        json.dumps(landmark_calibration, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    rear_policy.update(
        {
            "created_at": datetime.now(UTC).isoformat(),
            "reviewer": reviewer.strip(),
        }
    )
    rear_policy_path = run_dir / "rear_label_policy.json"
    rear_policy_path.write_text(
        json.dumps(rear_policy, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    elevation_counts = Counter(
        row["camera_elevation_class"]
        for row in approved_rows
        if row["pan_quality_pass"]
    )
    approval = {
        "approved": True,
        "stage": state["stage"],
        "local_batch_id": state["local_batch_id"],
        "reviewer": reviewer.strip(),
        "approved_at": datetime.now(UTC).isoformat(),
        "review_path": review_path.name,
        "review_sha256": sha256_file(review_path),
        "annotations_path": "approved_annotations.jsonl",
        "annotations_sha256": sha256_file(run_dir / "approved_annotations.jsonl"),
        "qa_report_path": qa_report_path.name,
        "qa_report_sha256": sha256_file(qa_report_path),
        **expected_model_hashes,
        "landmark_calibration_path": landmark_calibration_path.name,
        "landmark_calibration_sha256": sha256_file(landmark_calibration_path),
        "evaluation_protocol": str(evaluation_protocol.resolve()),
        "evaluation_protocol_sha256": sha256_file(evaluation_protocol),
        "usage_report": str(usage_report.resolve()),
        "usage_report_sha256": sha256_file(usage_report),
        "sign_calibration_path": sign_path.name,
        "sign_calibration_sha256": sha256_file(sign_path),
        "sign_calibration_approved": True,
        "rear_label_policy_path": rear_policy_path.name,
        "rear_label_policy_sha256": sha256_file(rear_policy_path),
        "pitch_calibration": str(calibration_path.resolve()),
        "pitch_calibration_sha256": sha256_file(calibration_path),
        "crop_margin": float(crop_margin),
        "account_verified_snapshot": account_verified_snapshot,
        "pan_quality_pass": sum(row["pan_quality_pass"] for row in approved_rows),
        "elevation_counts": dict(elevation_counts),
    }
    approval_path = run_dir / "approval.json"
    approval_path.write_text(
        json.dumps(approval, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return approval
