from pathlib import Path

import numpy as np
import pytest

from biternionnet.synthetic.generate import load_config
from biternionnet.synthetic.landmarks import LandmarkResult, crop_transform, landmark_annotation
from biternionnet.synthetic.qa import (
    calibrate_pitch,
    classify_elevation,
    direction_consistent,
    landmark_human_calibration,
    quality_reasons,
    rear_reliability_policy,
    summarize_landmarks,
)

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "synthetic_towncentre.yaml"


def _quality_row(abs_pan, residual, pitch, cam=45):
    return {
        "abs_pan_bin": abs_pan,
        "pose_status": "ok",
        "quality_gate_pass": True,
        "pitch_residual_deg": residual,
        "sixd_pitch_deg": pitch,
        "camera_elevation": cam,
    }


def test_pitch_calibration_classifies_but_does_not_reject_eye_level():
    config = load_config(CONFIG)
    rows = [_quality_row(value, residual, -35 + index) for index, (value, residual) in enumerate(
        zip(range(0, 91, 10), [8, 9, 10, 10, 11, 12, 9, 8, 11, 10], strict=True)
    )]
    calibration = calibrate_pitch(rows, config)
    assert calibration["valid"] is True
    high = _quality_row(90, calibration["bias_deg"], -40)
    eye = _quality_row(90, 45, 0)
    assert classify_elevation(high, calibration, config) == ("high_angle_match", True)
    assert classify_elevation(eye, calibration, config) == ("eye_level_or_low_angle", False)


def test_back_view_is_allowed_and_direction_mapping_is_explicit():
    config = load_config(CONFIG)
    row = {
        "exists": True,
        "image_valid": True,
        "dimension_match": True,
        "duplicate_of": None,
        "detector_status": "ok",
        "head_count": 1,
        "body_count": 1,
        "head_height_ratio": 0.35,
        "margin_left_head_ratio": 1.0,
        "margin_right_head_ratio": 1.0,
        "margin_top_head_ratio": 1.0,
        "margin_bottom_head_ratio": 1.0,
        "direction_consistent": True,
        "direction": "back",
    }
    reasons, complete = quality_reasons(row, config)
    assert complete and reasons == []
    assert direction_consistent("left_side", 90)
    assert direction_consistent("right_side", 270)
    assert not direction_consistent("right_side", 90)


def test_validation_pitch_trend_and_pilot_rear_policy_are_fail_closed():
    config = load_config(CONFIG)
    rows = [
        _quality_row(10 * index, 10, pitch, cam)
        for index, (cam, pitch) in enumerate(
            [(30, -20), (30, -21), (45, -35), (45, -36), (60, -50), (60, -51)]
        )
    ]
    calibrated = calibrate_pitch(rows, config)
    assert calibrated["valid"] is True
    assert calibrated["negative_pitch_trend_valid"] is True
    for row in rows:
        row["sixd_pitch_deg"] *= -1
    assert calibrate_pitch(rows, config)["valid"] is False

    qa_rows = {}
    reviews = []
    for index in range(10):
        custom_id = f"rear-{index}"
        qa_rows[custom_id] = {
            "custom_id": custom_id,
            "abs_pan_bin": 130,
            "expected_direction": "left_back",
            "pose_status": "ok",
            "pan_error_deg": 10 if index < 7 else 40,
            "direction_consistent": index != 9,
        }
        reviews.append({"custom_id": custom_id})
    policy = rear_reliability_policy(qa_rows, reviews, config, stage="pilot")
    assert policy["sectors"]["left_back"]["sixd_allowed"] is True
    qa_rows["rear-8"]["direction_consistent"] = False
    policy = rear_reliability_policy(qa_rows, reviews, config, stage="pilot")
    assert policy["sectors"]["left_back"]["sixd_allowed"] is False


def test_hrffa_crop_uses_five_percent_per_side_of_deim_long_side():
    transform, crop_box = crop_transform([10, 20, 110, 70], out_size=320, pad=0.05)
    assert crop_box == pytest.approx((5.0, -10.0, 115.0, 100.0))
    centre = transform @ np.asarray([60.0, 45.0, 1.0])
    assert centre[:2] == pytest.approx([160.0, 160.0])
    assert crop_box[2] - crop_box[0] == pytest.approx(100 * 1.1)


def test_hrffa_annotation_is_diagnostic_and_sector_summarized():
    points = np.stack(
        [np.linspace(0.2, 0.8, 68), np.linspace(0.25, 0.75, 68)], axis=1
    ).astype(np.float32)
    logits = np.zeros((68, 3), dtype=np.float32)
    logits[:, 2] = 4.0
    probabilities = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    result = LandmarkResult(
        points_normalized=points,
        points_xy=points * 100,
        visibility_logits=logits,
        visibility_probabilities=probabilities,
        visibility=np.full(68, 2, dtype=np.int64),
        crop_transform=np.eye(3),
        crop_box_xyxy=(0.0, 0.0, 100.0, 100.0),
        crop_image_coverage_ratio=1.0,
    )
    row = {
        "expected_direction": "front",
        **landmark_annotation(result, (100, 100, 3)),
    }
    groups = row["hrffa_landmark_groups"]
    row["hrffa_core_face_high_conf_visible_count"] = sum(
        groups[name]["high_confidence_visible_count"]
        for name in (
            "subject_right_eye",
            "subject_left_eye",
            "nose",
            "outer_mouth",
            "inner_mouth",
        )
    )
    summary = summarize_landmarks([row])
    assert row["hrffa_crop_pad"] == 0.05
    assert row["hrffa_high_confidence_visible_count"] == 68
    assert row["hrffa_core_face_high_conf_visible_count"] == 41
    assert row["hrffa_diagnostic_only"] is True
    assert summary["hard_gate_active"] is False
    assert summary["by_expected_direction"]["front"]["count"] == 1
    calibration = landmark_human_calibration(
        {"front-1": {"custom_id": "front-1", **row}},
        [{"custom_id": "front-1", "landmark_alignment": "match"}],
        stage="pilot",
    )
    assert calibration["hard_gate_active"] is False
    assert calibration["metrics_by_human_alignment"]["match"]["count"] == 1
    assert calibration["review_counts_by_expected_direction"]["front"] == {"match": 1}
