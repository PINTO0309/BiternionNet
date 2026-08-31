import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from biternionnet.synthetic.generate import (
    PipelineError,
    load_config,
    sha256_file,
    write_jsonl,
)
from biternionnet.synthetic.landmarks import (
    LandmarkResult,
    crop_transform,
    landmark_annotation,
)
from biternionnet.synthetic.qa import (
    QA_IMPLEMENTATION_VERSION,
    REVIEW_COLUMNS,
    calibrate_pitch,
    classify_elevation,
    direction_bin_distance,
    direction_consistent,
    effective_qa_config,
    landmark_human_calibration,
    promote_direct_production_labels,
    quality_reasons,
    rear_reliability_policy,
    run_auto_qa,
    summarize_landmarks,
)

CONFIG = (
    Path(__file__).resolve().parents[1] / "configs" / "synthetic_towncentre_batch.yaml"
)


def test_human_integrity_review_is_scoped_to_head_surroundings():
    assert "head_neck_shoulders_integrity" in REVIEW_COLUMNS
    assert "body_integrity" not in REVIEW_COLUMNS


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
    rows = [
        _quality_row(value, residual, -35 + index)
        for index, (value, residual) in enumerate(
            zip(range(0, 91, 10), [8, 9, 10, 10, 11, 12, 9, 8, 11, 10], strict=True)
        )
    ]
    calibration = calibrate_pitch(rows, config)
    assert calibration["valid"] is True
    high = _quality_row(60, calibration["bias_deg"], -40)
    eye = _quality_row(60, 45, 0)
    assert classify_elevation(high, calibration, config) == ("high_angle_match", True)
    assert classify_elevation(eye, calibration, config) == (
        "eye_level_or_low_angle",
        False,
    )
    profile = _quality_row(90, calibration["bias_deg"], -40)
    assert classify_elevation(profile, calibration, config) == ("unresolved", False)


def test_near_level_camera_regime_never_counts_as_high_angle():
    config = load_config(CONFIG)
    row = _quality_row(20, 0.0, -20.0, cam=20)
    row["camera_regime"] = "near_level"
    calibration = {
        "valid": True,
        "maximum_abs_pan_deg": 60,
        "bias_deg": 0.0,
        "threshold_deg": 25.0,
    }
    assert classify_elevation(row, calibration, config) == (
        "eye_level_or_low_angle",
        False,
    )


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
        "head_square_crop_within_image": True,
        "direction_consistent": True,
        "direction": "back",
    }
    reasons, complete = quality_reasons(row, config)
    assert complete and reasons == []
    row["direction_consistent"] = False
    reasons, complete = quality_reasons(row, config)
    assert complete and reasons == []
    assert direction_consistent("left_side", "left_side")
    assert direction_consistent("left_front", "left_side")
    assert direction_consistent("right_front", "front")
    assert not direction_consistent("front", "left_side")
    assert not direction_consistent("right_side", "left_side")
    assert direction_bin_distance("front", "left_side") == 2
    assert direction_bin_distance("right_front", "front") == 1


def test_qa_policy_disables_head_height_gate_but_keeps_measurement(tmp_path):
    config = load_config(CONFIG)
    row = {
        "exists": True,
        "image_valid": True,
        "dimension_match": True,
        "duplicate_of": None,
        "detector_status": "ok",
        "head_count": 1,
        "body_count": 1,
        "head_height_ratio": 0.10,
        "margin_left_head_ratio": 1.0,
        "margin_right_head_ratio": 1.0,
        "margin_top_head_ratio": 1.0,
        "margin_bottom_head_ratio": 1.0,
        "head_square_crop_within_image": True,
        "direction_consistent": True,
    }
    assert "head_too_small" in quality_reasons(row, config)[0]

    policy_path = tmp_path / "qa-policy.yaml"
    policy_path.write_text(
        "schema_version: 1\n"
        "pan_tolerance_deg: 30.0\n"
        "enforce_head_height_ratio: false\n"
        "deim_direction_max_bin_distance: 1\n"
        "deim_crop_margin: 0.05\n",
        encoding="utf-8",
    )
    effective, metadata = effective_qa_config(config, policy_path)
    reasons, complete = quality_reasons(row, effective)
    assert complete and reasons == []
    assert row["head_height_ratio"] == 0.10
    assert effective["qa"]["pan_tolerance_deg"] == 30.0
    assert effective["qa"]["enforce_head_height_ratio"] is False
    assert effective["qa"]["deim_crop_margin"] == 0.05
    assert metadata["source"] == "qa_policy_override"
    assert len(metadata["sha256"]) == 64


def test_quality_margin_gate_uses_five_percent_square_crop_not_legacy_ratios():
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
        "margin_left_head_ratio": 0.01,
        "margin_right_head_ratio": 0.01,
        "margin_top_head_ratio": 0.01,
        "margin_bottom_head_ratio": 0.01,
        "head_square_crop_within_image": True,
        "direction_consistent": True,
    }
    reasons, complete = quality_reasons(row, config)
    assert complete and reasons == []

    row["head_square_crop_within_image"] = False
    reasons, complete = quality_reasons(row, config)
    assert complete and reasons == ["head_crop_outside_image"]


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
    assert policy["sectors"]["left_back"]["sixd_allowed"] is True
    assert policy["sectors"]["left_back"]["deim_conflict_gate_active"] is False


def _write_minimal_uniform_qa_run(run_dir: Path, *, direct: bool) -> Path:
    run_dir.mkdir()
    (run_dir / "images").mkdir()
    image_path = run_dir / "images" / "sample.jpg"
    Image.new("RGB", (32, 32), "white").save(image_path, format="JPEG")
    plan_path = run_dir / "generation_plan.jsonl"
    write_jsonl(
        plan_path,
        [
            {
                "serial": 1,
                "custom_id": "sample",
                "filename": image_path.name,
                "abs_pan_bin": 130,
                "intent_pan_deg": 130,
                "camera_elevation": 45,
                "expected_direction": "left_back",
                "size": "32x32",
            }
        ],
    )
    state = {
        "local_batch_id": run_dir.name,
        "stage": "uniform_200",
        "config_path": str(CONFIG),
        "config_sha256": sha256_file(CONFIG),
        "plan_path": plan_path.name,
        "plan_sha256": sha256_file(plan_path),
        "direct_production": direct,
        "approval_policy": (
            "operator_direct_no_human_review" if direct else "staged_human_review"
        ),
        "intermediate_stages_waived": direct,
    }
    (run_dir / "batch_state.json").write_text(
        json.dumps(state) + "\n", encoding="utf-8"
    )
    calibration_path = run_dir / "calibration.json"
    calibration_path.write_text(
        json.dumps({"valid": True, "maximum_abs_pan_deg": 60}) + "\n",
        encoding="utf-8",
    )
    return calibration_path


def _add_passed_parent_qa(run_dir: Path, calibration_path: Path) -> Path:
    parent_dir = run_dir.parent / f"{run_dir.name}-parent"
    parent_dir.mkdir()
    image_path = run_dir / "images" / "sample.jpg"
    digest = sha256_file(image_path)
    parent_row = {
        "custom_id": "parent-sample",
        "filename": "parent-sample.jpg",
        "abs_pan_bin": 130,
        "intent_pan_deg": 130,
        "camera_elevation": 45,
        "expected_direction": "left_back",
        "exists": True,
        "image_valid": True,
        "dimension_match": True,
        "actual_size": "32x32",
        "sha256": digest,
        "duplicate_of": None,
        "detector_status": "ok",
        "head_count": 1,
        "body_count": 1,
        "head_square_crop_within_image": True,
        "direction": "left_back",
        "quality_gate_reasons": [],
        "quality_gate_complete": True,
        "quality_gate_pass": True,
        "pose_status": "skipped",
        "landmark_status": "skipped",
        "camera_elevation_class_auto": "high_angle_match",
        "counts_toward_high_angle_quota_auto": True,
        "label_source_auto": "intent_rear",
        "angle_deg_auto": 130,
        "pan_quality_pass_auto": True,
    }
    parent_qa_path = parent_dir / "auto_qa.jsonl"
    write_jsonl(parent_qa_path, [parent_row])
    _, qa_policy = effective_qa_config(load_config(CONFIG), None)
    (parent_dir / "qa_report.json").write_text(
        json.dumps(
            {
                "qa_implementation_version": QA_IMPLEMENTATION_VERSION,
                "qa_policy": qa_policy,
                "calibration_sha256": sha256_file(calibration_path),
                "detector_sha256": None,
                "pose_sha256": None,
                "landmark_sha256": None,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    state_path = run_dir / "batch_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state.update(
        {
            "parent_batch_dir": str(parent_dir.resolve()),
            "parent_qa_sha256": sha256_file(parent_qa_path),
            "edit_round": 1,
            "items": {
                "sample": {
                    "operation": "carry_forward",
                    "status": "success",
                    "parent_custom_id": "parent-sample",
                    "parent_sha256": digest,
                    "sha256": digest,
                }
            },
        }
    )
    state_path.write_text(json.dumps(state) + "\n", encoding="utf-8")
    return parent_dir


def test_direct_production_qa_uses_intent_rear_without_pilot_policy(tmp_path):
    direct_dir = tmp_path / "direct"
    calibration_path = _write_minimal_uniform_qa_run(direct_dir, direct=True)
    report = run_auto_qa(
        direct_dir,
        detector_model=None,
        pose_model=None,
        landmark_model=None,
        calibration_path=calibration_path,
        qa_policy_path=None,
    )
    assert report["rear_policy_mode"] == "direct_all_quality_intent_fallback"
    assert report["direct_production"] is True
    qa = json.loads((direct_dir / "auto_qa.jsonl").read_text(encoding="utf-8"))
    assert qa["label_source_auto"] == "intent_rear"

    staged_dir = tmp_path / "staged"
    calibration_path = _write_minimal_uniform_qa_run(staged_dir, direct=False)
    with pytest.raises(PipelineError, match="requires --rear-policy"):
        run_auto_qa(
            staged_dir,
            detector_model=None,
            pose_model=None,
            landmark_model=None,
            calibration_path=calibration_path,
            qa_policy_path=None,
        )


def test_operator_promotes_every_quality_passed_direct_label(tmp_path):
    run_dir = tmp_path / "direct-promotion"
    calibration_path = _write_minimal_uniform_qa_run(run_dir, direct=True)
    image_path = run_dir / "images" / "sample.jpg"
    write_jsonl(
        run_dir / "auto_qa.jsonl",
        [
            {
                "custom_id": "sample",
                "filename": "sample.jpg",
                "quality_gate_pass": True,
                "pan_quality_pass_auto": False,
                "intent_pan_deg": 130,
                "label_source_auto": None,
                "angle_deg_auto": None,
                "sha256": sha256_file(image_path),
            }
        ],
    )
    config = load_config(CONFIG)
    _, qa_policy = effective_qa_config(config, None)
    report = {
        "direct_production": True,
        "qa_implementation_version": QA_IMPLEMENTATION_VERSION - 1,
        "qa_policy": qa_policy,
        "total": 1,
        "quality_pass": 1,
        "calibration_path": str(calibration_path),
        "calibration_sha256": sha256_file(calibration_path),
        "detector_sha256": config["models"]["deimv2"]["sha256"],
        "pose_sha256": config["models"]["sixdrepnet360"]["sha256"],
        "landmark_sha256": config["models"]["hrffa_vitl_ibug68"]["sha256"],
    }
    (run_dir / "qa_report.json").write_text(json.dumps(report) + "\n", encoding="utf-8")

    result = promote_direct_production_labels(run_dir)

    assert result["promoted"] == 1
    assert result["total_accepted"] == 1
    row = json.loads((run_dir / "auto_qa.jsonl").read_text(encoding="utf-8"))
    assert row["pan_quality_pass_auto"] is True
    assert row["angle_deg_auto"] == 130.0
    assert row["label_source_auto"] == "intent_operator_promoted"
    assert row["label_promoted_auto"] is True
    assert row["label_confidence_auto"] == 1.0


def test_edit_qa_reuses_passed_carry_forward_without_re_evaluation(tmp_path):
    run_dir = tmp_path / "edit"
    calibration_path = _write_minimal_uniform_qa_run(run_dir, direct=True)
    _add_passed_parent_qa(run_dir, calibration_path)

    report = run_auto_qa(
        run_dir,
        detector_model=None,
        pose_model=None,
        landmark_model=None,
        calibration_path=calibration_path,
        qa_policy_path=None,
    )

    assert report["qa_reuse"]["compatibility"] == "matched"
    assert report["qa_reuse"]["reused_passed_records"] == 1
    assert report["qa_reuse"]["evaluated_current_run_records"] == 0
    row = json.loads((run_dir / "auto_qa.jsonl").read_text(encoding="utf-8"))
    assert row["custom_id"] == "sample"
    assert row["qa_reused_from_parent"] is True
    assert row["qa_reused_parent_custom_id"] == "parent-sample"
    assert row["quality_gate_pass"] is True


def test_edit_qa_refuses_to_re_evaluate_incompatible_passed_parent(tmp_path):
    run_dir = tmp_path / "edit-incompatible"
    calibration_path = _write_minimal_uniform_qa_run(run_dir, direct=True)
    parent_dir = _add_passed_parent_qa(run_dir, calibration_path)
    report_path = parent_dir / "qa_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["qa_implementation_version"] = QA_IMPLEMENTATION_VERSION - 1
    report_path.write_text(json.dumps(report) + "\n", encoding="utf-8")

    with pytest.raises(PipelineError, match="refusing to re-evaluate 1 passed"):
        run_auto_qa(
            run_dir,
            detector_model=None,
            pose_model=None,
            landmark_model=None,
            calibration_path=calibration_path,
            qa_policy_path=None,
        )


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
