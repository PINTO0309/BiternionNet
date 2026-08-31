import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from biternionnet.data import write_manifest
from biternionnet.synthetic.generate import (
    PipelineError,
    create_plan,
    load_config,
    load_state,
    read_plan,
    sha256_file,
    write_jsonl,
)
from biternionnet.synthetic.materialize import materialize_run, square_head_crop
from biternionnet.synthetic.qa import (
    DIRECT_ALL_QUALITY_LABEL_POLICY,
    QA_IMPLEMENTATION_VERSION,
    effective_qa_config,
)

CONFIG = (
    Path(__file__).resolve().parents[1] / "configs" / "synthetic_towncentre_batch.yaml"
)


def _image(path: Path, height: int, width: int, value: int = 120):
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), np.full((height, width, 3), value, dtype=np.uint8))


def test_square_head_crop_uses_long_side_and_five_percent_per_side():
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    image[:, :, 0] = np.arange(200, dtype=np.uint8)[None, :]
    image[:, :, 1] = np.arange(200, dtype=np.uint8)[:, None]

    crop = square_head_crop(image, [60, 75, 140, 125], margin=0.05)

    assert crop.shape == (88, 88, 3)
    assert crop[0, 0, 0] == 56
    assert crop[0, 0, 1] == 56


def test_square_head_crop_rejects_crop_outside_image():
    image = np.zeros((200, 200, 3), dtype=np.uint8)

    with pytest.raises(PipelineError, match="extends outside image"):
        square_head_crop(image, [0, 0, 100, 100], margin=0.05)


def test_materialize_retains_eye_level_but_keeps_it_out_of_high_angle_view(tmp_path):
    run = create_plan(
        CONFIG, "validation", "fixture-validation", tmp_path / "synthetic", seed=1
    )
    state = load_state(run)
    record = next(iter(read_plan(run, state).values()))
    _image(run / "images" / record["filename"], 200, 200)
    annotation = {
        "custom_id": record["custom_id"],
        "filename": record["filename"],
        "head_box_xyxy": [60, 50, 140, 150],
        "pan_quality_pass": True,
        "angle_deg": 90.0,
        "abs_pan_bin": 90,
        "label_source": "sixdrepnet360",
        "label_confidence": 1.0,
        "camera_elevation_class": "eye_level_or_low_angle",
        "counts_toward_high_angle_quota": False,
    }
    annotations = run / "approved_annotations.jsonl"
    write_jsonl(annotations, [annotation])
    review = run / "human_review.csv"
    review.write_text("fixture-review\n", encoding="utf-8")
    approval = {
        "approved": True,
        "crop_margin": 0.05,
        "annotations_path": annotations.name,
        "annotations_sha256": sha256_file(annotations),
        "review_path": review.name,
        "review_sha256": sha256_file(review),
    }
    (run / "approval.json").write_text(json.dumps(approval), encoding="utf-8")

    _image(tmp_path / "real" / "train.jpg", 20, 18)
    _image(tmp_path / "real" / "test.jpg", 30, 25)
    anchor = tmp_path / "towncentre" / "manifest.jsonl"
    neighbour = tmp_path / "towncentre" / "manifest_nb3.jsonl"
    rows = [
        {
            "split": "train",
            "task": "angle_deg",
            "angle_deg": 0.0,
            "image": "../real/train.jpg",
        },
        {
            "split": "test",
            "task": "angle_deg",
            "angle_deg": 180.0,
            "image": "../real/test.jpg",
        },
    ]
    write_manifest(rows, anchor)
    write_manifest(rows, neighbour)
    original = neighbour.read_bytes()
    output = tmp_path / "materialized"
    bad_approval = {**approval, "crop_margin": 0.15}
    (run / "approval.json").write_text(json.dumps(bad_approval), encoding="utf-8")
    with pytest.raises(PipelineError, match="must be fixed to 0.05"):
        materialize_run(
            run,
            output_root=output,
            anchor_manifest=anchor,
            neighbour_manifest=neighbour,
            seed=1,
        )
    (run / "approval.json").write_text(json.dumps(approval), encoding="utf-8")
    report = materialize_run(
        run,
        output_root=output,
        anchor_manifest=anchor,
        neighbour_manifest=neighbour,
        seed=1,
    )
    assert report["materialized_this_run"] == 1
    assert report["crop_margin"] == 0.05
    assert report["crop_rule"] == (
        "max(head_box_width,head_box_height)*(1+2*margin) square"
    )
    assert report["high_angle_total"] == 0
    all_rows = [
        json.loads(line)
        for line in (output / "manifest.jsonl").read_text().splitlines()
    ]
    assert all_rows[0]["camera_elevation_class"] == "eye_level_or_low_angle"
    assert all_rows[0]["source_crop_rule"] == "deim_long_side_square_5pct_per_side"
    assert (output / "manifest_high_angle.jsonl").read_text() == ""
    high_combined = tmp_path / "towncentre" / "manifest_nb3_synthetic.jsonl"
    all_combined = (
        tmp_path / "towncentre" / "manifest_nb3_synthetic_all_elevations.jsonl"
    )
    assert high_combined.read_bytes() == original
    assert len(all_combined.read_text().splitlines()) == 3


def test_materialize_accepts_all_promoted_direct_production_rows(tmp_path):
    run = create_plan(
        CONFIG, "validation", "fixture-direct", tmp_path / "synthetic", seed=1
    )
    state = load_state(run)
    state["direct_production"] = True
    state["approval_policy"] = "operator_direct_no_human_review"
    (run / "batch_state.json").write_text(
        json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    record = next(iter(read_plan(run, state).values()))
    record["augmentation_type"] = "face_mask"
    record["mask_description"] = "white surgical mask"
    record["parent_custom_id"] = "fixture-parent"
    write_jsonl(run / state["plan_path"], [record])
    state["plan_sha256"] = sha256_file(run / state["plan_path"])
    (run / "batch_state.json").write_text(
        json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    image_path = run / "images" / record["filename"]
    _image(image_path, 200, 200)
    qa_row = {
        "custom_id": record["custom_id"],
        "filename": record["filename"],
        "sha256": sha256_file(image_path),
        "head_box_xyxy": [60, 50, 140, 150],
        "quality_gate_pass": True,
        "pan_quality_pass_auto": True,
        "angle_deg_auto": 270.0,
        "abs_pan_bin": 90,
        "label_source_auto": "intent_operator_promoted",
        "label_confidence_auto": 1.0,
        "camera_elevation_class_auto": "unresolved",
        "counts_toward_high_angle_quota_auto": False,
        "label_acceptance_policy_auto": DIRECT_ALL_QUALITY_LABEL_POLICY,
    }
    write_jsonl(run / "auto_qa.jsonl", [qa_row])
    calibration = run / "pitch_calibration.json"
    calibration.write_text('{"valid": true}\n', encoding="utf-8")
    config = load_config(CONFIG)
    _, recorded_policy = effective_qa_config(config, None)
    qa_report = {
        "direct_production": True,
        "qa_implementation_version": QA_IMPLEMENTATION_VERSION,
        "label_acceptance_policy_auto": DIRECT_ALL_QUALITY_LABEL_POLICY,
        "total": 1,
        "quality_pass": 1,
        "pan_quality_pass_auto": 1,
        "operator_label_promotion": {
            "policy": DIRECT_ALL_QUALITY_LABEL_POLICY,
            "total_accepted": 1,
        },
        "detector_sha256": config["models"]["deimv2"]["sha256"],
        "pose_sha256": config["models"]["sixdrepnet360"]["sha256"],
        "landmark_sha256": config["models"]["hrffa_vitl_ibug68"]["sha256"],
        "calibration_path": str(calibration),
        "calibration_sha256": sha256_file(calibration),
        "qa_policy": recorded_policy,
    }
    (run / "qa_report.json").write_text(
        json.dumps(qa_report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    _image(tmp_path / "real" / "train.jpg", 20, 18)
    _image(tmp_path / "real" / "test.jpg", 30, 25)
    anchor = tmp_path / "towncentre" / "manifest.jsonl"
    neighbour = tmp_path / "towncentre" / "manifest_nb3.jsonl"
    real_rows = [
        {
            "split": "train",
            "task": "angle_deg",
            "angle_deg": 0.0,
            "image": "../real/train.jpg",
        },
        {
            "split": "test",
            "task": "angle_deg",
            "angle_deg": 180.0,
            "image": "../real/test.jpg",
        },
    ]
    write_manifest(real_rows, anchor)
    write_manifest(real_rows, neighbour)

    output = tmp_path / "materialized"
    report = materialize_run(
        run,
        output_root=output,
        anchor_manifest=anchor,
        neighbour_manifest=neighbour,
        seed=1,
    )

    assert report["annotation_source"] == "direct_operator_promoted_auto_qa"
    assert report["all_elevations_trainable_total"] == 1
    assert report["high_angle_total"] == 0
    accepted = [
        json.loads(line)
        for line in (run / "accepted_annotations.jsonl").read_text().splitlines()
    ]
    assert accepted[0]["angle_deg"] == 270.0
    assert accepted[0]["pan_quality_pass"] is True
    all_rows = [
        json.loads(line)
        for line in (output / "manifest.jsonl").read_text().splitlines()
    ]
    assert all_rows[0]["annotation_acceptance_source"] == (
        "direct_operator_promoted_auto_qa"
    )
    assert all_rows[0]["augmentation_type"] == "face_mask"
    assert all_rows[0]["mask_description"] == "white surgical mask"
    assert all_rows[0]["parent_custom_id"] == "fixture-parent"
    assert report["augmentation_counts"] == {"face_mask": 1}
    combined_all = (
        tmp_path / "towncentre" / "manifest_nb3_synthetic_all_elevations.jsonl"
    )
    assert len(combined_all.read_text().splitlines()) == 3
