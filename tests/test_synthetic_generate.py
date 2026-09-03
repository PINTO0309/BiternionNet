import base64
import hashlib
import io
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from biternionnet.synthetic.generate import (
    PipelineError,
    _edit_prompt,
    _pitch_calibration_tail_candidates,
    _qa_edit_reasons,
    _refresh_yawpose_regeneration_prompt,
    _regeneration_prompt,
    _token_based_batch_plan,
    advance_sequential_batches,
    build_plan,
    build_usage_report,
    create_edit_cycle,
    create_mask_augmentation,
    create_plan,
    finalize_standalone_run,
    load_config,
    load_state,
    pending_request_count,
    prepare_standalone_run,
    process_output_jsonl,
    read_plan,
    refresh_status,
    save_state,
    seal_collected_prefix,
    sha256_file,
    submit_pending,
    write_jsonl,
)
from biternionnet.synthetic.models import install_model_assets

CONFIG = (
    Path(__file__).resolve().parents[1] / "configs" / "synthetic_towncentre_batch.yaml"
)
SNAPSHOT_CONFIG = (
    Path(__file__).resolve().parents[1] / "configs" / "synthetic_towncentre.yaml"
)
NEAR_LEVEL_CONFIG = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "synthetic_towncentre_near_level_batch.yaml"
)
YAWPOSE_CONFIG = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "synthetic_yawpose_rear8000_batch.yaml"
)
YAWPOSE_FULL_CONFIG = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "synthetic_yawpose_full11382_batch.yaml"
)
YAWPOSE_ACCESSORY_CONFIG = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "synthetic_yawpose_accessories680_batch.yaml"
)
YAWPOSE_S005_CONFIG = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "synthetic_yawpose_s005_5710_batch.yaml"
)
YAWPOSE_S006_CONFIG = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "synthetic_yawpose_s006_712_batch.yaml"
)


def test_validation_and_pilot_are_deterministic_and_cover_directions():
    config = load_config(CONFIG)
    validation = build_plan(config, "validation", seed=7)
    assert len(validation) == 19
    assert [row["abs_pan_bin"] for row in validation] == list(range(0, 181, 10))
    assert {row["expected_direction"] for row in validation} == {
        "front",
        "left_front",
        "left_side",
        "left_back",
        "back",
        "right_back",
        "right_side",
        "right_front",
    }
    assert any(
        row["abs_pan_bin"] == 180 and row["expected_direction"] == "back"
        for row in validation
    )
    assert {row["camera_elevation"] for row in validation} == {30, 45, 60}
    assert (
        "toward image-right"
        in next(row for row in validation if row["signed_pan"] == 20)["pan_detail"]
    )
    assert (
        "toward image-left"
        in next(row for row in validation if row["signed_pan"] == -30)["pan_detail"]
    )
    assert (
        "exact full-back view"
        in next(row for row in validation if row["signed_pan"] == 180)["pan_detail"]
    )

    first = build_plan(config, "pilot", seed=19)
    second = build_plan(config, "pilot", seed=19)
    assert first == second
    assert len(first) == 380
    assert Counter(row["abs_pan_bin"] for row in first) == {
        value: 20 for value in range(0, 181, 10)
    }
    for value in range(10, 180, 10):
        signs = [row["signed_pan"] > 0 for row in first if row["abs_pan_bin"] == value]
        assert Counter(signs) == {True: 10, False: 10}


def test_pitch_edit_excludes_side_profile_euler_fold():
    config = load_config(CONFIG)
    row = {
        "abs_pan_bin": 70,
        "camera_elevation": 45,
        "pose_status": "ok",
        "pan_error_deg": 0.0,
        "sixd_pitch_deg": 167.0,
        "direction_consistent": True,
        "quality_gate_reasons": [],
    }
    assert "head_looks_up_at_camera" not in _qa_edit_reasons(row, config)
    row["abs_pan_bin"] = 60
    assert "head_looks_up_at_camera" in _qa_edit_reasons(row, config)


def test_edit_cycle_uses_effective_pan_tolerance_recorded_by_qa():
    config = load_config(CONFIG)
    row = {
        "abs_pan_bin": 20,
        "camera_elevation": 45,
        "pose_status": "ok",
        "pan_error_deg": 29.9,
        "sixd_pitch_deg": -45.0,
        "direction_consistent": True,
        "quality_gate_reasons": [],
        "qa_pan_tolerance_deg": 30.0,
    }
    assert "pan_out_of_tolerance" not in _qa_edit_reasons(row, config)
    row["pan_error_deg"] = 30.1
    assert "pan_out_of_tolerance" in _qa_edit_reasons(row, config)


def test_operator_promoted_label_does_not_trigger_pose_or_pitch_edit():
    config = load_config(CONFIG)
    row = {
        "abs_pan_bin": 50,
        "camera_elevation": 45,
        "pose_status": "ok",
        "pan_error_deg": 120.0,
        "sixd_pitch_deg": 10.0,
        "quality_gate_pass": True,
        "quality_gate_reasons": [],
        "label_acceptance_policy_auto": ("direct_all_quality_pass_intent_fallback_v1"),
    }
    assert _qa_edit_reasons(row, config) == []


def test_pitch_calibration_tail_selects_two_controlling_residuals():
    config = load_config(CONFIG)
    residuals = [-12.5829, -10.3286, 2.6038, 22.0412, 26.1672, 3.4859]
    qa_rows = {
        f"row-{index}": {
            "custom_id": f"row-{index}",
            "filename": f"row-{index}.jpg",
            "abs_pan_bin": index * 10,
            "pose_status": "ok",
            "quality_gate_pass": True,
            "pitch_residual_deg": residual,
        }
        for index, residual in enumerate(residuals)
    }
    calibration = {
        "valid": False,
        "sample_count": 6,
        "bias_deg": 3.04485,
        "data_threshold_deg": 26.05935,
        "hard_maximum_deg": 25.0,
    }
    selected = _pitch_calibration_tail_candidates(qa_rows, calibration, config)
    assert [row["custom_id"] for row in selected] == ["row-4", "row-3"]
    assert [row["centred_abs_residual_deg"] for row in selected] == pytest.approx(
        [23.12235, 18.99635]
    )


def test_observed_token_batch_plan_uses_the_minimum_batch_count():
    one = _token_based_batch_plan(432, 2312.0)
    assert one["expected_total_input_tokens"] == pytest.approx(998_784)
    assert one["max_records_per_batch"] == 432
    assert one["minimum_batch_count"] == 1

    two = _token_based_batch_plan(433, 2312.0)
    assert two["expected_total_input_tokens"] == pytest.approx(1_001_096)
    assert two["max_records_per_batch"] == 432
    assert two["minimum_batch_count"] == 2


def test_pan_edit_prompt_uses_relative_correction_from_current_pose():
    record = {
        "intent_pan_deg": 20,
        "signed_pan": 20,
        "pan_detail": "Keep the final view mostly frontal.",
        "prompt": "Original target.",
    }
    row = {"pose_status": "ok", "estimated_pan_deg": 57.2929}
    prompt = _edit_prompt(record, row, ["pan_out_of_tolerance"])
    assert "estimated at pan +57.3 degrees" in prompt
    assert "37.3 degrees toward image-left" in prompt
    assert "not an instruction to turn farther" in prompt


def test_crop_edit_prompt_translates_person_away_from_overflowing_edges():
    record = {
        "size": "1024x1024",
        "prompt": "Original target.",
    }
    row = {
        "actual_size": "1024x1024",
        "head_square_crop_box_xyxy": [-12.0, -7.0, 700.0, 705.0],
    }
    prompt = _edit_prompt(record, row, ["head_crop_outside_image"])
    assert "4% of the image width toward image-right" in prompt
    assert "3% of the image height downward" in prompt
    assert "not a crop, zoom, head rotation" in prompt
    assert "times 1.10 must stay fully inside" in prompt
    assert (
        "explicitly overrides any generic target instruction not to translate" in prompt
    )

    row["head_square_crop_box_xyxy"] = [-1.0, 100.0, 1031.0, 1132.0]
    prompt = _edit_prompt(record, row, ["head_crop_outside_image"])
    assert "uniformly reduce the entire person by approximately 3%" in prompt


def test_multiple_head_edit_prompt_removes_person_shaped_background_content():
    record = {
        "size": "1024x1024",
        "prompt": "Original target.",
    }
    row = {
        "head_count": 3,
        "actual_size": "1024x1024",
    }

    prompt = _edit_prompt(record, row, ["head_count_not_one"])

    assert "Keep only the main foreground subject" in prompt
    assert "mannequin" in prompt
    assert "human reflection" in prompt
    assert "plain opaque wall" in prompt


def test_deim_direction_does_not_create_an_edit_reason():
    config = load_config(CONFIG)
    reasons = _qa_edit_reasons(
        {
            "abs_pan_bin": 120,
            "quality_gate_reasons": ["direction_conflict"],
            "direction_consistent": False,
        },
        config,
    )
    assert reasons == []


def test_plan_fixes_low_quality_and_is_immutable(tmp_path):
    run = create_plan(CONFIG, "validation", "validation-v001", tmp_path, seed=3)
    state = load_state(run)
    assert state["request_count"] == 19
    assert state["api_request"]["model"] == "gpt-image-2"
    assert state["api_request"]["quality"] == "low"
    assert len(state["shards"]) == 1
    requests = [
        json.loads(line)
        for line in (run / state["shards"][0]["attempts"][0]["input_path"])
        .read_text()
        .splitlines()
    ]
    assert all(row["body"]["quality"] == "low" for row in requests)
    assert all(row["body"]["output_format"] == "jpeg" for row in requests)
    assert all(row["body"]["output_compression"] == 92 for row in requests)
    assert all(row["custom_id"].startswith("validation-v001--") for row in requests)
    plan = read_plan(run, state)
    filenames = [row["filename"] for row in plan.values()]
    assert filenames == sorted(filenames)
    assert filenames[1] == "validation-v001_000002--pan-010_cam+45_pitch+002.jpg"
    assert state["items"][requests[1]["custom_id"]]["filename"] == filenames[1]
    local_status = refresh_status(run)
    assert local_status["status"] == "planned"
    assert local_status["pending_requests"] == 19
    assert local_status["batches"] == []
    (run / "generation_plan.jsonl").write_text("{}\n", encoding="utf-8")
    with pytest.raises(PipelineError, match="changed"):
        load_state(run)


def test_plan_rejects_snapshot_that_batch_api_does_not_support(tmp_path):
    with pytest.raises(PipelineError, match="dated GPT-Image-2 snapshots"):
        create_plan(
            SNAPSHOT_CONFIG, "validation", "validation-snapshot", tmp_path, seed=3
        )


def test_direct_uniform_production_requires_explicit_single_batch(tmp_path):
    root = tmp_path / "runs"
    with pytest.raises(PipelineError, match="requires --single-batch"):
        create_plan(
            CONFIG,
            "uniform_200",
            "production-invalid",
            root,
            seed=3,
            direct_production=True,
        )

    run = create_plan(
        CONFIG,
        "uniform_200",
        "production-uniform200-v001",
        root,
        seed=3,
        direct_production=True,
        single_batch=True,
        compact_prompts=True,
    )
    state = load_state(run)
    assert state["request_count"] == 6700
    assert state["direct_production"] is True
    assert state["single_batch"] is True
    assert state["prompt_profile"] == "compact_direct_v1"
    assert state["approval_policy"] == "operator_direct_no_human_review"
    assert state["intermediate_stages_waived"] is True
    assert state["parent_batch_dir"] is None
    assert len(state["shards"]) == 1
    assert len(state["shards"][0]["custom_ids"]) == 6700
    plan = read_plan(run, state)
    assert all("Photorealistic overhead CCTV" in row["prompt"] for row in plan.values())
    assert all("detected head" not in row["prompt"] for row in plan.values())


def test_near_level_plan_preserves_pan_distribution_and_submits_serially(tmp_path):
    run = create_plan(
        NEAR_LEVEL_CONFIG,
        "near_level_8400",
        "production-nearlevel8400-v001",
        tmp_path,
        seed=20260831,
        direct_production=True,
        sequential_batches=True,
        compact_prompts=True,
    )
    state = load_state(run)
    rows = list(read_plan(run, state).values())

    assert state["request_count"] == 8400
    assert state["single_batch"] is False
    assert state["sequential_batches"] is True
    assert [len(shard["custom_ids"]) for shard in state["shards"]] == [500] * 16 + [400]
    assert Counter(row["abs_pan_bin"] for row in rows) == Counter(
        {
            0: 182,
            10: 441,
            20: 638,
            30: 594,
            40: 331,
            50: 501,
            60: 548,
            70: 728,
            80: 800,
            90: 687,
            100: 564,
            110: 592,
            120: 556,
            130: 395,
            140: 277,
            150: 225,
            160: 251,
            170: 90,
        }
    )
    sign_counts = Counter(
        1 if row["signed_pan"] > 0 else -1 if row["signed_pan"] < 0 else 0
        for row in rows
    )
    assert sign_counts == Counter({1: 4109, -1: 4109, 0: 182})

    elevation_counts = Counter(row["camera_elevation"] for row in rows)
    assert set(elevation_counts) == set(range(-20, 21))
    assert set(elevation_counts.values()) == {204, 205}
    assert sum(elevation_counts[value] for value in range(-20, 0)) == 4098
    assert sum(elevation_counts[value] for value in range(1, 21)) == 4098
    assert elevation_counts[0] == 204

    masked = [row for row in rows if row.get("augmentation_type") == "face_mask"]
    assert len(masked) == 1680
    assert max(row["abs_pan_bin"] for row in masked) == 90
    assert set(Counter(row["mask_description"] for row in masked).values()) == {210}
    assert all(row["camera_regime"] == "near_level" for row in rows)
    assert "above the subject's eye level" in next(
        row["prompt"] for row in rows if row["camera_elevation"] == 20
    )
    assert "below the subject's eye level" in next(
        row["prompt"] for row in rows if row["camera_elevation"] == -20
    )
    assert "exactly at the subject's eye level" in next(
        row["prompt"] for row in rows if row["camera_elevation"] == 0
    )
    assert "Correctly wears" in masked[0]["prompt"]

    client = SimpleNamespace(files=_FakeFiles(), batches=_FakeBatches())
    advance = advance_sequential_batches(
        run,
        spend_cap_usd=100.0,
        client=client,
    )
    assert advance["action"] == "submitted_next_batch"
    assert advance["submitted_batch_ids"] == ["batch-1"]
    assert advance["retry_requests"] == 0
    assert pending_request_count(load_state(run)) == 7900
    assert (
        submit_pending(
            run,
            approved_request_count=7900,
            spend_cap_usd=100.0,
            client=client,
        )
        == []
    )
    assert client.batches.created == 1


def test_yawpose_rear_plan_matches_shortage_distribution_and_sign_contract(tmp_path):
    run = create_plan(
        YAWPOSE_CONFIG,
        "yawpose_rear_8000",
        "production-yawpose-rear8000-v001",
        tmp_path,
        seed=20260901,
        direct_production=True,
        sequential_batches=True,
    )
    state = load_state(run)
    rows = list(read_plan(run, state).values())
    expected = [
        91,
        250,
        221,
        258,
        419,
        540,
        591,
        566,
        728,
        818,
        727,
        565,
        592,
        539,
        420,
        258,
        221,
        196,
    ]

    assert state["target_count"] == 8000
    assert state["label_convention"] == "yawpose"
    assert state["sequential_batches"] is True
    assert [len(shard["custom_ids"]) for shard in state["shards"]] == [500] * 16
    assert Counter(row["bin"] for row in rows) == Counter(
        {
            f"yaw_{start}_{start + 10}": count
            for start, count in zip(range(90, 270, 10), expected, strict=True)
        }
    )
    for start, count in zip(range(90, 270, 10), expected, strict=True):
        integer_counts = Counter(
            row["yaw_yawpose"]
            for row in rows
            if row["bin"] == f"yaw_{start}_{start + 10}"
        )
        assert set(integer_counts) == set(range(start, start + 10))
        assert max(integer_counts.values()) - min(integer_counts.values()) <= 1
        assert sum(integer_counts.values()) == count

    camera_counts = Counter(row["cam"] for row in rows)
    assert camera_counts[0] == 5600
    assert sum(camera_counts[value] for value in range(-30, 0)) == 1200
    assert sum(camera_counts[value] for value in range(1, 31)) == 1200
    assert set(camera_counts[value] for value in [*range(-30, 0), *range(1, 31)]) == {
        40
    }
    assert Counter(row["size"] for row in rows) == {
        "1024x1536": 4000,
        "1536x1024": 4000,
    }
    assert all(-10 <= row["pitch"] <= 10 and row["roll"] == 0 for row in rows)
    masked = [row for row in rows if row.get("augmentation_type") == "face_mask"]
    assert len(masked) == 152
    assert {row["bin"] for row in masked} == {
        "yaw_90_100",
        "yaw_100_110",
        "yaw_250_260",
        "yaw_260_270",
    }
    assert all(
        row["filename"]
        == (
            f"yawp{row['yaw_yawpose']:+04d}_pitch{row['pitch']:+03d}_"
            f"cam{row['cam']:+03d}_{row['serial']:06d}.jpg"
        )
        for row in rows
    )
    required = {
        "serial",
        "bin",
        "yaw_yawpose",
        "pitch",
        "cam",
        "roll",
        "visible_side",
        "size",
        "context",
        "anchor",
        "gender",
        "age",
        "skin_tone",
        "hair",
        "clothing",
        "headwear",
        "lens_feel",
        "background",
        "lighting",
        "custom_id",
        "filename",
        "prompt",
    }
    assert all(required <= row.keys() for row in rows)
    assert all("+90 degrees faces screen-left" in row["prompt"] for row in rows)
    assert "pure centered back-of-head" in next(
        row["anchor"] for row in rows if row["yaw_yawpose"] == 180
    )


def test_yawpose_v11_remainder_adds_front_bins_and_continues_serials(tmp_path):
    run = create_plan(
        YAWPOSE_FULL_CONFIG,
        "yawpose_full_remainder_10382",
        "production-yawpose-full11382-v002",
        tmp_path,
        seed=20260901,
        serial_offset=1000,
        direct_production=True,
        sequential_batches=True,
    )
    state = load_state(run)
    rows = list(read_plan(run, state).values())

    assert state["target_count"] == 10382
    assert state["serial_offset"] == 1000
    assert min(row["serial"] for row in rows) == 1001
    assert max(row["serial"] for row in rows) == 11382
    assert [len(shard["custom_ids"]) for shard in state["shards"]] == [500] * 20 + [382]
    assert sum(row["visible_side"] == "three_quarter_left" for row in rows) == 1718
    assert sum(row["visible_side"] == "three_quarter_right" for row in rows) == 552
    assert sum(abs(row["pitch"]) > 10 for row in rows) == 2276
    assert sum(row.get("augmentation_type") == "face_mask" for row in rows) == 610
    assert all(
        "both eyes remain visible" in row["anchor"]
        for row in rows
        if row["visible_side"].startswith("three_quarter")
    )


def test_yawpose_accessories_are_uniform_per_eligible_bin(tmp_path):
    run = create_plan(
        YAWPOSE_ACCESSORY_CONFIG,
        "yawpose_accessories_680",
        "production-yawpose-accessories680-v001",
        tmp_path,
        seed=20260902,
        serial_offset=11382,
        direct_production=True,
        sequential_batches=True,
    )
    state = load_state(run)
    rows = list(read_plan(run, state).values())

    assert state["target_count"] == 680
    assert state["serial_offset"] == 11382
    assert [len(shard["custom_ids"]) for shard in state["shards"]] == [500, 180]
    assert min(row["serial"] for row in rows) == 11383
    assert max(row["serial"] for row in rows) == 12062
    assert Counter(row["accessory_type"] for row in rows) == {
        "sunglasses": 180,
        "ear_piercing": 500,
    }
    assert all(row["augmentation_type"] == "accessory" for row in rows)
    assert not any(row.get("mask_description") for row in rows)
    assert sum(abs(row["pitch"]) > 10 for row in rows) == 136
    assert Counter(row["size"] for row in rows) == {
        "1024x1536": 340,
        "1536x1024": 340,
    }

    sunglasses = [row for row in rows if row["accessory_type"] == "sunglasses"]
    assert {int(row["bin"].split("_")[1]) for row in sunglasses} <= {
        20,
        30,
        40,
        50,
        60,
        70,
        300,
        310,
        320,
    }
    assert set(Counter(row["bin"] for row in sunglasses).values()) == {20}
    assert all("Required additional accessory:" in row["prompt"] for row in rows)
    assert all(row["accessory_description"] in row["prompt"] for row in rows)

    ear_piercings = [row for row in rows if row["accessory_type"] == "ear_piercing"]
    assert not {int(row["bin"].split("_")[1]) for row in ear_piercings}.intersection(
        {170, 180}
    )
    assert set(Counter(row["bin"] for row in ear_piercings).values()) == {20}


def test_yawpose_s005_plan_matches_spec_and_accessory_policy(tmp_path):
    run = create_plan(
        YAWPOSE_S005_CONFIG,
        "yawpose_s005_5710",
        "production-yawpose-s005-v001",
        tmp_path,
        seed=20260903,
        direct_production=True,
        sequential_batches=True,
    )
    state = load_state(run)
    rows = list(read_plan(run, state).values())
    starts = [120, 130, 140, 150, 160, 170, 210, 220, 230, 240, 250, 260]
    expected = [495, 458, 389, 400, 494, 359, 450, 494, 603, 642, 569, 357]

    assert state["target_count"] == 5710
    assert state["sequential_batches"] is True
    assert state["api_request"]["quality"] == "low"
    assert [len(shard["custom_ids"]) for shard in state["shards"]] == [500] * 11 + [210]
    assert Counter(row["bin"] for row in rows) == Counter(
        {
            f"yaw_{start}_{start + 10}": count
            for start, count in zip(starts, expected, strict=True)
        }
    )
    for start, count in zip(starts, expected, strict=True):
        integer_counts = Counter(
            row["yaw_yawpose"]
            for row in rows
            if row["bin"] == f"yaw_{start}_{start + 10}"
        )
        assert set(integer_counts) == set(range(start, start + 10))
        assert max(integer_counts.values()) - min(integer_counts.values()) <= 1
        assert sum(integer_counts.values()) == count

    assert sum(abs(row["pitch"]) > 10 for row in rows) == 1142
    assert Counter(row["size"] for row in rows) == {
        "1024x1536": 2855,
        "1536x1024": 2855,
    }
    assert Counter(
        row.get("accessory_type") for row in rows if row.get("accessory_type")
    ) == {
        "eyeglasses": 120,
        "sunglasses": 120,
        "earring": 330,
    }
    assert sum(row.get("augmentation_type") == "face_mask" for row in rows) == 120
    assert {
        int(row["bin"].split("_")[1])
        for row in rows
        if row.get("augmentation_type") == "face_mask"
    } == {250, 260}
    assert {
        int(row["bin"].split("_")[1])
        for row in rows
        if row.get("accessory_type") in {"eyeglasses", "sunglasses"}
    } == {120, 130, 250, 260}
    assert 170 not in {
        int(row["bin"].split("_")[1])
        for row in rows
        if row.get("accessory_type") == "earring"
    }
    assert all(
        "Never rotate the head" in row["prompt"]
        for row in rows
        if row.get("accessory_type")
    )
    assert all(
        "mask silhouette may replace the bare nose or mouth outline" in row["prompt"]
        for row in rows
        if row.get("augmentation_type") == "face_mask"
    )


def test_yawpose_regeneration_uses_fresh_prompt_and_scene_attributes():
    config = load_config(YAWPOSE_S005_CONFIG)
    record = build_plan(config, "yawpose_s005_5710", seed=20260903)[0]
    original = dict(record)
    original_prompt = str(record["prompt"])
    invariant_fields = (
        "yaw_yawpose",
        "head_pitch",
        "camera_elevation",
        "size",
        "headwear",
        "augmentation_type",
        "accessory_type",
        "accessory_description",
        "mask_description",
    )

    changes = _refresh_yawpose_regeneration_prompt(record, config, edit_round=13)

    expected_changed = {
        "context",
        "background",
        "lighting",
        "lens_feel",
        "gender",
        "age",
        "skin_tone",
        "hair",
        "clothing",
    }
    assert set(changes) == expected_changed
    assert all(record[field] != original[field] for field in expected_changed)
    assert record["scene"] == record["background"]
    assert record["prompt"] != original_prompt
    assert all(record.get(field) == original.get(field) for field in invariant_fields)

    wrapped = _regeneration_prompt(
        record,
        {"pose_status": "ok", "estimated_pan_deg": 303.2},
        ["yawpose_out_of_tolerance"],
    )
    assert original_prompt not in wrapped
    assert str(record["prompt"]) in wrapped
    assert "Do not reuse the failed candidate's person" in wrapped
    assert "newly sampled complete target prompt" in wrapped


def test_yawpose_s006_plan_matches_spec(tmp_path):
    run = create_plan(
        YAWPOSE_S006_CONFIG,
        "yawpose_s006_712",
        "production-yawpose-s006-v001",
        tmp_path,
        seed=20260904,
        direct_production=True,
        sequential_batches=True,
    )
    state = load_state(run)
    rows = list(read_plan(run, state).values())

    assert state["target_count"] == 712
    assert state["sequential_batches"] is True
    assert state["api_request"]["quality"] == "low"
    assert [len(shard["custom_ids"]) for shard in state["shards"]] == [500, 212]
    assert Counter(row["bin"] for row in rows) == {
        "yaw_100_110": 242,
        "yaw_110_120": 470,
    }
    for start, count in ((100, 242), (110, 470)):
        integer_counts = Counter(
            row["yaw_yawpose"]
            for row in rows
            if row["bin"] == f"yaw_{start}_{start + 10}"
        )
        assert set(integer_counts) == set(range(start, start + 10))
        assert max(integer_counts.values()) - min(integer_counts.values()) <= 1
        assert sum(integer_counts.values()) == count
    assert Counter(row["visible_side"] for row in rows) == {
        "profile_left": 242,
        "left_ear": 470,
    }
    assert sum(abs(row["pitch"]) > 10 for row in rows) == 142
    assert Counter(row["size"] for row in rows) == {
        "1024x1536": 356,
        "1536x1024": 356,
    }
    assert all(not row.get("augmentation_type") for row in rows)
    assert all(not row.get("accessory_type") for row in rows)
    assert all("The subject wears no face mask" in row["prompt"] for row in rows)
    assert all(row["anchor"] in row["prompt"] for row in rows)


def test_seal_collected_prefix_archives_only_never_submitted_tail(tmp_path):
    config = load_config(CONFIG)
    config["stages"]["uniform_200"].update(
        {
            "count": 2,
            "shard_size": 1,
            "bin_counts": [1, 1] + [0] * 17,
        }
    )
    config_path = tmp_path / "scope-fixture.yaml"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    run = create_plan(
        config_path,
        "uniform_200",
        "scope-fixture",
        tmp_path,
        seed=4,
        direct_production=True,
        sequential_batches=True,
    )
    state = load_state(run)
    plan = read_plan(run, state)
    first_id = state["shards"][0]["custom_ids"][0]
    first = plan[first_id]
    image_path = run / "images" / first["filename"]
    width, height = map(int, first["size"].split("x"))
    Image.new("RGB", (width, height), "gray").save(image_path, "JPEG")
    state["items"][first_id].update(
        {"status": "success", "sha256": sha256_file(image_path)}
    )
    state["shards"][0]["attempts"][0].update(
        {"status": "completed", "batch_id": "batch-completed"}
    )
    save_state(run, state)

    result = seal_collected_prefix(run)
    sealed = load_state(run)

    assert result["sealed_target_count"] == 1
    assert result["discarded_unsubmitted_requests"] == 1
    assert sealed["status"] == "collected"
    assert sealed["target_count"] == sealed["request_count"] == 1
    assert len(sealed["shards"]) == len(sealed["items"]) == 1
    archive = run / "superseded_unsubmitted"
    assert (archive / "original_batch_state.json").is_file()
    assert (
        len((archive / "original_generation_plan.jsonl").read_text().splitlines()) == 2
    )
    assert len((run / "generation_plan.jsonl").read_text().splitlines()) == 1


class _FakeFiles:
    def __init__(self):
        self.created = 0

    def create(self, *, file, purpose):
        assert purpose == "batch" and file.read(1)
        self.created += 1
        return SimpleNamespace(id=f"file-{self.created}")


class _FakeBatches:
    def __init__(self):
        self.created = 0

    def list(self, *, limit):
        assert limit == 100
        return SimpleNamespace(data=[])

    def create(self, **kwargs):
        self.created += 1
        return SimpleNamespace(
            id=f"batch-{self.created}",
            status="validating",
            output_file_id=None,
            error_file_id=None,
            request_counts=SimpleNamespace(total=19, completed=0, failed=0),
            metadata=kwargs["metadata"],
            errors=None,
        )


def test_submit_requires_exact_count_and_spend_cap(tmp_path):
    run = create_plan(CONFIG, "validation", "validation-v002", tmp_path, seed=3)
    client = SimpleNamespace(files=_FakeFiles(), batches=_FakeBatches())
    with pytest.raises(PipelineError, match="does not match"):
        submit_pending(run, approved_request_count=18, spend_cap_usd=1.0, client=client)
    with pytest.raises(PipelineError, match="exceeds spend cap"):
        submit_pending(
            run, approved_request_count=19, spend_cap_usd=0.01, client=client
        )
    state = load_state(run)
    state["planning_cost_per_request_usd"] = 0.05
    save_state(run, state)
    assert submit_pending(
        run, approved_request_count=19, spend_cap_usd=0.95, client=client
    ) == ["batch-1"]


def test_token_planned_batches_are_submitted_serially(tmp_path):
    run = create_plan(CONFIG, "validation", "validation-serialized", tmp_path, seed=3)
    state = load_state(run)
    second_shard = json.loads(json.dumps(state["shards"][0]))
    second_shard["index"] = 1
    state["shards"].append(second_shard)
    state["token_batch_plans"] = {
        "/v1/images/edits": {"queued_token_limit_exclusive": 1_000_000}
    }
    save_state(run, state)
    client = SimpleNamespace(files=_FakeFiles(), batches=_FakeBatches())

    remote = submit_pending(
        run,
        approved_request_count=38,
        spend_cap_usd=1.0,
        client=client,
    )

    assert remote == ["batch-1"]
    assert client.files.created == 1
    assert client.batches.created == 1
    assert pending_request_count(load_state(run)) == 19


def test_process_output_reconciles_custom_id_and_usage(tmp_path):
    run = create_plan(CONFIG, "validation", "validation-v003", tmp_path, seed=3)
    state = load_state(run)
    plan = read_plan(run, state)
    custom_id, record = next(iter(plan.items()))
    width, height = map(int, record["size"].split("x"))
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (20, 30, 40)).save(buffer, "JPEG", quality=92)
    row = {
        "custom_id": custom_id,
        "response": {
            "status_code": 200,
            "request_id": "request-1",
            "body": {
                "model": state["api_request"]["model"],
                "data": [{"b64_json": base64.b64encode(buffer.getvalue()).decode()}],
                "usage": {"output_tokens": 123},
            },
        },
    }
    output = run / "fixture_output.jsonl"
    output.write_text(json.dumps(row) + "\n", encoding="utf-8")
    changed, usage = process_output_jsonl(output, run, state, plan)
    assert changed == {custom_id}
    assert usage[0]["usage"] == {"output_tokens": 123}
    assert (run / "images" / record["filename"]).exists()


def test_qa_failures_create_hash_bound_high_fidelity_edit_cycle(tmp_path):
    root = tmp_path / "runs"
    parent = create_plan(CONFIG, "validation", "validation-parent", root, seed=3)
    state = load_state(parent)
    plan = read_plan(parent, state)
    qa_rows = []
    failed_parent_id = None
    for record in plan.values():
        width, height = map(int, record["size"].split("x"))
        image = Image.new("RGB", (width, height), (20 + int(record["serial"]), 40, 60))
        image.save(parent / "images" / record["filename"], "JPEG", quality=92)
        failed = int(record["serial"]) == 2
        if failed:
            failed_parent_id = record["custom_id"]
        qa_rows.append(
            {
                "custom_id": record["custom_id"],
                "abs_pan_bin": record["abs_pan_bin"],
                "camera_elevation": record["camera_elevation"],
                "pose_status": "ok",
                "pan_error_deg": 40.0 if failed else 0.0,
                "sixd_pitch_deg": 0.0 if failed else -float(record["camera_elevation"]),
                "direction_consistent": not failed,
                "quality_gate_reasons": ["head_too_small", "direction_conflict"]
                if failed
                else [],
                "margin_left_head_ratio": 1.0,
                "margin_right_head_ratio": 1.0,
                "margin_top_head_ratio": 1.0,
                "margin_bottom_head_ratio": 1.0,
            }
        )
    (parent / "auto_qa.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in qa_rows), encoding="utf-8"
    )

    child = create_edit_cycle(
        parent,
        "validation-edit01",
        root,
        max_edit_rounds=2,
        planning_cost_per_request_usd=0.03,
    )
    child_state = load_state(child)
    child_plan = read_plan(child, child_state)
    assert child_state["edit_round"] == 1
    assert child_state["request_count"] == 1
    assert child_state["planning_projected_cost_usd"] == pytest.approx(0.03)
    request = json.loads(
        (child / child_state["shards"][0]["attempts"][0]["input_path"]).read_text()
    )
    assert request["url"] == "/v1/images/edits"
    assert request["body"]["images"][0]["image_url"].startswith(
        "data:image/jpeg;base64,"
    )
    assert "input_fidelity" not in request["body"]
    assert "head_too_small" in request["body"]["prompt"]
    assert "physical gaze-reference object" not in request["body"]["prompt"]
    assert (
        "visible lower-body artifacts alone are acceptable" in request["body"]["prompt"]
    )
    failed_child_id = next(
        custom_id
        for custom_id, item in child_state["items"].items()
        if item["parent_custom_id"] == failed_parent_id
    )
    assert child_state["items"][failed_child_id]["operation"] == "edit"
    assert not (child / "images" / child_plan[failed_child_id]["filename"]).exists()
    carried = next(
        (custom_id, item)
        for custom_id, item in child_state["items"].items()
        if item["operation"] == "carry_forward"
    )
    assert (child / "images" / child_plan[carried[0]]["filename"]).exists()

    failed_child_record = child_plan[failed_child_id]
    width, height = map(int, failed_child_record["size"].split("x"))
    Image.new("RGB", (width, height), (80, 60, 40)).save(
        child / "images" / failed_child_record["filename"], "JPEG", quality=92
    )
    second_qa_rows = []
    for record in child_plan.values():
        failed = record["custom_id"] == failed_child_id
        second_qa_rows.append(
            {
                "custom_id": record["custom_id"],
                "abs_pan_bin": record["abs_pan_bin"],
                "camera_elevation": record["camera_elevation"],
                "pose_status": "ok",
                "pan_error_deg": 0.0,
                "sixd_pitch_deg": 0.0 if failed else -float(record["camera_elevation"]),
                "direction_consistent": True,
                "quality_gate_reasons": [],
            }
        )
    (child / "auto_qa.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in second_qa_rows), encoding="utf-8"
    )

    second_child = create_edit_cycle(
        child,
        "validation-edit02",
        root,
        max_edit_rounds=2,
        planning_cost_per_request_usd=0.03,
    )
    second_state = load_state(second_child)
    second_request = json.loads(
        (
            second_child / second_state["shards"][0]["attempts"][0]["input_path"]
        ).read_text()
    )
    second_prompt = second_request["body"]["prompt"]
    assert "The previous direct pitch correction failed" in second_prompt
    assert "a small plain red paper cup lying on its side" in second_prompt
    assert "directly below the subject's nose" in second_prompt
    assert "change pitch without changing yaw" in second_prompt
    assert (
        "rotating the entire head downward at the neck by about 25 degrees"
        in second_prompt
    )
    assert "not merely moving the eyes" in second_prompt
    assisted = next(
        item for item in second_state["items"].values() if item["operation"] == "edit"
    )
    assert assisted["pitch_reference_object"]["downward_correction_deg"] == 25
    assert assisted["pitch_reference_object"]["position"] == "lower-nose-aligned"


def test_final_edit_failures_can_be_regenerated_without_another_image_edit(tmp_path):
    root = tmp_path / "runs"
    parent = create_plan(CONFIG, "validation", "validation-edit04", root, seed=3)
    state = load_state(parent)
    state["edit_round"] = 4
    save_state(parent, state)
    plan = read_plan(parent, state)
    failed_parent_id = next(iter(plan))
    qa_rows = []
    for record in plan.values():
        width, height = map(int, record["size"].split("x"))
        Image.new("RGB", (width, height), (30, 60, 90)).save(
            parent / "images" / record["filename"], "JPEG", quality=92
        )
        failed = record["custom_id"] == failed_parent_id
        qa_rows.append(
            {
                "custom_id": record["custom_id"],
                "abs_pan_bin": record["abs_pan_bin"],
                "camera_elevation": record["camera_elevation"],
                "pose_status": "ok",
                "pan_error_deg": 40.0 if failed else 0.0,
                "sixd_pitch_deg": -float(record["camera_elevation"]),
                "direction_consistent": True,
                "quality_gate_reasons": ["rear_face_visible"] if failed else [],
            }
        )
    write_jsonl(parent / "auto_qa.jsonl", qa_rows)

    child = create_edit_cycle(
        parent,
        "validation-regen01",
        root,
        max_edit_rounds=4,
        planning_cost_per_request_usd=0.03,
        regenerate_quality_failures=True,
    )
    child_state = load_state(child)
    request = json.loads(
        (child / child_state["shards"][0]["attempts"][0]["input_path"])
        .read_text()
        .splitlines()[0]
    )
    regenerated = next(
        item
        for item in child_state["items"].values()
        if item["operation"] == "regenerate_quality_failure"
    )
    assert child_state["edit_round"] == 5
    assert child_state["max_edit_rounds"] == 4
    assert child_state["regenerate_quality_failures"] is True
    assert request["url"] == "/v1/images/generations"
    assert "no eye, eyebrow, nose bridge" in request["body"]["prompt"]
    assert "Generate a completely new independent image" in request["body"]["prompt"]
    assert "edit_prompt" not in regenerated
    assert regenerated["regeneration_prompt"] == request["body"]["prompt"]
    assert not list((child / "edit_inputs").glob("*.jpg"))


@pytest.mark.parametrize(("labelled_yaw", "retry_yaw"), [(97, 122), (253, 228)])
def test_rear_face_regeneration_aims_within_tolerance_toward_full_back(
    labelled_yaw, retry_yaw
):
    prompt = _regeneration_prompt(
        {
            "label_convention": "yawpose",
            "yaw_yawpose": labelled_yaw,
            "anchor": "Keep the named rear-side anchor.",
            "prompt": "Original labelled target.",
        },
        {},
        ["rear_face_visible"],
    )
    assert f"physical head yaw visually {retry_yaw:+d} degrees" in prompt
    assert "within the allowed 30-degree label tolerance" in prompt


@pytest.mark.parametrize(
    ("labelled_yaw", "estimated_yaw", "retry_yaw"),
    [(92, 52.4, 117), (20, 60.6, 355), (268, 300.8, 243)],
)
def test_pan_regeneration_compensates_observed_bias_within_tolerance(
    labelled_yaw, estimated_yaw, retry_yaw
):
    prompt = _regeneration_prompt(
        {
            "label_convention": "yawpose",
            "yaw_yawpose": labelled_yaw,
            "pan_detail": "Keep the requested directional anchor.",
            "prompt": "Original labelled target.",
        },
        {"pose_status": "ok", "estimated_pan_deg": estimated_yaw},
        ["yawpose_out_of_tolerance"],
    )
    assert f"physical head visually at {retry_yaw:+d} degrees" in prompt
    assert "within the allowed 30-degree tolerance" in prompt
    assert f"annotation label at {labelled_yaw:+d} degrees" in prompt
    if labelled_yaw == 20:
        assert "both eyes and both cheeks must be visible" in prompt
    else:
        assert "unmistakable strict side profile" in prompt


def test_mask_augmentation_plans_twenty_percent_as_separate_edits(tmp_path):
    root = tmp_path / "runs"
    parent = create_plan(CONFIG, "validation", "mask-parent", root, seed=3)
    state = load_state(parent)
    plan = read_plan(parent, state)
    qa_rows = []
    usage_rows = []
    for record in plan.values():
        width, height = map(int, record["size"].split("x"))
        Image.new("RGB", (width, height), (60, 80, 100)).save(
            parent / "images" / record["filename"], "JPEG", quality=92
        )
        qa_rows.append(
            {
                "custom_id": record["custom_id"],
                "quality_gate_pass": True,
                "pan_quality_pass_auto": True,
            }
        )
        usage_rows.append(
            {"custom_id": record["custom_id"], "usage": {"input_tokens": 1500}}
        )
    write_jsonl(parent / "auto_qa.jsonl", qa_rows)
    (parent / "qa_report.json").write_text(
        json.dumps(
            {
                "total": len(plan),
                "quality_pass": len(plan),
                "pan_quality_pass_auto": len(plan),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    write_jsonl(parent / "usage.jsonl", usage_rows)
    attempt = state["shards"][0]["attempts"][0]
    attempt["endpoint"] = "/v1/images/edits"
    attempt["custom_ids"] = list(plan)
    state["shards"][0]["custom_ids"] = list(plan)
    save_state(parent, state)

    child = create_mask_augmentation(
        parent,
        "production-mask20-v001",
        root,
        target_fraction=0.20,
        planning_cost_per_request_usd=0.009,
        seed=7,
    )
    child_state = load_state(child)
    child_plan = read_plan(child, child_state)

    assert child_state["base_dataset_count"] == 19
    assert child_state["request_count"] == 5
    assert child_state["projected_combined_count"] == 24
    assert child_state["projected_mask_fraction"] == pytest.approx(5 / 24)
    assert child_state["augmentation_type"] == "face_mask"
    assert len(child_state["shards"]) == 1
    assert (
        child_state["token_batch_plans"]["/v1/images/edits"][
            "observed_mean_input_tokens"
        ]
        == 1500
    )
    assert all(record["abs_pan_bin"] <= 90 for record in child_plan.values())
    assert all(
        record["augmentation_type"] == "face_mask" for record in child_plan.values()
    )
    assert all(
        "over the nose, mouth, and chin" in record["prompt"]
        for record in child_plan.values()
    )
    assert all(
        child_state["items"][custom_id]["operation"] == "mask_augmentation_edit"
        for custom_id in child_plan
    )
    assert len(list((child / "edit_inputs").glob("*.jpg"))) == 5
    request = json.loads(
        (child / child_state["shards"][0]["attempts"][0]["input_path"])
        .read_text()
        .splitlines()[0]
    )
    assert request["url"] == "/v1/images/edits"
    assert request["body"]["quality"] == "low"
    assert "input_fidelity" not in request["body"]


def test_usage_report_and_model_install_are_explicit_and_hash_checked(tmp_path):
    run = create_plan(
        CONFIG, "validation", "validation-v004", tmp_path / "runs", seed=3
    )
    report = build_usage_report(run)
    assert report["cost_basis"] == "documented_reference_only"
    assert report["actual_cost_per_completed_usd"] is None
    state = load_state(run)
    requested_id = state["shards"][0]["custom_ids"][0]
    for item in state["items"].values():
        item["status"] = "success"
    state["request_count"] = 1
    state["shards"][0]["custom_ids"] = [requested_id]
    save_state(run, state)
    write_jsonl(
        run / "auto_qa.jsonl",
        [
            {
                "pan_quality_pass_auto": True,
                "camera_elevation_class_auto": "high_angle_match",
            }
            for _ in state["items"]
        ],
    )
    edit_style_report = build_usage_report(run)
    assert edit_style_report["completed_requests"] == 1
    assert edit_style_report["completed_images"] == 19
    assert edit_style_report["failed_or_missing"] == 0
    assert edit_style_report["pan_quality_pass"] == 19
    assert edit_style_report["quality_source"] == "auto_qa"
    with pytest.raises(PipelineError, match="positive"):
        build_usage_report(run, actual_cost_usd=0.0)

    source_repo = tmp_path / "source"
    source = source_repo / "weights" / "fixture.onnx"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"verified model fixture")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    assets = {
        "fixture": {
            "source": "weights/fixture.onnx",
            "target": "data/models/fixture.onnx",
            "sha256": digest,
        }
    }
    installed = install_model_assets(
        assets, source_repo=source_repo, repository_root=tmp_path / "repo"
    )
    assert installed[0]["status"] == "copied"
    installed_again = install_model_assets(
        assets, source_repo=source_repo, repository_root=tmp_path / "repo"
    )
    assert installed_again[0]["status"] == "present"


def test_standalone_conversion_archives_parent_evidence_and_requires_full_qa(
    tmp_path,
):
    run = create_plan(CONFIG, "validation", "validation-standalone", tmp_path, seed=3)
    state = load_state(run)
    for item in state["items"].values():
        item["status"] = "success"
    for shard in state["shards"]:
        for attempt in shard["attempts"]:
            attempt["status"] = "completed"
    state.update(
        {
            "direct_production": True,
            "approval_policy": "operator_direct_no_human_review",
            "edit_round": 1,
            "parent_batch_dir": str(tmp_path / "parent"),
            "parent_state_sha256": "state-hash",
            "parent_plan_sha256": "plan-hash",
            "parent_qa_sha256": "qa-hash",
            "parent_approval_sha256": None,
        }
    )
    evidence = tmp_path / "parent_usage.jsonl"
    evidence.write_text('{"usage": {"input_tokens": 10}}\n', encoding="utf-8")
    state["token_batch_plans"] = {
        "/v1/images/edits": {
            "evidence_run": str(tmp_path / "parent"),
            "usage_path": str(evidence),
            "usage_sha256": sha256_file(evidence),
        }
    }
    save_state(run, state)
    count = int(state["target_count"])
    write_jsonl(run / "auto_qa.jsonl", ({"custom_id": str(i)} for i in range(count)))
    write_jsonl(
        run / "accepted_annotations.jsonl",
        ({"custom_id": str(i)} for i in range(count)),
    )
    (run / "qa_report.json").write_text(
        json.dumps(
            {
                "total": count,
                "quality_pass": count,
                "operator_label_promotion": {"total_accepted": count},
            }
        ),
        encoding="utf-8",
    )

    prepared = prepare_standalone_run(run)
    detached = load_state(run)
    assert prepared["parent_detached"] is True
    assert detached["parent_batch_dir"] is None
    assert detached["standalone_conversion"]["status"] == "prepared_for_full_qa"
    local_usage = Path(detached["token_batch_plans"]["/v1/images/edits"]["usage_path"])
    assert local_usage.is_file()
    assert run.resolve() in local_usage.parents
    provenance = json.loads(
        Path(prepared["provenance_manifest"]).read_text(encoding="utf-8")
    )
    assert provenance["original_parent"]["parent_qa_sha256"] == "qa-hash"

    with pytest.raises(PipelineError, match="non-reused"):
        finalize_standalone_run(run)
    (run / "qa_report.json").write_text(
        json.dumps(
            {
                "total": count,
                "quality_pass": count,
                "qa_reuse": {
                    "compatibility": "not_applicable",
                    "reused_passed_records": 0,
                    "evaluated_current_run_records": count,
                },
                "operator_label_promotion": {"total_accepted": count},
            }
        ),
        encoding="utf-8",
    )
    finalized = finalize_standalone_run(run)
    assert finalized["status"] == "verified_standalone"
    assert finalized["evaluated_current_run_records"] == count


def test_standalone_conversion_can_preserve_hash_bound_passed_qa(tmp_path):
    run = create_plan(CONFIG, "validation", "validation-reused-qa", tmp_path, seed=3)
    state = load_state(run)
    state.update(
        {
            "direct_production": True,
            "approval_policy": "operator_direct_no_human_review",
            "edit_round": 1,
            "parent_batch_dir": str(tmp_path / "parent"),
            "parent_state_sha256": "state-hash",
            "parent_plan_sha256": "plan-hash",
            "parent_qa_sha256": "qa-hash",
            "parent_approval_sha256": None,
        }
    )
    for item in state["items"].values():
        item["status"] = "success"
    for shard in state["shards"]:
        for attempt in shard["attempts"]:
            attempt["status"] = "completed"
    save_state(run, state)
    plan = read_plan(run, state)
    qa_rows = []
    for record in plan.values():
        image_path = run / "images" / record["filename"]
        image_path.write_bytes(f"image:{record['custom_id']}".encode())
        qa_rows.append(
            {
                "custom_id": record["custom_id"],
                "filename": record["filename"],
                "quality_gate_pass": True,
                "pan_quality_pass_auto": True,
                "sha256": sha256_file(image_path),
            }
        )
    write_jsonl(run / "auto_qa.jsonl", qa_rows)
    write_jsonl(
        run / "accepted_annotations.jsonl",
        ({"custom_id": row["custom_id"]} for row in qa_rows),
    )
    count = len(qa_rows)
    reused = count - 2
    (run / "qa_report.json").write_text(
        json.dumps(
            {
                "total": count,
                "quality_pass": count,
                "pan_quality_pass_auto": count,
                "qa_reuse": {
                    "compatibility": "matched",
                    "reused_passed_records": reused,
                    "evaluated_current_run_records": 2,
                },
                "operator_label_promotion": {"total_accepted": count},
            }
        ),
        encoding="utf-8",
    )

    prepare_standalone_run(run)
    with pytest.raises(PipelineError, match="allow_reused_passed_qa"):
        finalize_standalone_run(run)
    finalized = finalize_standalone_run(run, allow_reused_passed_qa=True)

    assert finalized["status"] == "verified_standalone"
    assert finalized["qa_verification_mode"] == "hash_bound_passed_qa_reuse"
    assert finalized["reused_passed_records"] == reused
    assert finalized["evaluated_current_run_records"] == 2


def test_pilot_cost_projection_uses_hash_bound_validation_account_cost(tmp_path):
    root = tmp_path / "runs"
    parent = create_plan(CONFIG, "validation", "validation-v005", root, seed=3)
    state = load_state(parent)
    first = next(iter(state["items"].values()))
    first["status"] = "success"
    save_state(parent, state)
    usage = build_usage_report(parent, actual_cost_usd=0.02)

    review = parent / "human_review.csv"
    protocol = tmp_path / "test_profiles_protocol.json"
    profile_manifest = tmp_path / "test_profiles.jsonl"
    sign = parent / "sign_calibration.json"
    calibration = parent / "pitch_calibration.json"
    review.write_text("reviewed\n", encoding="utf-8")
    profile_manifest.write_text("{}\n", encoding="utf-8")
    protocol.write_text(
        json.dumps(
            {
                "split": "test_profiles",
                "records": 250,
                "people": 200,
                "bootstrap_resamples": 10_000,
                "promotion_rule": "primary 95% CI wholly above zero",
                "manifest_path": str(profile_manifest.resolve()),
                "manifest_sha256": sha256_file(profile_manifest),
            }
        ),
        encoding="utf-8",
    )
    sign.write_text("{}\n", encoding="utf-8")
    calibration.write_text('{"valid": true}\n', encoding="utf-8")
    usage_path = parent / "usage_report.json"
    approval = {
        "approved": True,
        "stage": "validation",
        "review_path": review.name,
        "review_sha256": sha256_file(review),
        "evaluation_protocol": str(protocol.resolve()),
        "evaluation_protocol_sha256": sha256_file(protocol),
        "usage_report": str(usage_path.resolve()),
        "usage_report_sha256": sha256_file(usage_path),
        "sign_calibration_approved": True,
        "sign_calibration_path": sign.name,
        "sign_calibration_sha256": sha256_file(sign),
        "pitch_calibration": str(calibration.resolve()),
        "pitch_calibration_sha256": sha256_file(calibration),
        "account_verified_snapshot": state["api_request"]["model"],
    }
    (parent / "approval.json").write_text(json.dumps(approval), encoding="utf-8")

    pilot = create_plan(
        CONFIG,
        "pilot",
        "pilot-v001",
        root,
        seed=3,
        approved_batch_dir=parent,
    )
    pilot_state = load_state(pilot)
    assert usage["actual_cost_per_completed_usd"] == pytest.approx(0.02)
    assert pilot_state["planning_cost_basis"] == "parent_account_observed"
    assert pilot_state["planning_projected_cost_usd"] == pytest.approx(7.6)
