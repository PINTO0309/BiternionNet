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
    build_plan,
    build_usage_report,
    create_edit_cycle,
    create_plan,
    load_config,
    load_state,
    process_output_jsonl,
    read_plan,
    refresh_status,
    save_state,
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


def test_pan_edit_prompt_uses_relative_correction_from_current_pose():
    record = {
        "intent_pan_deg": 20,
        "signed_pan": 20,
        "pan_detail": "Keep the final view mostly frontal.",
        "prompt": "Original target.",
    }
    row = {"pose_status": "ok", "estimated_pan_deg": 57.2929}
    prompt = _edit_prompt(record, row, ["direction_conflict"])
    assert "estimated at pan +57.3 degrees" in prompt
    assert "37.3 degrees toward image-left" in prompt
    assert "not an instruction to turn farther" in prompt


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
