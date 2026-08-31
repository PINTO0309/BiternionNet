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
    build_plan,
    build_usage_report,
    create_plan,
    load_config,
    load_state,
    process_output_jsonl,
    read_plan,
    refresh_status,
    save_state,
    sha256_file,
    submit_pending,
)
from biternionnet.synthetic.models import install_model_assets

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "synthetic_towncentre.yaml"


def test_validation_and_pilot_are_deterministic_and_cover_directions():
    config = load_config(CONFIG)
    validation = build_plan(config, "validation", seed=7)
    assert len(validation) == 19
    assert [row["abs_pan_bin"] for row in validation] == list(range(0, 181, 10))
    assert {row["expected_direction"] for row in validation} == {
        "front", "left_front", "left_side", "left_back", "back",
        "right_back", "right_side", "right_front",
    }
    assert any(row["abs_pan_bin"] == 180 and row["expected_direction"] == "back" for row in validation)
    assert {row["camera_elevation"] for row in validation} == {30, 45, 60}

    first = build_plan(config, "pilot", seed=19)
    second = build_plan(config, "pilot", seed=19)
    assert first == second
    assert len(first) == 380
    assert Counter(row["abs_pan_bin"] for row in first) == {value: 20 for value in range(0, 181, 10)}
    for value in range(10, 180, 10):
        signs = [row["signed_pan"] > 0 for row in first if row["abs_pan_bin"] == value]
        assert Counter(signs) == {True: 10, False: 10}


def test_plan_fixes_low_quality_and_is_immutable(tmp_path):
    run = create_plan(CONFIG, "validation", "validation-v001", tmp_path, seed=3)
    state = load_state(run)
    assert state["request_count"] == 19
    assert state["api_request"]["quality"] == "low"
    assert len(state["shards"]) == 1
    requests = [json.loads(line) for line in (run / state["shards"][0]["attempts"][0]["input_path"]).read_text().splitlines()]
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
        submit_pending(run, approved_request_count=19, spend_cap_usd=0.01, client=client)
    assert submit_pending(run, approved_request_count=19, spend_cap_usd=1.0, client=client) == ["batch-1"]


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


def test_usage_report_and_model_install_are_explicit_and_hash_checked(tmp_path):
    run = create_plan(CONFIG, "validation", "validation-v004", tmp_path / "runs", seed=3)
    report = build_usage_report(run)
    assert report["cost_basis"] == "documented_reference_only"
    assert report["actual_cost_per_completed_usd"] is None
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
    installed = install_model_assets(assets, source_repo=source_repo, repository_root=tmp_path / "repo")
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
