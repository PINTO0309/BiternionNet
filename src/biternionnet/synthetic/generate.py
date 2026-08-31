"""Deterministic GPT-Image-2 Batch pipeline for TownCentre synthetic heads.

Adapted from HRFFA's MIT-licensed ``gpt_head_gen.py`` at commit
1155c7f7b3f07c649c64f45516750f86ca0e7015.  This version deliberately keeps
planning, paid submission, collection, and retries as separate operations.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import io
import json
import os
import random
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

import yaml
from PIL import Image, UnidentifiedImageError

ENDPOINT = "/v1/images/generations"
QUALITY = "low"
BACKGROUND = "opaque"
OUTPUT_FORMAT = "jpeg"
OUTPUT_COMPRESSION = 92
N_IMAGES = 1
COMPLETION_WINDOW = "24h"
BATCH_IMAGE_MODEL = "gpt-image-2"
ALLOWED_SIZES = {"1024x1536", "1024x1024", "1536x1024"}
STATE_NAME = "batch_state.json"
PLAN_NAME = "generation_plan.jsonl"
TERMINAL_STATUSES = {"completed", "failed", "expired", "cancelled"}
ACTIVE_STATUSES = {"validating", "in_progress", "finalizing", "cancelling"}

DIR8_BY_CENTRE = {
    0: "front",
    45: "left_front",
    90: "left_side",
    135: "left_back",
    180: "back",
    225: "right_back",
    270: "right_side",
    315: "right_front",
}

ORIENTATION_BY_CENTRE = {
    0: "front, facing the camera",
    45: "front-left, turning toward image-right",
    90: "subject's left side visible, facing image-right",
    135: "back-left, turning away toward image-right",
    180: "unambiguous back of the head",
    225: "back-right, turning away toward image-left",
    270: "subject's right side visible, facing image-left",
    315: "front-right, turning toward image-left",
}


class PipelineError(RuntimeError):
    """A user-actionable synthetic-pipeline error."""


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "__dict__"):
        return {str(key): _jsonable(item) for key, item in vars(value).items()}
    return value


def _get(value: Any, key: str, default: Any = None) -> Any:
    return value.get(key, default) if isinstance(value, dict) else getattr(value, key, default)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise PipelineError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
    return rows


def validate_evaluation_protocol(path: Path) -> dict[str, Any]:
    try:
        protocol = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PipelineError(f"invalid evaluation protocol: {path}") from exc
    if protocol.get("split") != "test_profiles":
        raise PipelineError("evaluation protocol split must be test_profiles")
    records = int(protocol.get("records", 0))
    people = int(protocol.get("people", 0))
    if not 200 <= records <= 300 or people < 150:
        raise PipelineError("evaluation protocol needs 200..300 records and at least 150 people")
    if int(protocol.get("bootstrap_resamples", 0)) < 10_000:
        raise PipelineError("evaluation protocol needs at least 10000 bootstrap resamples")
    if "95% CI" not in str(protocol.get("promotion_rule", "")):
        raise PipelineError("evaluation protocol promotion rule must use a 95% CI")
    manifest_digest = str(protocol.get("manifest_sha256", ""))
    if len(manifest_digest) != 64:
        raise PipelineError("evaluation protocol must bind the test_profiles manifest SHA-256")
    manifest_value = protocol.get("manifest_path")
    if not manifest_value:
        raise PipelineError("evaluation protocol must record the test_profiles manifest path")
    manifest_path = Path(str(manifest_value))
    if not manifest_path.exists() or sha256_file(manifest_path) != manifest_digest:
        raise PipelineError("test_profiles manifest is missing or changed")
    return protocol


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise PipelineError("synthetic config must be a mapping")
    api = config.get("api") or {}
    fixed = {
        "endpoint": ENDPOINT,
        "completion_window": COMPLETION_WINDOW,
        "quality": QUALITY,
        "background": BACKGROUND,
        "output_format": OUTPUT_FORMAT,
        "output_compression": OUTPUT_COMPRESSION,
        "n": N_IMAGES,
    }
    for key, expected in fixed.items():
        if api.get(key) != expected:
            raise PipelineError(f"config api.{key} must be fixed to {expected!r}")
    if not isinstance(api.get("model"), str) or not api["model"].strip():
        raise PipelineError("config api.model must be non-empty")
    if set(api.get("allowed_sizes") or []) != ALLOWED_SIZES:
        raise PipelineError(f"config api.allowed_sizes must equal {sorted(ALLOWED_SIZES)}")
    storage = config.get("storage") or {}
    if storage.get("format") != "jpeg" or storage.get("quality") != OUTPUT_COMPRESSION:
        raise PipelineError("storage must be JPEG quality 92")
    models = config.get("models") or {}
    expected_models = {"deimv2", "sixdrepnet360", "hrffa_vitl_ibug68"}
    if set(models) != expected_models:
        raise PipelineError(f"models must define exactly {sorted(expected_models)}")
    for name, asset in models.items():
        if not all(isinstance(asset.get(key), str) and asset[key] for key in ("source", "target")):
            raise PipelineError(f"models.{name} must define source and target")
        digest = asset.get("sha256")
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise PipelineError(f"models.{name}.sha256 must be a lowercase SHA-256 digest")
    targets = config.get("targets") or {}
    bins = list(targets.get("abs_pan_bins") or [])
    if bins != list(range(0, 181, 10)):
        raise PipelineError("targets.abs_pan_bins must be 0..180 in 10-degree steps")
    validation = config.get("validation") or []
    if len(validation) != 19 or [int(row["abs_pan"]) for row in validation] != bins:
        raise PipelineError("validation must contain exactly one ordered record per absolute pan bin")
    stages = config.get("stages") or {}
    for stage, stage_config in stages.items():
        count = int(stage_config["count"])
        shard_size = int(stage_config["shard_size"])
        if count <= 0 or not 1 <= shard_size <= 500:
            raise PipelineError(f"invalid stage size for {stage}")
        bin_counts = stage_config.get("bin_counts")
        if bin_counts is not None and (len(bin_counts) != 19 or sum(map(int, bin_counts)) != count):
            raise PipelineError(f"invalid bin_counts for {stage}")
    return config


def wrap180(angle: float) -> float:
    return (float(angle) + 180.0) % 360.0 - 180.0


def circular_error_deg(left: float, right: float) -> float:
    return abs(wrap180(float(left) - float(right)))


def sector_centre(pan_deg: float) -> int:
    return int(((float(pan_deg) % 360.0 + 22.5) // 45.0) % 8) * 45


def expected_direction(pan_deg: float) -> str:
    return DIR8_BY_CENTRE[sector_centre(pan_deg)]


def _exact_schedule(config_rows: list[dict[str, Any]], total: int, rng: random.Random) -> list[str]:
    result: list[str] = []
    allocated = 0
    for index, row in enumerate(config_rows):
        count = total - allocated if index == len(config_rows) - 1 else total * int(row["share"]) // 100
        result.extend([str(row["value"])] * count)
        allocated += count
    rng.shuffle(result)
    return result


def _make_prompt(config: dict[str, Any], record: dict[str, Any]) -> str:
    prompt = config["prompt"]
    appearance = (
        f"Subject: a fictional adult {record['gender']} in their {record['age']}, "
        f"skin tone {record['skin_tone']}, {record['hair']}, wearing {record['clothing']}; "
        f"accessory: {record['accessory']}. Scene: {record['scene']}; {record['lighting']}."
    )
    return " ".join(
        [
            prompt["preamble"],
            prompt["camera"].format(**record),
            prompt["pan"].format(**record),
            appearance,
            prompt["framing"],
            prompt["realism"],
        ]
    )


def _custom_id(signed_pan: int, camera_elevation: int, head_pitch: int, serial: int) -> str:
    return (
        f"pan{signed_pan:+04d}_cam{camera_elevation:+03d}_"
        f"pitch{head_pitch:+04d}_{serial:06d}"
    )


def _image_filename(
    signed_pan: int,
    camera_elevation: int,
    head_pitch: int,
    serial: int,
    *,
    batch_id: str | None = None,
) -> str:
    prefix = f"{batch_id}_" if batch_id else ""
    return (
        f"{prefix}{serial:06d}--pan{signed_pan:+04d}_"
        f"cam{camera_elevation:+03d}_pitch{head_pitch:+04d}.jpg"
    )


def _record(
    config: dict[str, Any],
    *,
    stage: str,
    serial: int,
    abs_pan: int,
    signed_pan: int,
    camera_elevation: int,
    head_pitch: int,
    size: str,
) -> dict[str, Any]:
    prompt = config["prompt"]
    pan = signed_pan % 360
    centre = sector_centre(pan)
    index = serial - 1
    record: dict[str, Any] = {
        "serial": serial,
        "stage": stage,
        "abs_pan_bin": abs_pan,
        "signed_pan": signed_pan,
        "intent_pan_deg": float(pan),
        "camera_elevation": camera_elevation,
        "head_pitch": head_pitch,
        "roll": 0,
        "size": size,
        "expected_direction": DIR8_BY_CENTRE[centre],
        "orientation": ORIENTATION_BY_CENTRE[centre],
        "scene": prompt["scenes"][index % len(prompt["scenes"])],
        "lighting": prompt["lighting"][index % len(prompt["lighting"])],
        "age": prompt["ages"][index % len(prompt["ages"])],
        "gender": prompt["gender_presentations"][index % len(prompt["gender_presentations"])],
        "skin_tone": prompt["skin_tones"][index % len(prompt["skin_tones"])],
        "hair": prompt["hair"][index % len(prompt["hair"])],
        "clothing": prompt["clothing"][index % len(prompt["clothing"])],
        "accessory": prompt["accessories"][index % len(prompt["accessories"])],
    }
    record["custom_id"] = _custom_id(signed_pan, camera_elevation, head_pitch, serial)
    record["filename"] = _image_filename(
        signed_pan,
        camera_elevation,
        head_pitch,
        serial,
    )
    record["prompt"] = _make_prompt(config, record)
    return record


def build_plan(
    config: dict[str, Any],
    stage: str,
    seed: int,
    *,
    bin_counts: list[int] | None = None,
    serial_offset: int = 0,
) -> list[dict[str, Any]]:
    if stage not in config["stages"]:
        raise PipelineError(f"unknown stage: {stage}")
    rng = random.Random(seed)
    generation = config["generation"]
    if stage == "validation":
        if bin_counts is not None or serial_offset:
            raise PipelineError("validation does not support partial overrides")
        assignments = [dict(row) for row in config["validation"]]
    else:
        counts = list(map(int, bin_counts or config["stages"][stage]["bin_counts"]))
        if len(counts) != 19:
            raise PipelineError("bin_counts must contain 19 values")
        assignments = [
            {"abs_pan": abs_pan, "occurrence": occurrence}
            for abs_pan, count in zip(config["targets"]["abs_pan_bins"], counts)
            for occurrence in range(count)
        ]
        rng.shuffle(assignments)
    sizes = _exact_schedule(generation["size_schedule"], len(assignments), rng)
    records: list[dict[str, Any]] = []
    local_counts = {value: 0 for value in config["targets"]["abs_pan_bins"]}
    for index, assignment in enumerate(assignments):
        abs_pan = int(assignment["abs_pan"])
        occurrence = local_counts[abs_pan]
        local_counts[abs_pan] += 1
        if "signed_pan" in assignment:
            signed_pan = int(assignment["signed_pan"])
        elif abs_pan in {0, 180}:
            signed_pan = abs_pan
        else:
            signed_pan = abs_pan if occurrence % 2 == 0 else -abs_pan
        camera_elevation = int(
            assignment.get(
                "camera_elevation",
                rng.randint(
                    int(generation["camera_elevation"]["min"]),
                    int(generation["camera_elevation"]["max"]),
                ),
            )
        )
        head_pitch = int(
            assignment.get(
                "head_pitch",
                rng.randint(
                    int(generation["head_pitch"]["min"]),
                    int(generation["head_pitch"]["max"]),
                ),
            )
        )
        records.append(
            _record(
                config,
                stage=stage,
                serial=serial_offset + index + 1,
                abs_pan=abs_pan,
                signed_pan=signed_pan,
                camera_elevation=camera_elevation,
                head_pitch=head_pitch,
                size=sizes[index],
            )
        )
    custom_ids = [row["custom_id"] for row in records]
    if len(custom_ids) != len(set(custom_ids)):
        raise PipelineError("planner produced duplicate custom IDs")
    return records


def batch_request(record: dict[str, Any], api: dict[str, Any]) -> dict[str, Any]:
    request = {
        "custom_id": record["custom_id"],
        "method": "POST",
        "url": ENDPOINT,
        "body": {
            "model": api["model"],
            "prompt": record["prompt"],
            "n": N_IMAGES,
            "size": record["size"],
            "quality": QUALITY,
            "background": BACKGROUND,
            "output_format": OUTPUT_FORMAT,
            "output_compression": OUTPUT_COMPRESSION,
        },
    }
    validate_batch_request(request, api)
    return request


def validate_batch_request(request: dict[str, Any], api: dict[str, Any]) -> None:
    if request.get("method") != "POST" or request.get("url") != ENDPOINT:
        raise PipelineError(f"every request must POST to {ENDPOINT}")
    if not isinstance(request.get("custom_id"), str) or not request["custom_id"]:
        raise PipelineError("custom_id must be non-empty")
    body = request.get("body")
    if not isinstance(body, dict):
        raise PipelineError("request body must be an object")
    fixed = {
        "model": api["model"],
        "n": N_IMAGES,
        "quality": QUALITY,
        "background": BACKGROUND,
        "output_format": OUTPUT_FORMAT,
        "output_compression": OUTPUT_COMPRESSION,
    }
    for key, expected in fixed.items():
        if body.get(key) != expected:
            raise PipelineError(f"request body.{key} must be {expected!r}")
    if body.get("size") not in ALLOWED_SIZES:
        raise PipelineError(f"unsupported image size: {body.get('size')!r}")
    if not isinstance(body.get("prompt"), str) or not body["prompt"].strip():
        raise PipelineError("prompt must be non-empty")
    if set(body) != {
        "model", "prompt", "n", "size", "quality", "background", "output_format",
        "output_compression",
    }:
        raise PipelineError("unexpected or missing image generation request options")


def validate_batch_jsonl(path: Path, api: dict[str, Any]) -> list[str]:
    custom_ids: list[str] = []
    for row in read_jsonl(path):
        validate_batch_request(row, api)
        custom_ids.append(row["custom_id"])
    if not custom_ids or len(custom_ids) != len(set(custom_ids)):
        raise PipelineError(f"empty or duplicate custom IDs in {path}")
    return custom_ids


def _approved_parent(parent: Path, required_stage: str) -> dict[str, Any]:
    approval_path = parent / "approval.json"
    if not approval_path.exists():
        raise PipelineError(f"{required_stage} approval missing: {approval_path}")
    approval = json.loads(approval_path.read_text(encoding="utf-8"))
    if approval.get("approved") is not True or approval.get("stage") != required_stage:
        raise PipelineError(f"approval is not an approved {required_stage} run")
    reviewed = parent / str(approval.get("review_path", "human_review.csv"))
    if not reviewed.exists() or approval.get("review_sha256") != sha256_file(reviewed):
        raise PipelineError("approved human review is missing or changed")
    protocol_value = approval.get("evaluation_protocol")
    if not protocol_value:
        raise PipelineError("approved run does not bind an evaluation protocol")
    protocol = Path(protocol_value)
    if not protocol.exists() or approval.get("evaluation_protocol_sha256") != sha256_file(protocol):
        raise PipelineError("approved evaluation protocol is missing or changed")
    validate_evaluation_protocol(protocol)
    usage_value = approval.get("usage_report")
    if not usage_value:
        raise PipelineError("approved run does not bind a usage/cost report")
    usage_report = Path(usage_value)
    if not usage_report.exists() or approval.get("usage_report_sha256") != sha256_file(usage_report):
        raise PipelineError("approved usage/cost report is missing or changed")
    sign_value = approval.get("sign_calibration_path")
    if approval.get("sign_calibration_approved") is not True or not sign_value:
        raise PipelineError("approved run does not bind sign calibration")
    sign_path = parent / str(sign_value)
    if not sign_path.exists() or approval.get("sign_calibration_sha256") != sha256_file(sign_path):
        raise PipelineError("approved sign calibration is missing or changed")
    calibration_value = approval.get("pitch_calibration")
    if not calibration_value:
        raise PipelineError("approved run does not bind pitch calibration")
    calibration_path = Path(str(calibration_value))
    if (
        not calibration_path.exists()
        or approval.get("pitch_calibration_sha256") != sha256_file(calibration_path)
    ):
        raise PipelineError("approved pitch calibration is missing or changed")
    if required_stage == "pilot":
        rear_value = approval.get("rear_label_policy_path")
        if not rear_value:
            raise PipelineError("approved Pilot does not bind a rear label policy")
        rear_path = parent / str(rear_value)
        if not rear_path.exists() or approval.get("rear_label_policy_sha256") != sha256_file(rear_path):
            raise PipelineError("approved Pilot rear label policy is missing or changed")
    return approval


def create_plan(
    config_path: Path,
    stage: str,
    batch_id: str,
    output_root: Path,
    seed: int,
    approved_batch_dir: Path | None = None,
    *,
    bin_counts: list[int] | None = None,
) -> Path:
    if not batch_id or any(character in batch_id for character in "/\\"):
        raise PipelineError("batch-id must be one safe path component")
    config_path = config_path.resolve()
    config = load_config(config_path)
    if config["api"]["model"] != BATCH_IMAGE_MODEL:
        raise PipelineError(
            f"Batch image generation requires api.model={BATCH_IMAGE_MODEL!r}; "
            "dated GPT-Image-2 snapshots are not accepted by the Batch API"
        )
    required_parent = {"pilot": "validation", "floor_120": "pilot", "uniform_200": "pilot"}.get(stage)
    parent_approval = None
    if required_parent:
        if approved_batch_dir is None:
            raise PipelineError(f"{stage} planning requires an approved {required_parent} directory")
        parent_approval = _approved_parent(approved_batch_dir, required_parent)
        if required_parent == "validation" and parent_approval.get("account_verified_snapshot") != config["api"]["model"]:
            raise PipelineError("Validation did not verify the configured snapshot for Pilot")
        parent_usage = json.loads(
            Path(parent_approval["usage_report"]).read_text(encoding="utf-8")
        )
        if parent_usage.get("actual_cost_per_completed_usd") is None:
            raise PipelineError(
                f"{stage} planning requires the approved {required_parent} run's actual account cost"
            )
    records = build_plan(config, stage, seed, bin_counts=bin_counts)
    for record in records:
        record["custom_id"] = f"{batch_id}--{record['custom_id']}"
        record["filename"] = _image_filename(
            int(record["signed_pan"]),
            int(record["camera_elevation"]),
            int(record["head_pitch"]),
            int(record["serial"]),
            batch_id=batch_id,
        )
    run_dir = output_root / "batches" / batch_id
    if run_dir.exists():
        raise PipelineError(f"refusing to overwrite existing run: {run_dir}")
    run_dir.mkdir(parents=True)
    (run_dir / "images").mkdir()
    write_jsonl(run_dir / PLAN_NAME, records)
    shard_size = int(config["stages"][stage]["shard_size"])
    shards: list[dict[str, Any]] = []
    for shard_index, start in enumerate(range(0, len(records), shard_size)):
        chunk = records[start : start + shard_size]
        input_name = f"batch_input_{shard_index:03d}_attempt_00.jsonl"
        input_path = run_dir / input_name
        write_jsonl(input_path, (batch_request(row, config["api"]) for row in chunk))
        custom_ids = validate_batch_jsonl(input_path, config["api"])
        shards.append(
            {
                "index": shard_index,
                "custom_ids": custom_ids,
                "attempts": [
                    {
                        "number": 0,
                        "input_path": input_name,
                        "input_sha256": sha256_file(input_path),
                        "custom_ids": custom_ids,
                        "input_file_id": None,
                        "batch_id": None,
                        "status": "planned",
                        "output_file_id": None,
                        "error_file_id": None,
                        "request_counts": None,
                        "history": [{"at": utc_now(), "status": "planned"}],
                    }
                ],
            }
        )
    reference_cost = float(config["api"].get("documented_reference_cost_per_image_usd", 0.0))
    planning_cost = reference_cost
    cost_basis = "documented_reference"
    if parent_approval is not None:
        usage_report = json.loads(Path(parent_approval["usage_report"]).read_text(encoding="utf-8"))
        observed_cost = usage_report.get("actual_cost_per_completed_usd")
        if observed_cost is not None:
            planning_cost = float(observed_cost)
            cost_basis = "parent_account_observed"
    state = {
        "schema_version": 1,
        "local_batch_id": batch_id,
        "stage": stage,
        "status": "planned",
        "seed": seed,
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "api_request": config["api"],
        "plan_path": PLAN_NAME,
        "plan_sha256": sha256_file(run_dir / PLAN_NAME),
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "parent_batch_dir": str(approved_batch_dir.resolve()) if approved_batch_dir else None,
        "parent_approval_sha256": (
            sha256_file(approved_batch_dir / "approval.json") if parent_approval else None
        ),
        "target_count": len(records),
        "request_count": len(records),
        "reference_cost_per_request_usd": reference_cost,
        "reference_projected_cost_usd": round(reference_cost * len(records), 6),
        "planning_cost_per_request_usd": planning_cost,
        "planning_projected_cost_usd": round(planning_cost * len(records), 6),
        "planning_cost_basis": cost_basis,
        "items": {
            row["custom_id"]: {"status": "planned", "filename": row["filename"]}
            for row in records
        },
        "shards": shards,
    }
    _atomic_json(run_dir / STATE_NAME, state)
    return run_dir


def load_state(run_dir: Path) -> dict[str, Any]:
    state_path = run_dir / STATE_NAME
    if not state_path.exists():
        raise PipelineError(f"state not found: {state_path}")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    plan_path = run_dir / state["plan_path"]
    if not plan_path.exists() or sha256_file(plan_path) != state["plan_sha256"]:
        raise PipelineError("generation plan is missing or changed")
    config_path = Path(state["config_path"])
    if not config_path.exists() or sha256_file(config_path) != state["config_sha256"]:
        raise PipelineError("generation config is missing or changed")
    return state


def save_state(run_dir: Path, state: dict[str, Any]) -> None:
    attempts = [attempt for shard in state["shards"] for attempt in shard["attempts"]]
    statuses = [str(attempt.get("status")) for attempt in attempts]
    if state["items"] and all(item.get("status") == "success" for item in state["items"].values()):
        state["status"] = "collected"
    elif any(status in ACTIVE_STATUSES for status in statuses):
        state["status"] = next(status for status in statuses if status in ACTIVE_STATUSES)
    elif any(status == "planned" for status in statuses):
        state["status"] = "planned"
    elif statuses and all(status == "completed" for status in statuses):
        state["status"] = "completed"
    elif statuses and all(status in TERMINAL_STATUSES for status in statuses):
        state["status"] = "terminal_with_failures"
    state["updated_at"] = utc_now()
    _atomic_json(run_dir / STATE_NAME, state)


def read_plan(run_dir: Path, state: dict[str, Any] | None = None) -> dict[str, dict[str, Any]]:
    state = state or load_state(run_dir)
    return {row["custom_id"]: row for row in read_jsonl(run_dir / state["plan_path"])}


def _client() -> Any:
    if not os.environ.get("OPENAI_API_KEY"):
        raise PipelineError("OPENAI_API_KEY is not set")
    from openai import OpenAI

    return OpenAI()


def _sync_attempt(attempt: dict[str, Any], batch: Any) -> None:
    status = str(_get(batch, "status", "unknown"))
    attempt.update(
        {
            "batch_id": _get(batch, "id", attempt.get("batch_id")),
            "status": status,
            "output_file_id": _get(batch, "output_file_id"),
            "error_file_id": _get(batch, "error_file_id"),
            "request_counts": _jsonable(_get(batch, "request_counts")),
            "batch_errors": _jsonable(_get(batch, "errors")),
        }
    )
    history = attempt.setdefault("history", [])
    if not history or history[-1].get("status") != status:
        history.append({"at": utc_now(), "status": status})


def _find_remote_duplicate(client: Any, metadata: dict[str, str]) -> Any | None:
    try:
        page = client.batches.list(limit=100)
    except Exception:
        return None
    for batch in _get(page, "data", []) or []:
        remote = _get(batch, "metadata", {}) or {}
        if all(remote.get(key) == value for key, value in metadata.items()):
            return batch
    return None


def pending_request_count(state: dict[str, Any]) -> int:
    return sum(
        len(attempt["custom_ids"])
        for shard in state["shards"]
        for attempt in shard["attempts"]
        if attempt.get("status") == "planned" and not attempt.get("batch_id")
    )


def submit_pending(
    run_dir: Path,
    *,
    approved_request_count: int,
    spend_cap_usd: float,
    client: Any | None = None,
) -> list[str]:
    state = load_state(run_dir)
    pending = pending_request_count(state)
    if pending == 0:
        return []
    if approved_request_count != pending:
        raise PipelineError(
            f"explicit approved request count {approved_request_count} does not match pending {pending}"
        )
    projected = float(
        state.get("planning_cost_per_request_usd", state.get("reference_cost_per_request_usd", 0.0))
    ) * pending
    if spend_cap_usd <= 0 or projected > spend_cap_usd:
        raise PipelineError(
            f"{state.get('planning_cost_basis', 'documented_reference')} projection "
            f"${projected:.4f} exceeds spend cap ${spend_cap_usd:.4f}"
        )
    if state.get("parent_batch_dir"):
        required = {"pilot": "validation", "floor_120": "pilot", "uniform_200": "pilot"}[state["stage"]]
        parent = Path(state["parent_batch_dir"])
        _approved_parent(parent, required)
        if sha256_file(parent / "approval.json") != state.get("parent_approval_sha256"):
            raise PipelineError("parent approval changed after planning")
    client = client or _client()
    remote_ids: list[str] = []
    for shard in state["shards"]:
        for attempt in shard["attempts"]:
            if attempt.get("status") != "planned":
                continue
            input_path = run_dir / attempt["input_path"]
            if sha256_file(input_path) != attempt["input_sha256"]:
                raise PipelineError(f"Batch input changed: {input_path}")
            custom_ids = validate_batch_jsonl(input_path, state["api_request"])
            if custom_ids != attempt["custom_ids"]:
                raise PipelineError(f"Batch input IDs changed: {input_path}")
            metadata = {
                "local_batch_id": state["local_batch_id"][:64],
                "stage": state["stage"][:64],
                "shard": str(shard["index"]),
                "attempt": str(attempt["number"]),
                "input_sha256": attempt["input_sha256"],
            }
            duplicate = _find_remote_duplicate(client, metadata)
            if duplicate is not None:
                _sync_attempt(attempt, duplicate)
                save_state(run_dir, state)
                remote_ids.append(str(attempt["batch_id"]))
                continue
            with input_path.open("rb") as stream:
                uploaded = client.files.create(file=stream, purpose="batch")
            attempt["input_file_id"] = _get(uploaded, "id")
            attempt["history"].append({"at": utc_now(), "status": "input_uploaded"})
            save_state(run_dir, state)
            batch = client.batches.create(
                input_file_id=attempt["input_file_id"],
                endpoint=ENDPOINT,
                completion_window=COMPLETION_WINDOW,
                metadata=metadata,
            )
            _sync_attempt(attempt, batch)
            save_state(run_dir, state)
            remote_ids.append(str(attempt["batch_id"]))
    return remote_ids


def refresh_status(run_dir: Path, client: Any | None = None) -> dict[str, Any]:
    state = load_state(run_dir)
    remote_attempts = [
        (shard, attempt)
        for shard in state["shards"]
        for attempt in shard["attempts"]
        if attempt.get("batch_id")
    ]
    if remote_attempts:
        client = client or _client()
    batches: list[dict[str, Any]] = []
    for shard, attempt in remote_attempts:
        _sync_attempt(attempt, client.batches.retrieve(attempt["batch_id"]))
        batches.append(
            {
                "shard": shard["index"],
                "attempt": attempt["number"],
                "batch_id": attempt["batch_id"],
                "status": attempt["status"],
                "request_counts": attempt.get("request_counts"),
            }
        )
    save_state(run_dir, state)
    return {
        "stage": state["stage"],
        "status": state["status"],
        "batches": batches,
        "local_success": sum(item.get("status") == "success" for item in state["items"].values()),
        "pending_requests": pending_request_count(state),
        "total": len(state["items"]),
    }


def _download_file(client: Any, file_id: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    with client.files.with_streaming_response.content(file_id) as response:
        response.stream_to_file(temporary)
    temporary.replace(target)


def _valid_image(path: Path, expected_size: str) -> bool:
    if not path.exists():
        return False
    expected = tuple(int(value) for value in expected_size.split("x"))
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            return image.format == "JPEG" and image.size == expected
    except (OSError, UnidentifiedImageError):
        return False


def process_output_jsonl(
    path: Path,
    run_dir: Path,
    state: dict[str, Any],
    plan: dict[str, dict[str, Any]],
) -> tuple[set[str], list[dict[str, Any]]]:
    changed: set[str] = set()
    usage_rows: list[dict[str, Any]] = []
    for line_number, row in enumerate(read_jsonl(path), 1):
        custom_id = row.get("custom_id")
        try:
            if custom_id not in plan:
                raise PipelineError(f"unknown custom_id {custom_id!r}")
            response = row.get("response") or {}
            if response.get("status_code") != 200:
                raise PipelineError(f"response status {response.get('status_code')}")
            body = response.get("body") or {}
            data = body.get("data") or []
            if len(data) != 1 or not isinstance(data[0].get("b64_json"), str):
                raise PipelineError("response does not contain exactly one b64_json image")
            raw = base64.b64decode(data[0]["b64_json"], validate=True)
            target = run_dir / "images" / plan[custom_id]["filename"]
            temporary = target.with_suffix(".jpg.tmp")
            expected = tuple(int(value) for value in plan[custom_id]["size"].split("x"))
            with Image.open(io.BytesIO(raw)) as image:
                if image.format != "JPEG" or image.size != expected:
                    raise PipelineError("API image format or size mismatch")
            temporary.write_bytes(raw)
            if not _valid_image(temporary, plan[custom_id]["size"]):
                raise PipelineError("collected JPEG failed validation")
            temporary.replace(target)
            digest = sha256_file(target)
            state["items"][custom_id].update(
                {"status": "success", "sha256": digest, "collected_at": utc_now()}
            )
            changed.add(custom_id)
            usage_rows.append(
                {
                    "custom_id": custom_id,
                    "model": body.get("model"),
                    "usage": body.get("usage"),
                    "response_id": response.get("request_id"),
                }
            )
        except (PipelineError, OSError, ValueError, binascii.Error, UnidentifiedImageError) as exc:
            if custom_id in state["items"]:
                state["items"][custom_id].update({"status": "collect_error", "error": str(exc)})
            else:
                state.setdefault("collection_errors", []).append(
                    {"file": path.name, "line": line_number, "error": str(exc)}
                )
    return changed, usage_rows


def _process_error_jsonl(path: Path, state: dict[str, Any]) -> None:
    for row in read_jsonl(path):
        custom_id = row.get("custom_id")
        if custom_id in state["items"] and state["items"][custom_id].get("status") != "success":
            state["items"][custom_id].update({"status": "api_error", "api_error": row.get("error")})


def _hash_manifest(run_dir: Path, state: dict[str, Any]) -> None:
    seen: dict[str, str] = {}
    rows: list[dict[str, Any]] = []
    for custom_id, item in sorted(state["items"].items()):
        path = run_dir / "images" / item["filename"]
        if item.get("status") != "success" or not path.exists():
            continue
        digest = sha256_file(path)
        duplicate_of = seen.get(digest)
        seen.setdefault(digest, custom_id)
        item["sha256"] = digest
        item["duplicate_of"] = duplicate_of
        rows.append(
            {"custom_id": custom_id, "filename": path.name, "sha256": digest, "duplicate_of": duplicate_of}
        )
    write_jsonl(run_dir / "image_sha256.jsonl", rows)


def collect_results(run_dir: Path, client: Any | None = None) -> dict[str, Any]:
    state = load_state(run_dir)
    plan = read_plan(run_dir, state)
    client = client or _client()
    usage: dict[str, dict[str, Any]] = {}
    usage_path = run_dir / "usage.jsonl"
    if usage_path.exists():
        usage = {row["custom_id"]: row for row in read_jsonl(usage_path)}
    changed: set[str] = set()
    for shard in state["shards"]:
        for attempt in shard["attempts"]:
            if not attempt.get("batch_id"):
                continue
            _sync_attempt(attempt, client.batches.retrieve(attempt["batch_id"]))
            prefix = f"shard_{shard['index']:03d}_attempt_{attempt['number']:02d}"
            if attempt.get("output_file_id"):
                output = run_dir / f"{prefix}_output.jsonl"
                if not output.exists():
                    _download_file(client, attempt["output_file_id"], output)
                new_ids, new_usage = process_output_jsonl(output, run_dir, state, plan)
                changed.update(new_ids)
                usage.update({row["custom_id"]: row for row in new_usage})
                attempt["local_output_path"] = output.name
                attempt["local_output_sha256"] = sha256_file(output)
            if attempt.get("error_file_id"):
                error = run_dir / f"{prefix}_error.jsonl"
                if not error.exists():
                    _download_file(client, attempt["error_file_id"], error)
                _process_error_jsonl(error, state)
                attempt["local_error_path"] = error.name
                attempt["local_error_sha256"] = sha256_file(error)
            save_state(run_dir, state)
    write_jsonl(usage_path, (usage[key] for key in sorted(usage)))
    _hash_manifest(run_dir, state)
    save_state(run_dir, state)
    return {
        "stage": state["stage"],
        "changed": len(changed),
        "success": sum(item.get("status") == "success" for item in state["items"].values()),
        "total": len(state["items"]),
        "usage_records": len(usage),
    }


def prepare_resume(run_dir: Path) -> int:
    state = load_state(run_dir)
    plan = read_plan(run_dir, state)
    created = 0
    for shard in state["shards"]:
        missing = [
            custom_id
            for custom_id in shard["custom_ids"]
            if not _valid_image(run_dir / "images" / plan[custom_id]["filename"], plan[custom_id]["size"])
        ]
        if not missing:
            continue
        latest = shard["attempts"][-1]
        if latest.get("status") in ACTIVE_STATUSES:
            continue
        if latest.get("status") == "planned" and not latest.get("batch_id"):
            created += len(latest["custom_ids"])
            continue
        number = len(shard["attempts"])
        input_name = f"batch_input_{shard['index']:03d}_attempt_{number:02d}.jsonl"
        input_path = run_dir / input_name
        write_jsonl(input_path, (batch_request(plan[custom_id], state["api_request"]) for custom_id in missing))
        validate_batch_jsonl(input_path, state["api_request"])
        shard["attempts"].append(
            {
                "number": number,
                "input_path": input_name,
                "input_sha256": sha256_file(input_path),
                "custom_ids": missing,
                "input_file_id": None,
                "batch_id": None,
                "status": "planned",
                "output_file_id": None,
                "error_file_id": None,
                "request_counts": None,
                "history": [{"at": utc_now(), "status": "planned_retry"}],
            }
        )
        created += len(missing)
    save_state(run_dir, state)
    return created


def _sum_numeric_usage(rows: list[dict[str, Any]]) -> dict[str, float]:
    totals: dict[str, float] = {}

    def visit(prefix: str, value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                visit(f"{prefix}.{key}" if prefix else str(key), child)
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            totals[prefix] = totals.get(prefix, 0.0) + float(value)

    for row in rows:
        visit("", row.get("usage") or {})
    return totals


def build_usage_report(run_dir: Path, *, actual_cost_usd: float | None = None) -> dict[str, Any]:
    state = load_state(run_dir)
    usage_path = run_dir / "usage.jsonl"
    usage_rows = read_jsonl(usage_path) if usage_path.exists() else []
    success = sum(item.get("status") == "success" for item in state["items"].values())
    image_paths = [
        run_dir / "images" / item["filename"]
        for item in state["items"].values()
        if item.get("status") == "success"
    ]
    approved_path = run_dir / "approved_annotations.jsonl"
    approved = read_jsonl(approved_path) if approved_path.exists() else []
    pan_quality = sum(bool(row.get("pan_quality_pass")) for row in approved)
    high_angle = sum(bool(row.get("counts_toward_high_angle_quota")) for row in approved)
    eye_level = sum(
        bool(row.get("pan_quality_pass"))
        and row.get("camera_elevation_class") == "eye_level_or_low_angle"
        for row in approved
    )
    if actual_cost_usd is not None and actual_cost_usd <= 0:
        raise PipelineError("actual cost must be positive when supplied")
    models = sorted({str(row["model"]) for row in usage_rows if row.get("model")})
    total_bytes = sum(path.stat().st_size for path in image_paths if path.exists())
    report = {
        "stage": state["stage"],
        "local_batch_id": state["local_batch_id"],
        "requested_model": state["api_request"]["model"],
        "response_models": models,
        "requested": state["request_count"],
        "completed_images": success,
        "failed_or_missing": state["request_count"] - success,
        "usage_records": len(usage_rows),
        "usage_path": str(usage_path) if usage_path.exists() else None,
        "usage_sha256": sha256_file(usage_path) if usage_path.exists() else None,
        "usage_totals": _sum_numeric_usage(usage_rows),
        "image_bytes": total_bytes,
        "bytes_per_completed": total_bytes / success if success else None,
        "pan_quality_pass": pan_quality,
        "high_angle_qualified": high_angle,
        "retained_eye_level": eye_level,
        "actual_cost_usd": actual_cost_usd,
        "actual_cost_per_completed_usd": (
            actual_cost_usd / success if actual_cost_usd is not None and success else None
        ),
        "actual_cost_per_pan_quality_usd": (
            actual_cost_usd / pan_quality if actual_cost_usd is not None and pan_quality else None
        ),
        "actual_cost_per_high_angle_usd": (
            actual_cost_usd / high_angle if actual_cost_usd is not None and high_angle else None
        ),
        "documented_reference_cost_per_request_usd": state.get("reference_cost_per_request_usd"),
        "cost_basis": "account_observed" if actual_cost_usd is not None else "documented_reference_only",
        "created_at": utc_now(),
    }
    report_path = run_dir / "usage_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report
