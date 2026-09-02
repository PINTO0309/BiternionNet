"""Materialize yawpose-rear synthetic runs as standalone 320x320 head crops."""

from __future__ import annotations

import json
import os
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import cv2

from .generate import (
    DEIM_CROP_MARGIN,
    PipelineError,
    load_config,
    load_state,
    read_jsonl,
    read_plan,
    sha256_file,
    write_jsonl,
)
from .materialize import square_head_crop
from .qa import (
    DIRECT_ALL_QUALITY_LABEL_POLICY,
    QA_IMPLEMENTATION_VERSION,
    config_from_recorded_qa_policy,
)


def _atomic_copy(source: Path, target: Path) -> None:
    temporary = target.with_suffix(target.suffix + ".tmp")
    shutil.copy2(source, temporary)
    temporary.replace(target)


def materialize_yawpose_run(run_dir: Path, *, output_root: Path) -> dict[str, Any]:
    """Create a self-contained s004 delivery with fixed 320x320 DEIM crops."""
    run_dir = run_dir.resolve()
    output_root = output_root.resolve()
    state = load_state(run_dir)
    config = load_config(Path(state["config_path"]))
    if state.get("label_convention") != "yawpose":
        raise PipelineError("yawpose materialization requires a yawpose run")
    target_count = int(state["target_count"])
    if len(state.get("items") or {}) != target_count or not all(
        item.get("status") == "success" for item in state["items"].values()
    ):
        raise PipelineError("yawpose materialization requires every source image")

    qa_path = run_dir / "auto_qa.jsonl"
    report_path = run_dir / "qa_report.json"
    if not qa_path.is_file() or not report_path.is_file():
        raise PipelineError("yawpose materialization requires automatic QA")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    promotion = report.get("operator_label_promotion") or {}
    if (
        report.get("qa_implementation_version") != QA_IMPLEMENTATION_VERSION
        or report.get("total") != target_count
        or report.get("quality_pass") != target_count
        or report.get("pan_quality_pass_auto") != target_count
        or report.get("label_acceptance_policy_auto")
        != DIRECT_ALL_QUALITY_LABEL_POLICY
        or promotion.get("total_accepted") != target_count
    ):
        raise PipelineError("yawpose materialization requires fully promoted QA")
    config_from_recorded_qa_policy(config, report)
    expected_hashes = {
        "detector_sha256": config["models"]["deimv2"]["sha256"],
        "pose_sha256": config["models"]["sixdrepnet360"]["sha256"],
        "landmark_sha256": config["models"]["hrffa_vitl_ibug68"]["sha256"],
    }
    for key, expected in expected_hashes.items():
        if report.get(key) != expected:
            raise PipelineError(f"yawpose QA does not bind configured {key}")

    plan = read_plan(run_dir, state)
    qa_rows = {
        row["custom_id"]: row
        for row in (
            json.loads(line)
            for line in qa_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }
    if len(qa_rows) != target_count or set(qa_rows) != set(plan):
        raise PipelineError("yawpose QA rows do not exactly match the generation plan")
    if output_root.exists():
        raise PipelineError(f"refusing to overwrite yawpose delivery: {output_root}")
    staging = output_root.parent / f".{output_root.name}.tmp"
    if staging.exists():
        raise PipelineError(f"stale yawpose materialization staging directory: {staging}")
    images_dir = staging / "images"
    images_dir.mkdir(parents=True)

    crop_rows: list[dict[str, Any]] = []
    image_hash_rows: list[dict[str, Any]] = []
    accepted_rows: list[dict[str, Any]] = []
    try:
        for custom_id, record in sorted(
            plan.items(), key=lambda item: int(item[1]["serial"])
        ):
            qa = qa_rows[custom_id]
            if (
                qa.get("quality_gate_pass") is not True
                or qa.get("pan_quality_pass_auto") is not True
                or not qa.get("head_box_xyxy")
            ):
                raise PipelineError(f"yawpose row is not accepted: {custom_id}")
            source = run_dir / "images" / record["filename"]
            if not source.is_file() or qa.get("sha256") != sha256_file(source):
                raise PipelineError(f"QA-bound yawpose source changed: {custom_id}")
            image = cv2.imread(str(source), cv2.IMREAD_COLOR)
            if image is None:
                raise PipelineError(f"cannot decode yawpose source: {custom_id}")
            crop = square_head_crop(image, qa["head_box_xyxy"], DEIM_CROP_MARGIN)
            interpolation = (
                cv2.INTER_AREA
                if crop.shape[0] >= 320 and crop.shape[1] >= 320
                else cv2.INTER_LANCZOS4
            )
            resized = cv2.resize(crop, (320, 320), interpolation=interpolation)
            output = images_dir / record["filename"]
            if not cv2.imwrite(
                str(output),
                resized,
                [cv2.IMWRITE_JPEG_QUALITY, int(config["storage"]["quality"])],
            ):
                raise PipelineError(f"failed to write yawpose crop: {output}")
            digest = sha256_file(output)
            crop_row = {
                "custom_id": custom_id,
                "filename": record["filename"],
                "image": f"images/{record['filename']}",
                "bin": record["bin"],
                "yaw_yawpose": int(record["yaw_yawpose"]),
                "pitch": int(record["pitch"]),
                "cam": int(record["cam"]),
                "roll": int(record["roll"]),
                "visible_side": record["visible_side"],
                "augmentation_type": record.get("augmentation_type"),
                "mask_description": record.get("mask_description"),
                "accessory_type": record.get("accessory_type"),
                "accessory_description": record.get("accessory_description"),
                "label_source": qa["label_source_auto"],
                "label_confidence": float(qa["label_confidence_auto"]),
                "source_run": str(run_dir),
                "source_filename": record["filename"],
                "source_sha256": qa["sha256"],
                "head_box_xyxy": qa["head_box_xyxy"],
                "head_square_crop_box_xyxy": qa["head_square_crop_box_xyxy"],
                "source_crop_rule": "deim_long_side_square_5pct_per_side",
                "crop_margin_per_side": DEIM_CROP_MARGIN,
                "output_width": 320,
                "output_height": 320,
                "sha256": digest,
            }
            crop_rows.append(crop_row)
            image_hash_rows.append(
                {
                    "custom_id": custom_id,
                    "filename": record["filename"],
                    "sha256": digest,
                    "duplicate_of": None,
                }
            )
            accepted_rows.append(
                {
                    **crop_row,
                    "annotation_acceptance_source": (
                        "direct_production_operator_promoted_auto_qa"
                    ),
                }
            )

        hashes: dict[str, str] = {}
        for row in image_hash_rows:
            row["duplicate_of"] = hashes.get(row["sha256"])
            hashes.setdefault(row["sha256"], row["custom_id"])
        if any(row["duplicate_of"] for row in image_hash_rows):
            raise PipelineError("materialized yawpose crops contain duplicates")

        _atomic_copy(run_dir / state["plan_path"], staging / "generation_plan.jsonl")
        _atomic_copy(qa_path, staging / "auto_qa.jsonl")
        _atomic_copy(report_path, staging / "qa_report.json")
        _atomic_copy(run_dir / "batch_state.json", staging / "batch_state.json")
        write_jsonl(staging / "crop_meta.jsonl", crop_rows)
        write_jsonl(staging / "image_sha256.jsonl", image_hash_rows)
        write_jsonl(staging / "accepted_annotations.jsonl", accepted_rows)
        delivery = {
            "schema_version": 1,
            "source_run": str(run_dir),
            "source_batch_state_sha256": sha256_file(run_dir / "batch_state.json"),
            "target_count": target_count,
            "materialized_count": len(crop_rows),
            "output_size": "320x320",
            "crop_margin_per_side": DEIM_CROP_MARGIN,
            "crop_rule": "max(head_box_width,head_box_height)*(1+2*margin) square",
            "yaw_bin_counts": dict(Counter(row["bin"] for row in crop_rows)),
            "yaw_integer_min": min(row["yaw_yawpose"] for row in crop_rows),
            "yaw_integer_max": max(row["yaw_yawpose"] for row in crop_rows),
            "mask_count": sum(
                row.get("augmentation_type") == "face_mask" for row in crop_rows
            ),
            "accessory_counts": dict(
                Counter(
                    row["accessory_type"]
                    for row in crop_rows
                    if row.get("accessory_type")
                )
            ),
            "duplicate_crops": 0,
        }
        report_target = staging / "delivery_report.json"
        temporary = report_target.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(delivery, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        with temporary.open("a", encoding="utf-8") as stream:
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(report_target)
        staging.replace(output_root)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {**delivery, "output_root": str(output_root)}


def materialize_yawpose_runs(
    run_dirs: list[Path], *, output_root: Path
) -> dict[str, Any]:
    """Materialize and merge fully promoted yawpose runs into one delivery."""
    if not run_dirs:
        raise PipelineError("at least one yawpose source run is required")
    resolved_runs = [run_dir.resolve() for run_dir in run_dirs]
    if len(set(resolved_runs)) != len(resolved_runs):
        raise PipelineError("yawpose source runs must be unique")
    if len(resolved_runs) == 1:
        return materialize_yawpose_run(resolved_runs[0], output_root=output_root)

    output_root = output_root.resolve()
    if output_root.exists():
        raise PipelineError(f"refusing to overwrite yawpose delivery: {output_root}")
    staging = output_root.parent / f".{output_root.name}.tmp"
    if staging.exists():
        raise PipelineError(f"stale yawpose materialization staging directory: {staging}")
    staging.mkdir(parents=True)
    images_dir = staging / "images"
    images_dir.mkdir()

    plan_rows: list[dict[str, Any]] = []
    qa_rows: list[dict[str, Any]] = []
    crop_rows: list[dict[str, Any]] = []
    hash_rows: list[dict[str, Any]] = []
    accepted_rows: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    temporary_deliveries: list[Path] = []
    try:
        for index, run_dir in enumerate(resolved_runs):
            temporary_delivery = staging / f"source_{index:02d}"
            source_report = materialize_yawpose_run(
                run_dir, output_root=temporary_delivery
            )
            temporary_deliveries.append(temporary_delivery)
            source_state = load_state(run_dir)
            source_rows.append(
                {
                    "run_dir": str(run_dir),
                    "local_batch_id": source_state["local_batch_id"],
                    "target_count": int(source_state["target_count"]),
                    "batch_state_sha256": sha256_file(run_dir / "batch_state.json"),
                    "generation_plan_sha256": sha256_file(
                        temporary_delivery / "generation_plan.jsonl"
                    ),
                    "auto_qa_sha256": sha256_file(
                        temporary_delivery / "auto_qa.jsonl"
                    ),
                    "qa_report_sha256": sha256_file(
                        temporary_delivery / "qa_report.json"
                    ),
                    "delivery_report_sha256": sha256_file(
                        temporary_delivery / "delivery_report.json"
                    ),
                    "materialized_count": int(source_report["materialized_count"]),
                }
            )
            plan_rows.extend(read_jsonl(temporary_delivery / "generation_plan.jsonl"))
            qa_rows.extend(read_jsonl(temporary_delivery / "auto_qa.jsonl"))
            crop_rows.extend(read_jsonl(temporary_delivery / "crop_meta.jsonl"))
            hash_rows.extend(read_jsonl(temporary_delivery / "image_sha256.jsonl"))
            accepted_rows.extend(
                read_jsonl(temporary_delivery / "accepted_annotations.jsonl")
            )
            for image_path in (temporary_delivery / "images").iterdir():
                target = images_dir / image_path.name
                if target.exists():
                    raise PipelineError(
                        f"duplicate yawpose output filename: {image_path.name}"
                    )
                image_path.replace(target)

        target_count = len(plan_rows)
        serials = [int(row["serial"]) for row in plan_rows]
        custom_ids = [str(row["custom_id"]) for row in plan_rows]
        filenames = [str(row["filename"]) for row in plan_rows]
        if sorted(serials) != list(range(1, target_count + 1)):
            raise PipelineError(
                "combined yawpose serials must be contiguous from 1 through target_count"
            )
        if len(set(custom_ids)) != target_count:
            raise PipelineError("combined yawpose custom IDs are not unique")
        if len(set(filenames)) != target_count:
            raise PipelineError("combined yawpose filenames are not unique")
        expected_ids = set(custom_ids)
        for name, rows in (
            ("automatic QA", qa_rows),
            ("crop metadata", crop_rows),
            ("image hashes", hash_rows),
            ("accepted annotations", accepted_rows),
        ):
            row_ids = [str(row["custom_id"]) for row in rows]
            if len(row_ids) != target_count or set(row_ids) != expected_ids:
                raise PipelineError(f"combined {name} does not exactly match the plan")
        crop_hashes = [str(row["sha256"]) for row in hash_rows]
        if len(set(crop_hashes)) != target_count or any(
            row.get("duplicate_of") for row in hash_rows
        ):
            raise PipelineError("combined yawpose crops contain duplicates")

        order = {custom_id: serial for custom_id, serial in zip(custom_ids, serials)}
        plan_rows.sort(key=lambda row: int(row["serial"]))
        qa_rows.sort(key=lambda row: order[str(row["custom_id"])])
        crop_rows.sort(key=lambda row: order[str(row["custom_id"])])
        hash_rows.sort(key=lambda row: order[str(row["custom_id"])])
        accepted_rows.sort(key=lambda row: order[str(row["custom_id"])])
        write_jsonl(staging / "generation_plan.jsonl", plan_rows)
        write_jsonl(staging / "auto_qa.jsonl", qa_rows)
        write_jsonl(staging / "crop_meta.jsonl", crop_rows)
        write_jsonl(staging / "image_sha256.jsonl", hash_rows)
        write_jsonl(staging / "accepted_annotations.jsonl", accepted_rows)

        combined_state = {
            "schema_version": 1,
            "local_batch_id": output_root.name,
            "status": "materialized",
            "label_convention": "yawpose",
            "target_count": target_count,
            "source_runs": source_rows,
        }
        combined_qa = {
            "schema_version": 1,
            "qa_implementation_version": QA_IMPLEMENTATION_VERSION,
            "total": target_count,
            "quality_pass": target_count,
            "pan_quality_pass_auto": target_count,
            "label_acceptance_policy_auto": DIRECT_ALL_QUALITY_LABEL_POLICY,
            "operator_label_promotion": {
                "policy": DIRECT_ALL_QUALITY_LABEL_POLICY,
                "total_accepted": target_count,
            },
            "source_runs": source_rows,
        }
        delivery = {
            "schema_version": 1,
            "source_runs": source_rows,
            "target_count": target_count,
            "materialized_count": len(crop_rows),
            "output_size": "320x320",
            "crop_margin_per_side": DEIM_CROP_MARGIN,
            "crop_rule": "max(head_box_width,head_box_height)*(1+2*margin) square",
            "yaw_bin_counts": dict(Counter(row["bin"] for row in crop_rows)),
            "yaw_integer_min": min(row["yaw_yawpose"] for row in crop_rows),
            "yaw_integer_max": max(row["yaw_yawpose"] for row in crop_rows),
            "mask_count": sum(
                row.get("augmentation_type") == "face_mask" for row in crop_rows
            ),
            "accessory_counts": dict(
                Counter(
                    row["accessory_type"]
                    for row in crop_rows
                    if row.get("accessory_type")
                )
            ),
            "duplicate_crops": 0,
        }
        for path, payload in (
            (staging / "batch_state.json", combined_state),
            (staging / "qa_report.json", combined_qa),
            (staging / "delivery_report.json", delivery),
        ):
            path.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        for temporary_delivery in temporary_deliveries:
            shutil.rmtree(temporary_delivery)
        staging.replace(output_root)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {**delivery, "output_root": str(output_root)}
