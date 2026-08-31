"""Materialize accepted synthetic heads at the native TownCentre size distribution."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..data import read_manifest
from .generate import (
    DEIM_CROP_MARGIN,
    PipelineError,
    load_config,
    load_state,
    sector_centre,
    sha256_file,
    write_jsonl,
)
from .landmarks import crop_transform
from .qa import (
    DIRECT_ALL_QUALITY_LABEL_POLICY,
    QA_IMPLEMENTATION_VERSION,
    config_from_recorded_qa_policy,
)

DIRECT_ANNOTATIONS_NAME = "accepted_annotations.jsonl"
ELEVATION_CLASSES = {"high_angle_match", "eye_level_or_low_angle", "unresolved"}


def square_head_crop(image: np.ndarray, box: list[float], margin: float) -> np.ndarray:
    """Crop the DEIM head as a long-side square with fixed per-side margin."""
    height, width = image.shape[:2]
    try:
        _, crop_box = crop_transform(box, pad=margin)
    except ValueError as exc:
        raise PipelineError(str(exc)) from exc
    crop_side = crop_box[2] - crop_box[0]
    side_px = max(1, int(round(crop_side)))
    centre_x = (crop_box[0] + crop_box[2]) / 2.0
    centre_y = (crop_box[1] + crop_box[3]) / 2.0
    left = int(round(centre_x - side_px / 2.0))
    top = int(round(centre_y - side_px / 2.0))
    right, bottom = left + side_px, top + side_px
    if left < 0 or top < 0 or right > width or bottom > height:
        raise PipelineError("5% long-side square head crop extends outside image")
    crop = image[top:bottom, left:right]
    if crop.shape[0] != side_px or crop.shape[1] != side_px:
        raise PipelineError("square head crop has inconsistent dimensions")
    return crop


def _real_sizes(manifest: Path) -> list[tuple[int, int]]:
    records = [row for row in read_manifest(manifest) if row.get("split") == "train"]
    sizes: list[tuple[int, int]] = []
    root = manifest.parent
    for row in records:
        image_path = Path(row["image"])
        if not image_path.is_absolute():
            image_path = root / image_path
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is not None:
            sizes.append(tuple(map(int, image.shape[:2])))
    if not sizes:
        raise PipelineError(f"no readable training images in {manifest}")
    return sizes


def _real_training_paths(manifest: Path) -> list[Path]:
    paths: list[Path] = []
    for row in read_manifest(manifest):
        if row.get("split") != "train":
            continue
        path = Path(row["image"])
        if not path.is_absolute():
            path = manifest.parent / path
        if path.is_file():
            paths.append(path)
    return paths


def _summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "q05": float(np.quantile(array, 0.05)),
        "median": float(np.median(array)),
        "q95": float(np.quantile(array, 0.95)),
    }


def _domain_stats(paths: list[Path]) -> dict[str, Any]:
    values: dict[str, list[float]] = {
        "height_px": [],
        "width_px": [],
        "brightness_mean": [],
        "contrast_std": [],
        "laplacian_variance": [],
    }
    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        values["height_px"].append(float(image.shape[0]))
        values["width_px"].append(float(image.shape[1]))
        values["brightness_mean"].append(float(gray.mean()))
        values["contrast_std"].append(float(gray.std()))
        values["laplacian_variance"].append(
            float(cv2.Laplacian(gray, cv2.CV_64F).var())
        )
    if not values["height_px"]:
        raise PipelineError("cannot compute domain statistics without readable images")
    return {
        "count": len(values["height_px"]),
        **{key: _summary(value) for key, value in values.items()},
    }


def _comparison_sheet(
    real_paths: list[Path],
    synthetic_paths: list[Path],
    output: Path,
    *,
    resize_50: bool,
) -> None:
    rows = min(12, len(real_paths), len(synthetic_paths))
    if rows == 0:
        return
    cell = 96
    canvas = np.full(((rows + 1) * cell, 2 * cell, 3), 255, dtype=np.uint8)
    for column, label in enumerate(("TownCentre", "synthetic")):
        cv2.putText(
            canvas,
            label,
            (column * cell + 6, cell // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
    for row_index in range(rows):
        for column, path in enumerate(
            (real_paths[row_index], synthetic_paths[row_index])
        ):
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                continue
            if resize_50:
                image = cv2.resize(image, (50, 50), interpolation=cv2.INTER_LANCZOS4)
            elif image.shape[0] > cell - 8 or image.shape[1] > cell - 8:
                scale = min((cell - 8) / image.shape[0], (cell - 8) / image.shape[1])
                image = cv2.resize(
                    image,
                    (
                        max(1, round(image.shape[1] * scale)),
                        max(1, round(image.shape[0] * scale)),
                    ),
                    interpolation=cv2.INTER_AREA,
                )
            top = (row_index + 1) * cell + (cell - image.shape[0]) // 2
            left = column * cell + (cell - image.shape[1]) // 2
            canvas[top : top + image.shape[0], left : left + image.shape[1]] = image
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), canvas, [cv2.IMWRITE_JPEG_QUALITY, 92]):
        raise PipelineError(f"failed to write comparison sheet: {output}")


def _stable_rng(custom_id: str, seed: int) -> random.Random:
    digest = hashlib.sha256(f"{seed}:{custom_id}".encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def _atomic_combined_manifest(
    real_manifest: Path, synthetic_rows: list[dict[str, Any]], output: Path
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    original = real_manifest.read_bytes()
    with temporary.open("wb") as stream:
        stream.write(original)
        if original and not original.endswith(b"\n"):
            stream.write(b"\n")
        for row in synthetic_rows:
            adjusted = dict(row)
            adjusted["image"] = "../synthetic/" + str(row["image"])
            stream.write(
                (
                    json.dumps(adjusted, ensure_ascii=False, sort_keys=True) + "\n"
                ).encode()
            )
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(output)


def _direct_production_annotations(
    run_dir: Path, state: dict[str, Any], config: dict[str, Any]
) -> tuple[list[dict[str, Any]], Path]:
    """Normalize operator-promoted automatic QA into accepted annotations."""
    if (
        state.get("direct_production") is not True
        or state.get("approval_policy") != "operator_direct_no_human_review"
    ):
        raise PipelineError("human approval is required before materialization")
    qa_path = run_dir / "auto_qa.jsonl"
    report_path = run_dir / "qa_report.json"
    if not qa_path.exists() or not report_path.exists():
        raise PipelineError("automatic QA and its report are required")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (
        report.get("direct_production") is not True
        or report.get("qa_implementation_version") != QA_IMPLEMENTATION_VERSION
        or report.get("label_acceptance_policy_auto") != DIRECT_ALL_QUALITY_LABEL_POLICY
    ):
        raise PipelineError("direct-production labels were not operator-promoted")
    config_from_recorded_qa_policy(config, report)
    expected_model_hashes = {
        "detector_sha256": config["models"]["deimv2"]["sha256"],
        "pose_sha256": config["models"]["sixdrepnet360"]["sha256"],
        "landmark_sha256": config["models"]["hrffa_vitl_ibug68"]["sha256"],
    }
    for key, expected in expected_model_hashes.items():
        if report.get(key) != expected:
            raise PipelineError(f"QA report does not bind the configured {key}")
    calibration_path = Path(str(report.get("calibration_path", "")))
    if not calibration_path.is_file() or report.get(
        "calibration_sha256"
    ) != sha256_file(calibration_path):
        raise PipelineError("QA pitch calibration is unavailable or changed")

    source_rows = [
        json.loads(line)
        for line in qa_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    total = len(source_rows)
    promotion = report.get("operator_label_promotion")
    if (
        report.get("total") != total
        or report.get("quality_pass") != total
        or report.get("pan_quality_pass_auto") != total
        or not isinstance(promotion, dict)
        or promotion.get("policy") != DIRECT_ALL_QUALITY_LABEL_POLICY
        or promotion.get("total_accepted") != total
    ):
        raise PipelineError("direct-production QA report does not accept every image")

    accepted_rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in source_rows:
        custom_id = str(row.get("custom_id", ""))
        if not custom_id or custom_id in seen:
            raise PipelineError("automatic QA has a missing or duplicate custom_id")
        seen.add(custom_id)
        if (
            row.get("quality_gate_pass") is not True
            or row.get("pan_quality_pass_auto") is not True
            or row.get("label_acceptance_policy_auto")
            != DIRECT_ALL_QUALITY_LABEL_POLICY
        ):
            raise PipelineError(f"automatic QA row is not accepted: {custom_id}")
        angle = row.get("angle_deg_auto")
        confidence = row.get("label_confidence_auto")
        label_source = row.get("label_source_auto")
        elevation = row.get("camera_elevation_class_auto")
        if (
            not isinstance(angle, (int, float))
            or not math.isfinite(float(angle))
            or not isinstance(confidence, (int, float))
            or float(confidence) <= 0.0
            or not isinstance(label_source, str)
            or not label_source
            or elevation not in ELEVATION_CLASSES
        ):
            raise PipelineError(f"accepted label is incomplete: {custom_id}")
        accepted_rows.append(
            {
                **row,
                "pan_quality_pass": True,
                "angle_deg": float(angle) % 360.0,
                "label_source": label_source,
                "label_confidence": float(confidence),
                "camera_elevation_class": elevation,
                "counts_toward_high_angle_quota": bool(
                    row.get("counts_toward_high_angle_quota_auto")
                ),
                "annotation_acceptance_source": (
                    "direct_production_operator_promoted_auto_qa"
                ),
            }
        )
    accepted_path = run_dir / DIRECT_ANNOTATIONS_NAME
    write_jsonl(accepted_path, accepted_rows)
    return accepted_rows, accepted_path


def _materialization_annotations(
    run_dir: Path, state: dict[str, Any], config: dict[str, Any]
) -> tuple[list[dict[str, Any]], float, str, Path]:
    approval_path = run_dir / "approval.json"
    if not approval_path.exists():
        rows, annotations_path = _direct_production_annotations(run_dir, state, config)
        return (
            rows,
            DEIM_CROP_MARGIN,
            "direct_operator_promoted_auto_qa",
            annotations_path,
        )
    approval = json.loads(approval_path.read_text(encoding="utf-8"))
    if approval.get("approved") is not True:
        raise PipelineError("run is not approved")
    annotations_path = run_dir / approval["annotations_path"]
    if sha256_file(annotations_path) != approval["annotations_sha256"]:
        raise PipelineError("approved annotations changed after approval")
    if sha256_file(run_dir / approval["review_path"]) != approval["review_sha256"]:
        raise PipelineError("human review changed after approval")
    margin_value = approval.get("crop_margin")
    if not isinstance(margin_value, (int, float)):
        raise PipelineError("approval does not bind the fixed DEIM crop margin")
    margin = float(margin_value)
    if not np.isclose(margin, DEIM_CROP_MARGIN, rtol=0.0, atol=1e-12):
        raise PipelineError(
            f"approved DEIM crop margin must be fixed to {DEIM_CROP_MARGIN:.2f}"
        )
    rows = [
        json.loads(line)
        for line in annotations_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return rows, margin, "human_approved", annotations_path


def materialize_run(
    run_dir: Path,
    *,
    output_root: Path,
    anchor_manifest: Path,
    neighbour_manifest: Path,
    seed: int = 20260831,
) -> dict[str, Any]:
    state = load_state(run_dir)
    config = load_config(Path(state["config_path"]))
    rows, margin, annotation_source, annotations_path = _materialization_annotations(
        run_dir, state, config
    )
    sizes = _real_sizes(anchor_manifest)
    crops_dir = output_root / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)
    run_manifest_rows: list[dict[str, Any]] = []
    rejected: list[dict[str, str]] = []
    for row in rows:
        if not row.get("pan_quality_pass"):
            continue
        source = run_dir / "images" / row["filename"]
        image = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if image is None or not row.get("head_box_xyxy"):
            rejected.append(
                {"custom_id": row["custom_id"], "reason": "missing_image_or_head_box"}
            )
            continue
        if row.get("sha256") and row["sha256"] != sha256_file(source):
            raise PipelineError(f"QA-bound image changed: {row['custom_id']}")
        try:
            crop = square_head_crop(image, row["head_box_xyxy"], margin)
        except PipelineError as exc:
            rejected.append({"custom_id": row["custom_id"], "reason": str(exc)})
            continue
        rng = _stable_rng(row["custom_id"], seed)
        target_height, target_width = sizes[rng.randrange(len(sizes))]
        resized = cv2.resize(
            crop, (target_width, target_height), interpolation=cv2.INTER_AREA
        )
        output_name = Path(str(row["filename"])).name
        output_path = crops_dir / output_name
        if output_path.exists():
            raise PipelineError(
                f"refusing to overwrite existing materialized crop: {output_path}"
            )
        if not cv2.imwrite(
            str(output_path),
            resized,
            [cv2.IMWRITE_JPEG_QUALITY, int(config["storage"]["quality"])],
        ):
            raise PipelineError(f"failed to write crop: {output_path}")
        run_manifest_rows.append(
            {
                "split": "train",
                "task": "angle_deg",
                "angle_deg": float(row["angle_deg"]),
                "image": f"crops/{output_name}",
                "source": "synthetic",
                "custom_id": row["custom_id"],
                "abs_pan_bin": int(row["abs_pan_bin"]),
                "label_source": row["label_source"],
                "label_confidence": float(row["label_confidence"]),
                "camera_elevation_class": row["camera_elevation_class"],
                "counts_toward_high_angle_quota": bool(
                    row["counts_toward_high_angle_quota"]
                ),
                "generation_run": state["local_batch_id"],
                "sha256": sha256_file(output_path),
                "native_height": target_height,
                "native_width": target_width,
                "source_crop_rule": "deim_long_side_square_5pct_per_side",
                "annotation_acceptance_source": annotation_source,
            }
        )
    output_root.mkdir(parents=True, exist_ok=True)
    annotations_store = output_root / "annotations.jsonl"
    existing: dict[str, dict[str, Any]] = {}
    if annotations_store.exists():
        existing = {
            row["custom_id"]: row
            for row in (
                json.loads(line)
                for line in annotations_store.read_text(encoding="utf-8").splitlines()
            )
        }
    for row in run_manifest_rows:
        if row["custom_id"] in existing and existing[row["custom_id"]] != row:
            raise PipelineError(
                f"conflicting materialized annotation: {row['custom_id']}"
            )
        existing[row["custom_id"]] = row
    all_rows = [existing[key] for key in sorted(existing)]
    high_rows = [row for row in all_rows if row["counts_toward_high_angle_quota"]]
    write_jsonl(annotations_store, all_rows)
    all_manifest = output_root / "manifest.jsonl"
    high_manifest = output_root / "manifest_high_angle.jsonl"
    combined_high_manifest = neighbour_manifest.parent / "manifest_nb3_synthetic.jsonl"
    combined_all_manifest = (
        neighbour_manifest.parent / "manifest_nb3_synthetic_all_elevations.jsonl"
    )
    write_jsonl(all_manifest, all_rows)
    write_jsonl(high_manifest, high_rows)
    _atomic_combined_manifest(
        neighbour_manifest,
        high_rows,
        combined_high_manifest,
    )
    _atomic_combined_manifest(
        neighbour_manifest,
        all_rows,
        combined_all_manifest,
    )
    real_paths = _real_training_paths(anchor_manifest)
    synthetic_paths = [output_root / row["image"] for row in all_rows]
    domain_comparison = {
        "towncentre_train": _domain_stats(real_paths),
        "synthetic_all_elevations": _domain_stats(synthetic_paths),
    }
    comparison_path = output_root / "domain_comparison.json"
    comparison_path.write_text(
        json.dumps(domain_comparison, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _comparison_sheet(
        real_paths,
        synthetic_paths,
        output_root / "comparison_native.jpg",
        resize_50=False,
    )
    _comparison_sheet(
        real_paths,
        synthetic_paths,
        output_root / "comparison_resize50.jpg",
        resize_50=True,
    )
    report = {
        "run_id": state["local_batch_id"],
        "materialized_this_run": len(run_manifest_rows),
        "materialized_total": len(all_rows),
        "all_elevations_trainable_total": len(all_rows),
        "high_angle_total": len(high_rows),
        "elevation_counts": dict(
            Counter(row["camera_elevation_class"] for row in all_rows)
        ),
        "rejected": rejected,
        "crop_margin": margin,
        "crop_rule": "max(head_box_width,head_box_height)*(1+2*margin) square",
        "annotation_source": annotation_source,
        "source_annotations": str(annotations_path),
        "source_annotations_sha256": sha256_file(annotations_path),
        "all_elevations_manifest": str(all_manifest),
        "high_angle_manifest": str(high_manifest),
        "combined_all_elevations_manifest": str(combined_all_manifest),
        "combined_high_angle_manifest": str(combined_high_manifest),
        "anchor_manifest": str(anchor_manifest),
        "anchor_manifest_sha256": sha256_file(anchor_manifest),
        "neighbour_manifest": str(neighbour_manifest),
        "neighbour_manifest_sha256": sha256_file(neighbour_manifest),
        "domain_comparison": str(comparison_path),
        "comparison_native": str(output_root / "comparison_native.jpg"),
        "comparison_resize50": str(output_root / "comparison_resize50.jpg"),
    }
    (output_root / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def render_deim_crop_margin_sheet(
    run_dir: Path,
    *,
    output: Path,
    anchor_manifest: Path,
    limit: int = 19,
) -> Path:
    margins = (DEIM_CROP_MARGIN,)
    qa_path = run_dir / "auto_qa.jsonl"
    if not qa_path.exists():
        raise PipelineError("auto_qa.jsonl is required")
    rows = [
        row
        for row in (
            json.loads(line)
            for line in qa_path.read_text(encoding="utf-8").splitlines()
        )
        if row.get("head_box_xyxy")
    ][:limit]
    real_by_sector: dict[int, list[tuple[int, Path]]] = {}
    for anchor in read_manifest(anchor_manifest):
        if anchor.get("split") != "train" or anchor.get("angle_deg") is None:
            continue
        image_path = Path(anchor["image"])
        if not image_path.is_absolute():
            image_path = anchor_manifest.parent / image_path
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        sector = sector_centre(float(anchor["angle_deg"]))
        real_by_sector.setdefault(sector, []).append(
            (int(image.shape[0] * image.shape[1]), image_path)
        )
    for candidates in real_by_sector.values():
        candidates.sort(key=lambda value: (value[0], str(value[1])))
    if not real_by_sector:
        raise PipelineError(
            f"no readable TownCentre training anchors in {anchor_manifest}"
        )
    cell = 160
    columns = len(margins) + 1
    canvas = np.full(((len(rows) + 1) * cell, columns * cell, 3), 255, dtype=np.uint8)
    for row_index, row in enumerate(rows, 1):
        sector = sector_centre(float(row["intent_pan_deg"]))
        candidates = real_by_sector.get(sector) or [
            item for group in real_by_sector.values() for item in group
        ]
        quantile = (row_index - 1) / max(1, len(rows) - 1)
        real_path = candidates[round(quantile * (len(candidates) - 1))][1]
        real = cv2.imread(str(real_path), cv2.IMREAD_COLOR)
        if real is not None:
            real_thumb = cv2.resize(real, (cell, cell), interpolation=cv2.INTER_AREA)
            canvas[row_index * cell : (row_index + 1) * cell, 0:cell] = real_thumb
        image = cv2.imread(str(run_dir / "images" / row["filename"]), cv2.IMREAD_COLOR)
        if image is None:
            continue
        for column, margin in enumerate(margins):
            crop = square_head_crop(image, row["head_box_xyxy"], margin)
            thumb = cv2.resize(crop, (cell, cell), interpolation=cv2.INTER_AREA)
            left = (column + 1) * cell
            canvas[row_index * cell : (row_index + 1) * cell, left : left + cell] = (
                thumb
            )
    cv2.putText(
        canvas,
        "TownCentre real",
        (8, cell // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )
    for column, margin in enumerate(margins):
        left = (column + 1) * cell
        cv2.putText(
            canvas,
            f"margin={margin:.0%}",
            (left + 8, cell // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), canvas, [cv2.IMWRITE_JPEG_QUALITY, 92]):
        raise PipelineError(f"failed to write margin contact sheet: {output}")
    return output
