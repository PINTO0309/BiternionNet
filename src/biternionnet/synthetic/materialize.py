"""Materialize approved synthetic heads at the native TownCentre size distribution."""

from __future__ import annotations

import hashlib
import json
import os
import random
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..data import read_manifest
from .generate import PipelineError, load_config, load_state, sector_centre, sha256_file, write_jsonl


def expanded_crop(image: np.ndarray, box: list[float], margin: float) -> np.ndarray:
    height, width = image.shape[:2]
    x1, y1, x2, y2 = map(float, box)
    box_width, box_height = x2 - x1, y2 - y1
    left = max(0, int(round(x1 - box_width * margin)))
    right = min(width, int(round(x2 + box_width * margin)))
    top = max(0, int(round(y1 - box_height * margin)))
    bottom = min(height, int(round(y2 + box_height * margin)))
    if right <= left or bottom <= top:
        raise PipelineError("head crop is empty")
    return image[top:bottom, left:right]


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
        values["laplacian_variance"].append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
    if not values["height_px"]:
        raise PipelineError("cannot compute domain statistics without readable images")
    return {"count": len(values["height_px"]), **{key: _summary(value) for key, value in values.items()}}


def _comparison_sheet(
    real_paths: list[Path], synthetic_paths: list[Path], output: Path, *, resize_50: bool
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
        for column, path in enumerate((real_paths[row_index], synthetic_paths[row_index])):
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                continue
            if resize_50:
                image = cv2.resize(image, (50, 50), interpolation=cv2.INTER_LANCZOS4)
            elif image.shape[0] > cell - 8 or image.shape[1] > cell - 8:
                scale = min((cell - 8) / image.shape[0], (cell - 8) / image.shape[1])
                image = cv2.resize(
                    image,
                    (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale))),
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
            stream.write((json.dumps(adjusted, ensure_ascii=False, sort_keys=True) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(output)


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
    approval_path = run_dir / "approval.json"
    if not approval_path.exists():
        raise PipelineError("human approval is required before materialization")
    approval = json.loads(approval_path.read_text(encoding="utf-8"))
    if approval.get("approved") is not True:
        raise PipelineError("run is not approved")
    annotations_path = run_dir / approval["annotations_path"]
    if sha256_file(annotations_path) != approval["annotations_sha256"]:
        raise PipelineError("approved annotations changed after approval")
    if sha256_file(run_dir / approval["review_path"]) != approval["review_sha256"]:
        raise PipelineError("human review changed after approval")
    margin = float(approval["crop_margin"])
    rows = [json.loads(line) for line in annotations_path.read_text(encoding="utf-8").splitlines()]
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
            rejected.append({"custom_id": row["custom_id"], "reason": "missing_image_or_head_box"})
            continue
        try:
            crop = expanded_crop(image, row["head_box_xyxy"], margin)
        except PipelineError as exc:
            rejected.append({"custom_id": row["custom_id"], "reason": str(exc)})
            continue
        rng = _stable_rng(row["custom_id"], seed)
        target_height, target_width = sizes[rng.randrange(len(sizes))]
        resized = cv2.resize(crop, (target_width, target_height), interpolation=cv2.INTER_AREA)
        output_name = Path(str(row["filename"])).name
        output_path = crops_dir / output_name
        if output_path.exists():
            raise PipelineError(f"refusing to overwrite existing materialized crop: {output_path}")
        if not cv2.imwrite(
            str(output_path), resized, [cv2.IMWRITE_JPEG_QUALITY, int(config["storage"]["quality"])]
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
                "counts_toward_high_angle_quota": bool(row["counts_toward_high_angle_quota"]),
                "generation_run": state["local_batch_id"],
                "sha256": sha256_file(output_path),
                "native_height": target_height,
                "native_width": target_width,
            }
        )
    output_root.mkdir(parents=True, exist_ok=True)
    annotations_store = output_root / "annotations.jsonl"
    existing: dict[str, dict[str, Any]] = {}
    if annotations_store.exists():
        existing = {
            row["custom_id"]: row
            for row in (json.loads(line) for line in annotations_store.read_text(encoding="utf-8").splitlines())
        }
    for row in run_manifest_rows:
        if row["custom_id"] in existing and existing[row["custom_id"]] != row:
            raise PipelineError(f"conflicting materialized annotation: {row['custom_id']}")
        existing[row["custom_id"]] = row
    all_rows = [existing[key] for key in sorted(existing)]
    high_rows = [row for row in all_rows if row["counts_toward_high_angle_quota"]]
    write_jsonl(annotations_store, all_rows)
    write_jsonl(output_root / "manifest.jsonl", all_rows)
    write_jsonl(output_root / "manifest_high_angle.jsonl", high_rows)
    _atomic_combined_manifest(
        neighbour_manifest,
        high_rows,
        neighbour_manifest.parent / "manifest_nb3_synthetic.jsonl",
    )
    _atomic_combined_manifest(
        neighbour_manifest,
        all_rows,
        neighbour_manifest.parent / "manifest_nb3_synthetic_all_elevations.jsonl",
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
    _comparison_sheet(real_paths, synthetic_paths, output_root / "comparison_native.jpg", resize_50=False)
    _comparison_sheet(real_paths, synthetic_paths, output_root / "comparison_resize50.jpg", resize_50=True)
    report = {
        "run_id": state["local_batch_id"],
        "materialized_this_run": len(run_manifest_rows),
        "materialized_total": len(all_rows),
        "high_angle_total": len(high_rows),
        "elevation_counts": dict(Counter(row["camera_elevation_class"] for row in all_rows)),
        "rejected": rejected,
        "crop_margin": margin,
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


def render_crop_margin_candidates(
    run_dir: Path,
    *,
    output: Path,
    margins: list[float],
    anchor_manifest: Path,
    limit: int = 19,
) -> Path:
    qa_path = run_dir / "auto_qa.jsonl"
    if not qa_path.exists():
        raise PipelineError("auto_qa.jsonl is required")
    rows = [
        row
        for row in (json.loads(line) for line in qa_path.read_text(encoding="utf-8").splitlines())
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
        real_by_sector.setdefault(sector, []).append((int(image.shape[0] * image.shape[1]), image_path))
    for candidates in real_by_sector.values():
        candidates.sort(key=lambda value: (value[0], str(value[1])))
    if not real_by_sector:
        raise PipelineError(f"no readable TownCentre training anchors in {anchor_manifest}")
    cell = 160
    columns = len(margins) + 1
    canvas = np.full(((len(rows) + 1) * cell, columns * cell, 3), 255, dtype=np.uint8)
    for row_index, row in enumerate(rows, 1):
        sector = sector_centre(float(row["intent_pan_deg"]))
        candidates = real_by_sector.get(sector) or [item for group in real_by_sector.values() for item in group]
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
            crop = expanded_crop(image, row["head_box_xyxy"], margin)
            thumb = cv2.resize(crop, (cell, cell), interpolation=cv2.INTER_AREA)
            left = (column + 1) * cell
            canvas[row_index * cell : (row_index + 1) * cell, left : left + cell] = thumb
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
