"""Deterministic candidate selection and finalization for ``test_profiles``."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from ..data import read_manifest
from .generate import PipelineError, circular_error_deg, sha256_file, write_jsonl

PROFILE_COLUMNS = [
    "selection_rank",
    "selection_seed",
    "selection_source_manifest_sha256",
    "person_id",
    "image",
    "annotator1_id",
    "annotator1_deg",
    "annotator2_id",
    "annotator2_deg",
    "final_angle_deg",
    "adjudicated_by",
    "notes",
]


def _order_key(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def plan_profile_candidates(
    image_root: Path,
    existing_manifest: Path,
    output_csv: Path,
    *,
    candidate_count: int = 600,
    max_frames_per_person: int = 2,
    seed: int = 20260831,
) -> dict[str, Any]:
    used_people = {
        Path(row["image"]).parent.name
        for row in read_manifest(existing_manifest)
        if row.get("split") in {"train", "test", "val"}
    }
    people = sorted(
        [path for path in image_root.iterdir() if path.is_dir() and path.name not in used_people],
        key=lambda path: _order_key(seed, path.name),
    )
    candidates: list[dict[str, str]] = []
    source_digest = sha256_file(existing_manifest)
    for person in people:
        frames = sorted(person.glob("*.jpg"), key=lambda path: _order_key(seed, str(path)))
        for frame in frames[:max_frames_per_person]:
            candidates.append(
                {
                    "selection_rank": str(len(candidates) + 1),
                    "selection_seed": str(seed),
                    "selection_source_manifest_sha256": source_digest,
                    "person_id": person.name,
                    "image": str(frame.resolve()),
                    "annotator1_id": "",
                    "annotator1_deg": "",
                    "annotator2_id": "",
                    "annotator2_deg": "",
                    "final_angle_deg": "",
                    "adjudicated_by": "",
                    "notes": "",
                }
            )
            if len(candidates) >= candidate_count:
                break
        if len(candidates) >= candidate_count:
            break
    if len(candidates) < candidate_count:
        raise PipelineError(
            f"only {len(candidates)} train/test-disjoint candidate frames are available"
        )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=PROFILE_COLUMNS)
        writer.writeheader()
        writer.writerows(candidates)
    return {
        "candidates": len(candidates),
        "people": len({row["person_id"] for row in candidates}),
        "output": str(output_csv),
        "sha256": sha256_file(output_csv),
        "seed": seed,
    }


def _in_profile_range(angle: float) -> bool:
    value = float(angle) % 360.0
    return 60.0 <= value <= 120.0 or 240.0 <= value <= 300.0


def finalize_test_profiles(
    reviewed_csv: Path,
    existing_manifest: Path,
    output_jsonl: Path,
    protocol_output: Path,
    *,
    minimum: int = 200,
    maximum: int = 300,
    max_frames_per_person: int = 2,
) -> dict[str, Any]:
    existing_digest = sha256_file(existing_manifest)
    used_people = {
        Path(row["image"]).parent.name
        for row in read_manifest(existing_manifest)
        if row.get("split") in {"train", "test", "val"}
    }
    with reviewed_csv.open(encoding="utf-8", newline="") as stream:
        source = list(csv.DictReader(stream))
    rows: list[dict[str, Any]] = []
    people: Counter[str] = Counter()
    side_counts: Counter[str] = Counter()
    for line_number, row in enumerate(source, 2):
        if not row.get("final_angle_deg", "").strip():
            continue
        person = row["person_id"].strip()
        if row.get("selection_source_manifest_sha256") != existing_digest:
            raise PipelineError(f"selection provenance mismatch at line {line_number}")
        if person in used_people:
            raise PipelineError(f"identity leakage at {reviewed_csv}:{line_number}: {person}")
        try:
            first = float(row["annotator1_deg"]) % 360.0
            second = float(row["annotator2_deg"]) % 360.0
            final = float(row["final_angle_deg"]) % 360.0
        except ValueError as exc:
            raise PipelineError(f"invalid angle at {reviewed_csv}:{line_number}") from exc
        annotator1 = row.get("annotator1_id", "").strip()
        annotator2 = row.get("annotator2_id", "").strip()
        if not annotator1 or not annotator2 or annotator1 == annotator2:
            raise PipelineError(
                f"two distinct non-empty annotator IDs are required at line {line_number}"
            )
        if circular_error_deg(first, second) > 15.0 and not row.get("adjudicated_by", "").strip():
            raise PipelineError(f"unadjudicated >15-degree disagreement at line {line_number}")
        if not _in_profile_range(final):
            raise PipelineError(f"final angle outside profile ranges at line {line_number}: {final}")
        people[person] += 1
        if people[person] > max_frames_per_person:
            raise PipelineError(f"too many retained frames for person {person}")
        side = "60_120" if 60.0 <= final <= 120.0 else "240_300"
        side_counts[side] += 1
        rows.append(
            {
                "split": "test_profiles",
                "task": "angle_deg",
                "angle_deg": final,
                "image": str(Path(row["image"]).resolve()),
                "person_id": person,
                "source": "towncentre_manual_profile",
                "selection_rank": int(row["selection_rank"]),
                "selection_seed": int(row["selection_seed"]),
                "selection_source_manifest_sha256": row[
                    "selection_source_manifest_sha256"
                ],
                "annotator1_id": annotator1,
                "annotator1_deg": first,
                "annotator2_id": annotator2,
                "annotator2_deg": second,
                "adjudicated_by": row.get("adjudicated_by", "").strip() or None,
                "notes": row.get("notes", "").strip(),
            }
        )
    if not minimum <= len(rows) <= maximum:
        raise PipelineError(f"test_profiles count must be {minimum}..{maximum}, got {len(rows)}")
    if min(side_counts.values(), default=0) < int(0.4 * len(rows)):
        raise PipelineError(f"profile sides are not sufficiently balanced: {dict(side_counts)}")
    if len(people) < 150:
        raise PipelineError(f"test_profiles needs at least 150 people, got {len(people)}")
    write_jsonl(output_jsonl, rows)
    protocol = {
        "schema_version": 1,
        "split": "test_profiles",
        "records": len(rows),
        "people": len(people),
        "side_counts": dict(side_counts),
        "selection_source_sha256": sha256_file(reviewed_csv),
        "selection_population_manifest_sha256": existing_digest,
        "manifest_path": str(output_jsonl.resolve()),
        "manifest_sha256": sha256_file(output_jsonl),
        "identity_rule": "person IDs disjoint from existing train/test/val",
        "annotation_rule": "two independent labels; >15 degree disagreement adjudicated",
        "primary_metric": "paired person-cluster bootstrap mean circular absolute error difference",
        "bootstrap_resamples": 10000,
        "promotion_rule": "primary 95% CI wholly above zero",
    }
    protocol_output.parent.mkdir(parents=True, exist_ok=True)
    protocol_output.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {**protocol, "protocol_sha256": sha256_file(protocol_output)}
