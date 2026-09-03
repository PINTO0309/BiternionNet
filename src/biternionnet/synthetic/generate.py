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
import math
import os
import random
import shutil
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable

import yaml
from PIL import Image, UnidentifiedImageError

ENDPOINT = "/v1/images/generations"
EDIT_ENDPOINT = "/v1/images/edits"
QUALITY = "low"
BACKGROUND = "opaque"
OUTPUT_FORMAT = "jpeg"
OUTPUT_COMPRESSION = 92
N_IMAGES = 1
COMPLETION_WINDOW = "24h"
BATCH_IMAGE_MODEL = "gpt-image-2"
BATCH_ENQUEUED_TOKEN_LIMIT = 1_000_000
ALLOWED_SIZES = {"1024x1536", "1024x1024", "1536x1024"}
YAWPOSE_ALLOWED_BIN_STARTS = tuple(range(0, 360, 10))
DEIM_CROP_MARGIN = 0.05
PITCH_EDIT_REASONS = {
    "head_looks_up_at_camera",
    "pitch_unusable",
    "pitch_calibration_tail",
}
PITCH_REFERENCE_OBJECTS = (
    "a small matte yellow tennis ball",
    "a small plain red paper cup lying on its side",
    "a small matte blue beanbag",
    "a small flat orange pavement marker disc",
)
MASK_VARIANTS = (
    "a plain light-blue disposable surgical face mask",
    "a plain white disposable surgical face mask",
    "a plain pale-green disposable surgical face mask",
    "a plain black cloth face mask",
    "a plain navy cloth face mask",
    "a plain grey cloth face mask",
    "a plain white cup-shaped respirator mask without a valve",
    "a plain light-grey cup-shaped respirator mask without a valve",
)
ACCESSORY_VARIANTS = {
    "eyeglasses": (
        "realistic thin-framed clear prescription eyeglasses, correctly seated on the nose and ears",
        "realistic plain dark-framed clear eyeglasses, correctly aligned on the head",
        "realistic rimless clear prescription eyeglasses with natural transparent lenses",
        "realistic modest tortoiseshell clear eyeglasses, correctly worn",
    ),
    "sunglasses": (
        "realistic non-reflective dark sunglasses, correctly seated on the nose and both ears",
        "realistic thin-framed grey sunglasses with clearly tinted lenses, correctly worn",
        "realistic plain black sunglasses with modest frames and non-mirrored lenses",
        "realistic brown-tinted sunglasses with simple frames, correctly aligned on the face",
    ),
    "ear_piercing": (
        "a single small silver stud piercing on the ear exposed by this head orientation",
        "a single small gold stud piercing on the ear exposed by this head orientation",
        "a single small plain silver hoop earring on the ear exposed by this head orientation",
        "a single small plain gold hoop earring on the ear exposed by this head orientation",
    ),
    "earring": (
        "a single small silver stud earring on the ear exposed by this head orientation",
        "a single small gold stud earring on the ear exposed by this head orientation",
        "a single small plain silver hoop earring on the ear exposed by this head orientation",
        "a single small plain gold hoop earring on the ear exposed by this head orientation",
    ),
}
STATE_NAME = "batch_state.json"
PLAN_NAME = "generation_plan.jsonl"
STANDALONE_PROVENANCE_DIR = "standalone_provenance"
SUPERSEDED_UNSUBMITTED_DIR = "superseded_unsubmitted"
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
    return (
        value.get(key, default)
        if isinstance(value, dict)
        else getattr(value, key, default)
    )


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


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise PipelineError(
                    f"invalid JSONL at {path}:{line_number}: {exc}"
                ) from exc


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return list(iter_jsonl(path))


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
        raise PipelineError(
            "evaluation protocol needs 200..300 records and at least 150 people"
        )
    if int(protocol.get("bootstrap_resamples", 0)) < 10_000:
        raise PipelineError(
            "evaluation protocol needs at least 10000 bootstrap resamples"
        )
    if "95% CI" not in str(protocol.get("promotion_rule", "")):
        raise PipelineError("evaluation protocol promotion rule must use a 95% CI")
    manifest_digest = str(protocol.get("manifest_sha256", ""))
    if len(manifest_digest) != 64:
        raise PipelineError(
            "evaluation protocol must bind the test_profiles manifest SHA-256"
        )
    manifest_value = protocol.get("manifest_path")
    if not manifest_value:
        raise PipelineError(
            "evaluation protocol must record the test_profiles manifest path"
        )
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
        raise PipelineError(
            f"config api.allowed_sizes must equal {sorted(ALLOWED_SIZES)}"
        )
    storage = config.get("storage") or {}
    if storage.get("format") != "jpeg" or storage.get("quality") != OUTPUT_COMPRESSION:
        raise PipelineError("storage must be JPEG quality 92")
    models = config.get("models") or {}
    expected_models = {"deimv2", "sixdrepnet360", "hrffa_vitl_ibug68"}
    if set(models) != expected_models:
        raise PipelineError(f"models must define exactly {sorted(expected_models)}")
    for name, asset in models.items():
        if not all(
            isinstance(asset.get(key), str) and asset[key]
            for key in ("source", "target")
        ):
            raise PipelineError(f"models.{name} must define source and target")
        digest = asset.get("sha256")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)
        ):
            raise PipelineError(
                f"models.{name}.sha256 must be a lowercase SHA-256 digest"
            )
    generation = config.get("generation") or {}
    convention = generation.get("label_convention", "towncentre_pan")
    if convention not in {"towncentre_pan", "yawpose"}:
        raise PipelineError(
            "generation.label_convention must be towncentre_pan or yawpose"
        )
    targets = config.get("targets") or {}
    if convention == "yawpose":
        bins = list(map(int, targets.get("yaw_bins") or []))
        if (
            not bins
            or bins != sorted(set(bins))
            or any(value not in YAWPOSE_ALLOWED_BIN_STARTS for value in bins)
        ):
            raise PipelineError(
                "targets.yaw_bins must be unique sorted 10-degree starts in [0, 350]"
            )
    else:
        bins = list(targets.get("abs_pan_bins") or [])
        if bins != list(range(0, 181, 10)):
            raise PipelineError(
                "targets.abs_pan_bins must be 0..180 in 10-degree steps"
            )
        validation = config.get("validation") or []
        if len(validation) != 19 or [int(row["abs_pan"]) for row in validation] != bins:
            raise PipelineError(
                "validation must contain exactly one ordered record per absolute pan bin"
            )
    stages = config.get("stages") or {}
    for stage, stage_config in stages.items():
        count = int(stage_config["count"])
        shard_size = int(stage_config["shard_size"])
        if count <= 0 or not 1 <= shard_size <= 500:
            raise PipelineError(f"invalid stage size for {stage}")
        bin_counts = stage_config.get("bin_counts")
        if bin_counts is not None and (
            len(bin_counts) != len(bins) or sum(map(int, bin_counts)) != count
        ):
            raise PipelineError(f"invalid bin_counts for {stage}")
        mask_bin_counts = stage_config.get("mask_bin_counts")
        masks = [0] * len(bins)
        if mask_bin_counts is not None:
            masks = list(map(int, mask_bin_counts))
            counts = list(map(int, bin_counts or []))
            if (
                len(masks) != len(bins)
                or not counts
                or any(mask < 0 or mask > total for mask, total in zip(masks, counts))
            ):
                raise PipelineError(f"invalid mask_bin_counts for {stage}")
        accessory_bin_counts = stage_config.get("accessory_bin_counts")
        if accessory_bin_counts is not None:
            if not isinstance(accessory_bin_counts, dict) or not accessory_bin_counts:
                raise PipelineError(f"invalid accessory_bin_counts for {stage}")
            unknown = set(accessory_bin_counts) - set(ACCESSORY_VARIANTS)
            if unknown:
                raise PipelineError(
                    f"unsupported accessory types for {stage}: {sorted(unknown)}"
                )
            accessory_counts: list[list[int]] = []
            for accessory_type, values in accessory_bin_counts.items():
                per_bin = list(map(int, values))
                if len(per_bin) != len(bins) or any(value < 0 for value in per_bin):
                    raise PipelineError(
                        f"invalid accessory_bin_counts.{accessory_type} for {stage}"
                    )
                accessory_counts.append(per_bin)
            counts = list(map(int, bin_counts or []))
            if not counts or any(
                masks[index] + sum(values[index] for values in accessory_counts)
                > counts[index]
                for index in range(len(bins))
            ):
                raise PipelineError(
                    f"mask and accessory counts exceed bin_counts for {stage}"
                )
    elevation = generation.get("camera_elevation") or {}
    minimum = int(elevation.get("min", 0))
    maximum = int(elevation.get("max", 0))
    if minimum > maximum:
        raise PipelineError("generation.camera_elevation min must not exceed max")
    schedule = elevation.get("schedule", "random_integer")
    if schedule not in {"random_integer", "balanced_integer"}:
        raise PipelineError(
            "generation.camera_elevation.schedule must be random_integer or balanced_integer"
        )
    if generation.get("camera_regime", "high_angle") not in {
        "high_angle",
        "near_level",
        "yawpose_rear",
    }:
        raise PipelineError(
            "generation.camera_regime must be high_angle, near_level, or yawpose_rear"
        )
    if convention == "yawpose":
        if generation.get("camera_regime") != "yawpose_rear":
            raise PipelineError(
                "yawpose generation requires camera_regime=yawpose_rear"
            )
        zero_share = int(elevation.get("zero_share", 70))
        if not 0 <= zero_share <= 100:
            raise PipelineError("yawpose camera zero_share must be in [0, 100]")
    return config


def wrap180(angle: float) -> float:
    return (float(angle) + 180.0) % 360.0 - 180.0


def circular_error_deg(left: float, right: float) -> float:
    return abs(wrap180(float(left) - float(right)))


def sector_centre(pan_deg: float) -> int:
    return int(((float(pan_deg) % 360.0 + 22.5) // 45.0) % 8) * 45


def expected_direction(pan_deg: float) -> str:
    return DIR8_BY_CENTRE[sector_centre(pan_deg)]


def _pan_detail(signed_pan: int) -> str:
    """Spell out the continuous turn so image generation preserves sign and sector."""
    amount = abs(int(signed_pan))
    if amount == 0:
        return (
            "Keep the head exactly frontal and symmetric; do not turn it left or right."
        )
    image_side = "right" if signed_pan > 0 else "left"
    subject_side = "left" if signed_pan > 0 else "right"
    if amount <= 20:
        return (
            f"Turn the nose exactly {amount} degrees from frontal toward image-{image_side}, "
            f"showing slightly more of the subject's {subject_side} side. Keep the face mostly "
            "frontal; do not produce a three-quarter view, profile, or the opposite turn."
        )
    if amount <= 60:
        return (
            f"Turn the nose exactly {amount} degrees from frontal toward image-{image_side}, "
            f"showing the subject's {subject_side} three-quarter-front view. Keep facial features "
            "visible; do not produce a pure side profile, rear view, or the opposite turn."
        )
    if amount <= 90:
        return (
            f"Turn exactly {amount} degrees from frontal toward image-{image_side}, showing the "
            f"subject's {subject_side} near-profile view. Do not cross into a rear view or the "
            "opposite side."
        )
    if amount <= 110:
        past_profile = amount - 90
        return (
            f"Turn exactly {amount} degrees from frontal: {past_profile} degrees past the subject's "
            f"{subject_side} side profile toward the rear. Keep this close to profile, not a "
            "three-quarter rear or full-back view."
        )
    if amount <= 150:
        return (
            f"Turn exactly {amount} degrees from frontal through the subject's {subject_side} side, "
            "forming a three-quarter-rear view. Show the back of the skull plus only the intended "
            "side; do not produce a profile, full-back view, or the opposite side."
        )
    if amount < 180:
        short_of_back = 180 - amount
        return (
            f"Turn exactly {amount} degrees from frontal through the subject's {subject_side} side, "
            f"only {short_of_back} degrees short of a full-back view. The back of the head must "
            "dominate, with at most a narrow sliver of the intended side and none of the opposite side."
        )
    return "Show an exact full-back view centered on the rear of the skull; no face or side profile."


def _camera_detail(camera_elevation: int) -> str:
    """Describe signed camera elevation without introducing overhead bias."""
    angle = int(camera_elevation)
    if angle > 0:
        return (
            f"Place the camera exactly {angle} degrees above the subject's eye level, with its "
            f"optical axis angled downward by {angle} degrees. Use only a subtle elevated-camera "
            "cue; do not exaggerate this into a steep overhead or bird's-eye view."
        )
    if angle < 0:
        amount = abs(angle)
        return (
            f"Place the camera exactly {amount} degrees below the subject's eye level, with its "
            f"optical axis angled upward by {amount} degrees. Use only a subtle low-camera cue; "
            "do not exaggerate this into a dramatic worm's-eye view."
        )
    return (
        "Place the camera exactly at the subject's eye level with a horizontal optical axis. "
        "Do not use top-down, overhead, low-angle, or upward-looking camera cues."
    )


def _balanced_integer_schedule(
    minimum: int, maximum: int, total: int, rng: random.Random
) -> list[int]:
    """Allocate integer elevations almost uniformly, preserving sign symmetry when possible."""
    values = list(range(minimum, maximum + 1))
    if not values:
        raise PipelineError("camera elevation schedule is empty")
    quotient, remainder = divmod(total, len(values))
    counts = {value: quotient for value in values}
    if minimum == -maximum and 0 in counts:
        pairs = list(range(1, maximum + 1))
        rng.shuffle(pairs)
        if remainder % 2:
            counts[0] += 1
            remainder -= 1
        for magnitude in pairs[: remainder // 2]:
            counts[-magnitude] += 1
            counts[magnitude] += 1
    else:
        extras = values[:]
        rng.shuffle(extras)
        for value in extras[:remainder]:
            counts[value] += 1
    schedule = [value for value in values for _ in range(counts[value])]
    if len(schedule) != total:
        raise PipelineError("balanced camera schedule did not match the target count")
    rng.shuffle(schedule)
    return schedule


def _balanced_discrete_schedule(
    values: list[int], total: int, rng: random.Random
) -> list[int]:
    """Repeat discrete values as evenly as possible in deterministic random order."""
    if not values or total < 0:
        raise PipelineError("balanced discrete schedule has invalid inputs")
    quotient, remainder = divmod(total, len(values))
    extras = values[:]
    rng.shuffle(extras)
    counts = {value: quotient for value in values}
    for value in extras[:remainder]:
        counts[value] += 1
    schedule = [value for value in values for _ in range(counts[value])]
    rng.shuffle(schedule)
    return schedule


def _yawpose_camera_schedule(
    config: dict[str, Any], total: int, rng: random.Random
) -> list[int]:
    elevation = config["generation"]["camera_elevation"]
    minimum = int(elevation["min"])
    maximum = int(elevation["max"])
    if minimum >= 0 or maximum <= 0:
        raise PipelineError(
            "yawpose camera schedule requires negative and positive angles"
        )
    zero_count = round(total * int(elevation.get("zero_share", 70)) / 100)
    nonzero_values = [*range(minimum, 0), *range(1, maximum + 1)]
    schedule = [0] * zero_count + _balanced_discrete_schedule(
        nonzero_values, total - zero_count, rng
    )
    rng.shuffle(schedule)
    return schedule


def _yawpose_pitch_schedule(
    config: dict[str, Any], total: int, rng: random.Random
) -> list[int]:
    pitch = config["generation"]["head_pitch"]
    inner_values = list(range(int(pitch["min"]), int(pitch["max"]) + 1))
    outer_count = int(pitch.get("outer_count", 0))
    if not 0 <= outer_count <= total:
        raise PipelineError("yawpose pitch outer_count must be in [0, stage count]")
    schedule = _balanced_discrete_schedule(inner_values, total - outer_count, rng)
    if outer_count:
        outer_min = int(pitch.get("outer_min_abs", 11))
        outer_max = int(pitch.get("outer_max_abs", 25))
        if not 0 < outer_min <= outer_max:
            raise PipelineError("yawpose outer pitch bounds are invalid")
        outer_values = [
            *range(-outer_max, -outer_min + 1),
            *range(outer_min, outer_max + 1),
        ]
        schedule.extend(_balanced_discrete_schedule(outer_values, outer_count, rng))
        rng.shuffle(schedule)
    return schedule


def _yawpose_visible_side(yaw: int) -> tuple[str, str]:
    """Return the required side label and literal visual anchor from the s004 spec."""
    if 20 <= yaw < 80:
        return (
            "three_quarter_left",
            "A front three-quarter view facing screen-left: both eyes remain visible, the farther eye lies close to the facial outline, the nose bridge begins to overlap the cheek, and the ear on the turned-toward side is visible.",
        )
    if 90 <= yaw < 110:
        return (
            "profile_left",
            "A complete to shallow-rear profile facing screen-left: show one ear and the nose-tip silhouette; at most one eye is barely visible.",
        )
    if 110 <= yaw < 140:
        return (
            "left_ear",
            "A left-rear quarter view: show the left ear and only the left cheek and jaw outline; no eye, nose, or mouth may be visible.",
        )
    if 140 <= yaw < 175:
        return (
            "left_ear",
            "An almost pure rear view reached through the left side: the back of the skull dominates and only part of the left ear may remain visible; show no facial feature.",
        )
    if 175 <= yaw < 185:
        return (
            "none",
            "A pure centered back-of-head view: ears are absent or barely visible, with no face at all; use the nape and natural hair flow to establish anatomy.",
        )
    if 185 <= yaw < 220:
        return (
            "right_ear",
            "An almost pure rear view reached through the right side: the back of the skull dominates and only part of the right ear may remain visible; show no facial feature.",
        )
    if 220 <= yaw < 250:
        return (
            "right_ear",
            "A right-rear quarter view: show the right ear and only the right cheek and jaw outline; no eye, nose, or mouth may be visible.",
        )
    if 250 <= yaw < 270:
        return (
            "profile_right",
            "A shallow-rear to complete profile facing screen-right: show one ear and the nose-tip silhouette; at most one eye is barely visible.",
        )
    if 280 <= yaw < 340:
        return (
            "three_quarter_right",
            "A front three-quarter view facing screen-right: both eyes remain visible, the farther eye lies close to the facial outline, the nose bridge begins to overlap the cheek, and the ear on the turned-toward side is visible.",
        )
    raise PipelineError(f"yawpose angle has no configured visual anchor: {yaw}")


def _yawpose_prompt(record: dict[str, Any]) -> str:
    mask = (
        f"The subject correctly wears {record['mask_description']} over the nose, mouth, and chin, naturally foreshortened for this orientation. Only the portion physically visible from the requested yaw may appear; the mask silhouette may replace the bare nose or mouth outline, but must not rotate the head or reveal a forbidden facial feature."
        if record.get("mask_description")
        else "The subject wears no face mask."
    )
    accessory = (
        f"Required additional accessory: {record['accessory_description']}. Keep its naturally visible portion recognizable and physically plausible from the requested yaw. Never rotate the head, expose a forbidden eye, nose, or mouth, mirror the image, or invent transparent anatomy merely to display the accessory."
        if record.get("accessory_description")
        else ""
    )
    return " ".join(
        [
            "Create one photorealistic, unedited-looking natural surveillance or smartphone photograph of exactly one fictional adult who resembles no real person.",
            "Use the yawpose label convention: +90 degrees faces screen-left, +180 degrees is the centered back of the head, and +270 degrees faces screen-right.",
            f"Set yaw_yawpose to exactly {int(record['yaw_yawpose']):+d} degrees in bin {record['bin']}; do not mirror the image or swap left and right.",
            record["anchor"],
            f"{record['camera_detail']} Keep the person's own head pitch near {int(record['pitch']):+d} degrees and image-plane roll at 0 degrees.",
            f"Subject: a fictional {record['gender']} aged {record['age']}, {record['skin_tone']}, {record['hair']}, wearing {record['clothing']}; headwear or accessory: {record['headwear']}.",
            accessory,
            mask,
            f"Natural context: {record['context']}. Background: {record['background']}; lighting: {record['lighting']}; camera feel: {record['lens_feel']}.",
            "Show the complete hair, crown, skull, chin when visible, neck, and parts of both shoulders. Keep the head about 30% to 42% of image height and leave generous clear margin on all four sides so a square head crop with 5% margin cannot leave the canvas.",
            "Keep the head, neck, shoulders, and visible upper torso anatomically coherent. Lower-body detail is unimportant. No other person, face, mannequin, poster face, human reflection, text, watermark, border, collage, illustration, CGI, beauty filter, or visible AI artifact.",
        ]
    )


def _refresh_yawpose_regeneration_prompt(
    record: dict[str, Any], config: dict[str, Any], edit_round: int
) -> dict[str, dict[str, str]]:
    """Replace every non-label scene attribute before a fresh regeneration."""
    prompt_config = config["prompt"]
    attribute_options = {
        "context": prompt_config["contexts"],
        "background": prompt_config["backgrounds"],
        "lighting": prompt_config["lighting"],
        "lens_feel": prompt_config["lens_feels"],
        "gender": prompt_config["gender_presentations"],
        "age": prompt_config["ages"],
        "skin_tone": prompt_config["skin_tones"],
        "hair": prompt_config["hair"],
        "clothing": prompt_config["clothing"],
    }
    seed_material = (
        f"{record['custom_id']}|fresh-yawpose-regeneration|{edit_round}"
    ).encode("utf-8")
    rng = random.Random(int.from_bytes(hashlib.sha256(seed_material).digest()[:8]))
    changes: dict[str, dict[str, str]] = {}
    for field, configured_options in attribute_options.items():
        previous = str(record[field])
        alternatives = [
            str(option) for option in configured_options if str(option) != previous
        ]
        if not alternatives:
            raise PipelineError(
                f"fresh yawpose regeneration requires an alternative {field} value"
            )
        replacement = rng.choice(alternatives)
        record[field] = replacement
        changes[field] = {"from": previous, "to": replacement}
    record["scene"] = record["background"]
    record["prompt"] = _yawpose_prompt(record)
    return changes


def _yawpose_filename(yaw: int, pitch: int, camera: int, serial: int) -> str:
    return f"yawp{yaw:+04d}_pitch{pitch:+03d}_cam{camera:+03d}_{serial:06d}.jpg"


def _build_yawpose_plan(
    config: dict[str, Any],
    stage: str,
    seed: int,
    *,
    bin_counts: list[int] | None,
    serial_offset: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    stage_config = config["stages"][stage]
    starts = list(map(int, config["targets"]["yaw_bins"]))
    counts = list(map(int, bin_counts or stage_config["bin_counts"]))
    masks = list(map(int, stage_config.get("mask_bin_counts") or [0] * len(starts)))
    accessory_bin_counts = {
        str(accessory_type): list(map(int, values))
        for accessory_type, values in (
            stage_config.get("accessory_bin_counts") or {}
        ).items()
    }
    if (
        accessory_bin_counts
        and bin_counts is not None
        and counts != list(map(int, stage_config["bin_counts"]))
    ):
        raise PipelineError(
            "bin_counts overrides are not supported for accessory yawpose stages"
        )
    assignments: list[dict[str, Any]] = []
    for bin_index, (start, count, mask_count) in enumerate(
        zip(starts, counts, masks, strict=True)
    ):
        yaw_values = _balanced_discrete_schedule(
            list(range(start, start + 10)), count, rng
        )
        augmentation_types: list[str | None] = ["face_mask"] * mask_count
        for accessory_type, values in accessory_bin_counts.items():
            augmentation_types.extend([accessory_type] * values[bin_index])
        augmentation_types.extend([None] * (count - len(augmentation_types)))
        rng.shuffle(augmentation_types)
        assignments.extend(
            {
                "yaw_bin_start": start,
                "yaw_yawpose": yaw,
                "native_mask": augmentation_type == "face_mask",
                "accessory_type": (
                    augmentation_type
                    if augmentation_type in ACCESSORY_VARIANTS
                    else None
                ),
            }
            for yaw, augmentation_type in zip(
                yaw_values, augmentation_types, strict=True
            )
        )
    rng.shuffle(assignments)
    sizes = _exact_schedule(
        config["generation"]["size_schedule"], len(assignments), rng
    )
    cameras = _yawpose_camera_schedule(config, len(assignments), rng)
    pitches = _yawpose_pitch_schedule(config, len(assignments), rng)
    prompt = config["prompt"]
    mask_serial = 0
    accessory_serials = {key: 0 for key in ACCESSORY_VARIANTS}
    records: list[dict[str, Any]] = []
    for index, assignment in enumerate(assignments):
        serial = serial_offset + index + 1
        yaw = int(assignment["yaw_yawpose"])
        signed_yaw = int(wrap180(yaw))
        pitch = pitches[index]
        camera = cameras[index]
        visible_side, anchor = _yawpose_visible_side(yaw)
        mask_description = None
        if assignment["native_mask"]:
            mask_description = MASK_VARIANTS[mask_serial % len(MASK_VARIANTS)]
            mask_serial += 1
        accessory_type = assignment["accessory_type"]
        accessory_description = None
        if accessory_type is not None:
            variants = ACCESSORY_VARIANTS[accessory_type]
            accessory_description = variants[
                accessory_serials[accessory_type] % len(variants)
            ]
            accessory_serials[accessory_type] += 1
        attribute_index = serial - 1
        filename = _yawpose_filename(yaw, pitch, camera, serial)
        record: dict[str, Any] = {
            "serial": serial,
            "stage": stage,
            "label_convention": "yawpose",
            "bin": f"yaw_{int(assignment['yaw_bin_start'])}_{int(assignment['yaw_bin_start']) + 10}",
            "yaw_yawpose": yaw,
            "yaw_signed": signed_yaw,
            "abs_pan_bin": abs(signed_yaw),
            "signed_pan": signed_yaw,
            "intent_pan_deg": float(yaw),
            "pitch": pitch,
            "head_pitch": pitch,
            "cam": camera,
            "camera_elevation": camera,
            "camera_regime": "yawpose_rear",
            "camera_detail": _camera_detail(camera),
            "roll": 0,
            "visible_side": visible_side,
            "anchor": anchor,
            "expected_direction": expected_direction((-yaw) % 360),
            "orientation": anchor,
            "pan_detail": anchor,
            "size": sizes[index],
            "context": prompt["contexts"][attribute_index % len(prompt["contexts"])],
            "background": prompt["backgrounds"][
                attribute_index % len(prompt["backgrounds"])
            ],
            "scene": prompt["backgrounds"][
                attribute_index % len(prompt["backgrounds"])
            ],
            "lighting": prompt["lighting"][attribute_index % len(prompt["lighting"])],
            "lens_feel": prompt["lens_feels"][
                attribute_index % len(prompt["lens_feels"])
            ],
            "gender": prompt["gender_presentations"][
                attribute_index % len(prompt["gender_presentations"])
            ],
            "age": prompt["ages"][attribute_index % len(prompt["ages"])],
            "skin_tone": prompt["skin_tones"][
                attribute_index % len(prompt["skin_tones"])
            ],
            "hair": prompt["hair"][attribute_index % len(prompt["hair"])],
            "clothing": prompt["clothing"][attribute_index % len(prompt["clothing"])],
            "headwear": prompt["headwear"][attribute_index % len(prompt["headwear"])],
            "accessory": prompt["headwear"][attribute_index % len(prompt["headwear"])],
            "custom_id": filename.removesuffix(".jpg"),
            "filename": filename,
        }
        if mask_description is not None:
            record["augmentation_type"] = "face_mask"
            record["mask_description"] = mask_description
        if accessory_description is not None:
            record["augmentation_type"] = "accessory"
            record["accessory_type"] = accessory_type
            record["accessory_description"] = accessory_description
        record["prompt"] = _yawpose_prompt(record)
        records.append(record)
    custom_ids = [row["custom_id"] for row in records]
    if len(custom_ids) != len(set(custom_ids)):
        raise PipelineError("yawpose planner produced duplicate custom IDs")
    return records


def _exact_schedule(
    config_rows: list[dict[str, Any]], total: int, rng: random.Random
) -> list[str]:
    result: list[str] = []
    allocated = 0
    for index, row in enumerate(config_rows):
        count = (
            total - allocated
            if index == len(config_rows) - 1
            else total * int(row["share"]) // 100
        )
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
    mask = (
        f"Face covering: the subject is correctly wearing {record['mask_description']} "
        "over the nose, mouth, and chin."
        if record.get("mask_description")
        else "Face covering: none; keep the nose and mouth naturally visible."
    )
    return " ".join(
        [
            prompt["preamble"],
            prompt["camera"].format(**record),
            prompt["pan"].format(**record),
            appearance,
            mask,
            prompt["framing"],
            prompt["realism"],
        ]
    )


def _make_compact_production_prompt(record: dict[str, Any]) -> str:
    """Keep the production controls while minimizing queued Batch text tokens."""
    mask = (
        f"Correctly wears {record['mask_description']} over nose, mouth, chin. "
        if record.get("mask_description")
        else "No face mask. "
    )
    near_level = record.get("camera_regime") == "near_level"
    opening = (
        "Photorealistic natural CCTV photo"
        if near_level
        else "Photorealistic overhead CCTV"
    )
    camera = (
        record["camera_detail"]
        if near_level
        else (
            f"Camera {record['camera_elevation']:+d}deg above, looking down; "
            "crown visible."
        )
    )
    neck = "upright natural neck" if near_level else "upright neck, never look up"
    return (
        f"{opening}: one fictional {record['age']} {record['gender']}; "
        f"{record['skin_tone']}, {record['hair']}, {record['clothing']}, {record['accessory']}. "
        f"{camera} "
        f"Head pan {record['signed_pan']:+d}deg ({record['expected_direction']}), "
        f"pitch {record['head_pitch']:+d}, roll 0; {neck}. "
        f"{mask}"
        "One uncropped head, neck, shoulders, upper torso; head 30-40% height, clear margins. "
        f"{record['scene']}; {record['lighting']}. "
        "No extra people or faces, legs, text, watermark, CGI, or anatomy defects."
    )


def _custom_id(
    signed_pan: int, camera_elevation: int, head_pitch: int, serial: int
) -> str:
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
    camera_regime: str,
    mask_description: str | None = None,
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
        "camera_regime": camera_regime,
        "camera_detail": _camera_detail(camera_elevation),
        "head_pitch": head_pitch,
        "roll": 0,
        "size": size,
        "expected_direction": DIR8_BY_CENTRE[centre],
        "orientation": ORIENTATION_BY_CENTRE[centre],
        "pan_detail": _pan_detail(signed_pan),
        "scene": prompt["scenes"][index % len(prompt["scenes"])],
        "lighting": prompt["lighting"][index % len(prompt["lighting"])],
        "age": prompt["ages"][index % len(prompt["ages"])],
        "gender": prompt["gender_presentations"][
            index % len(prompt["gender_presentations"])
        ],
        "skin_tone": prompt["skin_tones"][index % len(prompt["skin_tones"])],
        "hair": prompt["hair"][index % len(prompt["hair"])],
        "clothing": prompt["clothing"][index % len(prompt["clothing"])],
        "accessory": prompt["accessories"][index % len(prompt["accessories"])],
    }
    if mask_description is not None:
        record["augmentation_type"] = "face_mask"
        record["mask_description"] = mask_description
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
    if config["generation"].get("label_convention") == "yawpose":
        return _build_yawpose_plan(
            config,
            stage,
            seed,
            bin_counts=bin_counts,
            serial_offset=serial_offset,
        )
    rng = random.Random(seed)
    generation = config["generation"]
    stage_config = config["stages"][stage]
    if stage == "validation":
        if bin_counts is not None or serial_offset:
            raise PipelineError("validation does not support partial overrides")
        assignments = [dict(row) for row in config["validation"]]
    else:
        counts = list(map(int, bin_counts or stage_config["bin_counts"]))
        if len(counts) != len(config["targets"]["abs_pan_bins"]):
            raise PipelineError("bin_counts length must match configured target bins")
        mask_counts = list(map(int, stage_config.get("mask_bin_counts") or [0] * 19))
        assignments = []
        odd_sign_index = 0
        for abs_pan, count, mask_count in zip(
            config["targets"]["abs_pan_bins"], counts, mask_counts
        ):
            positive_first = True
            if stage_config.get("balance_pan_signs") and abs_pan not in {0, 180}:
                if count % 2:
                    positive_first = odd_sign_index % 2 == 0
                    odd_sign_index += 1
            for occurrence in range(count):
                assignment = {
                    "abs_pan": abs_pan,
                    "occurrence": occurrence,
                    "native_mask": occurrence < mask_count,
                }
                if stage_config.get("balance_pan_signs") and abs_pan not in {0, 180}:
                    positive = (occurrence % 2 == 0) == positive_first
                    assignment["signed_pan"] = abs_pan if positive else -abs_pan
                assignments.append(assignment)
        rng.shuffle(assignments)
    sizes = _exact_schedule(generation["size_schedule"], len(assignments), rng)
    elevation_config = generation["camera_elevation"]
    elevation_schedule = None
    if elevation_config.get("schedule", "random_integer") == "balanced_integer":
        elevation_schedule = _balanced_integer_schedule(
            int(elevation_config["min"]),
            int(elevation_config["max"]),
            len(assignments),
            rng,
        )
    records: list[dict[str, Any]] = []
    local_counts = {value: 0 for value in config["targets"]["abs_pan_bins"]}
    mask_serial = 0
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
        if elevation_schedule is not None and "camera_elevation" not in assignment:
            camera_elevation = elevation_schedule[index]
        else:
            sampled_camera_elevation = rng.randint(
                int(elevation_config["min"]), int(elevation_config["max"])
            )
            camera_elevation = int(
                assignment.get("camera_elevation", sampled_camera_elevation)
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
        mask_description = None
        if assignment.get("native_mask"):
            mask_description = MASK_VARIANTS[mask_serial % len(MASK_VARIANTS)]
            mask_serial += 1
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
                camera_regime=str(generation.get("camera_regime", "high_angle")),
                mask_description=mask_description,
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


def edit_batch_request(
    record: dict[str, Any], api: dict[str, Any], source: Path, prompt: str
) -> dict[str, Any]:
    if not _valid_image(source, record["size"]):
        raise PipelineError(f"edit source is not the expected JPEG: {source}")
    encoded = base64.b64encode(source.read_bytes()).decode("ascii")
    request = {
        "custom_id": record["custom_id"],
        "method": "POST",
        "url": EDIT_ENDPOINT,
        "body": {
            "model": api["model"],
            "images": [{"image_url": f"data:image/jpeg;base64,{encoded}"}],
            "prompt": prompt,
            "n": N_IMAGES,
            "size": record["size"],
            "quality": QUALITY,
            "background": BACKGROUND,
            "output_format": OUTPUT_FORMAT,
            "output_compression": OUTPUT_COMPRESSION,
        },
    }
    validate_batch_request(request, api, expected_endpoint=EDIT_ENDPOINT)
    return request


def validate_batch_request(
    request: dict[str, Any],
    api: dict[str, Any],
    *,
    expected_endpoint: str | None = None,
) -> None:
    endpoint = request.get("url")
    if request.get("method") != "POST" or endpoint not in {ENDPOINT, EDIT_ENDPOINT}:
        raise PipelineError(f"every request must POST to {ENDPOINT} or {EDIT_ENDPOINT}")
    if expected_endpoint is not None and endpoint != expected_endpoint:
        raise PipelineError(
            f"request endpoint {endpoint!r} does not match {expected_endpoint!r}"
        )
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
    expected_keys = {
        "model",
        "prompt",
        "n",
        "size",
        "quality",
        "background",
        "output_format",
        "output_compression",
    }
    if endpoint == EDIT_ENDPOINT:
        expected_keys.add("images")
        images = body.get("images")
        if (
            not isinstance(images, list)
            or len(images) != 1
            or not isinstance(images[0], dict)
        ):
            raise PipelineError(
                "image edit request must contain exactly one image reference"
            )
        reference = images[0]
        if set(reference) != {"image_url"}:
            raise PipelineError("image edit reference must contain only image_url")
        image_url = reference.get("image_url")
        if not isinstance(image_url, str) or not image_url.startswith(
            "data:image/jpeg;base64,"
        ):
            raise PipelineError("image edit reference must be a base64 JPEG data URL")
        if "input_fidelity" in body:
            raise PipelineError("gpt-image-2 edit requests must omit input_fidelity")
    if set(body) != expected_keys:
        raise PipelineError("unexpected or missing image request options")


def validate_batch_jsonl(
    path: Path, api: dict[str, Any], *, expected_endpoint: str | None = None
) -> list[str]:
    custom_ids: list[str] = []
    endpoints: set[str] = set()
    for row in read_jsonl(path):
        validate_batch_request(row, api, expected_endpoint=expected_endpoint)
        custom_ids.append(row["custom_id"])
        endpoints.add(str(row["url"]))
    if not custom_ids or len(custom_ids) != len(set(custom_ids)):
        raise PipelineError(f"empty or duplicate custom IDs in {path}")
    if len(endpoints) != 1:
        raise PipelineError("one Batch input file cannot mix endpoints")
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
    if not protocol.exists() or approval.get(
        "evaluation_protocol_sha256"
    ) != sha256_file(protocol):
        raise PipelineError("approved evaluation protocol is missing or changed")
    validate_evaluation_protocol(protocol)
    usage_value = approval.get("usage_report")
    if not usage_value:
        raise PipelineError("approved run does not bind a usage/cost report")
    usage_report = Path(usage_value)
    if not usage_report.exists() or approval.get("usage_report_sha256") != sha256_file(
        usage_report
    ):
        raise PipelineError("approved usage/cost report is missing or changed")
    sign_value = approval.get("sign_calibration_path")
    if approval.get("sign_calibration_approved") is not True or not sign_value:
        raise PipelineError("approved run does not bind sign calibration")
    sign_path = parent / str(sign_value)
    if not sign_path.exists() or approval.get("sign_calibration_sha256") != sha256_file(
        sign_path
    ):
        raise PipelineError("approved sign calibration is missing or changed")
    calibration_value = approval.get("pitch_calibration")
    if not calibration_value:
        raise PipelineError("approved run does not bind pitch calibration")
    calibration_path = Path(str(calibration_value))
    if not calibration_path.exists() or approval.get(
        "pitch_calibration_sha256"
    ) != sha256_file(calibration_path):
        raise PipelineError("approved pitch calibration is missing or changed")
    if required_stage == "pilot":
        rear_value = approval.get("rear_label_policy_path")
        if not rear_value:
            raise PipelineError("approved Pilot does not bind a rear label policy")
        rear_path = parent / str(rear_value)
        if not rear_path.exists() or approval.get(
            "rear_label_policy_sha256"
        ) != sha256_file(rear_path):
            raise PipelineError(
                "approved Pilot rear label policy is missing or changed"
            )
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
    serial_offset: int = 0,
    direct_production: bool = False,
    single_batch: bool = False,
    sequential_batches: bool = False,
    compact_prompts: bool = False,
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
    stage_config = config["stages"].get(stage) or {}
    direct_stage_allowed = stage == "uniform_200" or bool(
        stage_config.get("direct_production_allowed")
    )
    if direct_production and not direct_stage_allowed:
        raise PipelineError(f"direct production is not allowed for {stage}")
    if direct_production and approved_batch_dir is not None:
        raise PipelineError("direct production cannot also use an approved parent")
    if single_batch and sequential_batches:
        raise PipelineError("choose either --single-batch or --sequential-batches")
    if direct_production and not (single_batch or sequential_batches):
        raise PipelineError(
            "direct production requires --single-batch or --sequential-batches"
        )
    if compact_prompts and not direct_production:
        raise PipelineError("compact prompts require direct production")
    required_parent = (
        {
            "pilot": "validation",
            "floor_120": "pilot",
            "uniform_200": "pilot",
        }.get(stage)
        if not direct_production
        else None
    )
    parent_approval = None
    if required_parent:
        if approved_batch_dir is None:
            raise PipelineError(
                f"{stage} planning requires an approved {required_parent} directory"
            )
        parent_approval = _approved_parent(approved_batch_dir, required_parent)
        if (
            required_parent == "validation"
            and parent_approval.get("account_verified_snapshot")
            != config["api"]["model"]
        ):
            raise PipelineError(
                "Validation did not verify the configured snapshot for Pilot"
            )
        parent_usage = json.loads(
            Path(parent_approval["usage_report"]).read_text(encoding="utf-8")
        )
        if parent_usage.get("actual_cost_per_completed_usd") is None:
            raise PipelineError(
                f"{stage} planning requires the approved {required_parent} run's actual account cost"
            )
    if serial_offset < 0:
        raise PipelineError("serial offset must be non-negative")
    records = build_plan(
        config,
        stage,
        seed,
        bin_counts=bin_counts,
        serial_offset=serial_offset,
    )
    if compact_prompts:
        if config["generation"].get("label_convention") == "yawpose":
            raise PipelineError(
                "yawpose generation keeps its explicit left/right anchor prompts; do not compact"
            )
        for record in records:
            record["prompt"] = _make_compact_production_prompt(record)
    yawpose = config["generation"].get("label_convention") == "yawpose"
    for record in records:
        record["custom_id"] = f"{batch_id}--{record['custom_id']}"
        if not yawpose:
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
    shard_size = (
        len(records) if single_batch else int(config["stages"][stage]["shard_size"])
    )
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
                        "endpoint": ENDPOINT,
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
    reference_cost = float(
        config["api"].get("documented_reference_cost_per_image_usd", 0.0)
    )
    planning_cost = reference_cost
    cost_basis = "documented_reference"
    if parent_approval is not None:
        usage_report = json.loads(
            Path(parent_approval["usage_report"]).read_text(encoding="utf-8")
        )
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
        "parent_batch_dir": str(approved_batch_dir.resolve())
        if approved_batch_dir
        else None,
        "parent_approval_sha256": (
            sha256_file(approved_batch_dir / "approval.json")
            if parent_approval
            else None
        ),
        "target_count": len(records),
        "request_count": len(records),
        "serial_offset": serial_offset,
        "reference_cost_per_request_usd": reference_cost,
        "reference_projected_cost_usd": round(reference_cost * len(records), 6),
        "planning_cost_per_request_usd": planning_cost,
        "planning_projected_cost_usd": round(planning_cost * len(records), 6),
        "planning_cost_basis": cost_basis,
        "direct_production": direct_production,
        "single_batch": single_batch,
        "sequential_batches": sequential_batches,
        "prompt_profile": (
            "yawpose_rear_full_v1"
            if yawpose
            else ("compact_direct_v1" if compact_prompts else "full_v4")
        ),
        "label_convention": config["generation"].get(
            "label_convention", "towncentre_pan"
        ),
        "approval_policy": (
            "operator_direct_no_human_review"
            if direct_production
            else "staged_human_review"
        ),
        "intermediate_stages_waived": direct_production,
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
    if any(status in ACTIVE_STATUSES for status in statuses):
        state["status"] = next(
            status for status in statuses if status in ACTIVE_STATUSES
        )
    elif any(status == "planned" for status in statuses):
        state["status"] = "planned"
    elif state["items"] and all(
        item.get("status") == "success" for item in state["items"].values()
    ):
        state["status"] = "collected"
    elif statuses and all(status == "completed" for status in statuses):
        state["status"] = "completed"
    elif statuses and all(status in TERMINAL_STATUSES for status in statuses):
        state["status"] = "terminal_with_failures"
    state["updated_at"] = utc_now()
    _atomic_json(run_dir / STATE_NAME, state)


def seal_collected_prefix(run_dir: Path) -> dict[str, Any]:
    """Discard only an unsubmitted shard tail after preserving its original plan."""
    run_dir = run_dir.resolve()
    state = load_state(run_dir)
    if state.get("scope_revision"):
        raise PipelineError("run scope has already been revised")
    if any(
        attempt.get("status") in ACTIVE_STATUSES
        for shard in state["shards"]
        for attempt in shard["attempts"]
    ):
        raise PipelineError("cannot revise scope while a remote Batch is active")
    plan = read_plan(run_dir, state)
    kept_shards: list[dict[str, Any]] = []
    dropped_shards: list[dict[str, Any]] = []
    tail_started = False
    for shard in sorted(state["shards"], key=lambda value: int(value["index"])):
        ids = [str(value) for value in shard["custom_ids"]]
        all_success = all(
            state["items"].get(custom_id, {}).get("status") == "success"
            and _valid_image(
                run_dir / "images" / plan[custom_id]["filename"],
                plan[custom_id]["size"],
            )
            for custom_id in ids
        )
        if all_success:
            if tail_started:
                raise PipelineError("collected shards are not a contiguous prefix")
            kept_shards.append(shard)
            continue
        tail_started = True
        if any(
            attempt.get("batch_id") or attempt.get("status") != "planned"
            for attempt in shard["attempts"]
        ):
            raise PipelineError(
                "scope revision may discard only never-submitted planned shards"
            )
        if any(
            state["items"].get(custom_id, {}).get("status") == "success"
            for custom_id in ids
        ):
            raise PipelineError(
                "scope revision cannot discard a partially collected shard"
            )
        dropped_shards.append(shard)
    if not kept_shards or not dropped_shards:
        raise PipelineError(
            "scope revision requires collected prefix and unsubmitted tail"
        )

    archive = run_dir / SUPERSEDED_UNSUBMITTED_DIR
    if archive.exists():
        raise PipelineError(f"scope revision archive already exists: {archive}")
    archive.mkdir()
    shutil.copy2(run_dir / STATE_NAME, archive / "original_batch_state.json")
    shutil.copy2(run_dir / PLAN_NAME, archive / "original_generation_plan.jsonl")
    archived_inputs: list[dict[str, Any]] = []
    for shard in dropped_shards:
        for attempt in shard["attempts"]:
            source = run_dir / str(attempt["input_path"])
            if not source.is_file() or sha256_file(source) != attempt["input_sha256"]:
                raise PipelineError(
                    f"unsubmitted Batch input is missing or changed: {source}"
                )
            target = archive / source.name
            source.replace(target)
            archived_inputs.append(
                {
                    "shard": int(shard["index"]),
                    "input": target.name,
                    "sha256": sha256_file(target),
                    "requests": len(attempt["custom_ids"]),
                }
            )

    kept_ids = {
        str(custom_id) for shard in kept_shards for custom_id in shard["custom_ids"]
    }
    kept_plan = sorted(
        (plan[custom_id] for custom_id in kept_ids), key=lambda row: int(row["serial"])
    )
    original_target = int(state["target_count"])
    original_plan_sha256 = str(state["plan_sha256"])
    write_jsonl(run_dir / PLAN_NAME, kept_plan)
    state["items"] = {
        custom_id: state["items"][custom_id] for custom_id in sorted(kept_ids)
    }
    state["shards"] = kept_shards
    state["target_count"] = len(kept_ids)
    state["request_count"] = len(kept_ids)
    state["plan_sha256"] = sha256_file(run_dir / PLAN_NAME)
    state["reference_projected_cost_usd"] = round(
        float(state["reference_cost_per_request_usd"]) * len(kept_ids), 6
    )
    state["planning_projected_cost_usd"] = round(
        float(state["planning_cost_per_request_usd"]) * len(kept_ids), 6
    )
    revision_manifest = {
        "schema_version": 1,
        "created_at": utc_now(),
        "reason": "generation specification changed before the remaining shards were submitted",
        "original_target_count": original_target,
        "sealed_target_count": len(kept_ids),
        "discarded_unsubmitted_requests": original_target - len(kept_ids),
        "original_plan_sha256": original_plan_sha256,
        "sealed_plan_sha256": state["plan_sha256"],
        "original_state_snapshot": "original_batch_state.json",
        "original_plan_snapshot": "original_generation_plan.jsonl",
        "archived_inputs": archived_inputs,
    }
    _atomic_json(archive / "scope_revision.json", revision_manifest)
    state["scope_revision"] = {
        **revision_manifest,
        "archive": str(archive),
        "archive_manifest_sha256": sha256_file(archive / "scope_revision.json"),
    }
    _hash_manifest(run_dir, state)
    save_state(run_dir, state)
    return {
        "batch_dir": str(run_dir),
        "status": state["status"],
        "sealed_target_count": len(kept_ids),
        "discarded_unsubmitted_requests": original_target - len(kept_ids),
        "kept_shards": len(kept_shards),
        "archived_shards": len(dropped_shards),
        "archive": str(archive),
    }


def prepare_standalone_run(run_dir: Path) -> dict[str, Any]:
    """Detach an edit run from its parent after preserving immutable evidence."""
    run_dir = run_dir.resolve()
    state = load_state(run_dir)
    parent_dir = state.get("parent_batch_dir")
    if not parent_dir or not state.get("edit_round"):
        raise PipelineError("standalone preparation requires a parent-backed edit run")
    if state.get("standalone_conversion"):
        raise PipelineError("standalone conversion is already recorded")
    if any(
        attempt.get("status") in ACTIVE_STATUSES
        for shard in state["shards"]
        for attempt in shard["attempts"]
    ):
        raise PipelineError("cannot detach a run while a remote Batch is active")
    if len(state.get("items") or {}) != int(state["target_count"]) or not all(
        item.get("status") == "success" for item in state["items"].values()
    ):
        raise PipelineError(
            "standalone preparation requires every image locally collected"
        )

    required = ["auto_qa.jsonl", "qa_report.json", "accepted_annotations.jsonl"]
    for name in required:
        if not (run_dir / name).is_file():
            raise PipelineError(f"standalone preparation requires {name}")
    report = json.loads((run_dir / "qa_report.json").read_text(encoding="utf-8"))
    target_count = int(state["target_count"])
    promotion = report.get("operator_label_promotion") or {}
    if (
        report.get("total") != target_count
        or report.get("quality_pass") != target_count
        or promotion.get("total_accepted") != target_count
    ):
        raise PipelineError(
            "standalone preparation requires fully accepted automatic QA"
        )

    final_dir = run_dir / STANDALONE_PROVENANCE_DIR
    staging_dir = run_dir / f".{STANDALONE_PROVENANCE_DIR}.tmp"
    if final_dir.exists() or staging_dir.exists():
        raise PipelineError("standalone provenance directory already exists")
    staging_dir.mkdir()
    backup_names = [
        STATE_NAME,
        "auto_qa.jsonl",
        "qa_report.json",
        "accepted_annotations.jsonl",
    ]
    for optional in ("completion_audit.json", "usage_report.json"):
        if (run_dir / optional).is_file():
            backup_names.append(optional)
    backups: list[dict[str, Any]] = []
    for name in backup_names:
        source = run_dir / name
        target_name = f"parent_reuse_{name}"
        target = staging_dir / target_name
        shutil.copy2(source, target)
        backups.append(
            {
                "source": name,
                "snapshot": target_name,
                "sha256": sha256_file(target),
                "size_bytes": target.stat().st_size,
            }
        )

    token_evidence: list[dict[str, Any]] = []
    for index, (endpoint, plan) in enumerate(
        sorted((state.get("token_batch_plans") or {}).items())
    ):
        usage_value = plan.get("usage_path")
        if not usage_value:
            continue
        usage_path = Path(str(usage_value)).resolve()
        if not usage_path.is_file():
            raise PipelineError(f"token evidence is unavailable: {usage_path}")
        expected_sha256 = plan.get("usage_sha256")
        actual_sha256 = sha256_file(usage_path)
        if expected_sha256 and actual_sha256 != expected_sha256:
            raise PipelineError("token evidence changed after edit planning")
        target_name = f"token_evidence_{index:02d}_{usage_path.name}"
        target = staging_dir / target_name
        shutil.copy2(usage_path, target)
        original_evidence_run = plan.get("evidence_run")
        plan["evidence_run"] = str(final_dir)
        plan["usage_path"] = str(final_dir / target_name)
        token_evidence.append(
            {
                "endpoint": endpoint,
                "original_evidence_run": original_evidence_run,
                "original_usage_path": str(usage_path),
                "snapshot": target_name,
                "sha256": actual_sha256,
                "size_bytes": target.stat().st_size,
            }
        )

    parent_keys = (
        "parent_batch_dir",
        "parent_state_sha256",
        "parent_plan_sha256",
        "parent_qa_sha256",
        "parent_approval_sha256",
    )
    original_parent = {key: state.get(key) for key in parent_keys}
    provenance = {
        "schema_version": 1,
        "created_at": utc_now(),
        "run_id": state["local_batch_id"],
        "purpose": "preserve parent-reuse evidence before full standalone QA",
        "original_parent": original_parent,
        "backups": backups,
        "token_evidence": token_evidence,
    }
    _atomic_json(staging_dir / "provenance_manifest.json", provenance)
    staging_dir.replace(final_dir)
    provenance_path = final_dir / "provenance_manifest.json"

    for key in parent_keys:
        state[key] = None
    state["standalone_conversion"] = {
        "schema_version": 1,
        "status": "prepared_for_full_qa",
        "prepared_at": utc_now(),
        "provenance_manifest": str(provenance_path),
        "provenance_manifest_sha256": sha256_file(provenance_path),
        "previous_parent_batch_dir": str(parent_dir),
    }
    save_state(run_dir, state)
    return {
        "batch_dir": str(run_dir),
        "status": "prepared_for_full_qa",
        "target_count": target_count,
        "parent_detached": True,
        "provenance_manifest": str(provenance_path),
        "backups": len(backups),
        "token_evidence_snapshots": len(token_evidence),
    }


def finalize_standalone_run(
    run_dir: Path, *, allow_reused_passed_qa: bool = False
) -> dict[str, Any]:
    """Seal a detached run after full QA or verified passed-QA reuse."""
    run_dir = run_dir.resolve()
    state = load_state(run_dir)
    conversion = state.get("standalone_conversion")
    if not isinstance(conversion, dict) or conversion.get("status") != (
        "prepared_for_full_qa"
    ):
        raise PipelineError("standalone run is not awaiting full QA verification")
    if state.get("parent_batch_dir") is not None:
        raise PipelineError("standalone run still has an operational parent")
    provenance_path = Path(str(conversion.get("provenance_manifest", "")))
    if not provenance_path.is_file() or sha256_file(provenance_path) != conversion.get(
        "provenance_manifest_sha256"
    ):
        raise PipelineError("standalone provenance is unavailable or changed")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    for backup in provenance.get("backups") or []:
        snapshot = provenance_path.parent / str(backup["snapshot"])
        if not snapshot.is_file() or sha256_file(snapshot) != backup.get("sha256"):
            raise PipelineError(
                "standalone provenance backup is unavailable or changed"
            )
    qa_path = run_dir / "auto_qa.jsonl"
    report_path = run_dir / "qa_report.json"
    if not qa_path.is_file() or not report_path.is_file():
        raise PipelineError("full automatic QA is required before finalization")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    target_count = int(state["target_count"])
    reuse = report.get("qa_reuse") or {}
    promotion = report.get("operator_label_promotion") or {}
    fully_quality_passed = (
        report.get("total") == target_count
        and report.get("quality_pass") == target_count
        and promotion.get("total_accepted") == target_count
    )
    full_non_reused = fully_quality_passed and (
        reuse.get("compatibility") == "not_applicable"
        and reuse.get("reused_passed_records") == 0
        and reuse.get("evaluated_current_run_records") == target_count
    )
    reused_passed = fully_quality_passed and (
        report.get("pan_quality_pass_auto") == target_count
        and reuse.get("compatibility") == "matched"
        and int(reuse.get("reused_passed_records", -1))
        + int(reuse.get("evaluated_current_run_records", -1))
        == target_count
    )
    if not full_non_reused and not (allow_reused_passed_qa and reused_passed):
        raise PipelineError(
            "standalone finalization requires fully accepted, non-reused automatic QA; "
            "pass allow_reused_passed_qa only when previously passed records must not be re-evaluated"
        )
    if reused_passed:
        rows = read_jsonl(qa_path)
        plan = read_plan(run_dir, state)
        if len(rows) != target_count or {str(row["custom_id"]) for row in rows} != set(
            plan
        ):
            raise PipelineError("standalone automatic QA rows do not match the plan")
        for row in rows:
            if (
                row.get("quality_gate_pass") is not True
                or row.get("pan_quality_pass_auto") is not True
            ):
                raise PipelineError(
                    f"standalone QA row is not accepted: {row['custom_id']}"
                )
            image_path = run_dir / "images" / str(row["filename"])
            if not image_path.is_file() or row.get("sha256") != sha256_file(image_path):
                raise PipelineError(
                    f"standalone QA-bound image changed: {row['custom_id']}"
                )
        accepted_path = run_dir / "accepted_annotations.jsonl"
        if (
            not accepted_path.is_file()
            or sum(1 for line in accepted_path.open(encoding="utf-8") if line.strip())
            != target_count
        ):
            raise PipelineError("standalone accepted annotations are incomplete")
    elif (
        sum(1 for line in qa_path.open(encoding="utf-8") if line.strip())
        != target_count
    ):
        raise PipelineError("standalone automatic QA row count is incomplete")
    verification_mode = (
        "full_non_reused_qa" if full_non_reused else "hash_bound_passed_qa_reuse"
    )
    conversion.update(
        {
            "status": "verified_standalone",
            "verified_at": utc_now(),
            "qa_verification_mode": verification_mode,
            "auto_qa_sha256": sha256_file(qa_path),
            "qa_report_sha256": sha256_file(report_path),
            "evaluated_current_run_records": int(
                reuse["evaluated_current_run_records"]
            ),
            "reused_passed_records": int(reuse["reused_passed_records"]),
        }
    )
    save_state(run_dir, state)
    return {
        "batch_dir": str(run_dir),
        "status": conversion["status"],
        "target_count": target_count,
        "quality_pass": int(report["quality_pass"]),
        "qa_verification_mode": verification_mode,
        "evaluated_current_run_records": int(reuse["evaluated_current_run_records"]),
        "reused_passed_records": int(reuse["reused_passed_records"]),
        "provenance_manifest": str(provenance_path),
    }


def read_plan(
    run_dir: Path, state: dict[str, Any] | None = None
) -> dict[str, dict[str, Any]]:
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
    if state.get("direct_production") and not state.get("parent_batch_dir"):
        if state.get("direct_production") is not True:
            raise PipelineError(
                "production without a parent requires an explicit direct-production plan"
            )
        single = state.get("single_batch") is True and len(state["shards"]) == 1
        sequential = state.get("sequential_batches") is True
        if not (single or sequential):
            raise PipelineError(
                "direct production must remain single-batch or strictly sequential"
            )
        if sequential and any(
            len(shard.get("custom_ids") or []) > 500 for shard in state["shards"]
        ):
            raise PipelineError("sequential production Batch exceeds 500 requests")
    cost_per_request = state.get(
        "planning_cost_per_request_usd",
        state.get("reference_cost_per_request_usd", 0.0),
    )
    projected_decimal = Decimal(str(cost_per_request)) * pending
    spend_cap_decimal = Decimal(str(spend_cap_usd))
    projected = float(projected_decimal)
    if spend_cap_decimal <= 0 or projected_decimal > spend_cap_decimal:
        raise PipelineError(
            f"{state.get('planning_cost_basis', 'documented_reference')} projection "
            f"${projected:.4f} exceeds spend cap ${spend_cap_usd:.4f}"
        )
    if state.get("parent_batch_dir") and state.get("parent_approval_sha256"):
        required = {
            "pilot": "validation",
            "floor_120": "pilot",
            "uniform_200": "pilot",
        }[state["stage"]]
        parent = Path(state["parent_batch_dir"])
        _approved_parent(parent, required)
        if sha256_file(parent / "approval.json") != state.get("parent_approval_sha256"):
            raise PipelineError("parent approval changed after planning")
    serialized_token_batches = bool(
        state.get("token_batch_plans") or state.get("sequential_batches")
    )
    if serialized_token_batches and any(
        attempt.get("batch_id") and attempt.get("status") in ACTIVE_STATUSES
        for shard in state["shards"]
        for attempt in shard["attempts"]
    ):
        return []
    client = client or _client()
    remote_ids: list[str] = []
    planned_attempts = [
        (shard, attempt)
        for shard in state["shards"]
        for attempt in shard["attempts"]
        if attempt.get("status") == "planned" and not attempt.get("batch_id")
    ]
    if serialized_token_batches:
        planned_attempts = planned_attempts[:1]
    for shard, attempt in planned_attempts:
        input_path = run_dir / attempt["input_path"]
        if sha256_file(input_path) != attempt["input_sha256"]:
            raise PipelineError(f"Batch input changed: {input_path}")
        endpoint = str(attempt.get("endpoint", ENDPOINT))
        custom_ids = validate_batch_jsonl(
            input_path, state["api_request"], expected_endpoint=endpoint
        )
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
            endpoint=endpoint,
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
        "local_success": sum(
            item.get("status") == "success" for item in state["items"].values()
        ),
        "pending_requests": pending_request_count(state),
        "total": len(state["items"]),
    }


def advance_sequential_batches(
    run_dir: Path,
    *,
    spend_cap_usd: float,
    client: Any | None = None,
) -> dict[str, Any]:
    """Collect a terminal Batch and submit at most one next sequential Batch."""
    state = load_state(run_dir)
    if state.get("sequential_batches") is not True:
        raise PipelineError("advance-sequential requires a sequential-batches plan")

    status = refresh_status(run_dir, client=client)
    state = load_state(run_dir)
    active = [
        attempt
        for shard in state["shards"]
        for attempt in shard["attempts"]
        if attempt.get("batch_id") and attempt.get("status") in ACTIVE_STATUSES
    ]
    if active:
        return {
            "stage": status["stage"],
            "status": status["status"],
            "action": "waiting_for_active_batch",
            "active_batches": [
                {
                    "batch_id": str(attempt["batch_id"]),
                    "status": str(attempt["status"]),
                    "request_counts": attempt.get("request_counts"),
                }
                for attempt in active
            ],
            "local_success": status["local_success"],
            "pending_requests": status["pending_requests"],
            "total": status["total"],
            "submitted_batch_ids": [],
        }

    remote_exists = any(
        attempt.get("batch_id")
        for shard in state["shards"]
        for attempt in shard["attempts"]
    )
    collection = collect_results(run_dir, client=client) if remote_exists else None
    attempts_before = {
        str(attempt["input_path"])
        for shard in state["shards"]
        for attempt in shard["attempts"]
    }
    prepare_resume(run_dir)
    state = load_state(run_dir)
    retry_requests = sum(
        len(attempt["custom_ids"])
        for shard in state["shards"]
        for attempt in shard["attempts"]
        if str(attempt["input_path"]) not in attempts_before
    )
    local_success = sum(
        item.get("status") == "success" for item in state["items"].values()
    )
    if local_success == len(state["items"]):
        return {
            "stage": state["stage"],
            "status": state["status"],
            "action": "all_images_collected",
            "local_success": local_success,
            "total": len(state["items"]),
            "pending_requests": pending_request_count(state),
            "retry_requests": retry_requests,
            "collection": collection,
            "submitted_batch_ids": [],
        }

    pending = pending_request_count(state)
    if pending == 0:
        raise PipelineError(
            "sequential run is incomplete but has no active or planned requests"
        )
    submitted = submit_pending(
        run_dir,
        approved_request_count=pending,
        spend_cap_usd=spend_cap_usd,
        client=client,
    )
    return {
        "stage": state["stage"],
        "status": load_state(run_dir)["status"],
        "action": "submitted_next_batch" if submitted else "no_submission",
        "local_success": local_success,
        "total": len(state["items"]),
        "pending_requests_before_submission": pending,
        "pending_requests": pending_request_count(load_state(run_dir)),
        "retry_requests": retry_requests,
        "collection": collection,
        "submitted_batch_ids": submitted,
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


def _qa_edit_reasons(row: dict[str, Any], config: dict[str, Any]) -> list[str]:
    reasons = [
        str(value)
        for value in row.get("quality_gate_reasons") or []
        if str(value) != "direction_conflict"
    ]
    if (
        row.get("label_acceptance_policy_auto")
        == "direct_all_quality_pass_intent_fallback_v1"
        and row.get("quality_gate_pass") is True
    ):
        return reasons
    abs_pan = int(row["abs_pan_bin"])
    if abs_pan <= 90:
        if row.get("pose_status") != "ok":
            reasons.append("pose_unusable")
        elif float(row.get("pan_error_deg", 999.0)) > float(
            row.get("qa_pan_tolerance_deg", config["qa"]["pan_tolerance_deg"])
        ):
            reasons.append("pan_out_of_tolerance")
    # Match the empirically stable SixD pitch range used by QA calibration.
    # Around side profiles the Euler pitch folds toward 180 degrees.
    if abs_pan <= 60:
        if row.get("pose_status") != "ok":
            reasons.append("pitch_unusable")
        else:
            requested_camera = float(row["camera_elevation"])
            minimum_ratio = float(
                config["qa"]["elevation"]["minimum_negative_camera_ratio"]
            )
            if (
                float(row.get("sixd_pitch_deg", 999.0))
                > -minimum_ratio * requested_camera
            ):
                reasons.append("head_looks_up_at_camera")
    return list(dict.fromkeys(reasons))


def _pitch_calibration_tail_candidates(
    qa_rows: dict[str, dict[str, Any]],
    calibration: dict[str, Any],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    if calibration.get("valid") is not False:
        raise PipelineError("pitch calibration tail retry requires invalid calibration")
    data_threshold = float(calibration.get("data_threshold_deg", 0.0))
    hard_maximum = float(calibration.get("hard_maximum_deg", 0.0))
    if data_threshold <= hard_maximum:
        raise PipelineError(
            "pitch calibration tail retry requires a data-threshold hard-limit failure"
        )
    eligible = [
        row
        for row in qa_rows.values()
        if int(row["abs_pan_bin"]) <= 60
        and row.get("pose_status") == "ok"
        and row.get("quality_gate_pass")
    ]
    if len(eligible) != int(calibration.get("sample_count", -1)):
        raise PipelineError("pitch calibration no longer matches automatic QA rows")
    bias = float(calibration["bias_deg"])
    quantile = float(config["qa"]["elevation"]["quantile"])
    # Linear q90 for a small Validation set is controlled by the two largest
    # centred residuals. Retry at least those two records so the next
    # calibration is not dependent on one lucky edit.
    tail_count = min(len(eligible), max(2, math.ceil((1.0 - quantile) * len(eligible))))
    ranked = sorted(
        eligible,
        key=lambda row: (
            -abs(float(row["pitch_residual_deg"]) - bias),
            int(row["abs_pan_bin"]),
            str(row["custom_id"]),
        ),
    )
    return [
        {
            "custom_id": row["custom_id"],
            "filename": row["filename"],
            "abs_pan_bin": int(row["abs_pan_bin"]),
            "pitch_residual_deg": float(row["pitch_residual_deg"]),
            "centred_abs_residual_deg": round(
                abs(float(row["pitch_residual_deg"]) - bias), 6
            ),
        }
        for row in ranked[:tail_count]
    ]


def _pitch_reference_object_instruction(
    record: dict[str, Any], row: dict[str, Any]
) -> dict[str, Any]:
    """Build a deterministic second-stage physical gaze/pitch reference."""
    horizontal_position = "lower-nose-aligned"
    object_description = PITCH_REFERENCE_OBJECTS[
        (int(record["serial"]) - 1) % len(PITCH_REFERENCE_OBJECTS)
    ]
    requested_camera = float(record["camera_elevation"])
    observed_pitch = row.get("sixd_pitch_deg")
    try:
        correction = float(observed_pitch) - (-requested_camera)
    except (TypeError, ValueError):
        correction = 15.0
    if not math.isfinite(correction):
        correction = 15.0
    downward_deg = int(round(max(10.0, min(25.0, correction)) / 5.0) * 5)
    instruction = (
        "The previous direct pitch correction failed. Change only the head/neck pose and add exactly one "
        f"physical gaze-reference object: {object_description}. Place it on the pavement near the "
        "lower edge directly below the subject's nose at the same horizontal image coordinate as the nose. "
        "Do not place it farther left or right than the nose, because the object must change pitch without "
        "changing yaw. Keep it fully visible, smaller than 5% of image height, separated from the person, "
        "and outside the head/neck/shoulder target region. Have the subject naturally focus on that "
        f"object by rotating the entire head downward at the neck by about {downward_deg} degrees from the "
        "current pose, not merely moving the eyes. The nose and visible eye(s) must point toward the object "
        "while the requested left/right head pan remains unchanged. Show more crown and less underside of the "
        "chin; the subject must not look toward the elevated camera. This object-assisted instruction "
        "overrides only the original numeric head-pitch instruction. Preserve the camera position, camera "
        "angle, person position, framing, identity, background, and requested pan. The object must be plain, "
        "matte, non-reflective, and contain no text, logo, face, or person-like shape. Do not add another "
        "person, another object, a hand interaction, or an occlusion of the subject."
    )
    return {
        "description": object_description,
        "position": horizontal_position,
        "downward_correction_deg": downward_deg,
        "instruction": instruction,
    }


def _edit_prompt(
    record: dict[str, Any],
    row: dict[str, Any],
    reasons: list[str],
    *,
    pitch_reference: dict[str, Any] | None = None,
) -> str:
    corrections: list[str] = []
    if "head_too_small" in reasons:
        corrections.append(
            "Recompose closer so the complete detected head occupies 30% to 40% of image height; "
            "show only the upper torso, never the full body."
        )
    if "head_too_large" in reasons:
        corrections.append(
            "Recompose slightly wider so the complete detected head occupies 30% to 40% of image height."
        )
    if "head_crop_outside_image" in reasons:
        actual_size = str(row.get("actual_size") or record["size"])
        image_width, image_height = map(int, actual_size.split("x"))
        crop_box = row.get("head_square_crop_box_xyxy")
        if not isinstance(crop_box, list) or len(crop_box) != 4:
            raise PipelineError(
                "head_crop_outside_image edit requires the recorded square crop box"
            )
        crop_x1, crop_y1, crop_x2, crop_y2 = map(float, crop_box)
        crop_side = max(crop_x2 - crop_x1, crop_y2 - crop_y1)
        available_side = min(image_width, image_height)
        shift_x = max(0.0, -crop_x1) - max(0.0, crop_x2 - image_width)
        shift_y = max(0.0, -crop_y1) - max(0.0, crop_y2 - image_height)
        movements: list[str] = []
        if abs(shift_x) >= 0.01:
            x_percent = max(3, math.ceil(abs(shift_x) / image_width * 100.0) + 2)
            movements.append(
                f"{x_percent}% of the image width toward image-"
                f"{'right' if shift_x > 0 else 'left'}"
            )
        if abs(shift_y) >= 0.01:
            y_percent = max(3, math.ceil(abs(shift_y) / image_height * 100.0) + 2)
            movements.append(
                f"{y_percent}% of the image height "
                f"{'downward' if shift_y > 0 else 'upward'}"
            )
        movement = " and ".join(movements) or "slightly toward the image centre"
        scale_instruction = "Keep the person's proportions and head size unchanged."
        if crop_side > available_side:
            reduction_percent = max(
                3,
                math.ceil((crop_side / available_side - 1.0) * 100.0) + 2,
            )
            scale_instruction = (
                "This recorded square is wider than the canvas, so uniformly reduce the entire "
                f"person by approximately {reduction_percent}% before applying the positional "
                "shift; preserve all body proportions."
            )
        corrections.append(
            "Translate the entire person—head, neck, shoulders, and visible upper torso—together "
            f"by approximately {movement}. This must be a small positional shift within the same "
            "canvas, not a crop, zoom, head rotation, pose change, or camera change. "
            f"{scale_instruction} Naturally reconstruct the small "
            "background area exposed by the move. After the shift, a square crop centred on the "
            "detected head with side max(head width, head height) times 1.10 must stay fully inside "
            "the image, with a small safety clearance from every edge. This small whole-person "
            "translation explicitly overrides any generic target instruction not to translate; "
            "it does not override the target head pose, mask, identity, or scene requirements."
        )
    pan_reasons = {
        "pan_out_of_tolerance",
        "pose_unusable",
        "yawpose_out_of_tolerance",
        "yawpose_pose_unusable",
    }
    if pan_reasons.intersection(reasons):
        if row.get("pose_status") == "ok" and row.get("estimated_pan_deg") is not None:
            current_pan = float(row["estimated_pan_deg"])
            target_pan = float(record["intent_pan_deg"])
            relative_correction = wrap180(target_pan - current_pan)
            if abs(relative_correction) >= 1.0:
                positive_side = (
                    "image-left"
                    if record.get("label_convention") == "yawpose"
                    else "image-right"
                )
                negative_side = (
                    "image-right"
                    if record.get("label_convention") == "yawpose"
                    else "image-left"
                )
                correction_direction = (
                    f"toward {positive_side}"
                    if relative_correction > 0
                    else f"toward {negative_side}"
                )
                corrections.append(
                    f"The current image is estimated at pan {wrap180(current_pan):+.1f} degrees. "
                    f"From this current pose, rotate the entire head and nose about "
                    f"{abs(relative_correction):.1f} degrees {correction_direction}, back toward the "
                    f"target pan {wrap180(target_pan):+.1f} degrees. This is a relative correction "
                    "from the supplied image, not an instruction to turn farther in its current direction."
                )
        corrections.append(record["pan_detail"])
        if record.get("label_convention") == "yawpose":
            corrections.append(
                f"The final yaw_yawpose must be {int(record['yaw_yawpose']):+d} degrees under the convention +90=screen-left profile, +180=back, +270=screen-right profile. Do not mirror or reverse the named visible side."
            )
        else:
            corrections.append(
                f"The final head pan must be {int(record['signed_pan']):+d} degrees; do not mirror the "
                "image or reverse left and right."
            )
    if "rear_face_visible" in reasons:
        corrections.append(
            "The previous rear-view image exposed a face where none is allowed. Rotate only the head farther toward the back until no eye, nose, or mouth is visible, while preserving the requested yaw bin and the explicitly named left/right ear or pure-back anchor. Do not mirror the image."
        )
        corrections.append(record["anchor"])
    if any(reason in reasons for reason in PITCH_EDIT_REASONS):
        if pitch_reference is not None:
            corrections.append(str(pitch_reference["instruction"]))
        else:
            corrections.append(
                f"Keep the camera {int(record['camera_elevation']):+d} degrees above the subject, but "
                f"set the person's own head pitch to {int(record['head_pitch']):+d} degrees relative "
                "to the ground. Keep the neck upright, show the crown from above, and do not let the "
                "subject tilt the face upward toward the camera."
            )
    if any(reason in reasons for reason in ("head_not_detected", "head_count_not_one")):
        corrections.append(
            "Show exactly one complete, clearly detectable human head and no other person."
        )
        if int(row.get("head_count") or 0) > 1:
            corrections.append(
                "The detector found more than one head/person because the background contains "
                "person-shaped content. Keep only the main foreground subject. Remove every "
                "background human, face, head, mannequin, shop-window dummy, statue, poster, "
                "photograph, screen image, silhouette, and human reflection. Replace any such "
                "storefront display or reflective window with a plain opaque wall, closed shutter, "
                "or empty facade that has no human-like shapes. Do not duplicate, reflect, or "
                "partially repeat the foreground subject anywhere in the image."
            )
    if "body_not_detected" in reasons:
        corrections.append(
            "Keep the neck, both shoulders, and a coherent upper torso clearly visible."
        )
    if "duplicate_image" in reasons:
        corrections.append(
            "Keep the requested geometry but change non-label attributes such as clothing texture and background details."
        )
    if any(reason in reasons for reason in ("invalid_image", "wrong_dimensions")):
        corrections.append(f"Return one valid JPEG at exactly {record['size']} pixels.")
    if not corrections:
        corrections.append(
            "Correct the recorded QA failure while satisfying the full target description."
        )
    reason_text = ", ".join(reasons)
    return " ".join(
        [
            "Edit the supplied source image instead of creating an unrelated scene.",
            "Preserve the same fictional person's identity, age, hair, clothing, background, lighting, "
            "surveillance style, and camera location except where a correction below requires reframing.",
            "Preserve photorealism and anatomical integrity of the head, neck, shoulders, and visible upper "
            "torso. Any visible lower-body artifacts alone are acceptable and are outside QA scope; they "
            "must not distract from the requested correction. Do not add people, text, watermarks, borders, "
            "collages, reflections, or AI artifacts around the target region.",
            f"Recorded QA failures: {reason_text}.",
            *corrections,
            "All requirements from the target remain binding:",
            record["prompt"],
        ]
    )


def _regeneration_prompt(
    record: dict[str, Any], row: dict[str, Any], reasons: list[str]
) -> str:
    corrections: list[str] = []
    if "rear_face_visible" in reasons:
        corrections.append(
            "The failed candidate showed forbidden facial features. Render the head as a strict "
            "rear or rear-oblique view: no eye, eyebrow, nose bridge, nostril, cheek, lips, mouth, "
            "chin front, or facial reflection may be visible. Show only the back/side of the skull, "
            "hair, and the explicitly requested ear anchor. Do not solve this with a mask, hand, "
            "object, blur, darkness, or crop; use the correct physical head rotation."
        )
        if record.get("anchor"):
            corrections.append(str(record["anchor"]))
        if record.get("label_convention") == "yawpose":
            labelled_yaw = int(record["yaw_yawpose"]) % 360
            toward_back = wrap180(180.0 - labelled_yaw)
            rearward_shift = math.copysign(min(25.0, abs(toward_back)), toward_back)
            retry_yaw = int(round((labelled_yaw + rearward_shift) % 360))
            corrections.append(
                f"For this retry, make the physical head yaw visually {retry_yaw:+d} degrees, "
                f"which moves {abs(int(round(rearward_shift)))} degrees toward the full-back "
                f"anchor from the labelled yaw {labelled_yaw:+d}. This retry aim remains within "
                "the allowed 30-degree label tolerance and is mandatory so the rear skull "
                "surface dominates and the face disappears; keep the annotation label unchanged."
            )
    if any(
        reason
        in {
            "pan_out_of_tolerance",
            "pose_unusable",
            "yawpose_out_of_tolerance",
            "yawpose_pose_unusable",
        }
        for reason in reasons
    ):
        corrections.append(str(record["pan_detail"]))
        if record.get("label_convention") == "yawpose":
            labelled_yaw = int(record["yaw_yawpose"]) % 360
            signed_labelled_yaw = wrap180(labelled_yaw)
            if abs(signed_labelled_yaw) <= 45.0:
                corrections.append(
                    "Use an unmistakably near-front facial geometry: both eyes and both cheeks "
                    "must be visible, the nose bridge must stay near the facial centreline, and "
                    "the turn must remain subtle. Do not render a profile or rear-oblique pose."
                )
            elif 70.0 <= abs(signed_labelled_yaw) <= 110.0:
                screen_side = (
                    "screen-left" if signed_labelled_yaw > 0 else "screen-right"
                )
                corrections.append(
                    f"Use an unmistakable strict side profile facing {screen_side}: show no more "
                    "than one eye, keep the nose as a side silhouette, and do not expose the "
                    "front plane of the face or turn toward a three-quarter frontal view."
                )
            estimated_pan = row.get("estimated_pan_deg")
            if row.get("pose_status") == "ok" and isinstance(
                estimated_pan, (int, float)
            ):
                relative_correction = wrap180(labelled_yaw - float(estimated_pan))
                retry_shift = math.copysign(
                    min(25.0, abs(relative_correction)), relative_correction
                )
                retry_yaw = int(round((labelled_yaw + retry_shift) % 360))
                corrections.append(
                    f"The failed candidate was estimated at yaw_yawpose "
                    f"{float(estimated_pan):+.1f} degrees for the labelled target "
                    f"{labelled_yaw:+d}. Counter that observed generation bias by aiming the "
                    f"physical head visually at {retry_yaw:+d} degrees for this retry. This "
                    f"internal retry aim is {abs(int(round(retry_shift)))} degrees from the "
                    "label, remains within the allowed 30-degree tolerance, and overrides only "
                    "the exact rendered-yaw wording in the newly sampled target; keep the annotation "
                    f"label at {labelled_yaw:+d} degrees and never mirror the requested side."
                )
            else:
                corrections.append(
                    f"The head yaw must be {labelled_yaw:+d} degrees under yaw_yawpose "
                    "(+90=screen-left profile, +180=back, +270=screen-right profile); never "
                    "mirror the requested side."
                )
    if any(reason in reasons for reason in ("head_not_detected", "head_count_not_one")):
        corrections.append(
            "Show exactly one complete foreground human head. Include no other person, face, "
            "head, mannequin, statue, poster, screen portrait, silhouette, or human reflection."
        )
    if "head_crop_outside_image" in reasons:
        corrections.append(
            "Place the complete head safely near the image centre so a square centred on the "
            "detected head with side 1.10 times max(head width, head height) remains fully inside "
            "the canvas."
        )
    if any(reason in reasons for reason in PITCH_EDIT_REASONS):
        corrections.append(
            f"Keep camera elevation {int(record['camera_elevation']):+d} degrees and set the "
            f"person's own head pitch to {int(record['head_pitch']):+d} degrees relative to the "
            "ground without changing yaw."
        )
    if not corrections:
        corrections.append(
            "Correct every recorded QA failure while preserving the exact requested labels."
        )
    return " ".join(
        [
            "Generate a completely new independent image from the newly sampled target below; "
            "do not reproduce or edit the failed candidate. Do not reuse the failed candidate's "
            "person, age, skin appearance, hair, clothing, context, background, lighting, camera "
            "feel, or prompt wording. The following QA corrections are mandatory and override "
            "any conflicting visual tendency while preserving the requested yaw, pitch, camera, "
            "accessory, mask, and headwear labels.",
            f"Recorded QA failures: {', '.join(reasons)}.",
            *corrections,
            "Use this newly sampled complete target prompt:",
            str(record["prompt"]),
        ]
    )


def _chunk_batch_requests(
    requests: list[dict[str, Any]],
    *,
    max_records: int,
    max_bytes: int = 190 * 1024 * 1024,
) -> list[list[dict[str, Any]]]:
    chunks: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_bytes = 0
    for request in requests:
        request_bytes = len(json.dumps(request, ensure_ascii=False).encode("utf-8")) + 1
        if request_bytes > max_bytes:
            raise PipelineError(
                "one image edit request exceeds the safe Batch JSONL size"
            )
        if current and (
            len(current) >= max_records or current_bytes + request_bytes > max_bytes
        ):
            chunks.append(current)
            current = []
            current_bytes = 0
        current.append(request)
        current_bytes += request_bytes
    if current:
        chunks.append(current)
    return chunks


def _token_based_batch_plan(
    request_count: int,
    observed_mean_input_tokens: float,
    *,
    token_limit: int = BATCH_ENQUEUED_TOKEN_LIMIT,
) -> dict[str, Any]:
    if request_count <= 0:
        raise PipelineError("token-based Batch planning requires requests")
    if observed_mean_input_tokens <= 0:
        raise PipelineError("observed mean input tokens must be positive")
    if token_limit <= 1:
        raise PipelineError("Batch queued-token limit must exceed one")
    max_records = max(1, int((token_limit - 1) // observed_mean_input_tokens))
    batches = math.ceil(request_count / max_records)
    return {
        "request_count": request_count,
        "observed_mean_input_tokens": observed_mean_input_tokens,
        "expected_total_input_tokens": observed_mean_input_tokens * request_count,
        "queued_token_limit_exclusive": token_limit,
        "max_records_per_batch": max_records,
        "minimum_batch_count": batches,
    }


def _observed_input_token_profile(
    run_dir: Path, expected_endpoint: str
) -> dict[str, Any]:
    state = load_state(run_dir)
    endpoint_by_custom_id: dict[str, str] = {}
    for shard in state["shards"]:
        for attempt in shard["attempts"]:
            endpoint = str(attempt.get("endpoint", ENDPOINT))
            for custom_id in attempt.get("custom_ids") or []:
                endpoint_by_custom_id[str(custom_id)] = endpoint
    usage_path = run_dir / "usage.jsonl"
    if not usage_path.exists():
        raise PipelineError(f"token evidence has no usage.jsonl: {run_dir}")
    values = [
        int((row.get("usage") or {}).get("input_tokens") or 0)
        for row in read_jsonl(usage_path)
        if endpoint_by_custom_id.get(str(row.get("custom_id"))) == expected_endpoint
    ]
    values = [value for value in values if value > 0]
    if not values:
        raise PipelineError(
            f"token evidence has no observed {expected_endpoint} input tokens: {run_dir}"
        )
    return {
        "evidence_run": str(run_dir.resolve()),
        "usage_path": str(usage_path.resolve()),
        "usage_sha256": sha256_file(usage_path),
        "observed_records": len(values),
        "observed_mean_input_tokens": sum(values) / len(values),
        "observed_min_input_tokens": min(values),
        "observed_max_input_tokens": max(values),
    }


def _mask_augmentation_prompt(record: dict[str, Any], mask_description: str) -> str:
    """Build a mask-only edit prompt while binding the pose and scene invariants."""
    return " ".join(
        [
            "Edit the supplied source image; do not create an unrelated person or scene.",
            f"Add exactly {mask_description}, correctly worn over the nose, mouth, and chin.",
            "Make the mask visibly distinct from skin and hair, correctly foreshortened for the current "
            "head orientation, naturally fitted to the cheeks and nose bridge, with plausible ear loops "
            "or straps following the visible ear and side of the head.",
            "Change only the face mask and the tiny immediately occluded facial area. Preserve the same "
            "fictional identity, skull shape, head size, hair, hat or hood, eyeglasses, clothing, neck, "
            "shoulders, upper torso, background, lighting, shadows, camera position, framing, and image size.",
            f"Keep head pan {int(record['signed_pan']):+d} degrees, head pitch "
            f"{int(record['head_pitch']):+d} degrees, roll 0 degrees, and camera elevation "
            f"{int(record['camera_elevation']):+d} degrees unchanged. Do not mirror, rotate, translate, "
            "zoom, crop, or re-pose the person.",
            "The mask must be photorealistic and anatomically wearable. Do not add logos, text, valves, "
            "hands, people, faces, reflections, borders, watermarks, or other objects. Do not turn the "
            "mask into a beard, scarf, balaclava, costume mask, gas mask, or oxygen mask.",
            "All unedited pixels and scene content should remain as close to the source image as possible.",
        ]
    )


def _stratified_mask_selection(
    records: list[dict[str, Any]], target_count: int, seed: int
) -> list[dict[str, Any]]:
    """Select face-visible records proportionally across pan bin and direction sign."""
    eligible = [record for record in records if int(record["abs_pan_bin"]) <= 90]
    if target_count <= 0 or target_count > len(eligible):
        raise PipelineError(
            f"mask target {target_count} exceeds {len(eligible)} face-visible records"
        )
    groups: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for record in eligible:
        signed_pan = int(record["signed_pan"])
        sign = 0 if signed_pan == 0 else (1 if signed_pan > 0 else -1)
        groups.setdefault((int(record["abs_pan_bin"]), sign), []).append(record)
    allocations: dict[tuple[int, int], int] = {}
    remainders: list[tuple[float, tuple[int, int]]] = []
    for key, group in groups.items():
        exact = target_count * len(group) / len(eligible)
        allocations[key] = math.floor(exact)
        remainders.append((exact - allocations[key], key))
    remaining = target_count - sum(allocations.values())
    for _, key in sorted(remainders, key=lambda value: (-value[0], value[1]))[
        :remaining
    ]:
        allocations[key] += 1
    selected: list[dict[str, Any]] = []
    for key in sorted(groups):
        group = sorted(groups[key], key=lambda row: str(row["custom_id"]))
        random.Random(f"{seed}:{key[0]}:{key[1]}").shuffle(group)
        selected.extend(group[: allocations[key]])
    return sorted(selected, key=lambda row: str(row["custom_id"]))


def create_mask_augmentation(
    parent_run_dir: Path,
    batch_id: str,
    output_root: Path,
    *,
    target_fraction: float,
    planning_cost_per_request_usd: float,
    edit_token_evidence_run_dir: Path | None = None,
    seed: int = 20260831,
) -> Path:
    """Plan mask-only image edits so masks reach a fraction of the combined set."""
    if not batch_id or any(character in batch_id for character in "/\\"):
        raise PipelineError("batch-id must be one safe path component")
    if not 0.0 < target_fraction < 1.0:
        raise PipelineError("mask target fraction must be between zero and one")
    if planning_cost_per_request_usd <= 0:
        raise PipelineError("planning cost per edit request must be positive")
    parent_run_dir = parent_run_dir.resolve()
    parent_state = load_state(parent_run_dir)
    parent_plan = read_plan(parent_run_dir, parent_state)
    qa_path = parent_run_dir / "auto_qa.jsonl"
    report_path = parent_run_dir / "qa_report.json"
    if not qa_path.exists() or not report_path.exists():
        raise PipelineError("accepted automatic QA is required before mask planning")
    qa_rows = {row["custom_id"]: row for row in read_jsonl(qa_path)}
    if set(qa_rows) != set(parent_plan):
        raise PipelineError("parent QA rows do not exactly match its generation plan")
    if any(
        row.get("quality_gate_pass") is not True
        or row.get("pan_quality_pass_auto") is not True
        for row in qa_rows.values()
    ):
        raise PipelineError(
            "mask augmentation requires every parent image to be accepted"
        )
    qa_report = json.loads(report_path.read_text(encoding="utf-8"))
    if (
        qa_report.get("total") != len(parent_plan)
        or qa_report.get("quality_pass") != len(parent_plan)
        or qa_report.get("pan_quality_pass_auto") != len(parent_plan)
    ):
        raise PipelineError("parent QA report does not accept the complete dataset")

    config_path = Path(parent_state["config_path"])
    config = load_config(config_path)
    if config["api"]["model"] != BATCH_IMAGE_MODEL:
        raise PipelineError(
            f"Batch image editing requires api.model={BATCH_IMAGE_MODEL!r}"
        )
    base_count = len(parent_plan)
    target_count = math.ceil(base_count * target_fraction / (1.0 - target_fraction))
    selected = _stratified_mask_selection(
        [parent_plan[custom_id] for custom_id in sorted(parent_plan)],
        target_count,
        seed,
    )
    run_dir = output_root / "batches" / batch_id
    if run_dir.exists():
        raise PipelineError(f"refusing to overwrite existing run: {run_dir}")
    run_dir.mkdir(parents=True)
    (run_dir / "images").mkdir()
    (run_dir / "edit_inputs").mkdir()

    records: list[dict[str, Any]] = []
    items: dict[str, dict[str, Any]] = {}
    lineage: list[dict[str, Any]] = []
    requests: list[dict[str, Any]] = []
    for serial, parent_record in enumerate(selected, 1):
        parent_id = str(parent_record["custom_id"])
        source = parent_run_dir / "images" / parent_record["filename"]
        if not _valid_image(source, parent_record["size"]):
            raise PipelineError(f"mask edit source is unavailable: {source}")
        source_sha256 = sha256_file(source)
        record = dict(parent_record)
        record.update(
            {
                "serial": serial,
                "source_serial": int(parent_record["serial"]),
                "custom_id": f"{batch_id}--{_custom_id(int(parent_record['signed_pan']), int(parent_record['camera_elevation']), int(parent_record['head_pitch']), serial)}",
                "filename": _image_filename(
                    int(parent_record["signed_pan"]),
                    int(parent_record["camera_elevation"]),
                    int(parent_record["head_pitch"]),
                    serial,
                    batch_id=batch_id,
                ),
                "parent_custom_id": parent_id,
                "parent_filename": parent_record["filename"],
                "augmentation_type": "face_mask",
            }
        )
        mask_description = MASK_VARIANTS[(serial - 1) % len(MASK_VARIANTS)]
        edit_prompt = _mask_augmentation_prompt(record, mask_description)
        record["base_prompt"] = parent_record["prompt"]
        record["prompt"] = edit_prompt
        record["mask_description"] = mask_description
        edit_source = run_dir / "edit_inputs" / parent_record["filename"]
        shutil.copy2(source, edit_source)
        if sha256_file(edit_source) != source_sha256:
            raise PipelineError("mask edit source changed while copying")
        request = edit_batch_request(record, config["api"], edit_source, edit_prompt)
        requests.append(request)
        item = {
            "status": "planned",
            "operation": "mask_augmentation_edit",
            "filename": record["filename"],
            "parent_custom_id": parent_id,
            "parent_filename": parent_record["filename"],
            "parent_sha256": source_sha256,
            "edit_round": 1,
            "edit_reasons": ["add_face_mask"],
            "edit_prompt": edit_prompt,
            "mask_description": mask_description,
        }
        items[record["custom_id"]] = item
        lineage.append({"custom_id": record["custom_id"], **item})
        records.append(record)
    write_jsonl(run_dir / PLAN_NAME, records)
    write_jsonl(run_dir / "edit_lineage.jsonl", lineage)

    evidence_run = (edit_token_evidence_run_dir or parent_run_dir).resolve()
    token_profile = _observed_input_token_profile(evidence_run, EDIT_ENDPOINT)
    token_plan = _token_based_batch_plan(
        len(requests), token_profile["observed_mean_input_tokens"]
    )
    chunks = _chunk_batch_requests(
        requests, max_records=int(token_plan["max_records_per_batch"])
    )
    shards: list[dict[str, Any]] = []
    for shard_index, chunk in enumerate(chunks):
        input_name = f"batch_input_{shard_index:03d}_attempt_00.jsonl"
        input_path = run_dir / input_name
        write_jsonl(input_path, chunk)
        custom_ids = validate_batch_jsonl(
            input_path, config["api"], expected_endpoint=EDIT_ENDPOINT
        )
        if input_path.stat().st_size > 200 * 1024 * 1024:
            raise PipelineError("Batch input exceeds the 200 MB API limit")
        shards.append(
            {
                "index": shard_index,
                "custom_ids": custom_ids,
                "attempts": [
                    {
                        "number": 0,
                        "endpoint": EDIT_ENDPOINT,
                        "input_path": input_name,
                        "input_sha256": sha256_file(input_path),
                        "custom_ids": custom_ids,
                        "input_file_id": None,
                        "batch_id": None,
                        "status": "planned",
                        "output_file_id": None,
                        "error_file_id": None,
                        "request_counts": None,
                        "history": [
                            {"at": utc_now(), "status": "planned_mask_augmentation"}
                        ],
                    }
                ],
            }
        )
    final_count = base_count + len(records)
    reference_cost = float(
        config["api"].get("documented_reference_cost_per_image_usd", 0.0)
    )
    state = {
        "schema_version": 1,
        "local_batch_id": batch_id,
        "stage": parent_state["stage"],
        "status": "planned",
        "seed": seed,
        "config_path": str(config_path.resolve()),
        "config_sha256": sha256_file(config_path),
        "api_request": config["api"],
        "plan_path": PLAN_NAME,
        "plan_sha256": sha256_file(run_dir / PLAN_NAME),
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "parent_batch_dir": str(parent_run_dir),
        "parent_state_sha256": sha256_file(parent_run_dir / STATE_NAME),
        "parent_plan_sha256": parent_state["plan_sha256"],
        "parent_qa_sha256": sha256_file(qa_path),
        "parent_approval_sha256": None,
        "direct_production": True,
        "approval_policy": "operator_direct_no_human_review",
        "intermediate_stages_waived": True,
        "edit_round": 1,
        "max_edit_rounds": 2,
        "augmentation_type": "face_mask",
        "base_dataset_count": base_count,
        "target_fraction": target_fraction,
        "target_count": len(records),
        "projected_combined_count": final_count,
        "projected_mask_fraction": len(records) / final_count,
        "eligible_abs_pan_max": 90,
        "request_count": len(records),
        "reference_cost_per_request_usd": reference_cost,
        "reference_projected_cost_usd": round(reference_cost * len(records), 6),
        "planning_cost_per_request_usd": float(planning_cost_per_request_usd),
        "planning_projected_cost_usd": round(
            planning_cost_per_request_usd * len(records), 6
        ),
        "planning_cost_basis": "operator_supplied_observed_edit_cost",
        "token_batch_plans": {EDIT_ENDPOINT: {**token_profile, **token_plan}},
        "items": items,
        "shards": shards,
    }
    _atomic_json(run_dir / STATE_NAME, state)
    return run_dir


def create_edit_cycle(
    parent_run_dir: Path,
    batch_id: str,
    output_root: Path,
    *,
    max_edit_rounds: int,
    planning_cost_per_request_usd: float,
    include_pitch_calibration_tail: bool = False,
    edit_token_evidence_run_dir: Path | None = None,
    generation_token_evidence_run_dir: Path | None = None,
    only_edit_reasons: set[str] | None = None,
    regenerate_quality_failures: bool = False,
) -> Path:
    if not batch_id or any(character in batch_id for character in "/\\"):
        raise PipelineError("batch-id must be one safe path component")
    if max_edit_rounds < 1:
        raise PipelineError("max edit rounds must be positive")
    if planning_cost_per_request_usd <= 0:
        raise PipelineError("planning cost per edit request must be positive")
    parent_run_dir = parent_run_dir.resolve()
    parent_state = load_state(parent_run_dir)
    parent_plan = read_plan(parent_run_dir, parent_state)
    qa_path = parent_run_dir / "auto_qa.jsonl"
    if not qa_path.exists():
        raise PipelineError("run auto QA before planning image edits")
    qa_rows = {row["custom_id"]: row for row in read_jsonl(qa_path)}
    if set(qa_rows) != set(parent_plan):
        raise PipelineError(
            "auto QA rows do not exactly match the parent generation plan"
        )
    edit_round = int(parent_state.get("edit_round", 0)) + 1
    if edit_round > max_edit_rounds and not regenerate_quality_failures:
        raise PipelineError(
            f"edit round {edit_round} exceeds configured maximum {max_edit_rounds}; discard or regenerate"
        )
    config_path = Path(parent_state["config_path"])
    config = load_config(config_path)
    if config["api"]["model"] != BATCH_IMAGE_MODEL:
        raise PipelineError(
            f"Batch image editing requires api.model={BATCH_IMAGE_MODEL!r}"
        )
    calibration_tail: list[dict[str, Any]] = []
    calibration_path = parent_run_dir / "pitch_calibration.json"
    if include_pitch_calibration_tail:
        if parent_state["stage"] != "validation" or not calibration_path.exists():
            raise PipelineError(
                "pitch calibration tail retry requires Validation pitch_calibration.json"
            )
        calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
        if calibration.get("run_id") != parent_state["local_batch_id"]:
            raise PipelineError("pitch calibration belongs to a different run")
        calibration_tail = _pitch_calibration_tail_candidates(
            qa_rows, calibration, config
        )
    calibration_tail_ids = {
        str(candidate["custom_id"]) for candidate in calibration_tail
    }
    run_dir = output_root / "batches" / batch_id
    if run_dir.exists():
        raise PipelineError(f"refusing to overwrite existing run: {run_dir}")
    run_dir.mkdir(parents=True)
    (run_dir / "images").mkdir()
    (run_dir / "edit_inputs").mkdir()

    records: list[dict[str, Any]] = []
    old_to_new: dict[str, str] = {}
    yawpose = config["generation"].get("label_convention") == "yawpose"
    for parent_record in sorted(
        parent_plan.values(), key=lambda value: int(value["serial"])
    ):
        record = dict(parent_record)
        core_id = str(parent_record["custom_id"]).split("--", 1)[-1]
        record["custom_id"] = f"{batch_id}--{core_id}"
        if yawpose:
            record["filename"] = _yawpose_filename(
                int(record["yaw_yawpose"]),
                int(record["head_pitch"]),
                int(record["camera_elevation"]),
                int(record["serial"]),
            )
        else:
            record["filename"] = _image_filename(
                int(record["signed_pan"]),
                int(record["camera_elevation"]),
                int(record["head_pitch"]),
                int(record["serial"]),
                batch_id=batch_id,
            )
        record["edit_round"] = edit_round
        record["parent_custom_id"] = parent_record["custom_id"]
        records.append(record)
        old_to_new[parent_record["custom_id"]] = record["custom_id"]
    lineage: list[dict[str, Any]] = []
    items: dict[str, dict[str, Any]] = {}
    requests_by_endpoint: dict[str, list[dict[str, Any]]] = {
        ENDPOINT: [],
        EDIT_ENDPOINT: [],
    }
    records_by_id = {record["custom_id"]: record for record in records}
    for parent_id, parent_record in parent_plan.items():
        record = records_by_id[old_to_new[parent_id]]
        row = qa_rows[parent_id]
        reasons = _qa_edit_reasons(row, config)
        if only_edit_reasons is not None:
            reasons = [reason for reason in reasons if reason in only_edit_reasons]
        if parent_id in calibration_tail_ids:
            reasons = list(dict.fromkeys([*reasons, "pitch_calibration_tail"]))
        parent_item = parent_state.get("items", {}).get(parent_id, {})
        previous_reasons = {
            str(value) for value in parent_item.get("edit_reasons") or []
        }
        pitch_reference = (
            _pitch_reference_object_instruction(record, row)
            if PITCH_EDIT_REASONS.intersection(reasons)
            and (
                PITCH_EDIT_REASONS.intersection(previous_reasons)
                or "pitch_calibration_tail" in reasons
                or int(record["edit_round"]) > 1
            )
            else None
        )
        source = parent_run_dir / "images" / parent_record["filename"]
        source_valid = _valid_image(source, parent_record["size"])
        source_sha256 = sha256_file(source) if source_valid else None
        item = {
            "filename": record["filename"],
            "parent_custom_id": parent_id,
            "parent_filename": parent_record["filename"],
            "parent_sha256": source_sha256,
            "edit_round": edit_round,
            "edit_reasons": reasons,
            "previous_edit_reasons": sorted(previous_reasons),
            "pitch_reference_object": pitch_reference,
        }
        if not reasons:
            if not source_valid:
                raise PipelineError(
                    f"QA passed but parent image is unavailable: {source}"
                )
            target = run_dir / "images" / record["filename"]
            shutil.copy2(source, target)
            if sha256_file(target) != source_sha256:
                raise PipelineError("carried-forward image changed while copying")
            item.update(
                {
                    "status": "success",
                    "operation": "carry_forward",
                    "sha256": source_sha256,
                }
            )
            lineage.append({"custom_id": record["custom_id"], **item})
            items[record["custom_id"]] = item
            continue
        if source_valid and not regenerate_quality_failures:
            edit_source = run_dir / "edit_inputs" / parent_record["filename"]
            shutil.copy2(source, edit_source)
            prompt = _edit_prompt(record, row, reasons, pitch_reference=pitch_reference)
            request = edit_batch_request(record, config["api"], edit_source, prompt)
            operation = "edit"
            item["edit_prompt"] = prompt
        else:
            request_record = dict(record)
            if regenerate_quality_failures:
                parent_prompt = str(record["prompt"])
                fresh_attribute_changes = (
                    _refresh_yawpose_regeneration_prompt(record, config, edit_round)
                    if yawpose
                    else {}
                )
                request_record = dict(record)
                regeneration_prompt = _regeneration_prompt(record, row, reasons)
                request_record["prompt"] = regeneration_prompt
                item["regeneration_prompt"] = regeneration_prompt
                item["fresh_prompt"] = yawpose
                item["parent_prompt_sha256"] = hashlib.sha256(
                    parent_prompt.encode("utf-8")
                ).hexdigest()
                item["fresh_base_prompt_sha256"] = hashlib.sha256(
                    str(record["prompt"]).encode("utf-8")
                ).hexdigest()
                item["fresh_attribute_changes"] = fresh_attribute_changes
            request = batch_request(request_record, config["api"])
            operation = (
                "regenerate_quality_failure"
                if regenerate_quality_failures
                else "regenerate_missing_source"
            )
        requests_by_endpoint[request["url"]].append(request)
        item.update({"status": "planned", "operation": operation})
        lineage.append({"custom_id": record["custom_id"], **item})
        items[record["custom_id"]] = item
    selected_count = sum(len(requests) for requests in requests_by_endpoint.values())
    if selected_count == 0:
        raise PipelineError("auto QA found no actionable image edit candidates")
    write_jsonl(run_dir / PLAN_NAME, records)
    write_jsonl(run_dir / "edit_lineage.jsonl", lineage)

    direct_production = bool(parent_state.get("direct_production"))
    token_batch_plans: dict[str, dict[str, Any]] = {}
    max_records_by_endpoint: dict[str, int] = {}
    if direct_production:
        for endpoint, requests in requests_by_endpoint.items():
            if not requests:
                continue
            evidence_run = (
                (generation_token_evidence_run_dir or parent_run_dir)
                if endpoint == ENDPOINT
                else (edit_token_evidence_run_dir or parent_run_dir)
            )
            profile = _observed_input_token_profile(evidence_run, endpoint)
            plan = _token_based_batch_plan(
                len(requests), profile["observed_mean_input_tokens"]
            )
            token_batch_plans[endpoint] = {**profile, **plan}
            configured_shard_size = int(
                config["stages"][parent_state["stage"]]["shard_size"]
            )
            max_records_by_endpoint[endpoint] = min(
                configured_shard_size, int(plan["max_records_per_batch"])
            )
    else:
        configured_shard_size = int(
            config["stages"][parent_state["stage"]]["shard_size"]
        )
        for endpoint, requests in requests_by_endpoint.items():
            if requests:
                max_records_by_endpoint[endpoint] = configured_shard_size
    shards: list[dict[str, Any]] = []
    shard_index = 0
    for endpoint in (EDIT_ENDPOINT, ENDPOINT):
        for chunk in _chunk_batch_requests(
            requests_by_endpoint[endpoint],
            max_records=max_records_by_endpoint.get(endpoint, 1),
        ):
            input_name = f"batch_input_{shard_index:03d}_attempt_00.jsonl"
            input_path = run_dir / input_name
            write_jsonl(input_path, chunk)
            custom_ids = validate_batch_jsonl(
                input_path, config["api"], expected_endpoint=endpoint
            )
            if input_path.stat().st_size > 200 * 1024 * 1024:
                raise PipelineError("Batch input exceeds the 200 MB API limit")
            shards.append(
                {
                    "index": shard_index,
                    "custom_ids": custom_ids,
                    "attempts": [
                        {
                            "number": 0,
                            "endpoint": endpoint,
                            "input_path": input_name,
                            "input_sha256": sha256_file(input_path),
                            "custom_ids": custom_ids,
                            "input_file_id": None,
                            "batch_id": None,
                            "status": "planned",
                            "output_file_id": None,
                            "error_file_id": None,
                            "request_counts": None,
                            "history": [
                                {"at": utc_now(), "status": "planned_edit_cycle"}
                            ],
                        }
                    ],
                }
            )
            shard_index += 1
    state = {
        "schema_version": 1,
        "local_batch_id": batch_id,
        "stage": parent_state["stage"],
        "status": "planned",
        "seed": parent_state["seed"],
        "config_path": str(config_path.resolve()),
        "config_sha256": sha256_file(config_path),
        "api_request": config["api"],
        "plan_path": PLAN_NAME,
        "plan_sha256": sha256_file(run_dir / PLAN_NAME),
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "parent_batch_dir": str(parent_run_dir),
        "parent_state_sha256": sha256_file(parent_run_dir / STATE_NAME),
        "parent_plan_sha256": parent_state["plan_sha256"],
        "parent_qa_sha256": sha256_file(qa_path),
        "parent_approval_sha256": None,
        "direct_production": direct_production,
        "single_batch": False,
        "sequential_batches": bool(parent_state.get("sequential_batches")),
        "label_convention": parent_state.get(
            "label_convention", config["generation"].get("label_convention")
        ),
        "approval_policy": parent_state.get("approval_policy"),
        "intermediate_stages_waived": bool(
            parent_state.get("intermediate_stages_waived")
        ),
        "edit_round": edit_round,
        "max_edit_rounds": max_edit_rounds,
        "regenerate_quality_failures": regenerate_quality_failures,
        "target_count": len(records),
        "request_count": selected_count,
        "reference_cost_per_request_usd": float(
            config["api"].get("documented_reference_cost_per_image_usd", 0.0)
        ),
        "reference_projected_cost_usd": round(
            float(config["api"].get("documented_reference_cost_per_image_usd", 0.0))
            * selected_count,
            6,
        ),
        "planning_cost_per_request_usd": float(planning_cost_per_request_usd),
        "planning_projected_cost_usd": round(
            float(planning_cost_per_request_usd) * selected_count, 6
        ),
        "planning_cost_basis": "operator_supplied_observed_edit_cost",
        "token_batch_plans": token_batch_plans,
        "edit_reason_filter": (
            sorted(only_edit_reasons) if only_edit_reasons is not None else None
        ),
        "forced_edit_policy": (
            {
                "type": "pitch_calibration_tail",
                "calibration_path": str(calibration_path),
                "calibration_sha256": sha256_file(calibration_path),
                "selection": calibration_tail,
            }
            if include_pitch_calibration_tail
            else None
        ),
        "items": items,
        "shards": shards,
    }
    _atomic_json(run_dir / STATE_NAME, state)
    return run_dir


def process_output_jsonl(
    path: Path,
    run_dir: Path,
    state: dict[str, Any],
    plan: dict[str, dict[str, Any]],
) -> tuple[set[str], list[dict[str, Any]]]:
    changed: set[str] = set()
    usage_rows: list[dict[str, Any]] = []
    for line_number, row in enumerate(iter_jsonl(path), 1):
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
                raise PipelineError(
                    "response does not contain exactly one b64_json image"
                )
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
        except (
            PipelineError,
            OSError,
            ValueError,
            binascii.Error,
            UnidentifiedImageError,
        ) as exc:
            if custom_id in state["items"]:
                state["items"][custom_id].update(
                    {"status": "collect_error", "error": str(exc)}
                )
            else:
                state.setdefault("collection_errors", []).append(
                    {"file": path.name, "line": line_number, "error": str(exc)}
                )
    return changed, usage_rows


def _process_error_jsonl(path: Path, state: dict[str, Any]) -> None:
    for row in iter_jsonl(path):
        custom_id = row.get("custom_id")
        if (
            custom_id in state["items"]
            and state["items"][custom_id].get("status") != "success"
        ):
            response = row.get("response") or {}
            body = response.get("body") or {}
            error = row.get("error") or body.get("error")
            state["items"][custom_id].update(
                {
                    "status": "api_error",
                    "api_error": error,
                    "api_status_code": response.get("status_code"),
                    "api_request_id": response.get("request_id"),
                }
            )


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
            {
                "custom_id": custom_id,
                "filename": path.name,
                "sha256": digest,
                "duplicate_of": duplicate_of,
            }
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
                output_already_processed = (
                    attempt.get("local_output_path") == output.name
                    and output.exists()
                    and attempt.get("local_output_sha256") == sha256_file(output)
                )
                if not output.exists():
                    _download_file(client, attempt["output_file_id"], output)
                if not output_already_processed:
                    new_ids, new_usage = process_output_jsonl(
                        output, run_dir, state, plan
                    )
                    changed.update(new_ids)
                    usage.update({row["custom_id"]: row for row in new_usage})
                    attempt["local_output_path"] = output.name
                    attempt["local_output_sha256"] = sha256_file(output)
            if attempt.get("error_file_id"):
                error = run_dir / f"{prefix}_error.jsonl"
                error_already_processed = (
                    attempt.get("local_error_path") == error.name
                    and error.exists()
                    and attempt.get("local_error_sha256") == sha256_file(error)
                )
                if not error.exists():
                    _download_file(client, attempt["error_file_id"], error)
                if not error_already_processed:
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
        "success": sum(
            item.get("status") == "success" for item in state["items"].values()
        ),
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
            if not _valid_image(
                run_dir / "images" / plan[custom_id]["filename"],
                plan[custom_id]["size"],
            )
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
        endpoint = str(latest.get("endpoint", ENDPOINT))
        if state.get("edit_round"):
            previous_requests = {
                row["custom_id"]: row
                for row in read_jsonl(run_dir / latest["input_path"])
            }
            if not set(missing).issubset(previous_requests):
                raise PipelineError("edit retry source requests are incomplete")
            retry_requests = [previous_requests[custom_id] for custom_id in missing]
        else:
            retry_requests = [
                batch_request(plan[custom_id], state["api_request"])
                for custom_id in missing
            ]
        write_jsonl(input_path, retry_requests)
        validate_batch_jsonl(
            input_path, state["api_request"], expected_endpoint=endpoint
        )
        shard["attempts"].append(
            {
                "number": number,
                "endpoint": endpoint,
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


def build_usage_report(
    run_dir: Path, *, actual_cost_usd: float | None = None
) -> dict[str, Any]:
    state = load_state(run_dir)
    usage_path = run_dir / "usage.jsonl"
    usage_rows = read_jsonl(usage_path) if usage_path.exists() else []
    success = sum(item.get("status") == "success" for item in state["items"].values())
    requested_ids = {
        str(custom_id) for shard in state["shards"] for custom_id in shard["custom_ids"]
    }
    completed_requests = sum(
        state["items"].get(custom_id, {}).get("status") == "success"
        for custom_id in requested_ids
    )
    image_paths = [
        run_dir / "images" / item["filename"]
        for item in state["items"].values()
        if item.get("status") == "success"
    ]
    approved_path = run_dir / "approved_annotations.jsonl"
    approved = read_jsonl(approved_path) if approved_path.exists() else []
    if approved:
        quality_rows = approved
        quality_source = "approved_annotations"
        pan_quality = sum(bool(row.get("pan_quality_pass")) for row in quality_rows)
        high_angle = sum(
            bool(row.get("counts_toward_high_angle_quota")) for row in quality_rows
        )
        eye_level = sum(
            bool(row.get("pan_quality_pass"))
            and row.get("camera_elevation_class") == "eye_level_or_low_angle"
            for row in quality_rows
        )
    else:
        auto_path = run_dir / "auto_qa.jsonl"
        quality_rows = read_jsonl(auto_path) if auto_path.exists() else []
        quality_source = "auto_qa" if quality_rows else None
        pan_quality = sum(
            bool(row.get("pan_quality_pass_auto")) for row in quality_rows
        )
        high_angle = sum(
            bool(row.get("pan_quality_pass_auto"))
            and row.get("camera_elevation_class_auto") == "high_angle_match"
            for row in quality_rows
        )
        eye_level = sum(
            bool(row.get("pan_quality_pass_auto"))
            and row.get("camera_elevation_class_auto") == "eye_level_or_low_angle"
            for row in quality_rows
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
        "completed_requests": completed_requests,
        "completed_images": success,
        "failed_or_missing": max(0, state["request_count"] - completed_requests),
        "usage_records": len(usage_rows),
        "usage_path": str(usage_path) if usage_path.exists() else None,
        "usage_sha256": sha256_file(usage_path) if usage_path.exists() else None,
        "usage_totals": _sum_numeric_usage(usage_rows),
        "image_bytes": total_bytes,
        "bytes_per_completed": total_bytes / success if success else None,
        "pan_quality_pass": pan_quality,
        "quality_source": quality_source,
        "high_angle_qualified": high_angle,
        "retained_eye_level": eye_level,
        "actual_cost_usd": actual_cost_usd,
        "actual_cost_per_completed_usd": (
            actual_cost_usd / completed_requests
            if actual_cost_usd is not None and completed_requests
            else None
        ),
        "actual_cost_per_pan_quality_usd": (
            actual_cost_usd / pan_quality
            if actual_cost_usd is not None and pan_quality
            else None
        ),
        "actual_cost_per_high_angle_usd": (
            actual_cost_usd / high_angle
            if actual_cost_usd is not None and high_angle
            else None
        ),
        "documented_reference_cost_per_request_usd": state.get(
            "reference_cost_per_request_usd"
        ),
        "cost_basis": "account_observed"
        if actual_cost_usd is not None
        else "documented_reference_only",
        "created_at": utc_now(),
    }
    report_path = run_dir / "usage_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report
