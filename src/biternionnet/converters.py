from __future__ import annotations

import json
import os
import random
import re
from pathlib import Path

import numpy as np

from .data import open_pickle, write_manifest


PAN_RE = re.compile(r"pan = ([+-]?\d+(?:\.\d+)?)")
VALID_RE = re.compile(r"valid = ([01])")


def _relpath(path: Path, start: Path) -> str:
    return os.path.relpath(path.resolve(), start.resolve())


def convert_tosato_classification(source: str | Path, output: str | Path) -> None:
    source = Path(source)
    data = json.loads(source.read_text(encoding="utf-8"))
    records = []
    for split in ("train", "test"):
        for label, images in data[split].items():
            for image in images:
                records.append({"split": split, "image": image, "task": "classification", "label": label})
    write_manifest(records, output)


def convert_pickle_classification(source: str | Path, output: str | Path) -> None:
    _xtr, _xte, ytr, yte, ntr, nte, le = open_pickle(source)
    records = []
    for split, labels, names in (("train", ytr, ntr), ("test", yte, nte)):
        for label, image in zip(labels, names):
            records.append({"split": split, "image": image, "task": "classification", "label": str(le.inverse_transform([label])[0])})
    write_manifest(records, output)


def convert_towncentre_pickle(source: str | Path, output: str | Path, image_root: str = "TownCentreHeadImages") -> None:
    _x, y, names = open_pickle(source)
    records = []
    for i, (angle, name) in enumerate(zip(y, names)):
        split = "test" if i % 10 == 0 else "train"
        records.append({"split": split, "image": f"{image_root}/{name}", "task": "angle_deg", "angle_deg": float(angle)})
    write_manifest(records, output)


def convert_towncentre_raw(source: str | Path, output: str | Path, train_split: float = 0.9, seed: int = 0) -> None:
    source = Path(source)
    output = Path(output)
    rng = random.Random(seed)
    person_splits: dict[int, str] = {}
    records = []

    for label_path in sorted(source.glob("*/*.txt")):
        lines = label_path.read_text(encoding="utf-8").splitlines()
        if len(lines) < 2:
            continue
        pan_match = PAN_RE.match(lines[0])
        valid_match = VALID_RE.match(lines[1])
        if pan_match is None or valid_match is None or valid_match.group(1) == "0":
            continue

        stem = label_path.stem
        image_candidates = [path for path in label_path.parent.iterdir() if path.is_file() and path.stem.startswith(stem) and path.suffix.lower() != ".txt"]
        if len(image_candidates) != 1:
            raise ValueError(f"Expected exactly one image for {label_path}, found {len(image_candidates)}")

        image_path = image_candidates[0]
        try:
            person_id = int(image_path.name.split("_")[1])
        except (IndexError, ValueError) as exc:
            raise ValueError(f"Could not extract TownCentre person id from {image_path.name}") from exc

        if person_id not in person_splits:
            person_splits[person_id] = "train" if rng.random() < train_split else "test"

        records.append(
            {
                "split": person_splits[person_id],
                "image": _relpath(image_path, output.parent),
                "task": "angle_deg",
                "angle_deg": float((float(pan_match.group(1)) + 720.0) % 360.0),
            }
        )

    if not records:
        raise ValueError(f"No valid TownCentre records found under {source}")
    write_manifest(records, output)


def convert_idiap_pickle(source: str | Path, output: str | Path, train_root: str = "IHDPHeadPose/train", test_root: str = "IHDPHeadPose/test") -> None:
    (xtr, ptr, ttr, rtr, ntr), (xte, pte, tte, rte, nte) = open_pickle(source)
    del xtr, xte
    records = []
    for split, root, pan, tilt, roll, names in (
        ("train", train_root, ptr, ttr, rtr, ntr),
        ("test", test_root, pte, tte, rte, nte),
    ):
        for values in zip(pan, tilt, roll, names):
            p, t, r, name = values
            records.append({"split": split, "image": f"{root}/{name}", "task": "pose_rad", "pan": float(p), "tilt": float(t), "roll": float(r)})
    write_manifest(records, output)


def convert_caviar_pickle(source: str | Path, output: str | Path, train_root: str, test_root: str) -> None:
    (xtr, ptr, *_train), (xte, pte, *_test) = open_pickle(source)
    del xtr, xte
    ntr = _train[-1]
    nte = _test[-1]
    records = []
    for split, root, angles, names in (("train", train_root, ptr, ntr), ("test", test_root, pte, nte)):
        for angle, name in zip(np.asarray(angles), names):
            records.append({"split": split, "image": f"{root}/{name}.jpg", "task": "angle_deg", "angle_deg": float(angle)})
    write_manifest(records, output)
