from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .data import open_pickle, write_manifest


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

