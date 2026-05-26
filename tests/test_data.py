import json
from pathlib import Path

import cv2
import numpy as np
import torch

from biternionnet.data import CropConfig, ManifestDataset, flip_label, write_manifest


def _write_image(path: Path, value: int = 127) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((50, 50, 3), value, dtype=np.uint8)
    cv2.imwrite(str(path), image)


def test_flip_label_classification_and_angle():
    assert flip_label({"task": "classification", "label": "left"}, {"left": "right"})["label"] == "right"
    assert flip_label({"task": "angle_deg", "angle_deg": 30.0})["angle_deg"] == 330.0


def test_manifest_dataset_shapes_for_tasks(tmp_path):
    _write_image(tmp_path / "images" / "a.jpg")
    _write_image(tmp_path / "images" / "b.jpg")
    manifest = tmp_path / "manifest.jsonl"
    write_manifest(
        [
            {"split": "train", "image": "images/a.jpg", "task": "classification", "label": "front"},
            {"split": "test", "image": "images/b.jpg", "task": "classification", "label": "left"},
        ],
        manifest,
    )
    dataset = ManifestDataset(
        manifest,
        "train",
        "classification",
        crop=CropConfig((46, 46)),
        class_to_idx={"front": 0, "left": 1},
    )
    image, target = dataset[0]
    assert tuple(image.shape) == (3, 46, 46)
    assert target.dtype == torch.long

