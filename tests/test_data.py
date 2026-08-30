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


def _write_sized_image(path: Path, size: tuple[int, int], value: int = 127) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((size[0], size[1], 3), value, dtype=np.uint8)
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


def test_manifest_dataset_resizes_small_images_before_crop(tmp_path):
    _write_sized_image(tmp_path / "images" / "small.jpg", (25, 23))
    manifest = tmp_path / "manifest.jsonl"
    write_manifest(
        [{"split": "train", "image": "images/small.jpg", "task": "angle_deg", "angle_deg": 90.0}],
        manifest,
    )
    dataset = ManifestDataset(manifest, "train", "angle_deg", crop=CropConfig((46, 46)))
    image, target = dataset[0]
    assert tuple(image.shape) == (3, 46, 46)
    assert target.item() == 90.0


def test_crop_config_resize_then_random_crop(tmp_path):
    from biternionnet.data import crop_image
    import random

    image = np.zeros((25, 23, 3), dtype=np.float32)
    random.seed(0)
    cropped = crop_image(image, CropConfig((46, 46), random_crop=True, resize=(50, 50)))
    assert cropped.shape == (46, 46, 3)
    # 50x50 -> 46x46 leaves a 4px margin so random crops actually vary.
    offsets = set()
    for _ in range(50):
        big = np.arange(50 * 50, dtype=np.float32).reshape(50, 50, 1).repeat(3, axis=2)
        offsets.add(float(crop_image(big, CropConfig((46, 46), random_crop=True, resize=(50, 50)))[0, 0, 0]))
    assert len(offsets) > 1


def test_flip_label_rejects_pose_rad():
    import pytest

    with pytest.raises(ValueError):
        flip_label({"task": "pose_rad", "pan": 0.1, "tilt": 0.0, "roll": 0.0})
