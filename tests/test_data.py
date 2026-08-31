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


def test_scale_jitter_varies_resize_but_keeps_crop_size():
    import random

    from biternionnet.data import prepare_image, resize_for_crop

    image = np.zeros((25, 23, 3), dtype=np.float32)
    random.seed(1)
    crop = CropConfig((46, 46), random_crop=True, resize=(50, 50), scale_jitter=(0.9, 1.1))
    sizes = {resize_for_crop(image, crop).shape[:2] for _ in range(60)}
    assert len(sizes) > 1
    assert all(46 <= h <= 55 and 46 <= w <= 55 for h, w in sizes)  # never below the crop size
    assert all(prepare_image(image, crop).shape == (46, 46, 3) for _ in range(20))
    # jitter is ignored for deterministic (test) crops
    fixed = CropConfig((46, 46), random_crop=False, resize=(50, 50), scale_jitter=(0.9, 1.1))
    assert resize_for_crop(image, fixed).shape[:2] == (50, 50)


def test_dataset_applies_photometric_only_when_configured(tmp_path):
    import random

    from biternionnet.augment import PHOTOMETRIC_PRESETS

    _write_image(tmp_path / "images" / "a.jpg")
    manifest = tmp_path / "manifest.jsonl"
    write_manifest([{"split": "train", "image": "images/a.jpg", "task": "angle_deg", "angle_deg": 10.0}], manifest)
    plain = ManifestDataset(manifest, "train", "angle_deg", crop=CropConfig((46, 46), resize=(50, 50)))
    aug = ManifestDataset(manifest, "train", "angle_deg", crop=CropConfig((46, 46), resize=(50, 50)), photometric=PHOTOMETRIC_PRESETS["cctv"])
    noop = ManifestDataset(manifest, "train", "angle_deg", crop=CropConfig((46, 46), resize=(50, 50)), photometric=PHOTOMETRIC_PRESETS["none"])
    assert noop.photometric is None
    a, _ = plain[0]
    random.seed(0)
    b, _ = aug[0]
    assert a.shape == b.shape == (3, 46, 46)
    assert not torch.equal(a, b)
    assert torch.equal(a, noop[0][0])


def test_resize_for_crop_clips_lanczos_overshoot():
    from biternionnet.data import resize_for_crop

    image = np.zeros((25, 23, 3), dtype=np.float32)
    image[10:15, 10:14] = 1.0  # sharp edge -> ringing when upscaled
    out = resize_for_crop(image, CropConfig((46, 46), resize=(50, 50)))
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_resize_equal_to_crop_disables_translation():
    import random

    from biternionnet.data import prepare_image

    rng = np.random.default_rng(0)
    image = rng.uniform(0, 1, (30, 25, 3)).astype(np.float32)
    crop = CropConfig((46, 46), random_crop=True, resize=(46, 46))
    random.seed(0)
    first = prepare_image(image, crop)
    for seed in range(1, 20):
        random.seed(seed)
        assert np.array_equal(prepare_image(image, crop), first)  # no crop jitter left
    assert first.shape == (46, 46, 3)


def test_neighbor_label_jitter_applies_only_to_neighbors(tmp_path):
    import random

    _write_image(tmp_path / "images" / "a.jpg")
    _write_image(tmp_path / "images" / "b.jpg")
    manifest = tmp_path / "manifest.jsonl"
    write_manifest(
        [
            {"split": "train", "image": "images/a.jpg", "task": "angle_deg", "angle_deg": 100.0},
            {"split": "train", "image": "images/b.jpg", "task": "angle_deg", "angle_deg": 100.0, "source": "neighbor", "anchor_frame": 1, "frame_offset": 1},
        ],
        manifest,
    )
    ds = ManifestDataset(manifest, "train", "angle_deg", crop=CropConfig((46, 46)), neighbor_label_jitter_deg=2.0)
    random.seed(0)
    anchor_targets = {float(ds[0][1]) for _ in range(10)}
    neighbor_targets = [float(ds[1][1]) for _ in range(50)]
    assert anchor_targets == {100.0}  # anchors untouched
    assert all(98.0 <= t <= 102.0 for t in neighbor_targets)
    assert len(set(neighbor_targets)) > 10  # fresh draw each access
    # records themselves are not mutated
    assert ds.records[1]["angle_deg"] == 100.0
