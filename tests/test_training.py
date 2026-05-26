from pathlib import Path

import cv2
import numpy as np

from biternionnet.train import evaluate_checkpoint, train_model
from biternionnet.data import write_manifest


def _image(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.full((50, 50, 3), value, dtype=np.uint8))


def test_smoke_biternion_training_creates_checkpoint(tmp_path):
    for i in range(4):
        _image(tmp_path / "images" / f"{i}.jpg", 40 + i * 30)
    manifest = tmp_path / "manifest.jsonl"
    write_manifest(
        [
            {"split": "train", "image": "images/0.jpg", "task": "angle_deg", "angle_deg": 0.0},
            {"split": "train", "image": "images/1.jpg", "task": "angle_deg", "angle_deg": 90.0},
            {"split": "test", "image": "images/2.jpg", "task": "angle_deg", "angle_deg": 180.0},
            {"split": "test", "image": "images/3.jpg", "task": "angle_deg", "angle_deg": 270.0},
        ],
        manifest,
    )
    result = train_model(
        "smoke-biternion",
        manifest,
        tmp_path / "run",
        epochs=1,
        batch_size=2,
        device_name="cpu",
        train_flip_probability=0.0,
    )
    checkpoint = Path(result["best_checkpoint"])
    assert checkpoint.exists()
    metrics = evaluate_checkpoint(checkpoint, manifest, device_name="cpu", batch_size=2)
    assert "maad_deg" in metrics

