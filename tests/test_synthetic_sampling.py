from collections import Counter

import cv2
import numpy as np
import pytest

from biternionnet.data import write_manifest
from biternionnet.synthetic.sampling import SourceQuotaSampler
from biternionnet.train import train_model


def test_source_quota_sampler_is_exact_balanced_and_capped():
    records = [{"source": "neighbor"} for _ in range(20)] + [{"source": "synthetic"} for _ in range(8)]
    sampler = SourceQuotaSampler(
        records,
        synthetic_fraction=0.25,
        num_samples=32,
        max_synthetic_repeats=2,
        seed=7,
    )
    first = list(sampler)
    synthetic = [index for index in first if index >= 20]
    assert len(first) == 32 and len(synthetic) == 8
    assert max(Counter(synthetic).values()) == 1
    assert sampler.last_report["realized_synthetic_fraction"] == 0.25
    sampler.set_epoch(1)
    assert list(sampler) != first


def test_source_quota_sampler_fails_when_repeat_cap_is_impossible():
    records = [{"source": "real"} for _ in range(20)] + [{"source": "synthetic"} for _ in range(2)]
    with pytest.raises(ValueError, match="need at least"):
        SourceQuotaSampler(records, synthetic_fraction=0.5, num_samples=20, max_synthetic_repeats=4)


def test_training_uses_source_quota_sampler_and_logs_realized_fraction(tmp_path):
    rows = []
    for index in range(8):
        path = tmp_path / "images" / f"{index}.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), np.full((50, 50, 3), 30 + index * 20, dtype=np.uint8))
        if index < 4:
            rows.append(
                {
                    "split": "train",
                    "task": "angle_deg",
                    "angle_deg": float(index * 45),
                    "image": f"images/{index}.jpg",
                    "source": "real",
                }
            )
        elif index < 6:
            rows.append(
                {
                    "split": "train",
                    "task": "angle_deg",
                    "angle_deg": float(index * 45),
                    "image": f"images/{index}.jpg",
                    "source": "synthetic",
                }
            )
        else:
            rows.append(
                {
                    "split": "test",
                    "task": "angle_deg",
                    "angle_deg": float(index * 45),
                    "image": f"images/{index}.jpg",
                }
            )
    manifest = tmp_path / "manifest.jsonl"
    write_manifest(rows, manifest)
    result = train_model(
        "smoke-biternion",
        manifest,
        tmp_path / "run",
        epochs=1,
        batch_size=2,
        device_name="cpu",
        synthetic_fraction=0.25,
        epoch_samples=8,
        synthetic_max_repeats=2,
    )
    report = result["history"][0]["source_sampling"]
    assert report["synthetic_draws"] == 2
    assert report["realized_synthetic_fraction"] == 0.25
