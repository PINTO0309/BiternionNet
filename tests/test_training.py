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
    predictions = tmp_path / "predictions.jsonl"
    metrics = evaluate_checkpoint(
        checkpoint,
        manifest,
        device_name="cpu",
        batch_size=2,
        predictions_output=predictions,
    )
    assert "maad_deg" in metrics
    assert metrics["bin_180_count"] == 1 and metrics["bin_270_count"] == 1
    assert len(predictions.read_text().splitlines()) == 2



def _angle_manifest(tmp_path: Path, with_val: bool = False) -> Path:
    records = []
    for i, angle in enumerate([0.0, 90.0, 180.0, 270.0, 45.0, 225.0]):
        _image(tmp_path / "images" / f"{i}.jpg", 40 + i * 30)
        split = "train" if i < 2 else ("val" if with_val and i < 4 else "test")
        records.append({"split": split, "image": f"images/{i}.jpg", "task": "angle_deg", "angle_deg": angle})
    manifest = tmp_path / "manifest.jsonl"
    write_manifest(records, manifest)
    return manifest


def test_best_checkpoint_only_written_with_val_split(tmp_path):
    manifest = _angle_manifest(tmp_path, with_val=False)
    result = train_model("smoke-biternion", manifest, tmp_path / "run", epochs=1, batch_size=2, device_name="cpu")
    assert result["best_checkpoint"] == result["last_checkpoint"]
    assert not (tmp_path / "run" / "best.pt").exists()

    manifest = _angle_manifest(tmp_path, with_val=True)
    result = train_model("smoke-biternion", manifest, tmp_path / "run2", epochs=1, batch_size=2, device_name="cpu")
    assert Path(result["best_checkpoint"]).name == "best.pt"
    assert "val_maad_deg" in result["history"][0]


def test_quantized_softmax_evaluates_against_continuous_angle(tmp_path):
    manifest = _angle_manifest(tmp_path)
    result = train_model("towncentre-q4x-softmax", manifest, tmp_path / "run", epochs=1, batch_size=2, device_name="cpu")
    record = result["history"][0]
    assert {"maad_deg", "maad_quadint_deg", "bin_accuracy"} <= set(record)
    metrics = evaluate_checkpoint(result["last_checkpoint"], manifest, device_name="cpu", batch_size=2)
    assert "maad_quadint_deg" in metrics


def test_quantized_biternion_evaluates_against_continuous_angle(tmp_path):
    manifest = _angle_manifest(tmp_path)
    result = train_model("towncentre-q4x-biternion", manifest, tmp_path / "run", epochs=1, batch_size=2, device_name="cpu")
    assert "maad_deg" in result["history"][0]


def test_legacy_checkpoint_without_pool_ceil_mode_loads(tmp_path):
    import torch
    from biternionnet.experiments import get_experiment
    from biternionnet.models import ModelConfig, build_model

    config = get_experiment("smoke-biternion")
    legacy = {k: v for k, v in config.__dict__.items() if k not in {"pool_ceil_mode", "resize_size", "flip_augmentation"}}
    model = build_model(ModelConfig(output_dim=2, head="biternion", pool_ceil_mode=False))
    checkpoint = tmp_path / "legacy.pt"
    torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": {}, "experiment": legacy, "class_to_idx": {}, "history": []}, checkpoint)
    manifest = _angle_manifest(tmp_path)
    metrics = evaluate_checkpoint(checkpoint, manifest, device_name="cpu", batch_size=2)
    assert "maad_deg" in metrics


def test_wsd_lr_factor_shape():
    from biternionnet.train import wsd_lr_factor

    total = 100
    kwargs = dict(total_steps=total, warmup_fraction=0.1, decay_fraction=0.3, final_lr_ratio=0.1)
    factors = [wsd_lr_factor(step, **kwargs) for step in range(total)]
    assert factors[0] == 0.1  # 1/warmup_steps
    assert factors[9] == 1.0  # end of warmup
    assert all(f == 1.0 for f in factors[10:70])  # stable
    assert factors[70] < 1.0 and factors[70] > factors[80] > factors[99]
    assert abs(factors[99] - 0.1) < 1e-9  # reaches final ratio on the last step
    assert wsd_lr_factor(total + 5, **kwargs) == 0.1
    # No warmup / no decay degenerates to constant.
    assert all(wsd_lr_factor(s, total, 0.0, 0.0, 0.1) == 1.0 for s in range(total))


def test_with_overrides_validates_schedule():
    import pytest
    from biternionnet.experiments import get_experiment, with_overrides

    config = get_experiment("smoke-biternion")
    assert with_overrides(config, None, None, None, lr_schedule="wsd").lr_schedule == "wsd"
    with pytest.raises(ValueError):
        with_overrides(config, None, None, None, lr_schedule="cosine")
    with pytest.raises(ValueError):
        with_overrides(config, None, None, None, warmup_fraction=0.6, decay_fraction=0.6)


def test_training_with_wsd_records_decayed_lr(tmp_path):
    manifest = _angle_manifest(tmp_path)
    result = train_model(
        "smoke-biternion",
        manifest,
        tmp_path / "run",
        epochs=4,
        batch_size=1,
        lr_schedule="wsd",
        warmup_fraction=0.25,
        decay_fraction=0.5,
        final_lr_ratio=0.1,
        device_name="cpu",
    )
    # 2 steps/epoch, 8 steps total: warmup 2, stable 2, decay 4. The logged lr is the value the
    # *next* step will use, so the end of the stable epoch already shows the first decay value.
    lrs = [record["lr"] for record in result["history"]]
    assert lrs[0] == 1.0  # end of warmup epoch
    assert lrs[0] > lrs[1] > lrs[2] > lrs[3]
    assert abs(lrs[-1] - 0.1) < 1e-9

    constant = train_model("smoke-biternion", manifest, tmp_path / "run-const", epochs=2, batch_size=1, device_name="cpu")
    assert all(record["lr"] == 1.0 for record in constant["history"])


def test_resume_continues_epochs_and_switches_to_cosine(tmp_path):
    import torch

    manifest = _angle_manifest(tmp_path)
    first = train_model("smoke-biternion", manifest, tmp_path / "run", epochs=3, batch_size=1, device_name="cpu")
    assert [r["epoch"] for r in first["history"]] == [1, 2, 3]
    assert torch.load(first["last_checkpoint"], map_location="cpu")["global_step"] == 6

    resumed = train_model(
        "smoke-biternion",
        manifest,
        tmp_path / "run",
        epochs=10,
        batch_size=1,
        lr_schedule="plateau_cosine",
        decay_start_epoch=5,
        cosine_epochs=3,
        resume_from=first["last_checkpoint"],
        device_name="cpu",
    )
    epochs = [r["epoch"] for r in resumed["history"]]
    assert epochs == [1, 2, 3, 4, 5, 6, 7]  # stops after the 3 cosine epochs
    assert resumed["stopped_early"] is True
    phases = [r.get("phase", "constant") for r in resumed["history"]]
    assert phases[3] == "constant" and phases[4:] == ["cosine"] * 3
    # The logged lr is the value the next step will use: epoch 3 (still constant) logs 1.0,
    # epoch 4 already shows the first cosine step.
    assert resumed["history"][2]["lr"] == 1.0
    assert resumed["history"][3]["lr"] < 1.0
    assert resumed["history"][-1]["lr"] < resumed["history"][-2]["lr"] < 1.0
    assert abs(resumed["history"][-1]["lr"] - 0.1) < 1e-9
    assert resumed["schedule_state"]["trigger_reason"] == "manual"


def test_training_with_photometric_and_scale_jitter(tmp_path):
    manifest = _angle_manifest(tmp_path)
    result = train_model(
        "towncentre-biternion",  # has resize_size, required for scale_jitter
        manifest,
        tmp_path / "run",
        epochs=1,
        batch_size=2,
        photometric="cctv",
        scale_jitter=(0.9, 1.1),
        device_name="cpu",
    )
    assert "maad_deg" in result["history"][0]
    import torch

    saved = torch.load(result["last_checkpoint"], map_location="cpu")["experiment"]
    assert saved["photometric"] == "cctv" and tuple(saved["scale_jitter"]) == (0.9, 1.1)


def test_with_overrides_validates_augmentation():
    import pytest
    from biternionnet.experiments import get_experiment, with_overrides

    config = get_experiment("towncentre-biternion")
    assert with_overrides(config, None, None, None, photometric="none").photometric is None
    with pytest.raises(ValueError):
        with_overrides(config, None, None, None, photometric="bogus")
    with pytest.raises(ValueError):
        with_overrides(config, None, None, None, scale_jitter=(1.1, 0.9))
    with pytest.raises(ValueError):
        with_overrides(get_experiment("qmul"), None, None, None, scale_jitter=(0.9, 1.1))  # no resize_size


def test_text_logs_written_and_consistent_after_resume(tmp_path):
    import json

    manifest = _angle_manifest(tmp_path)
    out = tmp_path / "run"
    first = train_model("smoke-biternion", manifest, out, epochs=2, batch_size=1, device_name="cpu")
    lines = [json.loads(l) for l in (out / "history.jsonl").read_text().splitlines()]
    assert [r["epoch"] for r in lines] == [1, 2]
    assert lines == first["history"]
    assert all("time" in r and "epoch_seconds" in r for r in lines)
    run_info = json.loads((out / "run.json").read_text())
    assert run_info["experiment"]["name"] == "smoke-biternion" and run_info["steps_per_epoch"] == 2
    events = [json.loads(l) for l in (out / "events.jsonl").read_text().splitlines()]
    assert [e["event"] for e in events] == ["start", "finish"]

    # resume into a fresh directory: history.jsonl is rebuilt from the checkpoint, then appended
    out2 = tmp_path / "run2"
    resumed = train_model(
        "smoke-biternion", manifest, out2, epochs=5, batch_size=1, device_name="cpu",
        resume_from=first["last_checkpoint"], lr_schedule="plateau_cosine", decay_start_epoch=4, cosine_epochs=1,
    )
    lines2 = [json.loads(l) for l in (out2 / "history.jsonl").read_text().splitlines()]
    assert [r["epoch"] for r in lines2] == [1, 2, 3, 4]
    assert lines2 == resumed["history"]
    events2 = [json.loads(l) for l in (out2 / "events.jsonl").read_text().splitlines()]
    assert events2[0]["event"] == "resume" and events2[0]["start_epoch"] == 3
    assert any(e["event"] == "schedule" for e in events2) and events2[-1]["event"] == "finish"
    assert events2[-1]["stopped_early"] is True


def test_evaluate_checkpoint_reports_per_bin_metrics(tmp_path):
    manifest = _angle_manifest(tmp_path)
    result = train_model("smoke-biternion", manifest, tmp_path / "run", epochs=1, batch_size=2, device_name="cpu")
    metrics = evaluate_checkpoint(result["last_checkpoint"], manifest, device_name="cpu", batch_size=2)
    assert "bin_macro_maad_deg" in metrics
    counts = [metrics.get(f"bin_{c:03d}_count", 0) for c in range(0, 360, 45)]
    assert sum(counts) == 4  # the fixture's 4 test records
