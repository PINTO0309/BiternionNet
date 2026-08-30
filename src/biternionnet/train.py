from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import CropConfig, ManifestDataset, build_class_mapping, read_manifest
from .experiments import (
    ExperimentConfig,
    config_from_checkpoint,
    evaluation_target_kind,
    get_experiment,
    quantization_arrays,
    with_overrides,
)
from .losses import (
    CosineLoss,
    ModuloMAELoss,
    VonMisesBiternionLoss,
    VonMisesLoss,
    angle_difference_deg,
    bit2deg,
    probs2deg_centre,
    probs2deg_quadint,
    quantize_labels,
)
from .models import ModelConfig, build_model
from .schedules import build_controller, wsd_lr_factor  # noqa: F401  (wsd_lr_factor re-exported for tests)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _loss(config: ExperimentConfig) -> nn.Module:
    if config.loss == "cross_entropy":
        return nn.CrossEntropyLoss()
    if config.loss == "l1":
        return nn.L1Loss()
    if config.loss == "modulo_mae":
        return ModuloMAELoss()
    if config.loss == "vonmises_deg":
        return VonMisesLoss(config.kappa, radians=False)
    if config.loss == "vonmises_rad":
        return VonMisesLoss(config.kappa, radians=True)
    if config.loss == "cosine":
        return CosineLoss()
    if config.loss == "vonmises_biternion":
        return VonMisesBiternionLoss(config.kappa)
    raise ValueError(f"Unsupported loss: {config.loss}")


def _optimizer(config: ExperimentConfig, model: nn.Module) -> torch.optim.Optimizer:
    if config.optimizer == "adadelta":
        return torch.optim.Adadelta(model.parameters(), lr=config.lr, rho=config.rho, eps=config.eps)
    if config.optimizer == "adam":
        return torch.optim.Adam(model.parameters(), lr=config.lr)
    raise ValueError(f"Unsupported optimizer: {config.optimizer}")


def _output_dim(config: ExperimentConfig, class_to_idx: dict[str, int]) -> int:
    if config.output_dim is not None:
        return config.output_dim
    if config.target_kind == "classification":
        return len(class_to_idx)
    raise ValueError(f"Experiment {config.name} requires explicit output_dim")


def model_config(config: ExperimentConfig, class_to_idx: dict[str, int]) -> ModelConfig:
    return ModelConfig(
        output_dim=_output_dim(config, class_to_idx),
        head=config.model_head,
        variant=config.model_variant,
        input_size=config.input_size,
        backbone_activation=config.backbone_activation,
        pool_ceil_mode=config.pool_ceil_mode,
    )


def _eval_dataset(config: ExperimentConfig, manifest: str | Path, split: str, class_to_idx: dict[str, int]) -> ManifestDataset:
    borders, centres = quantization_arrays(config)
    return ManifestDataset(
        manifest,
        split,
        evaluation_target_kind(config),
        crop=CropConfig(config.input_size, random_crop=False, resize=config.resize_size),
        class_to_idx=class_to_idx,
        exclude_label=config.exclude_label,
        flip_probability=0.0,
        quantization_borders=borders,
        quantization_centres=centres,
    )


def build_datasets(
    config: ExperimentConfig,
    manifest: str | Path,
    train_flip_probability: float,
) -> tuple[ManifestDataset, ManifestDataset, ManifestDataset | None, dict[str, int]]:
    """Return (train, test, val-or-None, class_to_idx).

    The test dataset yields the evaluation target (continuous angle for quantized experiments).
    A validation dataset is returned only if the manifest contains ``split == "val"`` records.
    """
    records = read_manifest(manifest)
    class_to_idx = build_class_mapping(records, exclude_label=config.exclude_label) if config.target_kind == "classification" else {}
    borders, centres = quantization_arrays(config)
    flip_probability = train_flip_probability if config.flip_augmentation else 0.0
    train_dataset = ManifestDataset(
        manifest,
        "train",
        config.target_kind,
        crop=CropConfig(config.input_size, random_crop=True, resize=config.resize_size),
        class_to_idx=class_to_idx,
        class_flip_map=config.class_flip_map,
        exclude_label=config.exclude_label,
        flip_probability=flip_probability,
        quantization_borders=borders,
        quantization_centres=centres,
    )
    test_dataset = _eval_dataset(config, manifest, "test", class_to_idx)
    val_dataset = None
    if any(r.get("split") == "val" for r in records):
        val_dataset = _eval_dataset(config, manifest, "val", class_to_idx)
    return train_dataset, test_dataset, val_dataset, class_to_idx


def _target_deg(config: ExperimentConfig, target: torch.Tensor) -> torch.Tensor:
    """Convert an evaluation target to degrees with shape (N, 1)."""
    kind = evaluation_target_kind(config)
    if kind == "angle_rad":
        return torch.rad2deg(target).reshape(-1, 1)
    if kind == "biternion":
        return bit2deg(target).reshape(-1, 1)
    return target.reshape(-1, 1)


@torch.no_grad()
def evaluate_model(model: nn.Module, loader: DataLoader, config: ExperimentConfig, device: torch.device) -> dict[str, float]:
    model.eval()
    correct = 0
    total = 0
    angle_errors: list[torch.Tensor] = []
    quadint_errors: list[torch.Tensor] = []
    pose_errors: list[torch.Tensor] = []
    borders, centres = quantization_arrays(config)

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)
        outputs = model(images)
        if config.target_kind == "classification":
            predicted = outputs.argmax(dim=1)
            correct += int((predicted == targets).sum().item())
            total += int(targets.numel())
        elif config.target_kind == "quantized_classification":
            # Paper Sec. 5: train on bins, evaluate the angle against the continuous ground truth.
            target_deg = _target_deg(config, targets).cpu()
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            true_bins = quantize_labels(target_deg.numpy().reshape(-1), borders)
            correct += int((probs.argmax(axis=1) == true_bins).sum())
            total += int(len(true_bins))
            pred_centre = torch.as_tensor(probs2deg_centre(probs, centres), dtype=torch.float32).reshape(-1, 1)
            pred_quadint = torch.as_tensor(probs2deg_quadint(probs, centres), dtype=torch.float32).reshape(-1, 1)
            angle_errors.append(angle_difference_deg(pred_centre, target_deg))
            quadint_errors.append(angle_difference_deg(pred_quadint, target_deg))
        elif config.model_head == "biternion":
            pred_deg = bit2deg(outputs).reshape(-1, 1)
            angle_errors.append(angle_difference_deg(pred_deg, _target_deg(config, targets)).detach().cpu())
        elif config.target_kind == "pose_rad":
            err = torch.rad2deg(torch.abs(torch.atan2(torch.sin(targets - outputs), torch.cos(targets - outputs))))
            pose_errors.append(err.detach().cpu())
        else:
            pred_deg = outputs.reshape(-1, 1)
            angle_errors.append(angle_difference_deg(pred_deg, _target_deg(config, targets)).detach().cpu())

    if config.target_kind == "classification":
        return {"accuracy": correct / total}
    if pose_errors:
        errors = torch.cat(pose_errors, dim=0)
        return {
            "maad_deg": float(errors.mean().item()),
            "pan_maad_deg": float(errors[:, 0].mean().item()),
            "tilt_maad_deg": float(errors[:, 1].mean().item()),
            "roll_maad_deg": float(errors[:, 2].mean().item()),
        }
    metrics = {"maad_deg": float(torch.cat(angle_errors, dim=0).mean().item())}
    if quadint_errors:
        metrics["maad_quadint_deg"] = float(torch.cat(quadint_errors, dim=0).mean().item())
        metrics["bin_accuracy"] = correct / total
    return metrics


def _selection_score(metrics: dict[str, float]) -> float:
    return metrics.get("accuracy", -metrics.get("maad_deg", float("inf")))


def train_model(
    experiment: str,
    manifest: str | Path,
    output: str | Path,
    *,
    epochs: int | None = None,
    batch_size: int | None = None,
    lr: float | None = None,
    backbone_activation: str | None = None,
    lr_schedule: str | None = None,
    warmup_fraction: float | None = None,
    decay_fraction: float | None = None,
    final_lr_ratio: float | None = None,
    plateau_window: int | None = None,
    plateau_threshold: float | None = None,
    plateau_min_epochs: int | None = None,
    cosine_epochs: int | None = None,
    decay_start_epoch: int | None = None,
    disable_plateau_trigger: bool = False,
    resume_from: str | Path | None = None,
    seed: int = 0,
    device_name: str | None = None,
    num_workers: int = 0,
    train_flip_probability: float = 0.5,
) -> dict[str, Any]:
    """Train an experiment preset.

    ``last.pt`` is always written (the original notebooks train a fixed number of epochs and
    evaluate the final model). ``best.pt`` is written only when the manifest has a ``val`` split,
    so model selection never looks at the test split.

    ``resume_from`` continues a run from one of its checkpoints: model and optimizer state,
    metric history, global step and schedule state are restored and epoch numbering continues.
    Schedule options given here override the ones stored in the checkpoint, which is how a
    constant-lr run is switched to ``plateau_cosine`` with a hand-picked ``decay_start_epoch``.
    """
    set_seed(seed)
    config = with_overrides(
        get_experiment(experiment),
        epochs,
        batch_size,
        lr,
        backbone_activation,
        lr_schedule=lr_schedule,
        warmup_fraction=warmup_fraction,
        decay_fraction=decay_fraction,
        final_lr_ratio=final_lr_ratio,
        plateau_window=plateau_window,
        plateau_threshold=plateau_threshold,
        plateau_min_epochs=plateau_min_epochs,
        cosine_epochs=cosine_epochs,
        decay_start_epoch=decay_start_epoch,
        disable_plateau_trigger=disable_plateau_trigger,
    )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))

    resume_data = None
    if resume_from is not None:
        resume_data = torch.load(resume_from, map_location="cpu")
        resumed = config_from_checkpoint(resume_data["experiment"])
        if resumed.name != config.name:
            raise ValueError(f"Cannot resume experiment {config.name!r} from a {resumed.name!r} checkpoint")
        # Keep the architecture/data settings the checkpoint was trained with.
        config = with_overrides(
            config,
            None,
            None,
            None,
            resumed.backbone_activation,
        )
        config = config.__class__(**{**config.__dict__, "pool_ceil_mode": resumed.pool_ceil_mode, "resize_size": resumed.resize_size})

    train_dataset, test_dataset, val_dataset, class_to_idx = build_datasets(config, manifest, train_flip_probability)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=num_workers)
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=num_workers)

    model = build_model(model_config(config, class_to_idx)).to(device)
    criterion = _loss(config)
    optimizer = _optimizer(config, model)
    base_lr = config.lr

    history: list[dict[str, Any]] = []
    global_step = 0
    schedule_state: dict[str, Any] = {}
    if resume_data is not None:
        model.load_state_dict(resume_data["model_state_dict"])
        optimizer.load_state_dict(resume_data["optimizer_state_dict"])
        history = list(resume_data.get("history", []))
        global_step = int(resume_data.get("global_step", len(history) * len(train_loader)))
        schedule_state = dict(resume_data.get("schedule_state", {}))
        for group in optimizer.param_groups:
            group["lr"] = base_lr
        set_seed(seed + len(history))

    controller = build_controller(config, steps_per_epoch=len(train_loader), state=schedule_state)

    best_score: float | None = None
    best_path = output / "best.pt"
    last_path = output / "last.pt"
    start_epoch = len(history) + 1
    if resume_data is not None:
        print(json.dumps({"resumed_from": str(resume_from), "start_epoch": start_epoch, "global_step": global_step, "schedule_state": controller.state()}, sort_keys=True))

    stopped_early = False
    for epoch in range(start_epoch, config.epochs + 1):
        model.train()
        losses = []
        iterator = tqdm(train_loader, desc=f"epoch {epoch}/{config.epochs}", leave=False)
        for images, targets in iterator:
            images = images.to(device)
            targets = targets.to(device)
            factor = controller.lr_factor(global_step)
            for group in optimizer.param_groups:
                group["lr"] = base_lr * factor
            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            global_step += 1
            losses.append(float(loss.detach().cpu().item()))
            iterator.set_postfix(loss=np.mean(losses), lr=base_lr * factor)

        metrics = evaluate_model(model, test_loader, config, device)
        epoch_record: dict[str, Any] = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "lr": float(base_lr * controller.lr_factor(global_step)),
            **metrics,
        }
        if val_loader is not None:
            val_metrics = evaluate_model(model, val_loader, config, device)
            epoch_record.update({f"val_{k}": v for k, v in val_metrics.items()})
        epoch_record.update(controller.on_epoch_end(epoch, history + [epoch_record]))
        if val_loader is not None:
            score = _selection_score(val_metrics)
            if best_score is None or score > best_score:
                best_score = score
                save_checkpoint(best_path, model, optimizer, config, class_to_idx, history + [epoch_record], global_step, controller.state())
        history.append(epoch_record)
        save_checkpoint(last_path, model, optimizer, config, class_to_idx, history, global_step, controller.state())
        print(json.dumps(epoch_record, sort_keys=True))
        if controller.should_stop(epoch):
            stopped_early = True
            print(json.dumps({"schedule_event": f"cosine decay complete after epoch {epoch}; stopping"}, sort_keys=True))
            break

    result: dict[str, Any] = {
        "history": history,
        "last_checkpoint": str(last_path),
        "best_checkpoint": str(best_path if val_loader is not None else last_path),
        "class_to_idx": class_to_idx,
        "schedule_state": controller.state(),
        "stopped_early": stopped_early,
    }
    return result


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    config: ExperimentConfig,
    class_to_idx: dict[str, int],
    history: list[dict[str, Any]],
    global_step: int = 0,
    schedule_state: dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "experiment": config.__dict__,
            "class_to_idx": class_to_idx,
            "history": history,
            "global_step": global_step,
            "schedule_state": schedule_state or {},
        },
        path,
    )


def evaluate_checkpoint(
    checkpoint: str | Path,
    manifest: str | Path,
    *,
    split: str = "test",
    device_name: str | None = None,
    batch_size: int | None = None,
    num_workers: int = 0,
) -> dict[str, float]:
    checkpoint_data = torch.load(checkpoint, map_location="cpu")
    config = config_from_checkpoint(checkpoint_data["experiment"])
    if batch_size is not None:
        config = with_overrides(config, None, batch_size, None)
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    class_to_idx = checkpoint_data.get("class_to_idx", {})
    dataset = _eval_dataset(config, manifest, split, class_to_idx)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False, num_workers=num_workers)
    model = build_model(model_config(config, class_to_idx)).to(device)
    model.load_state_dict(checkpoint_data["model_state_dict"])
    metrics = evaluate_model(model, loader, config, device)
    print(json.dumps(metrics, sort_keys=True))
    return metrics
