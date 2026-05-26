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
from .experiments import ExperimentConfig, get_experiment, quantization_arrays, with_overrides
from .losses import (
    CosineLoss,
    ModuloMAELoss,
    VonMisesBiternionLoss,
    VonMisesLoss,
    angle_difference_deg,
    bit2deg,
    cyclic_mae_deg,
)
from .models import ModelConfig, build_model


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


def build_datasets(
    config: ExperimentConfig,
    manifest: str | Path,
    train_flip_probability: float,
) -> tuple[ManifestDataset, ManifestDataset, dict[str, int]]:
    records = read_manifest(manifest)
    class_to_idx = build_class_mapping(records, exclude_label=config.exclude_label) if config.target_kind == "classification" else {}
    borders, centres = quantization_arrays(config)
    train_dataset = ManifestDataset(
        manifest,
        "train",
        config.target_kind,
        crop=CropConfig(config.input_size, random_crop=True),
        class_to_idx=class_to_idx,
        class_flip_map=config.class_flip_map,
        exclude_label=config.exclude_label,
        flip_probability=train_flip_probability,
        quantization_borders=borders,
        quantization_centres=centres,
    )
    test_dataset = ManifestDataset(
        manifest,
        "test",
        config.target_kind,
        crop=CropConfig(config.input_size, random_crop=False),
        class_to_idx=class_to_idx,
        exclude_label=config.exclude_label,
        flip_probability=0.0,
        quantization_borders=borders,
        quantization_centres=centres,
    )
    return train_dataset, test_dataset, class_to_idx


def _prepare_target_for_metric(config: ExperimentConfig, target: torch.Tensor) -> torch.Tensor:
    if config.target_kind == "angle_rad":
        return torch.rad2deg(target)
    return target


@torch.no_grad()
def evaluate_model(model: nn.Module, loader: DataLoader, config: ExperimentConfig, device: torch.device) -> dict[str, float]:
    model.eval()
    correct = 0
    total = 0
    angle_errors: list[torch.Tensor] = []
    pose_errors: list[torch.Tensor] = []

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)
        outputs = model(images)
        if config.target_kind in {"classification", "quantized_classification"}:
            predicted = outputs.argmax(dim=1)
            correct += int((predicted == targets).sum().item())
            total += int(targets.numel())
        elif config.model_head == "biternion":
            pred_deg = bit2deg(outputs).reshape(-1, 1)
            if config.target_kind == "biternion":
                target_deg = bit2deg(targets).reshape(-1, 1)
            else:
                target_deg = _prepare_target_for_metric(config, targets)
            angle_errors.append(angle_difference_deg(pred_deg, target_deg).detach().cpu())
        elif config.target_kind == "pose_rad":
            err = torch.rad2deg(torch.abs(torch.atan2(torch.sin(targets - outputs), torch.cos(targets - outputs))))
            pose_errors.append(err.detach().cpu())
        else:
            pred_deg = outputs
            target_deg = _prepare_target_for_metric(config, targets)
            angle_errors.append(angle_difference_deg(pred_deg, target_deg).detach().cpu())

    if total:
        return {"accuracy": correct / total}
    if pose_errors:
        errors = torch.cat(pose_errors, dim=0)
        return {
            "maad_deg": float(errors.mean().item()),
            "pan_maad_deg": float(errors[:, 0].mean().item()),
            "tilt_maad_deg": float(errors[:, 1].mean().item()),
            "roll_maad_deg": float(errors[:, 2].mean().item()),
        }
    errors = torch.cat(angle_errors, dim=0)
    return {"maad_deg": float(errors.mean().item())}


def train_model(
    experiment: str,
    manifest: str | Path,
    output: str | Path,
    *,
    epochs: int | None = None,
    batch_size: int | None = None,
    lr: float | None = None,
    backbone_activation: str | None = None,
    seed: int = 0,
    device_name: str | None = None,
    num_workers: int = 0,
    train_flip_probability: float = 0.5,
) -> dict[str, Any]:
    set_seed(seed)
    config = with_overrides(get_experiment(experiment), epochs, batch_size, lr, backbone_activation)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))

    train_dataset, test_dataset, class_to_idx = build_datasets(config, manifest, train_flip_probability)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=num_workers)

    model = build_model(
        ModelConfig(
            output_dim=_output_dim(config, class_to_idx),
            head=config.model_head,
            variant=config.model_variant,
            input_size=config.input_size,
            backbone_activation=config.backbone_activation,
        )
    ).to(device)
    criterion = _loss(config)
    optimizer = _optimizer(config, model)

    history: list[dict[str, Any]] = []
    best_score: float | None = None
    best_path = output / "best.pt"

    for epoch in range(1, config.epochs + 1):
        model.train()
        losses = []
        iterator = tqdm(train_loader, desc=f"epoch {epoch}/{config.epochs}", leave=False)
        for images, targets in iterator:
            images = images.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
            iterator.set_postfix(loss=np.mean(losses))

        metrics = evaluate_model(model, test_loader, config, device)
        epoch_record = {"epoch": epoch, "train_loss": float(np.mean(losses)), **metrics}
        history.append(epoch_record)
        score = metrics.get("accuracy", -metrics.get("maad_deg", float("inf")))
        if best_score is None or score > best_score:
            best_score = score
            save_checkpoint(best_path, model, optimizer, config, class_to_idx, history)
        save_checkpoint(output / "last.pt", model, optimizer, config, class_to_idx, history)
        print(json.dumps(epoch_record, sort_keys=True))

    return {"history": history, "best_checkpoint": str(best_path), "class_to_idx": class_to_idx}


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    config: ExperimentConfig,
    class_to_idx: dict[str, int],
    history: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "experiment": config.__dict__,
            "class_to_idx": class_to_idx,
            "history": history,
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
    config = ExperimentConfig(**checkpoint_data["experiment"])
    if batch_size is not None:
        config = with_overrides(config, None, batch_size, None)
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    class_to_idx = checkpoint_data.get("class_to_idx", {})
    borders, centres = quantization_arrays(config)
    dataset = ManifestDataset(
        manifest,
        split,
        config.target_kind,
        crop=CropConfig(config.input_size, random_crop=False),
        class_to_idx=class_to_idx,
        exclude_label=config.exclude_label,
        quantization_borders=borders,
        quantization_centres=centres,
    )
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False, num_workers=num_workers)
    model = build_model(
        ModelConfig(
            _output_dim(config, class_to_idx),
            config.model_head,
            config.model_variant,
            config.input_size,
            config.backbone_activation,
        )
    ).to(device)
    model.load_state_dict(checkpoint_data["model_state_dict"])
    metrics = evaluate_model(model, loader, config, device)
    print(json.dumps(metrics, sort_keys=True))
    return metrics
