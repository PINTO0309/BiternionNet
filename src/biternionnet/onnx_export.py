from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import onnx
import torch
from onnxsim import simplify

from .experiments import ExperimentConfig
from .models import ModelConfig, build_model


def _output_dim(config: ExperimentConfig, class_to_idx: dict[str, int]) -> int:
    if config.output_dim is not None:
        return config.output_dim
    if config.target_kind in {"classification", "quantized_classification"}:
        return len(class_to_idx)
    raise ValueError(f"Checkpoint experiment {config.name!r} does not define output_dim")


def load_checkpoint_model(checkpoint: str | Path, device: torch.device) -> tuple[torch.nn.Module, ExperimentConfig, dict[str, int], dict[str, Any]]:
    checkpoint_data = torch.load(checkpoint, map_location=device)
    config = ExperimentConfig(**checkpoint_data["experiment"])
    class_to_idx = checkpoint_data.get("class_to_idx", {})
    model = build_model(
        ModelConfig(
            output_dim=_output_dim(config, class_to_idx),
            head=config.model_head,
            variant=config.model_variant,
            input_size=config.input_size,
            backbone_activation=config.backbone_activation,
        )
    ).to(device)
    model.load_state_dict(checkpoint_data["model_state_dict"])
    model.eval()
    return model, config, class_to_idx, checkpoint_data


def _add_metadata(path: Path, metadata: dict[str, str]) -> None:
    model = onnx.load(path)
    existing = {item.key: item for item in model.metadata_props}
    for key, value in metadata.items():
        item = existing.get(key)
        if item is None:
            item = model.metadata_props.add()
            item.key = key
        item.value = value
    onnx.save(model, path)


def _check_onnx(path: Path) -> None:
    model = onnx.load(path)
    onnx.checker.check_model(model)


def _simplify_onnx(source: Path, destination: Path, input_shape: tuple[int, int, int, int], dynamic_batch: bool) -> None:
    model, ok = simplify(
        str(source),
        test_input_shapes={"input": list(input_shape)},
    )
    if not ok:
        raise RuntimeError(f"onnxsim validation failed for {source}")
    onnx.save(model, destination)
    _check_onnx(destination)


def _export_one(
    model: torch.nn.Module,
    dummy_input: torch.Tensor,
    output_path: Path,
    opset: int,
    dynamic_batch: bool,
) -> None:
    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {"input": {0: "batch"}, "output": {0: "batch"}}
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
        dynamo=False,
    )
    _check_onnx(output_path)


def export_checkpoint_to_onnx(
    checkpoint: str | Path,
    output_dir: str | Path,
    *,
    prefix: str | None = None,
    opset: int = 17,
    batch_size: int = 1,
    device_name: str = "cpu",
    simplify_models: bool = True,
) -> dict[str, str]:
    checkpoint = Path(checkpoint)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(device_name)
    model, config, class_to_idx, checkpoint_data = load_checkpoint_model(checkpoint, device)
    height, width = config.input_size
    dummy_input = torch.zeros(batch_size, 3, height, width, dtype=torch.float32, device=device)
    prefix = prefix or checkpoint.parent.name or checkpoint.stem

    static_path = output_dir / f"{prefix}_static_opset{opset}.onnx"
    dynamic_path = output_dir / f"{prefix}_dynamic_batch_opset{opset}.onnx"
    static_simplified_path = output_dir / f"{prefix}_static_opset{opset}_sim.onnx"
    dynamic_simplified_path = output_dir / f"{prefix}_dynamic_batch_opset{opset}_sim.onnx"

    metadata = {
        "checkpoint": str(checkpoint),
        "experiment": config.name,
        "input_size": json.dumps(config.input_size),
        "backbone_activation": config.backbone_activation,
        "class_to_idx": json.dumps(class_to_idx, sort_keys=True),
        "history_length": str(len(checkpoint_data.get("history", []))),
    }

    _export_one(model, dummy_input, static_path, opset, dynamic_batch=False)
    _add_metadata(static_path, metadata | {"input_shape_mode": "static"})

    _export_one(model, dummy_input, dynamic_path, opset, dynamic_batch=True)
    _add_metadata(dynamic_path, metadata | {"input_shape_mode": "dynamic_batch"})

    outputs = {
        "static": str(static_path),
        "dynamic_batch": str(dynamic_path),
    }
    if simplify_models:
        input_shape = tuple(dummy_input.shape)
        _simplify_onnx(static_path, static_simplified_path, input_shape, dynamic_batch=False)
        _add_metadata(static_simplified_path, metadata | {"input_shape_mode": "static", "onnxsim": "true"})
        _simplify_onnx(dynamic_path, dynamic_simplified_path, input_shape, dynamic_batch=True)
        _add_metadata(dynamic_simplified_path, metadata | {"input_shape_mode": "dynamic_batch", "onnxsim": "true"})
        outputs["static_simplified"] = str(static_simplified_path)
        outputs["dynamic_batch_simplified"] = str(dynamic_simplified_path)
    return outputs
