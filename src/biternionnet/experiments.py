from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    task: str
    target_kind: str
    model_head: str
    loss: str
    input_size: tuple[int, int] = (46, 46)
    model_variant: str = "standard"
    output_dim: int | None = None
    epochs: int = 50
    batch_size: int = 100
    optimizer: str = "adadelta"
    lr: float = 1.0
    rho: float = 0.95
    eps: float = 1e-7
    kappa: float = 1.0
    class_flip_map: dict[str, str] | None = None
    exclude_label: str | None = None
    quantization_borders: tuple[float, ...] | None = None
    quantization_centres: tuple[float, ...] | None = None


CLASS_FLIPS_4 = {"front": "front", "back": "back", "left": "right", "right": "left", "background": "background"}
CLASS_FLIPS_6 = {"frnt": "frnt", "rear": "rear", "left": "rght", "rght": "left", "frlf": "frrg", "frrg": "frlf"}


_EXPERIMENTS: dict[str, ExperimentConfig] = {
    "hiit": ExperimentConfig("hiit", "classification", "classification", "classification", "cross_entropy", class_flip_map=CLASS_FLIPS_6),
    "hocoffee": ExperimentConfig("hocoffee", "classification", "classification", "classification", "cross_entropy", class_flip_map=CLASS_FLIPS_6),
    "hoc": ExperimentConfig("hoc", "classification", "classification", "classification", "cross_entropy", input_size=(123, 54), model_variant="hoc", class_flip_map=CLASS_FLIPS_4),
    "qmul": ExperimentConfig("qmul", "classification", "classification", "classification", "cross_entropy", class_flip_map=CLASS_FLIPS_4),
    "qmul-no-background": ExperimentConfig("qmul-no-background", "classification", "classification", "classification", "cross_entropy", class_flip_map=CLASS_FLIPS_4, exclude_label="background"),
    "idiap": ExperimentConfig("idiap", "pose_rad", "pose_rad", "linear", "l1", input_size=(68, 68), model_variant="idiap", output_dim=3),
    "caviar": ExperimentConfig("caviar", "angle_deg", "angle_deg", "linear", "l1", output_dim=1),
    "caviar-occluded": ExperimentConfig("caviar-occluded", "angle_deg", "angle_deg", "linear", "l1", output_dim=1),
    "towncentre-linreg": ExperimentConfig("towncentre-linreg", "angle_deg", "angle_deg", "linear", "l1", output_dim=1),
    "towncentre-linreg-rad": ExperimentConfig("towncentre-linreg-rad", "angle_deg", "angle_rad", "linear", "l1", output_dim=1),
    "towncentre-mod-mae": ExperimentConfig("towncentre-mod-mae", "angle_deg", "angle_deg", "linear", "modulo_mae", output_dim=1),
    "towncentre-vonmises": ExperimentConfig("towncentre-vonmises", "angle_deg", "angle_deg", "linear", "vonmises_deg", output_dim=1, kappa=1.0),
    "towncentre-biternion": ExperimentConfig("towncentre-biternion", "angle_deg", "biternion", "biternion", "cosine", output_dim=2),
    "towncentre-biternion-vonmises": ExperimentConfig("towncentre-biternion-vonmises", "angle_deg", "biternion", "biternion", "vonmises_biternion", output_dim=2, kappa=1.0),
    "smoke-classification": ExperimentConfig("smoke-classification", "classification", "classification", "classification", "cross_entropy", epochs=1, batch_size=2),
    "smoke-biternion": ExperimentConfig("smoke-biternion", "angle_deg", "biternion", "biternion", "cosine", output_dim=2, epochs=1, batch_size=2),
}


def _add_quantized_towncentre() -> None:
    specs = {
        "3": ([0, 120, 240, 361], [60, 180, 320]),
        "4x": ([315, 45, 135, 225, 315], [0, 90, 180, 270]),
        "4p": ([0, 90, 180, 270, 361], [45, 135, 225, 315]),
        "6x": ([330, 30, 90, 150, 210, 270, 330], [0, 60, 120, 180, 240, 300]),
        "8x": ([337.5, 22.5, 67.5, 112.5, 157.5, 202.5, 247.5, 292.5, 337.5], [0, 45, 90, 135, 180, 225, 270, 315]),
        "8p": ([0, 45, 90, 135, 180, 225, 270, 315, 361], [22.5, 67.5, 112.5, 157.5, 202.5, 247.5, 292.5, 337.5]),
        "10x": ([342, 18, 54, 90, 126, 162, 198, 234, 270, 306, 342], [0, 36, 72, 108, 144, 180, 216, 252, 288, 324]),
        "12x": ([345, 15, 45, 75, 105, 135, 165, 195, 225, 255, 285, 315, 345], [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]),
    }
    for name, (borders, centres) in specs.items():
        for suffix, target_kind, head, loss, output_dim in (
            ("softmax", "quantized_classification", "classification", "cross_entropy", len(centres)),
            ("linreg", "quantized_angle_deg", "linear", "l1", 1),
            ("linreg-vonmises", "quantized_angle_deg", "linear", "vonmises_deg", 1),
            ("biternion", "quantized_biternion", "biternion", "cosine", 2),
            ("biternion-vonmises", "quantized_biternion", "biternion", "vonmises_biternion", 2),
        ):
            key = f"towncentre-q{name}-{suffix}"
            _EXPERIMENTS[key] = ExperimentConfig(
                key,
                "angle_deg",
                target_kind,
                head,
                loss,
                output_dim=output_dim,
                quantization_borders=tuple(float(x) for x in borders),
                quantization_centres=tuple(float(x) for x in centres),
            )


_add_quantized_towncentre()


def list_experiments() -> list[str]:
    return sorted(_EXPERIMENTS)


def get_experiment(name: str) -> ExperimentConfig:
    try:
        return _EXPERIMENTS[name]
    except KeyError as exc:
        known = ", ".join(list_experiments())
        raise KeyError(f"Unknown experiment {name!r}. Known experiments: {known}") from exc


def with_overrides(config: ExperimentConfig, epochs: int | None, batch_size: int | None, lr: float | None) -> ExperimentConfig:
    changes = {}
    if epochs is not None:
        changes["epochs"] = epochs
    if batch_size is not None:
        changes["batch_size"] = batch_size
    if lr is not None:
        changes["lr"] = lr
    return replace(config, **changes)


def quantization_arrays(config: ExperimentConfig) -> tuple[np.ndarray | None, np.ndarray | None]:
    borders = None if config.quantization_borders is None else np.array(config.quantization_borders, dtype=np.float32)
    centres = None if config.quantization_centres is None else np.array(config.quantization_centres, dtype=np.float32)
    return borders, centres

