from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .augment import get_photometric_preset


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    task: str
    target_kind: str
    model_head: str
    loss: str
    input_size: tuple[int, int] = (46, 46)
    model_variant: str = "standard"
    backbone_activation: str = "relu"
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
    # Resize every image to this (H, W) before cropping, like ``prepare_data.scale_all``.
    resize_size: tuple[int, int] | None = None
    # Whether horizontal-flip augmentation is valid for this experiment.
    flip_augmentation: bool = True
    # Theano/DeepFried2 pooling rounds up (ignore_border=False); PyTorch floors by default.
    pool_ceil_mode: bool = True
    # Learning-rate schedule applied per optimizer step:
    #   "constant"       - paper setting (fixed AdaDelta step for ``epochs`` epochs)
    #   "wsd"            - warmup -> stable -> linear decay to ``lr * final_lr_ratio``
    #   "plateau_cosine" - constant until the train loss plateaus (or ``decay_start_epoch``),
    #                      then cosine decay over ``cosine_epochs`` epochs to ``lr * final_lr_ratio``
    lr_schedule: str = "constant"
    warmup_fraction: float = 0.05
    decay_fraction: float = 0.3
    final_lr_ratio: float = 0.1
    # plateau_cosine: window (epochs) and relative-decrease threshold for the plateau detector,
    # earliest epoch at which it may fire, cosine phase length, and optional manual start epoch.
    plateau_window: int = 10
    plateau_threshold: float | None = 0.02
    plateau_min_epochs: int = 10
    cosine_epochs: int = 15
    decay_start_epoch: int | None = None
    # Data augmentation beyond the paper's random crop + flip (opt-in ablations):
    # photometric preset name (see augment.PHOTOMETRIC_PRESETS) and multiplicative jitter of
    # ``resize_size`` before the random crop.
    photometric: str | None = None
    scale_jitter: tuple[float, float] | None = None
    # Uniform +-J degrees added online to the angle label of neighbour-frame records during
    # training (fresh draw every access). Neighbour labels are copies of their anchor's label,
    # so tens of records share one integer degree; a small jitter (~the measured 0.14 deg/frame
    # head-turn rate times the offset) spreads those 1-deg spikes without changing the bin.
    neighbor_label_jitter_deg: float = 0.0


CLASS_FLIPS_4 = {"front": "front", "back": "back", "left": "right", "right": "left", "background": "background"}
CLASS_FLIPS_6 = {"frnt": "frnt", "rear": "rear", "left": "rght", "rght": "left", "frlf": "frrg", "frrg": "frlf"}
TOWNCENTRE_RESIZE = (50, 50)


_EXPERIMENTS: dict[str, ExperimentConfig] = {
    "hiit": ExperimentConfig("hiit", "classification", "classification", "classification", "cross_entropy", class_flip_map=CLASS_FLIPS_6),
    "hocoffee": ExperimentConfig("hocoffee", "classification", "classification", "classification", "cross_entropy", class_flip_map=CLASS_FLIPS_6),
    "hoc": ExperimentConfig("hoc", "classification", "classification", "classification", "cross_entropy", input_size=(123, 54), model_variant="hoc", class_flip_map=CLASS_FLIPS_4),
    "qmul": ExperimentConfig("qmul", "classification", "classification", "classification", "cross_entropy", class_flip_map=CLASS_FLIPS_4),
    "qmul-no-background": ExperimentConfig("qmul-no-background", "classification", "classification", "classification", "cross_entropy", class_flip_map=CLASS_FLIPS_4, exclude_label="background"),
    # IDIAP and CAVIAR were not flip-augmented in the original notebooks (IDIAP is already
    # mirrored, CAVIAR has no augmentation), and there is no label flip rule for pan/tilt/roll.
    "idiap": ExperimentConfig("idiap", "pose_rad", "pose_rad", "linear", "l1", input_size=(68, 68), model_variant="idiap", output_dim=3, flip_augmentation=False),
    "caviar": ExperimentConfig("caviar", "angle_deg", "angle_deg", "linear", "l1", output_dim=1, flip_augmentation=False),
    "caviar-occluded": ExperimentConfig("caviar-occluded", "angle_deg", "angle_deg", "linear", "l1", output_dim=1, flip_augmentation=False),
    # TownCentre head crops are tiny (~25x23); the notebooks rescale them to 50x50 first and random-crop 46x46.
    "towncentre-linreg": ExperimentConfig("towncentre-linreg", "angle_deg", "angle_deg", "linear", "l1", output_dim=1, resize_size=TOWNCENTRE_RESIZE),
    "towncentre-linreg-rad": ExperimentConfig("towncentre-linreg-rad", "angle_deg", "angle_rad", "linear", "l1", output_dim=1, resize_size=TOWNCENTRE_RESIZE),
    "towncentre-mod-mae": ExperimentConfig("towncentre-mod-mae", "angle_deg", "angle_deg", "linear", "modulo_mae", output_dim=1, resize_size=TOWNCENTRE_RESIZE),
    # The notebook uses VonMisesCriterion(0.5, radians=False) for this experiment.
    "towncentre-vonmises": ExperimentConfig("towncentre-vonmises", "angle_deg", "angle_deg", "linear", "vonmises_deg", output_dim=1, kappa=0.5, resize_size=TOWNCENTRE_RESIZE),
    "towncentre-biternion": ExperimentConfig("towncentre-biternion", "angle_deg", "biternion", "biternion", "cosine", output_dim=2, resize_size=TOWNCENTRE_RESIZE),
    "towncentre-biternion-vonmises": ExperimentConfig("towncentre-biternion-vonmises", "angle_deg", "biternion", "biternion", "vonmises_biternion", output_dim=2, kappa=1.0, resize_size=TOWNCENTRE_RESIZE),
    # Long schedule chosen from the 1000-epoch constant-lr sweep (runs/tc-btvm-c1000): test MAAD keeps
    # improving slowly up to ~1000 epochs without overfitting, so train 1000 epochs at the paper's
    # constant AdaDelta step and then cosine-decay for 100 epochs. Not a paper setting.
    "towncentre-biternion-vonmises-long": ExperimentConfig(
        "towncentre-biternion-vonmises-long",
        "angle_deg",
        "biternion",
        "biternion",
        "vonmises_biternion",
        output_dim=2,
        kappa=1.0,
        resize_size=TOWNCENTRE_RESIZE,
        epochs=1100,
        lr_schedule="plateau_cosine",
        plateau_threshold=None,
        decay_start_epoch=1001,
        cosine_epochs=100,
    ),
    "towncentre-biternion-long": ExperimentConfig(
        "towncentre-biternion-long",
        "angle_deg",
        "biternion",
        "biternion",
        "cosine",
        output_dim=2,
        resize_size=TOWNCENTRE_RESIZE,
        epochs=1100,
        lr_schedule="plateau_cosine",
        plateau_threshold=None,
        decay_start_epoch=1001,
        cosine_epochs=100,
    ),
    # Default configuration adopted on 2026-08-30 (history/004 §3.2): neighbour-frame manifest
    # (manifest_nb3.jsonl, 269 steps/epoch) + photometric "cctv" + scale jitter, 300 constant epochs
    # then a 50-epoch cosine decay (~94k steps). Chosen because the dataset is small and noisy; dataset
    # reinforcement is worked on before re-tuning the augmentation strength.
    "towncentre-biternion-vonmises-aug": ExperimentConfig(
        "towncentre-biternion-vonmises-aug",
        "angle_deg",
        "biternion",
        "biternion",
        "vonmises_biternion",
        output_dim=2,
        kappa=1.0,
        resize_size=TOWNCENTRE_RESIZE,
        epochs=350,
        lr_schedule="plateau_cosine",
        plateau_threshold=None,
        decay_start_epoch=301,
        cosine_epochs=50,
        photometric="cctv",
        scale_jitter=(0.9, 1.1),
    ),
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
                resize_size=TOWNCENTRE_RESIZE,
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


LR_SCHEDULES = ("constant", "wsd", "plateau_cosine")


def with_overrides(
    config: ExperimentConfig,
    epochs: int | None,
    batch_size: int | None,
    lr: float | None,
    backbone_activation: str | None = None,
    *,
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
    photometric: str | None = None,
    scale_jitter: tuple[float, float] | None = None,
    resize_size: tuple[int, int] | None = None,
    input_size: tuple[int, int] | None = None,
    neighbor_label_jitter_deg: float | None = None,
) -> ExperimentConfig:
    changes = {}
    if epochs is not None:
        changes["epochs"] = epochs
    if batch_size is not None:
        changes["batch_size"] = batch_size
    if lr is not None:
        changes["lr"] = lr
    if backbone_activation is not None:
        if backbone_activation not in {"relu", "swish"}:
            raise ValueError("backbone_activation must be 'relu' or 'swish'")
        changes["backbone_activation"] = backbone_activation
    if lr_schedule is not None:
        if lr_schedule not in LR_SCHEDULES:
            raise ValueError(f"lr_schedule must be one of {LR_SCHEDULES}")
        changes["lr_schedule"] = lr_schedule
    if warmup_fraction is not None:
        changes["warmup_fraction"] = warmup_fraction
    if decay_fraction is not None:
        changes["decay_fraction"] = decay_fraction
    if final_lr_ratio is not None:
        changes["final_lr_ratio"] = final_lr_ratio
    if plateau_window is not None:
        changes["plateau_window"] = plateau_window
    if plateau_threshold is not None:
        changes["plateau_threshold"] = plateau_threshold
    if disable_plateau_trigger:
        changes["plateau_threshold"] = None
    if plateau_min_epochs is not None:
        changes["plateau_min_epochs"] = plateau_min_epochs
    if cosine_epochs is not None:
        changes["cosine_epochs"] = cosine_epochs
    if decay_start_epoch is not None:
        changes["decay_start_epoch"] = decay_start_epoch
    if photometric is not None:
        changes["photometric"] = None if photometric == "none" else photometric
    if scale_jitter is not None:
        changes["scale_jitter"] = tuple(float(x) for x in scale_jitter)
    if resize_size is not None:
        changes["resize_size"] = tuple(int(x) for x in resize_size)
    if input_size is not None:
        changes["input_size"] = tuple(int(x) for x in input_size)
    if neighbor_label_jitter_deg is not None:
        if neighbor_label_jitter_deg < 0:
            raise ValueError("neighbor_label_jitter_deg must be >= 0")
        changes["neighbor_label_jitter_deg"] = float(neighbor_label_jitter_deg)
    result = replace(config, **changes)
    get_photometric_preset(result.photometric)  # validates the preset name
    if result.scale_jitter is not None:
        lo, hi = result.scale_jitter
        if not 0.0 < lo <= hi:
            raise ValueError("scale_jitter must satisfy 0 < low <= high")
        if result.resize_size is None:
            raise ValueError("scale_jitter requires a preset with resize_size")
    if result.resize_size is not None and (result.resize_size[0] < result.input_size[0] or result.resize_size[1] < result.input_size[1]):
        raise ValueError("resize_size must be >= input_size in both dimensions")
    if result.warmup_fraction < 0.0 or result.decay_fraction < 0.0:
        raise ValueError("warmup_fraction and decay_fraction must be non-negative")
    if result.warmup_fraction + result.decay_fraction > 1.0:
        raise ValueError("warmup_fraction + decay_fraction must not exceed 1.0")
    if not 0.0 <= result.final_lr_ratio <= 1.0:
        raise ValueError("final_lr_ratio must be in [0, 1]")
    if result.plateau_window < 2:
        raise ValueError("plateau_window must be >= 2")
    if result.cosine_epochs < 1:
        raise ValueError("cosine_epochs must be >= 1")
    if result.decay_start_epoch is not None and result.decay_start_epoch < 1:
        raise ValueError("decay_start_epoch must be >= 1")
    return result


def config_from_checkpoint(data: dict) -> ExperimentConfig:
    """Rebuild an ExperimentConfig from a checkpoint's ``experiment`` dict.

    Checkpoints written before ``pool_ceil_mode`` existed were trained with floor pooling
    (4x4 feature maps on 46x46 inputs); keep them loadable by defaulting to that behaviour.
    """
    data = dict(data)
    data.setdefault("pool_ceil_mode", False)
    for key in ("input_size", "resize_size", "quantization_borders", "quantization_centres", "scale_jitter"):
        if data.get(key) is not None:
            data[key] = tuple(data[key])
    return ExperimentConfig(**data)


def evaluation_target_kind(config: ExperimentConfig) -> str:
    """Target kind used for evaluation: quantized experiments are scored against the continuous angle."""
    if config.target_kind.startswith("quantized_"):
        return "angle_deg"
    return config.target_kind


def quantization_arrays(config: ExperimentConfig) -> tuple[np.ndarray | None, np.ndarray | None]:
    borders = None if config.quantization_borders is None else np.array(config.quantization_borders, dtype=np.float32)
    centres = None if config.quantization_centres is None else np.array(config.quantization_centres, dtype=np.float32)
    return borders, centres
