"""Learning-rate controllers applied on top of the optimizer's base ``lr``.

All controllers expose the same small interface used by ``train.train_model``:

- ``lr_factor(global_step)``: multiplier for the base lr before optimizer step ``global_step``.
- ``on_epoch_end(epoch, history)``: inspect the metric history (including the epoch that just
  finished) and return extra fields for the epoch log; may change the controller state.
- ``should_stop(epoch)``: whether training is complete after ``epoch`` (before ``config.epochs``).
- ``state()``: JSON-serialisable state stored in checkpoints so a run can be resumed.
"""

from __future__ import annotations

import math
from typing import Any

from .experiments import ExperimentConfig


def wsd_lr_factor(step: int, total_steps: int, warmup_fraction: float, decay_fraction: float, final_lr_ratio: float) -> float:
    """Warmup-Stable-Decay multiplier for optimizer step ``step`` (0-based).

    Linear warmup from ``1/warmup_steps`` to 1, constant 1 during the stable phase, then linear
    decay reaching ``final_lr_ratio`` exactly on the last step. Steps past ``total_steps`` keep
    ``final_lr_ratio``.
    """
    warmup_steps = int(round(total_steps * warmup_fraction))
    decay_steps = int(round(total_steps * decay_fraction))
    decay_start = total_steps - decay_steps
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    if step >= total_steps:
        return final_lr_ratio
    if step >= decay_start and decay_steps > 0:
        progress = (step - decay_start + 1) / decay_steps
        return 1.0 - (1.0 - final_lr_ratio) * progress
    return 1.0


def plateau_rate(train_losses: list[float], window: int) -> float | None:
    """Relative loss decrease over the last ``window`` epochs.

    Compares the mean loss of the first half of the window with the mean of the second half:
    ``(mean(first) - mean(second)) / mean(first)``. Returns ``None`` until ``window`` epochs exist.
    A value of 0.02 means the loss fell by 2% between the two halves; negative means it rose.
    """
    if window < 2 or len(train_losses) < window:
        return None
    recent = train_losses[-window:]
    half = window // 2
    first = recent[: window - half]
    second = recent[window - half :]
    first_mean = sum(first) / len(first)
    second_mean = sum(second) / len(second)
    if first_mean == 0.0:
        return 0.0
    return (first_mean - second_mean) / first_mean


class ConstantController:
    def lr_factor(self, step: int) -> float:
        return 1.0

    def on_epoch_end(self, epoch: int, history: list[dict[str, Any]]) -> dict[str, Any]:
        return {}

    def should_stop(self, epoch: int) -> bool:
        return False

    def state(self) -> dict[str, Any]:
        return {}


class WsdController(ConstantController):
    def __init__(self, config: ExperimentConfig, steps_per_epoch: int) -> None:
        self.config = config
        self.total_steps = config.epochs * steps_per_epoch

    def lr_factor(self, step: int) -> float:
        return wsd_lr_factor(step, self.total_steps, self.config.warmup_fraction, self.config.decay_fraction, self.config.final_lr_ratio)


class PlateauCosineController(ConstantController):
    """Constant lr until the training loss plateaus, then cosine decay over ``cosine_epochs``.

    The decay is triggered (decay begins with the *next* epoch) by the first of:

    - ``config.decay_start_epoch`` (manual choice, e.g. after inspecting the logs),
    - ``plateau_rate`` over ``config.plateau_window`` epochs dropping below
      ``config.plateau_threshold`` once ``epoch >= config.plateau_min_epochs``
      (disabled when ``plateau_threshold`` is ``None``),
    - the epoch budget: decay is forced to start at ``config.epochs - cosine_epochs + 1`` so
      it always completes within ``config.epochs``.

    Training stops once the cosine phase has run for ``cosine_epochs`` epochs.
    """

    def __init__(self, config: ExperimentConfig, steps_per_epoch: int, state: dict[str, Any] | None = None) -> None:
        if config.cosine_epochs < 1:
            raise ValueError("cosine_epochs must be >= 1")
        self.config = config
        self.steps_per_epoch = steps_per_epoch
        state = dict(state or {})
        self.decay_start_epoch: int | None = state.get("decay_start_epoch")
        self.trigger_reason: str | None = state.get("trigger_reason")
        if config.decay_start_epoch is not None and self.decay_start_epoch is None:
            self.decay_start_epoch = int(config.decay_start_epoch)
            self.trigger_reason = "manual"

    # -- lr -----------------------------------------------------------------
    def lr_factor(self, step: int) -> float:
        if self.decay_start_epoch is None:
            return 1.0
        first_decay_step = (self.decay_start_epoch - 1) * self.steps_per_epoch
        if step < first_decay_step:
            return 1.0
        total_decay_steps = self.config.cosine_epochs * self.steps_per_epoch
        progress = min(1.0, (step - first_decay_step + 1) / total_decay_steps)
        final = self.config.final_lr_ratio
        return final + (1.0 - final) * 0.5 * (1.0 + math.cos(math.pi * progress))

    # -- epoch bookkeeping --------------------------------------------------
    def phase(self, epoch: int) -> str:
        if self.decay_start_epoch is not None and epoch >= self.decay_start_epoch:
            return "cosine"
        return "constant"

    def on_epoch_end(self, epoch: int, history: list[dict[str, Any]]) -> dict[str, Any]:
        losses = [float(record["train_loss"]) for record in history]
        rate = plateau_rate(losses, self.config.plateau_window)
        record: dict[str, Any] = {"phase": self.phase(epoch), "plateau_rate": rate}

        if self.decay_start_epoch is None:
            forced_start = self.config.epochs - self.config.cosine_epochs + 1
            auto = (
                self.config.plateau_threshold is not None
                and epoch >= self.config.plateau_min_epochs
                and rate is not None
                and rate < self.config.plateau_threshold
            )
            if auto:
                self.decay_start_epoch = epoch + 1
                self.trigger_reason = "plateau"
            elif epoch + 1 >= forced_start:
                self.decay_start_epoch = max(epoch + 1, forced_start)
                self.trigger_reason = "budget"
            if self.decay_start_epoch is not None:
                record["schedule_event"] = (
                    f"cosine decay starts at epoch {self.decay_start_epoch} ({self.trigger_reason}); "
                    f"ends after epoch {self.decay_start_epoch + self.config.cosine_epochs - 1}"
                )
        if self.decay_start_epoch is not None:
            record["decay_start_epoch"] = self.decay_start_epoch
        return record

    def should_stop(self, epoch: int) -> bool:
        return self.decay_start_epoch is not None and epoch >= self.decay_start_epoch + self.config.cosine_epochs - 1

    def state(self) -> dict[str, Any]:
        return {"decay_start_epoch": self.decay_start_epoch, "trigger_reason": self.trigger_reason}


def build_controller(config: ExperimentConfig, steps_per_epoch: int, state: dict[str, Any] | None = None) -> ConstantController:
    if config.lr_schedule == "constant":
        return ConstantController()
    if config.lr_schedule == "wsd":
        return WsdController(config, steps_per_epoch)
    if config.lr_schedule == "plateau_cosine":
        return PlateauCosineController(config, steps_per_epoch, state)
    raise ValueError(f"Unsupported lr_schedule: {config.lr_schedule}")
