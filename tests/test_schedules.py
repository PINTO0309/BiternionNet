import math

import pytest

from biternionnet.experiments import get_experiment, with_overrides
from biternionnet.schedules import PlateauCosineController, plateau_rate


def _config(**kwargs):
    base = get_experiment("smoke-biternion")
    return with_overrides(base, kwargs.pop("epochs", 50), None, None, lr_schedule="plateau_cosine", **kwargs)


def _run(controller, epochs, losses):
    """Feed per-epoch losses to the controller, return the list of epoch log dicts."""
    history, logs = [], []
    for epoch, loss in zip(range(1, epochs + 1), losses):
        record = {"epoch": epoch, "train_loss": loss}
        info = controller.on_epoch_end(epoch, history + [record])
        record.update(info)
        history.append(record)
        logs.append(record)
        if controller.should_stop(epoch):
            break
    return logs


def test_plateau_rate_compares_window_halves():
    assert plateau_rate([1.0, 1.0, 1.0], 4) is None
    # first half mean 1.0, second half mean 0.9 -> 10% decrease
    assert math.isclose(plateau_rate([1.0, 1.0, 0.9, 0.9], 4), 0.1)
    assert plateau_rate([1.0, 1.0, 1.1, 1.1], 4) < 0.0
    assert plateau_rate([5.0, 1.0, 1.0, 1.0, 1.0, 1.0], 4) == 0.0  # only the last window counts


def test_auto_trigger_after_plateau_then_cosine_and_stop():
    config = _config(plateau_window=4, plateau_threshold=0.05, plateau_min_epochs=4, cosine_epochs=3, epochs=50)
    controller = PlateauCosineController(config, steps_per_epoch=10)
    # Loss drops quickly, then flattens from epoch 5 on.
    losses = [1.0, 0.8, 0.6, 0.4, 0.39, 0.385, 0.38, 0.379, 0.378, 0.377, 0.376]
    logs = _run(controller, 50, losses)
    trigger = next(r for r in logs if "schedule_event" in r)
    assert controller.trigger_reason == "plateau"
    assert trigger["phase"] == "constant"
    assert controller.decay_start_epoch == trigger["epoch"] + 1
    assert logs[-1]["epoch"] == controller.decay_start_epoch + 2  # 3 cosine epochs, then stop
    assert [r["phase"] for r in logs[-3:]] == ["cosine"] * 3
    # lr: 1.0 before decay, cosine down to final ratio on the last decay step.
    first_decay_step = (controller.decay_start_epoch - 1) * 10
    assert controller.lr_factor(first_decay_step - 1) == 1.0
    assert controller.lr_factor(first_decay_step) < 1.0
    assert math.isclose(controller.lr_factor(first_decay_step + 30 - 1), config.final_lr_ratio, abs_tol=1e-9)
    assert controller.lr_factor(first_decay_step + 100) == config.final_lr_ratio


def test_budget_forces_decay_to_fit_in_epochs():
    config = _config(plateau_threshold=None, cosine_epochs=5, epochs=20)
    controller = PlateauCosineController(config, steps_per_epoch=1)
    logs = _run(controller, 20, [1.0 / (i + 1) for i in range(20)])  # keeps improving
    assert controller.trigger_reason == "budget"
    assert controller.decay_start_epoch == 16
    assert logs[-1]["epoch"] == 20
    assert [r["phase"] for r in logs[15:]] == ["cosine"] * 5


def test_manual_decay_start_epoch_and_state_roundtrip():
    config = _config(decay_start_epoch=7, cosine_epochs=2, plateau_threshold=0.5, epochs=50)
    controller = PlateauCosineController(config, steps_per_epoch=1)
    assert controller.trigger_reason == "manual"
    logs = _run(controller, 50, [1.0] * 50)  # flat loss would auto-trigger, manual wins
    assert [r["phase"] for r in logs] == ["constant"] * 6 + ["cosine"] * 2
    assert logs[-1]["epoch"] == 8

    resumed = PlateauCosineController(_config(cosine_epochs=2, epochs=50), steps_per_epoch=1, state=controller.state())
    assert resumed.decay_start_epoch == 7 and resumed.trigger_reason == "manual"
    assert resumed.should_stop(8)


def test_with_overrides_validates_plateau_settings():
    with pytest.raises(ValueError):
        _config(plateau_window=1)
    with pytest.raises(ValueError):
        _config(cosine_epochs=0)
    assert _config(disable_plateau_trigger=True).plateau_threshold is None
