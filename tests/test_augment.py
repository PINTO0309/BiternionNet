import numpy as np
import pytest

from biternionnet.augment import PHOTOMETRIC_PRESETS, PhotometricConfig, apply_photometric, get_photometric_preset, motion_blur_kernel, scaled


def _image(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(0.2, 0.8, (50, 50, 3)).astype(np.float32)


def test_presets_and_lookup():
    assert get_photometric_preset(None) is None
    assert get_photometric_preset("cctv") is PHOTOMETRIC_PRESETS["cctv"]
    assert get_photometric_preset("none").is_noop()
    with pytest.raises(ValueError):
        get_photometric_preset("nope")


def test_apply_photometric_shape_range_dtype_and_determinism():
    image = _image()
    config = PHOTOMETRIC_PRESETS["cctv"]
    outs = [apply_photometric(image, config, np.random.default_rng(7)) for _ in range(2)]
    assert np.array_equal(outs[0], outs[1])
    out = outs[0]
    assert out.shape == image.shape and out.dtype == np.float32
    assert out.min() >= 0.0 and out.max() <= 1.0
    # different seeds differ, and the input is not modified in place
    assert not np.array_equal(out, apply_photometric(image, config, np.random.default_rng(8)))
    assert np.array_equal(image, _image())


def test_noop_config_is_identity():
    image = _image()
    out = apply_photometric(image, PHOTOMETRIC_PRESETS["none"], np.random.default_rng(0))
    assert np.array_equal(out, image)


def test_each_operation_changes_the_image():
    image = _image()
    base = PHOTOMETRIC_PRESETS["none"]
    for field in ("color_prob", "motion_blur_prob", "noise_prob", "erase_prob", "blur_prob", "jpeg_prob", "grayscale_prob"):
        config = PhotometricConfig(**{**base.__dict__, field: 1.0})
        out = apply_photometric(image, config, np.random.default_rng(1))
        assert not np.array_equal(out, image), field
        assert out.shape == image.shape


def test_erase_is_local():
    image = _image()
    config = PhotometricConfig(**{**PHOTOMETRIC_PRESETS["none"].__dict__, "erase_prob": 1.0, "erase_fraction": (0.1, 0.1)})
    out = apply_photometric(image, config, np.random.default_rng(3))
    changed = np.any(out != image, axis=2)
    assert 0 < changed.sum() <= 6 * 6  # one 5x5 box (rounding tolerance)


def test_motion_blur_kernel_normalised_and_odd():
    k = motion_blur_kernel(3, 0.0)
    assert k.shape == (3, 3) and np.isclose(k.sum(), 1.0)
    assert k[1].sum() == pytest.approx(1.0)  # horizontal line
    assert motion_blur_kernel(4, 1.0).shape == (5, 5)


def test_scaled_probabilities():
    half = scaled(PHOTOMETRIC_PRESETS["cctv"], 0.5)
    assert half.color_prob == pytest.approx(0.4)
    assert scaled(PHOTOMETRIC_PRESETS["cctv"], 10.0).color_prob == 1.0
