"""Photometric augmentation for small head crops.

Operates on ``float32`` RGB images in ``[0, 1]`` (the format ``data.load_image`` produces) and
leaves labels untouched. The default preset ``"cctv"`` is tuned for ~50 px surveillance head
crops: strong brightness/contrast/gamma changes, short motion blur, sensor noise, small
occluders, and only mild blur / JPEG re-compression (the source images are already ~28 px
JPEGs, so heavy degradation would erase the remaining cues). Grayscale is off because hair,
skin and clothing colour are primary orientation cues at this resolution.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import cv2
import numpy as np


@dataclass(frozen=True)
class PhotometricConfig:
    # brightness / contrast / gamma
    color_prob: float = 0.8
    contrast_range: tuple[float, float] = (0.6, 1.4)
    brightness_max: float = 25.0 / 255.0
    gamma_range: tuple[float, float] = (0.7, 1.4)
    # linear motion blur, kernel length in pixels
    motion_blur_prob: float = 0.3
    motion_blur_length: tuple[int, int] = (2, 3)
    # additive gaussian noise, sigma in [0, 1] units
    noise_prob: float = 0.3
    noise_sigma: tuple[float, float] = (3.0 / 255.0, 12.0 / 255.0)
    # random erasing: rectangle side as a fraction of the image side
    erase_prob: float = 0.3
    erase_fraction: tuple[float, float] = (0.10, 0.20)
    erase_count: int = 1
    # gaussian blur (odd kernel)
    blur_prob: float = 0.15
    blur_kernel: int = 3
    # jpeg re-compression
    jpeg_prob: float = 0.15
    jpeg_quality: tuple[int, int] = (50, 90)
    # grayscale conversion
    grayscale_prob: float = 0.0

    def is_noop(self) -> bool:
        return all(
            p <= 0.0
            for p in (self.color_prob, self.motion_blur_prob, self.noise_prob, self.erase_prob, self.blur_prob, self.jpeg_prob, self.grayscale_prob)
        )


PHOTOMETRIC_PRESETS: dict[str, PhotometricConfig] = {
    "cctv": PhotometricConfig(),
    "cctv-light": PhotometricConfig(color_prob=0.4, motion_blur_prob=0.15, noise_prob=0.15, erase_prob=0.15, blur_prob=0.08, jpeg_prob=0.08),
    "none": PhotometricConfig(color_prob=0.0, motion_blur_prob=0.0, noise_prob=0.0, erase_prob=0.0, blur_prob=0.0, jpeg_prob=0.0, grayscale_prob=0.0),
}


def get_photometric_preset(name: str | None) -> PhotometricConfig | None:
    if name is None:
        return None
    try:
        return PHOTOMETRIC_PRESETS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown photometric preset {name!r}. Known presets: {', '.join(sorted(PHOTOMETRIC_PRESETS))}") from exc


def _color(image: np.ndarray, config: PhotometricConfig, rng: np.random.Generator) -> np.ndarray:
    contrast = rng.uniform(*config.contrast_range)
    brightness = rng.uniform(-config.brightness_max, config.brightness_max)
    gamma = rng.uniform(*config.gamma_range)
    image = np.clip(image * contrast + brightness, 0.0, 1.0)
    return np.power(image, gamma, dtype=np.float32)


def motion_blur_kernel(length: int, angle_rad: float) -> np.ndarray:
    """Normalised line kernel of ``length`` pixels at ``angle_rad`` (port of the HRFFA helper)."""
    size = max(3, length + (length + 1) % 2)
    kernel = np.zeros((size, size), dtype=np.float32)
    centre = size // 2
    half = (length - 1) / 2.0
    for t in np.linspace(-half, half, max(2, length * 4)):
        x = int(round(centre + t * np.cos(angle_rad)))
        y = int(round(centre + t * np.sin(angle_rad)))
        if 0 <= x < size and 0 <= y < size:
            kernel[y, x] = 1.0
    return kernel / max(float(kernel.sum()), 1.0)


def _motion_blur(image: np.ndarray, config: PhotometricConfig, rng: np.random.Generator) -> np.ndarray:
    lo, hi = config.motion_blur_length
    length = int(rng.integers(lo, hi + 1))
    if length < 2:
        return image
    kernel = motion_blur_kernel(length, float(rng.uniform(0.0, np.pi)))
    return cv2.filter2D(image, -1, kernel, borderType=cv2.BORDER_REFLECT)


def _noise(image: np.ndarray, config: PhotometricConfig, rng: np.random.Generator) -> np.ndarray:
    sigma = rng.uniform(*config.noise_sigma)
    return np.clip(image + rng.normal(0.0, sigma, image.shape).astype(np.float32), 0.0, 1.0)


def _erase(image: np.ndarray, config: PhotometricConfig, rng: np.random.Generator) -> np.ndarray:
    """Occlusion: fill 1..erase_count rectangles with uniform noise or the local mean colour."""
    h, w = image.shape[:2]
    out = image.copy()
    for _ in range(int(rng.integers(1, config.erase_count + 1))):
        eh = max(1, int(round(h * rng.uniform(*config.erase_fraction))))
        ew = max(1, int(round(w * rng.uniform(*config.erase_fraction))))
        y0 = int(rng.integers(0, max(h - eh, 0) + 1))
        x0 = int(rng.integers(0, max(w - ew, 0) + 1))
        region = out[y0 : y0 + eh, x0 : x0 + ew]
        if rng.random() < 0.5:
            region[...] = rng.uniform(0.0, 1.0, region.shape).astype(np.float32)
        else:
            region[...] = region.mean(axis=(0, 1), keepdims=True)
    return out


def _blur(image: np.ndarray, config: PhotometricConfig) -> np.ndarray:
    k = int(config.blur_kernel) | 1
    return cv2.GaussianBlur(image, (k, k), 0)


def _jpeg(image: np.ndarray, config: PhotometricConfig, rng: np.random.Generator) -> np.ndarray:
    quality = int(rng.integers(config.jpeg_quality[0], config.jpeg_quality[1] + 1))
    encoded_input = np.clip(image * 255.0 + 0.5, 0, 255).astype(np.uint8)
    ok, encoded = cv2.imencode(".jpg", encoded_input, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return image
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    return decoded.astype(np.float32) / 255.0


def _grayscale(image: np.ndarray) -> np.ndarray:
    gray = image @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    return np.repeat(gray[..., None], 3, axis=2)


def apply_photometric(image: np.ndarray, config: PhotometricConfig, rng: np.random.Generator) -> np.ndarray:
    """Apply the configured photometric augmentations to a float32 RGB image in [0, 1]."""
    if image.dtype != np.float32:
        image = image.astype(np.float32)
    image = np.ascontiguousarray(image)
    if rng.random() < config.color_prob:
        image = _color(image, config, rng)
    if rng.random() < config.grayscale_prob:
        image = _grayscale(image)
    if rng.random() < config.noise_prob:
        image = _noise(image, config, rng)
    if rng.random() < config.blur_prob:
        image = _blur(image, config)
    if rng.random() < config.motion_blur_prob:
        image = _motion_blur(image, config, rng)
    if rng.random() < config.jpeg_prob:
        image = _jpeg(image, config, rng)
    if rng.random() < config.erase_prob:
        image = _erase(image, config, rng)
    return np.clip(image, 0.0, 1.0).astype(np.float32)


def scaled(config: PhotometricConfig, factor: float) -> PhotometricConfig:
    """Return a copy with every probability multiplied by ``factor`` (clipped to [0, 1])."""
    f = lambda p: float(min(1.0, max(0.0, p * factor)))  # noqa: E731
    return replace(
        config,
        color_prob=f(config.color_prob),
        motion_blur_prob=f(config.motion_blur_prob),
        noise_prob=f(config.noise_prob),
        erase_prob=f(config.erase_prob),
        blur_prob=f(config.blur_prob),
        jpeg_prob=f(config.jpeg_prob),
        grayscale_prob=f(config.grayscale_prob),
    )
