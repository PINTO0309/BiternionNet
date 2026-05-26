from __future__ import annotations

import gzip
import json
import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .losses import deg2bit, quantize_labels


@dataclass(frozen=True)
class CropConfig:
    size: tuple[int, int] | None
    random_crop: bool = False


def read_manifest(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
            records.append(record)
    return records


def write_manifest(records: Iterable[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def build_class_mapping(records: list[dict[str, Any]], exclude_label: str | None = None) -> dict[str, int]:
    labels = sorted({str(r["label"]) for r in records if r.get("task") == "classification"})
    if exclude_label is not None:
        labels = [label for label in labels if label != exclude_label]
    return {label: i for i, label in enumerate(labels)}


def load_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), flags=cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image.astype(np.float32) / 255.0


def crop_image(image: np.ndarray, crop: CropConfig | None) -> np.ndarray:
    if crop is None or crop.size is None:
        return image
    crop_h, crop_w = crop.size
    h, w = image.shape[:2]
    if h < crop_h or w < crop_w:
        resize_h = max(h, crop_h)
        resize_w = max(w, crop_w)
        image = cv2.resize(image, (resize_w, resize_h), interpolation=cv2.INTER_LANCZOS4)
        h, w = image.shape[:2]
    if crop.random_crop:
        top = random.randint(0, h - crop_h)
        left = random.randint(0, w - crop_w)
    else:
        top = (h - crop_h) // 2
        left = (w - crop_w) // 2
    return image[top : top + crop_h, left : left + crop_w]


def flip_label(record: dict[str, Any], class_flip_map: dict[str, str] | None = None) -> dict[str, Any]:
    flipped = dict(record)
    task = flipped.get("task")
    if task == "classification":
        if class_flip_map is not None:
            flipped["label"] = class_flip_map.get(str(flipped["label"]), str(flipped["label"]))
    elif task == "angle_deg":
        flipped["angle_deg"] = float((360.0 - float(flipped["angle_deg"])) % 360.0)
    return flipped


class ManifestDataset(Dataset):
    def __init__(
        self,
        manifest: str | Path,
        split: str,
        target_kind: str,
        crop: CropConfig | None = None,
        class_to_idx: dict[str, int] | None = None,
        class_flip_map: dict[str, str] | None = None,
        exclude_label: str | None = None,
        flip_probability: float = 0.0,
        quantization_borders: np.ndarray | None = None,
        quantization_centres: np.ndarray | None = None,
    ) -> None:
        self.manifest = Path(manifest)
        self.root = self.manifest.parent
        self.target_kind = target_kind
        self.crop = crop
        self.class_to_idx = class_to_idx or {}
        self.class_flip_map = class_flip_map or {}
        self.flip_probability = flip_probability
        self.quantization_borders = quantization_borders
        self.quantization_centres = quantization_centres

        records = [r for r in read_manifest(self.manifest) if r.get("split") == split]
        if exclude_label is not None:
            records = [r for r in records if r.get("label") != exclude_label]
        if not records:
            raise ValueError(f"No records for split={split!r} in {manifest}")
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        record = self.records[index]
        image_path = Path(record["image"])
        if not image_path.is_absolute():
            image_path = self.root / image_path
        image = load_image(image_path)

        if self.flip_probability > 0.0 and random.random() < self.flip_probability:
            image = np.ascontiguousarray(image[:, ::-1])
            record = flip_label(record, self.class_flip_map)

        image = crop_image(image, self.crop)
        image_tensor = torch.from_numpy(np.transpose(image, (2, 0, 1))).float()
        return image_tensor, self._target(record)

    def _target(self, record: dict[str, Any]) -> torch.Tensor:
        if self.target_kind == "classification":
            return torch.tensor(self.class_to_idx[str(record["label"])], dtype=torch.long)
        if self.target_kind == "angle_deg":
            return torch.tensor([float(record["angle_deg"])], dtype=torch.float32)
        if self.target_kind == "angle_rad":
            return torch.deg2rad(torch.tensor([float(record["angle_deg"])], dtype=torch.float32))
        if self.target_kind == "biternion":
            return torch.from_numpy(deg2bit(np.array(float(record["angle_deg"]), dtype=np.float32))).float()
        if self.target_kind == "pose_rad":
            return torch.tensor([float(record["pan"]), float(record["tilt"]), float(record["roll"])], dtype=torch.float32)
        if self.target_kind.startswith("quantized_"):
            if self.quantization_borders is None or self.quantization_centres is None:
                raise ValueError("Quantized target requires borders and centres")
            angle = np.array(float(record["angle_deg"]), dtype=np.float32)
            q = int(quantize_labels(angle.reshape(1), self.quantization_borders)[0])
            if self.target_kind == "quantized_classification":
                return torch.tensor(q, dtype=torch.long)
            centre = float(self.quantization_centres[q])
            if self.target_kind == "quantized_angle_deg":
                return torch.tensor([centre], dtype=torch.float32)
            if self.target_kind == "quantized_biternion":
                return torch.from_numpy(deg2bit(np.array(centre, dtype=np.float32))).float()
        raise ValueError(f"Unsupported target kind: {self.target_kind}")


def open_pickle(path: str | Path):
    path = Path(path)
    if path.suffix == ".gz":
        with gzip.open(path, "rb") as f:
            return pickle.load(f)
    with path.open("rb") as f:
        return pickle.load(f)
