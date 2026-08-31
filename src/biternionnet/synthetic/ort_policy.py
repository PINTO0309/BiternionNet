"""Fail-closed batch-1 ONNX Runtime policy for synthetic QA models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
import onnxruntime as ort

BATCH_SIZE = 1
PROVIDER_PRIORITY = (
    "TensorrtExecutionProvider",
    "CUDAExecutionProvider",
    "CPUExecutionProvider",
)
ProviderSpec: TypeAlias = str | tuple[str, dict[str, Any]]


@dataclass(frozen=True)
class OnnxProviderPlan:
    providers: list[ProviderSpec]
    preferred_provider: str
    tensorrt_cache_dir: Path | None
    forced_cpu: bool

    def report(self) -> dict[str, Any]:
        return {
            "batch_size": BATCH_SIZE,
            "multiple_batch_forbidden": True,
            "provider_priority": list(PROVIDER_PRIORITY),
            "preferred_provider": self.preferred_provider,
            "forced_cpu": self.forced_cpu,
            "tensorrt_cache_dir": (
                str(self.tensorrt_cache_dir.resolve()) if self.tensorrt_cache_dir else None
            ),
            "tensorrt_cache_scope": (
                "onnxruntime-version/model-sha256/batch1" if self.tensorrt_cache_dir else None
            ),
        }


def _cache_key(model_path: Path, model_sha256: str | None) -> str:
    if model_sha256:
        return model_sha256[:16]
    stat = model_path.stat()
    return f"size{stat.st_size:x}-mtime{stat.st_mtime_ns:x}"


def build_provider_plan(
    model_path: Path,
    *,
    model_sha256: str | None = None,
    force_cpu: bool = False,
) -> OnnxProviderPlan:
    """Prefer TensorRT and isolate its cache to one model revision and batch size 1."""
    available = set(ort.get_available_providers())
    if force_cpu:
        if "CPUExecutionProvider" not in available:
            raise RuntimeError("CPUExecutionProvider is not available")
        return OnnxProviderPlan(
            providers=["CPUExecutionProvider"],
            preferred_provider="CPUExecutionProvider",
            tensorrt_cache_dir=None,
            forced_cpu=True,
        )

    providers: list[ProviderSpec] = []
    cache_dir: Path | None = None
    if "TensorrtExecutionProvider" in available:
        cache_dir = (
            model_path.parent
            / "trt_cache"
            / f"ort-{ort.__version__}"
            / f"{model_path.stem}-{_cache_key(model_path, model_sha256)}-batch1"
        )
        cache_dir.mkdir(parents=True, exist_ok=True)
        providers.append(
            (
                "TensorrtExecutionProvider",
                {
                    "trt_engine_cache_enable": True,
                    "trt_engine_cache_path": str(cache_dir.resolve()),
                    "trt_timing_cache_enable": True,
                    "trt_timing_cache_path": str(cache_dir.resolve()),
                },
            )
        )
    if "CUDAExecutionProvider" in available:
        providers.append("CUDAExecutionProvider")
    if "CPUExecutionProvider" in available:
        providers.append("CPUExecutionProvider")
    if not providers:
        raise RuntimeError(
            "none of TensorRT, CUDA, or CPU ONNX Runtime execution providers is available"
        )
    first = providers[0][0] if isinstance(providers[0], tuple) else providers[0]
    return OnnxProviderPlan(
        providers=providers,
        preferred_provider=first,
        tensorrt_cache_dir=cache_dir,
        forced_cpu=False,
    )


def require_batch_one(tensor: np.ndarray, *, model_name: str) -> None:
    if tensor.ndim < 1 or tensor.shape[0] != BATCH_SIZE:
        observed = tensor.shape[0] if tensor.ndim else "scalar"
        raise ValueError(
            f"{model_name} inference requires batch size {BATCH_SIZE}; got {observed}"
        )


def validate_model_batch_axis(
    input_shape: list[Any], *, model_name: str, allow_dynamic: bool
) -> None:
    if not input_shape:
        raise ValueError(f"{model_name} input shape has no batch axis")
    batch = input_shape[0]
    if isinstance(batch, int):
        if batch != BATCH_SIZE:
            raise ValueError(
                f"{model_name} ONNX input must have fixed batch 1 or a permitted dynamic axis, "
                f"got {input_shape}"
            )
    elif not allow_dynamic:
        raise ValueError(f"{model_name} ONNX input must have fixed batch 1, got {input_shape}")
