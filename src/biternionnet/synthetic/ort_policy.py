"""Fail-closed batch-1 ONNX Runtime policy for synthetic QA models."""

from __future__ import annotations

import ctypes
import ctypes.util
import re
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
CUDA_PROVIDER_PRIORITY = (
    "CUDAExecutionProvider",
    "CPUExecutionProvider",
)
ProviderSpec: TypeAlias = str | tuple[str, dict[str, Any]]


@dataclass(frozen=True)
class TensorRTRuntimeFingerprint:
    onnxruntime: str
    tensorrt: str
    cuda_runtime: str
    compute_capability: str
    precision: str = "fp32"

    def cache_component(self) -> str:
        raw = (
            f"ort-{self.onnxruntime}_trt-{self.tensorrt}_cuda-{self.cuda_runtime}_"
            f"sm{self.compute_capability}_{self.precision}"
        )
        return re.sub(r"[^A-Za-z0-9_.-]+", "-", raw)

    def report(self) -> dict[str, str]:
        return {
            "onnxruntime": self.onnxruntime,
            "tensorrt": self.tensorrt,
            "cuda_runtime": self.cuda_runtime,
            "compute_capability": self.compute_capability,
            "precision": self.precision,
        }


@dataclass(frozen=True)
class OnnxProviderPlan:
    providers: list[ProviderSpec]
    provider_priority: tuple[str, ...]
    preferred_provider: str
    tensorrt_cache_dir: Path | None
    tensorrt_runtime: TensorRTRuntimeFingerprint | None
    forced_cpu: bool

    def report(self) -> dict[str, Any]:
        return {
            "batch_size": BATCH_SIZE,
            "multiple_batch_forbidden": True,
            "provider_priority": list(self.provider_priority),
            "preferred_provider": self.preferred_provider,
            "forced_cpu": self.forced_cpu,
            "tensorrt_cache_dir": (
                str(self.tensorrt_cache_dir.resolve()) if self.tensorrt_cache_dir else None
            ),
            "tensorrt_cache_scope": (
                "ort+tensorrt+cuda+compute-capability+precision/model-sha256/batch1"
                if self.tensorrt_cache_dir
                else None
            ),
            "tensorrt_runtime": (
                self.tensorrt_runtime.report() if self.tensorrt_runtime else None
            ),
        }


def _cache_key(model_path: Path, model_sha256: str | None) -> str:
    if model_sha256:
        return model_sha256[:16]
    stat = model_path.stat()
    return f"size{stat.st_size:x}-mtime{stat.st_mtime_ns:x}"


def _tensorrt_version() -> str:
    for name in ("nvinfer", "nvinfer.so.10", "nvinfer.so.8"):
        library_path = ctypes.util.find_library(name) if "." not in name else f"lib{name}"
        if not library_path:
            continue
        try:
            library = ctypes.CDLL(library_path)
            library.getInferLibVersion.restype = ctypes.c_int
            value = int(library.getInferLibVersion())
            if value > 0:
                return f"{value // 10000}.{(value // 100) % 100}.{value % 100}"
        except (AttributeError, OSError):
            continue
    raise RuntimeError(
        "TensorRTExecutionProvider is available but the TensorRT runtime version cannot be read; "
        "refusing to create or reuse an unversioned engine cache"
    )


def _runtime_fingerprint() -> TensorRTRuntimeFingerprint:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "TensorRT cache isolation requires PyTorch to identify the CUDA runtime and GPU"
        ) from exc
    if not torch.cuda.is_available() or torch.version.cuda is None:
        raise RuntimeError(
            "TensorRTExecutionProvider is available but CUDA runtime identity is unavailable"
        )
    major, minor = torch.cuda.get_device_capability(torch.cuda.current_device())
    return TensorRTRuntimeFingerprint(
        onnxruntime=ort.__version__,
        tensorrt=_tensorrt_version(),
        cuda_runtime=str(torch.version.cuda),
        compute_capability=f"{major}{minor}",
    )


def build_provider_plan(
    model_path: Path,
    *,
    model_sha256: str | None = None,
    force_cpu: bool = False,
    allow_tensorrt: bool = True,
) -> OnnxProviderPlan:
    """Prefer TensorRT and isolate its cache to one model revision and batch size 1."""
    available = set(ort.get_available_providers())
    if force_cpu:
        if "CPUExecutionProvider" not in available:
            raise RuntimeError("CPUExecutionProvider is not available")
        return OnnxProviderPlan(
            providers=["CPUExecutionProvider"],
            provider_priority=("CPUExecutionProvider",),
            preferred_provider="CPUExecutionProvider",
            tensorrt_cache_dir=None,
            tensorrt_runtime=None,
            forced_cpu=True,
        )

    providers: list[ProviderSpec] = []
    cache_dir: Path | None = None
    runtime: TensorRTRuntimeFingerprint | None = None
    if allow_tensorrt and "TensorrtExecutionProvider" in available:
        runtime = _runtime_fingerprint()
        cache_dir = (
            model_path.parent
            / "trt_cache"
            / runtime.cache_component()
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
        provider_priority=(PROVIDER_PRIORITY if allow_tensorrt else CUDA_PROVIDER_PRIORITY),
        preferred_provider=first,
        tensorrt_cache_dir=cache_dir,
        tensorrt_runtime=runtime,
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
