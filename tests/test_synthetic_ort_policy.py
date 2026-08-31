from pathlib import Path

import numpy as np
import pytest

from biternionnet.synthetic.detector import Deimv2Detector
from biternionnet.synthetic.ort_policy import (
    build_provider_plan,
    require_batch_one,
    validate_model_batch_axis,
)


def test_tensorrt_is_first_and_cache_is_isolated_to_model_hash_and_batch1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    model = tmp_path / "model.onnx"
    model.write_bytes(b"fixture")
    monkeypatch.setattr(
        "biternionnet.synthetic.ort_policy.ort.get_available_providers",
        lambda: [
            "CPUExecutionProvider",
            "CUDAExecutionProvider",
            "TensorrtExecutionProvider",
        ],
    )
    digest = "a" * 64
    plan = build_provider_plan(model, model_sha256=digest)
    provider, options = plan.providers[0]
    assert provider == "TensorrtExecutionProvider"
    assert plan.providers[1:] == ["CUDAExecutionProvider", "CPUExecutionProvider"]
    assert plan.tensorrt_cache_dir is not None
    assert plan.tensorrt_cache_dir.name == "model-aaaaaaaaaaaaaaaa-batch1"
    assert plan.tensorrt_cache_dir.is_dir()
    assert options["trt_engine_cache_enable"] is True
    assert options["trt_engine_cache_path"] == str(plan.tensorrt_cache_dir.resolve())
    assert plan.report()["multiple_batch_forbidden"] is True


def test_cpu_override_and_batch_guards_are_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    model = tmp_path / "model.onnx"
    model.write_bytes(b"fixture")
    monkeypatch.setattr(
        "biternionnet.synthetic.ort_policy.ort.get_available_providers",
        lambda: ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    plan = build_provider_plan(model, force_cpu=True)
    assert plan.providers == ["CPUExecutionProvider"]
    assert plan.tensorrt_cache_dir is None

    require_batch_one(np.zeros((1, 3, 8, 8), dtype=np.float32), model_name="fixture")
    with pytest.raises(ValueError, match="batch size 1"):
        require_batch_one(np.zeros((2, 3, 8, 8), dtype=np.float32), model_name="fixture")
    validate_model_batch_axis(["N", 3, 640, 640], model_name="DEIM", allow_dynamic=True)
    with pytest.raises(ValueError, match="fixed batch 1"):
        validate_model_batch_axis(["N", 3, 320, 320], model_name="HRFFA", allow_dynamic=False)


def test_deim_list_api_never_combines_images_into_a_multiple_batch():
    detector = Deimv2Detector.__new__(Deimv2Detector)
    calls = []
    detector.preprocess = lambda image: np.zeros((3, 640, 640), dtype=np.float32)

    def run_single(batch):
        calls.append(batch.shape)
        return np.zeros((1, 1, 6), dtype=np.float32)

    detector._run_single_with_oom_fallback = run_single
    detector._postprocess = lambda output, images: [[[]]]
    images = [np.zeros((20, 20, 3), dtype=np.uint8) for _ in range(3)]
    assert detector.infer_batch(images) == [[[]], [[]], [[]]]
    assert calls == [(1, 3, 640, 640)] * 3
