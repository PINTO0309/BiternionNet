"""SixDRepNet360 ONNX wrapper for full-resolution synthetic-head QA.

Adapted from HRFFA at commit 1155c7f7b3f07c649c64f45516750f86ca0e7015
(MIT). Model weights are not bundled here.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from .ort_policy import (
    ProviderSpec,
    build_provider_plan,
    require_batch_one,
    validate_model_batch_axis,
)

MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


class SixDRepNet360:
    def __init__(
        self, model_path: Path, providers: list[ProviderSpec] | None = None
    ) -> None:
        options = ort.SessionOptions()
        options.log_severity_level = 4
        self.session = ort.InferenceSession(
            str(model_path),
            sess_options=options,
            providers=(
                providers
                if providers is not None
                else build_provider_plan(model_path).providers
            ),
        )
        model_input = self.session.get_inputs()[0]
        validate_model_batch_axis(
            model_input.shape, model_name="SixDRepNet360", allow_dynamic=False
        )
        self.input_name = model_input.name
        self.execution_providers = self.session.get_providers()

    def infer(self, image_bgr: np.ndarray, head_bbox: list[float]) -> tuple[float, float, float]:
        height, width = image_bgr.shape[:2]
        x1, y1, x2, y2 = map(float, head_bbox)
        centre_x, centre_y = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        expanded_width, expanded_height = (x2 - x1) * 1.2, (y2 - y1) * 1.2
        left = max(int(centre_x - expanded_width / 2.0), 0)
        right = min(int(centre_x + expanded_width / 2.0), width)
        top = max(int(centre_y - expanded_height / 2.0), 0)
        bottom = min(int(centre_y + expanded_height / 2.0), height)
        crop = image_bgr[top:bottom, left:right]
        if crop.size == 0:
            return float("nan"), float("nan"), float("nan")
        resized = cv2.resize(crop, (256, 256), interpolation=cv2.INTER_LINEAR)[16:240, 16:240]
        rgb = resized[..., ::-1].astype(np.float32) / 255.0
        input_tensor = ((rgb - MEAN) / STD).transpose(2, 0, 1)[None].astype(np.float32)
        require_batch_one(input_tensor, model_name="SixDRepNet360")
        (pose,) = self.session.run(None, {self.input_name: input_tensor})
        return tuple(float(value) for value in pose[0])
