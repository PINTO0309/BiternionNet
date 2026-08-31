"""DEIMv2-Wholebody49 ONNX wrapper used by synthetic-image QA.

Adapted from HRFFA at commit 1155c7f7b3f07c649c64f45516750f86ca0e7015
(MIT). DEIMv2 itself is Apache-2.0; weights are not bundled here.
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

INPUT_SIZE = 640
CLASS_BODY = 0
CLASS_HEAD = 7
DIR8_CLASSES = {
    8: "front",
    9: "right_front",
    10: "right_side",
    11: "right_back",
    12: "back",
    13: "left_back",
    14: "left_side",
    15: "left_front",
}
KEEP_CLASSES = frozenset([CLASS_BODY, CLASS_HEAD, *DIR8_CLASSES, 16, 17, 18, 19, 20])


class Deimv2Detector:
    def __init__(
        self,
        model_path: Path,
        providers: list[ProviderSpec] | None = None,
        score_threshold: float = 0.25,
    ) -> None:
        self.score_threshold = score_threshold
        self._model_path = str(model_path)
        self._providers = (
            providers
            if providers is not None
            else build_provider_plan(model_path, allow_tensorrt=False).providers
        )
        self.session = self._make_session(self._providers)
        model_input = self.session.get_inputs()[0]
        validate_model_batch_axis(
            model_input.shape, model_name="DEIMv2", allow_dynamic=True
        )
        self.input_name = model_input.name
        self.execution_providers = self.session.get_providers()
        self._cpu_session: ort.InferenceSession | None = None

    def _make_session(self, providers: list[ProviderSpec]) -> ort.InferenceSession:
        options = ort.SessionOptions()
        options.log_severity_level = 4
        return ort.InferenceSession(self._model_path, sess_options=options, providers=providers)

    @staticmethod
    def preprocess(image_bgr: np.ndarray) -> np.ndarray:
        resized = cv2.resize(image_bgr, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        return rgb.transpose(2, 0, 1).astype(np.float32) / 255.0

    def infer_batch(self, images_bgr: list[np.ndarray]) -> list[list[list[float]]]:
        """Accept a list for API compatibility, but execute every image as an isolated batch of 1."""
        return [self.infer(image) for image in images_bgr]

    def infer(self, image_bgr: np.ndarray) -> list[list[float]]:
        batch = self.preprocess(image_bgr)[None]
        require_batch_one(batch, model_name="DEIMv2")
        output = self._run_single_with_oom_fallback(batch)
        return self._postprocess(output, [image_bgr])[0]

    def _run_single_with_oom_fallback(self, batch: np.ndarray) -> np.ndarray:
        require_batch_one(batch, model_name="DEIMv2")
        try:
            (output,) = self.session.run(["label_xyxy_score"], {self.input_name: batch})
            return output
        except Exception as exc:
            if "Failed to allocate memory" not in str(exc):
                raise
            self.session = self._make_session(self._providers)
            try:
                (output,) = self.session.run(
                    ["label_xyxy_score"], {self.input_name: batch}
                )
                return output
            except Exception as retry_exc:
                if "Failed to allocate memory" not in str(retry_exc):
                    raise
                if self._cpu_session is None:
                    self._cpu_session = self._make_session(["CPUExecutionProvider"])
                (output,) = self._cpu_session.run(
                    ["label_xyxy_score"], {self.input_name: batch}
                )
                return output

    def _postprocess(
        self, output: np.ndarray, images_bgr: list[np.ndarray]
    ) -> list[list[list[float]]]:
        results: list[list[list[float]]] = []
        for detections, image in zip(output, images_bgr, strict=True):
            height, width = image.shape[:2]
            rows: list[list[float]] = []
            for cls, x1, y1, x2, y2, score in detections[detections[:, 5] >= self.score_threshold]:
                cls = int(cls)
                if cls not in KEEP_CLASSES:
                    continue
                rows.append(
                    [
                        cls,
                        round(float(np.clip(x1, 0, 1)) * width, 2),
                        round(float(np.clip(y1, 0, 1)) * height, 2),
                        round(float(np.clip(x2, 0, 1)) * width, 2),
                        round(float(np.clip(y2, 0, 1)) * height, 2),
                        round(float(score), 4),
                    ]
                )
            results.append(rows)
        return results
