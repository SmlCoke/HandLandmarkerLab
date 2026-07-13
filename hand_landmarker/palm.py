"""Frozen AetherSign Palm Detector ONNX inference and board-compatible decode."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


ANCHOR_SIZES = {14: ((0.10, 0.10), (0.18, 0.18)), 7: ((0.25, 0.25), (0.40, 0.40))}
VALUES_PER_ANCHOR = 8
REG_CHANNELS = 16
CLS_CHANNELS = 2


@dataclass(frozen=True)
class PalmDetection:
    score: float
    bbox_norm: Tuple[float, float, float, float]
    wrist_norm: Tuple[float, float]
    middle_mcp_norm: Tuple[float, float]
    head_size: int
    anchor_index: int

    def bbox_px(self, width: int, height: int) -> Tuple[float, float, float, float]:
        width_scale = float(max(1, int(width) - 1))
        height_scale = float(max(1, int(height) - 1))
        return (
            _round_pixel(self.bbox_norm[0] * width_scale),
            _round_pixel(self.bbox_norm[1] * height_scale),
            _round_pixel(self.bbox_norm[2] * width_scale),
            _round_pixel(self.bbox_norm[3] * height_scale),
        )

    def wrist_px(self, width: int, height: int) -> Tuple[float, float]:
        return (
            _round_pixel(self.wrist_norm[0] * float(max(1, int(width) - 1))),
            _round_pixel(self.wrist_norm[1] * float(max(1, int(height) - 1))),
        )

    def middle_mcp_px(self, width: int, height: int) -> Tuple[float, float]:
        return (
            _round_pixel(self.middle_mcp_norm[0] * float(max(1, int(width) - 1))),
            _round_pixel(self.middle_mcp_norm[1] * float(max(1, int(height) - 1))),
        )

    def as_dict(self, width: int, height: int) -> Dict[str, Any]:
        return {
            "score": self.score,
            "bbox_norm": list(self.bbox_norm),
            "bbox_px": list(self.bbox_px(width, height)),
            "keypoints_norm": {"p0": list(self.wrist_norm), "p9": list(self.middle_mcp_norm)},
            "keypoints_px": {
                "p0": list(self.wrist_px(width, height)),
                "p9": list(self.middle_mcp_px(width, height)),
            },
            "head": "head{}".format(self.head_size),
            "anchor_index": self.anchor_index,
        }


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _round_pixel(value: float) -> float:
    """Match ``std::round`` for the non-negative coordinates used here."""

    return float(math.floor(float(value) + 0.5))


def _generate_anchors(feature_size: int):
    import numpy as np

    anchors = []
    step = 1.0 / float(feature_size)
    for y_index in range(feature_size):
        for x_index in range(feature_size):
            for width, height in ANCHOR_SIZES[feature_size]:
                anchors.append(
                    ((x_index + 0.5) * step, (y_index + 0.5) * step, float(width), float(height))
                )
    return np.asarray(anchors, dtype=np.float32)


def _infer_layout(array, feature_size: int, channels: int, default: str) -> str:
    import numpy as np

    value = np.asarray(array)
    if value.ndim == 4 and value.shape[0] == 1:
        value = value[0]
    if value.shape == (feature_size, feature_size, channels):
        return "hwc"
    if value.shape == (channels, feature_size, feature_size):
        return "nchw"
    return default


def _reshape(array, feature_size: int, channels: int, layout: str):
    import numpy as np

    value = np.asarray(array)
    if value.ndim == 4 and value.shape[0] == 1:
        value = value[0]
    expected = feature_size * feature_size * channels
    if value.size != expected:
        raise ValueError("Palm head has {} elements, expected {}".format(value.size, expected))
    if value.ndim == 3 and value.shape == (feature_size, feature_size, channels):
        return value.reshape(-1, channels)
    if value.ndim == 3 and value.shape == (channels, feature_size, feature_size):
        return value.transpose(1, 2, 0).reshape(-1, channels)
    if layout == "hwc":
        return value.reshape(feature_size, feature_size, channels).reshape(-1, channels)
    return value.reshape(channels, feature_size, feature_size).transpose(1, 2, 0).reshape(-1, channels)


def _find_output(outputs: Sequence[Any], element_count: int, used: set):
    import numpy as np

    for index, output in enumerate(outputs):
        if index not in used and int(np.asarray(output).size) == int(element_count):
            used.add(index)
            return output
    raise ValueError("Could not find Palm ONNX output with {} elements".format(element_count))


def split_outputs(outputs: Sequence[Any]) -> Tuple[Any, Any, Any, Any]:
    used = set()
    return (
        _find_output(outputs, 14 * 14 * REG_CHANNELS, used),
        _find_output(outputs, 14 * 14 * CLS_CHANNELS, used),
        _find_output(outputs, 7 * 7 * REG_CHANNELS, used),
        _find_output(outputs, 7 * 7 * CLS_CHANNELS, used),
    )


def decode_head(
    reg_output,
    cls_output,
    feature_size: int,
    score_threshold: float,
    output_layout: str = "auto",
) -> List[PalmDetection]:
    default = output_layout if output_layout in {"nchw", "hwc"} else "nchw"
    reg_layout = _infer_layout(reg_output, feature_size, REG_CHANNELS, default)
    cls_layout = _infer_layout(cls_output, feature_size, CLS_CHANNELS, default)
    reg = _reshape(reg_output, feature_size, REG_CHANNELS, reg_layout).reshape(-1, 2, VALUES_PER_ANCHOR)
    cls = _reshape(cls_output, feature_size, CLS_CHANNELS, cls_layout)
    anchors = _generate_anchors(feature_size)
    detections = []
    for cell_index in range(feature_size * feature_size):
        for anchor_id in range(2):
            score = float(cls[cell_index, anchor_id])
            if not math.isfinite(score) or score < float(score_threshold):
                continue
            anchor_index = cell_index * 2 + anchor_id
            anchor_cx, anchor_cy, anchor_w, anchor_h = [float(value) for value in anchors[anchor_index]]
            dx, dy, dw, dh = [float(value) for value in reg[cell_index, anchor_id, :4]]
            if not all(math.isfinite(value) for value in (dx, dy, dw, dh)):
                continue
            center_x = anchor_cx + dx * anchor_w
            center_y = anchor_cy + dy * anchor_h
            width = anchor_w * math.exp(max(-10.0, min(10.0, dw)))
            height = anchor_h * math.exp(max(-10.0, min(10.0, dh)))
            points = []
            for point_index in range(2):
                base = 4 + point_index * 2
                points.append(
                    (
                        _clamp01(anchor_cx + float(reg[cell_index, anchor_id, base]) * anchor_w),
                        _clamp01(anchor_cy + float(reg[cell_index, anchor_id, base + 1]) * anchor_h),
                    )
                )
            detections.append(
                PalmDetection(
                    score=score,
                    bbox_norm=(
                        _clamp01(center_x - width * 0.5),
                        _clamp01(center_y - height * 0.5),
                        _clamp01(center_x + width * 0.5),
                        _clamp01(center_y + height * 0.5),
                    ),
                    wrist_norm=points[0],
                    middle_mcp_norm=points[1],
                    head_size=feature_size,
                    anchor_index=anchor_index,
                )
            )
    return detections


def bbox_iou(first: Sequence[float], second: Sequence[float]) -> float:
    x1 = max(float(first[0]), float(second[0]))
    y1 = max(float(first[1]), float(second[1]))
    x2 = min(float(first[2]), float(second[2]))
    y2 = min(float(first[3]), float(second[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, float(first[2]) - float(first[0])) * max(0.0, float(first[3]) - float(first[1]))
    area_b = max(0.0, float(second[2]) - float(second[0])) * max(0.0, float(second[3]) - float(second[1]))
    union = area_a + area_b - intersection
    # The scheduler adds 1e-6 to the denominator before both intra-head and
    # cross-head threshold comparisons.
    return intersection / (union + 1e-6) if union > 0.0 else 0.0


def _nms(detections: Sequence[PalmDetection], threshold: float) -> List[PalmDetection]:
    pending = sorted(detections, key=lambda item: item.score, reverse=True)
    selected = []
    while pending:
        current = pending.pop(0)
        selected.append(current)
        pending = [item for item in pending if bbox_iou(current.bbox_norm, item.bbox_norm) <= threshold]
    return selected


def select_detections(
    candidates: Sequence[PalmDetection],
    nms_iou_threshold: float = 0.30,
    cross_head_suppress_iou: float = 0.35,
    max_detections: int = 2,
) -> List[PalmDetection]:
    selected = _nms([item for item in candidates if item.head_size == 14], nms_iou_threshold)
    # The A1 scheduler applies intra-head NMS only to the 14x14 head. The
    # 7x7 candidates are score-sorted and suppressed only across selected heads.
    head7 = [item for item in candidates if item.head_size == 7]
    for candidate in sorted(head7, key=lambda item: item.score, reverse=True):
        if any(bbox_iou(candidate.bbox_norm, previous.bbox_norm) > cross_head_suppress_iou for previous in selected):
            continue
        selected.append(candidate)
        # Match palm_detector.cpp exactly: once a 7x7 candidate makes the
        # selected list reach the board limit, the board stops considering
        # lower-scored 7x7 candidates before the final global score sort.
        if len(selected) >= max(0, int(max_detections)):
            break
    return sorted(selected, key=lambda item: item.score, reverse=True)[: max(0, int(max_detections))]


def decode_outputs(
    outputs: Sequence[Any],
    score_threshold: float = 0.50,
    nms_iou_threshold: float = 0.30,
    cross_head_suppress_iou: float = 0.35,
    max_detections: int = 2,
    output_layout: str = "auto",
) -> List[PalmDetection]:
    reg14, cls14, reg7, cls7 = split_outputs(outputs)
    candidates = decode_head(reg14, cls14, 14, score_threshold, output_layout)
    candidates.extend(decode_head(reg7, cls7, 7, score_threshold, output_layout))
    return select_detections(candidates, nms_iou_threshold, cross_head_suppress_iou, max_detections)


class PalmPredictor:
    """ONNX Runtime adapter for the frozen Palm model; it never trains Palm."""

    def __init__(
        self,
        model_path: str,
        input_size: int = 224,
        score_threshold: float = 0.50,
        nms_iou_threshold: float = 0.30,
        cross_head_suppress_iou: float = 0.35,
        max_detections: int = 2,
        output_layout: str = "auto",
        providers: Optional[Sequence[str]] = None,
    ) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("onnxruntime is required for the frozen Palm Detector") from exc
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError("Frozen Palm ONNX model not found: {}".format(path))
        self.session = ort.InferenceSession(
            str(path), providers=list(providers or ["CPUExecutionProvider"])
        )
        metadata = self.session.get_inputs()[0]
        self.input_name = metadata.name
        self.input_type = str(getattr(metadata, "type", "tensor(float)"))
        self.input_size = int(input_size)
        input_shape = list(getattr(metadata, "shape", []) or [])
        if input_shape != [1, 1, self.input_size, self.input_size]:
            raise ValueError(
                "Frozen Palm input must be [1,1,{0},{0}]; got {1}".format(
                    self.input_size, input_shape
                )
            )
        if self.input_type not in {"tensor(float)", "tensor(float32)", "tensor(uint8)"}:
            raise ValueError("Unsupported frozen Palm input dtype: {}".format(self.input_type))
        expected_output_sizes = sorted(
            [14 * 14 * REG_CHANNELS, 14 * 14 * CLS_CHANNELS, 7 * 7 * REG_CHANNELS, 7 * 7 * CLS_CHANNELS]
        )
        output_sizes = []
        for output in self.session.get_outputs():
            if str(getattr(output, "type", "")) not in {"tensor(float)", "tensor(float32)"}:
                raise ValueError("Frozen Palm ONNX outputs must be FLOAT32")
            shape = list(getattr(output, "shape", []) or [])
            if not shape or not all(isinstance(value, int) and value > 0 for value in shape):
                raise ValueError("Frozen Palm ONNX outputs must have static positive shapes: {}".format(shape))
            output_sizes.append(math.prod(shape))
        if sorted(output_sizes) != expected_output_sizes:
            raise ValueError(
                "Frozen Palm output element counts changed: got {}, expected {}".format(
                    sorted(output_sizes), expected_output_sizes
                )
            )
        self.score_threshold = float(score_threshold)
        self.nms_iou_threshold = float(nms_iou_threshold)
        self.cross_head_suppress_iou = float(cross_head_suppress_iou)
        self.max_detections = int(max_detections)
        self.output_layout = str(output_layout).lower()

    def preprocess(self, image):
        import numpy as np

        from .board_ops import resize_gray_uint8

        resized = resize_gray_uint8(image, self.input_size, self.input_size)
        if "uint8" in self.input_type:
            return resized[np.newaxis, np.newaxis, :, :].astype(np.uint8)
        return (resized.astype(np.float32) / 255.0)[np.newaxis, np.newaxis, :, :]

    def predict(self, image) -> List[PalmDetection]:
        outputs = self.session.run(None, {self.input_name: self.preprocess(image)})
        return decode_outputs(
            outputs,
            score_threshold=self.score_threshold,
            nms_iou_threshold=self.nms_iou_threshold,
            cross_head_suppress_iou=self.cross_head_suppress_iou,
            max_detections=self.max_detections,
            output_layout=self.output_layout,
        )
