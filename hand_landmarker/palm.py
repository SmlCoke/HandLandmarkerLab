"""EOS 2.0 Palm Detector ONNX inference and HLMF-compatible decode."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple


VALUES_PER_ANCHOR = 8


@dataclass(frozen=True)
class PalmDetection:
    score: float
    bbox_norm: Tuple[float, float, float, float]
    wrist_norm: Tuple[float, float]
    middle_mcp_norm: Tuple[float, float]
    feature_level: str
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
            "head": self.feature_level,
            "anchor_index": self.anchor_index,
        }


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _round_pixel(value: float) -> float:
    """Match ``std::round`` for the non-negative coordinates used here."""

    return float(math.floor(float(value) + 0.5))


def normalize_feature_levels(palm_config: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Validate rectangular EOS feature levels and derive their channel counts."""

    raw_levels = palm_config.get("feature_levels")
    if not isinstance(raw_levels, list) or not raw_levels:
        raise ValueError("palm.feature_levels must be a non-empty list")
    levels: List[Dict[str, Any]] = []
    names: Set[str] = set()
    shapes: Set[Tuple[int, int]] = set()
    for index, raw in enumerate(raw_levels):
        if not isinstance(raw, Mapping):
            raise ValueError("palm.feature_levels[{}] must be a mapping".format(index))
        name = str(raw.get("name") or "").strip()
        if not name or name in names:
            raise ValueError("palm feature-level names must be non-empty and unique")
        height = raw.get("height")
        width = raw.get("width")
        if (
            isinstance(height, bool)
            or not isinstance(height, int)
            or height < 1
            or isinstance(width, bool)
            or not isinstance(width, int)
            or width < 1
        ):
            raise ValueError(
                "palm.feature_levels[{}] height/width must be positive integers".format(index)
            )
        shape = (height, width)
        if shape in shapes:
            raise ValueError("palm feature-level shapes must be unique")
        raw_anchors = raw.get("anchor_sizes")
        if not isinstance(raw_anchors, list) or len(raw_anchors) != 2:
            raise ValueError(
                "palm.feature_levels[{}].anchor_sizes must contain exactly two sizes".format(index)
            )
        anchors: List[Tuple[float, float]] = []
        for anchor_index, raw_anchor in enumerate(raw_anchors):
            if not isinstance(raw_anchor, (list, tuple)) or len(raw_anchor) != 2:
                raise ValueError(
                    "palm.feature_levels[{}].anchor_sizes[{}] must be [width,height]".format(
                        index, anchor_index
                    )
                )
            anchor_width, anchor_height = [float(value) for value in raw_anchor]
            if not all(
                math.isfinite(value) and value > 0.0
                for value in (anchor_width, anchor_height)
            ):
                raise ValueError("palm anchor sizes must be finite and positive")
            anchors.append((anchor_width, anchor_height))
        levels.append(
            {
                "name": name,
                "height": height,
                "width": width,
                "anchor_sizes": tuple(anchors),
                "reg_channels": len(anchors) * VALUES_PER_ANCHOR,
                "cls_channels": len(anchors),
            }
        )
        names.add(name)
        shapes.add(shape)
    return levels


def feature_level_anchor_count(levels: Sequence[Mapping[str, Any]]) -> int:
    return sum(
        int(level["height"]) * int(level["width"]) * len(level["anchor_sizes"])
        for level in levels
    )


def generate_anchors(
    feature_height: int,
    feature_width: int,
    sizes: Sequence[Tuple[float, float]],
):
    import numpy as np

    anchors = []
    step_x = 1.0 / float(feature_width)
    step_y = 1.0 / float(feature_height)
    for y_index in range(feature_height):
        for x_index in range(feature_width):
            for width, height in sizes:
                anchors.append(
                    (
                        (x_index + 0.5) * step_x,
                        (y_index + 0.5) * step_y,
                        float(width),
                        float(height),
                    )
                )
    return np.asarray(anchors, dtype=np.float32)


def _infer_layout(array, feature_height: int, feature_width: int, channels: int, default: str) -> str:
    import numpy as np

    value = np.asarray(array)
    if value.ndim == 4 and value.shape[0] == 1:
        value = value[0]
    if value.shape == (feature_height, feature_width, channels):
        return "hwc"
    if value.shape == (channels, feature_height, feature_width):
        return "nchw"
    return default


def _reshape(array, feature_height: int, feature_width: int, channels: int, layout: str):
    import numpy as np

    value = np.asarray(array)
    if value.ndim == 4 and value.shape[0] == 1:
        value = value[0]
    expected = feature_height * feature_width * channels
    if value.size != expected:
        raise ValueError("Palm head has {} elements, expected {}".format(value.size, expected))
    if value.ndim == 3 and value.shape == (feature_height, feature_width, channels):
        return value.reshape(-1, channels)
    if value.ndim == 3 and value.shape == (channels, feature_height, feature_width):
        return value.transpose(1, 2, 0).reshape(-1, channels)
    if layout == "hwc":
        return value.reshape(feature_height, feature_width, channels).reshape(-1, channels)
    return (
        value.reshape(channels, feature_height, feature_width)
        .transpose(1, 2, 0)
        .reshape(-1, channels)
    )


def _find_output(outputs: Sequence[Any], element_count: int, used: Set[int]):
    import numpy as np

    matches = [
        index
        for index, output in enumerate(outputs)
        if index not in used and int(np.asarray(output).size) == int(element_count)
    ]
    if len(matches) != 1:
        raise ValueError(
            "Expected exactly one unused Palm ONNX output with {} elements, got {}".format(
                element_count, len(matches)
            )
        )
    used.add(matches[0])
    return outputs[matches[0]]


def split_outputs(
    outputs: Sequence[Any], feature_levels: Sequence[Mapping[str, Any]]
) -> List[Tuple[Mapping[str, Any], Any, Any]]:
    if len(outputs) != len(feature_levels) * 2:
        raise ValueError(
            "Palm ONNX must expose {} outputs, got {}".format(
                len(feature_levels) * 2, len(outputs)
            )
        )
    used: Set[int] = set()
    split = []
    for level in feature_levels:
        cells = int(level["height"]) * int(level["width"])
        regression = _find_output(outputs, cells * int(level["reg_channels"]), used)
        classification = _find_output(outputs, cells * int(level["cls_channels"]), used)
        split.append((level, regression, classification))
    if len(used) != len(outputs):
        raise ValueError("Palm ONNX contains unrecognized outputs")
    return split


def decode_head(
    reg_output,
    cls_output,
    level: Mapping[str, Any],
    score_threshold: float,
    output_layout: str = "nchw",
) -> List[PalmDetection]:
    feature_height = int(level["height"])
    feature_width = int(level["width"])
    anchor_sizes = level["anchor_sizes"]
    anchor_count = len(anchor_sizes)
    default = output_layout if output_layout in {"nchw", "hwc"} else "nchw"
    reg_layout = _infer_layout(
        reg_output, feature_height, feature_width, int(level["reg_channels"]), default
    )
    cls_layout = _infer_layout(
        cls_output, feature_height, feature_width, int(level["cls_channels"]), default
    )
    reg = _reshape(
        reg_output,
        feature_height,
        feature_width,
        int(level["reg_channels"]),
        reg_layout,
    ).reshape(-1, anchor_count, VALUES_PER_ANCHOR)
    cls = _reshape(
        cls_output,
        feature_height,
        feature_width,
        int(level["cls_channels"]),
        cls_layout,
    )
    anchors = generate_anchors(feature_height, feature_width, anchor_sizes)
    detections = []
    for cell_index in range(feature_height * feature_width):
        for anchor_id in range(anchor_count):
            score = float(cls[cell_index, anchor_id])
            if not math.isfinite(score) or score < float(score_threshold):
                continue
            anchor_index = cell_index * anchor_count + anchor_id
            anchor_cx, anchor_cy, anchor_w, anchor_h = [
                float(value) for value in anchors[anchor_index]
            ]
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
                        _clamp01(
                            anchor_cy
                            + float(reg[cell_index, anchor_id, base + 1]) * anchor_h
                        ),
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
                    feature_level=str(level["name"]),
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
    area_a = max(0.0, float(first[2]) - float(first[0])) * max(
        0.0, float(first[3]) - float(first[1])
    )
    area_b = max(0.0, float(second[2]) - float(second[0])) * max(
        0.0, float(second[3]) - float(second[1])
    )
    return intersection / (area_a + area_b - intersection + 1e-6)


def select_detections(
    candidates: Sequence[PalmDetection],
    nms_iou_threshold: float = 0.10,
    max_detections: int = 2,
) -> List[PalmDetection]:
    """Apply the EOS 2.0 global NMS after merging all feature levels."""

    if not candidates:
        return []
    import numpy as np

    boxes = np.asarray([item.bbox_norm for item in candidates], dtype=np.float32)
    scores = np.asarray([item.score for item in candidates], dtype=np.float32)
    x1, y1, x2, y2 = [boxes[:, index] for index in range(4)]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        current = int(order[0])
        keep.append(current)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[current], x1[rest])
        yy1 = np.maximum(y1[current], y1[rest])
        xx2 = np.minimum(x2[current], x2[rest])
        yy2 = np.minimum(y2[current], y2[rest])
        intersection = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = intersection / (areas[current] + areas[rest] - intersection + 1e-6)
        order = rest[np.where(iou <= float(nms_iou_threshold))[0]]
    selected = [candidates[index] for index in keep]
    selected.sort(key=lambda item: item.score, reverse=True)
    return selected[: max(0, int(max_detections))]


def decode_outputs(
    outputs: Sequence[Any],
    feature_levels: Sequence[Mapping[str, Any]],
    score_threshold: float = 0.25,
    nms_iou_threshold: float = 0.10,
    max_detections: int = 2,
    output_layout: str = "nchw",
) -> List[PalmDetection]:
    candidates = []
    for level, regression, classification in split_outputs(outputs, feature_levels):
        candidates.extend(
            decode_head(
                regression,
                classification,
                level,
                score_threshold,
                output_layout,
            )
        )
    return select_detections(candidates, nms_iou_threshold, max_detections)


def preprocess_for_onnx(
    image, input_width: int, input_height: int, input_type: str = "tensor(float)"
):
    import cv2
    import numpy as np

    from .io_utils import to_uint8_gray

    gray = to_uint8_gray(image)
    resized = cv2.resize(
        gray,
        (int(input_width), int(input_height)),
        interpolation=cv2.INTER_AREA,
    )
    if "uint8" in input_type:
        return resized[np.newaxis, np.newaxis, :, :].astype(np.uint8)
    return (resized.astype(np.float32) / 255.0)[np.newaxis, np.newaxis, :, :]


def _static_shape(value: Any, field: str) -> List[int]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError("{} must be one static NCHW rank-4 shape".format(field))
    shape = []
    for dimension in value:
        if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 1:
            raise ValueError("{} must contain only positive integer dimensions".format(field))
        shape.append(dimension)
    return shape


def palm_model_contract(
    session: Any,
    input_width: int,
    input_height: int,
    feature_levels: Sequence[Mapping[str, Any]],
    output_layout: str = "nchw",
) -> Dict[str, Any]:
    if output_layout not in {"nchw", "hwc"}:
        raise ValueError("palm.output_layout must be nchw or hwc")
    inputs = list(session.get_inputs())
    if len(inputs) != 1:
        raise ValueError("Palm ONNX must expose exactly one input, got {}".format(len(inputs)))
    input_meta = inputs[0]
    input_shape = _static_shape(input_meta.shape, "Palm ONNX input")
    expected_input_shape = [1, 1, int(input_height), int(input_width)]
    if input_shape != expected_input_shape:
        raise ValueError(
            "Palm ONNX input shape {} does not match configured {}".format(
                input_shape, expected_input_shape
            )
        )
    input_type = str(getattr(input_meta, "type", ""))
    if input_type not in {"tensor(float)", "tensor(float32)", "tensor(uint8)"}:
        raise ValueError("Palm ONNX input type must be float32 or uint8, got {!r}".format(input_type))
    expected_shapes = []
    for level in feature_levels:
        height = int(level["height"])
        width = int(level["width"])
        for channels in (int(level["reg_channels"]), int(level["cls_channels"])):
            expected_shapes.append(
                [1, channels, height, width]
                if output_layout == "nchw"
                else [1, height, width, channels]
            )
    outputs = list(session.get_outputs())
    output_shapes = [
        _static_shape(getattr(meta, "shape", None), "Palm ONNX output {}".format(meta.name))
        for meta in outputs
    ]
    if sorted(output_shapes) != sorted(expected_shapes):
        raise ValueError(
            "Palm ONNX output shapes {} do not match configured {}".format(
                output_shapes, expected_shapes
            )
        )
    if any(
        str(getattr(meta, "type", "")) not in {"tensor(float)", "tensor(float32)"}
        for meta in outputs
    ):
        raise ValueError("Palm ONNX outputs must all be float32")
    return {
        "input_name": str(input_meta.name),
        "input_shape": input_shape,
        "input_type": input_type,
        "output_shapes": output_shapes,
        "anchor_count": feature_level_anchor_count(feature_levels),
    }


class PalmPredictor:
    """ONNX Runtime adapter for EOS 2.0; it never trains Palm."""

    def __init__(
        self,
        model_path: str,
        input_width: int = 384,
        input_height: int = 224,
        feature_levels: Optional[Sequence[Mapping[str, Any]]] = None,
        score_threshold: float = 0.25,
        nms_iou_threshold: float = 0.10,
        max_detections: int = 2,
        output_layout: str = "nchw",
        providers: Optional[Sequence[str]] = None,
    ) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("onnxruntime is required for the EOS Palm Detector") from exc
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError("EOS Palm ONNX model not found: {}".format(path))
        if feature_levels is None:
            raise ValueError("palm.feature_levels is required")
        self.feature_levels = normalize_feature_levels(
            {"feature_levels": list(feature_levels)}
        )
        self.input_width = int(input_width)
        self.input_height = int(input_height)
        self.output_layout = str(output_layout).lower()
        self.session = ort.InferenceSession(
            str(path), providers=list(providers or ["CPUExecutionProvider"])
        )
        contract = palm_model_contract(
            self.session,
            self.input_width,
            self.input_height,
            self.feature_levels,
            self.output_layout,
        )
        self.input_name = contract["input_name"]
        self.input_type = contract["input_type"]
        self.score_threshold = float(score_threshold)
        self.nms_iou_threshold = float(nms_iou_threshold)
        self.max_detections = int(max_detections)
        if not math.isfinite(self.score_threshold) or not 0.0 <= self.score_threshold <= 1.0:
            raise ValueError("palm.score_threshold must be finite and in [0,1]")
        if not math.isfinite(self.nms_iou_threshold) or not 0.0 <= self.nms_iou_threshold <= 1.0:
            raise ValueError("palm.nms_iou_threshold must be finite and in [0,1]")
        if self.max_detections < 1:
            raise ValueError("palm.max_detections must be a positive integer")

    def preprocess(self, image):
        return preprocess_for_onnx(
            image, self.input_width, self.input_height, self.input_type
        )

    def predict(self, image) -> List[PalmDetection]:
        outputs = self.session.run(None, {self.input_name: self.preprocess(image)})
        return decode_outputs(
            outputs,
            feature_levels=self.feature_levels,
            score_threshold=self.score_threshold,
            nms_iou_threshold=self.nms_iou_threshold,
            max_detections=self.max_detections,
            output_layout=self.output_layout,
        )
