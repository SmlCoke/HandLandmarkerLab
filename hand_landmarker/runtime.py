"""Hand model adapters and the frozen-Palm -> ROI -> Hand cascade."""

from __future__ import annotations

from dataclasses import dataclass
import copy
from pathlib import Path
from typing import Any, List, Mapping, Optional, Sequence, Tuple

from .contracts import BOARD_CONTRACT
from .geometry import RoiRect, build_roi_rect, crop_hand_roi, project_normalized_points
from .palm import PalmDetection, PalmPredictor


@dataclass(frozen=True)
class HandPrediction:
    landmarks_norm: Tuple[Tuple[float, float], ...]
    hand_flag_score: float
    handedness_score: float
    landmark_raw_max_abs: float = 0.0
    board_landmark_scale_divisor: float = 1.0
    normalized_out_of_range_coordinate_count: int = 0
    hand_flag_raw_score: float = 0.0
    handedness_raw_score: float = 0.0

    @property
    def handedness(self) -> str:
        return "Right" if self.handedness_score >= 0.5 else "Left"


@dataclass(frozen=True)
class CascadeDetection:
    palm: PalmDetection
    roi: RoiRect
    hand: HandPrediction
    landmarks_image: Tuple[Tuple[float, float], ...]


def normalize_runtime_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    """Normalize the public YAML schema into the compact runtime schema."""

    from .config import resolve_path

    value = copy.deepcopy(dict(config))
    model_config = value.get("model", {})
    if model_config:
        fixed_values = (
            ("input_shape", [1, 256, 256]),
            ("input_layout", "NCHW"),
            ("input_dtype", "float32"),
            ("output_order", ["landmarks", "hand_flag", "handedness"]),
        )
        for key, expected in fixed_values:
            if key in model_config and model_config[key] != expected:
                raise ValueError("model.{} must remain {}; got {}".format(key, expected, model_config[key]))
        if "input_scale" in model_config and abs(float(model_config["input_scale"]) - 1.0 / 255.0) > 1e-12:
            raise ValueError("model.input_scale must remain 1/255")
        if "input_offset" in model_config and float(model_config["input_offset"]) != 0.0:
            raise ValueError("model.input_offset must remain 0")
        sizes = model_config.get("output_sizes")
        if sizes is not None and dict(sizes) != {"landmarks": 42, "hand_flag": 1, "handedness": 1}:
            raise ValueError("model.output_sizes must remain landmarks=42, hand_flag=1, handedness=1")
    pipeline = value.get("pipeline", {})
    pipeline_palm = pipeline.get("palm", {})
    pipeline_roi = pipeline.get("roi", {})
    pipeline_hand = pipeline.get("hand", {})

    hand = value.setdefault("hand", {})
    if hand.get("model_path") and model_config.get("checkpoint"):
        hand_path = resolve_path(str(hand["model_path"]), config)
        checkpoint_path = resolve_path(str(model_config["checkpoint"]), config)
        if hand_path != checkpoint_path:
            raise ValueError(
                "hand.model_path and model.checkpoint refer to different files: {} != {}".format(
                    hand_path, checkpoint_path
                )
            )
    if not hand.get("model_path") and model_config.get("checkpoint"):
        hand["model_path"] = model_config["checkpoint"]
    if hand.get("model_path"):
        hand["model_path"] = str(resolve_path(str(hand["model_path"]), config))
    hand.setdefault("backend", "onnx" if str(hand.get("model_path", "")).lower().endswith(".onnx") else "keras")

    palm = value.setdefault("palm", {})
    if not palm.get("model_path") and pipeline_palm.get("model"):
        palm["model_path"] = pipeline_palm["model"]
    for source_key, target_key in (
        ("score_threshold", "score_threshold"),
        ("nms_iou_threshold", "nms_iou_threshold"),
        ("cross_head_suppression_iou", "cross_head_suppress_iou"),
        ("max_detections", "max_detections"),
    ):
        if target_key not in palm and source_key in pipeline_palm:
            palm[target_key] = pipeline_palm[source_key]
    input_shape = pipeline_palm.get("input_shape") or []
    if "input_size" not in palm and input_shape:
        palm["input_size"] = input_shape[-1]
    if palm.get("model_path"):
        palm["model_path"] = str(resolve_path(str(palm["model_path"]), config))
    if palm.get("input_layout", "NCHW") != "NCHW":
        raise ValueError("palm.input_layout must remain NCHW")
    if palm.get("color_mode", "grayscale") != "grayscale":
        raise ValueError("palm.color_mode must remain grayscale")
    palm_fixed = (
        ("input_size", BOARD_CONTRACT["palm_input_size"]),
        ("score_threshold", BOARD_CONTRACT["palm_score_threshold"]),
        ("nms_iou_threshold", BOARD_CONTRACT["palm_nms_iou_threshold"]),
        ("cross_head_suppress_iou", BOARD_CONTRACT["palm_cross_head_suppress_iou"]),
        ("max_detections", BOARD_CONTRACT["palm_max_detections"]),
    )
    for key, expected in palm_fixed:
        if key in palm and abs(float(palm[key]) - float(expected)) > 1e-9:
            raise ValueError("palm.{} must match the A1 value {}".format(key, expected))

    roi = value.setdefault("hand_roi", {})
    scale = pipeline_roi.get("scale") or []
    shift = pipeline_roi.get("shift") or []
    if len(scale) == 2:
        roi.setdefault("scale_x", scale[0])
        roi.setdefault("scale_y", scale[1])
    if len(shift) == 2:
        roi.setdefault("shift_x", shift[0])
        roi.setdefault("shift_y", shift[1])
    if int(roi.get("output_size", BOARD_CONTRACT["hand_input_width"])) != 256:
        raise ValueError("hand_roi.output_size must remain 256")
    convention = roi.get("rotation_convention")
    if convention is not None and convention != "board_wrist_to_middle_mcp":
        raise ValueError("hand_roi.rotation_convention must remain board_wrist_to_middle_mcp")
    roi_fixed = (
        ("scale_x", BOARD_CONTRACT["hand_roi_scale_x"]),
        ("scale_y", BOARD_CONTRACT["hand_roi_scale_y"]),
        ("shift_x", BOARD_CONTRACT["hand_roi_shift_x"]),
        ("shift_y", BOARD_CONTRACT["hand_roi_shift_y"]),
    )
    for key, expected in roi_fixed:
        if key in roi and abs(float(roi[key]) - float(expected)) > 1e-9:
            raise ValueError("hand_roi.{} must match the A1 value {}".format(key, expected))

    inference = value.setdefault("inference", {})
    inference.setdefault("batch_size", value.get("evaluation", {}).get("batch_size", 64))
    if int(inference["batch_size"]) <= 0:
        raise ValueError("inference.batch_size must be positive")
    if "hand_flag_threshold" not in inference and "presence_threshold" in pipeline_hand:
        inference["hand_flag_threshold"] = pipeline_hand["presence_threshold"]
    return value


def configure_tensorflow_runtime(config: Mapping[str, Any]) -> None:
    """Apply safe TensorFlow runtime flags before creating the Keras model."""

    import tensorflow as tf

    runtime = config.get("runtime", {})
    if bool(runtime.get("gpu_memory_growth", True)):
        for device in tf.config.list_physical_devices("GPU"):
            try:
                tf.config.experimental.set_memory_growth(device, True)
            except (RuntimeError, ValueError):
                pass
    if bool(runtime.get("deterministic", False)):
        try:
            tf.config.experimental.enable_op_determinism()
        except (AttributeError, RuntimeError, TypeError):
            pass


def preprocess_hand_crops(crops: Sequence[Any]):
    import cv2
    import numpy as np

    from .io_utils import to_uint8_gray

    tensors = []
    for crop in crops:
        gray = to_uint8_gray(crop)
        if gray.shape[:2] != (256, 256):
            gray = cv2.resize(gray, (256, 256), interpolation=cv2.INTER_LINEAR)
        tensors.append(gray.astype(np.float32) / 255.0)
    if not tensors:
        return np.zeros((0, 1, 256, 256), dtype=np.float32)
    return np.stack(tensors, axis=0)[:, None, :, :].astype(np.float32)


def decode_hand_outputs(outputs: Any, batch_size: int) -> List[HandPrediction]:
    import numpy as np

    values = list(outputs) if isinstance(outputs, (list, tuple)) else [outputs]
    arrays = [np.asarray(value) for value in values]
    if isinstance(batch_size, bool) or int(batch_size) <= 0:
        raise ValueError("batch_size must be a positive integer")
    batch_size = int(batch_size)
    if len(arrays) != 3:
        raise ValueError("Hand model must return exactly three ordered outputs")
    per_sample_sizes = []
    for index, array in enumerate(arrays):
        if array.size % batch_size:
            raise ValueError(
                "Hand output {} has {} values, which is not divisible by batch {}".format(
                    index, array.size, batch_size
                )
            )
        if not np.all(np.isfinite(array)):
            raise ValueError("Hand output {} contains NaN or Inf".format(index))
        per_sample_sizes.append(int(array.size // batch_size))
    if per_sample_sizes != [42, 1, 1]:
        raise ValueError(
            "Hand output order must be landmarks, hand_flag, handedness; got per-sample sizes {}".format(
                per_sample_sizes
            )
        )
    landmarks = arrays[0].reshape(batch_size, 42)
    hand_flag = arrays[1].reshape(batch_size)
    handedness = arrays[2].reshape(batch_size)

    def normalize_score(value: float) -> float:
        # Mirror HANDLANDMARKER::NormalizeScore. Sigmoid models stay in [0,1],
        # while this also makes malformed/legacy 8-bit-like outputs auditable.
        normalized = float(value)
        if normalized > 1.0 and normalized <= 255.0:
            normalized /= 255.0
        return max(0.0, min(1.0, normalized))

    decoded = []
    for sample_index in range(batch_size):
        raw = landmarks[sample_index].astype(float)
        raw_max_abs = float(np.max(np.abs(raw)))
        scale = 256.0 if raw_max_abs > 2.0 else 1.0
        normalized = raw / scale
        points = tuple(
            (float(normalized[index * 2]), float(normalized[index * 2 + 1]))
            for index in range(21)
        )
        out_of_range = int(np.count_nonzero((normalized < 0.0) | (normalized > 1.0)))
        raw_flag = float(hand_flag[sample_index])
        raw_handedness = float(handedness[sample_index])
        decoded.append(
            HandPrediction(
                landmarks_norm=points,
                hand_flag_score=normalize_score(raw_flag),
                handedness_score=normalize_score(raw_handedness),
                landmark_raw_max_abs=raw_max_abs,
                board_landmark_scale_divisor=scale,
                normalized_out_of_range_coordinate_count=out_of_range,
                hand_flag_raw_score=raw_flag,
                handedness_raw_score=raw_handedness,
            )
        )
    return decoded


class KerasHandPredictor:
    def __init__(self, weights_path: str, model_version: str = "v1", num_iterations: Any = 8) -> None:
        from models.hand_landmarker.registry import build_model

        self.model = build_model(model_version, num_iterations=num_iterations)
        path = Path(weights_path)
        if not path.is_file():
            raise FileNotFoundError("Hand weights not found: {}".format(path))
        self.model.load_weights(str(path))

    def predict(self, crops: Sequence[Any], batch_size: int = 64) -> List[HandPrediction]:
        tensors = preprocess_hand_crops(crops)
        if tensors.shape[0] == 0:
            return []
        outputs = self.model.predict(tensors, batch_size=max(1, int(batch_size)), verbose=0)
        return decode_hand_outputs(outputs, int(tensors.shape[0]))


class OnnxHandPredictor:
    def __init__(self, model_path: str, providers: Optional[Sequence[str]] = None) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("onnxruntime is required for ONNX hand inference") from exc
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError("Hand ONNX model not found: {}".format(path))
        self.session = ort.InferenceSession(
            str(path), providers=list(providers or ["CPUExecutionProvider"])
        )
        inputs = self.session.get_inputs()
        if len(inputs) != 1:
            raise ValueError("Hand ONNX must expose exactly one input")
        input_metadata = inputs[0]
        self.input_name = input_metadata.name
        input_shape = list(getattr(input_metadata, "shape", []) or [])
        if input_shape != [1, 1, 256, 256]:
            raise ValueError("Hand ONNX input must be static [1,1,256,256]; got {}".format(input_shape))
        if str(getattr(input_metadata, "type", "")) not in {"tensor(float)", "tensor(float32)"}:
            raise ValueError("Hand ONNX input must be FLOAT32")
        output_metadata = self.session.get_outputs()
        if len(output_metadata) != 3:
            raise ValueError("Hand ONNX must expose exactly three ordered outputs")
        output_sizes = []
        for output in output_metadata:
            if str(getattr(output, "type", "")) not in {"tensor(float)", "tensor(float32)"}:
                raise ValueError("All Hand ONNX outputs must be FLOAT32")
            shape = list(getattr(output, "shape", []) or [])
            if not shape or not all(isinstance(value, int) and value > 0 for value in shape):
                raise ValueError("Hand ONNX outputs must have static positive shapes: {}".format(shape))
            size = 1
            for value in shape:
                size *= int(value)
            output_sizes.append(size)
        if output_sizes != [42, 1, 1]:
            raise ValueError(
                "Hand ONNX output order must be landmarks, hand_flag, handedness; got element counts {}".format(
                    output_sizes
                )
            )
        self.fixed_batch = int(input_shape[0]) if input_shape and isinstance(input_shape[0], int) else None

    def predict(self, crops: Sequence[Any], batch_size: int = 64) -> List[HandPrediction]:
        tensors = preprocess_hand_crops(crops)
        decoded: List[HandPrediction] = []
        batch_size = self.fixed_batch or max(1, int(batch_size))
        for start in range(0, int(tensors.shape[0]), batch_size):
            batch = tensors[start : start + batch_size]
            outputs = self.session.run(None, {self.input_name: batch})
            decoded.extend(decode_hand_outputs(outputs, int(batch.shape[0])))
        return decoded


def create_hand_predictor(config: Mapping[str, Any]):
    config = normalize_runtime_config(config)
    hand_config = config.get("hand", {})
    backend = str(hand_config.get("backend", "keras")).lower()
    providers = hand_config.get("providers") or ["CPUExecutionProvider"]
    if backend == "keras":
        configure_tensorflow_runtime(config)
        return KerasHandPredictor(
            weights_path=str(hand_config["model_path"]),
            model_version=str(config.get("model", {}).get("version", "v1")),
            num_iterations=config.get("model", {}).get("num_iterations", 8),
        )
    if backend == "onnx":
        return OnnxHandPredictor(str(hand_config["model_path"]), providers=providers)
    raise ValueError("Unsupported hand backend: {}".format(backend))


class CascadeRunner:
    """PC reference implementation of the board's Palm -> ROI -> Hand flow."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        config = normalize_runtime_config(config)
        palm_config = config.get("palm", {})
        self.palm = PalmPredictor(
            model_path=str(palm_config["model_path"]),
            input_size=int(palm_config.get("input_size", BOARD_CONTRACT["palm_input_size"])),
            score_threshold=float(palm_config.get("score_threshold", BOARD_CONTRACT["palm_score_threshold"])),
            nms_iou_threshold=float(palm_config.get("nms_iou_threshold", BOARD_CONTRACT["palm_nms_iou_threshold"])),
            cross_head_suppress_iou=float(
                palm_config.get("cross_head_suppress_iou", BOARD_CONTRACT["palm_cross_head_suppress_iou"])
            ),
            max_detections=int(palm_config.get("max_detections", BOARD_CONTRACT["palm_max_detections"])),
            output_layout=str(palm_config.get("output_layout", "auto")),
            providers=palm_config.get("providers") or ["CPUExecutionProvider"],
        )
        self.hand = create_hand_predictor(config)
        roi_config = config.get("hand_roi", {})
        self.scale_x = float(roi_config.get("scale_x", BOARD_CONTRACT["hand_roi_scale_x"]))
        self.scale_y = float(roi_config.get("scale_y", BOARD_CONTRACT["hand_roi_scale_y"]))
        self.shift_x = float(roi_config.get("shift_x", BOARD_CONTRACT["hand_roi_shift_x"]))
        self.shift_y = float(roi_config.get("shift_y", BOARD_CONTRACT["hand_roi_shift_y"]))
        self.batch_size = int(config.get("inference", {}).get("batch_size", 64))

    def predict(self, image) -> List[CascadeDetection]:
        height, width = image.shape[:2]
        palms = self.palm.predict(image)
        rois = []
        crops = []
        for palm in palms:
            rect = build_roi_rect(
                palm.bbox_px(width, height),
                palm.wrist_px(width, height),
                palm.middle_mcp_px(width, height),
                width,
                height,
                self.scale_x,
                self.scale_y,
                self.shift_x,
                self.shift_y,
            )
            rois.append(rect)
            crops.append(crop_hand_roi(image, rect))
        hands = self.hand.predict(crops, batch_size=self.batch_size)
        if len(hands) != len(palms):
            raise RuntimeError("Hand predictor returned {} results for {} Palm ROIs".format(len(hands), len(palms)))
        results = []
        for palm, rect, hand in zip(palms, rois, hands):
            projected = tuple(project_normalized_points(hand.landmarks_norm, rect))
            results.append(CascadeDetection(palm=palm, roi=rect, hand=hand, landmarks_image=projected))
        return results
