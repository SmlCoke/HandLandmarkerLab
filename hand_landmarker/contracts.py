"""Stable model, annotation, and board-runtime contracts.

Keep this module free of TensorFlow/OpenCV imports so configuration and dataset
checks remain usable before the training environment is installed.
"""

from __future__ import annotations

import math
import re

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


MODEL_CHECKPOINT_STAGES: Tuple[str, str] = ("pretrain", "finetune")


MODEL_IO: Dict[str, Any] = {
    "input_name": "inputs",
    "input_shape": [None, 1, 256, 256],
    "input_layout": "NCHW",
    "input_dtype": "float32",
    "input_range": [0.0, 1.0],
    "outputs": [
        {
            "semantic": "landmarks",
            "keras_shape": [None, 1, 1, 42],
            "deployed_onnx_shape": [1, 42, 1, 1],
            "elements_per_sample": 42,
        },
        {"semantic": "hand_flag", "shape": [None, 1, 1, 1], "elements_per_sample": 1},
        {"semantic": "handedness", "shape": [None, 1, 1, 1], "elements_per_sample": 1},
    ],
    "landmark_order": "x0,y0,x1,y1,...,x20,y20",
    "handedness_encoding": {"Left": 0.0, "Right": 1.0},
}

BOARD_CONTRACT: Dict[str, Any] = {
    "upright_image_width": 1280,
    "upright_image_height": 720,
    "image_channels": 1,
    "palm_input_size": 224,
    "palm_score_threshold": 0.50,
    "palm_nms_iou_threshold": 0.30,
    "palm_cross_head_suppress_iou": 0.35,
    "palm_max_detections": 2,
    "hand_input_width": 256,
    "hand_input_height": 256,
    "hand_roi_scale_x": 1.8,
    "hand_roi_scale_y": 1.8,
    "hand_roi_shift_x": 0.0,
    "hand_roi_shift_y": -0.1,
}

HAND_CONNECTIONS: Tuple[Tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)

# Anatomical parent-child vectors used by training and error ranking.  This is
# intentionally separate from HAND_CONNECTIONS, whose palm links are optimized
# for drawing a connected wireframe.
STRUCTURAL_BONES: Tuple[Tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
)


def validate_model_checkpoint_stage(
    config: Mapping[str, Any],
    required: bool = True,
) -> Optional[str]:
    """Return the declared model provenance after strict schema validation.

    Evaluation, export, and folder inference all consume an already-trained
    Hand Landmarker.  The checkpoint stage is therefore explicit provenance,
    not something inferred from a path or from the presence of a finetune
    dataset.
    """

    model = config.get("model", {})
    if not isinstance(model, Mapping):
        raise ValueError("model must be a mapping")
    stage = model.get("checkpoint_stage")
    if stage in (None, "") and not required:
        return None
    if not isinstance(stage, str) or stage not in MODEL_CHECKPOINT_STAGES:
        raise ValueError(
            "model.checkpoint_stage must be exactly one of {}; got {!r}".format(
                list(MODEL_CHECKPOINT_STAGES), stage
            )
        )
    return stage


def validate_checkpoint_path_stage(
    config: Mapping[str, Any],
    checkpoint_path: Any,
) -> str:
    """Reject an explicit checkpoint path that names the opposite stage.

    Stage-less custom paths are intentionally valid.  A marker is considered
    explicit when ``pretrain`` or ``finetune`` is a standalone token within a
    slash-delimited path component, including common ``_``/``-`` filename
    separators.
    """

    declared_stage = validate_model_checkpoint_stage(config)
    components = [
        component.lower()
        for component in re.split(r"[\\/]+", str(checkpoint_path))
        if component
    ]
    markers = set()
    for component in components:
        for candidate in MODEL_CHECKPOINT_STAGES:
            if re.search(
                r"(^|[^a-z0-9]){}([^a-z0-9]|$)".format(re.escape(candidate)),
                component,
            ):
                markers.add(candidate)
    conflicting = sorted(markers - {declared_stage})
    if conflicting:
        raise ValueError(
            "Checkpoint path {!r} explicitly names stage(s) {} but "
            "model.checkpoint_stage is {!r}".format(
                str(checkpoint_path), conflicting, declared_stage
            )
        )
    return declared_stage


def ordered_landmarks(row: Mapping[str, Any], key: str = "landmarks_crop_norm") -> List[Tuple[float, float]]:
    """Return 21 landmarks in MediaPipe order, rejecting malformed records."""

    raw = list(row.get(key) or [])
    if len(raw) != 21:
        raise ValueError("{} must contain exactly 21 landmarks; got {}".format(key, len(raw)))
    by_id: Dict[int, Tuple[float, float]] = {}
    for index, point in enumerate(raw):
        point_id = int(point.get("id", index))
        if point_id in by_id:
            raise ValueError("{} contains duplicate landmark id {}".format(key, point_id))
        by_id[point_id] = (float(point["x"]), float(point["y"]))
    if set(by_id) != set(range(21)):
        raise ValueError("{} landmark ids must be exactly 0..20".format(key))
    return [by_id[index] for index in range(21)]


def validate_label_record(row: Mapping[str, Any], split: str = "train") -> List[str]:
    """Validate the canonical HandLandmarkerFab JSONL schema for one row."""

    errors: List[str] = []
    record_id = row.get("global_crop_id") or row.get("crop_id")
    if not record_id:
        errors.append("missing crop_id/global_crop_id")
    if not row.get("crop_path"):
        errors.append("missing crop_path")
    if int(row.get("width", 256)) != 256 or int(row.get("height", 256)) != 256:
        errors.append("ROI size must be 256x256")

    presence = row.get("hand_presence") or {}
    if "present" not in presence:
        errors.append("missing hand_presence.present")
        present = False
    else:
        present = bool(presence.get("present"))

    landmarks = list(row.get("landmarks_crop_norm") or [])
    if present:
        try:
            normalized = ordered_landmarks(row)
            if not all(math.isfinite(value) for point in normalized for value in point):
                errors.append("landmarks_crop_norm contains NaN/Inf")
            if split in {"val", "test"} and any(
                value < 0.0 or value > 1.0 for point in normalized for value in point
            ):
                errors.append("Gold landmarks_crop_norm must be within [0,1]")
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(str(exc))
        handedness = str((row.get("handedness") or {}).get("label", "unknown")).lower()
        if handedness not in {"left", "right", "unknown"}:
            errors.append("invalid handedness label: {}".format(handedness))
        if split in {"val", "test"} and handedness not in {"left", "right"}:
            errors.append("Gold positive must have Left/Right handedness")
        for point_key in ("landmarks_crop_px", "landmarks_image_px"):
            points = list(row.get(point_key) or [])
            if len(points) != 21:
                errors.append("{} must contain exactly 21 landmarks".format(point_key))
    elif landmarks:
        errors.append("negative sample must not contain landmarks")
    elif any(row.get(key) for key in ("landmarks_crop_px", "landmarks_image_px")):
        errors.append("negative sample must have empty landmark arrays")
    if not present and str((row.get("handedness") or {}).get("label", "unknown")).lower() != "unknown":
        errors.append("negative sample handedness must be unknown")

    if row.get("ignore_for_training"):
        errors.append("ignored record was included in canonical labels")
    if split in {"val", "test"}:
        if row.get("ground_truth_valid") is not True:
            errors.append("evaluation record must have ground_truth_valid=true")
        if row.get("palm_valid") is not True:
            errors.append("evaluation record must be produced by a valid frozen Palm detection")
        if str(row.get("split", "")).lower() != split:
            errors.append("evaluation record split does not match {}".format(split))
    return errors


def normalize_supervision_tier_loss_weights(
    value: Any = None,
) -> Dict[str, float]:
    """Return the strict Gold/pseudo multipliers used by finetune losses.

    These multipliers are deliberately independent from ``sampling_weight``
    and ``training.gold_fraction``.  Both tiers must remain active; setting a
    multiplier to zero would silently bypass the mandatory replay contract.
    """

    if value is None:
        return {"gold": 1.0, "pseudo": 1.0}
    if not isinstance(value, Mapping) or set(value) != {"gold", "pseudo"}:
        raise ValueError(
            "losses.supervision_tier_weights must contain exactly gold and pseudo"
        )
    normalized = {str(tier): float(weight) for tier, weight in value.items()}
    for tier, weight in normalized.items():
        if not math.isfinite(weight) or weight <= 0.0:
            raise ValueError(
                "losses.supervision_tier_weights.{} must be finite and positive".format(
                    tier
                )
            )
    return normalized


def effective_head_weights(
    row: Mapping[str, Any],
    supervision_tier_weights: Any = None,
) -> Tuple[float, float, float]:
    """Return presence, landmark, handedness loss weights for a canonical row.

    ``sampling_weight`` deliberately does not participate here; it belongs only
    to the sampler, as required by the finalization contract.
    """

    supervision = float(row.get("supervision_loss_weight", 1.0))
    if supervision_tier_weights is not None:
        tier_weights = normalize_supervision_tier_loss_weights(supervision_tier_weights)
        tier = str(row.get("supervision_tier") or "").lower()
        if tier not in tier_weights:
            raise ValueError("unsupported supervision_tier for loss weighting: {!r}".format(tier))
        supervision *= tier_weights[tier]
    presence = (
        float(row.get("hand_presence_loss_weight", 1.0))
        * supervision
        * float(row.get("presence_quality_weight", 1.0))
    )
    landmark = (
        float(row.get("landmark_loss_weight", 1.0))
        * supervision
        * float(row.get("landmark_quality_weight", 1.0))
    )
    handedness = (
        float(row.get("handedness_loss_weight", 1.0))
        * supervision
        * float(row.get("handedness_quality_weight", 1.0))
    )
    if not bool((row.get("hand_presence") or {}).get("present", False)):
        landmark = 0.0
        handedness = 0.0
    if str((row.get("handedness") or {}).get("label", "unknown")).lower() not in {"left", "right"}:
        handedness = 0.0
    return presence, landmark, handedness
