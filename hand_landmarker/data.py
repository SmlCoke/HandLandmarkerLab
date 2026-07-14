"""Canonical JSONL data loading and deterministic Keras ``Sequence`` input.

TensorFlow is optional at import time so contract tests can run before the
server environment is created.  When TensorFlow 2.9 is present the sequence
inherits its real ``keras.utils.Sequence`` class.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from .config import load_config
from .contracts import effective_head_weights
from .inspect import EVALUATION_SCHEMA, DatasetContractError, audit_canonical_dataset, leakage_report
from .io_utils import read_image, to_uint8_gray
from .pretrain_curation import verify_curation_manifest


try:  # Keep ``make inspect`` independent of TensorFlow by importing this module only for training.
    from tensorflow.keras.utils import Sequence as _KerasSequence
except ImportError:  # pragma: no cover - used by lightweight local contract tests.
    class _KerasSequence(object):
        pass


OUTPUT_ORDER = ("landmarks", "hand_flag", "handedness")
CANONICAL_SAMPLE_TYPES = (
    "POS_RUNTIME",
    "POS_LOW_PALM",
    "NEG_RUNTIME_CANDIDATE",
    "NEG_LOW_PALM_CANDIDATE",
)


def _config_mapping(config: Union[Mapping[str, Any], str, Path]) -> Dict[str, Any]:
    return load_config(config) if isinstance(config, (str, Path)) else dict(config)


def _dotted_get(row: Mapping[str, Any], path: str) -> Any:
    current: Any = row
    for part in str(path).split("."):
        if not isinstance(current, Mapping) or part not in current:
            raise DatasetContractError("Missing configured target field {} in {}".format(path, row.get("crop_id")))
        current = current[part]
    return current


def _ordered_landmarks(row: Mapping[str, Any], field: str) -> np.ndarray:
    raw = list(row.get(field) or [])
    if len(raw) != 21:
        raise DatasetContractError("{} must contain 21 points for {}".format(field, row.get("crop_id")))
    by_id: Dict[int, Tuple[float, float]] = {}
    for offset, point in enumerate(raw):
        point_id = int(point.get("id", offset))
        if point_id in by_id:
            raise DatasetContractError("Duplicate landmark id {} in {}".format(point_id, row.get("crop_id")))
        by_id[point_id] = (float(point["x"]), float(point["y"]))
    if set(by_id) != set(range(21)):
        raise DatasetContractError("Landmark ids must be exactly 0..20 in {}".format(row.get("crop_id")))
    return np.asarray([by_id[index] for index in range(21)], dtype=np.float32)


def _handedness_target(
    row: Mapping[str, Any], encoding: Mapping[str, Any], field: str = "handedness.label"
) -> Tuple[float, bool]:
    label = str(_dotted_get(row, field))
    value = encoding.get(label)
    if value is None:
        return 0.0, False
    return float(value), True


def augment_image_and_targets(
    image: np.ndarray,
    landmarks_norm: np.ndarray,
    handedness_value: float,
    present: bool,
    augmentation: Mapping[str, Any],
    rng: np.random.RandomState,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Apply synchronized crop-space affine/flip and photometric augmentation.

    Returned image values are in ``[0,1]``.  Landmark coordinates are not
    clipped: clipping a transformed point to the border would fabricate a
    different training target.
    """

    raw_image = np.asarray(image)
    if raw_image.ndim != 2:
        raise ValueError("augment_image_and_targets expects a 2-D grayscale image")
    if not np.all(np.isfinite(raw_image)):
        raise ValueError("image values must be finite")
    minimum = float(np.min(raw_image))
    maximum = float(np.max(raw_image))
    if minimum < 0.0 or maximum > 255.0:
        raise ValueError("image values must be in [0,255]")
    gray = raw_image.astype(np.float32)
    if np.issubdtype(raw_image.dtype, np.integer):
        # Integer images are raw pixels even when an extremely dark crop only
        # happens to contain values 0 and 1.
        gray = gray / 255.0
    elif maximum > 1.0:
        gray = gray / 255.0
    points = np.asarray(landmarks_norm, dtype=np.float32).copy()
    if not bool(augmentation.get("enabled", False)):
        return gray, points, float(handedness_value)

    height, width = gray.shape
    rotation = float(augmentation.get("rotation_degrees", 0.0))
    if rotation < 0.0:
        raise ValueError("augmentation.rotation_degrees must be non-negative")
    scale_value = augmentation.get("scale_range", [1.0, 1.0])
    if not isinstance(scale_value, (list, tuple)) or len(scale_value) != 2:
        raise ValueError("augmentation.scale_range must contain [min,max]")
    scale_low, scale_high = float(scale_value[0]), float(scale_value[1])
    translation = float(augmentation.get("translation_fraction", 0.0))
    if scale_low <= 0.0 or scale_high < scale_low:
        raise ValueError("augmentation.scale_range must satisfy 0 < min <= max")
    if translation < 0.0:
        raise ValueError("augmentation.translation_fraction must be non-negative")
    use_affine = rotation > 0.0 or scale_low != 1.0 or scale_high != 1.0 or translation > 0.0
    if use_affine:
        try:
            import cv2
        except ImportError as exc:  # pragma: no cover - server dependency error.
            raise RuntimeError("OpenCV is required for affine data augmentation") from exc
        angle = float(rng.uniform(-rotation, rotation)) if rotation > 0.0 else 0.0
        scale = float(rng.uniform(scale_low, scale_high))
        tx = float(rng.uniform(-translation, translation) * (width - 1)) if translation > 0.0 else 0.0
        ty = float(rng.uniform(-translation, translation) * (height - 1)) if translation > 0.0 else 0.0
        matrix = cv2.getRotationMatrix2D(((width - 1) / 2.0, (height - 1) / 2.0), angle, scale)
        matrix[0, 2] += tx
        matrix[1, 2] += ty
        gray = cv2.warpAffine(
            gray,
            matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0.0,
        )
        if present:
            pixel = points * np.asarray([width - 1, height - 1], dtype=np.float32)
            homogeneous = np.concatenate([pixel, np.ones((len(pixel), 1), dtype=np.float32)], axis=1)
            transformed = np.matmul(homogeneous, np.asarray(matrix, dtype=np.float32).T)
            points = transformed / np.asarray([width - 1, height - 1], dtype=np.float32)

    flip_probability = float(augmentation.get("horizontal_flip_probability", 0.0))
    if flip_probability < 0.0 or flip_probability > 1.0:
        raise ValueError("horizontal_flip_probability must be in [0,1]")
    if flip_probability > 0.0 and float(rng.random_sample()) < flip_probability:
        gray = np.ascontiguousarray(gray[:, ::-1])
        if present:
            points[:, 0] = 1.0 - points[:, 0]
            handedness_value = 1.0 - float(handedness_value)

    contrast_value = augmentation.get("contrast_range", [1.0, 1.0])
    if not isinstance(contrast_value, (list, tuple)) or len(contrast_value) != 2:
        raise ValueError("augmentation.contrast_range must contain [min,max]")
    contrast_low, contrast_high = float(contrast_value[0]), float(contrast_value[1])
    if contrast_low <= 0.0 or contrast_high < contrast_low:
        raise ValueError("augmentation.contrast_range must satisfy 0 < min <= max")
    contrast = float(rng.uniform(contrast_low, contrast_high))
    mean = float(np.mean(gray))
    gray = (gray - mean) * contrast + mean
    brightness = float(augmentation.get("brightness_delta", 0.0))
    if brightness < 0.0:
        raise ValueError("augmentation.brightness_delta must be non-negative")
    if brightness > 0.0:
        gray = gray + float(rng.uniform(-brightness, brightness))
    noise = float(augmentation.get("gaussian_noise_stddev", 0.0))
    if noise < 0.0:
        raise ValueError("augmentation.gaussian_noise_stddev must be non-negative")
    if noise > 0.0:
        gray = gray + rng.normal(0.0, noise, size=gray.shape).astype(np.float32)
    return np.clip(gray, 0.0, 1.0).astype(np.float32), points.astype(np.float32), float(handedness_value)


class WeightedStratifiedSampler:
    """Exact per-batch tier/type quotas, then weighted rows inside each cell."""

    def __init__(
        self,
        records: Sequence[Mapping[str, Any]],
        stage: str,
        seed: int,
        weight_key: str = "sampling_weight",
        gold_fraction: Optional[float] = None,
        supervision_fractions: Optional[Mapping[str, Any]] = None,
        sample_type_fractions: Optional[Mapping[str, Any]] = None,
        tier_key: str = "supervision_tier",
        bucket_key: str = "sampling_bucket",
        sample_type_key: str = "sample_type",
        quota_tie_break: Sequence[str] = CANONICAL_SAMPLE_TYPES,
        require_all_tier_sample_type_cells: bool = True,
    ) -> None:
        if not records:
            raise DatasetContractError("Cannot sample an empty canonical dataset")
        self.records = records
        self.stage = str(stage)
        if self.stage not in {"pretrain", "finetune"}:
            raise DatasetContractError("stage must be pretrain or finetune")
        self.seed = int(seed)
        self.weight_key = str(weight_key)
        self.gold_fraction = None if gold_fraction is None else float(gold_fraction)
        self.tier_key = str(tier_key)
        self.bucket_key = str(bucket_key)
        self.sample_type_key = str(sample_type_key)
        self.quota_tie_break = tuple(str(value) for value in quota_tie_break)
        if self.quota_tie_break != CANONICAL_SAMPLE_TYPES:
            raise DatasetContractError(
                "sampling.quota_tie_break must be {}".format(list(CANONICAL_SAMPLE_TYPES))
            )
        if require_all_tier_sample_type_cells is not True:
            raise DatasetContractError(
                "sampling.require_all_tier_sample_type_cells must remain true"
            )
        self.supervision_fractions = {
            str(key): float(value) for key, value in dict(supervision_fractions or {}).items()
        }
        if sample_type_fractions is None:
            raise DatasetContractError("sampling.sample_type_fractions is required")
        try:
            raw_sample_type_fractions = {
                str(key): Fraction(str(value))
                for key, value in dict(sample_type_fractions).items()
            }
        except (TypeError, ValueError, ZeroDivisionError) as exc:
            raise DatasetContractError(
                "sampling.sample_type_fractions must contain finite numeric values"
            ) from exc
        if set(raw_sample_type_fractions) != set(CANONICAL_SAMPLE_TYPES):
            raise DatasetContractError(
                "sampling.sample_type_fractions must define exactly {}".format(
                    list(CANONICAL_SAMPLE_TYPES)
                )
            )
        if any(value < 0 for value in raw_sample_type_fractions.values()):
            raise DatasetContractError("sampling.sample_type_fractions must be non-negative")
        if sum(raw_sample_type_fractions.values(), Fraction(0, 1)) != Fraction(1, 1):
            raise DatasetContractError("sampling.sample_type_fractions must sum exactly to 1")
        self.sample_type_fractions = {
            name: raw_sample_type_fractions[name] for name in CANONICAL_SAMPLE_TYPES
        }
        self.groups: Dict[str, Dict[str, List[int]]] = defaultdict(lambda: defaultdict(list))
        try:
            self.weights = np.asarray([float(row[self.weight_key]) for row in records], dtype=np.float64)
        except (KeyError, TypeError, ValueError) as exc:
            raise DatasetContractError(
                "Every canonical row must provide numeric {}".format(self.weight_key)
            ) from exc
        for index, row in enumerate(records):
            tier = str(row.get(self.tier_key))
            sample_type = str(row.get(self.sample_type_key))
            if sample_type not in CANONICAL_SAMPLE_TYPES:
                raise DatasetContractError(
                    "Unsupported sample_type {} in {}".format(sample_type, row.get("crop_id"))
                )
            expected_bucket = "{}:{}".format(tier, sample_type)
            if row.get(self.bucket_key) != expected_bucket:
                raise DatasetContractError(
                    "{} must equal {!r} for {}".format(
                        self.bucket_key, expected_bucket, row.get("crop_id")
                    )
                )
            self.groups[tier][sample_type].append(index)
        if set(self.groups) - {"pseudo", "gold"}:
            raise DatasetContractError("Unsupported supervision tiers: {}".format(sorted(self.groups)))
        if self.stage == "pretrain" and set(self.groups) != {"pseudo"}:
            raise DatasetContractError(
                "Pretrain requires supervision_tier=pseudo only; got {}".format(
                    sorted(self.groups)
                )
            )
        if self.stage == "finetune":
            if self.gold_fraction is None:
                raise DatasetContractError("training.gold_fraction is required for finetune")
            if not 0.30 <= self.gold_fraction <= 0.50:
                raise DatasetContractError("training.gold_fraction must be within [0.30, 0.50]")
            if "gold" not in self.groups or "pseudo" not in self.groups:
                raise DatasetContractError("Finetune requires both gold and pseudo supervision tiers")
        self._validate_groups()

    def _validate_groups(self) -> None:
        if np.any(~np.isfinite(self.weights)) or np.any(self.weights < 0.0):
            raise DatasetContractError("sampling_weight must be finite and non-negative")
        for tier in self._tier_order():
            for sample_type in CANONICAL_SAMPLE_TYPES:
                if self.sample_type_fractions[sample_type] <= 0:
                    continue
                indices = self.groups[tier].get(sample_type, [])
                if not indices:
                    raise DatasetContractError(
                        "Missing canonical sampling cell supervision_tier={!r}, sample_type={!r}; "
                        "configured positive fractions require every active tier x sample_type cell".format(
                            tier, sample_type
                        )
                    )
                if float(np.sum(self.weights[indices])) <= 0.0:
                    raise DatasetContractError(
                        "Sampling cell supervision_tier={!r}, sample_type={!r} has no positive {}".format(
                            tier, sample_type, self.weight_key
                        )
                    )

    def _tier_order(self) -> List[str]:
        if "gold" in self.groups and "pseudo" in self.groups:
            return ["gold", "pseudo"]
        return sorted(self.groups)

    @staticmethod
    def _largest_remainder_counts(
        count: int,
        fractions: Mapping[str, Fraction],
        order: Sequence[str],
    ) -> Dict[str, int]:
        raw = {name: Fraction(count, 1) * fractions[name] for name in order}
        result = {name: int(raw[name].numerator // raw[name].denominator) for name in order}
        remaining = int(count) - sum(result.values())
        ranked = sorted(
            order,
            key=lambda name: (-(raw[name] - result[name]), order.index(name)),
        )
        for name in ranked[:remaining]:
            result[name] += 1
        return result

    @staticmethod
    def _bounded_proportional_counts(
        count: int,
        capacities: Mapping[str, int],
        order: Sequence[str],
    ) -> Dict[str, int]:
        available = sum(int(capacities[name]) for name in order)
        if count < 0 or count > available:
            raise DatasetContractError(
                "Infeasible cross-bucket quota: requested {} from capacity {}".format(count, available)
            )
        if available == 0:
            return {name: 0 for name in order}
        raw = {
            name: Fraction(int(capacities[name]) * int(count), int(available))
            for name in order
        }
        result = {
            name: min(
                int(capacities[name]),
                int(raw[name].numerator // raw[name].denominator),
            )
            for name in order
        }
        remaining = int(count) - sum(result.values())
        ranked = sorted(
            order,
            key=lambda name: (-(raw[name] - result[name]), order.index(name)),
        )
        for name in ranked:
            if remaining <= 0:
                break
            if result[name] < int(capacities[name]):
                result[name] += 1
                remaining -= 1
        if remaining:
            raise DatasetContractError("Could not construct a feasible cross-bucket quota")
        return result

    def _tier_counts(self, count: int) -> Dict[str, int]:
        tiers = self._tier_order()
        if self.stage == "finetune":
            assert self.gold_fraction is not None
            gold_count = int(math.floor(float(count) * self.gold_fraction + 0.5))
            gold_count = max(0, min(count, gold_count))
            return {"gold": gold_count, "pseudo": count - gold_count}
        if len(tiers) == 1:
            return {tiers[0]: count}
        if self.supervision_fractions:
            missing = [tier for tier in tiers if tier not in self.supervision_fractions]
            if missing:
                raise DatasetContractError("Missing supervision fraction(s): {}".format(missing))
            try:
                values = {tier: Fraction(str(self.supervision_fractions[tier])) for tier in tiers}
            except (TypeError, ValueError, ZeroDivisionError) as exc:
                raise DatasetContractError(
                    "supervision_fractions must contain finite numeric values"
                ) from exc
        else:
            values = {tier: Fraction(1, len(tiers)) for tier in tiers}
        if any(value < 0 for value in values.values()) or sum(values.values(), Fraction(0, 1)) <= 0:
            raise DatasetContractError("supervision_fractions must be finite, non-negative, and non-zero")
        total = sum(values.values(), Fraction(0, 1))
        normalized = {tier: values[tier] / total for tier in tiers}
        return self._largest_remainder_counts(count, normalized, tiers)

    def batch_quota(self, count: int) -> Dict[str, Dict[str, int]]:
        """Return exact tier x sample-type integer quotas for one batch."""

        if count < 0:
            raise ValueError("count must be non-negative")
        tiers = self._tier_order()
        tier_counts = self._tier_counts(int(count))
        type_counts = self._largest_remainder_counts(
            int(count), self.sample_type_fractions, CANONICAL_SAMPLE_TYPES
        )
        remaining_types = dict(type_counts)
        remaining_total = int(count)
        quota: Dict[str, Dict[str, int]] = {}
        for tier in tiers[:-1]:
            tier_count = int(tier_counts[tier])
            quota[tier] = self._bounded_proportional_counts(
                tier_count, remaining_types, CANONICAL_SAMPLE_TYPES
            )
            for sample_type in CANONICAL_SAMPLE_TYPES:
                remaining_types[sample_type] -= quota[tier][sample_type]
            remaining_total -= tier_count
        if tiers:
            last_tier = tiers[-1]
            if sum(remaining_types.values()) != int(tier_counts[last_tier]) or remaining_total != int(
                tier_counts[last_tier]
            ):
                raise DatasetContractError("Internal tier/type quota conservation failure")
            quota[last_tier] = dict(remaining_types)
        return quota

    def sample(self, count: int, epoch: int = 0, stream: int = 0) -> np.ndarray:
        if count <= 0:
            return np.empty((0,), dtype=np.int64)
        rng = np.random.RandomState(
            (self.seed + int(epoch) * 1000003 + int(stream) * 9176) % (2 ** 32 - 1)
        )
        quota = self.batch_quota(int(count))
        cells: List[Tuple[str, str]] = []
        for tier in self._tier_order():
            for sample_type in CANONICAL_SAMPLE_TYPES:
                cells.extend([(tier, sample_type)] * int(quota[tier][sample_type]))
        rng.shuffle(cells)
        result = np.empty((count,), dtype=np.int64)
        for position, (tier, sample_type) in enumerate(cells):
            indices = np.asarray(self.groups[tier][sample_type], dtype=np.int64)
            row_weights = self.weights[indices]
            result[position] = int(rng.choice(indices, p=row_weights / float(np.sum(row_weights))))
        return result

    def report(self) -> Dict[str, Any]:
        return {
            "stage": self.stage,
            "weight_key": self.weight_key,
            "tier_key": self.tier_key,
            "bucket_key": self.bucket_key,
            "sample_type_key": self.sample_type_key,
            "gold_fraction": self.gold_fraction,
            "gold_fraction_scope": "per_batch_half_up" if self.stage == "finetune" else None,
            "gold_fraction_rounding": (
                "nearest_feasible_count_half_up_per_batch" if self.stage == "finetune" else None
            ),
            "sample_type_fractions": {
                name: float(self.sample_type_fractions[name]) for name in CANONICAL_SAMPLE_TYPES
            },
            "sample_type_fraction_scope": "per_batch_largest_remainder",
            "quota_tie_break": list(CANONICAL_SAMPLE_TYPES),
            "cross_bucket_policy": "require_every_positive_type_in_every_active_tier",
            "row_sampling": "sampling_weight_within_selected_tier_and_sample_type_only",
            "tiers": {
                tier: {sample_type: len(indices) for sample_type, indices in sorted(types.items())}
                for tier, types in sorted(self.groups.items())
            },
        }


class CanonicalSequence(_KerasSequence):
    """Keras sequence returning ``x, y_list, effective_weight_list``."""

    def __init__(
        self,
        records: Sequence[Mapping[str, Any]],
        dataset_config: Mapping[str, Any],
        targets_config: Mapping[str, Any],
        batch_size: int,
        training: bool,
        stage: Optional[str] = None,
        seed: int = 0,
        steps_per_epoch: Optional[int] = None,
        augmentation_config: Optional[Mapping[str, Any]] = None,
        training_config: Optional[Mapping[str, Any]] = None,
        sampling_config: Optional[Mapping[str, Any]] = None,
        output_order: Sequence[str] = OUTPUT_ORDER,
    ) -> None:
        self.records = [dict(row) for row in records]
        if not self.records:
            raise DatasetContractError("CanonicalSequence requires at least one record")
        self.dataset_config = dict(dataset_config)
        self.targets_config = dict(targets_config)
        self.batch_size = int(batch_size)
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.training = bool(training)
        self.stage = str(stage or "")
        self.seed = int(seed)
        self.augmentation = dict(augmentation_config or {})
        self.training_config = dict(training_config or {})
        self.sampling_config = dict(sampling_config or {})
        self.output_order = tuple(str(value) for value in output_order)
        if self.output_order != OUTPUT_ORDER:
            raise DatasetContractError(
                "model.output_order must be {}; got {}".format(list(OUTPUT_ORDER), list(self.output_order))
            )
        self.landmark_field = str(self.targets_config.get("landmark_field", "landmarks_crop_norm"))
        self.presence_field = str(self.targets_config.get("presence_field", "hand_presence.present"))
        self.handedness_field = str(self.targets_config.get("handedness_field", "handedness.label"))
        if str(self.targets_config.get("landmark_space", "normalized_crop_xy")) != "normalized_crop_xy":
            raise DatasetContractError("targets.landmark_space must be normalized_crop_xy")
        if str(
            self.targets_config.get("landmark_order", "id_0_to_20_interleaved_xy")
        ) != "id_0_to_20_interleaved_xy":
            raise DatasetContractError("targets.landmark_order must be id_0_to_20_interleaved_xy")
        self.handedness_encoding = dict(
            self.targets_config.get("handedness_encoding", {"Left": 0, "Right": 1, "unknown": None})
        )
        if (
            self.handedness_encoding.get("Left") != 0
            or self.handedness_encoding.get("Right") != 1
            or self.handedness_encoding.get("unknown") is not None
        ):
            raise DatasetContractError("handedness encoding must be Left=0, Right=1, unknown=null")
        if int(self.targets_config.get("num_landmarks", 21)) != 21:
            raise DatasetContractError("targets.num_landmarks must remain 21")
        for row in self.records:
            presence_value = _dotted_get(row, self.presence_field)
            if not isinstance(presence_value, bool):
                raise DatasetContractError(
                    "Configured presence target must be boolean in {}".format(row.get("crop_id"))
                )
            _handedness_target(row, self.handedness_encoding, self.handedness_field)
            if presence_value:
                _ordered_landmarks(row, self.landmark_field)
        self.input_scale = float(self.dataset_config.get("input_scale", 1.0 / 255.0))
        self.input_offset = float(self.dataset_config.get("input_offset", 0.0))
        if not math.isclose(self.input_scale, 1.0 / 255.0, rel_tol=0.0, abs_tol=1.0e-12):
            raise DatasetContractError("data.input_scale must equal 1/255")
        if not math.isclose(self.input_offset, 0.0, rel_tol=0.0, abs_tol=1.0e-12):
            raise DatasetContractError("data.input_offset must equal 0")
        image_size = self.dataset_config.get("image_size", [256, 256])
        if not isinstance(image_size, (list, tuple)) or len(image_size) != 2:
            raise DatasetContractError("data.image_size must remain [256, 256]")
        self.expected_width, self.expected_height = [int(value) for value in image_size]
        if (self.expected_width, self.expected_height) != (256, 256):
            raise DatasetContractError("data.image_size must remain [256, 256]")
        if str(self.dataset_config.get("input_layout", "NCHW")) != "NCHW":
            raise DatasetContractError("Only NCHW input_layout is compatible with the board model")
        if str(self.dataset_config.get("input_dtype", "float32")) != "float32":
            raise DatasetContractError("Only float32 input_dtype is supported")
        if str(self.dataset_config.get("color_mode", "grayscale")) != "grayscale":
            raise DatasetContractError("Only grayscale color_mode is supported")
        if int(self.dataset_config.get("channels", 1)) != 1:
            raise DatasetContractError("Only one-channel grayscale crops are supported")
        self.cache_enabled = bool(self.dataset_config.get("cache", False))
        self._cache: Dict[str, np.ndarray] = {}
        self.epoch = 0
        configured_epoch_size = self.sampling_config.get("epoch_size")
        if steps_per_epoch not in (None, 0):
            self.steps = int(steps_per_epoch)
            self.epoch_size = self.steps * self.batch_size
        else:
            self.epoch_size = (
                int(configured_epoch_size)
                if configured_epoch_size not in (None, 0)
                else len(self.records)
            )
            self.steps = int(math.ceil(self.epoch_size / float(self.batch_size)))
        if self.steps <= 0:
            raise ValueError("steps_per_epoch must be positive")
        if self.epoch_size <= 0:
            raise ValueError("sampling.epoch_size must be positive")
        self.sampler: Optional[WeightedStratifiedSampler] = None
        if self.training:
            if self.sampling_config.get("enabled", True) is not True:
                raise DatasetContractError("sampling.enabled must remain true for canonical training")
            if self.sampling_config.get("replacement", True) is not True:
                raise DatasetContractError("Weighted canonical sampling requires sampling.replacement=true")
            if self.sampling_config.get("honor_record_sampling_weight", True) is not True:
                raise DatasetContractError("sampling.honor_record_sampling_weight must remain true")
            if self.sampling_config.get("quota_rounding", "largest_remainder") != "largest_remainder":
                raise DatasetContractError("sampling.quota_rounding must be largest_remainder")
            configured_strata = self.sampling_config.get("stratify_by")
            if configured_strata is not None:
                expected_strata = [
                    str(self.sampling_config.get("tier_key", "supervision_tier")),
                    str(self.sampling_config.get("sample_type_key", "sample_type")),
                ]
                if list(configured_strata) != expected_strata:
                    raise DatasetContractError(
                        "sampling.stratify_by must be {}; got {}".format(expected_strata, configured_strata)
                    )
            self.sampler = WeightedStratifiedSampler(
                self.records,
                stage=self.stage,
                seed=self.seed,
                weight_key=str(
                    self.sampling_config.get(
                        "weight_key", self.dataset_config.get("sampling_weight_key", "sampling_weight")
                    )
                ),
                gold_fraction=self.sampling_config.get(
                    "gold_fraction", self.training_config.get("gold_fraction")
                ),
                supervision_fractions=self.sampling_config.get(
                    "supervision_fractions", self.training_config.get("supervision_fractions")
                ),
                sample_type_fractions=self.sampling_config.get("sample_type_fractions"),
                tier_key=str(self.sampling_config.get("tier_key", "supervision_tier")),
                bucket_key=str(self.sampling_config.get("bucket_key", "sampling_bucket")),
                sample_type_key=str(self.sampling_config.get("sample_type_key", "sample_type")),
                quota_tie_break=self.sampling_config.get(
                    "quota_tie_break", CANONICAL_SAMPLE_TYPES
                ),
                require_all_tier_sample_type_cells=self.sampling_config.get(
                    "require_all_tier_sample_type_cells", True
                ),
            )
            self.set_epoch(0)
        else:
            self.steps = int(math.ceil(len(self.records) / float(self.batch_size)))
            self.epoch_indices = np.arange(len(self.records), dtype=np.int64)

    def __len__(self) -> int:
        return self.steps

    def on_epoch_end(self) -> None:
        if self.training and self.sampler is not None:
            self.set_epoch(self.epoch + 1)

    def set_epoch(self, epoch: int) -> None:
        """Set the absolute zero-based epoch for reproducible resume behavior."""

        if isinstance(epoch, bool) or int(epoch) != epoch or int(epoch) < 0:
            raise ValueError("epoch must be a non-negative integer")
        self.epoch = int(epoch)
        if self.training and self.sampler is not None:
            self.epoch_indices = self._sample_epoch_indices(epoch=self.epoch)

    def _sample_epoch_indices(self, epoch: int) -> np.ndarray:
        if self.sampler is None:
            raise RuntimeError("Training sampler is not initialized")
        batches: List[np.ndarray] = []
        remaining = self.epoch_size
        batch_number = 0
        while remaining > 0:
            current_size = min(self.batch_size, remaining)
            batches.append(self.sampler.sample(current_size, epoch=epoch, stream=batch_number))
            remaining -= current_size
            batch_number += 1
        return np.concatenate(batches) if batches else np.empty((0,), dtype=np.int64)

    def batch_record_indices(self, batch_index: int) -> np.ndarray:
        if batch_index < 0 or batch_index >= len(self):
            raise IndexError(batch_index)
        start = batch_index * self.batch_size
        stop = start + self.batch_size
        return self.epoch_indices[start:stop]

    def _load_gray(self, row: Mapping[str, Any]) -> np.ndarray:
        path = str(row["_resolved_crop_path"])
        if self.cache_enabled and path in self._cache:
            return self._cache[path].copy()
        image = read_image(path)
        if image is None:
            raise DatasetContractError("Could not read canonical crop: {}".format(path))
        gray = to_uint8_gray(image)
        if gray.shape != (self.expected_height, self.expected_width):
            raise DatasetContractError(
                "Canonical crop {} must be {}x{}; got {}".format(
                    path, self.expected_width, self.expected_height, gray.shape
                )
            )
        if self.cache_enabled:
            self._cache[path] = gray.copy()
        return gray

    def __getitem__(self, batch_index: int):
        indices = self.batch_record_indices(batch_index)
        images: List[np.ndarray] = []
        landmark_targets: List[np.ndarray] = []
        presence_targets: List[List[float]] = []
        handedness_targets: List[List[float]] = []
        presence_weights: List[float] = []
        landmark_weights: List[float] = []
        handedness_weights: List[float] = []
        batch_rng = np.random.RandomState(
            (self.seed + self.epoch * 1000003 + int(batch_index) * 9176 + 17) % (2 ** 32 - 1)
        )
        for record_index in indices:
            row = self.records[int(record_index)]
            present_value = _dotted_get(row, self.presence_field)
            if not isinstance(present_value, bool):
                raise DatasetContractError("Configured presence target must be boolean in {}".format(row.get("crop_id")))
            present = bool(present_value)
            points = _ordered_landmarks(row, self.landmark_field) if present else np.zeros((21, 2), dtype=np.float32)
            handedness_value, handedness_known = _handedness_target(
                row, self.handedness_encoding, self.handedness_field
            )
            gray = self._load_gray(row)
            sample_seed = int(batch_rng.randint(0, 2 ** 31 - 1))
            gray01, points, handedness_value = augment_image_and_targets(
                gray,
                points,
                handedness_value,
                present,
                self.augmentation if self.training else {"enabled": False},
                np.random.RandomState(sample_seed),
            )
            tensor = gray01 * (255.0 * self.input_scale) + self.input_offset
            images.append(tensor[np.newaxis, :, :].astype(np.float32))
            landmark_targets.append(points.reshape(42).astype(np.float32))
            presence_targets.append([1.0 if present else 0.0])
            handedness_targets.append([float(handedness_value) if handedness_known else 0.0])
            presence_weight, landmark_weight, handedness_weight = effective_head_weights(row)
            presence_weights.append(float(presence_weight))
            landmark_weights.append(float(landmark_weight))
            handedness_weights.append(float(handedness_weight))

        x_value = np.stack(images).astype(np.float32)
        y_values = [
            np.stack(landmark_targets).astype(np.float32),
            np.asarray(presence_targets, dtype=np.float32),
            np.asarray(handedness_targets, dtype=np.float32),
        ]
        sample_weights = [
            np.asarray(landmark_weights, dtype=np.float32),
            np.asarray(presence_weights, dtype=np.float32),
            np.asarray(handedness_weights, dtype=np.float32),
        ]
        return x_value, y_values, sample_weights

    def sampling_report(self) -> Dict[str, Any]:
        if self.sampler is None:
            return {"mode": "sequential", "records": len(self.records)}
        indices = [int(value) for value in self.epoch_indices]
        per_batch_tiers = []
        per_batch_sample_types = []
        per_batch_cross_cells = []
        for batch_index in range(len(self)):
            batch_indices = self.batch_record_indices(batch_index)
            per_batch_tiers.append(
                dict(
                    Counter(
                        str(self.records[int(index)].get("supervision_tier"))
                        for index in batch_indices
                    )
                )
            )
            per_batch_sample_types.append(
                dict(
                    Counter(
                        str(self.records[int(index)].get("sample_type"))
                        for index in batch_indices
                    )
                )
            )
            per_batch_cross_cells.append(
                {
                    "{}:{}".format(tier, sample_type): int(value)
                    for tier, type_counts in self.sampler.batch_quota(len(batch_indices)).items()
                    for sample_type, value in type_counts.items()
                }
            )
        return {
            "mode": "weighted_stratified",
            "absolute_epoch": self.epoch,
            "draws_per_epoch": len(indices),
            "drawn_supervision_tiers": dict(Counter(str(self.records[index].get("supervision_tier")) for index in indices)),
            "drawn_sample_types": dict(Counter(str(self.records[index].get("sample_type")) for index in indices)),
            "drawn_supervision_tiers_per_batch": per_batch_tiers,
            "drawn_sample_types_per_batch": per_batch_sample_types,
            "tier_sample_type_quota_per_batch": per_batch_cross_cells,
            "definition": self.sampler.report(),
        }


def _validation_dataset_config(config: Mapping[str, Any], dataset: Mapping[str, Any]) -> Dict[str, Any]:
    value = dict(dataset)
    validation = config.get("validation", {})
    value["labels"] = validation["labels"]
    if validation.get("ignored_labels"):
        value["ignored_labels"] = validation["ignored_labels"]
    else:
        value.pop("ignored_labels", None)
    value["require_schema_version"] = EVALUATION_SCHEMA
    value["require_split"] = "val"
    value.pop("require_training_stage", None)
    return value


def create_sequences(config: Union[Mapping[str, Any], str, Path]):
    """Build strict train/validation sequences and a serializable data report."""

    cfg = _config_mapping(config)
    data_value = cfg.get("data")
    dataset_value = cfg.get("dataset")
    if isinstance(data_value, Mapping) and isinstance(dataset_value, Mapping):
        if dict(data_value) != dict(dataset_value):
            raise DatasetContractError("config.data and config.dataset are both set but differ")
    dataset_cfg = dict(data_value or dataset_value or {})
    root_stage = str(cfg.get("stage") or "")
    data_stage = str(dataset_cfg.get("require_training_stage") or "")
    if root_stage and data_stage and root_stage != data_stage:
        raise DatasetContractError(
            "stage and data.require_training_stage conflict: {} != {}".format(
                root_stage, data_stage
            )
        )
    stage = data_stage or root_stage
    if stage not in {"pretrain", "finetune"}:
        raise DatasetContractError("create_sequences requires stage pretrain or finetune")
    targets_cfg = dict(cfg.get("targets", {}))
    training_cfg = dict(cfg.get("training", {}))
    sampling_cfg = dict(cfg.get("sampling", {}))
    experiment_cfg = dict(cfg.get("experiment", {}))
    seed = int(experiment_cfg.get("seed", 0))
    curation_manifest = verify_curation_manifest(
        cfg, dataset_cfg, error_type=DatasetContractError
    )
    train_records, train_report = audit_canonical_dataset(
        cfg,
        dataset=dataset_cfg,
        expected_stage=stage,
        check_images=True,
        hash_images=True,
        raise_on_error=True,
    )

    validation_records: Optional[List[Dict[str, Any]]] = None
    validation_report = None
    leakage_checks: List[Dict[str, Any]] = []
    validation_cfg = dict(cfg.get("validation", {}))
    if validation_cfg.get("enabled", False):
        if not validation_cfg.get("labels"):
            raise DatasetContractError("validation.labels is required when validation.enabled=true")
        val_dataset = _validation_dataset_config(cfg, dataset_cfg)
        val_records, validation_report = audit_canonical_dataset(
            cfg,
            dataset=val_dataset,
            expected_split="val",
            check_images=True,
            hash_images=True,
            raise_on_error=True,
        )
        validation_records = val_records
        leakage = leakage_report("train", train_records, "validation", val_records)
        leakage_checks.append(leakage)
        if leakage.get("status") != "ok":
            raise DatasetContractError(
                "Cross-split leakage detected before training: {}".format(leakage.get("fatal", []))
            )

    train_sequence = CanonicalSequence(
        train_records,
        dataset_config=dataset_cfg,
        targets_config=targets_cfg,
        batch_size=int(training_cfg.get("batch_size", 32)),
        training=True,
        stage=stage,
        seed=seed,
        steps_per_epoch=training_cfg.get("steps_per_epoch"),
        augmentation_config=cfg.get("augmentation", {}),
        training_config=training_cfg,
        sampling_config=sampling_cfg,
        output_order=cfg.get("model", {}).get("output_order", OUTPUT_ORDER),
    )

    validation_sequence = None
    if validation_records is not None:
        val_dataset = _validation_dataset_config(cfg, dataset_cfg)
        validation_sequence = CanonicalSequence(
            validation_records,
            dataset_config=val_dataset,
            targets_config=targets_cfg,
            batch_size=int(validation_cfg.get("batch_size", 128)),
            training=False,
            seed=seed,
            augmentation_config={"enabled": False},
            output_order=cfg.get("model", {}).get("output_order", OUTPUT_ORDER),
        )
    report = {
        "status": "ok",
        "stage": stage,
        "train": train_report,
        "validation": validation_report,
        "leakage": leakage_checks,
        "sampler": train_sequence.sampling_report(),
        "curation_manifest": curation_manifest,
        "tensor_contract": {
            "inputs": [None, 1, 256, 256],
            "targets": [[None, 42], [None, 1], [None, 1]],
            "target_order": list(OUTPUT_ORDER),
            "sample_weights": [[None], [None], [None]],
        },
    }
    return train_sequence, validation_sequence, report
