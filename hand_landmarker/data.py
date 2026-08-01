"""Canonical JSONL data loading and deterministic Keras ``Sequence`` input.

TensorFlow is optional at import time so contract tests can run before the
server environment is created.  When TensorFlow 2.9 is present the sequence
inherits its real ``keras.utils.Sequence`` class.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from .config import load_config
from .contracts import effective_head_weights, normalize_supervision_tier_loss_weights
from .inspect import (
    EVALUATION_SCHEMA,
    DatasetContractError,
    audit_canonical_dataset,
    leakage_report,
    verify_dataset_curation_manifest,
)
from .io_utils import read_image, to_uint8_gray


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
    """Deterministic tier/type quotas, then weighted rows inside each cell.

    Pretrain retains its exact per-batch sample-type contract.  Finetune can
    opt into tier-specific, epoch-level sample-type quotas so genuinely rare
    Gold cells can appear a few times per epoch instead of zero-or-every-batch.
    """

    def __init__(
        self,
        records: Sequence[Mapping[str, Any]],
        stage: str,
        seed: int,
        weight_key: str = "sampling_weight",
        gold_fraction: Optional[float] = None,
        supervision_fractions: Optional[Mapping[str, Any]] = None,
        sample_type_fractions: Optional[Mapping[str, Any]] = None,
        sample_type_fractions_by_tier: Optional[Mapping[str, Any]] = None,
        missing_cell_policy: Optional[Mapping[str, Any]] = None,
        rare_cell_policy: Optional[Mapping[str, Any]] = None,
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
        self.uses_epoch_type_plan = bool(
            self.stage == "finetune" and sample_type_fractions_by_tier is not None
        )
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
        def parse_fractions(value: Mapping[str, Any], label: str) -> Dict[str, Fraction]:
            try:
                parsed = {
                    str(key): Fraction(str(item))
                    for key, item in dict(value).items()
                }
            except (TypeError, ValueError, ZeroDivisionError) as exc:
                raise DatasetContractError(
                    "{} must contain finite numeric values".format(label)
                ) from exc
            if set(parsed) != set(CANONICAL_SAMPLE_TYPES):
                raise DatasetContractError(
                    "{} must define exactly {}".format(label, list(CANONICAL_SAMPLE_TYPES))
                )
            if any(item < 0 for item in parsed.values()):
                raise DatasetContractError("{} must be non-negative".format(label))
            if sum(parsed.values(), Fraction(0, 1)) != Fraction(1, 1):
                raise DatasetContractError("{} must sum exactly to 1".format(label))
            return {name: parsed[name] for name in CANONICAL_SAMPLE_TYPES}

        self.sample_type_fractions_by_tier: Dict[str, Dict[str, Fraction]] = {}
        if self.uses_epoch_type_plan:
            raw_by_tier = dict(sample_type_fractions_by_tier or {})
            if set(raw_by_tier) != {"gold", "pseudo"}:
                raise DatasetContractError(
                    "sampling.sample_type_fractions_by_tier must define gold and pseudo"
                )
            self.sample_type_fractions_by_tier = {
                tier: parse_fractions(
                    dict(raw_by_tier[tier]),
                    "sampling.sample_type_fractions_by_tier.{}".format(tier),
                )
                for tier in ("gold", "pseudo")
            }
            self.sample_type_fractions = dict(self.sample_type_fractions_by_tier["pseudo"])
        else:
            if sample_type_fractions is None:
                raise DatasetContractError("sampling.sample_type_fractions is required")
            self.sample_type_fractions = parse_fractions(
                sample_type_fractions, "sampling.sample_type_fractions"
            )
            if self.stage == "finetune":
                self.sample_type_fractions_by_tier = {
                    "gold": dict(self.sample_type_fractions),
                    "pseudo": dict(self.sample_type_fractions),
                }
        self.missing_cell_policy = {
            str(key): str(value) for key, value in dict(missing_cell_policy or {}).items()
        }
        self.rare_cell_policy = dict(rare_cell_policy or {})
        self.last_epoch_plan: Optional[Dict[str, Any]] = None
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
            if not 0.0 < self.gold_fraction < 1.0:
                raise DatasetContractError("training.gold_fraction must be strictly between 0 and 1")
            if "gold" not in self.groups or "pseudo" not in self.groups:
                raise DatasetContractError("Finetune requires both gold and pseudo supervision tiers")
        self._validate_groups()
        self._cell_indices: Dict[Tuple[str, str], np.ndarray] = {}
        self._cell_cdfs: Dict[Tuple[str, str], np.ndarray] = {}
        for tier, type_groups in self.groups.items():
            for sample_type, values in type_groups.items():
                indices = np.asarray(values, dtype=np.int64)
                if not len(indices):
                    continue
                weights = self.weights[indices]
                total = float(np.sum(weights))
                if total <= 0.0:
                    continue
                cdf = np.cumsum(weights / total, dtype=np.float64)
                cdf[-1] = 1.0
                key = (tier, sample_type)
                self._cell_indices[key] = indices
                self._cell_cdfs[key] = cdf

    def _validate_groups(self) -> None:
        if np.any(~np.isfinite(self.weights)) or np.any(self.weights < 0.0):
            raise DatasetContractError("sampling_weight must be finite and non-negative")
        for tier in self._tier_order():
            fractions = self.sample_type_fractions_by_tier.get(
                tier, self.sample_type_fractions
            )
            for sample_type in CANONICAL_SAMPLE_TYPES:
                if fractions[sample_type] <= 0:
                    continue
                indices = self.groups[tier].get(sample_type, [])
                has_weight = bool(indices) and float(np.sum(self.weights[indices])) > 0.0
                if not indices or not has_weight:
                    policy = self.missing_cell_policy.get(tier, "fail")
                    if self.uses_epoch_type_plan and policy == "redistribute_within_tier":
                        continue
                    raise DatasetContractError(
                        "Missing canonical sampling cell supervision_tier={!r}, sample_type={!r}; "
                        "policy={!r}".format(
                            tier, sample_type, policy
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

    def resolve_auto_epoch_size(
        self,
        batch_size: int,
        upper_bound: int,
        max_average_draws: float,
        max_expected_row_draws: float,
    ) -> Tuple[int, Dict[str, Any]]:
        """Choose the largest safe whole-batch epoch for pretrain sampling.

        The calculation deliberately calls :meth:`batch_quota` instead of
        estimating from floating-point fractions, so the gate and the actual
        sampler share the same integer rounding behavior.
        """

        if self.stage != "pretrain":
            raise DatasetContractError("sampling.epoch_size=auto is only supported for pretrain")
        if isinstance(batch_size, bool) or int(batch_size) <= 0:
            raise DatasetContractError("training.batch_size must be positive")
        if isinstance(upper_bound, bool) or int(upper_bound) < int(batch_size):
            raise DatasetContractError(
                "sampling.epoch_size_upper_bound must be at least one batch"
            )
        limits = (float(max_average_draws), float(max_expected_row_draws))
        if any(not math.isfinite(value) or value <= 0.0 for value in limits):
            raise DatasetContractError(
                "sampling repetition limits must be finite and positive"
            )

        batch_size = int(batch_size)
        maximum_batches = int(upper_bound) // batch_size
        one_batch = self.batch_quota(batch_size)
        feasible: List[Tuple[int, Dict[str, Any]]] = []
        rejected: List[Dict[str, Any]] = []
        for batch_count in range(1, maximum_batches + 1):
            cell_reports: Dict[str, Dict[str, Any]] = {}
            all_safe = True
            limiting: Optional[Dict[str, Any]] = None
            for tier in self._tier_order():
                for sample_type in CANONICAL_SAMPLE_TYPES:
                    draws = int(one_batch[tier][sample_type]) * batch_count
                    if draws <= 0:
                        continue
                    indices = np.asarray(self.groups[tier].get(sample_type, []), dtype=np.int64)
                    if len(indices) == 0:
                        raise DatasetContractError(
                            "Resolved positive quota for missing sampling cell {}:{}".format(
                                tier, sample_type
                            )
                        )
                    weights = self.weights[indices]
                    weight_total = float(np.sum(weights))
                    average = float(draws) / float(len(indices))
                    normalized = weights / weight_total
                    max_offset = int(np.argmax(normalized))
                    max_expected = float(draws) * float(normalized[max_offset])
                    record = self.records[int(indices[max_offset])]
                    cell_key = "{}:{}".format(tier, sample_type)
                    cell_report = {
                        "draws": draws,
                        "unique_records": int(len(indices)),
                        "average_draws_per_unique_record": average,
                        "max_expected_row_draws": max_expected,
                        "limiting_record_id": str(
                            record.get("global_crop_id") or record.get("crop_id") or indices[max_offset]
                        ),
                        "limiting_record_normalized_weight": float(normalized[max_offset]),
                    }
                    cell_reports[cell_key] = cell_report
                    ratio = max(
                        average / limits[0],
                        max_expected / limits[1],
                    )
                    if limiting is None or ratio > float(limiting["limit_ratio"]):
                        limiting = dict(cell_report, cell=cell_key, limit_ratio=ratio)
                    if average > limits[0] + 1.0e-12 or max_expected > limits[1] + 1.0e-12:
                        all_safe = False
            candidate = {
                "epoch_size": batch_count * batch_size,
                "batch_count": batch_count,
                "cell_reports": cell_reports,
                "limiting": limiting,
            }
            if all_safe:
                feasible.append((int(candidate["epoch_size"]), candidate))
            else:
                rejected.append(candidate)

        if not feasible:
            raise DatasetContractError(
                "No whole-batch epoch_size satisfies sampling repetition limits"
            )
        resolved, selected = feasible[-1]
        return int(resolved), {
            "mode": "auto_exact_batch_quota",
            "resolved_epoch_size": int(resolved),
            "batch_size": batch_size,
            "epoch_size_upper_bound": int(upper_bound),
            "max_average_cell_draws_per_unique_record": limits[0],
            "max_expected_row_draws_per_epoch": limits[1],
            "limiting_cell": (selected.get("limiting") or {}).get("cell"),
            "limiting_record_id": (selected.get("limiting") or {}).get("limiting_record_id"),
            "limiting_record_normalized_weight": (selected.get("limiting") or {}).get(
                "limiting_record_normalized_weight"
            ),
            "max_expected_row_draws": (selected.get("limiting") or {}).get(
                "max_expected_row_draws"
            ),
            "cell_reports": selected["cell_reports"],
            "first_rejected_above_resolved": next(
                (
                    value
                    for value in rejected
                    if int(value["epoch_size"]) > int(resolved)
                ),
                None,
            ),
        }

    def _effective_epoch_fractions(
        self, tier: str
    ) -> Tuple[Dict[str, Fraction], List[str], Dict[str, Any]]:
        configured = self.sample_type_fractions_by_tier[tier]
        missing = []
        available = []
        for sample_type in CANONICAL_SAMPLE_TYPES:
            if configured[sample_type] <= 0:
                continue
            indices = self.groups[tier].get(sample_type, [])
            if indices and float(np.sum(self.weights[indices])) > 0.0:
                available.append(sample_type)
            else:
                missing.append(sample_type)
        policy = self.missing_cell_policy.get(tier, "fail")
        if missing and policy != "redistribute_within_tier":
            raise DatasetContractError(
                "Missing finetune sampling cells {} for tier {!r}; policy={!r}".format(
                    missing, tier, policy
                )
            )
        total = sum((configured[name] for name in available), Fraction(0, 1))
        if total <= 0:
            raise DatasetContractError(
                "Finetune tier {!r} has no available positive-fraction sampling cell".format(tier)
            )
        effective = {
            name: (configured[name] / total if name in available else Fraction(0, 1))
            for name in CANONICAL_SAMPLE_TYPES
        }
        return effective, missing, {
            "policy": policy,
            "missing": list(missing),
            "configured": {name: float(configured[name]) for name in CANONICAL_SAMPLE_TYPES},
            "effective_before_rare_cap": {
                name: float(effective[name]) for name in CANONICAL_SAMPLE_TYPES
            },
        }

    def _cell_repetition(self, tier: str, sample_type: str, draws: int) -> Dict[str, Any]:
        indices = np.asarray(self.groups[tier].get(sample_type, []), dtype=np.int64)
        if draws <= 0:
            return {
                "draws": 0,
                "unique_records": int(len(indices)),
                "average_draws_per_unique_record": 0.0,
                "max_expected_row_draws": 0.0,
                "limiting_record_id": None,
                "limiting_record_normalized_weight": None,
            }
        if len(indices) == 0:
            raise DatasetContractError(
                "Positive epoch quota targets missing cell {}:{}".format(tier, sample_type)
            )
        weights = self.weights[indices]
        total = float(np.sum(weights))
        normalized = weights / total
        offset = int(np.argmax(normalized))
        row = self.records[int(indices[offset])]
        return {
            "draws": int(draws),
            "unique_records": int(len(indices)),
            "average_draws_per_unique_record": float(draws) / float(len(indices)),
            "max_expected_row_draws": float(draws) * float(normalized[offset]),
            "limiting_record_id": str(
                row.get("global_crop_id") or row.get("crop_id") or indices[offset]
            ),
            "limiting_record_normalized_weight": float(normalized[offset]),
        }

    def _apply_rare_cell_limits(
        self,
        tier: str,
        quotas: Dict[str, int],
        effective: Mapping[str, Fraction],
    ) -> Tuple[Dict[str, int], Dict[str, Any]]:
        max_average_value = self.rare_cell_policy.get(
            "max_average_draws_per_unique_record"
        )
        max_row_value = self.rare_cell_policy.get("max_expected_row_draws_per_epoch")
        if max_average_value is None or max_row_value is None:
            return dict(quotas), {"policy": "disabled", "caps": {}}
        max_average = float(max_average_value)
        max_row = float(max_row_value)
        if any(not math.isfinite(value) or value <= 0.0 for value in (max_average, max_row)):
            raise DatasetContractError("finetune rare-cell limits must be finite and positive")
        policy = str(self.rare_cell_policy.get(tier, "fail"))
        adjusted = dict(quotas)
        caps: Dict[str, Any] = {}
        removed = 0
        for sample_type in CANONICAL_SAMPLE_TYPES:
            draws = int(adjusted[sample_type])
            repetition = self._cell_repetition(tier, sample_type, draws)
            if (
                repetition["average_draws_per_unique_record"] <= max_average + 1.0e-12
                and repetition["max_expected_row_draws"] <= max_row + 1.0e-12
            ):
                continue
            if policy != "cap_fraction_then_redistribute_within_tier":
                raise DatasetContractError(
                    "Finetune sampling cell {}:{} exceeds rare-cell limits".format(
                        tier, sample_type
                    )
                )
            indices = np.asarray(self.groups[tier][sample_type], dtype=np.int64)
            weights = self.weights[indices]
            max_probability = float(np.max(weights / float(np.sum(weights))))
            cap = int(
                math.floor(
                    min(
                        max_average * float(len(indices)),
                        max_row / max_probability,
                    )
                    + 1.0e-12
                )
            )
            cap = max(0, min(draws, cap))
            adjusted[sample_type] = cap
            removed += draws - cap
            caps[sample_type] = {
                "configured_draws": draws,
                "capped_draws": cap,
                "before": repetition,
            }
        if removed:
            recipients = [
                name
                for name in ("POS_RUNTIME", "POS_LOW_PALM")
                if effective[name] > 0
                and self.groups[tier].get(name)
                and name not in caps
            ]
            if not recipients:
                raise DatasetContractError(
                    "Rare-cell quota cannot be redistributed to an uncapped Gold positive cell"
                )
            total = sum((effective[name] for name in recipients), Fraction(0, 1))
            fractions = {name: effective[name] / total for name in recipients}
            additions = self._largest_remainder_counts(removed, fractions, recipients)
            for name, value in additions.items():
                adjusted[name] += int(value)
        if sum(adjusted.values()) != sum(quotas.values()):
            raise DatasetContractError("Rare-cell quota redistribution did not conserve tier draws")
        after = {
            name: self._cell_repetition(tier, name, adjusted[name])
            for name in CANONICAL_SAMPLE_TYPES
            if adjusted[name] > 0
        }
        for name, repetition in after.items():
            if (
                repetition["average_draws_per_unique_record"] > max_average + 1.0e-12
                or repetition["max_expected_row_draws"] > max_row + 1.0e-12
            ):
                raise DatasetContractError(
                    "Finetune sampling cell {}:{} remains above rare-cell limits after redistribution".format(
                        tier, name
                    )
                )
        return adjusted, {
            "policy": policy,
            "max_average_draws_per_unique_record": max_average,
            "max_expected_row_draws_per_epoch": max_row,
            "caps": caps,
            "redistributed_draws": removed,
            "after": after,
        }

    @staticmethod
    def _balanced_type_stream(quotas: Mapping[str, int]) -> List[str]:
        total = int(sum(int(value) for value in quotas.values()))
        used = {name: 0 for name in CANONICAL_SAMPLE_TYPES}
        stream: List[str] = []
        for position in range(total):
            candidates = [
                name
                for name in CANONICAL_SAMPLE_TYPES
                if used[name] < int(quotas.get(name, 0))
            ]
            selected = max(
                candidates,
                key=lambda name: (
                    Fraction((position + 1) * int(quotas[name]), max(total, 1))
                    - used[name],
                    -CANONICAL_SAMPLE_TYPES.index(name),
                ),
            )
            used[selected] += 1
            stream.append(selected)
        if used != {name: int(quotas.get(name, 0)) for name in CANONICAL_SAMPLE_TYPES}:
            raise DatasetContractError("Balanced epoch type stream did not conserve quotas")
        return stream

    def _sample_cells(
        self,
        cells: Sequence[Tuple[str, str]],
        rng: np.random.RandomState,
    ) -> np.ndarray:
        result = np.empty((len(cells),), dtype=np.int64)
        if not cells:
            return result
        uniforms = rng.random_sample(len(cells))
        positions_by_cell: Dict[Tuple[str, str], List[int]] = defaultdict(list)
        for position, cell in enumerate(cells):
            positions_by_cell[cell].append(position)
        for cell, positions in positions_by_cell.items():
            indices = self._cell_indices[cell]
            cdf = self._cell_cdfs[cell]
            position_array = np.asarray(positions, dtype=np.int64)
            selected = np.searchsorted(cdf, uniforms[position_array], side="right")
            result[position_array] = indices[selected]
        return result

    def sample_epoch(self, batch_sizes: Sequence[int], epoch: int = 0) -> np.ndarray:
        """Sample a complete epoch, preserving per-batch tier counts."""

        sizes = [int(value) for value in batch_sizes]
        if any(value <= 0 for value in sizes):
            raise ValueError("batch_sizes must be positive")
        if not self.uses_epoch_type_plan:
            batches = [
                self.sample(size, epoch=epoch, stream=index)
                for index, size in enumerate(sizes)
            ]
            self.last_epoch_plan = None
            return np.concatenate(batches) if batches else np.empty((0,), dtype=np.int64)

        tier_counts_by_batch = [self._tier_counts(size) for size in sizes]
        tier_totals = {
            tier: sum(int(value[tier]) for value in tier_counts_by_batch)
            for tier in self._tier_order()
        }
        quotas_by_tier: Dict[str, Dict[str, int]] = {}
        missing_reports: Dict[str, Any] = {}
        rare_reports: Dict[str, Any] = {}
        for tier in self._tier_order():
            effective, _missing, missing_report = self._effective_epoch_fractions(tier)
            quota = self._largest_remainder_counts(
                tier_totals[tier], effective, CANONICAL_SAMPLE_TYPES
            )
            quota, rare_report = self._apply_rare_cell_limits(tier, quota, effective)
            quotas_by_tier[tier] = quota
            missing_reports[tier] = missing_report
            rare_reports[tier] = rare_report

        streams = {
            tier: self._balanced_type_stream(quotas_by_tier[tier])
            for tier in self._tier_order()
        }
        offsets = {tier: 0 for tier in self._tier_order()}
        batches: List[np.ndarray] = []
        batch_cell_quotas: List[Dict[str, int]] = []
        schedule_cells: List[str] = []
        for batch_index, (size, tier_counts) in enumerate(zip(sizes, tier_counts_by_batch)):
            cells: List[Tuple[str, str]] = []
            for tier in self._tier_order():
                count = int(tier_counts[tier])
                start = offsets[tier]
                selected_types = streams[tier][start : start + count]
                offsets[tier] += count
                cells.extend((tier, sample_type) for sample_type in selected_types)
            rng = np.random.RandomState(
                (self.seed + int(epoch) * 1000003 + batch_index * 9176) % (2 ** 32 - 1)
            )
            rng.shuffle(cells)
            batches.append(self._sample_cells(cells, rng))
            counter = Counter("{}:{}".format(tier, sample_type) for tier, sample_type in cells)
            batch_cell_quotas.append(dict(sorted(counter.items())))
            schedule_cells.extend("{}:{}".format(tier, sample_type) for tier, sample_type in cells)
            if len(cells) != size:
                raise DatasetContractError("Finetune batch plan size mismatch")
        if any(offsets[tier] != tier_totals[tier] for tier in offsets):
            raise DatasetContractError("Finetune epoch plan did not consume every tier slot")

        result = np.concatenate(batches) if batches else np.empty((0,), dtype=np.int64)
        actual = Counter(int(index) for index in result)
        actual_repetition = {
            "maximum": max(actual.values()) if actual else 0,
            "unique_records": len(actual),
        }
        self.last_epoch_plan = {
            "quota_scope": {
                "supervision_tier": "per_batch_half_up",
                "sample_type": "per_epoch_largest_remainder",
            },
            "batch_sizes": sizes,
            "batch_tier_quota": tier_counts_by_batch,
            "epoch_draw_quota_by_tier_type": quotas_by_tier,
            "configured_and_effective_fractions": missing_reports,
            "rare_cell_quota_cap": rare_reports,
            "batch_cell_quotas": batch_cell_quotas,
            "batch_type_schedule_sha256": hashlib.sha256(
                json.dumps(schedule_cells, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
            "actual_repetition": actual_repetition,
        }
        return result

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
        return self._sample_cells(cells, rng)

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
            "sample_type_fractions_by_tier": (
                {
                    tier: {
                        name: float(fractions[name])
                        for name in CANONICAL_SAMPLE_TYPES
                    }
                    for tier, fractions in sorted(
                        self.sample_type_fractions_by_tier.items()
                    )
                }
                if self.uses_epoch_type_plan
                else None
            ),
            "sample_type_fraction_scope": (
                "per_epoch_largest_remainder"
                if self.uses_epoch_type_plan
                else "per_batch_largest_remainder"
            ),
            "quota_tie_break": list(CANONICAL_SAMPLE_TYPES),
            "cross_bucket_policy": (
                dict(self.missing_cell_policy)
                if self.uses_epoch_type_plan
                else "require_every_positive_type_in_every_active_tier"
            ),
            "row_sampling": "sampling_weight_within_selected_tier_and_sample_type_only",
            "tiers": {
                tier: {sample_type: len(indices) for sample_type, indices in sorted(types.items())}
                for tier, types in sorted(self.groups.items())
            },
        }


class CanonicalSequence(_KerasSequence):
    """Keras sequence returning targets plus head and Gold-structure masks."""

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
        losses_config: Optional[Mapping[str, Any]] = None,
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
        self.losses_config = dict(losses_config or {})
        self.supervision_tier_loss_weights = normalize_supervision_tier_loss_weights(
            self.losses_config.get("supervision_tier_weights")
        )
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
        self.sampler: Optional[WeightedStratifiedSampler] = None
        self.epoch_resolution: Optional[Dict[str, Any]] = None
        if self.training:
            if self.sampling_config.get("enabled", True) is not True:
                raise DatasetContractError("sampling.enabled must remain true for canonical training")
            if self.sampling_config.get("replacement", True) is not True:
                raise DatasetContractError("Weighted canonical sampling requires sampling.replacement=true")
            if self.sampling_config.get("honor_record_sampling_weight", True) is not True:
                raise DatasetContractError("sampling.honor_record_sampling_weight must remain true")
            if self.sampling_config.get("quota_rounding", "largest_remainder") != "largest_remainder":
                raise DatasetContractError("sampling.quota_rounding must be largest_remainder")
            if self.stage == "finetune" and self.sampling_config.get(
                "sample_type_fractions_by_tier"
            ) is not None:
                if dict(self.sampling_config.get("quota_scope") or {}) != {
                    "supervision_tier": "per_batch_half_up",
                    "sample_type": "per_epoch_largest_remainder",
                }:
                    raise DatasetContractError(
                        "finetune sampling.quota_scope must declare per-batch tier and per-epoch type quotas"
                    )
                if self.sampling_config.get("batch_distribution") != "deterministic_balanced_deficit":
                    raise DatasetContractError(
                        "finetune sampling.batch_distribution must be deterministic_balanced_deficit"
                    )
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
                sample_type_fractions_by_tier=self.sampling_config.get(
                    "sample_type_fractions_by_tier"
                ),
                missing_cell_policy=self.sampling_config.get("missing_cell_policy"),
                rare_cell_policy=self.sampling_config.get("rare_cell_policy"),
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

        configured_epoch_size = self.sampling_config.get("epoch_size")
        if steps_per_epoch not in (None, 0):
            if configured_epoch_size == "auto":
                raise DatasetContractError(
                    "training.steps_per_epoch and sampling.epoch_size=auto are mutually exclusive"
                )
            self.steps = int(steps_per_epoch)
            self.epoch_size = self.steps * self.batch_size
        elif configured_epoch_size == "auto":
            if not self.training or self.sampler is None:
                raise DatasetContractError(
                    "sampling.epoch_size=auto requires weighted training"
                )
            required = (
                "epoch_size_upper_bound",
                "max_average_cell_draws_per_unique_record",
                "max_expected_row_draws_per_epoch",
            )
            missing = [name for name in required if self.sampling_config.get(name) is None]
            if missing:
                raise DatasetContractError(
                    "sampling.epoch_size=auto requires {}".format(missing)
                )
            self.epoch_size, self.epoch_resolution = self.sampler.resolve_auto_epoch_size(
                batch_size=self.batch_size,
                upper_bound=int(self.sampling_config["epoch_size_upper_bound"]),
                max_average_draws=float(
                    self.sampling_config["max_average_cell_draws_per_unique_record"]
                ),
                max_expected_row_draws=float(
                    self.sampling_config["max_expected_row_draws_per_epoch"]
                ),
            )
            self.steps = int(self.epoch_size // self.batch_size)
        else:
            try:
                self.epoch_size = (
                    int(configured_epoch_size)
                    if configured_epoch_size not in (None, 0)
                    else len(self.records)
                )
            except (TypeError, ValueError) as exc:
                raise DatasetContractError(
                    "sampling.epoch_size must be a positive integer, null, or auto"
                ) from exc
            self.steps = int(math.ceil(self.epoch_size / float(self.batch_size)))
        if self.steps <= 0:
            raise ValueError("steps_per_epoch must be positive")
        if self.epoch_size <= 0:
            raise ValueError("sampling.epoch_size must be positive")

        self.tail_batch_size = int(self.epoch_size % self.batch_size)
        self.tail_batch_policy = str(
            self.sampling_config.get("tail_batch_policy") or "full_batches_only"
        )
        if self.tail_batch_policy not in {
            "full_batches_only",
            "allow_smaller_final_batch",
        }:
            raise DatasetContractError(
                "sampling.tail_batch_policy must be full_batches_only or allow_smaller_final_batch"
            )
        if (
            self.training
            and self.stage == "finetune"
            and self.sampling_config.get("sample_type_fractions_by_tier") is not None
            and self.tail_batch_size
            and self.tail_batch_policy != "allow_smaller_final_batch"
        ):
            raise DatasetContractError(
                "A non-divisible finetune epoch requires "
                "sampling.tail_batch_policy=allow_smaller_final_batch"
            )

        if self.training:
            assert self.sampler is not None
            self.set_epoch(0)
        else:
            self.epoch_size = len(self.records)
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
        batch_sizes: List[int] = []
        remaining = self.epoch_size
        while remaining > 0:
            current_size = min(self.batch_size, remaining)
            batch_sizes.append(current_size)
            remaining -= current_size
        return self.sampler.sample_epoch(batch_sizes, epoch=epoch)

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
        structure_weights: List[float] = []
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
            tier_weights = self.supervision_tier_loss_weights if self.training else None
            presence_weight, landmark_weight, handedness_weight = effective_head_weights(
                row, tier_weights
            )
            presence_weights.append(float(presence_weight))
            landmark_weights.append(float(landmark_weight))
            handedness_weights.append(float(handedness_weight))
            structure_weights.append(
                float(landmark_weight)
                if present and str(row.get("supervision_tier")) == "gold"
                else 0.0
            )

        x_value = np.stack(images).astype(np.float32)
        y_values = [
            np.stack(landmark_targets).astype(np.float32),
            np.asarray(presence_targets, dtype=np.float32),
            np.asarray(handedness_targets, dtype=np.float32),
        ]
        sample_weights = {
            "landmarks": np.asarray(landmark_weights, dtype=np.float32),
            "hand_flag": np.asarray(presence_weights, dtype=np.float32),
            "handedness": np.asarray(handedness_weights, dtype=np.float32),
            "structure": np.asarray(structure_weights, dtype=np.float32),
        }
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
            if self.sampler.last_epoch_plan is not None:
                per_batch_cross_cells.append(
                    dict(self.sampler.last_epoch_plan["batch_cell_quotas"][batch_index])
                )
            else:
                per_batch_cross_cells.append(
                    {
                        "{}:{}".format(tier, sample_type): int(value)
                        for tier, type_counts in self.sampler.batch_quota(len(batch_indices)).items()
                        for sample_type, value in type_counts.items()
                    }
                )
        dataset_key = str(self.sampling_config.get("dataset_key", "dataset_id"))
        source_group_key = str(
            self.sampling_config.get("source_group_key", "source_group_id")
        )
        inventory_by_dataset = Counter(
            str(row.get(dataset_key) or "") for row in self.records
        )
        if "" in inventory_by_dataset:
            raise DatasetContractError(
                "Every sampled row must provide {} for source auditing".format(dataset_key)
            )
        source_groups: Dict[str, List[int]] = defaultdict(list)
        source_group_dataset: Dict[str, str] = {}
        for index, row in enumerate(self.records):
            group = str(row.get(source_group_key) or "")
            if not group:
                raise DatasetContractError(
                    "Every sampled row must provide {} for source auditing".format(
                        source_group_key
                    )
                )
            dataset_id = str(row.get(dataset_key))
            existing = source_group_dataset.setdefault(group, dataset_id)
            if existing != dataset_id:
                raise DatasetContractError(
                    "{} {!r} crosses dataset IDs {} and {}".format(
                        source_group_key, group, existing, dataset_id
                    )
                )
            source_groups[group].append(index)

        batch_sizes = [len(self.batch_record_indices(index)) for index in range(len(self))]
        cell_draws: Counter = Counter()
        for size in batch_sizes:
            quota = self.sampler.batch_quota(size)
            for tier, type_counts in quota.items():
                for sample_type, value in type_counts.items():
                    cell_draws[(tier, sample_type)] += int(value)
        expected_by_dataset: Dict[str, float] = defaultdict(float)
        expected_by_source_group: Dict[str, float] = defaultdict(float)
        for tier, types in self.sampler.groups.items():
            for sample_type, cell_indices in types.items():
                draws = int(cell_draws[(tier, sample_type)])
                if draws <= 0:
                    continue
                values = np.asarray(cell_indices, dtype=np.int64)
                weights = self.sampler.weights[values]
                probabilities = weights / float(np.sum(weights))
                for record_index, probability in zip(values, probabilities):
                    row = self.records[int(record_index)]
                    expectation = float(draws) * float(probability)
                    expected_by_dataset[str(row[dataset_key])] += expectation
                    expected_by_source_group[str(row[source_group_key])] += expectation
        actual_by_dataset = Counter(
            str(self.records[index][dataset_key]) for index in indices
        )
        actual_by_source_group = Counter(
            str(self.records[index][source_group_key]) for index in indices
        )
        group_sizes = np.asarray(
            [len(values) for values in source_groups.values()], dtype=np.float64
        )
        expected_dataset_fraction = {
            key: float(value) / float(len(indices))
            for key, value in sorted(expected_by_dataset.items())
        }
        actual_dataset_fraction = {
            key: float(value) / float(len(indices))
            for key, value in sorted(actual_by_dataset.items())
        }
        source_audit = {
            "dataset_key": dataset_key,
            "source_group_key": source_group_key,
            "training_records_by_dataset": dict(sorted(inventory_by_dataset.items())),
            "unique_source_groups": len(source_groups),
            "unique_source_groups_by_dataset": dict(
                sorted(Counter(source_group_dataset.values()).items())
            ),
            "records_per_source_group": {
                "minimum": int(np.min(group_sizes)),
                "median": float(np.median(group_sizes)),
                "p90": float(np.percentile(group_sizes, 90)),
                "maximum": int(np.max(group_sizes)),
            },
            "expected_draws_by_dataset": {
                key: float(value) for key, value in sorted(expected_by_dataset.items())
            },
            "expected_draw_fraction_by_dataset": expected_dataset_fraction,
            "actual_epoch0_draws_by_dataset": dict(sorted(actual_by_dataset.items())),
            "actual_epoch0_draw_fraction_by_dataset": actual_dataset_fraction,
            "maximum_expected_dataset_fraction": max(
                expected_dataset_fraction.values(), default=0.0
            ),
            "maximum_expected_source_group_draws": max(
                expected_by_source_group.values(), default=0.0
            ),
            "maximum_actual_source_group_draws": max(
                actual_by_source_group.values(), default=0
            ),
        }
        return {
            "mode": "weighted_stratified",
            "absolute_epoch": self.epoch,
            "draws_per_epoch": len(indices),
            "tail_batch_policy": self.tail_batch_policy,
            "tail_batch_size": self.tail_batch_size,
            "drawn_supervision_tiers": dict(Counter(str(self.records[index].get("supervision_tier")) for index in indices)),
            "drawn_sample_types": dict(Counter(str(self.records[index].get("sample_type")) for index in indices)),
            "drawn_supervision_tiers_per_batch": per_batch_tiers,
            "drawn_sample_types_per_batch": per_batch_sample_types,
            "tier_sample_type_quota_per_batch": per_batch_cross_cells,
            "epoch_size_resolution": self.epoch_resolution,
            "epoch_type_plan": self.sampler.last_epoch_plan,
            "definition": self.sampler.report(),
            "supervision_tier_loss_weights": dict(self.supervision_tier_loss_weights),
            "source_audit": source_audit,
        }


def _validation_dataset_config(config: Mapping[str, Any], dataset: Mapping[str, Any]) -> Dict[str, Any]:
    value = dict(dataset)
    validation = dict(config.get("validation", {}))
    for key in (
        "data_root",
        "labels",
        "ignored_labels",
        "crop_path_key",
        "path_policy",
        "crop_image_roots",
        "allowed_crop_roots",
        "image_size",
        "channels",
        "color_mode",
    ):
        if key in validation:
            value[key] = validation[key]
    value["labels"] = validation["labels"]
    if validation.get("ignored_labels"):
        value["ignored_labels"] = validation["ignored_labels"]
    else:
        value.pop("ignored_labels", None)
    value["require_schema_version"] = (
        "hlml_fixed_roi_evaluation_v1"
        if str(dataset.get("require_curation_schema", "")) == "hlml_warehouse_snapshot_v1"
        else EVALUATION_SCHEMA
    )
    value["require_split"] = "val"
    value.pop("require_training_stage", None)
    return value


def create_sequences(config: Union[Mapping[str, Any], str, Path]):
    """Build strict train/validation sequences and a serializable data report."""

    cfg = _config_mapping(config)
    if "dataset" in cfg:
        raise DatasetContractError("config.dataset is obsolete; use config.data")
    dataset_cfg = dict(cfg.get("data") or {})
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
    curation_manifest = verify_dataset_curation_manifest(
        cfg, dataset_cfg, error_type=DatasetContractError
    )
    warehouse_snapshot = str(dataset_cfg.get("require_curation_schema", "")) == "hlml_warehouse_snapshot_v1"
    train_records, train_report = audit_canonical_dataset(
        cfg,
        dataset=dataset_cfg,
        expected_stage=stage,
        check_images=True,
        hash_images=not warehouse_snapshot,
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
            hash_images=not warehouse_snapshot,
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
        losses_config=cfg.get("losses", {}),
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
            "sample_weights": {
                "landmarks": [None],
                "hand_flag": [None],
                "handedness": [None],
                "structure": [None],
            },
        },
    }
    return train_sequence, validation_sequence, report
