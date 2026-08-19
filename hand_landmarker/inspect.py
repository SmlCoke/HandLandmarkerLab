"""Canonical HandLandmarkerFab dataset inspection without TensorFlow imports.

The finalizer JSONL is the sample index.  Files found below configured image
roots are used only to repair a moved canonical ``crop_path``; they never add
records to a dataset.
"""

from __future__ import annotations

import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

from .config import load_config, resolve_path
from .contracts import effective_head_weights
from .io_utils import build_basename_index, read_jsonl, sha256_file
from .pretrain_curation import verify_curation_manifest


TRAIN_SCHEMA = "train_finalize_v1"
EVALUATION_SCHEMA = "evaluation_gold_v1"
SAMPLE_TYPES = {
    "POS_RUNTIME",
    "POS_LOW_PALM",
    "NEG_RUNTIME_CANDIDATE",
    "NEG_LOW_PALM_CANDIDATE",
}
TRAIN_WEIGHT_FIELDS = (
    "hand_presence_loss_weight",
    "landmark_loss_weight",
    "handedness_loss_weight",
    "supervision_loss_weight",
    "presence_quality_weight",
    "landmark_quality_weight",
    "handedness_quality_weight",
    "sampling_weight",
)
EVALUATION_WEIGHT_FIELDS = (
    "hand_presence_loss_weight",
    "landmark_loss_weight",
    "handedness_loss_weight",
)


class DatasetContractError(ValueError):
    """Raised when canonical labels cannot safely be consumed."""


def verify_dataset_curation_manifest(
    config: Mapping[str, Any],
    dataset: Mapping[str, Any],
    error_type=DatasetContractError,
) -> Dict[str, Any]:
    """Dispatch to the stage-specific authenticated curation contract."""

    stage = str(dataset.get("require_training_stage") or config.get("stage") or "")
    schema = str(dataset.get("require_curation_schema") or "")
    if schema == "hlml_warehouse_snapshot_v1":
        from .warehouse import verify_snapshot_manifest

        return verify_snapshot_manifest(config, dataset, error_type=error_type)
    if stage == "finetune" or schema == "finetune_curation_v1":
        from .finetune_curation import verify_finetune_curation_manifest

        return verify_finetune_curation_manifest(config, dataset, error_type=error_type)
    return verify_curation_manifest(config, dataset, error_type=error_type)


def _record_id(row: Mapping[str, Any]) -> str:
    return str(row.get("global_crop_id") or row.get("crop_id") or "")


def _finite_number(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _identity_values(row: Mapping[str, Any]) -> List[str]:
    """Return every canonical identity usable for included/ignored checks."""

    return sorted(
        {
            str(row[key])
            for key in ("global_crop_id", "crop_id")
            if row.get(key) not in (None, "")
        }
    )


def _configured_path(value: Any, config: Mapping[str, Any]) -> Path:
    return resolve_path(str(value), config)


def _as_path_list(value: Any, config: Mapping[str, Any]) -> List[Path]:
    if value in (None, ""):
        return []
    values = value if isinstance(value, (list, tuple)) else [value]
    return [_configured_path(item, config) for item in values]


def configured_image_roots(dataset: Mapping[str, Any], config: Mapping[str, Any]) -> List[Path]:
    """Return only explicitly configured roots, with stable de-duplication."""

    values: List[Path] = []
    values.extend(_as_path_list(dataset.get("image_roots"), config))
    values.extend(_as_path_list(dataset.get("crop_image_roots"), config))
    # ``data_root`` is only a compatibility fallback.  Once dedicated image
    # roots are declared, indexing the (usually much broader) data root both
    # wastes time and weakens the allowed-root boundary.
    if not values:
        values.extend(_as_path_list(dataset.get("data_root"), config))
    unique: Dict[str, Path] = {}
    for path in values:
        lexical = Path(os.path.abspath(os.fspath(path.expanduser())))
        unique.setdefault(os.path.normcase(str(lexical)), lexical)
    return list(unique.values())


class CanonicalPathResolver:
    """Resolve ``crop_path`` while deliberately ignoring ``source_crop_path``."""

    def __init__(
        self,
        dataset: Mapping[str, Any],
        config: Mapping[str, Any],
        labels_path: Path,
    ) -> None:
        policy = str(dataset.get("path_policy", "canonical_crop_path_only"))
        if policy != "canonical_crop_path_only":
            raise DatasetContractError("Unsupported dataset.path_policy: {}".format(policy))
        self.path_key = str(dataset.get("crop_path_key", "crop_path"))
        self.labels_path = Path(labels_path).resolve()
        self.config = config
        self.roots = configured_image_roots(dataset, config)
        self.allowed_roots = _as_path_list(dataset.get("allowed_crop_roots"), config)
        self._allowed_root_cache: Optional[List[Tuple[Path, Path]]] = None
        self._basename_index: Optional[Dict[str, List[Path]]] = None

    @staticmethod
    def _lexical_absolute(path: Path) -> Path:
        """Normalize ``.``/``..`` without following filesystem links."""

        return Path(os.path.abspath(os.fspath(path.expanduser())))

    @staticmethod
    def _symlink_component(path: Path) -> Optional[Path]:
        """Return the first symlink in an absolute path, including parents."""

        absolute = CanonicalPathResolver._lexical_absolute(path)
        current = Path(absolute.anchor)
        for part in absolute.parts[1:]:
            current = current / part
            if current.is_symlink():
                return current
        return None

    def _resolved_allowed_roots(self) -> List[Tuple[Path, Path]]:
        if self._allowed_root_cache is not None:
            return list(self._allowed_root_cache)
        values: List[Tuple[Path, Path]] = []
        for configured in self.allowed_roots:
            lexical = self._lexical_absolute(configured)
            if not lexical.is_dir():
                raise DatasetContractError(
                    "Configured allowed_crop_root is not a readable directory: {}".format(lexical)
                )
            linked = self._symlink_component(lexical)
            if linked is not None:
                raise DatasetContractError(
                    "Configured allowed_crop_root contains a symlink component: {}".format(linked)
                )
            values.append((lexical, lexical.resolve(strict=True)))
        self._allowed_root_cache = values
        return list(values)

    def _validate_allowed_path(self, path: Path, *, require_file: bool) -> Path:
        """Resolve a path strictly and enforce containment plus no-symlink rules."""

        lexical = self._lexical_absolute(path)
        try:
            resolved = lexical.resolve(strict=True)
        except (FileNotFoundError, OSError) as exc:
            raise FileNotFoundError("Canonical image path is unreadable: {}".format(lexical)) from exc
        if require_file and not resolved.is_file():
            raise FileNotFoundError("Canonical image path is not a file: {}".format(lexical))
        if not require_file and not resolved.is_dir():
            raise DatasetContractError("Configured image root is not a directory: {}".format(lexical))

        if not self.allowed_roots:
            return resolved

        allowed_pairs = self._resolved_allowed_roots()
        lexical_matches: List[Tuple[Path, Path]] = []
        for allowed_lexical, allowed_resolved in allowed_pairs:
            try:
                lexical.relative_to(allowed_lexical)
            except ValueError:
                continue
            lexical_matches.append((allowed_lexical, allowed_resolved))
        if not lexical_matches:
            raise DatasetContractError(
                "Canonical path escapes allowed_crop_roots: {}".format(lexical)
            )

        for allowed_lexical, _ in lexical_matches:
            current = allowed_lexical
            for part in lexical.relative_to(allowed_lexical).parts:
                current = current / part
                if current.is_symlink():
                    raise DatasetContractError(
                        "Canonical path contains a symlink component: {}".format(current)
                    )

        for _, allowed_resolved in lexical_matches:
            try:
                resolved.relative_to(allowed_resolved)
                return resolved
            except ValueError:
                continue
        raise DatasetContractError(
            "Resolved canonical path escapes allowed_crop_roots: {} -> {}".format(
                lexical, resolved
            )
        )

    def _index(self) -> Dict[str, List[Path]]:
        if self._basename_index is None:
            missing_roots = [str(path) for path in self.roots if not path.is_dir()]
            valid_roots = [path for path in self.roots if path.is_dir()]
            if not valid_roots:
                suffix = "; configured roots missing: {}".format(missing_roots) if missing_roots else ""
                raise FileNotFoundError("No configured image root is readable{}".format(suffix))
            validated_roots = [
                self._validate_allowed_path(path, require_file=False) for path in valid_roots
            ]
            self._basename_index = build_basename_index(validated_roots)
        return self._basename_index

    def resolve(self, row: Mapping[str, Any]) -> Tuple[Path, str]:
        recorded = row.get(self.path_key)
        if recorded in (None, ""):
            raise FileNotFoundError("Record {} has no {}".format(_record_id(row), self.path_key))
        canonical = Path(str(recorded))
        direct_candidates: List[Path]
        if canonical.is_absolute():
            direct_candidates = [canonical]
        else:
            direct_candidates = [
                self.labels_path.parent / canonical,
                _configured_path(canonical, self.config),
            ]
        seen = set()
        for candidate in direct_candidates:
            key = str(candidate.resolve())
            if key in seen:
                continue
            seen.add(key)
            if candidate.is_file():
                return self._validate_allowed_path(candidate, require_file=True), "canonical"

        basename = canonical.name
        if not basename:
            raise FileNotFoundError("Record {} has an invalid {}".format(_record_id(row), self.path_key))
        matches = list(self._index().get(basename, []))
        checked = [self._validate_allowed_path(path, require_file=True) for path in matches]
        unique = sorted({str(path): path for path in checked}.values(), key=str)
        if not unique:
            raise FileNotFoundError(
                "Canonical image {} for {} was not found under configured roots".format(
                    basename, _record_id(row)
                )
            )
        if len(unique) > 1:
            raise DatasetContractError(
                "Ambiguous canonical image basename {} for {}: {}".format(
                    basename, _record_id(row), [str(path) for path in unique]
                )
            )
        return unique[0], "rebased"


def _validate_points(points: Any, field: str, expected_count: int = 21) -> Tuple[List[str], List[Tuple[float, float]]]:
    errors: List[str] = []
    if not isinstance(points, list):
        return ["{} must be a list".format(field)], []
    if len(points) != expected_count:
        return ["{} must contain {} points; got {}".format(field, expected_count, len(points))], []
    by_id: Dict[int, Tuple[float, float]] = {}
    for offset, point in enumerate(points):
        if not isinstance(point, Mapping):
            errors.append("{} point {} is not an object".format(field, offset))
            continue
        try:
            point_id = int(point.get("id"))
        except (TypeError, ValueError):
            errors.append("{} point {} has invalid id".format(field, offset))
            continue
        if point_id in by_id:
            errors.append("{} has duplicate id {}".format(field, point_id))
            continue
        x_value, y_value = point.get("x"), point.get("y")
        if not _finite_number(x_value) or not _finite_number(y_value):
            errors.append("{} point {} has non-finite coordinates".format(field, point_id))
            continue
        by_id[point_id] = (float(x_value), float(y_value))
    if set(by_id) != set(range(expected_count)):
        errors.append("{} ids must be exactly 0..{}".format(field, expected_count - 1))
    ordered = [by_id[index] for index in range(expected_count)] if not errors else []
    return errors, ordered


def _expected_sample_type(present: bool, palm_valid: bool) -> str:
    if present:
        return "POS_RUNTIME" if palm_valid else "POS_LOW_PALM"
    return "NEG_RUNTIME_CANDIDATE" if palm_valid else "NEG_LOW_PALM_CANDIDATE"


def _validate_geometry(row: Mapping[str, Any]) -> List[str]:
    errors: List[str] = []
    if not _finite_number(row.get("palm_score")):
        errors.append("palm_score must be finite")
    if not isinstance(row.get("palm_valid"), bool):
        errors.append("palm_valid must be boolean")
    rect = row.get("roi_rect")
    if not isinstance(rect, Mapping):
        errors.append("roi_rect must be an object")
    else:
        for key in ("x_center", "y_center", "width", "height", "rotation_rad"):
            if not _finite_number(rect.get(key)):
                errors.append("roi_rect.{} must be finite".format(key))
        if _finite_number(rect.get("width")) and float(rect["width"]) <= 0.0:
            errors.append("roi_rect.width must be positive")
        if _finite_number(rect.get("height")) and float(rect["height"]) <= 0.0:
            errors.append("roi_rect.height must be positive")
    corners = row.get("roi_corners_px")
    if not isinstance(corners, list) or len(corners) != 4:
        errors.append("roi_corners_px must contain four corners")
    else:
        for index, corner in enumerate(corners):
            if (
                not isinstance(corner, (list, tuple))
                or len(corner) != 2
                or not _finite_number(corner[0])
                or not _finite_number(corner[1])
            ):
                errors.append("roi_corners_px corner {} is invalid".format(index))
    return errors


def validate_canonical_record(
    row: Mapping[str, Any],
    dataset: Mapping[str, Any],
    expected_stage: Optional[str] = None,
    expected_split: Optional[str] = None,
) -> Tuple[List[str], List[str]]:
    """Validate one finalized row and return ``(errors, warnings)``."""

    errors: List[str] = []
    warnings: List[str] = []
    global_id = row.get("global_crop_id")
    crop_id = row.get("crop_id")
    if not global_id:
        errors.append("missing global_crop_id")
    if not crop_id:
        errors.append("missing crop_id")
    if global_id and crop_id and str(global_id) != str(crop_id):
        errors.append("crop_id must equal global_crop_id in canonical labels")
    for key in ("dataset_id", "source_crop_id", "source_group_id"):
        if row.get(key) in (None, ""):
            errors.append("missing {}".format(key))

    path_key = str(dataset.get("crop_path_key", "crop_path"))
    if row.get(path_key) in (None, ""):
        errors.append("missing {}".format(path_key))
    image_size = list(dataset.get("image_size", [256, 256]))
    if len(image_size) != 2:
        errors.append("dataset.image_size must have two values")
        expected_width = expected_height = 256
    else:
        expected_width, expected_height = int(image_size[0]), int(image_size[1])
    try:
        width, height = int(row.get("width")), int(row.get("height"))
    except (TypeError, ValueError):
        width = height = -1
    if (width, height) != (expected_width, expected_height):
        errors.append(
            "record size must be {}x{}; got {}x{}".format(expected_width, expected_height, width, height)
        )

    required_schema = dataset.get("require_schema_version")
    if required_schema is None:
        required_schema = TRAIN_SCHEMA if expected_stage else EVALUATION_SCHEMA if expected_split else None
    if required_schema is not None and str(row.get("schema_version")) != str(required_schema):
        errors.append(
            "schema_version mismatch: expected {}, got {}".format(required_schema, row.get("schema_version"))
        )

    if expected_stage:
        if str(row.get("training_stage")) != expected_stage:
            errors.append(
                "training_stage mismatch: expected {}, got {}".format(expected_stage, row.get("training_stage"))
            )
        if row.get("selection_action") != "include":
            errors.append("canonical train record must have selection_action=include")
        tier = str(row.get("supervision_tier", ""))
        if tier not in {"pseudo", "gold"}:
            errors.append("supervision_tier must be pseudo or gold")
        provenance = str(row.get("annotation_provenance", ""))
        if str(row.get("schema_version")) == "hlml_warehouse_train_v1":
            origin = str(row.get("label_origin"))
            expected_provenance = (
                "human_gold"
                if tier == "gold"
                or origin == "human"
                or origin.endswith("_human_corrected")
                else "mediapipe_pseudo"
            )
        else:
            expected_provenance = {"pseudo": "mediapipe_pseudo", "gold": "human_gold"}.get(tier)
        if expected_provenance and provenance != expected_provenance:
            errors.append(
                "annotation_provenance {} conflicts with supervision_tier {}".format(provenance, tier)
            )
        if str(row.get("quality_tier", "")) not in {"HIGH", "MEDIUM", "PRESENCE_ONLY"}:
            errors.append("included quality_tier must be HIGH, MEDIUM, or PRESENCE_ONLY")
        if not isinstance(row.get("quality_flags"), list):
            errors.append("quality_flags must be a list")
        if row.get("sampling_bucket") in (None, ""):
            errors.append("missing sampling_bucket")
    if expected_split:
        if str(row.get("split")) != expected_split:
            errors.append("split mismatch: expected {}, got {}".format(expected_split, row.get("split")))
        if row.get("ground_truth_valid") is not True:
            errors.append("evaluation record must have ground_truth_valid=true")
        if row.get("palm_valid") is not True:
            errors.append("evaluation record must have palm_valid=true")

    if row.get("ignore_for_training") is True:
        errors.append("ignored row is not allowed in canonical included labels")
    presence = row.get("hand_presence")
    if not isinstance(presence, Mapping) or not isinstance(presence.get("present"), bool):
        errors.append("hand_presence.present must be boolean")
        present = False
    else:
        present = bool(presence["present"])
    handedness = row.get("handedness")
    if not isinstance(handedness, Mapping):
        errors.append("handedness must be an object")
        handedness_label = ""
        handedness_score = None
    else:
        handedness_label = str(handedness.get("label", ""))
        handedness_score = handedness.get("score")
    if handedness_label not in {"Left", "Right", "unknown"}:
        errors.append("handedness.label must be Left, Right, or unknown")
    if handedness_score is not None and not _finite_number(handedness_score):
        errors.append("handedness.score must be null or finite")

    landmark_fields = ("landmarks_crop_norm", "landmarks_crop_px", "landmarks_image_px")
    ordered_fields: Dict[str, List[Tuple[float, float]]] = {}
    if present:
        for field in landmark_fields:
            field_errors, ordered = _validate_points(row.get(field), field)
            errors.extend(field_errors)
            ordered_fields[field] = ordered
        # Fixed-ROI Val/Test may contain a reviewed positive whose handedness
        # remains unknown. Landmark/presence evaluation stays valid and the
        # handedness metric explicitly excludes those rows.
        norm = ordered_fields.get("landmarks_crop_norm") or []
        crop_px = ordered_fields.get("landmarks_crop_px") or []
        outside = sum(not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0) for x, y in norm)
        if outside:
            # HLMF may preserve reviewed landmarks just outside a truncated
            # fixed ROI. Keep the original target and report it without making
            # an otherwise usable Train/Val/Test snapshot unreadable.
            warnings.append("{} normalized landmark(s) are outside [0,1]".format(outside))
        if len(norm) == len(crop_px) == 21 and width > 0 and height > 0:
            for point_id, ((norm_x, norm_y), (px_x, px_y)) in enumerate(zip(norm, crop_px)):
                if abs(norm_x * (width - 1) - px_x) > 0.08 or abs(norm_y * (height - 1) - px_y) > 0.08:
                    errors.append("norm/crop-pixel coordinates disagree at landmark {}".format(point_id))
                    break
    else:
        for field in landmark_fields:
            if row.get(field) not in (None, []):
                errors.append("negative record must have empty {}".format(field))
        if handedness_label != "unknown" or handedness_score is not None:
            errors.append("negative record handedness must be unknown with null score")
        if row.get("hand_id") is not None:
            errors.append("negative record hand_id must be null")

    errors.extend(_validate_geometry(row))
    if expected_stage:
        sample_type = str(row.get("sample_type", ""))
        if sample_type not in SAMPLE_TYPES:
            errors.append("invalid sample_type: {}".format(sample_type))
        elif isinstance(row.get("palm_valid"), bool):
            expected_type = _expected_sample_type(present, bool(row["palm_valid"]))
            if sample_type != expected_type:
                errors.append("sample_type mismatch: expected {}, got {}".format(expected_type, sample_type))

    required_weights = TRAIN_WEIGHT_FIELDS if expected_stage else EVALUATION_WEIGHT_FIELDS
    for key in required_weights:
        value = row.get(key)
        if not _finite_number(value) or float(value) < 0.0:
            errors.append("{} must be finite and non-negative".format(key))
    for key in ("hand_presence_loss_weight", "landmark_loss_weight", "handedness_loss_weight"):
        if _finite_number(row.get(key)) and float(row[key]) not in {0.0, 1.0}:
            errors.append("{} must be a 0/1 head mask".format(key))
    if not errors:
        try:
            effective = effective_head_weights(row)
            if any(not math.isfinite(value) or value < 0.0 for value in effective):
                errors.append("effective head weights must be finite and non-negative")
        except (TypeError, ValueError) as exc:
            errors.append("invalid effective head weights: {}".format(exc))
    return sorted(set(errors)), sorted(set(warnings))


def _inspect_image(path: Path) -> Tuple[int, int, int, str]:
    """Read image header/metadata without importing TensorFlow."""

    try:
        from PIL import Image

        with Image.open(str(path)) as image:
            width, height = image.size
            bands = len(image.getbands())
            return int(width), int(height), int(bands), str(image.mode)
    except ImportError:
        pass
    except Exception as exc:
        raise ValueError("Image header is unreadable: {} ({})".format(path, exc)) from exc

    try:
        from .io_utils import read_image

        image = read_image(path)
    except ImportError as exc:
        raise RuntimeError("Pillow or OpenCV is required to inspect image headers") from exc
    if image is None:
        raise ValueError("Image is unreadable: {}".format(path))
    height, width = image.shape[:2]
    channels = 1 if image.ndim == 2 else int(image.shape[2])
    return int(width), int(height), channels, str(image.dtype)


def audit_canonical_dataset(
    config: Mapping[str, Any],
    dataset: Optional[Mapping[str, Any]] = None,
    labels: Optional[Any] = None,
    expected_stage: Optional[str] = None,
    expected_split: Optional[str] = None,
    check_images: bool = True,
    hash_images: bool = False,
    raise_on_error: bool = False,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Audit one canonical JSONL and return resolved records plus a report."""

    dataset_cfg: Dict[str, Any] = dict(dataset or config.get("data") or {})
    labels_value = labels if labels is not None else dataset_cfg.get("labels")
    if not labels_value:
        raise DatasetContractError("data.labels is required")
    labels_path = _configured_path(labels_value, config)
    report: Dict[str, Any] = {
        "labels": str(labels_path),
        "labels_sha256": None,
        "ignored_labels": None,
        "ignored_labels_sha256": None,
        "ignored_records": 0,
        "ignored": {
            "configured": False,
            "path": None,
            "sha256": None,
            "count": 0,
            "identity_count": 0,
            "missing_identity_count": 0,
            "overlap_count": 0,
            "overlap_ids": [],
        },
        "status": "failed",
        "records": 0,
        "valid_records": 0,
        "invalid_records": 0,
        "counts": {},
        "path_resolution": {"canonical": 0, "rebased": 0},
        "image_shapes": {},
        "duplicate_image_hashes": [],
        "errors": [],
        "warnings": [],
    }
    if not labels_path.is_file():
        report["errors"].append({"scope": "labels", "errors": ["file does not exist"]})
        if raise_on_error:
            raise DatasetContractError("Canonical labels file does not exist: {}".format(labels_path))
        return [], report
    report["labels_sha256"] = sha256_file(labels_path)
    try:
        rows = read_jsonl(labels_path)
    except (OSError, ValueError) as exc:
        report["errors"].append({"scope": "labels", "errors": [str(exc)]})
        if raise_on_error:
            raise DatasetContractError(str(exc)) from exc
        return [], report
    report["records"] = len(rows)
    if not rows:
        report["errors"].append({"scope": "labels", "errors": ["canonical JSONL is empty"]})

    ignored_value = dataset_cfg.get("ignored_labels")
    if not ignored_value and expected_split == "val":
        validation_config = config.get("validation", {})
        if isinstance(validation_config, Mapping):
            ignored_value = validation_config.get("ignored_labels")
    if ignored_value:
        ignored_path = _configured_path(ignored_value, config)
        report["ignored_labels"] = str(ignored_path)
        report["ignored"]["configured"] = True
        report["ignored"]["path"] = str(ignored_path)
        if not ignored_path.is_file():
            report["errors"].append(
                {
                    "scope": "ignored_labels",
                    "errors": ["configured ignored JSONL does not exist: {}".format(ignored_path)],
                }
            )
        else:
            report["ignored_labels_sha256"] = sha256_file(ignored_path)
            report["ignored"]["sha256"] = report["ignored_labels_sha256"]
            try:
                ignored_rows = read_jsonl(ignored_path)
            except (OSError, ValueError) as exc:
                report["errors"].append(
                    {"scope": "ignored_labels", "errors": [str(exc)]}
                )
                ignored_rows = []
            report["ignored_records"] = len(ignored_rows)
            report["ignored"]["count"] = len(ignored_rows)

            included_by_id: Dict[str, List[str]] = defaultdict(list)
            ignored_by_id: Dict[str, List[str]] = defaultdict(list)
            for row in rows:
                for identity in _identity_values(row):
                    included_by_id[identity].append(_record_id(row))
            missing_identity_count = 0
            for row in ignored_rows:
                identities = _identity_values(row)
                if not identities:
                    missing_identity_count += 1
                    continue
                for identity in identities:
                    ignored_by_id[identity].append(_record_id(row))
            overlap = sorted(set(included_by_id) & set(ignored_by_id))
            report["ignored"]["identity_count"] = len(ignored_by_id)
            report["ignored"]["missing_identity_count"] = missing_identity_count
            report["ignored"]["overlap_count"] = len(overlap)
            report["ignored"]["overlap_ids"] = overlap[:20]
            if missing_identity_count:
                report["warnings"].append(
                    {
                        "scope": "ignored_labels",
                        "warnings": [
                            "{} ignored row(s) have neither global_crop_id nor crop_id".format(
                                missing_identity_count
                            )
                        ],
                    }
                )
            if overlap:
                report["errors"].append(
                    {
                        "scope": "included_ignored_identity_overlap",
                        "errors": [
                            "canonical included and ignored JSONL overlap on {} global/crop ID(s)".format(
                                len(overlap)
                            )
                        ],
                        "examples": [
                            {
                                "id": identity,
                                "included_records": sorted(set(included_by_id[identity])),
                                "ignored_records": sorted(set(ignored_by_id[identity])),
                            }
                            for identity in overlap[:20]
                        ],
                    }
                )

    try:
        resolver = CanonicalPathResolver(dataset_cfg, config, labels_path)
    except (DatasetContractError, OSError) as exc:
        report["errors"].append({"scope": "path_policy", "errors": [str(exc)]})
        if raise_on_error:
            raise
        return [], report

    seen_ids: Dict[str, int] = {}
    samples: List[Dict[str, Any]] = []
    hash_to_ids: Dict[str, List[str]] = defaultdict(list)
    counters: Dict[str, Counter] = {
        "presence": Counter(),
        "handedness": Counter(),
        "sample_type": Counter(),
        "supervision_tier": Counter(),
        "sampling_bucket": Counter(),
        "dataset_id": Counter(),
    }
    expected_width, expected_height = [int(value) for value in dataset_cfg.get("image_size", [256, 256])]
    expected_channels = int(dataset_cfg.get("channels", 1))
    for offset, raw in enumerate(rows, start=1):
        row = dict(raw)
        line_number = int(row.get("_jsonl_line", offset))
        record_id = _record_id(row) or "line:{}".format(line_number)
        row_errors, row_warnings = validate_canonical_record(
            row, dataset_cfg, expected_stage=expected_stage, expected_split=expected_split
        )
        if record_id in seen_ids:
            row_errors.append("duplicate global record id; first seen at line {}".format(seen_ids[record_id]))
        else:
            seen_ids[record_id] = line_number
        resolved_path: Optional[Path] = None
        resolution = None
        try:
            resolved_path, resolution = resolver.resolve(row)
        except (OSError, ValueError, DatasetContractError) as exc:
            row_errors.append(str(exc))
        if resolved_path is not None and check_images:
            try:
                width, height, channels, mode = _inspect_image(resolved_path)
                shape_key = "{}x{}x{}:{}".format(width, height, channels, mode)
                report["image_shapes"][shape_key] = report["image_shapes"].get(shape_key, 0) + 1
                if (width, height) != (expected_width, expected_height):
                    row_errors.append(
                        "image shape must be {}x{}; got {}x{}".format(
                            expected_width, expected_height, width, height
                        )
                    )
                if channels != expected_channels:
                    row_errors.append(
                        "image channels must be {}; got {} ({})".format(expected_channels, channels, mode)
                    )
            except (OSError, RuntimeError, ValueError) as exc:
                row_errors.append(str(exc))
        image_hash = None
        if resolved_path is not None and hash_images:
            try:
                image_hash = sha256_file(resolved_path)
                hash_to_ids[image_hash].append(record_id)
                curation = row.get("pretrain_curation") or {}
                expected_image_hash = (
                    curation.get("image_sha256")
                    if isinstance(curation, Mapping)
                    else None
                )
                if expected_image_hash and image_hash != str(expected_image_hash):
                    row_errors.append(
                        "source image hash does not match pretrain_curation.image_sha256"
                    )
            except OSError as exc:
                row_errors.append("could not hash image: {}".format(exc))

        if row_warnings:
            report["warnings"].append(
                {"record_id": record_id, "line": line_number, "warnings": sorted(set(row_warnings))}
            )
        if row_errors:
            report["errors"].append(
                {"record_id": record_id, "line": line_number, "errors": sorted(set(row_errors))}
            )
            continue
        assert resolved_path is not None and resolution is not None
        row["_resolved_crop_path"] = str(resolved_path)
        row["_path_resolution"] = resolution
        if image_hash is not None:
            row["_image_sha256"] = image_hash
        samples.append(row)
        report["path_resolution"][resolution] += 1
        present = bool(row["hand_presence"]["present"])
        counters["presence"]["positive" if present else "negative"] += 1
        counters["handedness"][str(row.get("handedness", {}).get("label", "unknown"))] += 1
        for key in ("sample_type", "supervision_tier", "sampling_bucket", "dataset_id"):
            if row.get(key) is not None:
                counters[key][str(row[key])] += 1

    report["counts"] = {key: dict(value) for key, value in counters.items()}
    report["valid_records"] = len(samples)
    report["invalid_records"] = len(rows) - len(samples)
    report["duplicate_image_hashes"] = [
        {"sha256": digest, "record_ids": record_ids}
        for digest, record_ids in sorted(hash_to_ids.items())
        if len(record_ids) > 1
    ]
    if report["duplicate_image_hashes"]:
        report["warnings"].append(
            {
                "scope": "within_dataset_exact_duplicates",
                "count": len(report["duplicate_image_hashes"]),
                "warnings": ["multiple canonical rows resolve to byte-identical crop images"],
            }
        )
    report["status"] = "ok" if not report["errors"] else "failed"
    if raise_on_error and report["errors"]:
        preview = report["errors"][:5]
        raise DatasetContractError(
            "Dataset contract failed for {} ({} invalid row(s)): {}".format(
                labels_path, report["invalid_records"], preview
            )
        )
    return samples, report


def leakage_report(
    first_name: str,
    first: Sequence[Mapping[str, Any]],
    second_name: str,
    second: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Find exact cross-split identity/path/content leakage and name warnings."""

    fatal: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []
    exact_keys = (
        ("global_crop_id", "global_crop_id"),
        ("source_group_id", "source_group_id"),
        ("_resolved_crop_path", "resolved_crop_path"),
        ("_image_sha256", "image_sha256"),
    )
    for row_key, label in exact_keys:
        left = {str(row.get(row_key)): _record_id(row) for row in first if row.get(row_key) not in (None, "")}
        right = {str(row.get(row_key)): _record_id(row) for row in second if row.get(row_key) not in (None, "")}
        overlap = sorted(set(left) & set(right))
        if overlap:
            fatal.append(
                {
                    "kind": label,
                    "count": len(overlap),
                    "examples": [
                        {"value": value, first_name: left[value], second_name: right[value]}
                        for value in overlap[:20]
                    ],
                }
            )
    for row_key in ("source_crop_id", "source_image", "image"):
        left = {str(row.get(row_key)) for row in first if row.get(row_key) not in (None, "")}
        right = {str(row.get(row_key)) for row in second if row.get(row_key) not in (None, "")}
        overlap = sorted(left & right)
        if overlap:
            warnings.append(
                {
                    "kind": "same_{}_name".format(row_key),
                    "count": len(overlap),
                    "examples": overlap[:20],
                    "note": "names alone are not proof of leakage across namespaced sources",
                }
            )
    return {
        "first": first_name,
        "second": second_name,
        "status": "failed" if fatal else "ok",
        "fatal": fatal,
        "warnings": warnings,
    }


def inspect_config(
    config: Mapping[str, Any],
    compare_labels: Optional[Sequence[Any]] = None,
    check_images: bool = True,
    hash_images: bool = True,
) -> Dict[str, Any]:
    """Inspect the primary configured dataset and configured comparison sets."""

    dataset_cfg = dict(config.get("data") or {})
    curation_manifest = verify_dataset_curation_manifest(
        config, dataset_cfg, error_type=DatasetContractError
    )
    task = str(config.get("task", ""))
    stage = str(dataset_cfg.get("require_training_stage") or config.get("stage") or "") or None
    split = str(dataset_cfg.get("require_split") or config.get("split") or "") or None
    primary_rows, primary_report = audit_canonical_dataset(
        config,
        dataset=dataset_cfg,
        expected_stage=stage if task == "train" or stage else None,
        expected_split=split if not stage else None,
        check_images=check_images,
        hash_images=hash_images,
        raise_on_error=False,
    )
    datasets: Dict[str, Dict[str, Any]] = {"primary": primary_report}
    row_sets: Dict[str, List[Dict[str, Any]]] = {"primary": primary_rows}

    validation_cfg = config.get("validation", {})
    if task == "train" and validation_cfg.get("enabled", False) and validation_cfg.get("labels"):
        val_dataset = dict(dataset_cfg)
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
            if key in validation_cfg:
                val_dataset[key] = validation_cfg[key]
        val_dataset["labels"] = validation_cfg["labels"]
        if validation_cfg.get("ignored_labels"):
            val_dataset["ignored_labels"] = validation_cfg["ignored_labels"]
        val_dataset["require_schema_version"] = EVALUATION_SCHEMA
        val_dataset.pop("require_training_stage", None)
        val_dataset["require_split"] = "val"
        rows, value = audit_canonical_dataset(
            config,
            dataset=val_dataset,
            expected_split="val",
            check_images=check_images,
            hash_images=hash_images,
            raise_on_error=False,
        )
        datasets["validation"] = value
        row_sets["validation"] = rows

    inspection_cfg = config.get("inspection", {})
    configured_value = (
        inspection_cfg.get("compare_datasets", [])
        if isinstance(inspection_cfg, Mapping)
        else []
    )
    configured_specs: List[Tuple[str, Mapping[str, Any]]] = []
    if isinstance(configured_value, Mapping):
        for name, specification in configured_value.items():
            if not isinstance(specification, Mapping):
                raise DatasetContractError(
                    "inspection.compare_datasets.{} must be a mapping".format(name)
                )
            configured_specs.append((str(name), specification))
    elif isinstance(configured_value, list):
        for index, specification in enumerate(configured_value, start=1):
            if not isinstance(specification, Mapping):
                raise DatasetContractError(
                    "inspection.compare_datasets item {} must be a mapping".format(index)
                )
            name = str(specification.get("name") or "configured_comparison_{}".format(index))
            configured_specs.append((name, specification))
    elif configured_value not in (None, ""):
        raise DatasetContractError("inspection.compare_datasets must be a mapping or list")

    for name, specification in configured_specs:
        if name in row_sets:
            raise DatasetContractError("Duplicate inspection dataset name: {}".format(name))
        comparison_dataset = dict(dataset_cfg)
        for key in (
            "require_training_stage",
            "require_split",
            "require_schema_version",
            "ignored_labels",
        ):
            comparison_dataset.pop(key, None)
        comparison_dataset.update(
            {key: value for key, value in specification.items() if key != "name"}
        )
        comparison_stage = str(comparison_dataset.get("require_training_stage") or "") or None
        comparison_split = str(comparison_dataset.get("require_split") or "") or None
        if comparison_stage and comparison_split:
            raise DatasetContractError(
                "inspection dataset {} cannot require both a training stage and an evaluation split".format(
                    name
                )
            )
        if comparison_stage:
            if comparison_stage not in {"pretrain", "finetune"}:
                raise DatasetContractError(
                    "inspection dataset {} has invalid require_training_stage {!r}".format(
                        name, comparison_stage
                    )
                )
            if comparison_dataset.get("require_schema_version") != TRAIN_SCHEMA:
                raise DatasetContractError(
                    "inspection training dataset {} must require schema {}".format(
                        name, TRAIN_SCHEMA
                    )
                )
        if comparison_split:
            if comparison_split not in {"val", "test"}:
                raise DatasetContractError(
                    "inspection dataset {} has invalid require_split {!r}".format(
                        name, comparison_split
                    )
                )
            if comparison_dataset.get("require_schema_version") != EVALUATION_SCHEMA:
                raise DatasetContractError(
                    "inspection evaluation dataset {} must require schema {}".format(
                        name, EVALUATION_SCHEMA
                    )
                )
        rows, value = audit_canonical_dataset(
            config,
            dataset=comparison_dataset,
            expected_stage=comparison_stage,
            expected_split=comparison_split,
            check_images=check_images,
            hash_images=hash_images,
            raise_on_error=False,
        )
        datasets[name] = value
        row_sets[name] = rows

    for index, labels_value in enumerate(compare_labels or [], start=1):
        name = "comparison_{}".format(index)
        comparison_dataset = dict(dataset_cfg)
        comparison_dataset["labels"] = labels_value
        comparison_dataset.pop("require_training_stage", None)
        comparison_dataset.pop("require_split", None)
        comparison_dataset.pop("require_schema_version", None)
        comparison_dataset.pop("ignored_labels", None)
        rows, value = audit_canonical_dataset(
            config,
            dataset=comparison_dataset,
            check_images=check_images,
            hash_images=hash_images,
            raise_on_error=False,
        )
        datasets[name] = value
        row_sets[name] = rows

    leakages: List[Dict[str, Any]] = []
    names = list(row_sets)
    for left_index, left_name in enumerate(names):
        for right_name in names[left_index + 1 :]:
            leakages.append(
                leakage_report(left_name, row_sets[left_name], right_name, row_sets[right_name])
            )
    failed_datasets = [name for name, value in datasets.items() if value.get("status") != "ok"]
    failed_leakage = [value for value in leakages if value.get("status") != "ok"]
    return {
        "status": "failed" if failed_datasets or failed_leakage else "ok",
        "datasets": datasets,
        "leakage": leakages,
        "failed_datasets": failed_datasets,
        "failed_leakage_checks": len(failed_leakage),
        "curation_manifest": curation_manifest,
    }


def inspect_config_path(
    config_path: Any,
    compare_labels: Optional[Sequence[Any]] = None,
    check_images: bool = True,
    hash_images: bool = True,
) -> Dict[str, Any]:
    return inspect_config(
        load_config(config_path),
        compare_labels=compare_labels,
        check_images=check_images,
        hash_images=hash_images,
    )
