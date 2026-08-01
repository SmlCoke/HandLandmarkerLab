"""Gold-set Hand Landmarker evaluation on the provided Val/Test ROI crops.

Val and Test JSONL rows already identify the 256x256 Hand ROI that is the
model input.  Palm detection belongs only to arbitrary full-image inference
and is intentionally unavailable from this evaluation entry point.
"""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .config import resolve_path
from .contracts import (
    ordered_landmarks,
    validate_checkpoint_path_stage,
    validate_label_record,
    validate_model_checkpoint_stage,
)
from .io_utils import read_image, sha256_file, write_json, write_jsonl
from .inspect import audit_canonical_dataset
from .metrics import EvaluationMetrics, threshold_sweep
from .runtime import create_hand_predictor, normalize_runtime_config


def _labels_path(config: Mapping[str, Any]) -> Path:
    value = config.get("data", {}).get("labels")
    if not value:
        raise KeyError("Evaluation config requires data.labels")
    return resolve_path(str(value), config)


def _output_dir(config: Mapping[str, Any]) -> Path:
    value = config.get("output", {}).get("dir")
    if not value:
        raise KeyError("Evaluation config requires output.dir")
    path = resolve_path(str(value), config)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _validate_rows(rows: Sequence[Mapping[str, Any]], split: str) -> None:
    seen = set()
    failures = []
    for row in rows:
        record_id = str(row.get("global_crop_id") or row.get("crop_id") or "")
        if record_id in seen:
            failures.append({"id": record_id, "errors": ["duplicate record id"]})
        seen.add(record_id)
        errors = validate_label_record(row, split)
        if errors:
            failures.append({"id": record_id, "errors": errors})
    if failures:
        preview = failures[:10]
        raise ValueError("Evaluation labels failed validation ({} rows): {}".format(len(failures), preview))


def _expected_crop_points(row: Mapping[str, Any]) -> Optional[List[Tuple[float, float]]]:
    if not bool((row.get("hand_presence") or {}).get("present", False)):
        return None
    values = row.get("landmarks_crop_px") or []
    if len(values) == 21:
        ordered = sorted(values, key=lambda point: int(point.get("id", 0)))
        return [(float(point["x"]), float(point["y"])) for point in ordered]
    return [(x_value * 255.0, y_value * 255.0) for x_value, y_value in ordered_landmarks(row)]


def _row_metrics(
    row: Mapping[str, Any],
    hand,
    threshold: float,
    predicted_crop_points: Optional[Sequence[Tuple[float, float]]],
) -> Dict[str, Any]:
    expected_presence = bool((row.get("hand_presence") or {}).get("present", False))
    expected_handedness = str((row.get("handedness") or {}).get("label", "unknown")).title()
    expected_points = _expected_crop_points(row)
    predicted_presence = bool(hand is not None and hand.hand_flag_score >= threshold)
    result: Dict[str, Any] = {
        "crop_id": row.get("global_crop_id") or row.get("crop_id"),
        "dataset_id": row.get("dataset_id"),
        "source_image": row.get("source_image") or row.get("image"),
        "crop_path": row.get("crop_path"),
        "resolved_crop_path": row.get("_resolved_crop_path"),
        "expected_presence": expected_presence,
        "predicted_presence": predicted_presence,
        "hand_flag_score": None if hand is None else hand.hand_flag_score,
        "expected_handedness": expected_handedness,
        "predicted_handedness": None if hand is None else hand.handedness,
        "handedness_score": None if hand is None else hand.handedness_score,
        "landmark_prediction_available": predicted_crop_points is not None,
    }
    if hand is not None:
        normalized_points = [
            [float(point[0]), float(point[1])] for point in hand.landmarks_norm
        ]
        fallback_max_abs = max(
            (abs(value) for point in normalized_points for value in point),
            default=0.0,
        )
        fallback_out_of_range = sum(
            value < 0.0 or value > 1.0
            for point in normalized_points
            for value in point
        )
        result.update(
            {
                "landmarks_roi_norm": normalized_points,
                "landmark_raw_max_abs": float(
                    getattr(hand, "landmark_raw_max_abs", fallback_max_abs)
                ),
                "normalized_out_of_range_coordinate_count": int(
                    getattr(
                        hand,
                        "normalized_out_of_range_coordinate_count",
                        fallback_out_of_range,
                    )
                ),
            }
        )
    else:
        result.update(
            {
                "landmarks_roi_norm": None,
                "landmark_raw_max_abs": None,
                "normalized_out_of_range_coordinate_count": None,
            }
        )
    if expected_points is not None and predicted_crop_points is not None:
        errors = [
            math.hypot(float(pred[0]) - float(target[0]), float(pred[1]) - float(target[1]))
            for target, pred in zip(expected_points, predicted_crop_points)
        ]
        normalizer = max(
            math.hypot(
                max(point[0] for point in expected_points) - min(point[0] for point in expected_points),
                max(point[1] for point in expected_points) - min(point[1] for point in expected_points),
            ),
            1e-6,
        )
        result["mean_landmark_error_px"] = sum(errors) / 21.0
        result["nme"] = result["mean_landmark_error_px"] / normalizer
        result["landmark_errors_px_by_id"] = errors
    else:
        result["mean_landmark_error_px"] = None
        result["nme"] = None
        result["landmark_errors_px_by_id"] = None
    return result


def _update_aggregate(
    metrics: EvaluationMetrics,
    row: Mapping[str, Any],
    hand,
    threshold: float,
    predicted_crop_points: Optional[Sequence[Tuple[float, float]]],
) -> None:
    expected_presence = bool((row.get("hand_presence") or {}).get("present", False))
    expected_handedness = str((row.get("handedness") or {}).get("label", "unknown")).title()
    metrics.update(
        expected_presence=expected_presence,
        predicted_presence=bool(hand is not None and hand.hand_flag_score >= threshold),
        expected_landmarks=_expected_crop_points(row),
        predicted_landmarks=predicted_crop_points,
        expected_handedness=expected_handedness,
        predicted_handedness=None if hand is None else hand.handedness,
    )


def evaluate_hand_rois(
    config: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    model_checkpoint_stage: Optional[str] = None,
) -> Dict[str, Any]:
    """Run only Hand Landmarker on each canonical ``crop_path`` ROI."""

    if model_checkpoint_stage is None:
        # Keep this lower-level helper usable in isolated metric tests while
        # making any supplied provenance fail closed.  The public config entry
        # point below always requires the field.
        model_checkpoint_stage = validate_model_checkpoint_stage(config, required=False)
    predictor = create_hand_predictor(normalize_runtime_config(config))
    batch_size = int(config.get("inference", {}).get("batch_size", 64))
    if batch_size <= 0:
        raise ValueError("inference.batch_size must be positive")
    threshold = float(config.get("evaluation", {}).get("hand_flag_threshold", 0.5))
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("evaluation.hand_flag_threshold must be finite and in [0,1]")
    metrics = EvaluationMetrics(config.get("evaluation", {}).get("pck_thresholds", [0.05, 0.10, 0.15]))
    details: List[Dict[str, Any]] = []
    scores: List[Tuple[bool, float]] = []
    raw_max_abs_values: List[float] = []
    out_of_range_hand_count = 0
    out_of_range_coordinate_count = 0
    collapse_positive_count = 0
    collapse_count = 0
    group_fields = (
        "dataset_id",
        "capture_source_id",
        "label_origin",
        "annotation_style",
        "distance",
        "lighting",
    )
    grouped_metrics: Dict[str, Dict[str, EvaluationMetrics]] = {
        field: {} for field in group_fields
    }
    grouped_collapse: Dict[str, Dict[str, List[int]]] = {
        field: defaultdict(lambda: [0, 0]) for field in group_fields
    }

    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start : start + batch_size]
        crops = []
        for row in batch_rows:
            resolved_value = row.get("_resolved_crop_path")
            if resolved_value in (None, ""):
                raise ValueError(
                    "Evaluation row {} is missing audited _resolved_crop_path".format(
                        row.get("global_crop_id") or row.get("crop_id")
                    )
                )
            path = Path(str(resolved_value))
            if not path.is_file():
                raise FileNotFoundError("Audited ROI crop no longer exists: {}".format(path))
            image = read_image(path)
            if image is None:
                raise ValueError("Could not read ROI crop: {}".format(path))
            if image.ndim != 2 or tuple(image.shape) != (256, 256):
                raise ValueError(
                    "Val/Test crop_path must be an existing 256x256 one-channel Hand ROI; got {} for {}".format(
                        getattr(image, "shape", None), path
                    )
                )
            if str(image.dtype) != "uint8":
                raise ValueError(
                    "Val/Test Hand ROI must be uint8 before /255 normalization; got {} for {}".format(
                        image.dtype, path
                    )
                )
            crops.append(image)
        predictions = predictor.predict(crops, batch_size=batch_size)
        if len(predictions) != len(batch_rows):
            raise RuntimeError(
                "Hand predictor returned {} rows for an evaluation batch of {}".format(
                    len(predictions), len(batch_rows)
                )
            )
        for row, hand in zip(batch_rows, predictions):
            predicted_points = [(point[0] * 255.0, point[1] * 255.0) for point in hand.landmarks_norm]
            fallback_max_abs = max(
                (abs(float(value)) for point in hand.landmarks_norm for value in point),
                default=0.0,
            )
            raw_max_abs = float(getattr(hand, "landmark_raw_max_abs", fallback_max_abs))
            out_of_range = int(
                getattr(
                    hand,
                    "normalized_out_of_range_coordinate_count",
                    sum(
                        float(value) < 0.0 or float(value) > 1.0
                        for point in hand.landmarks_norm
                        for value in point
                    ),
                )
            )
            raw_max_abs_values.append(raw_max_abs)
            out_of_range_hand_count += int(out_of_range > 0)
            out_of_range_coordinate_count += out_of_range
            _update_aggregate(metrics, row, hand, threshold, predicted_points)
            detail = _row_metrics(row, hand, threshold, predicted_points)
            detail["evaluation_scope"] = "provided_hand_roi"
            detail["model_checkpoint_stage"] = model_checkpoint_stage
            expected_positive = bool((row.get("hand_presence") or {}).get("present", False))
            if predicted_points:
                xs = [float(point[0]) for point in predicted_points]
                ys = [float(point[1]) for point in predicted_points]
                collapse = math.hypot(max(xs) - min(xs), max(ys) - min(ys)) < (
                    0.05 * math.hypot(255.0, 255.0)
                )
            else:
                collapse = False
            detail["landmark_collapse"] = bool(collapse)
            for field in group_fields:
                value = str(row.get(field) if row.get(field) not in (None, "") else "unknown")
                detail[field] = row.get(field)
                metric = grouped_metrics[field].setdefault(
                    value,
                    EvaluationMetrics(config.get("evaluation", {}).get("pck_thresholds", [0.05, 0.10, 0.15])),
                )
                _update_aggregate(metric, row, hand, threshold, predicted_points)
                if expected_positive:
                    grouped_collapse[field][value][1] += 1
                    grouped_collapse[field][value][0] += int(collapse)
            if expected_positive:
                collapse_positive_count += 1
                collapse_count += int(collapse)
            details.append(detail)
            scores.append((bool((row.get("hand_presence") or {}).get("present", False)), hand.hand_flag_score))

    evaluation_config = config.get("evaluation", {})
    report = {
        "model_checkpoint_stage": model_checkpoint_stage,
        "scope": "hand_landmarker_on_provided_hand_roi",
        "protocol": "Val/Test crop_path is the model input; Palm Detector is not loaded or run.",
        "metrics": metrics.report(),
        "landmark_output_health": {
            "prediction_count": len(raw_max_abs_values),
            "non_finite_outputs_rejected": True,
            "maximum_raw_absolute_value": max(raw_max_abs_values) if raw_max_abs_values else None,
            "normalized_out_of_range_hand_count": out_of_range_hand_count,
            "normalized_out_of_range_coordinate_count": out_of_range_coordinate_count,
        },
        "presence_threshold": threshold,
        "collapse": {
            "count": collapse_count,
            "eligible_positive_count": collapse_positive_count,
            "rate": (
                float(collapse_count) / float(collapse_positive_count)
                if collapse_positive_count
                else 0.0
            ),
            "definition": "predicted 21-point bounding diagonal below 0.05 of 256px ROI diagonal",
        },
        "grouped_metrics": {
            field: {
                value: {
                    **metric.report(),
                    "collapse": {
                        "count": grouped_collapse[field][value][0],
                        "eligible_positive_count": grouped_collapse[field][value][1],
                        "rate": (
                            float(grouped_collapse[field][value][0])
                            / float(grouped_collapse[field][value][1])
                            if grouped_collapse[field][value][1]
                            else 0.0
                        ),
                    },
                }
                for value, metric in sorted(values.items())
            }
            for field, values in grouped_metrics.items()
        },
        "details": details,
    }
    if bool(evaluation_config.get("tune_thresholds", False)):
        sweep_values = evaluation_config.get("threshold_sweep") or [
            round(value / 20.0, 2) for value in range(1, 20)
        ]
        if any(
            not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0
            for value in sweep_values
        ):
            raise ValueError("evaluation.threshold_sweep values must be finite and in [0,1]")
        report["presence_threshold_sweep"] = threshold_sweep(scores, sweep_values)
    return report


def evaluate_from_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    if config.get("schema_version", 1) != 1:
        raise ValueError("Unsupported configuration schema_version")
    if str(config.get("task", "evaluate")).lower() != "evaluate":
        raise ValueError("Evaluation entry point requires task: evaluate")
    split = str(config.get("split", "")).lower()
    if split not in {"val", "test"}:
        raise ValueError("Evaluation split must be val or test; got {!r}".format(split))
    if split == "test" and bool(config.get("evaluation", {}).get("tune_thresholds", False)):
        raise ValueError("Test is locked evaluation and must not tune or sweep thresholds")
    mode = str(config.get("evaluation", {}).get("mode", "")).lower()
    if mode != "roi":
        raise ValueError(
            "Unsupported evaluation.mode {!r}; Val/Test evaluation accepts provided Hand ROIs only".format(
                mode
            )
        )
    output_dir = _output_dir(config)
    predictions_path = output_dir / "predictions.jsonl"
    metrics_path = output_dir / "metrics.json"
    if not bool(config.get("output", {}).get("overwrite", False)):
        existing = [str(path) for path in (predictions_path, metrics_path) if path.exists()]
        if existing:
            raise FileExistsError(
                "Evaluation output already exists; set output.overwrite=true or choose a new output.dir: {}".format(
                    existing
                )
            )
    model_checkpoint_stage = validate_model_checkpoint_stage(config)
    runtime_config = normalize_runtime_config(config)
    validate_checkpoint_path_stage(
        config,
        runtime_config.get("hand", {}).get("model_path", ""),
    )
    labels_path = _labels_path(config)
    data_config = config.get("data", {})
    rows, data_contract_report = audit_canonical_dataset(
        config,
        dataset=data_config,
        expected_split=split,
        check_images=True,
        hash_images=False,
        raise_on_error=True,
    )
    _validate_rows(rows, split)
    result = evaluate_hand_rois(
        config,
        rows,
        model_checkpoint_stage=model_checkpoint_stage,
    )

    details = result.pop("details")
    hand_config = runtime_config.get("hand", {})
    model_path = Path(str(hand_config.get("model_path", "")))
    if not model_path.is_file():
        raise FileNotFoundError("Evaluated Hand model is not a file: {}".format(model_path))
    config_path_value = config.get("_meta", {}).get("config_path")
    config_path = Path(str(config_path_value)) if config_path_value else None
    labels_sha256 = sha256_file(labels_path)
    model_sha256 = sha256_file(model_path)
    report = {
        "labels_path": str(labels_path),
        "labels_sha256": labels_sha256,
        "hand_model": {
            "backend": str(hand_config.get("backend", "keras")),
            "path": str(model_path),
            "sha256": model_sha256,
            "checkpoint_stage": model_checkpoint_stage,
        },
        "config_path": str(config_path) if config_path else None,
        "config_sha256": (
            sha256_file(config_path) if config_path is not None and config_path.is_file() else None
        ),
        "label_count": len(rows),
        "data_contract": data_contract_report,
        "split": split,
        **result,
    }
    for detail in details:
        detail["labels_sha256"] = labels_sha256
        detail["hand_model_sha256"] = model_sha256
    write_jsonl(predictions_path, details)
    write_json(metrics_path, report)
    return report
