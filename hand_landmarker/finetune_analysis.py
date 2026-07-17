"""Paired finetune evaluation and inference analysis with bounded overlays."""

from __future__ import annotations

import json
import math
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from .contracts import STRUCTURAL_BONES
from .io_utils import read_image, read_jsonl, sha256_file, write_json, write_jsonl


def _identity(row: Mapping[str, Any]) -> str:
    return str(row.get("global_crop_id") or row.get("crop_id") or row.get("image") or "")


def _points(value: Any) -> List[Tuple[float, float]]:
    if not isinstance(value, Sequence) or len(value) != 21:
        return []
    if isinstance(value[0], Mapping):
        ordered = sorted(value, key=lambda point: int(point.get("id", 0)))
        return [(float(point["x"]), float(point["y"])) for point in ordered]
    return [(float(point[0]), float(point[1])) for point in value]


def _gold_points(row: Mapping[str, Any]) -> List[Tuple[float, float]]:
    normalized = _points(row.get("landmarks_crop_norm"))
    if normalized:
        return normalized
    pixels = _points(row.get("landmarks_crop_px"))
    return [(x / 255.0, y / 255.0) for x, y in pixels]


def _spread(points: Sequence[Tuple[float, float]]) -> float | None:
    if len(points) != 21:
        return None
    return math.hypot(
        max(point[0] for point in points) - min(point[0] for point in points),
        max(point[1] for point in points) - min(point[1] for point in points),
    )


def _bone_error(
    expected: Sequence[Tuple[float, float]], predicted: Sequence[Tuple[float, float]]
) -> float | None:
    if len(expected) != 21 or len(predicted) != 21:
        return None
    return sum(
        math.hypot(
            (predicted[end][0] - predicted[start][0]) - (expected[end][0] - expected[start][0]),
            (predicted[end][1] - predicted[start][1]) - (expected[end][1] - expected[start][1]),
        )
        for start, end in STRUCTURAL_BONES
    ) / len(STRUCTURAL_BONES)


def _mean(values: Iterable[float | None]) -> float | None:
    valid = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return sum(valid) / len(valid) if valid else None


def _percentile(values: Iterable[float | None], fraction: float) -> float | None:
    valid = sorted(float(value) for value in values if value is not None and math.isfinite(float(value)))
    if not valid:
        return None
    return valid[int(round((len(valid) - 1) * fraction))]


def _decorate_predictions(
    predictions: Sequence[Mapping[str, Any]], labels: Mapping[str, Mapping[str, Any]]
) -> List[Dict[str, Any]]:
    result = []
    for raw in predictions:
        row = dict(raw)
        identity = _identity(row)
        gold = labels.get(identity, {})
        expected = _gold_points(gold)
        predicted = _points(row.get("landmarks_roi_norm"))
        gold_spread = _spread(expected)
        predicted_spread = _spread(predicted)
        row.update(
            {
                "crop_id": identity,
                "dataset_id": row.get("dataset_id") or gold.get("dataset_id"),
                "image_sha256": (
                    gold.get("image_sha256")
                    or gold.get("normalized_pixel_sha256")
                ),
                "gold_spread": gold_spread,
                "predicted_spread": predicted_spread,
                "predicted_gold_spread_ratio": (
                    predicted_spread / gold_spread
                    if predicted_spread is not None and gold_spread not in (None, 0.0)
                    else None
                ),
                "bone_vector_error_norm": _bone_error(expected, predicted),
                "gold_landmarks_roi_norm": [list(point) for point in expected],
            }
        )
        result.append(row)
    return result


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    positives = [row for row in rows if row.get("mean_landmark_error_px") is not None]
    by_dataset: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_dataset[str(row.get("dataset_id") or "unknown")].append(row)
    per_landmark = []
    for index in range(21):
        values = [
            row["landmark_errors_px_by_id"][index]
            for row in positives
            if isinstance(row.get("landmark_errors_px_by_id"), list)
            and len(row["landmark_errors_px_by_id"]) == 21
        ]
        per_landmark.append({"landmark_id": index, "mean_error_px": _mean(values)})
    return {
        "records": len(rows),
        "positive_metric_records": len(positives),
        "mean_landmark_error_px": _mean(row.get("mean_landmark_error_px") for row in rows),
        "p90_landmark_error_px": _percentile((row.get("mean_landmark_error_px") for row in rows), 0.9),
        "mean_nme": _mean(row.get("nme") for row in rows),
        "mean_bone_vector_error_norm": _mean(row.get("bone_vector_error_norm") for row in rows),
        "collapse_count_spread_lt_0_10": sum(
            row.get("predicted_spread") is not None and float(row["predicted_spread"]) < 0.10
            for row in rows
        ),
        "presence_accuracy": _mean(
            float(row.get("expected_presence") == row.get("predicted_presence")) for row in rows
        ),
        "handedness_accuracy_known": _mean(
            float(row.get("expected_handedness") == row.get("predicted_handedness"))
            for row in rows
            if str(row.get("expected_handedness")) in {"Left", "Right"}
        ),
        "pck": {
            str(threshold): _mean(
                float(error <= threshold * 255.0)
                for row in positives
                for error in (row.get("landmark_errors_px_by_id") or [])
            )
            for threshold in (0.05, 0.10, 0.15)
        },
        "per_dataset": {
            dataset: {
                "records": len(group),
                "mean_landmark_error_px": _mean(row.get("mean_landmark_error_px") for row in group),
                "collapse_count": sum(
                    row.get("predicted_spread") is not None
                    and float(row["predicted_spread"]) < 0.10
                    for row in group
                ),
            }
            for dataset, group in sorted(by_dataset.items())
        },
        "per_landmark": per_landmark,
    }


def _paired(
    baseline: Sequence[Mapping[str, Any]], candidate: Sequence[Mapping[str, Any]]
) -> List[Dict[str, Any]]:
    left = {_identity(row): row for row in baseline}
    right = {_identity(row): row for row in candidate}
    result = []
    for identity in sorted(set(left) & set(right)):
        first, second = left[identity], right[identity]
        base_error = first.get("mean_landmark_error_px")
        candidate_error = second.get("mean_landmark_error_px")
        result.append(
            {
                "crop_id": identity,
                "dataset_id": second.get("dataset_id"),
                "baseline_error_px": base_error,
                "candidate_error_px": candidate_error,
                "error_delta_px": (
                    float(candidate_error) - float(base_error)
                    if candidate_error is not None and base_error is not None
                    else None
                ),
                "baseline_spread": first.get("predicted_spread"),
                "candidate_spread": second.get("predicted_spread"),
                "gold_spread": second.get("gold_spread"),
                "candidate_bone_error": second.get("bone_vector_error_norm"),
                "crop_path": second.get("resolved_crop_path") or second.get("crop_path"),
                "gold_landmarks_roi_norm": second.get("gold_landmarks_roi_norm"),
                "baseline_landmarks_roi_norm": first.get("landmarks_roi_norm"),
                "candidate_landmarks_roi_norm": second.get("landmarks_roi_norm"),
            }
        )
    return result


def _infer_summary(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {"status": "missing", "path": str(path)}
    rows = read_jsonl(path)
    detections = [item for row in rows for item in row.get("detections", [])]
    spreads = [_spread(_points(item.get("landmarks_roi_norm"))) for item in detections]
    return {
        "status": "ok",
        "path": str(path),
        "path_sha256": sha256_file(path),
        "image_count": len(rows),
        "detection_count": len(detections),
        "predicted_hand_count": sum(float(item.get("hand_flag_score", 0.0)) >= 0.5 for item in detections),
        "collapse_count_spread_lt_0_10": sum(value is not None and value < 0.10 for value in spreads),
        "mean_spread": _mean(spreads),
    }


def _paired_infer(baseline_path: Path, candidate_path: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if not baseline_path.is_file() or not candidate_path.is_file():
        return [], {"status": "missing", "matched_images": 0}
    baseline = {str(row.get("image") or ""): row for row in read_jsonl(baseline_path)}
    candidate = {str(row.get("image") or ""): row for row in read_jsonl(candidate_path)}
    rows: List[Dict[str, Any]] = []
    for image in sorted((set(baseline) & set(candidate)) - {""}):
        first = baseline[image]
        second = candidate[image]
        first_detections = list(first.get("detections") or [])
        second_detections = list(second.get("detections") or [])
        point_deltas = []
        for before, after in zip(first_detections, second_detections):
            before_points = _points(before.get("landmarks_roi_norm"))
            after_points = _points(after.get("landmarks_roi_norm"))
            if len(before_points) == 21 and len(after_points) == 21:
                point_deltas.append(
                    sum(
                        math.hypot(x1 - x0, y1 - y0)
                        for (x0, y0), (x1, y1) in zip(before_points, after_points)
                    )
                    / 21.0
                )
        candidate_spreads = [
            value
            for value in (_spread(_points(item.get("landmarks_roi_norm"))) for item in second_detections)
            if value is not None
        ]
        candidate_hand_count = sum(
            float(item.get("hand_flag_score", 0.0)) >= 0.5 for item in second_detections
        )
        # Palm count changes expose detector/ROI failures; low hand scores and
        # collapsed ROI landmarks expose Hand-stage failures on a valid Palm ROI.
        failure_score = (
            abs(len(first_detections) - len(second_detections)) * 3.0
            + max(0, len(second_detections) - candidate_hand_count) * 2.0
            + sum(value < 0.10 for value in candidate_spreads)
            + (1.0 if second.get("error") else 0.0)
        )
        rows.append(
            {
                "crop_id": image,
                "image": image,
                "candidate_rendered": second.get("rendered"),
                "baseline_palm_count": len(first_detections),
                "candidate_palm_count": len(second_detections),
                "candidate_hand_count": candidate_hand_count,
                "candidate_collapse_count": sum(value < 0.10 for value in candidate_spreads),
                "mean_landmark_delta_norm": _mean(point_deltas),
                "failure_score": failure_score,
                "overlay_source": "inference",
            }
        )
    return rows, {
        "status": "ok",
        "matched_images": len(rows),
        "mean_landmark_delta_norm": _mean(row.get("mean_landmark_delta_norm") for row in rows),
        "palm_or_roi_failure_images": sum(float(row["failure_score"]) > 0.0 for row in rows),
    }


def _draw_overlays(
    rows: Sequence[Mapping[str, Any]],
    infer_rows: Sequence[Mapping[str, Any]],
    output: Path,
    limit: int,
) -> List[str]:
    import cv2

    categories = {
        "student_collapse": sorted(
            rows,
            key=lambda row: float(row.get("candidate_spread") or math.inf),
        ),
        "high_landmark_error": sorted(
            rows,
            key=lambda row: float(row.get("candidate_error_px") or -math.inf),
            reverse=True,
        ),
        "improved_vs_baseline": sorted(
            rows,
            key=lambda row: float(row.get("error_delta_px") or math.inf),
        ),
        "palm_or_roi_failure": sorted(
            infer_rows,
            key=lambda row: float(row.get("failure_score") or 0.0),
            reverse=True,
        ),
    }
    written: List[str] = []
    per_category = max(1, limit // len(categories))
    for category, candidates in categories.items():
        accepted = 0
        for row in candidates:
            if len(written) >= limit or accepted >= per_category:
                break
            if category == "student_collapse" and not (
                row.get("candidate_spread") is not None and float(row["candidate_spread"]) < 0.10
            ):
                continue
            if category == "palm_or_roi_failure" and float(row.get("failure_score") or 0.0) <= 0.0:
                continue
            path = Path(
                str(
                    row.get("candidate_rendered")
                    or row.get("crop_path")
                    or row.get("image")
                    or ""
                )
            )
            image = read_image(path) if path.is_file() else None
            if image is None:
                continue
            if image.ndim == 2:
                canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            else:
                canvas = image.copy()
            if row.get("overlay_source") != "inference":
                for points, color in (
                    (row.get("gold_landmarks_roi_norm"), (0, 255, 0)),
                    (row.get("baseline_landmarks_roi_norm"), (0, 0, 255)),
                    (row.get("candidate_landmarks_roi_norm"), (255, 0, 0)),
                ):
                    for x, y in _points(points):
                        cv2.circle(
                            canvas,
                            (int(round(x * 255.0)), int(round(y * 255.0))),
                            2,
                            color,
                            -1,
                        )
            name = f"{category}_{accepted:02d}_{hashlib_sha(row['crop_id'])}.png"
            destination = output / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(destination), canvas):
                continue
            written.append(str(destination))
            accepted += 1
    return written


def hashlib_sha(value: Any) -> str:
    import hashlib

    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:12]


def analyze_finetune_runs(
    *,
    work_root: Path,
    baseline_id: str,
    candidate_id: str,
    labels_path: Path,
    output_dir: Path,
    overlay_limit: int = 40,
    overwrite: bool = False,
) -> Dict[str, Any]:
    if overlay_limit < 0 or overlay_limit > 40:
        raise ValueError("overlay_limit must be in [0, 40]")
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"analysis output already exists: {output_dir}")
        shutil.rmtree(output_dir)
    labels = {_identity(row): row for row in read_jsonl(labels_path)}
    if "" in labels:
        raise ValueError("evaluation labels contain an empty identity")
    base_path = work_root / "hand_landmarker_runs" / baseline_id / "eval" / "finetune" / "val" / "predictions.jsonl"
    candidate_path = work_root / "hand_landmarker_runs" / candidate_id / "eval" / "finetune" / "val" / "predictions.jsonl"
    baseline = _decorate_predictions(read_jsonl(base_path), labels)
    candidate = _decorate_predictions(read_jsonl(candidate_path), labels)
    paired_rows = _paired(baseline, candidate)
    baseline_infer_path = (
        work_root / "hand_landmarker_inference" / baseline_id / "finetune" / "predictions.jsonl"
    )
    candidate_infer_path = (
        work_root / "hand_landmarker_inference" / candidate_id / "finetune" / "predictions.jsonl"
    )
    infer_rows, infer_paired_summary = _paired_infer(baseline_infer_path, candidate_infer_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "per_roi_metrics.jsonl"
    write_jsonl(metrics_path, candidate)
    write_jsonl(output_dir / "inference_paired_metrics.jsonl", infer_rows)
    paired_report = {
        "schema_version": "finetune_paired_comparison_v1",
        "matched_records": len(paired_rows),
        "candidate_minus_baseline_mean_error_px": _mean(row.get("error_delta_px") for row in paired_rows),
        "improved_records": sum(
            row.get("error_delta_px") is not None and float(row["error_delta_px"]) < 0.0
            for row in paired_rows
        ),
        "regressed_records": sum(
            row.get("error_delta_px") is not None and float(row["error_delta_px"]) > 0.0
            for row in paired_rows
        ),
        "inference": {**infer_paired_summary, "rows": infer_rows},
        "rows": paired_rows,
    }
    write_json(output_dir / "paired_comparison.json", paired_report)
    overlays = (
        _draw_overlays(paired_rows, infer_rows, output_dir / "overlays", overlay_limit)
        if overlay_limit
        else []
    )
    summary = {
        "schema_version": "finetune_error_audit_v1",
        "status": "ok",
        "baseline_id": baseline_id,
        "candidate_id": candidate_id,
        "inputs": {
            "labels": {"path": str(labels_path), "sha256": sha256_file(labels_path)},
            "baseline_predictions": {"path": str(base_path), "sha256": sha256_file(base_path)},
            "candidate_predictions": {"path": str(candidate_path), "sha256": sha256_file(candidate_path)},
        },
        "baseline": _aggregate(baseline),
        "candidate": _aggregate(candidate),
        "paired": {
            key: value
            for key, value in paired_report.items()
            if key not in {"rows", "inference"}
        },
        "inference": {
            "baseline": _infer_summary(
                baseline_infer_path
            ),
            "candidate": _infer_summary(
                candidate_infer_path
            ),
            "paired": infer_paired_summary,
        },
        "overlays": {"count": len(overlays), "maximum": overlay_limit, "paths": overlays},
    }
    write_json(output_dir / "summary.json", summary)
    return summary
