"""Round-based Train-only hard-source mining for fixed HLMF Hand ROIs."""

from __future__ import annotations

import bisect
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from .io_utils import read_image, read_jsonl, write_json, write_jsonl
from .metrics import percentile, safe_div


MINING_SCHEMA = "hlml_train_hard_mining_v2"
LANDMARK_DIFFICULTY_WEIGHT = 0.80
PRESENCE_DIFFICULTY_WEIGHT = 0.10
HANDEDNESS_DIFFICULTY_WEIGHT = 0.10


def _points(row: Mapping[str, Any], key: str) -> List[Tuple[float, float]]:
    raw = list(row.get(key) or [])
    if len(raw) != 21:
        raise ValueError("{} must contain 21 landmarks for {}".format(key, row.get("roi_id")))
    by_id = {int(point.get("id", offset)): point for offset, point in enumerate(raw)}
    if set(by_id) != set(range(21)):
        raise ValueError("{} landmark IDs must be 0..20".format(key))
    return [(float(by_id[index]["x"]), float(by_id[index]["y"])) for index in range(21)]


def _brightness(row: Mapping[str, Any]) -> float:
    image = read_image(Path(str(row.get("crop_path") or "")))
    if image is None:
        raise ValueError("cannot decode mining ROI: {}".format(row.get("crop_path")))
    return float(image.mean())


def score_row(label: Mapping[str, Any], prediction: Mapping[str, Any]) -> Dict[str, Any]:
    """Score one student prediction against its current HLMF teacher label."""

    if str(label.get("split")) != "train":
        raise ValueError("hard mining accepts Train rows only")
    if not bool((label.get("hand_presence") or {}).get("present", False)):
        raise ValueError("hard-positive mining accepts positive labels only")
    expected = _points(label, "landmarks_crop_norm")
    predicted = _points(prediction, "student_landmarks_crop_norm")
    errors_norm = [
        math.hypot(estimate[0] - target[0], estimate[1] - target[1])
        for target, estimate in zip(expected, predicted)
    ]
    xs = [point[0] for point in predicted]
    ys = [point[1] for point in predicted]
    span = math.hypot(max(xs) - min(xs), max(ys) - min(ys))
    hand_flag = float(prediction.get("student_hand_flag", 1.0))
    handedness_score = float(prediction.get("student_handedness"))
    if not math.isfinite(hand_flag) or not 0.0 <= hand_flag <= 1.0:
        raise ValueError("student_hand_flag must be finite and within [0,1]")
    if not math.isfinite(handedness_score) or not 0.0 <= handedness_score <= 1.0:
        raise ValueError("student_handedness must be finite and within [0,1]")
    target_handedness = str((label.get("handedness") or {}).get("label", "")).title()
    handedness_eligible = target_handedness in {"Left", "Right"}
    target_handedness_probability = None
    if handedness_eligible:
        target_handedness_probability = (
            handedness_score if target_handedness == "Right" else 1.0 - handedness_score
        )
    mean_norm = sum(errors_norm) / len(errors_norm)
    result = {
        "roi_id": str(label.get("roi_id") or label.get("global_crop_id") or label.get("crop_id")),
        "capture_source_id": str(label.get("capture_source_id")),
        "split": "train",
        "mean_error_px": mean_norm * 255.0,
        "median_error_px": float(percentile([value * 255.0 for value in errors_norm], 0.5)),
        "p90_error_px": float(percentile([value * 255.0 for value in errors_norm], 0.9)),
        "pck_0_05": safe_div(sum(value <= 0.05 for value in errors_norm), 21),
        "pck_0_10": safe_div(sum(value <= 0.10 for value in errors_norm), 21),
        "collapse": bool(span < 0.05 or hand_flag < 0.5),
        "student_span_norm": span,
        "student_hand_flag": hand_flag,
        "presence_error": 1.0 - hand_flag,
        "presence_incorrect": bool(hand_flag < 0.5),
        "student_handedness": handedness_score,
        "teacher_handedness": target_handedness,
        "handedness_eligible": handedness_eligible,
        "handedness_error": (
            1.0 - target_handedness_probability if handedness_eligible else 0.0
        ),
        "handedness_incorrect": bool(
            handedness_eligible and target_handedness_probability < 0.5
        ),
        "brightness_mean": _brightness(label),
        "distance": label.get("distance"),
        "lighting": label.get("lighting"),
        "pose_span_norm": math.hypot(
            max(point[0] for point in expected) - min(point[0] for point in expected),
            max(point[1] for point in expected) - min(point[1] for point in expected),
        ),
    }
    return result


def aggregate_sources(
    labels: Sequence[Mapping[str, Any]], predictions: Sequence[Mapping[str, Any]]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Return ranked source reports and HLMF review-request rows."""

    if any(str(row.get("split")) != "train" for row in labels):
        raise ValueError("hard mining must never read Val/Test rows")
    prediction_by_id = {
        str(row.get("roi_id") or row.get("global_crop_id") or row.get("crop_id")): row
        for row in predictions
    }
    if len(prediction_by_id) != len(predictions):
        raise ValueError("student predictions contain duplicate ROI IDs")
    scored: List[Tuple[Dict[str, Any], Mapping[str, Any]]] = []
    for label in labels:
        roi_id = str(label.get("roi_id") or label.get("global_crop_id") or label.get("crop_id"))
        prediction = prediction_by_id.pop(roi_id, None)
        if prediction is None:
            raise ValueError("student prediction is missing ROI {}".format(roi_id))
        scored.append((score_row(label, prediction), label))
    if prediction_by_id:
        raise ValueError("predictions contain ROIs outside the Train snapshot")
    by_source: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    labels_by_id = {
        str(row.get("roi_id") or row.get("global_crop_id") or row.get("crop_id")): row
        for row in labels
    }
    for score, _ in scored:
        by_source[score["capture_source_id"]].append(score)
    ordered_errors = sorted(float(score["mean_error_px"]) for score, _ in scored)
    denominator = max(1, len(ordered_errors) - 1)
    for score, _ in scored:
        error = float(score["mean_error_px"])
        lower = bisect.bisect_left(ordered_errors, error)
        equal = bisect.bisect_right(ordered_errors, error) - lower
        landmark_rank = (lower + 0.5 * max(0, equal - 1)) / float(denominator)
        score["landmark_error_rank"] = min(1.0, max(0.0, landmark_rank))
        score["difficulty_score"] = (
            LANDMARK_DIFFICULTY_WEIGHT * score["landmark_error_rank"]
            + PRESENCE_DIFFICULTY_WEIGHT * float(score["presence_error"])
            + HANDEDNESS_DIFFICULTY_WEIGHT * float(score["handedness_error"])
        )
    reports = []
    for capture_id, rows in sorted(by_source.items()):
        errors = [float(row["mean_error_px"]) for row in rows]
        brightness = [float(row["brightness_mean"]) for row in rows]
        pose = [float(row["pose_span_norm"]) for row in rows]
        difficulty = [float(row["difficulty_score"]) for row in rows]
        handedness_rows = [row for row in rows if bool(row["handedness_eligible"])]
        reports.append(
            {
                "capture_source_id": capture_id,
                "sample_count": len(rows),
                "mean_error_px": safe_div(sum(errors), len(errors)),
                "median_error_px": percentile(errors, 0.50),
                "p90_error_px": percentile(errors, 0.90),
                "pck_0_05": safe_div(sum(float(row["pck_0_05"]) for row in rows), len(rows)),
                "pck_0_10": safe_div(sum(float(row["pck_0_10"]) for row in rows), len(rows)),
                "collapse_rate": safe_div(sum(bool(row["collapse"]) for row in rows), len(rows)),
                "mean_difficulty_score": safe_div(sum(difficulty), len(difficulty)),
                "p90_difficulty_score": percentile(difficulty, 0.90),
                "presence_incorrect_rate": safe_div(
                    sum(bool(row["presence_incorrect"]) for row in rows), len(rows)
                ),
                "handedness_incorrect_rate": safe_div(
                    sum(bool(row["handedness_incorrect"]) for row in handedness_rows),
                    len(handedness_rows),
                ),
                "handedness_eligible_count": len(handedness_rows),
                "brightness": {
                    "mean": safe_div(sum(brightness), len(brightness)),
                    "median": percentile(brightness, 0.50),
                },
                "pose_span_norm": {
                    "mean": safe_div(sum(pose), len(pose)),
                    "median": percentile(pose, 0.50),
                },
                "distance": rows[0].get("distance"),
                "lighting": rows[0].get("lighting"),
            }
        )
    reports.sort(
        key=lambda row: (
            float(row["p90_difficulty_score"] or 0.0),
            float(row["p90_error_px"] or 0.0),
        ),
        reverse=True,
    )
    ranks = {str(row["capture_source_id"]): index + 1 for index, row in enumerate(reports)}
    request = []
    for score, _ in sorted(
        scored,
        key=lambda item: (
            -float(item[0]["difficulty_score"]),
            -float(item[0]["mean_error_px"]),
            str(item[0]["roi_id"]),
        ),
    ):
        label = dict(labels_by_id[score["roi_id"]])
        label.pop("_jsonl_line", None)
        label.pop("_resolved_crop_path", None)
        # HLMF consumes dataset-root-relative zero-copy paths.
        label["crop_path"] = str(label.get("warehouse_crop_relpath") or label.get("crop_relpath"))
        label["crop_relpath"] = label["crop_path"]
        label["mining"] = {**score, "source_rank": ranks[score["capture_source_id"]]}
        request.append(label)
    return reports, request


def predict_rows(
    labels: Sequence[Mapping[str, Any]], predictor: Any, batch_size: int = 64
) -> List[Dict[str, Any]]:
    """Run the student on fixed training ROIs without image SHA-256 checks."""

    output: List[Dict[str, Any]] = []
    for start in range(0, len(labels), int(batch_size)):
        batch = labels[start : start + int(batch_size)]
        images = []
        for row in batch:
            if str(row.get("split")) != "train":
                raise ValueError("hard mining predictor accepts Train only")
            image = read_image(Path(str(row.get("crop_path") or "")))
            if image is None:
                raise ValueError("cannot decode mining ROI: {}".format(row.get("crop_path")))
            images.append(image)
        predictions = predictor.predict(images, batch_size=int(batch_size))
        if len(predictions) != len(batch):
            raise RuntimeError("predictor returned the wrong batch length")
        for row, prediction in zip(batch, predictions):
            output.append(
                {
                    "schema_version": MINING_SCHEMA,
                    "roi_id": str(row.get("roi_id") or row.get("global_crop_id") or row.get("crop_id")),
                    "student_landmarks_crop_norm": [
                        {"id": index, "x": float(point[0]), "y": float(point[1])}
                        for index, point in enumerate(prediction.landmarks_norm)
                    ],
                    "student_hand_flag": float(prediction.hand_flag_score),
                    "student_handedness": float(prediction.handedness_score),
                }
            )
    return output


def mine_hard_sources(
    labels_path: Path,
    output_dir: Path,
    snapshot_id: str,
    round_id: str,
    max_rois: int,
    predictions_path: Path | None = None,
    predictor: Any | None = None,
    batch_size: int = 64,
) -> Dict[str, Any]:
    snapshot_id = str(snapshot_id).strip()
    round_id = str(round_id).strip()
    if not snapshot_id or not round_id:
        raise ValueError("snapshot_id and round_id must be non-empty")
    if int(max_rois) <= 0:
        raise ValueError("max_rois must be positive")
    all_labels = read_jsonl(labels_path)
    if any(str(row.get("split")) != "train" for row in all_labels):
        raise ValueError("hard mining must never read Val/Test rows")
    labels = [
        row
        for row in all_labels
        if bool((row.get("hand_presence") or {}).get("present", False))
    ]
    if not labels:
        raise ValueError("hard mining snapshot contains no positive Train ROIs")
    positive_ids = {
        str(row.get("roi_id") or row.get("global_crop_id") or row.get("crop_id"))
        for row in labels
    }
    if predictions_path is not None:
        supplied = read_jsonl(predictions_path)
        predictions = [
            row
            for row in supplied
            if str(row.get("roi_id") or row.get("global_crop_id") or row.get("crop_id"))
            in positive_ids
        ]
    elif predictor is not None:
        predictions = predict_rows(labels, predictor, batch_size=batch_size)
    else:
        raise ValueError("predictions_path or predictor is required")
    reports, request = aggregate_sources(labels, predictions)
    output_dir = Path(output_dir)
    ledger_path = output_dir / "selection_ledger.json"
    if ledger_path.exists():
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        if str(ledger.get("schema_version")) != MINING_SCHEMA:
            raise ValueError("unsupported hard-mining selection ledger schema")
        if str(ledger.get("snapshot_id")) != snapshot_id:
            raise ValueError("hard-mining ledger belongs to another snapshot")
    else:
        ledger = {
            "schema_version": MINING_SCHEMA,
            "snapshot_id": snapshot_id,
            "dedupe_scope": "one_complete_geometry_multitask_multi_finetune_flow",
            "rounds": [],
            "selected_roi_ids": [],
        }
    if any(str(item.get("round_id")) == round_id for item in ledger.get("rounds") or []):
        raise ValueError("hard-mining round_id has already been used: {}".format(round_id))
    selected_before = {str(value) for value in ledger.get("selected_roi_ids") or []}
    eligible = [row for row in request if str(row.get("roi_id")) not in selected_before]
    selected = eligible[: int(max_rois)]
    if not selected:
        raise ValueError("no unselected hard-positive ROIs remain for this snapshot")
    round_dir = output_dir / "rounds" / round_id
    paths = [
        round_dir / "source_ranking.json",
        round_dir / "hlmf_review_request.jsonl",
        round_dir / "student_predictions.jsonl",
    ]
    if any(path.exists() for path in paths):
        raise FileExistsError("hard-mining outputs are immutable: {}".format(paths))
    round_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(paths[2], predictions)
    write_jsonl(paths[1], selected)
    report = {
        "schema_version": MINING_SCHEMA,
        "snapshot_id": snapshot_id,
        "round_id": round_id,
        "max_rois": int(max_rois),
        "scope": "train_fixed_hand_roi_only",
        "test_read": False,
        "source_count": len(reports),
        "positive_candidate_count": len(request),
        "already_selected_count": len(request) - len(eligible),
        "eligible_count": len(eligible),
        "selected_count": len(selected),
        "sources": reports,
        "difficulty_weights": {
            "landmark_error_rank": LANDMARK_DIFFICULTY_WEIGHT,
            "presence_error": PRESENCE_DIFFICULTY_WEIGHT,
            "handedness_error": HANDEDNESS_DIFFICULTY_WEIGHT,
        },
        "hlmf_request": paths[1].name,
        "dedupe_ledger": str(ledger_path),
    }
    write_json(paths[0], report)
    selected_ids = [str(row["roi_id"]) for row in selected]
    ledger["rounds"].append(
        {
            "round_id": round_id,
            "max_rois": int(max_rois),
            "selected_count": len(selected_ids),
            "request": str(paths[1]),
        }
    )
    ledger["selected_roi_ids"] = sorted(selected_before.union(selected_ids))
    write_json(ledger_path, ledger)
    return report
