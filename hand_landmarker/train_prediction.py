"""Batch inference over authenticated canonical training ROIs.

This runner deliberately bypasses Palm: every input row already names the
canonical 256x256 Hand ROI used by the teacher and by training.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .finetune_selection import clean_row, identity_value
from .io_utils import read_image, sha256_file, write_json, write_jsonl
from .runtime import KerasHandPredictor


PREDICTION_SCHEMA = "train_prediction_v1"


def _point_rows(points: Sequence[Tuple[float, float]]) -> List[Dict[str, float]]:
    if len(points) != 21:
        raise ValueError("Hand predictor must return exactly 21 landmarks")
    return [
        {"id": index, "x": float(point[0]), "y": float(point[1])}
        for index, point in enumerate(points)
    ]


def predict_rows(
    rows: Sequence[Mapping[str, Any]],
    predictor: Any,
    batch_size: int = 64,
) -> List[Dict[str, Any]]:
    """Predict all rows exactly once, preserving deterministic input order."""

    if isinstance(batch_size, bool) or int(batch_size) <= 0:
        raise ValueError("batch_size must be a positive integer")
    batch_size = int(batch_size)
    identities = [identity_value(row) for row in rows]
    if any(not value for value in identities):
        raise ValueError("Every prediction row requires a canonical crop identity")
    if len(set(identities)) != len(identities):
        raise ValueError("Prediction input contains duplicate crop identities")
    outputs: List[Dict[str, Any]] = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        images = []
        paths = []
        hashes = []
        for row in batch:
            path = Path(str(row.get("crop_path") or ""))
            if not path.is_absolute() or not path.is_file():
                raise FileNotFoundError("Canonical crop_path must be an existing absolute file: {}".format(path))
            if path.is_symlink():
                raise ValueError("Canonical prediction crops may not be symlinks: {}".format(path))
            digest = sha256_file(path)
            expected = str(row.get("image_sha256") or row.get("crop_image_sha256") or "")
            if expected and expected != digest:
                raise ValueError("Canonical crop SHA mismatch: {}".format(path))
            image = read_image(path)
            if image is None:
                raise ValueError("Could not decode canonical crop: {}".format(path))
            images.append(image)
            paths.append(path.resolve())
            hashes.append(digest)
        predictions = predictor.predict(images, batch_size=batch_size)
        if len(predictions) != len(batch):
            raise RuntimeError(
                "Predictor returned {} rows for input batch {}".format(len(predictions), len(batch))
            )
        for row, path, digest, prediction in zip(batch, paths, hashes, predictions):
            outputs.append(
                {
                    "schema_version": PREDICTION_SCHEMA,
                    "global_crop_id": str(row.get("global_crop_id") or identity_value(row)),
                    "parent_global_crop_id": row.get("parent_global_crop_id"),
                    "dataset_id": str(row.get("dataset_id") or ""),
                    "source_crop_id": str(row.get("source_crop_id") or row.get("crop_id") or ""),
                    "crop_path": str(path),
                    "image_sha256": digest,
                    "student_landmarks_crop_norm": _point_rows(prediction.landmarks_norm),
                    "student_hand_flag": float(prediction.hand_flag_score),
                    "student_handedness": float(prediction.handedness_score),
                    "landmark_raw_max_abs": float(prediction.landmark_raw_max_abs),
                    "normalized_out_of_range_coordinate_count": int(
                        prediction.normalized_out_of_range_coordinate_count
                    ),
                }
            )
    return outputs


def predict_training_labels(
    labels_path: Path,
    checkpoint_path: Path,
    model_config: Mapping[str, Any],
    output_path: Path,
    report_path: Path,
    batch_size: int = 64,
    predictor: Optional[Any] = None,
) -> Dict[str, Any]:
    """Run Keras inference and publish predictions plus authenticated provenance."""

    from .io_utils import read_jsonl

    labels_path = Path(labels_path)
    checkpoint_path = Path(checkpoint_path)
    output_path = Path(output_path)
    report_path = Path(report_path)
    if not labels_path.is_file():
        raise FileNotFoundError("Training labels not found: {}".format(labels_path))
    if not checkpoint_path.is_file():
        raise FileNotFoundError("Geometry checkpoint not found: {}".format(checkpoint_path))
    if output_path.exists() or report_path.exists():
        raise FileExistsError("Prediction outputs are immutable; choose a new finetune ID")
    rows = [clean_row(row) for row in read_jsonl(labels_path)]
    if not rows:
        raise ValueError("Training prediction input is empty")
    actual_predictor = predictor or KerasHandPredictor(
        weights_path=str(checkpoint_path),
        model_version=str(model_config.get("version", "v3-pro")),
        num_iterations=model_config.get("num_iterations", [2, 2, 3, 4, 4, 6, 6]),
    )
    predictions = predict_rows(rows, actual_predictor, batch_size=batch_size)
    labels_sha256 = sha256_file(labels_path)
    checkpoint_sha256 = sha256_file(checkpoint_path)
    for row in predictions:
        row["source_labels_sha256"] = labels_sha256
        row["checkpoint_sha256"] = checkpoint_sha256
        row["checkpoint_stage"] = "geometry"
        row["model_version"] = str(model_config.get("version", "v3-pro"))
    write_jsonl(output_path, predictions)
    report = {
        "status": "ok",
        "schema_version": PREDICTION_SCHEMA,
        "input": {
            "labels": str(labels_path.resolve()),
            "labels_sha256": labels_sha256,
            "record_count": len(rows),
        },
        "checkpoint": {
            "path": str(checkpoint_path.resolve()),
            "sha256": checkpoint_sha256,
            "stage": "geometry",
        },
        "model": clean_row(model_config),
        "batch_size": int(batch_size),
        "output": {
            "path": str(output_path.resolve()),
            "sha256": sha256_file(output_path),
            "record_count": len(predictions),
        },
    }
    write_json(report_path, report)
    return report
