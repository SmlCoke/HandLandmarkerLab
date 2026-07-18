#!/usr/bin/env python3
"""Fail unless the best checkpoint memorizes every persisted smoke ROI."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hand_landmarker.config import load_config, resolve_path
from hand_landmarker.io_utils import sha256_file, write_json


def _read_object(path: Path, label: str) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError("{} not found: {}".format(label, path))
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, Mapping):
        raise ValueError("{} must contain a JSON object: {}".format(label, path))
    return dict(value)


def _jsonable(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, sort_keys=True, default=str))


def _matching_hash_record(records: Any, path: Path, digest: str) -> bool:
    for record in records or []:
        if not isinstance(record, Mapping):
            continue
        try:
            same_path = Path(str(record.get("path"))).resolve() == path.resolve()
        except (OSError, ValueError):
            same_path = False
        if same_path and record.get("sha256") == digest and record.get("exists") is True:
            return True
    return False


def _semantic_batch_value(container: Any, semantic: str, index: int, label: str) -> Any:
    """Read a model field from either the semantic mapping or legacy sequence form."""

    if isinstance(container, Mapping):
        if semantic not in container:
            raise KeyError(
                "{} do not contain {!r}; available keys: {}".format(
                    label, semantic, sorted(str(key) for key in container)
                )
            )
        return container[semantic]
    if isinstance(container, (list, tuple)):
        if index >= len(container):
            raise IndexError(
                "{} do not contain index {} for {!r}; length={}".format(
                    label, index, semantic, len(container)
                )
            )
        return container[index]
    raise TypeError(
        "{} must be a semantic mapping or output sequence; got {}".format(
            label, type(container).__name__
        )
    )


def _verify_run_provenance(
    config: Mapping[str, Any], run_dir: Path
) -> Dict[str, Any]:
    report_path = run_dir / "training_report.json"
    report = _read_object(report_path, "Smoke training report")
    if report.get("status") != "complete":
        raise ValueError("Smoke training report is not complete: {}".format(report_path))

    metadata_path = Path(str(report.get("metadata_path") or ""))
    metadata = _read_object(metadata_path, "Smoke experiment metadata")
    if metadata.get("status") != "complete":
        raise ValueError("Smoke experiment metadata is not complete: {}".format(metadata_path))
    if metadata.get("resolved_config") != _jsonable(config):
        raise ValueError(
            "Smoke run resolved_config differs from the current config; use a new run ID and retrain"
        )
    saved_git = metadata.get("git")
    if isinstance(saved_git, Mapping):
        from hand_landmarker.training import _git_metadata

        repo_root = Path(str((config.get("_meta") or {}).get("repo_root") or "."))
        current_git = _git_metadata(repo_root)
        if any(
            saved_git.get(key) != current_git.get(key)
            for key in ("commit", "dirty", "status_short")
        ):
            raise ValueError(
                "Smoke run git provenance differs from the current worktree; retrain the smoke run"
            )

    labels_path = resolve_path(str(config["data"]["labels"]), config)
    labels_hash = sha256_file(labels_path)
    if not _matching_hash_record(
        (report.get("label_hashes") or {}).get("train"), labels_path, labels_hash
    ):
        raise ValueError("Smoke labels path/hash does not match the completed training report")

    artifacts = report.get("artifacts") or {}
    history_artifact = artifacts.get("history") or {}
    best_artifact = artifacts.get("best_checkpoint") or {}
    history_path = Path(str(history_artifact.get("path") or ""))
    best_path = Path(str(best_artifact.get("path") or ""))
    for name, path, artifact in (
        ("history", history_path, history_artifact),
        ("best checkpoint", best_path, best_artifact),
    ):
        if not path.is_file():
            raise FileNotFoundError("Smoke {} artifact not found: {}".format(name, path))
        actual = sha256_file(path)
        if artifact.get("sha256") != actual:
            raise ValueError("Smoke {} artifact hash mismatch: {}".format(name, path))

    selection = report.get("checkpoint_selection") or {}
    if (
        selection.get("monitor") != "landmark_mae"
        or selection.get("mode") != "min"
        or selection.get("verified_against_current_history") is not True
    ):
        raise ValueError(
            "Smoke best checkpoint must be scratch-history verified with landmark_mae/min; "
            "got {}/{} verified={}".format(
                selection.get("monitor"),
                selection.get("mode"),
                selection.get("verified_against_current_history"),
            )
        )
    return {
        "training_report": str(report_path),
        "experiment_metadata": str(metadata_path),
        "labels": str(labels_path),
        "labels_sha256": labels_hash,
        "history": str(history_path),
        "best_checkpoint": str(best_path),
        "best_checkpoint_sha256": best_artifact["sha256"],
    }


def _full_smoke_metrics(
    config: Mapping[str, Any], best_path: Path, expected_records: int
) -> Dict[str, Any]:
    # These imports stay inside the executed gate so --help and local contract
    # tests remain independent of TensorFlow.
    from hand_landmarker.inspect import DatasetContractError, audit_canonical_dataset
    from hand_landmarker.pretrain_curation import verify_curation_manifest
    from hand_landmarker.training import (
        _configure_tensorflow,
        _prepare_environment,
        assert_model_interface,
    )

    _prepare_environment(config)
    import numpy as np
    import tensorflow as tf

    from hand_landmarker.data import CanonicalSequence
    from models.hand_landmarker.registry import build_model

    _configure_tensorflow(tf, config)
    tf.keras.mixed_precision.set_global_policy("float32")
    data_config = dict(config.get("data") or {})
    verify_curation_manifest(
        config, data_config, error_type=DatasetContractError
    )
    records, _ = audit_canonical_dataset(
        config,
        dataset=data_config,
        expected_stage="pretrain",
        check_images=True,
        hash_images=True,
        raise_on_error=True,
    )
    if len(records) != int(expected_records):
        raise ValueError(
            "Smoke snapshot must contain exactly {} records; got {}".format(
                expected_records, len(records)
            )
        )
    sequence = CanonicalSequence(
        records,
        dataset_config=data_config,
        targets_config=dict(config.get("targets") or {}),
        batch_size=int((config.get("training") or {}).get("batch_size", 32)),
        training=False,
        stage="pretrain",
        seed=int((config.get("experiment") or {}).get("seed", 0)),
        augmentation_config={"enabled": False},
        training_config=dict(config.get("training") or {}),
        sampling_config={},
        output_order=(config.get("model") or {}).get(
            "output_order", ["landmarks", "hand_flag", "handedness"]
        ),
    )
    model_config = dict(config.get("model") or {})
    backbone = build_model(
        str(model_config.get("version", "v2")),
        num_iterations=model_config.get("num_iterations", 8),
    )
    interface = assert_model_interface(backbone)
    backbone.load_weights(str(best_path))

    sample_maes = []
    coordinate_errors = []
    for batch_index in range(len(sequence)):
        inputs, targets, sample_weights = sequence[batch_index]
        landmark_weights = np.asarray(
            _semantic_batch_value(sample_weights, "landmarks", 0, "Smoke sample weights"),
            dtype=np.float64,
        ).reshape(-1)
        if np.any(~np.isfinite(landmark_weights)) or np.any(landmark_weights <= 0.0):
            raise ValueError("Every smoke ROI must have a positive finite landmark weight")
        outputs = backbone.predict_on_batch(inputs)
        if isinstance(outputs, Mapping):
            if "convld_21_2d" in outputs:
                outputs = outputs["convld_21_2d"]
            elif "landmarks" in outputs:
                outputs = outputs["landmarks"]
            else:
                raise ValueError("Smoke model outputs do not contain the landmark head")
        elif isinstance(outputs, (list, tuple)):
            outputs = outputs[0]
        predicted = np.asarray(outputs, dtype=np.float64).reshape((-1, 42))
        expected = np.asarray(
            _semantic_batch_value(targets, "landmarks", 0, "Smoke targets"),
            dtype=np.float64,
        ).reshape((-1, 42))
        if predicted.shape != expected.shape:
            raise ValueError(
                "Smoke prediction/target shape mismatch: {} != {}".format(
                    predicted.shape, expected.shape
                )
            )
        absolute = np.abs(predicted - expected)
        if np.any(~np.isfinite(absolute)):
            raise FloatingPointError("Smoke inference produced a non-finite landmark error")
        coordinate_errors.extend(absolute.reshape(-1).tolist())
        sample_maes.extend(np.mean(absolute, axis=1).tolist())

    if len(sample_maes) != len(records):
        raise RuntimeError(
            "Sequential smoke evaluation covered {} of {} records".format(
                len(sample_maes), len(records)
            )
        )
    mean_mae = float(np.mean(coordinate_errors))
    p90_mae = float(np.percentile(sample_maes, 90.0))
    max_mae = float(np.max(sample_maes))
    return {
        "evaluation_mode": "sequential_all_records_no_augmentation",
        "record_count": len(records),
        "unique_evaluations": len(records),
        "coordinate_count": len(coordinate_errors),
        "mean_landmark_mae": mean_mae,
        "p90_sample_landmark_mae": p90_mae,
        "max_sample_landmark_mae": max_mae,
        "mean_coordinate_pixel_error": mean_mae * 255.0,
        "p90_sample_coordinate_pixel_error": p90_mae * 255.0,
        "max_sample_coordinate_pixel_error": max_mae * 255.0,
        "model_interface": interface,
    }


def check_smoke_run(
    config: Mapping[str, Any],
    maximum_mean_mae: Optional[float] = None,
    maximum_p90_mae: Optional[float] = None,
    maximum_max_mae: Optional[float] = None,
) -> Dict[str, Any]:
    run_dir = resolve_path(config["outputs"]["run_dir"], config)
    gate = dict(config.get("smoke_gate") or {})
    thresholds = {
        "maximum_mean_mae": float(
            gate.get("maximum_mean_mae", 0.01)
            if maximum_mean_mae is None
            else maximum_mean_mae
        ),
        "maximum_p90_mae": float(
            gate.get("maximum_p90_mae", 0.02)
            if maximum_p90_mae is None
            else maximum_p90_mae
        ),
        "maximum_max_mae": float(
            gate.get("maximum_max_mae", 0.05)
            if maximum_max_mae is None
            else maximum_max_mae
        ),
    }
    if any(not math.isfinite(value) or value < 0.0 for value in thresholds.values()):
        raise ValueError("Smoke MAE thresholds must be finite and non-negative")
    provenance = _verify_run_provenance(config, run_dir)
    metrics = _full_smoke_metrics(
        config,
        Path(provenance["best_checkpoint"]),
        int(gate.get("expected_records", 128)),
    )
    checks = {
        "mean_mae": metrics["mean_landmark_mae"] <= thresholds["maximum_mean_mae"],
        "p90_mae": metrics["p90_sample_landmark_mae"] <= thresholds["maximum_p90_mae"],
        "max_mae": metrics["max_sample_landmark_mae"] <= thresholds["maximum_max_mae"],
    }
    return {
        "status": "pass" if all(checks.values()) else "fail",
        "run_dir": str(run_dir),
        "thresholds": thresholds,
        "checks": checks,
        "metrics": metrics,
        "provenance": provenance,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Smoke training YAML file")
    parser.add_argument(
        "--maximum-mae",
        "--maximum-mean-mae",
        dest="maximum_mean_mae",
        type=float,
        default=None,
    )
    parser.add_argument("--maximum-p90-mae", type=float, default=None)
    parser.add_argument("--maximum-max-mae", type=float, default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    report = check_smoke_run(
        config,
        maximum_mean_mae=args.maximum_mean_mae,
        maximum_p90_mae=args.maximum_p90_mae,
        maximum_max_mae=args.maximum_max_mae,
    )
    run_dir = resolve_path(config["outputs"]["run_dir"], config)
    write_json(run_dir / "smoke_gate_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    if report["status"] != "pass":
        raise SystemExit(
            "Positive-only smoke overfit gate failed on the full persisted subset; "
            "do not launch full pretrain"
        )


if __name__ == "__main__":
    main()
