"""Immutable Val-winner descriptor and locked-Test gates for HLML 4.0."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping

from .io_utils import sha256_file, write_json


WINNER_SCHEMA = "hlml_val_winner_v1"


def _read_json(path: Path) -> Dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("JSON root must be an object: {}".format(path))
    return value


def freeze_winner(
    train_root: Path,
    release_id: str,
    val_metrics_path: Path,
    checkpoint_path: Path,
    checkpoint_stage: str,
    snapshot_id: str,
    model_version: str = "v2",
) -> Dict[str, Any]:
    """Freeze the only checkpoint descriptor that locked Test may consume."""

    val_metrics_path = Path(val_metrics_path).resolve()
    checkpoint_path = Path(checkpoint_path).resolve()
    if not val_metrics_path.is_file():
        raise FileNotFoundError("Val metrics do not exist: {}".format(val_metrics_path))
    if not checkpoint_path.is_file():
        raise FileNotFoundError("winner checkpoint does not exist: {}".format(checkpoint_path))
    metrics = _read_json(val_metrics_path)
    if str(metrics.get("split")) != "val":
        raise ValueError("winner must be frozen from fixed-ROI Val metrics")
    if str(metrics.get("scope")) != "hand_landmarker_on_provided_hand_roi":
        raise ValueError("winner Val scope must be provided fixed Hand ROIs")
    stage = str(checkpoint_stage)
    if stage not in {"pretrain", "finetune"}:
        raise ValueError("checkpoint_stage must be pretrain or finetune")
    release_root = Path(train_root).resolve() / "releases" / str(release_id)
    descriptor_path = release_root / "winner.json"
    if descriptor_path.exists() or release_root.exists() and any(release_root.iterdir()):
        raise FileExistsError("release winner is immutable: {}".format(release_root))
    report = {
        "schema_version": WINNER_SCHEMA,
        "release_id": str(release_id),
        "selection_split": "val",
        "evaluation_scope": "fixed_hand_roi_only",
        "model_version": str(model_version),
        "checkpoint_stage": stage,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "snapshot_id": str(snapshot_id),
        "presence_threshold": float(metrics.get("presence_threshold", 0.5)),
        "val_metrics_path": str(val_metrics_path),
        "val_metrics_sha256": sha256_file(val_metrics_path),
        "test_policy": {
            "single_winner_only": True,
            "overwrite": False,
            "may_feed_training": False,
            "may_feed_mining": False,
            "may_select_threshold": False,
        },
    }
    write_json(descriptor_path, report)
    return report


def load_winner(train_root: Path, release_id: str) -> Dict[str, Any]:
    path = Path(train_root).resolve() / "releases" / str(release_id) / "winner.json"
    if not path.is_file():
        raise FileNotFoundError("frozen winner descriptor does not exist: {}".format(path))
    report = _read_json(path)
    if report.get("schema_version") != WINNER_SCHEMA:
        raise ValueError("winner descriptor schema mismatch")
    checkpoint = Path(str(report.get("checkpoint_path", ""))).resolve()
    if not checkpoint.is_file() or sha256_file(checkpoint) != report.get("checkpoint_sha256"):
        raise ValueError("frozen winner checkpoint is missing or changed")
    return report


def locked_test_config(
    config: Mapping[str, Any], train_root: Path, release_id: str
) -> Dict[str, Any]:
    """Apply the winner to a Test config and refuse every tuning/overwrite path."""

    import copy

    winner = load_winner(train_root, release_id)
    value = copy.deepcopy(dict(config))
    if str(value.get("split")) != "test":
        raise ValueError("locked Test requires split: test")
    evaluation = value.setdefault("evaluation", {})
    if bool(evaluation.get("tune_thresholds", False)):
        raise ValueError("locked Test cannot tune thresholds")
    if str(evaluation.get("mode")) != "roi":
        raise ValueError("locked Test evaluates fixed Hand ROIs only")
    output = value.setdefault("output", {})
    if bool(output.get("overwrite", False)):
        raise ValueError("locked Test output.overwrite must remain false")
    output_dir = Path(str(output.get("dir", ""))).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("locked Test result already exists: {}".format(output_dir))
    value.setdefault("hand", {})["model_path"] = winner["checkpoint_path"]
    value.setdefault("model", {})["checkpoint_stage"] = winner["checkpoint_stage"]
    evaluation["hand_flag_threshold"] = winner["presence_threshold"]
    value["locked_winner"] = winner
    return value
