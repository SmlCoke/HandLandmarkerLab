#!/usr/bin/env python3
"""Authenticate and evaluate the full-config-bound finetune smoke run."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hand_landmarker.config import load_config, resolve_path
from hand_landmarker.contracts import (
    effective_head_weights,
    normalize_supervision_tier_loss_weights,
)
from hand_landmarker.io_utils import read_jsonl, sha256_file, write_json


CURATION_SCHEMA = "finetune_curation_v1"
SELECTION_SCHEMA = "finetune_smoke_selection_v1"
SAMPLE_TYPES = (
    "POS_RUNTIME",
    "POS_LOW_PALM",
    "NEG_RUNTIME_CANDIDATE",
    "NEG_LOW_PALM_CANDIDATE",
)
FIXED_GATE: Dict[str, Any] = {
    "expected_records": 256,
    "maximum_mean_landmark_mae": 0.02,
    "maximum_p90_landmark_mae": 0.04,
    "maximum_max_landmark_mae": 0.10,
    "maximum_hand_flag_bce": 0.08,
    "minimum_hand_flag_accuracy": 0.98,
    "maximum_handedness_bce": 0.15,
    "minimum_handedness_accuracy": 0.95,
}
FIXED_SMOKE_TYPE_FRACTIONS = {
    "gold": {
        "POS_RUNTIME": 0.35,
        "POS_LOW_PALM": 0.15,
        "NEG_RUNTIME_CANDIDATE": 0.25,
        "NEG_LOW_PALM_CANDIDATE": 0.25,
    },
    "pseudo": {
        "POS_RUNTIME": 0.35,
        "POS_LOW_PALM": 0.15,
        "NEG_RUNTIME_CANDIDATE": 0.25,
        "NEG_LOW_PALM_CANDIDATE": 0.25,
    },
}
ALLOWED_SMOKE_DIFFS = {
    "experiment.name",
    "data.labels",
    "training.epochs",
    "training.batch_size",
    "training.optimizer.learning_rate",
    "training.checkpoint.monitor",
    "training.checkpoint.mode",
    "training.learning_rate_schedule.monitor",
    "training.learning_rate_schedule.mode",
    "training.early_stopping.monitor",
    "training.early_stopping.mode",
    "sampling.epoch_size",
    *{
        "sampling.sample_type_fractions_by_tier.{}.{}".format(tier, sample_type)
        for tier in ("gold", "pseudo")
        for sample_type in SAMPLE_TYPES
    },
    "augmentation.enabled",
    "validation.enabled",
    "outputs.run_dir",
}
_EXPECTED_INTERFACE = {
    "input_shape": [None, 1, 256, 256],
    "output_shapes": [
        [None, 1, 1, 42],
        [None, 1, 1, 1],
        [None, 1, 1, 1],
    ],
    "output_names": [
        "convld_21_2d",
        "activation_handflag",
        "activation_handedness",
    ],
    "output_semantics": ["landmarks", "hand_flag", "handedness"],
}
_MISSING = object()
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _jsonable(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, sort_keys=True, default=str))


def _read_object(path: Path, label: str) -> Dict[str, Any]:
    _require_plain_file(path, label)
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, Mapping):
        raise ValueError("{} must contain a JSON object: {}".format(label, path))
    return dict(value)


def _require_plain_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError("{} not found: {}".format(label, path))
    if path.is_symlink():
        raise ValueError("{} must not be a symlink: {}".format(label, path))
    return path.resolve(strict=True)


def _within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _relative_artifact_path(value: Any, label: str) -> PurePosixPath:
    text = str(value or "")
    relative = PurePosixPath(text)
    if (
        not text
        or "\\" in text
        or relative.is_absolute()
        or any(part in {"", ".", ".."} or ":" in part for part in relative.parts)
    ):
        raise ValueError("{} must be a normalized relative POSIX path".format(label))
    return relative


def _manifest_artifact(
    manifest: Mapping[str, Any],
    output_root: Path,
    relative_value: Any,
    label: str,
    expected_count: Optional[int] = None,
) -> Dict[str, Any]:
    relative = _relative_artifact_path(relative_value, label)
    root = output_root.resolve(strict=True)
    unresolved = root
    for part in relative.parts:
        unresolved = unresolved / part
        if unresolved.is_symlink():
            raise ValueError("{} path contains a symlink: {}".format(label, unresolved))
    path = _require_plain_file(unresolved, label)
    if not _within(path, root):
        raise ValueError("{} escapes the curation output root".format(label))
    artifacts = manifest.get("artifacts") or {}
    entry = artifacts.get(relative.as_posix()) if isinstance(artifacts, Mapping) else None
    if not isinstance(entry, Mapping):
        raise ValueError("Curation manifest does not authenticate {}".format(label))
    digest = str(entry.get("sha256") or "")
    if not _SHA256_RE.fullmatch(digest) or sha256_file(path) != digest:
        raise ValueError("{} SHA256 mismatch".format(label))
    count = entry.get("count")
    if expected_count is not None and int(count if count is not None else -1) != expected_count:
        raise ValueError(
            "{} manifest count must be {}; got {}".format(label, expected_count, count)
        )
    actual_count: Optional[int] = None
    if count is not None or expected_count is not None:
        actual_count = len(read_jsonl(path))
        if count is None or int(count) != actual_count:
            raise ValueError(
                "{} JSONL count differs from the manifest: {} != {}".format(
                    label, actual_count, count
                )
            )
    return {
        "path": str(path),
        "relative_path": relative.as_posix(),
        "sha256": digest,
        "count": actual_count,
    }


def _leaf_differences(
    full: Any,
    smoke: Any,
    prefix: str = "",
) -> List[Tuple[str, Any, Any]]:
    if isinstance(full, Mapping) and isinstance(smoke, Mapping):
        result: List[Tuple[str, Any, Any]] = []
        for key in sorted(set(full) | set(smoke), key=str):
            if not prefix and key == "_meta":
                continue
            path = "{}.{}".format(prefix, key) if prefix else str(key)
            result.extend(
                _leaf_differences(full.get(key, _MISSING), smoke.get(key, _MISSING), path)
            )
        return result
    if full is _MISSING or smoke is _MISSING or _jsonable(full) != _jsonable(smoke):
        return [(prefix, full, smoke)]
    return []


def _diff_value(value: Any) -> Any:
    return "<missing>" if value is _MISSING else _jsonable(value)


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("{} must be an object".format(label))
    return value


def _assert_runtime_contract(config: Mapping[str, Any], label: str) -> None:
    if str(config.get("task")) != "train" or str(config.get("stage")) != "finetune":
        raise ValueError("{} must be a task=train, stage=finetune config".format(label))
    model = _require_mapping(config.get("model"), "{}.model".format(label))
    expected_model = {
        "version": "v2",
        "checkpoint_stage": "finetune",
        "input_shape": [1, 256, 256],
        "input_layout": "NCHW",
        "output_order": ["landmarks", "hand_flag", "handedness"],
        "output_sizes": {"landmarks": 42, "hand_flag": 1, "handedness": 1},
    }
    for key, expected in expected_model.items():
        if _jsonable(model.get(key)) != expected:
            raise ValueError("{}.model.{} must equal {!r}".format(label, key, expected))
    depth = model.get("num_iterations")
    if (
        not isinstance(depth, (list, tuple))
        or len(depth) != 7
        or any(isinstance(value, bool) or int(value) <= 0 for value in depth)
    ):
        raise ValueError("{}.model.num_iterations must contain seven positive integers".format(label))

    data = _require_mapping(config.get("data"), "{}.data".format(label))
    expected_input = {
        "require_curation_schema": CURATION_SCHEMA,
        "require_training_stage": "finetune",
        "image_size": [256, 256],
        "channels": 1,
        "color_mode": "grayscale",
        "input_layout": "NCHW",
        "input_dtype": "float32",
        "input_offset": 0.0,
    }
    for key, expected in expected_input.items():
        if _jsonable(data.get(key)) != expected:
            raise ValueError("{}.data.{} must equal {!r}".format(label, key, expected))
    if not math.isclose(float(data.get("input_scale", math.nan)), 1.0 / 255.0, abs_tol=1e-12):
        raise ValueError("{}.data.input_scale must equal 1/255".format(label))

    targets = _require_mapping(config.get("targets"), "{}.targets".format(label))
    expected_targets = {
        "num_landmarks": 21,
        "landmark_field": "landmarks_crop_norm",
        "landmark_space": "normalized_crop_xy",
        "landmark_order": "id_0_to_20_interleaved_xy",
        "presence_field": "hand_presence.present",
        "handedness_field": "handedness.label",
        "handedness_encoding": {"Left": 0, "Right": 1, "unknown": None},
    }
    for key, expected in expected_targets.items():
        if _jsonable(targets.get(key)) != expected:
            raise ValueError("{}.targets.{} must equal {!r}".format(label, key, expected))

    losses = _require_mapping(config.get("losses"), "{}.losses".format(label))
    if losses.get("honor_record_loss_weights") is not True:
        raise ValueError("{}.losses must honor record loss weights".format(label))
    normalize_supervision_tier_loss_weights(losses.get("supervision_tier_weights"))
    for head, expected_name in (
        ("landmarks", "huber"),
        ("hand_flag", "binary_crossentropy"),
        ("handedness", "binary_crossentropy"),
    ):
        loss = _require_mapping(losses.get(head), "{}.losses.{}".format(label, head))
        if str(loss.get("name", "")).lower() != expected_name:
            raise ValueError("{}.losses.{} has the wrong loss".format(label, head))
        coefficient = float(loss.get("coefficient", math.nan))
        if not math.isfinite(coefficient) or coefficient <= 0.0:
            raise ValueError("{}.losses.{}.coefficient must be finite and positive".format(label, head))
        if head != "landmarks" and loss.get("from_logits") is not False:
            raise ValueError("{}.losses.{}.from_logits must be false".format(label, head))


def compare_smoke_and_full_configs(
    smoke_config: Mapping[str, Any],
    full_config: Mapping[str, Any],
) -> Dict[str, Any]:
    """Reject every resolved config difference not explicitly allowed by 11.4."""

    _assert_runtime_contract(full_config, "Full config")
    _assert_runtime_contract(smoke_config, "Smoke config")
    if "smoke_gate" in full_config:
        raise ValueError("smoke_gate belongs only to the smoke config")
    gate = smoke_config.get("smoke_gate")
    if not isinstance(gate, Mapping) or dict(gate) != FIXED_GATE:
        raise ValueError("Finetune smoke_gate must equal the fixed P4 thresholds")

    differences = _leaf_differences(full_config, smoke_config)
    unexpected = [
        path
        for path, _full, _smoke in differences
        if path not in ALLOWED_SMOKE_DIFFS
        and path != "smoke_gate"
        and not path.startswith("smoke_gate.")
    ]
    if unexpected:
        raise ValueError(
            "Smoke config contains non-permitted full-config differences: {}".format(
                sorted(unexpected)
            )
        )
    changed = {path for path, _full, _smoke in differences}
    required_changes = {
        "experiment.name",
        "data.labels",
        "outputs.run_dir",
        "training.checkpoint.monitor",
        "training.learning_rate_schedule.monitor",
        "training.early_stopping.monitor",
        "sampling.epoch_size",
        "augmentation.enabled",
        "validation.enabled",
    }
    missing_changes = sorted(required_changes - changed)
    if missing_changes:
        raise ValueError("Smoke config is missing required overrides: {}".format(missing_changes))

    full_training = _require_mapping(full_config.get("training"), "Full config.training")
    smoke_training = _require_mapping(smoke_config.get("training"), "Smoke config.training")
    if full_training.get("resume_checkpoint") not in (None, "") or smoke_training.get("resume_checkpoint") not in (None, ""):
        raise ValueError("Finetune smoke/full configs must not use resume_checkpoint")
    if not full_training.get("initial_checkpoint") or not smoke_training.get("initial_checkpoint"):
        raise ValueError("Finetune smoke/full configs require initial_checkpoint")
    if bool((full_config.get("validation") or {}).get("enabled")) is not True:
        raise ValueError("Full finetune config must enable validation")
    if bool((smoke_config.get("validation") or {}).get("enabled")) is not False:
        raise ValueError("Smoke config must disable validation")
    if bool((smoke_config.get("augmentation") or {}).get("enabled")) is not False:
        raise ValueError("Smoke config must disable augmentation")
    if int((smoke_config.get("sampling") or {}).get("epoch_size", -1)) != 256:
        raise ValueError("Smoke sampling.epoch_size must be exactly 256")
    smoke_learning_rate = float(
        ((smoke_training.get("optimizer") or {}).get("learning_rate", math.nan))
    )
    if not math.isclose(smoke_learning_rate, 1.0e-3, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError("Smoke optimizer learning rate must be exactly 0.001")
    smoke_fractions = (smoke_config.get("sampling") or {}).get(
        "sample_type_fractions_by_tier"
    )
    try:
        normalized_smoke_fractions = {
            str(tier): {
                str(sample_type): float(fraction)
                for sample_type, fraction in dict(fractions).items()
            }
            for tier, fractions in dict(smoke_fractions).items()
        }
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("Smoke sample-type fractions are invalid") from exc
    if normalized_smoke_fractions != FIXED_SMOKE_TYPE_FRACTIONS:
        raise ValueError("Smoke sample-type fractions must equal the fixed balanced probe")
    for component in ("checkpoint", "learning_rate_schedule", "early_stopping"):
        full_monitor = _require_mapping(full_training.get(component), "Full training.{}".format(component))
        smoke_monitor = _require_mapping(smoke_training.get(component), "Smoke training.{}".format(component))
        if full_monitor.get("monitor") != "val_landmark_mae" or full_monitor.get("mode") != "min":
            raise ValueError("Full {} must monitor val_landmark_mae/min".format(component))
        if smoke_monitor.get("monitor") != "total_loss" or smoke_monitor.get("mode") != "min":
            raise ValueError("Smoke {} must monitor total_loss/min".format(component))
    if str((smoke_training.get("learning_rate_schedule") or {}).get("name")) != "reduce_on_plateau":
        raise ValueError("Smoke LR schedule must remain reduce_on_plateau")
    full_run = resolve_path(str((full_config.get("outputs") or {}).get("run_dir")), full_config)
    smoke_run = resolve_path(str((smoke_config.get("outputs") or {}).get("run_dir")), smoke_config)
    if full_run == smoke_run:
        raise ValueError("Full and smoke run directories must be independent")

    return {
        "status": "ok",
        "allowed_paths": sorted(ALLOWED_SMOKE_DIFFS),
        "differences": [
            {"path": path, "full": _diff_value(full), "smoke": _diff_value(smoke)}
            for path, full, smoke in differences
        ],
        "model_depth": list(smoke_config["model"]["num_iterations"]),
        "model_interface": {
            "input_shape": [None, *list(smoke_config["model"]["input_shape"])],
            "output_order": list(smoke_config["model"]["output_order"]),
            "output_sizes": dict(smoke_config["model"]["output_sizes"]),
        },
        "targets": _jsonable(smoke_config["targets"]),
        "losses": _jsonable(smoke_config["losses"]),
        "input_contract": {
            key: _jsonable(smoke_config["data"].get(key))
            for key in (
                "image_size",
                "channels",
                "color_mode",
                "input_layout",
                "input_dtype",
                "input_scale",
                "input_offset",
            )
        },
    }


def _artifact_relative_to_root(path: Path, root: Path, label: str) -> str:
    try:
        return path.resolve(strict=True).relative_to(root.resolve(strict=True)).as_posix()
    except (OSError, ValueError) as exc:
        raise ValueError("{} is outside the curation output root".format(label)) from exc


def verify_curation_binding(
    smoke_config: Mapping[str, Any],
    full_config: Mapping[str, Any],
) -> Dict[str, Any]:
    """Authenticate the manifest and the three full/smoke selection artifacts."""

    smoke_data = _require_mapping(smoke_config.get("data"), "Smoke config.data")
    full_data = _require_mapping(full_config.get("data"), "Full config.data")
    smoke_manifest_path = resolve_path(str(smoke_data.get("curation_manifest") or ""), smoke_config)
    full_manifest_path = resolve_path(str(full_data.get("curation_manifest") or ""), full_config)
    if smoke_manifest_path.resolve() != full_manifest_path.resolve():
        raise ValueError("Full and smoke configs must use the same curation manifest path")
    manifest_path = _require_plain_file(smoke_manifest_path, "Finetune curation manifest")
    manifest = _read_object(manifest_path, "Finetune curation manifest")
    if str(manifest.get("schema_version")) != CURATION_SCHEMA:
        raise ValueError("Finetune curation manifest schema mismatch")
    output_value = str(manifest.get("output_dir") or "")
    if not output_value or not Path(output_value).is_absolute():
        raise ValueError("Finetune curation manifest output_dir must be absolute")
    output_root_value = Path(output_value)
    if output_root_value.is_symlink() or not output_root_value.is_dir():
        raise ValueError("Finetune curation output_dir is missing or a symlink")
    output_root = output_root_value.resolve(strict=True)
    for label, config, data in (
        ("Full", full_config, full_data),
        ("Smoke", smoke_config, smoke_data),
    ):
        data_root = resolve_path(str(data.get("data_root") or ""), config)
        if data_root.resolve(strict=True) != output_root:
            raise ValueError("{} data_root is not the authenticated curation output".format(label))

    curation_config_value = manifest.get("config_path")
    curation_config_hash = str(manifest.get("config_sha256") or "")
    if not curation_config_value or not _SHA256_RE.fullmatch(curation_config_hash):
        raise ValueError("Curation manifest must authenticate its resolved curator config")
    curation_config_path = _require_plain_file(
        Path(str(curation_config_value)), "Finetune curator config"
    )
    if sha256_file(curation_config_path) != curation_config_hash:
        raise ValueError("Finetune curator config SHA256 mismatch")

    full_labels_path = _require_plain_file(
        resolve_path(str(full_data.get("labels") or ""), full_config), "Full finetune labels"
    )
    smoke_labels_path = _require_plain_file(
        resolve_path(str(smoke_data.get("labels") or ""), smoke_config), "Smoke finetune labels"
    )
    if full_labels_path == smoke_labels_path:
        raise ValueError("Full and smoke labels must be distinct persisted artifacts")
    full_relative = _artifact_relative_to_root(full_labels_path, output_root, "Full labels")
    smoke_relative = _artifact_relative_to_root(smoke_labels_path, output_root, "Smoke labels")
    smoke_manifest = _require_mapping(manifest.get("smoke"), "Curation manifest.smoke")
    if smoke_relative != str(smoke_manifest.get("labels") or ""):
        raise ValueError("Configured smoke labels differ from manifest.smoke.labels")
    if int(smoke_manifest.get("count", -1)) != 256:
        raise ValueError("Curation manifest smoke count must be 256")
    selection_hash = str(smoke_manifest.get("selection_config_sha256") or "")
    if not _SHA256_RE.fullmatch(selection_hash):
        raise ValueError("Curation manifest lacks a valid selection_config_sha256")
    curator_config = load_config(curation_config_path)
    expected_selection_hash = hashlib.sha256(
        json.dumps(
            curator_config.get("smoke") or {},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    if selection_hash != expected_selection_hash:
        raise ValueError("Curation manifest selection_config_sha256 mismatch")

    full_artifact = _manifest_artifact(
        manifest, output_root, full_relative, "Full finetune labels"
    )
    smoke_artifact = _manifest_artifact(
        manifest, output_root, smoke_relative, "Smoke finetune labels", expected_count=256
    )
    selection_artifact = _manifest_artifact(
        manifest,
        output_root,
        smoke_manifest.get("selection"),
        "Finetune smoke selection artifact",
        expected_count=256,
    )
    return {
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "curator_config": str(curation_config_path),
        "curator_config_sha256": curation_config_hash,
        "selection_config_sha256": selection_hash,
        "output_root": str(output_root),
        "full_labels": full_artifact,
        "smoke_labels": smoke_artifact,
        "selection": selection_artifact,
    }


def _identity(row: Mapping[str, Any]) -> str:
    return str(row.get("global_crop_id") or row.get("crop_id") or "")


def _category(row: Mapping[str, Any]) -> str:
    tier = str(row.get("supervision_tier") or "").lower()
    presence = (row.get("hand_presence") or {}).get("present")
    sample_type = str(row.get("sample_type") or "")
    if not isinstance(presence, bool):
        raise ValueError("Smoke hand_presence.present must be boolean: {}".format(_identity(row)))
    if tier == "gold":
        return "gold_positive" if presence else "gold_negative"
    if tier == "pseudo" and presence:
        return "pseudo_positive"
    if tier == "pseudo" and sample_type == "NEG_RUNTIME_CANDIDATE":
        return "pseudo_neg_runtime"
    if tier == "pseudo" and sample_type == "NEG_LOW_PALM_CANDIDATE":
        return "pseudo_neg_low"
    raise ValueError("Smoke row is outside the five permitted categories: {}".format(_identity(row)))


def _without_internal_line(row: Mapping[str, Any]) -> Dict[str, Any]:
    return {key: copy.deepcopy(value) for key, value in row.items() if key != "_jsonl_line"}


def _authenticate_smoke_derivation(
    full_rows: Sequence[Mapping[str, Any]],
    smoke_rows: Sequence[Mapping[str, Any]],
) -> None:
    full_by_id: Dict[str, Dict[str, Any]] = {}
    for row in full_rows:
        identity = _identity(row)
        if not identity or identity in full_by_id:
            raise ValueError("Full labels contain a missing or duplicate identity: {}".format(identity))
        full_by_id[identity] = _without_internal_line(row)
    for row in smoke_rows:
        identity = _identity(row)
        if identity not in full_by_id:
            raise ValueError("Smoke identity is absent from full labels: {}".format(identity))
        normalized = _without_internal_line(row)
        curation = normalized.get("finetune_curation")
        if not isinstance(curation, Mapping):
            raise ValueError("Smoke row lacks finetune_curation binding: {}".format(identity))
        curation = dict(curation)
        if float(normalized.get("sampling_weight", math.nan)) != 1.0 or float(
            curation.get("smoke_sampling_weight", math.nan)
        ) != 1.0:
            raise ValueError("Smoke sampling weight must be normalized to 1: {}".format(identity))
        original = curation.pop("smoke_original_sampling_weight", _MISSING)
        curation.pop("smoke_sampling_weight", None)
        if original is _MISSING:
            raise ValueError("Smoke row lacks its original sampling weight: {}".format(identity))
        normalized["sampling_weight"] = original
        if curation or "finetune_curation" in full_by_id[identity]:
            normalized["finetune_curation"] = curation
        else:
            normalized.pop("finetune_curation", None)
        if _jsonable(normalized) != _jsonable(full_by_id[identity]):
            raise ValueError("Smoke row differs from its authenticated full-label row: {}".format(identity))


def validate_smoke_snapshot(
    full_rows: Sequence[Mapping[str, Any]],
    smoke_rows: Sequence[Mapping[str, Any]],
    selection_rows: Sequence[Mapping[str, Any]],
    expected_records: int = 256,
) -> Dict[str, Any]:
    """Validate persisted categories, identities, and all three head masks."""

    if len(smoke_rows) != expected_records or len(selection_rows) != expected_records:
        raise ValueError("Smoke labels and selection must each contain exactly 256 rows")
    identities = [_identity(row) for row in smoke_rows]
    if any(not value for value in identities) or len(set(identities)) != expected_records:
        raise ValueError("Smoke labels require 256 unique non-empty identities")
    _authenticate_smoke_derivation(full_rows, smoke_rows)

    smoke_by_id = {_identity(row): row for row in smoke_rows}
    selection_ids: List[str] = []
    for row in selection_rows:
        if str(row.get("schema_version")) != SELECTION_SCHEMA:
            raise ValueError("Smoke selection schema mismatch")
        identity = str(row.get("global_crop_id") or "")
        if not identity or identity in selection_ids:
            raise ValueError("Smoke selection contains a missing or duplicate identity")
        selected = smoke_by_id.get(identity)
        if selected is None:
            raise ValueError("Smoke selection identity is absent from smoke labels: {}".format(identity))
        if str(row.get("category") or "") != _category(selected):
            raise ValueError("Smoke selection category does not match labels: {}".format(identity))
        if str(row.get("dataset_id") or "") != str(selected.get("dataset_id") or ""):
            raise ValueError("Smoke selection dataset_id does not match labels: {}".format(identity))
        digest = str(row.get("selection_hash") or "")
        if not _SHA256_RE.fullmatch(digest):
            raise ValueError("Smoke selection_hash must be SHA256-shaped: {}".format(identity))
        selection_ids.append(identity)
    if set(selection_ids) != set(identities):
        raise ValueError("Smoke labels and selection identities differ")

    categories = Counter(_category(row) for row in smoke_rows)
    gold_negative = int(categories.get("gold_negative", 0))
    expected_categories = {
        "gold_positive": 96 - gold_negative,
        "gold_negative": gold_negative,
        "pseudo_positive": 96,
        "pseudo_neg_runtime": 32,
        "pseudo_neg_low": 32,
    }
    if gold_negative < 0 or gold_negative > 16 or dict(categories) != {
        key: value for key, value in expected_categories.items() if value
    }:
        raise ValueError(
            "Smoke category quotas are invalid: got {}, expected {}".format(
                dict(categories), expected_categories
            )
        )

    handedness_counts: Counter = Counter()
    unknown_positive = 0
    for row in smoke_rows:
        identity = _identity(row)
        present = bool((row.get("hand_presence") or {}).get("present"))
        handedness = str((row.get("handedness") or {}).get("label", "unknown")).lower()
        try:
            presence_weight, landmark_weight, handedness_weight = effective_head_weights(row)
        except (TypeError, ValueError) as exc:
            raise ValueError("Invalid smoke head weight at {}: {}".format(identity, exc)) from exc
        weights = (presence_weight, landmark_weight, handedness_weight)
        if any(not math.isfinite(value) or value < 0.0 for value in weights):
            raise ValueError("Smoke head weights must be finite and non-negative: {}".format(identity))
        if presence_weight <= 0.0:
            raise ValueError("Every smoke row must supervise hand presence: {}".format(identity))
        if present:
            if landmark_weight <= 0.0:
                raise ValueError("Every positive smoke row must supervise landmarks: {}".format(identity))
            if handedness in {"left", "right"}:
                handedness_counts[handedness] += 1
                if handedness_weight <= 0.0:
                    raise ValueError("Known handedness must have a positive mask: {}".format(identity))
            elif handedness == "unknown":
                unknown_positive += 1
                if handedness_weight != 0.0:
                    raise ValueError("Unknown handedness must have a zero mask: {}".format(identity))
            else:
                raise ValueError("Positive smoke handedness must be Left/Right/unknown: {}".format(identity))
        elif landmark_weight != 0.0 or handedness_weight != 0.0:
            raise ValueError("Negative smoke rows may supervise only hand presence: {}".format(identity))
    if handedness_counts["left"] < 1 or handedness_counts["right"] < 1:
        raise ValueError("Handedness smoke requires at least one known Left and Right")

    return {
        "record_count": expected_records,
        "categories": dict(sorted(categories.items())),
        "gold_negative_runtime_check": (
            "selected" if gold_negative else "not_applicable_redistributed_to_gold_positive"
        ),
        "known_handedness": dict(sorted(handedness_counts.items())),
        "unknown_handedness_runtime_check": "required" if unknown_positive else "not_applicable",
        "unknown_handedness_positive_count": unknown_positive,
        "mask_contract": "pass",
    }


def validate_epoch_plan_coverage(
    training_report: Mapping[str, Any],
    smoke_rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    sampler = ((training_report.get("data_report") or {}).get("sampler") or {})
    if sampler.get("mode") != "weighted_stratified" or int(sampler.get("draws_per_epoch", -1)) != 256:
        raise ValueError("Smoke training report must contain a 256-draw weighted sampler plan")
    plan = sampler.get("epoch_type_plan")
    if not isinstance(plan, Mapping):
        raise ValueError("Smoke training report lacks the epoch-level tier/type plan")
    quotas = plan.get("epoch_draw_quota_by_tier_type")
    batches = plan.get("batch_cell_quotas")
    if not isinstance(quotas, Mapping) or not isinstance(batches, list) or not batches:
        raise ValueError("Smoke epoch plan lacks quotas or batch coverage")
    planned: Dict[str, int] = {}
    for tier, type_counts in quotas.items():
        if not isinstance(type_counts, Mapping):
            raise ValueError("Smoke epoch tier quota must be an object")
        for sample_type in SAMPLE_TYPES:
            count = int(type_counts.get(sample_type, 0))
            if count < 0:
                raise ValueError("Smoke epoch quota must be non-negative")
            planned["{}:{}".format(tier, sample_type)] = count
    drawn: Counter = Counter()
    for batch in batches:
        if not isinstance(batch, Mapping):
            raise ValueError("Smoke batch_cell_quotas entries must be objects")
        for cell, value in batch.items():
            count = int(value)
            if count < 0:
                raise ValueError("Smoke batch cell count must be non-negative")
            drawn[str(cell)] += count
    for cell, quota in planned.items():
        if int(drawn.get(cell, 0)) != quota:
            raise ValueError(
                "Smoke epoch cell was not fully drawn: {} planned {}, drawn {}".format(
                    cell, quota, drawn.get(cell, 0)
                )
            )
    if sum(planned.values()) != 256 or sum(drawn.values()) != 256:
        raise ValueError("Smoke epoch plan must conserve exactly 256 draws")
    schedule_hash = str(plan.get("batch_type_schedule_sha256") or "")
    if not _SHA256_RE.fullmatch(schedule_hash):
        raise ValueError("Smoke epoch plan lacks a valid batch_type_schedule_sha256")
    available = Counter(
        "{}:{}".format(str(row.get("supervision_tier")), str(row.get("sample_type")))
        for row in smoke_rows
    )
    missing = [cell for cell, quota in planned.items() if quota > 0 and available.get(cell, 0) <= 0]
    if missing:
        raise ValueError("Positive-quota smoke cells have no persisted records: {}".format(missing))
    return {
        "status": "pass",
        "epoch_draw_quota_by_tier_type": _jsonable(quotas),
        "drawn_by_tier_type": dict(sorted(drawn.items())),
        "batch_type_schedule_sha256": schedule_hash,
    }


def _matching_hash_record(records: Any, path: Path, digest: str) -> bool:
    for record in records or []:
        if not isinstance(record, Mapping):
            continue
        try:
            same_path = Path(str(record.get("path") or "")).resolve() == path.resolve()
        except (OSError, ValueError):
            same_path = False
        if same_path and record.get("sha256") == digest and record.get("exists") is True:
            return True
    return False


def _verify_history_selection(
    history: Mapping[str, Any], selection: Mapping[str, Any]
) -> Dict[str, Any]:
    monitor = str(selection.get("monitor") or "")
    mode = str(selection.get("mode") or "")
    epochs = list(history.get("epochs") or [])
    values = list((history.get("history") or {}).get(monitor) or [])
    if not epochs or len(epochs) != len(values):
        raise ValueError("Smoke history does not cover the checkpoint monitor")
    numeric = [float(value) for value in values]
    if any(not math.isfinite(value) for value in numeric):
        raise FloatingPointError("Smoke history contains a non-finite checkpoint metric")
    expected = min(numeric) if mode == "min" else max(numeric)
    index = numeric.index(expected)
    expected_epoch = int(epochs[index])
    if int(selection.get("history_best_epoch", -1)) != expected_epoch or not math.isclose(
        float(selection.get("history_best_value", math.nan)), expected, rel_tol=1e-7, abs_tol=1e-9
    ):
        raise ValueError("Smoke checkpoint selection differs from current history")
    if int(selection.get("completed_epoch", -1)) != expected_epoch or not math.isclose(
        float(selection.get("value", math.nan)), expected, rel_tol=1e-7, abs_tol=1e-9
    ):
        raise ValueError("Smoke best checkpoint state differs from current history")
    return {"best_epoch": expected_epoch, "best_value": expected}


def verify_smoke_run_provenance(
    smoke_config: Mapping[str, Any],
    curation: Mapping[str, Any],
) -> Dict[str, Any]:
    """Authenticate config/git/labels/initial state/history/best checkpoint."""

    outputs = _require_mapping(smoke_config.get("outputs"), "Smoke config.outputs")
    run_dir = resolve_path(str(outputs.get("run_dir") or ""), smoke_config)
    report_path = resolve_path(
        str(outputs.get("report_path", run_dir / "training_report.json")), smoke_config
    )
    report = _read_object(report_path, "Smoke training report")
    if (
        report.get("status") != "complete"
        or report.get("stage") != "finetune"
        or report.get("model_version") != "v2"
    ):
        raise ValueError("Smoke training report must be complete finetune")
    metadata_path = run_dir / "experiment_metadata.json"
    reported_metadata = Path(str(report.get("metadata_path") or ""))
    if reported_metadata.resolve() != metadata_path.resolve():
        raise ValueError("Smoke training report points to unexpected experiment metadata")
    metadata = _read_object(metadata_path, "Smoke experiment metadata")
    if (
        metadata.get("status") != "complete"
        or metadata.get("stage") != "finetune"
        or metadata.get("model_version") != "v2"
    ):
        raise ValueError("Smoke experiment metadata must be complete finetune")
    if metadata.get("resolved_config") != _jsonable(smoke_config):
        raise ValueError("Smoke run resolved_config differs from the current smoke config")

    config_path = _require_plain_file(
        Path(str((smoke_config.get("_meta") or {}).get("config_path") or "")),
        "Smoke config file",
    )
    if (
        Path(str(metadata.get("config_path") or "")).resolve() != config_path
        or metadata.get("config_sha256") != sha256_file(config_path)
    ):
        raise ValueError("Smoke metadata config path/SHA differs from the current config")
    saved_git = metadata.get("git")
    if not isinstance(saved_git, Mapping):
        raise ValueError("Smoke metadata lacks git provenance")
    from hand_landmarker.training import _git_metadata

    repo_root = Path(str((smoke_config.get("_meta") or {}).get("repo_root") or "."))
    current_git = _git_metadata(repo_root)
    if any(saved_git.get(key) != current_git.get(key) for key in ("commit", "dirty", "status_short")):
        raise ValueError("Smoke run git provenance differs from the current worktree")

    labels_path = Path(str((curation.get("smoke_labels") or {}).get("path") or ""))
    labels_hash = str((curation.get("smoke_labels") or {}).get("sha256") or "")
    if not _matching_hash_record((report.get("label_hashes") or {}).get("train"), labels_path, labels_hash):
        raise ValueError("Smoke labels path/SHA differs from the training report")
    if metadata.get("label_hashes") != report.get("label_hashes"):
        raise ValueError("Smoke report and metadata label hashes differ")
    if report.get("data_report") != metadata.get("data_report"):
        raise ValueError("Smoke report and metadata data_report differ")
    data_manifest = (report.get("data_report") or {}).get("curation_manifest") or {}
    if (
        Path(str(data_manifest.get("path") or "")).resolve()
        != Path(str(curation.get("manifest") or "")).resolve()
        or data_manifest.get("sha256") != curation.get("manifest_sha256")
    ):
        raise ValueError("Smoke training report used a different curation manifest")

    training = _require_mapping(smoke_config.get("training"), "Smoke config.training")
    initial_path = _require_plain_file(
        resolve_path(str(training.get("initial_checkpoint") or ""), smoke_config),
        "Finetune initial checkpoint",
    )
    initial_hash = sha256_file(initial_path)
    starting_state = metadata.get("starting_state") or {}
    if (
        starting_state.get("mode") != "initial_weights"
        or Path(str(starting_state.get("path") or "")).resolve() != initial_path
        or starting_state.get("sha256") != initial_hash
        or int(starting_state.get("initial_epoch", -1)) != 0
    ):
        raise ValueError("Smoke starting_state does not authenticate the initial checkpoint")

    report_artifacts = report.get("artifacts") or {}
    metadata_artifacts = metadata.get("artifacts") or {}
    checkpoints_dir = resolve_path(
        str(outputs.get("checkpoints_dir", run_dir / "checkpoints")), smoke_config
    )
    configured_paths = {
        "history": resolve_path(str(outputs.get("history_path", run_dir / "history.json")), smoke_config),
        "best_checkpoint": resolve_path(
            str(outputs.get("best_checkpoint", checkpoints_dir / "best.weights.h5")),
            smoke_config,
        ),
    }
    authenticated: Dict[str, Dict[str, Any]] = {}
    for name, configured in configured_paths.items():
        artifact = report_artifacts.get(name)
        metadata_artifact = metadata_artifacts.get(name)
        if not isinstance(artifact, Mapping) or not isinstance(metadata_artifact, Mapping):
            raise ValueError("Smoke report lacks {} provenance".format(name))
        path = _require_plain_file(Path(str(artifact.get("path") or "")), "Smoke {}".format(name))
        digest = sha256_file(path)
        if path != configured.resolve() or artifact.get("sha256") != digest:
            raise ValueError("Smoke {} path/SHA mismatch".format(name))
        if metadata_artifact.get("path") != artifact.get("path") or metadata_artifact.get("sha256") != digest:
            raise ValueError("Smoke metadata/report {} provenance differs".format(name))
        authenticated[name] = {"path": str(path), "sha256": digest}

    selection = report.get("checkpoint_selection") or {}
    if (
        selection.get("monitor") != "total_loss"
        or selection.get("mode") != "min"
        or selection.get("verified_against_current_history") is not True
        or metadata.get("checkpoint_selection") != selection
    ):
        raise ValueError("Smoke best checkpoint must be scratch-history verified with total_loss/min")
    history = _read_object(Path(authenticated["history"]["path"]), "Smoke history")
    history_selection = _verify_history_selection(history, selection)
    if list(report.get("completed_epochs") or []) != list(history.get("epochs") or []):
        raise ValueError("Smoke report completed_epochs differs from history")
    if list(metadata.get("completed_epochs") or []) != list(history.get("epochs") or []):
        raise ValueError("Smoke metadata completed_epochs differs from history")
    interface = metadata.get("model_interface") or {}
    if any(interface.get(key) != expected for key, expected in _EXPECTED_INTERFACE.items()):
        raise ValueError("Smoke metadata model interface is incompatible with HLML v2")
    if int(interface.get("parameter_count", 0)) <= 0:
        raise ValueError("Smoke metadata lacks a valid model parameter count")

    return {
        "training_report": str(report_path.resolve()),
        "experiment_metadata": str(metadata_path.resolve()),
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "git": _jsonable(saved_git),
        "labels": str(labels_path),
        "labels_sha256": labels_hash,
        "curation_manifest": str(curation["manifest"]),
        "curation_manifest_sha256": str(curation["manifest_sha256"]),
        "initial_checkpoint": str(initial_path),
        "initial_checkpoint_sha256": initial_hash,
        "history": authenticated["history"],
        "best_checkpoint": authenticated["best_checkpoint"],
        "checkpoint_selection": {**_jsonable(selection), **history_selection},
        "training_report_object": report,
    }


def _same_initial_checkpoint(
    smoke_config: Mapping[str, Any], full_config: Mapping[str, Any]
) -> Dict[str, Any]:
    smoke_path = _require_plain_file(
        resolve_path(str(smoke_config["training"]["initial_checkpoint"]), smoke_config),
        "Smoke initial checkpoint",
    )
    full_path = _require_plain_file(
        resolve_path(str(full_config["training"]["initial_checkpoint"]), full_config),
        "Full initial checkpoint",
    )
    smoke_hash = sha256_file(smoke_path)
    full_hash = sha256_file(full_path)
    if smoke_path != full_path or smoke_hash != full_hash:
        raise ValueError("Full and smoke configs do not share one initial checkpoint path/SHA")
    return {"path": str(smoke_path), "sha256": smoke_hash}


def _prediction_heads(outputs: Any) -> Tuple[Any, Any, Any]:
    if isinstance(outputs, Mapping):
        aliases = (
            ("landmarks", "convld_21_2d"),
            ("hand_flag", "activation_handflag"),
            ("handedness", "activation_handedness"),
        )
        values = []
        for names in aliases:
            matches = [outputs[name] for name in names if name in outputs]
            if len(matches) != 1:
                raise ValueError("Smoke model output mapping has missing/ambiguous heads")
            values.append(matches[0])
        return values[0], values[1], values[2]
    if not isinstance(outputs, (list, tuple)) or len(outputs) != 3:
        raise ValueError("Smoke model must produce exactly three ordered heads")
    return outputs[0], outputs[1], outputs[2]


def _metric_head_weights(sample_weights: Any) -> Tuple[Any, Any, Any]:
    """Return landmark, hand-flag and handedness weights in semantic order."""

    if not isinstance(sample_weights, Mapping):
        raise ValueError("Smoke sequence sample weights must be a semantic mapping")
    missing = [
        name for name in ("landmarks", "hand_flag", "handedness")
        if name not in sample_weights
    ]
    if missing:
        raise ValueError("Smoke sequence sample weights are missing {}".format(missing))
    return (
        sample_weights["landmarks"],
        sample_weights["hand_flag"],
        sample_weights["handedness"],
    )


def _binary_metrics(predictions: Any, targets: Any, label: str, np: Any) -> Tuple[float, float]:
    predicted = np.asarray(predictions, dtype=np.float64).reshape(-1)
    expected = np.asarray(targets, dtype=np.float64).reshape(-1)
    if predicted.shape != expected.shape or predicted.size == 0:
        raise ValueError("{} prediction/target coverage mismatch".format(label))
    if np.any(~np.isfinite(predicted)) or np.any(~np.isfinite(expected)):
        raise FloatingPointError("{} contains NaN/Inf".format(label))
    if np.any(predicted < 0.0) or np.any(predicted > 1.0):
        raise ValueError("{} predictions are outside [0,1]".format(label))
    clipped = np.clip(predicted, 1e-7, 1.0 - 1e-7)
    bce = float(np.mean(-(expected * np.log(clipped) + (1.0 - expected) * np.log(1.0 - clipped))))
    accuracy = float(np.mean((predicted >= 0.5) == (expected >= 0.5)))
    if not math.isfinite(bce) or not math.isfinite(accuracy):
        raise FloatingPointError("{} metrics are non-finite".format(label))
    return bce, accuracy


def full_smoke_metrics(
    config: Mapping[str, Any], best_path: Path, expected_records: int
) -> Dict[str, Any]:
    # TensorFlow/OpenCV remain inside the executed gate; --help and contract
    # tests therefore work in a lightweight local environment.
    from hand_landmarker.inspect import DatasetContractError, audit_canonical_dataset
    from hand_landmarker.finetune_curation import verify_finetune_curation_manifest
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
    verify_finetune_curation_manifest(config, data_config, error_type=DatasetContractError)
    records, _ = audit_canonical_dataset(
        config,
        dataset=data_config,
        expected_stage="finetune",
        check_images=True,
        hash_images=True,
        raise_on_error=True,
    )
    if len(records) != expected_records:
        raise ValueError("Smoke inference must cover exactly 256 persisted records")
    sequence = CanonicalSequence(
        records,
        dataset_config=data_config,
        targets_config=dict(config.get("targets") or {}),
        batch_size=int((config.get("training") or {}).get("batch_size", 32)),
        training=False,
        stage="finetune",
        seed=int((config.get("experiment") or {}).get("seed", 0)),
        augmentation_config={"enabled": False},
        training_config={},
        sampling_config={},
        output_order=(config.get("model") or {}).get("output_order"),
    )
    model_config = dict(config.get("model") or {})
    backbone = build_model(
        str(model_config.get("version")),
        num_iterations=model_config.get("num_iterations"),
    )
    interface = assert_model_interface(backbone)
    backbone.load_weights(str(best_path))

    sample_landmark_mae: List[float] = []
    coordinate_errors: List[float] = []
    hand_predictions: List[float] = []
    hand_targets: List[float] = []
    handedness_predictions: List[float] = []
    handedness_targets: List[float] = []
    covered = 0
    for batch_index in range(len(sequence)):
        inputs, targets, sample_weights = sequence[batch_index]
        outputs = _prediction_heads(backbone.predict_on_batch(inputs))
        landmark_output, hand_output, handedness_output = outputs
        batch_size = int(inputs.shape[0])
        predicted_landmarks = np.asarray(landmark_output, dtype=np.float64).reshape((batch_size, 42))
        predicted_hand = np.asarray(hand_output, dtype=np.float64).reshape((batch_size, 1))
        predicted_handedness = np.asarray(handedness_output, dtype=np.float64).reshape((batch_size, 1))
        expected_landmarks = np.asarray(targets[0], dtype=np.float64).reshape((batch_size, 42))
        expected_hand = np.asarray(targets[1], dtype=np.float64).reshape((batch_size, 1))
        expected_handedness = np.asarray(targets[2], dtype=np.float64).reshape((batch_size, 1))
        masks = [
            np.asarray(value, dtype=np.float64).reshape(batch_size)
            for value in _metric_head_weights(sample_weights)
        ]
        if any(np.any(~np.isfinite(value)) or np.any(value < 0.0) for value in masks):
            raise FloatingPointError("Smoke inference encountered an invalid head mask")
        if np.any(masks[1] <= 0.0):
            raise ValueError("Every smoke row must be included in hand-flag metrics")
        if any(
            np.any(~np.isfinite(value))
            for value in (
                predicted_landmarks,
                predicted_hand,
                predicted_handedness,
                expected_landmarks,
                expected_hand,
                expected_handedness,
            )
        ):
            raise FloatingPointError("Smoke inference produced NaN/Inf")
        positive = expected_hand.reshape(-1) >= 0.5
        if np.any((masks[0] > 0.0) != positive):
            raise ValueError("Runtime landmark masks do not match positive rows")
        known = masks[2] > 0.0
        if np.any(known & ~positive):
            raise ValueError("Runtime handedness mask includes a negative row")
        landmark_indices = np.flatnonzero(masks[0] > 0.0)
        if landmark_indices.size:
            absolute = np.abs(
                predicted_landmarks[landmark_indices] - expected_landmarks[landmark_indices]
            )
            coordinate_errors.extend(absolute.reshape(-1).tolist())
            sample_landmark_mae.extend(np.mean(absolute, axis=1).tolist())
        hand_predictions.extend(predicted_hand.reshape(-1).tolist())
        hand_targets.extend(expected_hand.reshape(-1).tolist())
        handedness_indices = np.flatnonzero(known)
        handedness_predictions.extend(
            predicted_handedness.reshape(-1)[handedness_indices].tolist()
        )
        handedness_targets.extend(expected_handedness.reshape(-1)[handedness_indices].tolist())
        covered += batch_size

    if covered != expected_records or len(hand_predictions) != expected_records:
        raise RuntimeError("Sequential smoke inference did not cover every persisted row exactly once")
    if not coordinate_errors or not handedness_predictions:
        raise ValueError("Smoke inference has no eligible landmark or handedness rows")
    if any(not math.isfinite(value) for value in coordinate_errors + sample_landmark_mae):
        raise FloatingPointError("Smoke landmark metrics contain NaN/Inf")
    hand_bce, hand_accuracy = _binary_metrics(hand_predictions, hand_targets, "hand flag", np)
    handedness_bce, handedness_accuracy = _binary_metrics(
        handedness_predictions, handedness_targets, "handedness", np
    )
    return {
        "evaluation_mode": "sequential_all_records_no_augmentation",
        "record_count": expected_records,
        "unique_evaluations": covered,
        "landmark_record_count": len(sample_landmark_mae),
        "hand_flag_record_count": len(hand_predictions),
        "handedness_record_count": len(handedness_predictions),
        "mean_landmark_mae": float(np.mean(coordinate_errors)),
        "p90_landmark_mae": float(np.percentile(sample_landmark_mae, 90.0)),
        "max_landmark_mae": float(np.max(sample_landmark_mae)),
        "hand_flag_bce": hand_bce,
        "hand_flag_accuracy": hand_accuracy,
        "handedness_bce": handedness_bce,
        "handedness_accuracy": handedness_accuracy,
        "model_interface": interface,
    }


def check_smoke_run(
    smoke_config: Mapping[str, Any], full_config: Mapping[str, Any]
) -> Dict[str, Any]:
    config_binding = compare_smoke_and_full_configs(smoke_config, full_config)
    initial = _same_initial_checkpoint(smoke_config, full_config)
    curation = verify_curation_binding(smoke_config, full_config)
    full_rows = read_jsonl(curation["full_labels"]["path"])
    smoke_rows = read_jsonl(curation["smoke_labels"]["path"])
    selection_rows = read_jsonl(curation["selection"]["path"])
    snapshot = validate_smoke_snapshot(full_rows, smoke_rows, selection_rows)
    provenance = verify_smoke_run_provenance(smoke_config, curation)
    if provenance["initial_checkpoint"] != initial["path"] or provenance["initial_checkpoint_sha256"] != initial["sha256"]:
        raise ValueError("Smoke run used a different initial checkpoint than the full config")
    coverage = validate_epoch_plan_coverage(
        provenance["training_report_object"], smoke_rows
    )
    best_path = Path(provenance["best_checkpoint"]["path"])
    metrics = full_smoke_metrics(smoke_config, best_path, 256)
    checks = {
        "mean_landmark_mae": metrics["mean_landmark_mae"] <= FIXED_GATE["maximum_mean_landmark_mae"],
        "p90_landmark_mae": metrics["p90_landmark_mae"] <= FIXED_GATE["maximum_p90_landmark_mae"],
        "max_landmark_mae": metrics["max_landmark_mae"] <= FIXED_GATE["maximum_max_landmark_mae"],
        "hand_flag_bce": metrics["hand_flag_bce"] <= FIXED_GATE["maximum_hand_flag_bce"],
        "hand_flag_accuracy": metrics["hand_flag_accuracy"] >= FIXED_GATE["minimum_hand_flag_accuracy"],
        "handedness_bce": metrics["handedness_bce"] <= FIXED_GATE["maximum_handedness_bce"],
        "handedness_accuracy": metrics["handedness_accuracy"] >= FIXED_GATE["minimum_handedness_accuracy"],
    }
    provenance.pop("training_report_object", None)
    return {
        "status": "pass" if all(checks.values()) else "fail",
        "full_config": str((full_config.get("_meta") or {}).get("config_path")),
        "smoke_config": str((smoke_config.get("_meta") or {}).get("config_path")),
        "thresholds": dict(FIXED_GATE),
        "checks": checks,
        "metrics": metrics,
        "config_binding": config_binding,
        "initial_checkpoint": initial,
        "curation": curation,
        "snapshot": snapshot,
        "epoch_plan_coverage": coverage,
        "provenance": provenance,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-config", required=True, help="Finetune smoke YAML")
    parser.add_argument("--full-config", required=True, help="Full finetune YAML")
    args = parser.parse_args()
    smoke_config = load_config(args.smoke_config)
    full_config = load_config(args.full_config)
    report = check_smoke_run(smoke_config, full_config)
    run_dir = resolve_path(str(smoke_config["outputs"]["run_dir"]), smoke_config)
    write_json(run_dir / "smoke_gate_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    if report["status"] != "pass":
        raise SystemExit(
            "Finetune smoke failed its fixed full-config-bound gate; do not launch full finetune"
        )


if __name__ == "__main__":
    main()
