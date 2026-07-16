"""Build the single authenticated pretrain-replay source for finetuning."""

from __future__ import annotations

import os
import shutil
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .finetune_selection import (
    NEGATIVE_SAMPLE_TYPES,
    POSITIVE_SAMPLE_TYPES,
    _bounded_quota,
    _dataset_quotas,
    _diverse_take,
    clean_row,
    identity_value,
)
from .io_utils import sha256_file, write_json, write_jsonl


SOURCE_SCHEMA = "finetune_source_v1"


def _within(path: Path, root: Path) -> bool:
    try:
        path.resolve(strict=True).relative_to(root.resolve(strict=True))
        return True
    except (OSError, ValueError):
        return False


def _is_confirmed_negative(row: Mapping[str, Any]) -> bool:
    if bool((row.get("hand_presence") or {}).get("present", False)):
        return False
    curation = row.get("pretrain_curation") or {}
    review = curation.get("review") or {}
    return (
        str(curation.get("action")) == "INCLUDE_CONFIRMED_NEGATIVE"
        and str(review.get("decision")) == "CONFIRMED_NEGATIVE"
        and all(str(review.get(key) or "").strip() for key in ("reviewer", "reviewed_at", "review_method", "review_image_sha256"))
    )


def select_replay_rows(
    canonical_rows: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Keep all reviewed negatives, then fill the remaining budget with positives."""

    if not bool(config.get("enabled", True)):
        return [], {"status": "disabled", "actual_selected": 0}
    max_records = int(config.get("max_records", 10000))
    if max_records <= 0:
        raise ValueError("pretrain_replay.max_records must be positive when enabled")
    if config.get("include_all_confirmed_negatives") is not True:
        raise ValueError("pretrain_replay.include_all_confirmed_negatives must be true")
    identities = [identity_value(row) for row in canonical_rows]
    if any(not value for value in identities) or len(set(identities)) != len(identities):
        raise ValueError("Replay input identities must be present and unique")
    negatives: List[Mapping[str, Any]] = []
    positives: List[Mapping[str, Any]] = []
    for row in canonical_rows:
        present = bool((row.get("hand_presence") or {}).get("present", False))
        sample_type = str(row.get("sample_type") or "")
        if present:
            if sample_type not in POSITIVE_SAMPLE_TYPES:
                raise ValueError("Replay positive has invalid sample_type: {}".format(identity_value(row)))
            positives.append(row)
        else:
            if sample_type not in NEGATIVE_SAMPLE_TYPES:
                raise ValueError("Replay negative has invalid sample_type: {}".format(identity_value(row)))
            if not _is_confirmed_negative(row):
                raise ValueError("Canonical replay contains an unreviewed negative: {}".format(identity_value(row)))
            negatives.append(row)
    if len(negatives) > max_records:
        raise ValueError(
            "Confirmed negatives ({}) exceed replay max_records ({})".format(len(negatives), max_records)
        )
    salt = str(config.get("salt", "finetune_replay_v1"))
    selected_negative = sorted((clean_row(row) for row in negatives), key=identity_value)
    positive_budget = max(0, max_records - len(selected_negative))
    positive_budget = min(positive_budget, len(positives))
    fractions = {
        key: float(value)
        for key, value in dict(config.get("positive_fractions") or {}).items()
    }
    capacities = Counter(str(row["sample_type"]) for row in positives)
    cell_quotas, cell_report = _bounded_quota(positive_budget, fractions, capacities)
    selected_positive: List[Dict[str, Any]] = []
    dataset_reports: Dict[str, Any] = {}
    for sample_type in POSITIVE_SAMPLE_TYPES:
        cell = [row for row in positives if str(row["sample_type"]) == sample_type]
        quota = int(cell_quotas.get(sample_type, 0))
        quotas, allocation = _dataset_quotas(
            cell, quota, None, config.get("dataset_weights")
        ) if cell else ({}, {})
        dataset_reports[sample_type] = allocation
        for dataset_id, dataset_quota in sorted(quotas.items()):
            rows = [row for row in cell if str(row.get("dataset_id")) == dataset_id]
            selected_positive.extend(_diverse_take(rows, dataset_quota, salt + ":" + sample_type))
    selected = selected_negative + selected_positive
    selected.sort(key=identity_value)
    output: List[Dict[str, Any]] = []
    for source in selected:
        row = clean_row(source)
        parent_global = str(row.get("parent_global_crop_id") or row.get("global_crop_id") or row.get("crop_id"))
        row["parent_global_crop_id"] = parent_global
        row.setdefault("parent_dataset_id", row.get("dataset_id"))
        row.setdefault("parent_source_crop_id", row.get("source_crop_id") or row.get("crop_id"))
        row["training_stage"] = "finetune"
        row["supervision_tier"] = "pseudo"
        output.append(row)
    report = {
        "status": "ok",
        "input_count": len(canonical_rows),
        "eligible_positive": len(positives),
        "confirmed_negative": len(negatives),
        "configured_budget": max_records,
        "actual_selected": len(output),
        "selected_positive": len(selected_positive),
        "selected_confirmed_negative": len(selected_negative),
        "selected_by_sample_type": dict(sorted(Counter(str(row.get("sample_type")) for row in output).items())),
        "cell_allocation": cell_report,
        "dataset_allocation": dataset_reports,
        "salt": salt,
    }
    return output, report


def build_replay_source(
    canonical_rows: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    output_dir: Path,
    parent_curation_manifest: Path,
    crop_root: Path,
    parent_pretrain_id: str,
    producer_version: str,
) -> Dict[str, Any]:
    """Atomically publish ``sources/replay/pretrain_replay``."""

    output_dir = Path(output_dir)
    parent_curation_manifest = Path(parent_curation_manifest)
    crop_root = Path(crop_root)
    if output_dir.exists():
        raise FileExistsError("Replay source already exists: {}".format(output_dir))
    if not parent_curation_manifest.is_file():
        raise FileNotFoundError("Parent curation manifest not found: {}".format(parent_curation_manifest))
    if not crop_root.is_dir() or not crop_root.is_absolute() or crop_root.is_symlink():
        raise ValueError("Replay crop root must be an existing absolute non-symlink directory")
    crop_root = crop_root.resolve(strict=True)
    selected, selection_report = select_replay_rows(canonical_rows, config)
    if not selected:
        raise ValueError("Enabled replay selection produced no rows")
    for row in selected:
        path = Path(str(row.get("crop_path") or ""))
        try:
            relative = path.relative_to(crop_root)
        except ValueError as exc:
            raise ValueError("Replay crop is outside the configured read-only root: {}".format(path)) from exc
        current = crop_root
        for part in relative.parts:
            current = current / part
            if current.is_symlink():
                raise ValueError("Replay crop traverses a symlink: {}".format(current))
        if not _within(path, crop_root):
            raise ValueError("Replay crop is outside the configured read-only root: {}".format(path))
        if path.is_symlink():
            raise ValueError("Replay crop may not be a symlink: {}".format(path))
        actual = sha256_file(path)
        expected = str(row.get("image_sha256") or row.get("crop_image_sha256") or "")
        if expected and expected != actual:
            raise ValueError("Replay crop SHA mismatch: {}".format(path))
        row["crop_path"] = str(path.resolve())
        row["image_sha256"] = actual

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=output_dir.name + ".tmp.", dir=str(output_dir.parent)))
    try:
        labels_path = temporary / "05_labels" / "canonical_source.jsonl"
        copied_parent = temporary / "qc" / "parent_curation_manifest.json"
        report_path = temporary / "qc" / "replay_report.json"
        write_jsonl(labels_path, selected)
        copied_parent.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(str(parent_curation_manifest), str(copied_parent))
        report = dict(selection_report)
        report.update(
            {
                "schema_version": SOURCE_SCHEMA,
                "parent_pretrain_id": str(parent_pretrain_id),
                "parent_curation_manifest": {
                    "path": str(parent_curation_manifest.resolve()),
                    "sha256": sha256_file(parent_curation_manifest),
                },
                "crop_root": str(crop_root.resolve()),
                "canonical_labels": "05_labels/canonical_source.jsonl",
                "canonical_labels_sha256": sha256_file(labels_path),
            }
        )
        write_json(report_path, report)
        descriptor = {
            "schema_version": SOURCE_SCHEMA,
            "source_id": "pretrain_replay",
            "dataset_id": "pretrain_replay_{}".format(parent_pretrain_id.replace("-", "_")),
            "source_kind": "pretrain_replay",
            "source_mode": "canonical_replay_index",
            "producer": "hlml_finetune_replay",
            "producer_version": str(producer_version),
            "created_at": datetime.now(timezone.utc).astimezone().isoformat(),
            "parent_pretrain_id": str(parent_pretrain_id),
            "enabled_stages": ["finetune"],
            "supervision_tier": "pseudo",
            "handedness_policy": "optional_per_row",
            "artifacts": {
                "canonical_labels": {
                    "path": "05_labels/canonical_source.jsonl",
                    "sha256": sha256_file(labels_path),
                    "count": len(selected),
                },
                "parent_curation_manifest": {
                    "path": "qc/parent_curation_manifest.json",
                    "sha256": sha256_file(copied_parent),
                },
                "qc_report": {
                    "path": "qc/replay_report.json",
                    "sha256": sha256_file(report_path),
                },
            },
            "external_crop_roots": [
                {
                    "root": str(crop_root.resolve()),
                    "read_only": True,
                    "row_image_sha256_required": True,
                }
            ],
            "counts": {
                "canonical_labels": len(selected),
                "confirmed_negative": int(selection_report["selected_confirmed_negative"]),
            },
        }
        descriptor_path = temporary / "finetune_source.json"
        write_json(descriptor_path, descriptor)
        os.replace(str(temporary), str(output_dir))
        temporary = None  # type: ignore[assignment]
        return {
            "status": "ok",
            "source_dir": str(output_dir.resolve()),
            "descriptor": str((output_dir / "finetune_source.json").resolve()),
            "descriptor_sha256": sha256_file(output_dir / "finetune_source.json"),
            "counts": descriptor["counts"],
        }
    finally:
        if temporary is not None and temporary.exists():
            shutil.rmtree(str(temporary), ignore_errors=True)
