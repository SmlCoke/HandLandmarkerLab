#!/usr/bin/env python3
"""Select b/c Gold-review requests and publish the one d replay source."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Mapping

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hand_landmarker.config import load_config, resolve_path
from hand_landmarker.finetune_replay import build_replay_source
from hand_landmarker.finetune_selection import (
    clean_row,
    gold_repository_occupied,
    identity_tokens,
    identity_value,
    select_negative_removed,
    select_teacher_student,
)
from hand_landmarker.io_utils import read_jsonl, sha256_file, write_json, write_jsonl
from hand_landmarker.train_prediction import predict_training_labels


def _git_version(repo_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(repo_root), stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown-working-tree"


def _validate_registry_report(registry_path: Path) -> Dict[str, Any]:
    report_path = registry_path.with_name("pretrain_source_registry_report.json")
    if report_path.is_symlink() or not report_path.is_file():
        raise ValueError("HLMF source registry sibling report is missing or a symlink")
    with report_path.open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    if not isinstance(report, Mapping):
        raise ValueError("HLMF source registry report root must be an object")
    if str(report.get("schema_version")) != "pretrain_source_registry_v1" or str(report.get("status")) != "ok":
        raise ValueError("HLMF source registry report schema/status mismatch")
    artifact = report.get("registry") or {}
    rows = read_jsonl(registry_path)
    if (
        Path(str(artifact.get("path") or "")).resolve() != registry_path.resolve()
        or str(artifact.get("sha256") or "") != sha256_file(registry_path)
        or int(artifact.get("count", -1)) != len(rows)
        or int(report.get("rows", -1)) != len(rows)
    ):
        raise ValueError("HLMF source registry report binding mismatch")
    return {
        "path": str(report_path.resolve()),
        "sha256": sha256_file(report_path),
        "count": len(rows),
    }


def _restore_rows(
    rows: List[Mapping[str, Any]],
    registry_rows: List[Mapping[str, Any]],
    hash_cache: Dict[Path, str],
) -> List[Dict[str, Any]]:
    """Join canonical rows to the authenticated HLMF source registry."""

    registry = {identity_value(row): clean_row(row) for row in registry_rows}
    if "" in registry or len(registry) != len(registry_rows):
        raise ValueError("Source registry identities must be present and unique")
    restored: List[Dict[str, Any]] = []
    for source in rows:
        identity = identity_value(source)
        entry = registry.get(identity)
        if entry is None:
            raise ValueError("Canonical row is absent from HLMF source registry: {}".format(identity))
        row = clean_row(source)
        if str(entry.get("schema_version")) != "pretrain_source_registry_v1":
            raise ValueError("Unsupported HLMF source registry schema")
        for path_key, sha_key in (
            ("parent_manifest_path", "parent_manifest_sha256"),
            ("parent_draft_path", "parent_draft_sha256"),
            ("parent_crop_path", "image_sha256"),
        ):
            path = Path(str(entry.get(path_key) or ""))
            expected = str(entry.get(sha_key) or "").lower()
            if not path.is_absolute() or not path.is_file() or not expected:
                raise ValueError("HLMF registry lacks authenticated {}: {}".format(path_key, identity))
            symlink = next(
                (candidate for candidate in (path, *path.parents) if candidate.is_symlink()),
                None,
            )
            if symlink is not None:
                raise ValueError("HLMF registry path traverses a symlink: {}".format(symlink))
            resolved = path.resolve(strict=True)
            actual = hash_cache.get(resolved)
            if actual is None:
                actual = sha256_file(resolved)
                hash_cache[resolved] = actual
            if actual != expected:
                raise ValueError("HLMF source provenance SHA mismatch: {}".format(path))
        for key in ("dataset_id", "source_crop_id", "global_crop_id"):
            if str(entry.get(key) or "") != str(row.get(key) or ""):
                raise ValueError("Source registry {} mismatch for {}".format(key, identity))
        row["parent_manifest_path"] = str(Path(entry["parent_manifest_path"]).resolve())
        row["parent_manifest_sha256"] = str(entry["parent_manifest_sha256"])
        row["parent_draft_path"] = str(Path(entry["parent_draft_path"]).resolve())
        row["parent_draft_sha256"] = str(entry["parent_draft_sha256"])
        row["crop_path"] = str(Path(entry["parent_crop_path"]).resolve())
        row["image_sha256"] = str(entry["image_sha256"])
        restored.append(row)
    return restored


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    if str(config.get("task")) != "prepare_finetune_sources":
        raise ValueError("Configuration task must be prepare_finetune_sources")
    inputs = config.get("inputs") or {}
    outputs = config.get("output") or {}
    selection_cfg = config.get("selection") or {}
    b_cfg = selection_cfg.get("negative_removed") or {}
    c_cfg = selection_cfg.get("teacher_student") or {}
    d_cfg = selection_cfg.get("pretrain_replay") or {}
    b_enabled = bool(b_cfg.get("enabled", True)) and int(b_cfg.get("max_items", 0)) > 0
    # A zero request budget is the normal HLML-3.0 score-pool mode: predict
    # every eligible ROI now, then freeze disjoint selections per round later.
    c_enabled = bool(c_cfg.get("enabled", True))
    d_enabled = bool(d_cfg.get("enabled", True))

    registry_path = resolve_path(str(inputs.get("source_registry") or ""), config)
    if not registry_path.is_file() and (b_enabled or c_enabled or d_enabled):
        raise FileNotFoundError("HLMF pretrain source registry is missing: {}".format(registry_path))
    registry_report = (
        _validate_registry_report(registry_path) if registry_path.is_file() else None
    )
    registry_rows = [clean_row(row) for row in read_jsonl(registry_path)] if registry_path.is_file() else []
    required_names = []
    if b_enabled:
        required_names.extend(["negative_removed_manifest", "pretrain_catalog"])
    if c_enabled:
        required_names.extend(["pretrain_geometry_labels", "geometry_checkpoint"])
    if d_enabled:
        required_names.extend(
            ["pretrain_multitask_labels", "pretrain_curation_manifest", "train_crop_root"]
        )
    input_paths = {
        name: resolve_path(str(inputs.get(name) or ""), config)
        for name in sorted(set(required_names))
    }
    missing = [
        str(path) for name, path in input_paths.items()
        if not (path.is_dir() if name == "train_crop_root" else path.is_file())
    ]
    if missing:
        raise FileNotFoundError("Missing prepare-finetune input(s): {}".format(missing))

    workspace = resolve_path(str(outputs.get("workspace") or ""), config)
    gold_repository = resolve_path(str(config.get("gold_repository_root") or ""), config)
    occupied_gold_tokens, occupied_gold_report = gold_repository_occupied(gold_repository)
    mining_target = workspace / "mining"
    replay_target = workspace / "sources" / "replay" / "pretrain_replay"
    if mining_target.exists() or (d_enabled and replay_target.exists()):
        raise FileExistsError("P2 outputs are immutable; choose a new HAND_FINETUNE_ID")
    workspace.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".prepare_finetune.", dir=str(workspace)))
    try:
        mining = temporary / "mining"
        replay_staging = temporary / "pretrain_replay"
        # Manifest and MediaPipe draft files are shared by tens of thousands of
        # registry rows.  Authenticate each distinct physical file once while
        # still hashing every distinct ROI image.
        provenance_hash_cache: Dict[Path, str] = {}
        restored_canonical = (
            _restore_rows(
                [clean_row(row) for row in read_jsonl(input_paths["pretrain_multitask_labels"])],
                registry_rows,
                provenance_hash_cache,
            )
            if d_enabled else []
        )
        restored_geometry = (
            _restore_rows(
                [clean_row(row) for row in read_jsonl(input_paths["pretrain_geometry_labels"])],
                registry_rows,
                provenance_hash_cache,
            )
            if c_enabled else []
        )
        catalog_rows = (
            _restore_rows(
                [clean_row(row) for row in read_jsonl(input_paths["pretrain_catalog"])],
                registry_rows,
                provenance_hash_cache,
            )
            if b_enabled else []
        )
        prediction_path = mining / "teacher_student" / "train_predictions.jsonl"
        prediction_report_path = mining / "teacher_student" / "prediction_report.json"
        if c_enabled:
            predict_training_labels(
                input_paths["pretrain_geometry_labels"],
                input_paths["geometry_checkpoint"],
                config.get("model") or {},
                prediction_path,
                prediction_report_path,
                batch_size=int((config.get("prediction") or {}).get("batch_size", 64)),
            )
            with prediction_report_path.open("r", encoding="utf-8") as handle:
                prediction_report = json.load(handle)
            prediction_report["output"]["path"] = str(
                (mining_target / "teacher_student" / "train_predictions.jsonl").resolve()
            )
            write_json(prediction_report_path, prediction_report)
            prediction_rows = [clean_row(row) for row in read_jsonl(prediction_path)]
        else:
            prediction_rows = []
            write_json(prediction_report_path, {"status": "disabled", "record_count": 0})
        # Prediction rows need only identity/output values; selector teacher rows
        # already carry the authenticated HLMF restoration paths.
        removed_rows = (
            [clean_row(row) for row in read_jsonl(input_paths["negative_removed_manifest"])]
            if b_enabled else []
        )
        b_requests, b_report = select_negative_removed(
            removed_rows,
            catalog_rows,
            b_cfg,
            occupied_identity_tokens=occupied_gold_tokens,
            provenance_hash_cache=provenance_hash_cache,
        )
        occupied = {str(row["parent_global_crop_id"]) for row in b_requests}
        occupied_tokens = set(occupied_gold_tokens)
        for row in b_requests:
            occupied_tokens.update(identity_tokens(row))
        c_requests, c_report, scored_rows = select_teacher_student(
            restored_geometry,
            prediction_rows,
            c_cfg,
            occupied_parent_ids=occupied,
            occupied_identity_tokens=occupied_tokens,
            provenance_hash_cache=provenance_hash_cache,
        )
        b_path = mining / "negative_removed_gold" / "selection_request.jsonl"
        b_report_path = mining / "negative_removed_gold" / "selection_report.json"
        c_path = mining / "disagreement_gold" / "selection_request.jsonl"
        c_report_path = mining / "disagreement_gold" / "selection_report.json"
        score_path = mining / "teacher_student" / "disagreement_scores.jsonl"
        write_jsonl(b_path, b_requests)
        write_jsonl(c_path, c_requests)
        write_jsonl(score_path, scored_rows)
        b_report["inputs"] = ({
            "negative_removed_manifest": str(input_paths["negative_removed_manifest"].resolve()),
            "negative_removed_sha256": sha256_file(input_paths["negative_removed_manifest"]),
            "catalog_sha256": sha256_file(input_paths["pretrain_catalog"]),
            "source_registry_sha256": sha256_file(registry_path),
        } if b_enabled else {})
        b_report["output"] = {
            "path": str((mining_target / "negative_removed_gold" / b_path.name).resolve()),
            "sha256": sha256_file(b_path),
        }
        c_report["inputs"] = ({
            "geometry_labels_sha256": sha256_file(input_paths["pretrain_geometry_labels"]),
            "predictions_sha256": sha256_file(prediction_path),
            "checkpoint_sha256": sha256_file(input_paths["geometry_checkpoint"]),
            "source_registry_sha256": sha256_file(registry_path),
        } if c_enabled else {})
        c_report["output"] = {
            "path": str((mining_target / "disagreement_gold" / c_path.name).resolve()),
            "sha256": sha256_file(c_path),
        }
        write_json(b_report_path, b_report)
        write_json(c_report_path, c_report)

        pretrain_id = str(config.get("pretrain_id") or "")
        if not pretrain_id:
            raise ValueError("pretrain_id is required")
        if d_enabled:
            replay_result = build_replay_source(
                restored_canonical,
                d_cfg,
                replay_staging,
                input_paths["pretrain_curation_manifest"],
                input_paths["train_crop_root"],
                pretrain_id,
                _git_version(Path(config["_meta"]["repo_root"])),
            )
            replay_result["source_dir"] = str(replay_target.resolve())
            replay_result["descriptor"] = str((replay_target / "finetune_source.json").resolve())
        else:
            replay_result = {"status": "disabled"}
        total_report = {
            "status": "ok",
            "schema_version": "finetune_source_preparation_v1",
            "pretrain_id": pretrain_id,
            "finetune_id": str(config.get("finetune_id") or ""),
            "config": {"path": str(Path(args.config).resolve()), "sha256": sha256_file(args.config)},
            "inputs": {
                **{
                    name + "_sha256": sha256_file(path)
                    for name, path in input_paths.items()
                    if path.is_file()
                },
                **(
                    {"source_registry_sha256": sha256_file(registry_path)}
                    if registry_path.is_file() else {}
                ),
                **(
                    {"source_registry_report_sha256": registry_report["sha256"]}
                    if registry_report else {}
                ),
            },
            "selectors": {"negative_removed": b_report, "teacher_student": c_report},
            "historical_gold_exclusion": occupied_gold_report,
            "replay": replay_result,
        }
        write_json(mining / "prepare_finetune_sources_report.json", total_report)

        mining_target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(str(mining), str(mining_target))
        if d_enabled:
            replay_target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(str(replay_staging), str(replay_target))
        shutil.rmtree(str(temporary), ignore_errors=True)
        temporary = None  # type: ignore[assignment]
        print(json.dumps({"status": "ok", "mining": str(mining_target), "replay": str(replay_target) if d_enabled else None}, ensure_ascii=False, indent=2))
    finally:
        if temporary is not None and temporary.exists():
            shutil.rmtree(str(temporary), ignore_errors=True)


if __name__ == "__main__":
    main()
