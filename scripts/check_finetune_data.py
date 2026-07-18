#!/usr/bin/env python3
"""Authenticate finalized finetune data and its training-time contracts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Mapping

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hand_landmarker.config import load_config, resolve_path
from hand_landmarker.finetune_curation import verify_finetune_curation_manifest
from hand_landmarker.finetune_source import (
    validate_finetune_source,
    validate_gold_aggregate,
    validate_source_set,
)
from hand_landmarker.io_utils import read_jsonl, sha256_file
from hand_landmarker.io_utils import write_json
from hand_landmarker.data import WeightedStratifiedSampler


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("JSON root must be an object: {}".format(path))
    return value


def _artifact_gate(manifest: Mapping[str, Any], output_root: Path) -> None:
    for relative, entry in dict(manifest.get("artifacts") or {}).items():
        path_value = Path(str(relative))
        if path_value.is_absolute() or ".." in path_value.parts:
            raise ValueError("Curation artifact path traversal: {}".format(relative))
        path = output_root / path_value
        if not path.is_file() or path.is_symlink():
            raise ValueError("Curation artifact is missing or a symlink: {}".format(path))
        if sha256_file(path) != str((entry or {}).get("sha256") or ""):
            raise ValueError("Curation artifact SHA mismatch: {}".format(path))
        if "count" in entry and len(read_jsonl(path)) != int(entry["count"]):
            raise ValueError("Curation artifact count mismatch: {}".format(path))


def _aggregate_repository_root(aggregate_path: Path) -> Path:
    """Recover the authenticated Gold repository root from its descriptor."""

    descriptor = _read_json(aggregate_path)
    value = Path(str(descriptor.get("gold_repository_root") or ""))
    if not value.is_absolute() or value.is_symlink() or not value.is_dir():
        raise ValueError("Gold aggregate repository root is missing or invalid")
    return value.resolve(strict=True)


def _sampling_gate(config: Mapping[str, Any], rows: Any) -> Dict[str, Any]:
    if str(config.get("stage")) != "finetune":
        raise ValueError("check_finetune_data accepts only stage=finetune")
    training = config.get("training") or {}
    sampling = config.get("sampling") or {}
    gold_fraction = float(training.get("gold_fraction", sampling.get("gold_fraction", -1.0)))
    if not 0.30 <= gold_fraction <= 0.50:
        raise ValueError("Finetune gold_fraction must be in [0.30,0.50]")
    by_tier = sampling.get("sample_type_fractions_by_tier")
    expected_types = {
        "POS_RUNTIME", "POS_LOW_PALM", "NEG_RUNTIME_CANDIDATE", "NEG_LOW_PALM_CANDIDATE"
    }
    if not isinstance(by_tier, Mapping) or set(by_tier) != {"gold", "pseudo"}:
        raise ValueError("Finetune sampling must define gold/pseudo type fractions")
    counts = Counter(
        (str(row.get("supervision_tier")), str(row.get("sample_type"))) for row in rows
    )
    for tier in ("gold", "pseudo"):
        fractions = {str(key): float(value) for key, value in dict(by_tier[tier]).items()}
        if set(fractions) != expected_types or any(not math.isfinite(value) or value < 0 for value in fractions.values()):
            raise ValueError("Invalid sample_type fractions for tier {}".format(tier))
        if not math.isclose(sum(fractions.values()), 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("Sample type fractions must sum to 1 for tier {}".format(tier))
        for sample_type, fraction in fractions.items():
            if tier == "pseudo" and fraction > 0 and counts[(tier, sample_type)] == 0:
                raise ValueError("Required pseudo sampling cell is empty: {}".format(sample_type))
    batch_size = int(training.get("batch_size", 0))
    epoch_size = sampling.get("epoch_size")
    if batch_size <= 0:
        raise ValueError("training.batch_size must be positive")
    tail_records = 0
    if epoch_size not in (None, "auto"):
        tail_records = int(epoch_size) % batch_size
        if tail_records and sampling.get("tail_batch_policy") != "allow_smaller_final_batch":
            raise ValueError(
                "Non-divisible sampling.epoch_size requires tail_batch_policy=allow_smaller_final_batch"
            )
    if sampling.get("replacement", True) is not True or sampling.get("honor_record_sampling_weight", True) is not True:
        raise ValueError("Finetune sampler must use replacement and row sampling weights")
    if epoch_size in (None, "auto"):
        raise ValueError("Formal finetune requires an explicit integer sampling.epoch_size")
    epoch_size_value = int(epoch_size)
    batch_sizes = [batch_size] * (epoch_size_value // batch_size)
    if tail_records:
        batch_sizes.append(tail_records)
    sampler = WeightedStratifiedSampler(
        rows,
        stage="finetune",
        seed=int((config.get("experiment") or {}).get("seed", 0)),
        weight_key=str(sampling.get("weight_key", "sampling_weight")),
        gold_fraction=gold_fraction,
        supervision_fractions=sampling.get("supervision_fractions"),
        sample_type_fractions=sampling.get("sample_type_fractions"),
        sample_type_fractions_by_tier=by_tier,
        missing_cell_policy=sampling.get("missing_cell_policy"),
        rare_cell_policy=sampling.get("rare_cell_policy"),
        tier_key=str(sampling.get("tier_key", "supervision_tier")),
        bucket_key=str(sampling.get("bucket_key", "sampling_bucket")),
        sample_type_key=str(sampling.get("sample_type_key", "sample_type")),
        quota_tie_break=sampling.get(
            "quota_tie_break",
            ["POS_RUNTIME", "POS_LOW_PALM", "NEG_RUNTIME_CANDIDATE", "NEG_LOW_PALM_CANDIDATE"],
        ),
        require_all_tier_sample_type_cells=sampling.get(
            "require_all_tier_sample_type_cells", True
        ),
    )
    sampled = sampler.sample_epoch(batch_sizes, epoch=0)
    if len(sampled) != epoch_size_value or sampler.last_epoch_plan is None:
        raise ValueError("Finetune sampler did not produce the exact epoch-0 plan")
    return {
        "gold_fraction": gold_fraction,
        "records_by_tier_sample_type": {
            "{}:{}".format(*key): value for key, value in sorted(counts.items())
        },
        "batch_size": batch_size,
        "epoch_size": epoch_size,
        "tail_batch_records": tail_records,
        "sampler_definition": sampler.report(),
        "epoch0_plan": sampler.last_epoch_plan,
    }


def _checkpoint_gate(
    config: Mapping[str, Any],
    checkpoint: Path,
    resume: bool,
) -> Dict[str, Any]:
    """Authenticate the complete training run that published the start weights."""

    output_value = str((config.get("outputs") or {}).get("run_dir") or "")
    if not output_value:
        raise ValueError("outputs.run_dir is required")
    output_run = resolve_path(output_value, config)
    if not resume and output_run.exists():
        raise FileExistsError("Fresh finetune outputs.run_dir must not exist: {}".format(output_run))
    if checkpoint.is_symlink() or not checkpoint.is_file():
        raise ValueError("Starting checkpoint must be a regular non-symlink file")
    checkpoint = checkpoint.resolve(strict=True)
    candidates = [checkpoint.parent, checkpoint.parent.parent, checkpoint.parent.parent.parent]
    run_dirs = [path for path in candidates if (path / "training_report.json").is_file()]
    if len(run_dirs) != 1:
        raise ValueError("Starting checkpoint must belong to exactly one authenticated training run")
    run_dir = run_dirs[0].resolve(strict=True)
    if _within(output_run.resolve(), run_dir) or _within(run_dir, output_run.resolve()):
        raise ValueError("Finetune output run overlaps its starting/pretrain run")
    report_path = run_dir / "training_report.json"
    metadata_path = run_dir / "experiment_metadata.json"
    if report_path.is_symlink() or metadata_path.is_symlink() or not metadata_path.is_file():
        raise ValueError("Starting checkpoint run report/metadata must be regular non-symlink files")
    report = _read_json(report_path)
    metadata = _read_json(metadata_path)
    expected_stage = "finetune" if resume else "pretrain"
    for name, value in (("training_report", report), ("experiment_metadata", metadata)):
        if str(value.get("status")) != "complete":
            raise ValueError("{} is not complete".format(name))
        if str(value.get("stage")) != expected_stage or str(value.get("model_version")) != "v2":
            raise ValueError("{} stage/model provenance mismatch".format(name))
        artifacts = value.get("artifacts") or {}
        names = ("best_checkpoint", "last_checkpoint") if resume else ("best_checkpoint",)
        matches = []
        for artifact_name in names:
            artifact = artifacts.get(artifact_name)
            if not isinstance(artifact, Mapping):
                continue
            artifact_path = Path(str(artifact.get("path") or ""))
            if artifact_path.resolve() == checkpoint and str(artifact.get("sha256") or "") == sha256_file(checkpoint):
                matches.append(artifact_name)
        if len(matches) != 1:
            raise ValueError("{} starting checkpoint path/SHA mismatch".format(name))
    if Path(str(report.get("metadata_path") or "")).resolve() != metadata_path.resolve():
        raise ValueError("Training report metadata_path mismatch")
    if not resume:
        report_experiment = str(report.get("experiment") or "")
        metadata_experiment = str((metadata.get("experiment") or {}).get("name") or "")
        resolved_experiment = str(
            (((metadata.get("resolved_config") or {}).get("experiment") or {}).get("name") or "")
        )
        if (
            run_dir.name != "multitask"
            or report_experiment != metadata_experiment
            or metadata_experiment != resolved_experiment
            or not metadata_experiment.endswith("_multitask")
        ):
            raise ValueError(
                "Fresh finetune must start from an authenticated multitask best run"
            )
    return {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "source_run": str(run_dir),
        "training_report": str(report_path),
        "training_report_sha256": sha256_file(report_path),
        "experiment_metadata": str(metadata_path),
        "experiment_metadata_sha256": sha256_file(metadata_path),
        "stage": expected_stage,
        "model_version": "v2",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--report", help="Optional gate report JSON path")
    args = parser.parse_args()
    config = load_config(args.config)
    dataset = config.get("data") or {}
    authenticated = verify_finetune_curation_manifest(config, dataset)
    manifest_path = Path(authenticated["path"])
    manifest = _read_json(manifest_path)
    output_root = Path(str(manifest["output_dir"])).resolve(strict=True)
    _artifact_gate(manifest, output_root)
    labels_path = resolve_path(str(dataset.get("labels") or ""), config)
    rows = read_jsonl(labels_path)
    if not rows:
        raise ValueError("Finetune labels are empty")

    allowed_roots = [Path(str(value)).resolve(strict=True) for value in dataset.get("allowed_crop_roots") or []]
    image_entries = []
    for row in rows:
        path = Path(str(row.get("crop_path") or ""))
        if not path.is_absolute() or not path.is_file() or path.is_symlink():
            raise ValueError("Finetune crop is missing/not absolute/a symlink: {}".format(path))
        lexical_roots = []
        for root in allowed_roots:
            try:
                path.relative_to(root)
                lexical_roots.append(root)
            except ValueError:
                pass
        resolved = path.resolve(strict=True)
        if not lexical_roots or not any(_within(resolved, root) for root in lexical_roots):
            raise ValueError("Finetune crop is outside data.allowed_crop_roots: {}".format(path))
        for root in lexical_roots:
            current = root
            for part in path.relative_to(root).parts:
                current = current / part
                if current.is_symlink():
                    raise ValueError("Finetune crop traverses a symlink: {}".format(current))
        digest = str(row.get("image_sha256") or "")
        if not digest or sha256_file(resolved) != digest:
            raise ValueError("Finetune crop SHA mismatch: {}".format(path))
        image_entries.append((str(resolved), digest))
    image_entries.sort()
    aggregate = hashlib.sha256(
        "".join("{}:{}\n".format(path, digest) for path, digest in image_entries).encode("utf-8")
    ).hexdigest()
    if aggregate != str((manifest.get("images") or {}).get("aggregate_sha256") or ""):
        raise ValueError("Finetune image aggregate SHA mismatch")
    if len(image_entries) != int((manifest.get("images") or {}).get("count", -1)):
        raise ValueError("Finetune image count mismatch")

    descriptors = list(manifest.get("source_descriptors") or [])
    if not any(item.get("source_kind") == "pretrain_replay" for item in descriptors):
        raise ValueError("Finetune snapshot has no replay source")
    if not any(item.get("source_kind") != "pretrain_replay" for item in descriptors):
        raise ValueError("Finetune snapshot has no Gold source")
    validated_sources = []
    for item in descriptors:
        path = Path(str(item.get("descriptor_path") or ""))
        if not path.is_file() or path.is_symlink() or sha256_file(path) != str(item.get("descriptor_sha256") or ""):
            raise ValueError("Source descriptor changed after curation: {}".format(path))
        validated_sources.append(validate_finetune_source(path, allowed_roots))
    validate_source_set(validated_sources)
    aggregate_ref = manifest.get("gold_aggregate") or {}
    aggregate_path = Path(str(aggregate_ref.get("path") or ""))
    if not aggregate_path.is_file() or aggregate_path.is_symlink() or sha256_file(aggregate_path) != str(aggregate_ref.get("sha256") or ""):
        raise ValueError("Gold aggregate changed after curation")
    gold_sources = [source for source in validated_sources if source["source_kind"] != "pretrain_replay"]
    validate_gold_aggregate(
        aggregate_path, _aggregate_repository_root(aggregate_path), gold_sources
    )

    smoke = manifest.get("smoke") or {}
    if int(smoke.get("count", -1)) != 256:
        raise ValueError("Authenticated finetune smoke snapshot must contain 256 rows")
    smoke_labels = output_root / str(smoke.get("labels") or "")
    smoke_selection = output_root / str(smoke.get("selection") or "")
    if len(read_jsonl(smoke_labels)) != 256 or len(read_jsonl(smoke_selection)) != 256:
        raise ValueError("Persisted smoke labels/selection coverage mismatch")

    sampling_report = _sampling_gate(config, rows)
    initial = config.get("training") or {}
    initial_path_value = initial.get("initial_checkpoint")
    resume_path_value = initial.get("resume_checkpoint")
    if initial_path_value and resume_path_value:
        raise ValueError("initial_checkpoint and resume_checkpoint are mutually exclusive")
    start_path = resolve_path(str(resume_path_value or initial_path_value or ""), config)
    if not start_path.is_file():
        raise FileNotFoundError("Finetune starting checkpoint is missing: {}".format(start_path))
    if str((config.get("model") or {}).get("version")) != "v2":
        raise ValueError("Finetune starting model contract must be version v2")
    checkpoint_provenance = _checkpoint_gate(
        config, start_path, resume=bool(resume_path_value)
    )
    run_dir = resolve_path(str((config.get("outputs") or {}).get("run_dir")), config)
    report = {
        "status": "ok",
        "schema_version": "finetune_data_gate_v1",
        "curation": authenticated,
        "records": len(rows),
        "sources": len(descriptors),
        "sampling": sampling_report,
        "starting_checkpoint": checkpoint_provenance,
    }
    if args.report:
        report_path = resolve_path(args.report, config)
    else:
        report_path = run_dir.parent / "finetune_data_gate.json"
    write_json(report_path, report)
    report["report_path"] = str(report_path.resolve())
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


def _within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


if __name__ == "__main__":
    main()
