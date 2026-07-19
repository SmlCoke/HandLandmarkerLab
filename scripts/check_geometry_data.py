#!/usr/bin/env python3
"""Gate geometry source isolation and emit a deterministic sampling audit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hand_landmarker.config import load_config, resolve_path
from hand_landmarker.data import CanonicalSequence, OUTPUT_ORDER
from hand_landmarker.inspect import audit_canonical_dataset, leakage_report
from hand_landmarker.io_utils import sha256_file, write_json
from hand_landmarker.pretrain_curation import verify_curation_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify the automatic teacher holdout and audit geometry sampling."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def check_geometry_data(config):
    if str(config.get("task")) != "train" or str(config.get("stage")) != "pretrain":
        raise ValueError("Geometry data gate requires task=train and stage=pretrain")
    dataset = dict(config.get("data") or {})
    training = dict(config.get("training") or {})
    sampling = dict(config.get("sampling") or {})
    targets = dict(config.get("targets") or {})
    holdout_cfg = dict(config.get("teacher_holdout") or {})
    if not bool(holdout_cfg.get("enabled", False)):
        raise ValueError("teacher_holdout.enabled must be true for formal geometry")

    manifest = verify_curation_manifest(config, dataset)
    if not manifest:
        raise ValueError("Formal geometry requires an authenticated curation manifest")
    manifest_path = Path(str(manifest["path"]))
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest_payload = json.load(handle)
    holdout_manifest = dict(manifest_payload.get("teacher_holdout") or {})
    if not bool(holdout_manifest.get("enabled", False)):
        raise ValueError("Curated snapshot does not contain an enabled teacher holdout")

    train_rows, train_report = audit_canonical_dataset(
        config,
        dataset=dataset,
        expected_stage="pretrain",
        check_images=False,
        hash_images=False,
        raise_on_error=True,
    )
    train_sequence = CanonicalSequence(
        train_rows,
        dataset_config=dataset,
        targets_config=targets,
        batch_size=int(training.get("batch_size", 32)),
        training=True,
        stage="pretrain",
        seed=int((config.get("experiment") or {}).get("seed", 0)),
        steps_per_epoch=training.get("steps_per_epoch"),
        augmentation_config=config.get("augmentation", {}),
        training_config=training,
        sampling_config=sampling,
        losses_config=config.get("losses", {}),
        output_order=(config.get("model") or {}).get("output_order", OUTPUT_ORDER),
    )

    holdout_labels = resolve_path(str(holdout_cfg.get("labels") or ""), config)
    if not holdout_labels.is_file():
        raise FileNotFoundError("Teacher holdout labels not found: {}".format(holdout_labels))
    try:
        holdout_relative = holdout_labels.resolve().relative_to(
            Path(str(manifest_payload["output_dir"])).resolve()
        ).as_posix()
    except (KeyError, OSError, ValueError) as exc:
        raise ValueError("Teacher holdout labels are outside the curated snapshot") from exc
    artifact = dict((manifest_payload.get("artifacts") or {}).get(holdout_relative) or {})
    if not artifact.get("sha256") or sha256_file(holdout_labels) != str(artifact["sha256"]):
        raise ValueError("Teacher holdout labels are not authenticated by curation manifest")

    holdout_dataset = dict(dataset)
    holdout_dataset["labels"] = str(holdout_labels)
    holdout_rows, holdout_report = audit_canonical_dataset(
        config,
        dataset=holdout_dataset,
        expected_stage="pretrain",
        check_images=False,
        hash_images=False,
        raise_on_error=True,
    )
    leakage = leakage_report("geometry_train", train_rows, "teacher_holdout", holdout_rows)
    if leakage.get("status") != "ok":
        raise ValueError("Teacher holdout leaks into geometry train: {}".format(leakage["fatal"]))

    selected_ids = set(str(value) for value in holdout_manifest.get("selected_dataset_ids") or [])
    holdout_ids = set(str(row.get("dataset_id") or "") for row in holdout_rows)
    train_ids = set(str(row.get("dataset_id") or "") for row in train_rows)
    if not selected_ids or selected_ids != holdout_ids:
        raise ValueError(
            "Teacher holdout dataset IDs do not match curation selection: {} != {}".format(
                sorted(holdout_ids), sorted(selected_ids)
            )
        )
    if selected_ids & train_ids:
        raise ValueError("A selected holdout dataset is still present in geometry train")
    minimum = int(holdout_cfg.get("minimum_positive_records", 5000))
    maximum = int(holdout_cfg.get("maximum_positive_records", 10000))
    if not minimum <= len(holdout_rows) <= maximum:
        raise ValueError(
            "Teacher holdout size {} is outside [{}, {}]".format(
                len(holdout_rows), minimum, maximum
            )
        )

    sampler_report = train_sequence.sampling_report()
    source_audit = dict(sampler_report.get("source_audit") or {})
    audit_cfg = dict(sampling.get("source_audit") or {})
    maximum_dataset_fraction = float(
        audit_cfg.get("maximum_expected_dataset_fraction", 0.15)
    )
    maximum_group_draws = float(
        audit_cfg.get("maximum_expected_source_group_draws", 25.0)
    )
    errors = []
    if float(source_audit.get("maximum_expected_dataset_fraction", 1.0)) > maximum_dataset_fraction:
        errors.append(
            "maximum expected dataset fraction exceeds {:.4f}".format(
                maximum_dataset_fraction
            )
        )
    if float(source_audit.get("maximum_expected_source_group_draws", 1.0e30)) > maximum_group_draws:
        errors.append(
            "maximum expected source-group draws exceeds {:.4f}".format(
                maximum_group_draws
            )
        )
    report = {
        "status": "failed" if errors else "ok",
        "schema_version": "geometry_sampling_audit_v1",
        "train_labels": str(resolve_path(str(dataset.get("labels")), config)),
        "train_labels_sha256": train_report.get("labels_sha256"),
        "teacher_holdout_labels": str(holdout_labels),
        "teacher_holdout_labels_sha256": holdout_report.get("labels_sha256"),
        "teacher_holdout_positive_records": len(holdout_rows),
        "teacher_holdout_dataset_ids": sorted(selected_ids),
        "train_teacher_holdout_leakage": leakage,
        "fixed_epoch_size": int(train_sequence.epoch_size),
        "batch_size": int(train_sequence.batch_size),
        "steps_per_epoch": int(len(train_sequence)),
        "sampler": sampler_report,
        "limits": {
            "maximum_expected_dataset_fraction": maximum_dataset_fraction,
            "maximum_expected_source_group_draws": maximum_group_draws,
        },
        "errors": errors,
    }
    return report


def main() -> int:
    args = parse_args()
    try:
        config = load_config(args.config)
        report = check_geometry_data(config)
        output_value = args.output or (config.get("inspection") or {}).get(
            "sampling_audit_report"
        )
        if not output_value:
            raise ValueError("inspection.sampling_audit_report is required")
        output = resolve_path(str(output_value), config)
        write_json(output, report)
        report["report_path"] = str(output)
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        return 0 if report["status"] == "ok" else 2
    except Exception as exc:
        print(
            json.dumps(
                {"status": "failed", "error": "{}: {}".format(type(exc).__name__, exc)},
                ensure_ascii=False,
                indent=2,
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
