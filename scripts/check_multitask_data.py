#!/usr/bin/env python3
"""Fail closed unless multitask pretrain uses enough human-confirmed negatives."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Mapping

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hand_landmarker.config import load_config, resolve_path
from hand_landmarker.io_utils import read_jsonl, write_json
from hand_landmarker.pretrain_curation import verify_curation_manifest


def check_multitask_data(config: Mapping[str, Any]) -> Dict[str, Any]:
    if str(config.get("task")) != "train" or str(config.get("stage")) != "pretrain":
        raise ValueError("Multitask gate requires a pretrain task: train config")
    data = dict(config.get("data") or {})
    manifest = verify_curation_manifest(config, data)
    labels_path = resolve_path(str(data.get("labels") or ""), config)
    rows = list(read_jsonl(labels_path))
    if not rows:
        raise ValueError("Multitask labels are empty: {}".format(labels_path))

    gate = dict(config.get("multitask_gate") or {})
    minimum_total = int(gate.get("minimum_confirmed_negatives", 1))
    minimum_by_type = {
        str(key): int(value)
        for key, value in dict(gate.get("minimum_confirmed_by_sample_type") or {}).items()
    }
    required_review_fields = [str(value) for value in gate.get("require_review_fields", [])]
    required_review_method = str(gate.get("require_review_method") or "")
    positives = 0
    confirmed = []
    violations = []
    for row in rows:
        present = bool((row.get("hand_presence") or {}).get("present", False))
        curation = row.get("pretrain_curation") or {}
        if present:
            positives += 1
            if curation.get("action") != "INCLUDE_LANDMARKS":
                violations.append("{}: positive lacks INCLUDE_LANDMARKS".format(row.get("crop_id")))
            continue
        review = curation.get("review") or {}
        if (
            curation.get("action") != "INCLUDE_CONFIRMED_NEGATIVE"
            or curation.get("negative_evidence") != "human_confirmed"
            or review.get("decision") != "CONFIRMED_NEGATIVE"
        ):
            violations.append("{}: unconfirmed negative entered multitask".format(row.get("crop_id")))
            continue
        missing = [field for field in required_review_fields if not str(review.get(field) or "").strip()]
        if missing:
            violations.append(
                "{}: confirmed negative lacks review fields {}".format(row.get("crop_id"), missing)
            )
            continue
        if required_review_method and str(review.get("review_method") or "") != required_review_method:
            violations.append(
                "{}: confirmed negative has review_method {!r}; expected {!r}".format(
                    row.get("crop_id"), review.get("review_method"), required_review_method
                )
            )
            continue
        confirmed.append(row)

    counts = Counter(str(row.get("sample_type")) for row in confirmed)
    checks = {
        "no_unreviewed_negative": not violations,
        "has_positive_geometry": positives > 0,
        "minimum_confirmed_negatives": len(confirmed) >= minimum_total,
        "minimum_confirmed_by_sample_type": all(
            counts.get(sample_type, 0) >= minimum
            for sample_type, minimum in minimum_by_type.items()
        ),
    }
    report = {
        "status": "pass" if all(checks.values()) else "fail",
        "labels": str(labels_path),
        "manifest": manifest,
        "record_count": len(rows),
        "positive_count": positives,
        "confirmed_negative_count": len(confirmed),
        "confirmed_negative_by_sample_type": dict(sorted(counts.items())),
        "requirements": {
            "minimum_confirmed_negatives": minimum_total,
            "minimum_confirmed_by_sample_type": minimum_by_type,
            "require_review_method": required_review_method,
            "require_review_fields": required_review_fields,
        },
        "checks": checks,
        "violations": violations[:50],
        "violation_count": len(violations),
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--report", default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    report = check_multitask_data(config)
    report_path = (
        resolve_path(args.report, config)
        if args.report
        else resolve_path(config["outputs"]["run_dir"], config).parent
        / "multitask_data_gate.json"
    )
    write_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    if report["status"] != "pass":
        raise SystemExit(
            "Multitask data gate failed; complete human negative review and rerun curation"
        )


if __name__ == "__main__":
    main()
