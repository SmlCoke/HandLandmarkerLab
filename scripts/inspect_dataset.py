#!/usr/bin/env python3
"""Validate canonical HandLandmarkerFab labels and cross-split leakage."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hand_landmarker.config import load_config, resolve_path
from hand_landmarker.inspect import inspect_config
from hand_landmarker.io_utils import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect canonical JSONL, crop paths/shapes/hashes, counts, and split leakage."
    )
    parser.add_argument("--config", required=True, help="Training/evaluation YAML configuration")
    parser.add_argument(
        "--compare-labels",
        action="append",
        default=[],
        help="Additional canonical JSONL to compare for leakage (repeatable)",
    )
    parser.add_argument("--output", default=None, help="Optional JSON report path")
    parser.add_argument(
        "--no-image-hash",
        action="store_true",
        help="Skip crop SHA-256 and content-leakage checks (path/schema checks still run)",
    )
    parser.add_argument(
        "--strict-warnings",
        action="store_true",
        help="Return non-zero when warnings are present even if all hard checks pass",
    )
    return parser.parse_args()


def _warning_count(report) -> int:
    count = sum(len(value.get("warnings", [])) for value in report.get("datasets", {}).values())
    count += sum(len(value.get("warnings", [])) for value in report.get("leakage", []))
    return count


def main() -> int:
    args = parse_args()
    try:
        config = load_config(args.config)
        report = inspect_config(
            config,
            compare_labels=args.compare_labels,
            check_images=True,
            hash_images=not args.no_image_hash,
        )
        output_value = args.output or config.get("inspection", {}).get("report")
        if output_value:
            output_path = resolve_path(output_value, config)
            write_json(output_path, report)
            report["report_path"] = str(output_path)
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        if report.get("status") != "ok":
            return 2
        if args.strict_warnings and _warning_count(report):
            return 3
        return 0
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

