#!/usr/bin/env python3
"""Read-only diagnostics for the documented training environment."""

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hand_landmarker.config import load_config
from hand_landmarker.doctor import environment_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="Optional YAML whose paths should be checked")
    parser.add_argument("--allow-nonproduction-python", action="store_true")
    args = parser.parse_args()
    report = environment_report(load_config(args.config) if args.config else None)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    failures = list(report["failures"])
    if args.allow_nonproduction_python:
        failures = [item for item in failures if not item.startswith("Production environment requires Python")]
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

