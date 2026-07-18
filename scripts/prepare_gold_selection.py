#!/usr/bin/env python3
"""Freeze explicit per-batch GoldSource decisions for one finetune run."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hand_landmarker.finetune_curation import prepare_gold_selection_from_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--enable-source-ids",
        default="",
        help="Comma/space separated source IDs to enable; every other published batch is disabled.",
    )
    args = parser.parse_args()
    enabled = [value for value in re.split(r"[\s,]+", args.enable_source_ids) if value]
    report = prepare_gold_selection_from_config(args.config, enabled)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
