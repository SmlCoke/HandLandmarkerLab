#!/usr/bin/env python3
"""Materialize an auditable positive-landmark pretrain snapshot."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hand_landmarker.config import load_config
from hand_landmarker.pretrain_curation import curate_pretrain_from_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Pretrain curation YAML file")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace exactly the configured snapshot directory",
    )
    parser.add_argument(
        "--finalize-retained-review",
        action="store_true",
        help="Treat every retained, manifest-matched review image as a confirmed negative",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    report = curate_pretrain_from_config(
        config,
        overwrite=True if args.overwrite else None,
        finalize_review=args.finalize_retained_review,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
