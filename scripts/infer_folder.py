#!/usr/bin/env python3
"""Run the frozen Palm -> Hand cascade on an arbitrary image folder."""

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hand_landmarker.config import load_config
from hand_landmarker.visualization import infer_folder_from_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Folder-inference YAML file")
    args = parser.parse_args()
    summary = infer_folder_from_config(load_config(args.config))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary.get("failed_count", 0):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
