#!/usr/bin/env python3
"""Export trained Hand Landmarker weights to a verified fixed-interface ONNX."""

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hand_landmarker.config import load_config
from hand_landmarker.export import export_from_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Export YAML file")
    args = parser.parse_args()
    print(json.dumps(export_from_config(load_config(args.config)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

