#!/usr/bin/env python3
"""Build deterministic NPY calibration/evaluation inputs without exporting ONNX."""

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hand_landmarker.config import load_config
from hand_landmarker.conversion_datasets import generate_conversion_datasets


def _apply_cli_overrides(config, args):
    export = config.setdefault("export", {})
    conversion = export.setdefault("conversion_datasets", {})
    conversion["enabled"] = True
    if args.output_dir:
        conversion["output_dir"] = args.output_dir
    if args.overwrite:
        export["overwrite"] = True
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Stage-specific export YAML file")
    parser.add_argument(
        "--output-dir", help="Override export.conversion_datasets.output_dir"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Authorize atomic replacement of the conversion dataset output directory",
    )
    args = parser.parse_args()
    config = _apply_cli_overrides(load_config(args.config), args)
    print(json.dumps(generate_conversion_datasets(config), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
