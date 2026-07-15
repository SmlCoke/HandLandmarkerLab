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


def _apply_cli_overrides(config, args):
    export = config.setdefault("export", {})
    if args.weights_path:
        config.setdefault("hand", {})["model_path"] = args.weights_path
    if args.output_path:
        export["model_path"] = args.output_path
    if args.contract_path:
        export["contract_path"] = args.contract_path
    if getattr(args, "conversion_output_dir", None):
        export.setdefault("conversion_datasets", {})[
            "output_dir"
        ] = args.conversion_output_dir
    if args.overwrite:
        export["overwrite"] = True
    if getattr(args, "force", False):
        export["force_a1_operator_export"] = True
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Export YAML file")
    parser.add_argument("--weights-path", help="Override Hand Landmarker weights")
    parser.add_argument("--output-path", help="Override export.model_path")
    parser.add_argument("--contract-path", help="Override export.contract_path")
    parser.add_argument(
        "--conversion-output-dir",
        help="Override export.conversion_datasets.output_dir",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Authorize replacement of existing ONNX, contract, and conversion dataset outputs",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Force export past only the A1 operator name/attribute gate; "
            "all other export validation and overwrite protection remain enabled"
        ),
    )
    args = parser.parse_args()
    config = _apply_cli_overrides(load_config(args.config), args)
    print(json.dumps(export_from_config(config), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
