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
        if "weights_path" in export:
            export["weights_path"] = args.weights_path
        model = config.setdefault("model", {})
        if "checkpoint" in model:
            model["checkpoint"] = args.weights_path
    if args.output_path:
        export["model_path"] = args.output_path
        export.pop("output", None)
        output = config.get("output")
        if isinstance(output, dict):
            output.pop("model_path", None)
    if args.contract_path:
        export["contract_path"] = args.contract_path
    if args.overwrite:
        export["overwrite"] = True
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Export YAML file")
    parser.add_argument("--weights-path", help="Override Hand Landmarker weights")
    parser.add_argument("--output-path", help="Override export.model_path")
    parser.add_argument("--contract-path", help="Override export.contract_path")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Authorize replacement of existing ONNX and contract outputs",
    )
    args = parser.parse_args()
    config = _apply_cli_overrides(load_config(args.config), args)
    print(json.dumps(export_from_config(config), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
