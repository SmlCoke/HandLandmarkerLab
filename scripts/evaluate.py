#!/usr/bin/env python3
"""Evaluate Hand Landmarker on canonical Val/Test Gold."""

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hand_landmarker.config import load_config
from hand_landmarker.evaluation import evaluate_from_config


def _apply_cli_overrides(config, args):
    if args.model_path:
        config.setdefault("hand", {})["model_path"] = args.model_path
        model = config.setdefault("model", {})
        if "checkpoint" in model:
            model["checkpoint"] = args.model_path
    if args.output_dir:
        output = config.setdefault("output", {})
        output["dir"] = args.output_dir
        output.pop("directory", None)
        paths = config.get("paths")
        if isinstance(paths, dict):
            paths.pop("output_dir", None)
        outputs = config.get("outputs")
        if isinstance(outputs, dict):
            outputs.pop("predictions", None)
            outputs.pop("metrics", None)
    if args.overwrite:
        config.setdefault("output", {})["overwrite"] = True
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Evaluation YAML file")
    parser.add_argument("--model-path", help="Override hand.model_path")
    parser.add_argument("--output-dir", help="Override output.dir")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Authorize replacement of existing evaluation outputs",
    )
    args = parser.parse_args()
    config = _apply_cli_overrides(load_config(args.config), args)
    report = evaluate_from_config(config)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
