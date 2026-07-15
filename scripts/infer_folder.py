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


def _apply_cli_overrides(config, args):
    if args.model_path:
        config.setdefault("hand", {})["model_path"] = args.model_path
    if args.output_dir:
        config.setdefault("output", {})["dir"] = args.output_dir
    if args.overwrite:
        config.setdefault("output", {})["overwrite"] = True
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Folder-inference YAML file")
    parser.add_argument("--model-path", help="Override hand.model_path")
    parser.add_argument("--output-dir", help="Override output.dir")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Authorize replacement of existing inference outputs",
    )
    args = parser.parse_args()
    config = _apply_cli_overrides(load_config(args.config), args)
    summary = infer_folder_from_config(config)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary.get("failed_count", 0):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
