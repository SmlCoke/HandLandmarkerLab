#!/usr/bin/env python3
"""Train the fixed-interface Hand Landmarker from one stage YAML config."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hand_landmarker.config import load_config
from hand_landmarker.training import train_from_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Pretrain or finetune YAML file")
    override = parser.add_mutually_exclusive_group()
    override.add_argument(
        "--resume-checkpoint",
        help="Override training.resume_checkpoint and restore its optimizer/epoch sidecar when available",
    )
    override.add_argument(
        "--initial-checkpoint",
        help="Override training.initial_checkpoint for fresh-optimizer finetuning",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    training = config.setdefault("training", {})
    if args.resume_checkpoint:
        training["resume_checkpoint"] = args.resume_checkpoint
        training["initial_checkpoint"] = None
    elif args.initial_checkpoint:
        training["initial_checkpoint"] = args.initial_checkpoint
        training["resume_checkpoint"] = None

    report = train_from_config(config)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
