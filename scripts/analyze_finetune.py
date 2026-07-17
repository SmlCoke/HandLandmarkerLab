#!/usr/bin/env python3
"""Compare two completed finetune runs without modifying either run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hand_landmarker.finetune_analysis import analyze_finetune_runs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--baseline-id", required=True)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overlay-limit", type=int, default=40)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    report = analyze_finetune_runs(
        work_root=args.work_root.resolve(),
        baseline_id=args.baseline_id,
        candidate_id=args.candidate_id,
        labels_path=args.labels.resolve(),
        output_dir=args.output_dir.resolve(),
        overlay_limit=args.overlay_limit,
        overwrite=args.overwrite,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
