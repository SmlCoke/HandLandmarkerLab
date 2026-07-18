#!/usr/bin/env python3
"""Run the read-only finetune source/aggregate/merge gate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hand_landmarker.finetune_curation import check_finetune_sources_from_config
from hand_landmarker.config import resolve_path
from hand_landmarker.io_utils import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--report", help="Optional gate report JSON path")
    args = parser.parse_args()
    checked = check_finetune_sources_from_config(args.config)
    report = {
        "status": "ok",
        "gold_records": len(checked["gold_rows"]),
        "replay_records_after_override": len(checked["replay_rows"]),
        "ignored_records": len(checked["ignored_rows"]),
        "superseded_by_gold": len(checked["superseded_rows"]),
        "smoke_records": len(checked["smoke_rows"]),
        "source_roles": checked["reports"]["source_roles"],
        "source_selection": checked["reports"]["source_selection"],
        "source_selection_manifest": checked["reports"]["source_selection_manifest"],
        "leakage": checked["reports"]["leakage"],
        "gold_aggregate": {
            "path": checked["aggregate"]["path"],
            "sha256": checked["aggregate"]["sha256"],
        },
    }
    if args.report:
        report_path = resolve_path(args.report, checked["config"])
    else:
        report_path = Path(checked["aggregate"]["path"]).parent.parent / "qc" / "finetune_sources_gate.json"
    write_json(report_path, report)
    report["report_path"] = str(report_path.resolve())
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
