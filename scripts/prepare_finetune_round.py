#!/usr/bin/env python3
"""Create one immutable, cumulative-disjoint disagreement Gold round."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Set

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hand_landmarker.config import load_config, resolve_path
from hand_landmarker.finetune_selection import (
    clean_row,
    identity_tokens,
    select_teacher_student,
)
from hand_landmarker.io_utils import read_jsonl, sha256_file, write_json, write_jsonl


HARD_GOLD_LIMIT = 800


def _existing_files(paths: Iterable[Path]) -> List[Path]:
    return sorted({path.resolve() for path in paths if path.is_file()}, key=str)


def _occupied(workspace: Path, config: Mapping[str, Any], target: Path) -> tuple[Set[str], Dict[str, Any]]:
    inputs = config.get("round_exclusion") or {}
    paths: List[Path] = []
    gold_root = workspace / "sources" / "gold"
    if gold_root.is_dir():
        paths.extend(gold_root.glob("*/03_reviewed/hand_landmarks_reviewed.jsonl"))
    mining = workspace / "mining"
    if mining.is_dir():
        paths.extend(path for path in mining.rglob("selection_request.jsonl") if path != target)
    cvat_root = workspace / "cvat"
    if cvat_root.is_dir():
        paths.extend(cvat_root.glob("*/02_roi_crops/hand_roi_crops_manifest.jsonl"))
    for name in ("validation_labels", "validation_ignored", "test_labels", "test_ignored"):
        value = inputs.get(name)
        if value:
            paths.append(resolve_path(str(value), config))
    tokens: Set[str] = set()
    records = 0
    reports = []
    for path in _existing_files(paths):
        rows = read_jsonl(path)
        records += len(rows)
        for row in rows:
            tokens.update(identity_tokens(row))
        reports.append({"path": str(path), "sha256": sha256_file(path), "rows": len(rows)})
    return tokens, {"files": reports, "records": records, "identity_token_count": len(tokens)}


def _validate_gold_budget(value: int) -> int:
    budget = int(value)
    if budget <= 0 or budget > HARD_GOLD_LIMIT:
        raise ValueError("gold-budget must be in [1, {}]".format(HARD_GOLD_LIMIT))
    return budget


def _new_recorded_count(workspace: Path, source_id: str, budget: int) -> tuple[int, Dict[str, Any]]:
    if not source_id:
        return 0, {"status": "not_configured", "count": 0}
    descriptor = workspace / "cvat" / source_id / "task_descriptor.json"
    if not descriptor.is_file():
        return 0, {"status": "task_missing", "source_id": source_id, "count": 0}
    value = json.loads(descriptor.read_text(encoding="utf-8"))
    manifest = workspace / "cvat" / source_id / str(value["artifacts"]["manifest"]["path"])
    rows = read_jsonl(manifest)
    if len(rows) > budget:
        raise ValueError(
            "new-recorded task exceeds the frozen Gold budget {}: {}".format(
                budget, len(rows)
            )
        )
    return len(rows), {
        "status": "ok",
        "source_id": source_id,
        "count": len(rows),
        "descriptor_sha256": sha256_file(descriptor),
        "manifest_sha256": sha256_file(manifest),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--round-id", required=True)
    parser.add_argument("--gold-budget", required=True, type=int)
    parser.add_argument("--new-recorded-source-id", default="")
    args = parser.parse_args()
    gold_budget = _validate_gold_budget(args.gold_budget)
    if not args.round_id or Path(args.round_id).name != args.round_id:
        raise ValueError("round-id must be one safe path component")
    config = load_config(args.config)
    workspace = resolve_path(str((config.get("output") or {}).get("workspace") or ""), config)
    source_id = "disagreement_gold_{}".format(args.round_id)
    output = workspace / "mining" / "rounds" / args.round_id / source_id
    request_path = output / "selection_request.jsonl"
    if output.exists():
        raise FileExistsError("finetune round is immutable and already exists: {}".format(output))

    new_count, new_report = _new_recorded_count(
        workspace, args.new_recorded_source_id, gold_budget
    )
    disagreement_budget = gold_budget - new_count
    score_path = workspace / "mining" / "teacher_student" / "disagreement_scores.jsonl"
    if not score_path.is_file():
        raise FileNotFoundError(
            "disagreement score pool is missing; run make prepare-finetune-sources first: {}".format(
                score_path
            )
        )
    scored = [clean_row(row) for row in read_jsonl(score_path)]
    predictions = [clean_row(row.get("student_prediction") or {}) for row in scored]
    occupied, occupied_report = _occupied(workspace, config, request_path)
    selection = dict(((config.get("selection") or {}).get("teacher_student") or {}))
    selection.update(
        {
            "enabled": True,
            "max_items": disagreement_budget,
            "per_dataset_max": None,
            "salt": "{}:{}".format(selection.get("salt", "disagreement"), args.round_id),
        }
    )
    requests, selector_report, ranked = select_teacher_student(
        scored,
        predictions,
        selection,
        occupied_identity_tokens=occupied,
    )
    if len(requests) > disagreement_budget:
        raise RuntimeError("selector exceeded disagreement budget")
    combined = new_count + len(requests)
    if combined > gold_budget or combined > HARD_GOLD_LIMIT:
        raise RuntimeError("combined CVAT task count exceeds frozen budget")
    output.mkdir(parents=True)
    write_jsonl(request_path, requests)
    write_jsonl(output / "ranked_eligible.jsonl", ranked)
    report = {
        "schema_version": "finetune_round_v1",
        "status": "ok",
        "round_id": args.round_id,
        "source_id": source_id,
        "frozen_gold_budget": gold_budget,
        "hard_limit": HARD_GOLD_LIMIT,
        "new_recorded": new_report,
        "disagreement_requested": disagreement_budget,
        "disagreement_selected": len(requests),
        "combined_task_count": combined,
        "occupied": occupied_report,
        "selector": selector_report,
        "selected_by_dataset": dict(sorted(Counter(row["dataset_id"] for row in requests).items())),
        "artifacts": {
            "selection_request": {
                "path": str(request_path),
                "sha256": sha256_file(request_path),
                "rows": len(requests),
            },
            "score_pool": {"path": str(score_path), "sha256": sha256_file(score_path)},
        },
    }
    write_json(output / "selection_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
