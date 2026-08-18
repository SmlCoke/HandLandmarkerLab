#!/usr/bin/env python3
"""HLML 4.0 high-level warehouse, training, mining, evaluation and export CLI."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hand_landmarker.config import load_config
from hand_landmarker.doctor import environment_report
from hand_landmarker.evaluation import evaluate_from_config
from hand_landmarker.export import export_from_config
from hand_landmarker.hard_mining import mine_hard_sources
from hand_landmarker.release import freeze_winner, locked_test_config
from hand_landmarker.runtime import KerasHandPredictor
from hand_landmarker.training import train_from_config
from hand_landmarker.visualization import infer_folder_from_config
from hand_landmarker.warehouse import build_snapshot, stage_paths


ROOT = Path(__file__).resolve().parents[1]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets-config", default=str(ROOT / "configs" / "datasets.yaml"))
    parser.add_argument("--training-config", default=str(ROOT / "configs" / "training.yaml"))
    parser.add_argument("--evaluation-config", default=str(ROOT / "configs" / "evaluation.yaml"))
    parser.add_argument("--inference-config", default=str(ROOT / "configs" / "inference.yaml"))
    parser.add_argument("--deploy-config", default=str(ROOT / "configs" / "deploy.yaml"))
    parser.add_argument(
        "--dataset-root",
        default=os.environ.get("HAND_DATASET_ROOT", "/root/autodl-tmp/DatesetFab"),
    )
    parser.add_argument(
        "--train-root",
        default=os.environ.get("HAND_TRAIN_ROOT", "/root/autodl-tmp/TrainFab/HLML-4.0"),
    )
    parser.add_argument("--snapshot-id", default=os.environ.get("HLML_SNAPSHOT_ID", "iris-v3-data-r1"))
    sub = parser.add_subparsers(dest="command", required=True)

    audit = sub.add_parser("data-audit", help="Build a zero-copy audited stage snapshot")
    audit.add_argument("--stage", choices=["geometry", "multitask", "multi_finetune"], required=True)
    audit.add_argument("--overwrite", action="store_true")

    train = sub.add_parser("train", help="Train one of the three fixed stages")
    train.add_argument("--stage", choices=["geometry", "multitask", "multi_finetune"], required=True)

    mine = sub.add_parser("mine-hard", help="Mine Train-only difficult capture sources")
    mine.add_argument("--predictions", help="Optional precomputed student prediction JSONL")
    mine.add_argument("--checkpoint", help="Multitask checkpoint when predictions are not provided")
    mine.add_argument("--output-dir")
    mine.add_argument("--batch-size", type=int, default=64)
    mine.add_argument("--round-id", required=True)
    mine.add_argument("--max-rois", type=int, required=True)

    sub.add_parser("eval-val", help="Evaluate the configured model on fixed reviewed Val ROIs")
    freeze = sub.add_parser("freeze-winner", help="Freeze the single Val-selected winner")
    freeze.add_argument("--release-id", default=os.environ.get("HLML_RELEASE_ID", "iris-v3-pro-r1"))
    freeze.add_argument("--metrics")
    freeze.add_argument("--checkpoint")
    freeze.add_argument("--stage", choices=["geometry", "multitask", "multi_finetune"], required=True)

    test = sub.add_parser("eval-test", help="Run the frozen winner once on locked Test")
    test.add_argument("--release-id", default=os.environ.get("HLML_RELEASE_ID", "iris-v3-pro-r1"))
    infer = sub.add_parser("infer", help="Run folder inference (Palm plus Hand)")
    infer.add_argument(
        "--palm-model-id",
        help="Temporarily override palm.model_id for this inference run",
    )
    sub.add_parser("export", help="Export and audit ONNX plus conversion datasets for A1")
    preflight = sub.add_parser(
        "export-preflight",
        help="Export an untrained selected architecture plus A1 conversion datasets",
    )
    preflight.add_argument("--output-dir")
    sub.add_parser("environment-check", help="Check the training environment")
    sub.add_parser("config-check", help="Parse every public configuration profile")
    return parser


def _set_roots(args: argparse.Namespace) -> None:
    os.environ["HAND_DATASET_ROOT"] = str(Path(args.dataset_root).resolve())
    os.environ["HAND_TRAIN_ROOT"] = str(Path(args.train_root).resolve())
    os.environ["HLML_SNAPSHOT_ID"] = str(args.snapshot_id)


def _training_config(path: str, stage: str) -> Dict[str, Any]:
    os.environ["HLML_STAGE"] = stage
    return load_config(path)


def _runtime_config(path: str) -> Dict[str, Any]:
    stage = os.environ.get("HLML_STAGE", "geometry")
    os.environ["HLML_CHECKPOINT_STAGE"] = (
        "finetune" if stage == "multi_finetune" else "pretrain"
    )
    return load_config(path)


def _evaluation_config(path: str, split: str) -> Dict[str, Any]:
    os.environ["HLML_EVALUATION_SPLIT"] = split
    return _runtime_config(path)


def _run(args: argparse.Namespace) -> Dict[str, Any]:
    _set_roots(args)
    if args.command == "data-audit":
        return build_snapshot(
            args.datasets_config,
            Path(args.dataset_root),
            Path(args.train_root),
            args.snapshot_id,
            args.stage,
            overwrite=args.overwrite,
        )
    if args.command == "train":
        paths = stage_paths(Path(args.train_root), args.snapshot_id, args.stage)
        if not paths["manifest"].is_file():
            raise FileNotFoundError("run data-audit before training: {}".format(paths["manifest"]))
        return train_from_config(_training_config(args.training_config, args.stage))
    if args.command == "mine-hard":
        paths = stage_paths(Path(args.train_root), args.snapshot_id, "multitask")
        output = Path(args.output_dir) if args.output_dir else (
            Path(args.train_root) / "mining" / args.snapshot_id
        )
        predictor = None
        if not args.predictions:
            config = _training_config(args.training_config, "multitask")
            checkpoint = Path(
                args.checkpoint
                or Path(args.train_root)
                / "runs"
                / os.environ.get("HLML_EXPERIMENT_ID", "iris-v3-pro-r1")
                / "multitask"
                / "checkpoints"
                / "best.weights.h5"
            )
            predictor = KerasHandPredictor(
                weights_path=str(checkpoint),
                model_version=str(config["model"].get("version", "v3-pro")),
                num_iterations=config["model"].get("num_iterations", [2, 2, 3, 4, 4, 6, 6]),
            )
        return mine_hard_sources(
            paths["train"],
            output,
            snapshot_id=args.snapshot_id,
            round_id=args.round_id,
            max_rois=args.max_rois,
            predictions_path=Path(args.predictions) if args.predictions else None,
            predictor=predictor,
            batch_size=args.batch_size,
        )
    if args.command == "eval-val":
        return evaluate_from_config(_evaluation_config(args.evaluation_config, "val"))
    if args.command == "freeze-winner":
        os.environ["HLML_STAGE"] = args.stage
        metrics = Path(args.metrics) if args.metrics else (
            Path(args.train_root)
            / "runs"
            / os.environ.get("HLML_EXPERIMENT_ID", "iris-v3-pro-r1")
            / "eval"
            / args.stage
            / "val"
            / "metrics.json"
        )
        checkpoint = Path(args.checkpoint) if args.checkpoint else (
            Path(args.train_root)
            / "runs"
            / os.environ.get("HLML_EXPERIMENT_ID", "iris-v3-pro-r1")
            / args.stage
            / "checkpoints"
            / "best.weights.h5"
        )
        return freeze_winner(
            Path(args.train_root),
            args.release_id,
            metrics,
            checkpoint,
            "finetune" if args.stage == "multi_finetune" else "pretrain",
            args.snapshot_id,
        )
    if args.command == "eval-test":
        os.environ["HLML_RELEASE_ID"] = args.release_id
        config = _evaluation_config(args.evaluation_config, "test")
        return evaluate_from_config(locked_test_config(config, Path(args.train_root), args.release_id))
    if args.command == "infer":
        config = _runtime_config(args.inference_config)
        if args.palm_model_id:
            config.setdefault("palm", {})["model_id"] = args.palm_model_id
            config.setdefault("palm", {}).pop("model_path", None)
        return infer_folder_from_config(config)
    if args.command == "export-preflight":
        config = _runtime_config(args.deploy_config)
        model_version = str(config.get("model", {}).get("version", "v3-pro"))
        output_root = (
            Path(args.output_dir).resolve()
            if args.output_dir
            else Path(args.train_root).resolve()
            / "preflight"
            / model_version
            / os.environ.get("HLML_STAGE", "geometry")
        )
        weights_path = output_root / "untrained-{}.weights.h5".format(model_version)
        export_path = output_root / "hand_landmarker_{}_untrained.onnx".format(
            model_version
        )
        config.setdefault("hand", {}).update(
            {"backend": "keras", "model_path": str(weights_path)}
        )
        export_config = config.setdefault("export", {})
        export_config["model_path"] = str(export_path)
        export_config["preflight_untrained"] = True
        export_config["overwrite"] = True
        export_config.setdefault("conversion_datasets", {})["output_dir"] = str(
            output_root / "model_conversion"
        )
        from scripts.build_export_preflight import build_preflight_bundle

        return build_preflight_bundle(config)
    if args.command == "export":
        return export_from_config(_runtime_config(args.deploy_config))
    if args.command == "environment-check":
        return environment_report(_training_config(args.training_config, "geometry"))
    parsed = {"datasets": load_config(args.datasets_config).get("contract")}
    parsed["training_profiles"] = {}
    for stage in ("geometry", "multitask", "multi_finetune"):
        parsed["training_profiles"][stage] = _training_config(args.training_config, stage)["resolved_profile"]
    parsed["evaluation_profiles"] = {}
    for split in ("val", "test"):
        parsed["evaluation_profiles"][split] = _evaluation_config(args.evaluation_config, split)["resolved_profile"]
    parsed["inference_task"] = _runtime_config(args.inference_config).get("task")
    parsed["deploy_task"] = _runtime_config(args.deploy_config).get("task")
    return {"status": "ok", **parsed}


def main() -> None:
    args = _parser().parse_args()
    try:
        result = _run(args)
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        print("ERROR: {}".format(exc), file=sys.stderr)
        raise SystemExit(2) from exc
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
