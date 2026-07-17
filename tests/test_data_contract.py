from __future__ import annotations

import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from hand_landmarker.config import load_config
from hand_landmarker.data import (
    CanonicalSequence,
    WeightedStratifiedSampler,
    augment_image_and_targets,
    create_sequences,
)
from hand_landmarker.inspect import DatasetContractError, audit_canonical_dataset, inspect_config


SAMPLE_TYPE_FRACTIONS = {
    "POS_RUNTIME": 0.56,
    "POS_LOW_PALM": 0.14,
    "NEG_RUNTIME_CANDIDATE": 0.25,
    "NEG_LOW_PALM_CANDIDATE": 0.05,
}
POS_ONLY_FRACTIONS = {
    "POS_RUNTIME": 1.0,
    "POS_LOW_PALM": 0.0,
    "NEG_RUNTIME_CANDIDATE": 0.0,
    "NEG_LOW_PALM_CANDIDATE": 0.0,
}
POS_NEG_RUNTIME_FRACTIONS = {
    "POS_RUNTIME": 0.5,
    "POS_LOW_PALM": 0.0,
    "NEG_RUNTIME_CANDIDATE": 0.5,
    "NEG_LOW_PALM_CANDIDATE": 0.0,
}


def _sampling_config(fractions=None, **overrides):
    value = {
        "enabled": True,
        "stratify_by": ["supervision_tier", "sample_type"],
        "tier_key": "supervision_tier",
        "bucket_key": "sampling_bucket",
        "sample_type_key": "sample_type",
        "weight_key": "sampling_weight",
        "sample_type_fractions": dict(fractions or SAMPLE_TYPE_FRACTIONS),
        "quota_rounding": "largest_remainder",
        "quota_tie_break": list(SAMPLE_TYPE_FRACTIONS),
        "require_all_tier_sample_type_cells": True,
        "replacement": True,
        "honor_record_sampling_weight": True,
    }
    value.update(overrides)
    return value


def _points_norm():
    return [
        {"id": index, "x": 0.2 + index * 0.02, "y": 0.3 + index * 0.01}
        for index in range(21)
    ]


def _train_row(
    name,
    crop_path,
    present=True,
    tier="pseudo",
    stage="pretrain",
    sampling_weight=0.25,
    sample_type=None,
):
    norm = _points_norm() if present else []
    crop = [
        {"id": point["id"], "x": point["x"] * 255.0, "y": point["y"] * 255.0}
        for point in norm
    ]
    image = [
        {"id": point["id"], "x": point["x"] * 255.0, "y": point["y"] * 255.0}
        for point in norm
    ]
    sample_type = sample_type or ("POS_RUNTIME" if present else "NEG_RUNTIME_CANDIDATE")
    if bool(sample_type.startswith("POS_")) != bool(present):
        raise ValueError("sample_type presence does not match present")
    palm_valid = sample_type in {"POS_RUNTIME", "NEG_RUNTIME_CANDIDATE"}
    provenance = "human_gold" if tier == "gold" else "mediapipe_pseudo"
    return {
        "schema_version": "train_finalize_v1",
        "dataset_id": "unit_train_v1",
        "source_crop_id": name + ":palm0:crop",
        "global_crop_id": "unit_train_v1:" + name + ":palm0:crop",
        "crop_id": "unit_train_v1:" + name + ":palm0:crop",
        "source_group_id": "unit_train_v1:" + name + ".tiff",
        "image": name + ".tiff",
        "crop_path": str(crop_path),
        "source_crop_path": "Z:/stale/" + Path(crop_path).name,
        "width": 256,
        "height": 256,
        "palm_det_id": name + ":palm0",
        "palm_valid": palm_valid,
        "palm_score": 0.8,
        "roi_rect": {
            "x_center": 128.0,
            "y_center": 128.0,
            "width": 255.0,
            "height": 255.0,
            "rotation_rad": 0.0,
        },
        "roi_corners_px": [[0.0, 0.0], [255.0, 0.0], [255.0, 255.0], [0.0, 255.0]],
        "hand_id": (name + ":hand") if present else None,
        "hand_presence": {"present": present},
        "handedness": {"label": "Left" if present else "unknown", "score": 0.9 if present else None},
        "landmarks_crop_norm": norm,
        "landmarks_crop_px": crop,
        "landmarks_image_px": image,
        "ignore_for_training": False,
        "annotation_provenance": provenance,
        "supervision_tier": tier,
        "training_stage": stage,
        "sample_type": sample_type,
        "quality_tier": "HIGH",
        "quality_flags": [],
        "selection_action": "include",
        "sampling_bucket": tier + ":" + sample_type,
        "sampling_weight": sampling_weight,
        "hand_presence_loss_weight": 1.0,
        "landmark_loss_weight": 1.0 if present else 0.0,
        "handedness_loss_weight": 1.0 if present else 0.0,
        "supervision_loss_weight": 1.0 if tier == "gold" else 0.7,
        "presence_quality_weight": 1.0,
        "landmark_quality_weight": 1.0 if present else 0.0,
        "handedness_quality_weight": 1.0 if present else 0.0,
    }


def _evaluation_row(name, crop_path, split="val"):
    row = _train_row(name, crop_path, present=True, tier="gold")
    dataset_id = "unit_{}_v1".format(split)
    row.update(
        {
            "schema_version": "evaluation_gold_v1",
            "dataset_id": dataset_id,
            "source_crop_id": name + ":palm0:crop",
            "global_crop_id": dataset_id + ":" + name + ":palm0:crop",
            "crop_id": dataset_id + ":" + name + ":palm0:crop",
            "source_group_id": dataset_id + ":" + name + ".tiff",
            "split": split,
            "ground_truth_valid": True,
            "annotation_provenance": "human_gold",
            "supervision_tier": "gold",
        }
    )
    for key in (
        "training_stage",
        "sample_type",
        "quality_tier",
        "quality_flags",
        "selection_action",
        "sampling_bucket",
        "sampling_weight",
        "supervision_loss_weight",
        "presence_quality_weight",
        "landmark_quality_weight",
        "handedness_quality_weight",
    ):
        row.pop(key, None)
    return row


def _write_jsonl(path, rows):
    with Path(path).open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")


def _write_image(path, value=127):
    Image.fromarray(np.full((256, 256), value, dtype=np.uint8), mode="L").save(str(path))


def _dataset_config(labels, data_root=None):
    value = {
        "labels": str(labels),
        "crop_path_key": "crop_path",
        "source_crop_path_key": "source_crop_path",
        "path_policy": "canonical_crop_path_only",
        "require_schema_version": "train_finalize_v1",
        "require_training_stage": "pretrain",
        "image_size": [256, 256],
        "channels": 1,
        "input_layout": "NCHW",
        "input_scale": 1.0 / 255.0,
        "input_offset": 0.0,
        "sampling_weight_key": "sampling_weight",
    }
    if data_root is not None:
        value["data_root"] = str(data_root)
    return value


class CanonicalDataContractTests(unittest.TestCase):
    def test_training_inspection_configs_compare_only_locked_test(self):
        repo_root = Path(__file__).resolve().parents[1]
        for filename in ("train_geometry.yaml", "train_multitask.yaml"):
            with self.subTest(config=filename):
                config = load_config(repo_root / "configs" / filename)
                comparisons = config["inspection"]["compare_datasets"]
                self.assertEqual({"test"}, set(comparisons))
                self.assertEqual("evaluation_gold_v1", comparisons["test"]["require_schema_version"])
                self.assertEqual("test", comparisons["test"]["require_split"])
                self.assertIn("ignored_labels", comparisons["test"])
                self.assertIn("ignored_labels", config["validation"])

    def test_create_sequences_reports_label_hashes_and_contract(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            train_image = root / "train.bmp"
            val_image = root / "val.bmp"
            _write_image(train_image, 80)
            _write_image(val_image, 120)
            train_labels = root / "train.jsonl"
            val_labels = root / "val.jsonl"
            _write_jsonl(train_labels, [_train_row("train", train_image)])
            _write_jsonl(val_labels, [_evaluation_row("val", val_image)])
            dataset = _dataset_config(train_labels, root)
            dataset.update({"input_dtype": "float32", "color_mode": "grayscale"})
            config = {
                "task": "train",
                "stage": "pretrain",
                "experiment": {"seed": 9},
                "data": dataset,
                "model": {"output_order": ["landmarks", "hand_flag", "handedness"]},
                "targets": {
                    "num_landmarks": 21,
                    "landmark_field": "landmarks_crop_norm",
                    "presence_field": "hand_presence.present",
                    "handedness_field": "handedness.label",
                    "handedness_encoding": {"Left": 0, "Right": 1, "unknown": None},
                },
                "training": {"batch_size": 1, "steps_per_epoch": None},
                "sampling": _sampling_config(POS_ONLY_FRACTIONS),
                "augmentation": {"enabled": False},
                "validation": {"enabled": True, "labels": str(val_labels), "batch_size": 1},
            }
            train_sequence, val_sequence, report = create_sequences(config)
            self.assertEqual(1, len(train_sequence))
            self.assertIsNotNone(val_sequence)
            self.assertEqual("ok", report["status"])
            self.assertEqual(64, len(report["train"]["labels_sha256"]))
            self.assertEqual(64, len(report["validation"]["labels_sha256"]))
            self.assertEqual("ok", report["leakage"][0]["status"])
            self.assertEqual([None, 42], report["tensor_contract"]["targets"][0])

    def test_source_crop_path_is_never_a_fallback(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            actual = root / "crop.bmp"
            _write_image(actual)
            labels = root / "labels.jsonl"
            row = _train_row("a", root / "missing" / "crop.bmp")
            row["source_crop_path"] = str(actual)
            _write_jsonl(labels, [row])
            config = {"data": _dataset_config(labels), "stage": "pretrain", "task": "train"}
            samples, report = audit_canonical_dataset(
                config,
                expected_stage="pretrain",
                check_images=True,
                raise_on_error=False,
            )
            self.assertEqual([], samples)
            self.assertEqual("failed", report["status"])

            config["data"]["data_root"] = str(root)
            samples, report = audit_canonical_dataset(
                config,
                expected_stage="pretrain",
                check_images=True,
                raise_on_error=True,
            )
            self.assertEqual(1, len(samples))
            self.assertEqual(str(actual.resolve()), samples[0]["_resolved_crop_path"])
            self.assertEqual("rebased", samples[0]["_path_resolution"])

    def test_allowed_crop_roots_apply_to_direct_canonical_paths(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            allowed = root / "allowed"
            outside = root / "outside"
            allowed.mkdir()
            outside.mkdir()
            image = outside / "crop.bmp"
            _write_image(image)
            labels = root / "labels.jsonl"
            _write_jsonl(labels, [_train_row("outside", image)])
            dataset = _dataset_config(labels)
            dataset.update(
                {
                    "crop_image_roots": [str(allowed)],
                    "allowed_crop_roots": [str(allowed)],
                }
            )

            samples, report = audit_canonical_dataset(
                {"data": dataset, "stage": "pretrain", "task": "train"},
                expected_stage="pretrain",
                check_images=True,
                raise_on_error=False,
            )
            self.assertEqual([], samples)
            self.assertEqual("failed", report["status"])
            self.assertIn("escapes allowed_crop_roots", " ".join(report["errors"][0]["errors"]))

    def test_allowed_crop_roots_reject_symlinked_crop(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            allowed = root / "allowed"
            outside = root / "outside"
            allowed.mkdir()
            outside.mkdir()
            target = outside / "crop.bmp"
            _write_image(target)
            linked = allowed / "crop.bmp"
            try:
                linked.symlink_to(target)
            except (OSError, NotImplementedError) as exc:
                self.skipTest("File symlinks are unavailable: {}".format(exc))
            labels = root / "labels.jsonl"
            _write_jsonl(labels, [_train_row("linked", linked)])
            dataset = _dataset_config(labels)
            dataset.update(
                {
                    "crop_image_roots": [str(allowed)],
                    "allowed_crop_roots": [str(allowed)],
                }
            )

            samples, report = audit_canonical_dataset(
                {"data": dataset, "stage": "pretrain", "task": "train"},
                expected_stage="pretrain",
                check_images=True,
                raise_on_error=False,
            )
            self.assertEqual([], samples)
            self.assertEqual("failed", report["status"])
            self.assertIn("symlink component", " ".join(report["errors"][0]["errors"]))

    def test_validation_roots_override_train_roots(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            train_root = root / "train"
            eval_root = root / "eval"
            train_root.mkdir()
            eval_root.mkdir()
            train_image = train_root / "train.bmp"
            val_image = eval_root / "val.bmp"
            _write_image(train_image, 40)
            _write_image(val_image, 90)
            train_labels = root / "train.jsonl"
            val_labels = root / "val.jsonl"
            _write_jsonl(train_labels, [_train_row("train", train_image)])
            _write_jsonl(val_labels, [_evaluation_row("val", val_image)])
            dataset = _dataset_config(train_labels)
            dataset.update(
                {
                    "data_root": str(train_root),
                    "crop_image_roots": [str(train_root)],
                    "allowed_crop_roots": [str(train_root)],
                }
            )
            config = {
                "task": "train",
                "stage": "pretrain",
                "data": dataset,
                "validation": {
                    "enabled": True,
                    "data_root": str(eval_root),
                    "labels": str(val_labels),
                    "crop_image_roots": [str(eval_root)],
                    "allowed_crop_roots": [str(eval_root)],
                },
            }

            report = inspect_config(config, check_images=True, hash_images=True)
            self.assertEqual("ok", report["datasets"]["primary"]["status"])
            self.assertEqual("ok", report["datasets"]["validation"]["status"])

    def test_stage_schema_and_weights_fail_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            image = root / "crop.bmp"
            _write_image(image)
            labels = root / "labels.jsonl"
            row = _train_row("bad", image)
            row["schema_version"] = "wrong"
            row["training_stage"] = "finetune"
            row["sampling_weight"] = -1.0
            _write_jsonl(labels, [row])
            config = {"data": _dataset_config(labels), "stage": "pretrain", "task": "train"}
            _, report = audit_canonical_dataset(
                config,
                expected_stage="pretrain",
                check_images=False,
                raise_on_error=False,
            )
            messages = " ".join(report["errors"][0]["errors"])
            self.assertIn("schema_version mismatch", messages)
            self.assertIn("training_stage mismatch", messages)
            self.assertIn("sampling_weight must be finite and non-negative", messages)

    def test_sequence_shapes_and_effective_weights(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            positive_image = root / "positive.bmp"
            negative_image = root / "negative.bmp"
            _write_image(positive_image, 64)
            _write_image(negative_image, 192)
            labels = root / "labels.jsonl"
            rows = [
                _train_row("positive", positive_image, present=True, sampling_weight=0.01),
                _train_row("negative", negative_image, present=False, sampling_weight=0.99),
            ]
            _write_jsonl(labels, rows)
            dataset = _dataset_config(labels)
            config = {"data": dataset, "stage": "pretrain", "task": "train"}
            samples, _ = audit_canonical_dataset(
                config,
                expected_stage="pretrain",
                check_images=True,
                raise_on_error=True,
            )
            sequence = CanonicalSequence(
                samples,
                dataset,
                {
                    "landmark_field": "landmarks_crop_norm",
                    "handedness_encoding": {"Left": 0, "Right": 1, "unknown": None},
                },
                batch_size=2,
                training=True,
                stage="pretrain",
                seed=3,
                augmentation_config={"enabled": False},
                sampling_config=_sampling_config(POS_NEG_RUNTIME_FRACTIONS),
            )
            x_value, targets, weights = sequence[0]
            self.assertEqual((2, 1, 256, 256), x_value.shape)
            self.assertEqual((2, 42), targets[0].shape)
            self.assertEqual((2, 1), targets[1].shape)
            self.assertEqual((2, 1), targets[2].shape)
            self.assertEqual(
                {"landmarks", "hand_flag", "handedness", "structure"}, set(weights)
            )
            self.assertEqual((2,), weights["structure"].shape)
            for position, record_index in enumerate(sequence.batch_record_indices(0)):
                row = samples[int(record_index)]
                # sampling_weight affects selection only; effective presence is 0.7 for all pseudo rows.
                self.assertAlmostEqual(0.7, float(weights["hand_flag"][position]), places=6)
                self.assertEqual(0.0, float(weights["structure"][position]))
                if not row["hand_presence"]["present"]:
                    self.assertEqual(0.0, float(weights["landmarks"][position]))
                    self.assertEqual(0.0, float(weights["handedness"][position]))

    def test_sequence_rejects_non_board_input_contract(self):
        row = _train_row("contract", "unused.bmp")
        targets = {
            "landmark_field": "landmarks_crop_norm",
            "handedness_encoding": {"Left": 0, "Right": 1, "unknown": None},
        }
        cases = (
            ({"input_scale": 1.0}, "input_scale must equal 1/255"),
            ({"input_offset": 1.0}, "input_offset must equal 0"),
            ({"image_size": [224, 224]}, "image_size must remain"),
        )
        for changes, message in cases:
            with self.subTest(changes=changes):
                dataset = _dataset_config("unused.jsonl")
                dataset.update(changes)
                with self.assertRaisesRegex(DatasetContractError, message):
                    CanonicalSequence(
                        [row],
                        dataset,
                        targets,
                        batch_size=1,
                        training=True,
                        stage="pretrain",
                        sampling_config=_sampling_config(POS_ONLY_FRACTIONS),
                    )

    def test_finetune_sampler_honors_exact_epoch_gold_fraction(self):
        records = []
        for index in range(3):
            records.append(
                {
                    "supervision_tier": "gold",
                    "sample_type": "POS_RUNTIME",
                    "sampling_bucket": "gold:POS_RUNTIME",
                    "sampling_weight": 1.0,
                }
            )
        for index in range(7):
            records.append(
                {
                    "supervision_tier": "pseudo",
                    "sample_type": "POS_RUNTIME",
                    "sampling_bucket": "pseudo:POS_RUNTIME",
                    "sampling_weight": 1.0,
                }
            )
        sampler = WeightedStratifiedSampler(
            records,
            "finetune",
            seed=11,
            gold_fraction=0.4,
            sample_type_fractions=POS_ONLY_FRACTIONS,
        )
        indices = sampler.sample(100, epoch=2)
        gold = sum(records[int(index)]["supervision_tier"] == "gold" for index in indices)
        self.assertEqual(40, gold)

    def test_pretrain_auto_epoch_size_uses_exact_batch_quota_and_row_weights(self):
        fractions = {
            "POS_RUNTIME": 0.72,
            "POS_LOW_PALM": 0.18,
            "NEG_RUNTIME_CANDIDATE": 0.08,
            "NEG_LOW_PALM_CANDIDATE": 0.02,
        }
        counts = {
            "POS_RUNTIME": 1600,
            "POS_LOW_PALM": 500,
            "NEG_RUNTIME_CANDIDATE": 128,
            "NEG_LOW_PALM_CANDIDATE": 894,
        }
        records = []
        for sample_type, count in counts.items():
            for index in range(count):
                records.append(
                    {
                        "crop_id": "{}:{}".format(sample_type, index),
                        "supervision_tier": "pseudo",
                        "sample_type": sample_type,
                        "sampling_bucket": "pseudo:{}".format(sample_type),
                        "sampling_weight": 0.25,
                    }
                )
        sampler = WeightedStratifiedSampler(
            records,
            "pretrain",
            seed=7,
            sample_type_fractions=fractions,
        )
        epoch_size, report = sampler.resolve_auto_epoch_size(
            batch_size=64,
            upper_bound=6400,
            max_average_draws=4.0,
            max_expected_row_draws=8.0,
        )
        self.assertEqual(6400, epoch_size)
        runtime = report["cell_reports"]["pseudo:NEG_RUNTIME_CANDIDATE"]
        self.assertEqual(500, runtime["draws"])
        self.assertAlmostEqual(3.90625, runtime["average_draws_per_unique_record"])
        self.assertAlmostEqual(3.90625, runtime["max_expected_row_draws"])
        self.assertEqual("pseudo:NEG_RUNTIME_CANDIDATE", report["limiting_cell"])

        one_runtime = [
            row
            for row in records
            if row["sample_type"] != "NEG_RUNTIME_CANDIDATE"
        ] + [
            {
                "crop_id": "only-runtime-negative",
                "supervision_tier": "pseudo",
                "sample_type": "NEG_RUNTIME_CANDIDATE",
                "sampling_bucket": "pseudo:NEG_RUNTIME_CANDIDATE",
                "sampling_weight": 1.0,
            }
        ]
        unsafe = WeightedStratifiedSampler(
            one_runtime,
            "pretrain",
            seed=7,
            sample_type_fractions=fractions,
        )
        with self.assertRaisesRegex(DatasetContractError, "No whole-batch epoch_size"):
            unsafe.resolve_auto_epoch_size(64, 6400, 4.0, 8.0)

    def test_finetune_sampler_enforces_exact_cross_bucket_quota(self):
        records = []
        for tier in ("gold", "pseudo"):
            for offset, sample_type in enumerate(SAMPLE_TYPE_FRACTIONS):
                records.append(
                    {
                        "crop_id": "{}:{}".format(tier, sample_type),
                        "supervision_tier": tier,
                        "sample_type": sample_type,
                        "sampling_bucket": "{}:{}".format(tier, sample_type),
                        # Deliberately unequal: weights cannot change cell quotas.
                        "sampling_weight": 10.0 ** (offset - 2),
                    }
                )
        sampler = WeightedStratifiedSampler(
            records,
            "finetune",
            seed=31,
            gold_fraction=0.4,
            sample_type_fractions=SAMPLE_TYPE_FRACTIONS,
        )
        self.assertEqual(
            {
                "gold": {
                    "POS_RUNTIME": 7,
                    "POS_LOW_PALM": 2,
                    "NEG_RUNTIME_CANDIDATE": 3,
                    "NEG_LOW_PALM_CANDIDATE": 1,
                },
                "pseudo": {
                    "POS_RUNTIME": 11,
                    "POS_LOW_PALM": 2,
                    "NEG_RUNTIME_CANDIDATE": 5,
                    "NEG_LOW_PALM_CANDIDATE": 1,
                },
            },
            sampler.batch_quota(32),
        )
        self.assertEqual(
            {
                "gold": {
                    "POS_RUNTIME": 15,
                    "POS_LOW_PALM": 4,
                    "NEG_RUNTIME_CANDIDATE": 6,
                    "NEG_LOW_PALM_CANDIDATE": 1,
                },
                "pseudo": {
                    "POS_RUNTIME": 21,
                    "POS_LOW_PALM": 5,
                    "NEG_RUNTIME_CANDIDATE": 10,
                    "NEG_LOW_PALM_CANDIDATE": 2,
                },
            },
            sampler.batch_quota(64),
        )
        self.assertEqual(
            {
                "gold": {
                    "POS_RUNTIME": 1,
                    "POS_LOW_PALM": 1,
                    "NEG_RUNTIME_CANDIDATE": 0,
                    "NEG_LOW_PALM_CANDIDATE": 0,
                },
                "pseudo": {
                    "POS_RUNTIME": 2,
                    "POS_LOW_PALM": 0,
                    "NEG_RUNTIME_CANDIDATE": 1,
                    "NEG_LOW_PALM_CANDIDATE": 0,
                },
            },
            sampler.batch_quota(5),
        )
        indices = sampler.sample(32, epoch=4)
        tier_counts = Counter(records[int(index)]["supervision_tier"] for index in indices)
        type_counts = Counter(records[int(index)]["sample_type"] for index in indices)
        self.assertEqual({"gold": 13, "pseudo": 19}, dict(tier_counts))
        self.assertEqual(
            {
                "POS_RUNTIME": 18,
                "POS_LOW_PALM": 4,
                "NEG_RUNTIME_CANDIDATE": 8,
                "NEG_LOW_PALM_CANDIDATE": 2,
            },
            dict(type_counts),
        )

        missing = [
            row
            for row in records
            if not (
                row["supervision_tier"] == "gold"
                and row["sample_type"] == "NEG_LOW_PALM_CANDIDATE"
            )
        ]
        with self.assertRaisesRegex(DatasetContractError, "Missing canonical sampling cell"):
            WeightedStratifiedSampler(
                missing,
                "finetune",
                seed=31,
                gold_fraction=0.4,
                sample_type_fractions=SAMPLE_TYPE_FRACTIONS,
            )

    def test_finetune_epoch_plan_redistributes_missing_gold_cells(self):
        records = []
        for tier, count in (("gold", 20), ("pseudo", 20)):
            for index in range(count):
                records.append(
                    {
                        "crop_id": "{}:{}".format(tier, index),
                        "supervision_tier": tier,
                        "sample_type": "POS_RUNTIME",
                        "sampling_bucket": "{}:POS_RUNTIME".format(tier),
                        "sampling_weight": 1.0,
                    }
                )
        sampler = WeightedStratifiedSampler(
            records,
            "finetune",
            seed=19,
            gold_fraction=0.4,
            sample_type_fractions_by_tier={
                "gold": {
                    "POS_RUNTIME": 0.70,
                    "POS_LOW_PALM": 0.20,
                    "NEG_RUNTIME_CANDIDATE": 0.07,
                    "NEG_LOW_PALM_CANDIDATE": 0.03,
                },
                "pseudo": POS_ONLY_FRACTIONS,
            },
            missing_cell_policy={"gold": "redistribute_within_tier", "pseudo": "fail"},
        )
        indices = sampler.sample_epoch([10] * 10, epoch=3)
        self.assertEqual(100, len(indices))
        self.assertEqual(
            {"gold": 40, "pseudo": 60},
            dict(Counter(records[int(index)]["supervision_tier"] for index in indices)),
        )
        plan = sampler.last_epoch_plan
        self.assertIsNotNone(plan)
        self.assertEqual(40, plan["epoch_draw_quota_by_tier_type"]["gold"]["POS_RUNTIME"])
        self.assertEqual(
            ["POS_LOW_PALM", "NEG_RUNTIME_CANDIDATE", "NEG_LOW_PALM_CANDIDATE"],
            plan["configured_and_effective_fractions"]["gold"]["missing"],
        )
        self.assertTrue(
            all(batch.get("gold:POS_RUNTIME") == 4 for batch in plan["batch_cell_quotas"])
        )

    def test_finetune_epoch_plan_caps_rare_gold_negative_across_batches(self):
        records = []
        for tier in ("gold", "pseudo"):
            for index in range(20):
                records.append(
                    {
                        "crop_id": "{}:pos:{}".format(tier, index),
                        "supervision_tier": tier,
                        "sample_type": "POS_RUNTIME",
                        "sampling_bucket": "{}:POS_RUNTIME".format(tier),
                        "sampling_weight": 1.0,
                    }
                )
        records.append(
            {
                "crop_id": "gold:rare-negative",
                "supervision_tier": "gold",
                "sample_type": "NEG_RUNTIME_CANDIDATE",
                "sampling_bucket": "gold:NEG_RUNTIME_CANDIDATE",
                "sampling_weight": 1.0,
            }
        )
        sampler = WeightedStratifiedSampler(
            records,
            "finetune",
            seed=23,
            gold_fraction=0.5,
            sample_type_fractions_by_tier={
                "gold": {
                    "POS_RUNTIME": 0.80,
                    "POS_LOW_PALM": 0.0,
                    "NEG_RUNTIME_CANDIDATE": 0.20,
                    "NEG_LOW_PALM_CANDIDATE": 0.0,
                },
                "pseudo": POS_ONLY_FRACTIONS,
            },
            missing_cell_policy={"gold": "redistribute_within_tier", "pseudo": "fail"},
            rare_cell_policy={
                "gold": "cap_fraction_then_redistribute_within_tier",
                "pseudo": "fail",
                "max_average_draws_per_unique_record": 4.0,
                "max_expected_row_draws_per_epoch": 8.0,
            },
        )
        indices = sampler.sample_epoch([10] * 10, epoch=0)
        plan = sampler.last_epoch_plan
        self.assertEqual(100, len(indices))
        self.assertEqual(
            4,
            plan["epoch_draw_quota_by_tier_type"]["gold"]["NEG_RUNTIME_CANDIDATE"],
        )
        negative_batches = sum(
            batch.get("gold:NEG_RUNTIME_CANDIDATE", 0) > 0
            for batch in plan["batch_cell_quotas"]
        )
        self.assertEqual(4, negative_batches)
        self.assertEqual(
            6,
            plan["rare_cell_quota_cap"]["gold"]["redistributed_draws"],
        )

    def test_sampler_rejects_pretrain_gold_and_non_finite_fractions(self):
        pseudo = {
            "crop_id": "pseudo",
            "supervision_tier": "pseudo",
            "sample_type": "POS_RUNTIME",
            "sampling_bucket": "pseudo:POS_RUNTIME",
            "sampling_weight": 1.0,
        }
        gold = dict(pseudo)
        gold.update(
            crop_id="gold",
            supervision_tier="gold",
            sampling_bucket="gold:POS_RUNTIME",
        )
        with self.assertRaisesRegex(DatasetContractError, "pseudo only"):
            WeightedStratifiedSampler(
                [pseudo, gold],
                "pretrain",
                seed=1,
                sample_type_fractions=POS_ONLY_FRACTIONS,
            )
        invalid = dict(POS_ONLY_FRACTIONS)
        invalid["POS_RUNTIME"] = float("nan")
        with self.assertRaisesRegex(DatasetContractError, "finite numeric values"):
            WeightedStratifiedSampler(
                [pseudo],
                "pretrain",
                seed=1,
                sample_type_fractions=invalid,
            )
        with self.assertRaisesRegex(DatasetContractError, r"within \[0.30, 0.50\]"):
            WeightedStratifiedSampler(
                [pseudo, gold],
                "finetune",
                seed=1,
                gold_fraction=0.8,
                sample_type_fractions=POS_ONLY_FRACTIONS,
            )

    def test_finetune_sequence_honors_gold_fraction_in_every_batch(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            image = root / "crop.bmp"
            _write_image(image)
            records = []
            for tier in ("gold", "gold", "pseudo", "pseudo"):
                row = _train_row(
                    "{}-{}".format(tier, len(records)),
                    image,
                    tier=tier,
                    stage="finetune",
                    sampling_weight=1.0,
                )
                row["_resolved_crop_path"] = str(image.resolve())
                records.append(row)

            sequence = CanonicalSequence(
                records,
                _dataset_config(root / "unused.jsonl"),
                {
                    "landmark_field": "landmarks_crop_norm",
                    "presence_field": "hand_presence.present",
                    "handedness_field": "handedness.label",
                    "handedness_encoding": {"Left": 0, "Right": 1, "unknown": None},
                },
                batch_size=10,
                training=True,
                stage="finetune",
                seed=17,
                augmentation_config={"enabled": False},
                sampling_config=_sampling_config(
                    POS_ONLY_FRACTIONS,
                    gold_fraction=0.4,
                    epoch_size=25,
                ),
            )

            gold_per_batch = []
            for batch_index in range(len(sequence)):
                indices = sequence.batch_record_indices(batch_index)
                gold_per_batch.append(
                    sum(records[int(index)]["supervision_tier"] == "gold" for index in indices)
                )
            self.assertEqual([4, 4, 2], gold_per_batch)
            report = sequence.sampling_report()
            self.assertEqual(
                "per_batch_half_up",
                report["definition"]["gold_fraction_scope"],
            )
            self.assertEqual(
                [{"gold": 4, "pseudo": 6}, {"gold": 4, "pseudo": 6}, {"gold": 2, "pseudo": 3}],
                report["drawn_supervision_tiers_per_batch"],
            )

    def test_set_epoch_is_absolute_reproducible_and_changes_random_stream(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            image = root / "crop.bmp"
            _write_image(image, 127)
            records = []
            for sample_type in SAMPLE_TYPE_FRACTIONS:
                for duplicate in range(2):
                    present = sample_type.startswith("POS_")
                    row = _train_row(
                        "{}-{}".format(sample_type, duplicate),
                        image,
                        present=present,
                        sample_type=sample_type,
                        sampling_weight=1.0 + duplicate,
                    )
                    row["_resolved_crop_path"] = str(image.resolve())
                    records.append(row)

            def make_sequence():
                return CanonicalSequence(
                    records,
                    _dataset_config(root / "unused.jsonl"),
                    {
                        "landmark_field": "landmarks_crop_norm",
                        "presence_field": "hand_presence.present",
                        "handedness_field": "handedness.label",
                        "handedness_encoding": {"Left": 0, "Right": 1, "unknown": None},
                    },
                    batch_size=8,
                    training=True,
                    stage="pretrain",
                    seed=41,
                    augmentation_config={
                        "enabled": True,
                        "rotation_degrees": 0.0,
                        "scale_range": [1.0, 1.0],
                        "translation_fraction": 0.0,
                        "horizontal_flip_probability": 0.0,
                        "contrast_range": [1.0, 1.0],
                        "brightness_delta": 0.0,
                        "gaussian_noise_stddev": 0.01,
                    },
                    sampling_config=_sampling_config(epoch_size=64),
                )

            first = make_sequence()
            epoch_zero_indices = first.epoch_indices.copy()
            epoch_zero_images = first[0][0]
            first.set_epoch(5)
            epoch_five_indices = first.epoch_indices.copy()
            epoch_five_images = first[0][0]
            first.set_epoch(5)
            self.assertTrue(np.array_equal(epoch_five_indices, first.epoch_indices))
            self.assertTrue(np.array_equal(epoch_five_images, first[0][0]))

            resumed = make_sequence()
            resumed.set_epoch(5)
            self.assertTrue(np.array_equal(epoch_five_indices, resumed.epoch_indices))
            self.assertTrue(np.array_equal(epoch_five_images, resumed[0][0]))
            self.assertFalse(np.array_equal(epoch_zero_indices, epoch_five_indices))
            self.assertFalse(np.array_equal(epoch_zero_images, epoch_five_images))

    def test_forced_flip_updates_x_and_handedness(self):
        image = np.tile(np.arange(256, dtype=np.float32), (256, 1)) / 255.0
        points = np.asarray([[0.2, 0.4]] * 21, dtype=np.float32)
        output, transformed, handedness = augment_image_and_targets(
            image,
            points,
            0.0,
            True,
            {
                "enabled": True,
                "rotation_degrees": 0.0,
                "scale_range": [1.0, 1.0],
                "translation_fraction": 0.0,
                "horizontal_flip_probability": 1.0,
                "contrast_range": [1.0, 1.0],
                "brightness_delta": 0.0,
                "gaussian_noise_stddev": 0.0,
            },
            np.random.RandomState(5),
        )
        self.assertTrue(np.allclose(0.8, transformed[:, 0]))
        self.assertAlmostEqual(1.0, handedness)
        self.assertTrue(np.allclose(image[:, ::-1], output))

    def test_dark_uint8_image_is_always_divided_by_255(self):
        image = np.asarray([[0, 1], [1, 0]], dtype=np.uint8)
        output, _, _ = augment_image_and_targets(
            image,
            np.zeros((21, 2), dtype=np.float32),
            0.0,
            False,
            {"enabled": False},
            np.random.RandomState(1),
        )
        self.assertEqual(np.float32(1.0 / 255.0), output[0, 1])
        self.assertLess(float(np.max(output)), 0.01)

    def test_inspect_detects_cross_split_content_leakage(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            train_image = root / "train.bmp"
            val_image = root / "val.bmp"
            _write_image(train_image, 33)
            val_image.write_bytes(train_image.read_bytes())
            train_labels = root / "train.jsonl"
            val_labels = root / "val.jsonl"
            _write_jsonl(train_labels, [_train_row("train", train_image)])
            _write_jsonl(val_labels, [_evaluation_row("val", val_image)])
            config = {
                "task": "train",
                "stage": "pretrain",
                "data": _dataset_config(train_labels, root),
                "validation": {"enabled": True, "labels": str(val_labels)},
            }
            report = inspect_config(config, check_images=True, hash_images=True)
            self.assertEqual("failed", report["status"])
            kinds = {
                item["kind"]
                for check in report["leakage"]
                for item in check["fatal"]
            }
            self.assertIn("image_sha256", kinds)

    def test_ignored_manifest_overlap_is_fatal_and_reported(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            image = root / "val.bmp"
            _write_image(image, 91)
            labels = root / "val.jsonl"
            ignored = root / "val_ignored.jsonl"
            included_row = _evaluation_row("val", image)
            _write_jsonl(labels, [included_row])
            # Cross-key overlap is also forbidden: an included global ID must
            # not reappear as the ignored row's crop ID.
            _write_jsonl(ignored, [{"crop_id": included_row["global_crop_id"]}])
            dataset = _dataset_config(labels, root)
            dataset.pop("require_training_stage")
            dataset.update(
                {
                    "ignored_labels": str(ignored),
                    "require_schema_version": "evaluation_gold_v1",
                    "require_split": "val",
                }
            )

            _, report = audit_canonical_dataset(
                {"data": dataset},
                dataset=dataset,
                expected_split="val",
                check_images=True,
                hash_images=True,
            )
            self.assertEqual("failed", report["status"])
            self.assertEqual(str(ignored.resolve()), report["ignored_labels"])
            self.assertEqual(1, report["ignored_records"])
            self.assertEqual(report["ignored_labels_sha256"], report["ignored"]["sha256"])
            self.assertEqual(1, report["ignored"]["count"])
            self.assertEqual(1, report["ignored"]["overlap_count"])
            self.assertEqual(
                [included_row["global_crop_id"]], report["ignored"]["overlap_ids"]
            )
            self.assertIn(
                "included_ignored_identity_overlap",
                {item.get("scope") for item in report["errors"]},
            )
            with self.assertRaisesRegex(DatasetContractError, "included and ignored"):
                audit_canonical_dataset(
                    {"data": dataset},
                    dataset=dataset,
                    expected_split="val",
                    check_images=True,
                    hash_images=True,
                    raise_on_error=True,
                )

            _write_jsonl(ignored, [{"global_crop_id": "ignored-only"}])
            _, disjoint_report = audit_canonical_dataset(
                {"data": dataset},
                dataset=dataset,
                expected_split="val",
                check_images=True,
                hash_images=True,
            )
            self.assertEqual("ok", disjoint_report["status"])
            self.assertEqual(1, disjoint_report["ignored"]["count"])
            self.assertEqual(0, disjoint_report["ignored"]["overlap_count"])

    def test_configured_test_is_strict_and_compared_with_train_and_val(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            train_image = root / "train.bmp"
            val_image = root / "val.bmp"
            test_image = root / "test.bmp"
            _write_image(train_image, 31)
            _write_image(val_image, 97)
            _write_image(test_image, 173)
            train_labels = root / "train.jsonl"
            val_labels = root / "val.jsonl"
            test_labels = root / "test.jsonl"
            val_ignored = root / "val_ignored.jsonl"
            test_ignored = root / "test_ignored.jsonl"
            train_row = _train_row("train", train_image)
            val_row = _evaluation_row("val", val_image)
            test_row = _evaluation_row("test", test_image, split="test")
            test_row["source_group_id"] = train_row["source_group_id"]
            _write_jsonl(train_labels, [train_row])
            _write_jsonl(val_labels, [val_row])
            _write_jsonl(test_labels, [test_row])
            _write_jsonl(val_ignored, [{"global_crop_id": "ignored-val-only"}])
            _write_jsonl(test_ignored, [{"global_crop_id": "ignored-test-only"}])

            config = {
                "task": "train",
                "stage": "pretrain",
                "data": _dataset_config(train_labels, root),
                "validation": {
                    "enabled": True,
                    "labels": str(val_labels),
                    "ignored_labels": str(val_ignored),
                },
                "inspection": {
                    "compare_datasets": {
                        "test": {
                            "data_root": str(root),
                            "labels": str(test_labels),
                            "ignored_labels": str(test_ignored),
                            "crop_path_key": "crop_path",
                            "path_policy": "canonical_crop_path_only",
                            "require_schema_version": "evaluation_gold_v1",
                            "require_split": "test",
                            "image_size": [256, 256],
                            "channels": 1,
                        }
                    }
                },
            }
            report = inspect_config(config, check_images=True, hash_images=True)
            self.assertEqual({"primary", "validation", "test"}, set(report["datasets"]))
            self.assertEqual("ok", report["datasets"]["test"]["status"])
            self.assertEqual(1, report["datasets"]["validation"]["ignored"]["count"])
            self.assertEqual(1, report["datasets"]["test"]["ignored"]["count"])
            self.assertEqual(
                {
                    ("primary", "validation"),
                    ("primary", "test"),
                    ("validation", "test"),
                },
                {(item["first"], item["second"]) for item in report["leakage"]},
            )
            primary_test = next(
                item
                for item in report["leakage"]
                if (item["first"], item["second"]) == ("primary", "test")
            )
            self.assertEqual("failed", primary_test["status"])
            self.assertIn("source_group_id", {item["kind"] for item in primary_test["fatal"]})

            # The configured comparison is not a loose --compare-labels set:
            # both Test split and evaluation schema remain mandatory.
            invalid_test = dict(test_row)
            invalid_test["split"] = "val"
            invalid_test["schema_version"] = "train_finalize_v1"
            _write_jsonl(test_labels, [invalid_test])
            strict_report = inspect_config(config, check_images=True, hash_images=True)
            self.assertIn("test", strict_report["failed_datasets"])
            strict_errors = "\n".join(
                error
                for item in strict_report["datasets"]["test"]["errors"]
                for error in item["errors"]
            )
            self.assertIn("schema_version mismatch", strict_errors)
            self.assertIn("split mismatch", strict_errors)

            cli_report = inspect_config(
                {
                    "task": "train",
                    "stage": "pretrain",
                    "data": _dataset_config(train_labels, root),
                },
                compare_labels=[str(test_labels)],
                check_images=True,
                hash_images=True,
            )
            self.assertIn("comparison_1", cli_report["datasets"])

    def test_create_sequences_rejects_cross_split_content_leakage(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            train_image = root / "train.bmp"
            val_image = root / "val.bmp"
            _write_image(train_image, 55)
            val_image.write_bytes(train_image.read_bytes())
            train_labels = root / "train.jsonl"
            val_labels = root / "val.jsonl"
            _write_jsonl(train_labels, [_train_row("train", train_image)])
            _write_jsonl(val_labels, [_evaluation_row("val", val_image)])
            data = _dataset_config(train_labels, root)
            data.update({"input_dtype": "float32", "color_mode": "grayscale"})
            config = {
                "task": "train",
                "stage": "pretrain",
                "experiment": {"seed": 3},
                "data": data,
                "model": {"output_order": ["landmarks", "hand_flag", "handedness"]},
                "targets": {
                    "num_landmarks": 21,
                    "landmark_field": "landmarks_crop_norm",
                    "presence_field": "hand_presence.present",
                    "handedness_field": "handedness.label",
                    "handedness_encoding": {"Left": 0, "Right": 1, "unknown": None},
                },
                "training": {"batch_size": 1},
                "sampling": _sampling_config(POS_ONLY_FRACTIONS),
                "augmentation": {"enabled": False},
                "validation": {"enabled": True, "labels": str(val_labels), "batch_size": 1},
            }
            with self.assertRaisesRegex(DatasetContractError, "image_sha256"):
                create_sequences(config)


if __name__ == "__main__":
    unittest.main()
