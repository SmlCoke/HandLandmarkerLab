from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from hand_landmarker.hard_mining import aggregate_sources
from hand_landmarker.evaluation import evaluate_hand_rois
from hand_landmarker.io_utils import read_jsonl, write_image, write_json, write_jsonl
from hand_landmarker.release import freeze_winner, locked_test_config
from hand_landmarker.warehouse import (
    WarehouseContractError,
    WarehouseReader,
    _assert_membership,
    build_snapshot,
)


def _points(offset: float = 0.0):
    return [
        {"id": index, "x": 0.15 + (index % 5) * 0.12 + offset, "y": 0.15 + (index // 5) * 0.12}
        for index in range(21)
    ]


class SyntheticWarehouse:
    variant = "eos-2.0"
    train_capture = "room-near-daylight-normal-train-s01-alice"
    val_capture = "room-near-daylight-normal-val-s01-bob"
    test_capture = "room-far-dim-normal-test-s01-carol"

    def __init__(self, root: Path):
        self.root = root
        self.registry = root / "Registry" / "registry.sqlite3"
        self.registry.parent.mkdir(parents=True)
        with closing(sqlite3.connect(str(self.registry))) as db:
            db.executescript(
                """
                CREATE TABLE rois(roi_id TEXT PRIMARY KEY,raw_image_id TEXT,capture_source_id TEXT,proposal_variant TEXT,proposal_slot INTEGER,crop_relpath TEXT);
                CREATE TABLE negative_datasets(negative_dataset_id TEXT PRIMARY KEY,status TEXT);
                CREATE TABLE published_negatives(roi_id TEXT PRIMARY KEY,negative_dataset_id TEXT,published_relpath TEXT);
                CREATE TABLE selections(selection_id TEXT PRIMARY KEY,status TEXT);
                """
            )
            db.commit()
        self.train_rows = [
            self._source_row("train-roi-1", "train-raw-1", self.train_capture, "PretrainSource/train-set", 70),
            self._source_row("train-roi-2", "train-raw-2", self.train_capture, "PretrainSource/train-set", 90),
        ]
        self.val_rows = [
            self._source_row("val-roi-1", "val-raw-1", self.val_capture, "EValSource/eval-set", 110)
        ]
        self.test_rows = [
            self._source_row("test-roi-1", "test-raw-1", self.test_capture, "EValSource/eval-set", 130)
        ]
        self._publish_dataset("pretrain", "train-set", [(self.train_capture, self.train_rows)])
        self._publish_dataset(
            "eval", "eval-set", [(self.val_capture, self.val_rows), (self.test_capture, self.test_rows)]
        )
        self._publish_negative()
        self._publish_selection()

    def _source_row(
        self,
        roi_id,
        raw_id,
        capture,
        dataset_parent,
        value,
        dataset_id=None,
        proposal_variant=None,
    ):
        split = capture.split("-")[4]
        dataset_id = dataset_id or ("train-set" if split == "train" else "eval-set")
        proposal_variant = proposal_variant or self.variant
        relative = "{}/{}/02_roi_crops/{}/{}.png".format(
            dataset_parent, capture, proposal_variant, roi_id
        )
        image = self.root / relative
        write_image(image, np.full((256, 256), value, dtype=np.uint8))
        row = {
            "schema_version": "hlmf_dataset_v1",
            "dataset_id": dataset_id,
            "capture_source_id": capture,
            "split": split,
            "raw_image_id": raw_id,
            "roi_id": roi_id,
            "crop_id": roi_id,
            "proposal_variant": proposal_variant,
            "proposal_slot": 0,
            "proposal_kind": "runtime",
            "crop_relpath": relative,
            "crop_path": relative,
            "hand_presence": {"present": True},
            "handedness": {"label": "Left", "score": 0.9},
            "landmarks_crop_norm": _points(),
            "label_origin": "mediapipe",
            "annotation_style": "mediapipe_v1",
            "palm_score": 0.8,
            "roi_rect": {"center_x": 100.0, "center_y": 100.0, "width": 180.0, "height": 180.0, "rotation": 0.0},
            "roi_corners_px": [[10.0, 10.0], [190.0, 10.0], [190.0, 190.0], [10.0, 190.0]],
        }
        with closing(sqlite3.connect(str(self.registry))) as db:
            db.execute(
                "INSERT INTO rois VALUES(?,?,?,?,?,?)",
                (roi_id, raw_id, capture, proposal_variant, 0, relative),
            )
            db.commit()
        return row

    def _publish_dataset(self, scope, dataset_id, sources, proposal_variant=None):
        bucket = "PretrainSource" if scope == "pretrain" else "EValSource"
        root = self.root / bucket / dataset_id
        proposal_variant = proposal_variant or self.variant
        descriptors = []
        for capture, rows in sources:
            labels = root / capture / "05_labels" / proposal_variant / (
                "hand_training_labels.jsonl" if scope == "pretrain" else "hand_evaluation_labels.jsonl"
            )
            write_jsonl(labels, rows)
            descriptors.append(
                {
                    "capture_source_id": capture,
                    "split": capture.split("-")[4],
                    "published_variants": [
                        {
                            "proposal_variant": proposal_variant,
                            "published_labels": len(rows),
                            "labels_relpath": str(labels.relative_to(self.root)).replace("\\", "/"),
                        }
                    ],
                }
            )
        write_json(
            root / "dataset_manifest.json",
            {
                "schema_version": "hlmf_dataset_v1",
                "dataset_id": dataset_id,
                "scope": scope,
                "capture_sources": descriptors,
            },
        )

    def _publish_negative(self):
        published_root = self.root / "GoldSource" / "NegativeSamples" / "neg-set" / "published"
        source_relative = "PretrainSource/train-set/{}/02_roi_crops/{}/neg-roi.png".format(
            self.train_capture, self.variant
        )
        published_relative = "GoldSource/NegativeSamples/neg-set/published/images/{}/neg-roi.png".format(
            self.train_capture
        )
        write_image(self.root / source_relative, np.zeros((256, 256), dtype=np.uint8))
        write_image(self.root / published_relative, np.zeros((256, 256), dtype=np.uint8))
        row = dict(self.train_rows[0])
        row.update(
            {
                "roi_id": "neg-roi",
                "crop_id": "neg-roi",
                "raw_image_id": "neg-raw",
                "proposal_kind": "negative_candidate",
                "crop_relpath": source_relative,
                "crop_path": source_relative,
                "published_relpath": published_relative,
                "hand_presence": {"present": False},
                "handedness": {"label": "unknown", "score": None},
                "landmarks_crop_norm": [],
                "negative_dataset_id": "neg-set",
            }
        )
        write_jsonl(published_root / "negative_labels.jsonl", [row])
        write_json(
            published_root / "manifest.json",
            {
                "schema_version": "hlmf_dataset_v1",
                "negative_dataset_id": "neg-set",
                "records": 1,
                "labels": "negative_labels.jsonl",
                "image_policy": "copied_review_and_published_images",
            },
        )
        with closing(sqlite3.connect(str(self.registry))) as db:
            db.execute(
                "INSERT INTO rois VALUES(?,?,?,?,?,?)",
                ("neg-roi", "neg-raw", self.train_capture, self.variant, 1, source_relative),
            )
            db.execute("INSERT INTO negative_datasets VALUES('neg-set','published')")
            db.execute(
                "INSERT INTO published_negatives VALUES('neg-roi','neg-set',?)",
                (published_relative,),
            )
            db.commit()

    def _publish_selection(self):
        root = self.root / "Selections" / "hard-set" / "published"
        row = dict(self.train_rows[0])
        row["selection_id"] = "hard-set"
        row["source_crop_relpath"] = row["crop_relpath"]
        row["published_relpath"] = "Selections/hard-set/published/images/{}/train-roi-1.png".format(
            self.train_capture
        )
        write_image(self.root / row["published_relpath"], np.full((256, 256), 70, dtype=np.uint8))
        write_jsonl(root / "selection.jsonl", [row])
        write_json(
            root / "manifest.json",
            {
                "schema_version": "hlmf_dataset_v1",
                "selection_id": "hard-set",
                "records": 1,
                "selection": "selection.jsonl",
                "image_policy": "copied_review_and_published_images",
            },
        )
        with closing(sqlite3.connect(str(self.registry))) as db:
            db.execute("INSERT INTO selections VALUES('hard-set','published')")
            db.commit()

    def config(self):
        dataset = {"dataset_id": "train-set", "proposal_variant": self.variant, "weight": 1.0}
        return {
            "stages": {
                "geometry": {"datasets": [dataset]},
                "multitask": {
                    "datasets": [dataset],
                    "negative_datasets": [{"negative_dataset_id": "neg-set", "weight": 2.0}],
                },
                "multi_finetune": {
                    "datasets": [dataset],
                    "selections": [{"selection_id": "hard-set", "weight": 1.0}],
                    "negative_datasets": [{"negative_dataset_id": "neg-set", "weight": 2.0}],
                    "hard_fraction": 0.55,
                    "replay_fraction": 0.45,
                },
            },
            "evaluation": {
                "val": [{"dataset_id": "eval-set", "proposal_variant": self.variant}],
                "test": [{"dataset_id": "eval-set", "proposal_variant": self.variant}],
            },
            "policies": {"performer_cross_split": "warn"},
        }


class WarehouseV4Tests(unittest.TestCase):
    def test_zero_copy_snapshots_and_three_stage_membership(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            warehouse = SyntheticWarehouse(root / "dataset")
            train = root / "train"
            geometry = build_snapshot(warehouse.config(), warehouse.root, train, "run", "geometry")
            multitask = build_snapshot(warehouse.config(), warehouse.root, train, "run", "multitask")
            finetune = build_snapshot(warehouse.config(), warehouse.root, train, "run", "multi_finetune")
            self.assertEqual({"train": 2, "val": 1, "test": 1}, geometry["records"])
            self.assertEqual(1, multitask["mix"]["negative"])
            self.assertEqual(0.55, finetune["mix"]["hard_fraction"])
            self.assertEqual(1, finetune["mix"]["replay"])
            self.assertEqual(1, finetune["mix"]["true_negative"])
            rows = read_jsonl(train / "snapshots" / "run" / "multitask" / "train.jsonl")
            self.assertTrue(all(warehouse.root in Path(row["crop_path"]).parents for row in rows))
            self.assertFalse(any(path.suffix.lower() in {".png", ".tif", ".tiff"} for path in train.rglob("*")))
            negative = next(row for row in rows if row.get("negative_dataset_id") == "neg-set")
            self.assertEqual(2.0, negative["sampling_weight"])
            negative_labels = read_jsonl(
                warehouse.root
                / "GoldSource"
                / "NegativeSamples"
                / "neg-set"
                / "published"
                / "negative_labels.jsonl"
            )
            (warehouse.root / negative_labels[0]["crop_relpath"]).unlink()
            negatives_after_source_delete = WarehouseReader(warehouse.root).negative_rows("neg-set")
            self.assertEqual(1, len(negatives_after_source_delete))
            self.assertTrue(Path(negatives_after_source_delete[0]["_absolute_crop_path"]).is_file())

            finetune_rows = read_jsonl(
                train / "snapshots" / "run" / "multi_finetune" / "train.jsonl"
            )
            selected = next(row for row in finetune_rows if row.get("selection_id") == "hard-set")
            self.assertIn("/Selections/hard-set/published/images/", selected["crop_path"].replace("\\", "/"))
            self.assertTrue(selected["source_crop_relpath"].startswith("PretrainSource/train-set/"))

            source_image = warehouse.root / selected["source_crop_relpath"]
            source_image.unlink()
            rows_after_source_delete = WarehouseReader(warehouse.root).selection_rows("hard-set")
            self.assertEqual(1, len(rows_after_source_delete))
            self.assertTrue(Path(rows_after_source_delete[0]["_absolute_crop_path"]).is_file())

    def test_geometry_combines_multiple_dataset_ids_variants_and_weights(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            warehouse = SyntheticWarehouse(root / "dataset")
            second_capture = "room-far-daylight-normal-train-s02-dave"
            second_row = warehouse._source_row(
                "train-roi-3",
                "train-raw-3",
                second_capture,
                "PretrainSource/train-set-2",
                150,
                dataset_id="train-set-2",
                proposal_variant="palm-v2",
            )
            warehouse._publish_dataset(
                "pretrain",
                "train-set-2",
                [(second_capture, [second_row])],
                proposal_variant="palm-v2",
            )
            config = warehouse.config()
            config["stages"]["geometry"]["datasets"].append(
                {
                    "dataset_id": "train-set-2",
                    "proposal_variant": "palm-v2",
                    "weight": 3.0,
                }
            )
            train_root = root / "train"
            report = build_snapshot(
                config, warehouse.root, train_root, "multi-source", "geometry"
            )
            self.assertEqual({"train": 3, "val": 1, "test": 1}, report["records"])
            rows = read_jsonl(
                train_root / "snapshots" / "multi-source" / "geometry" / "train.jsonl"
            )
            self.assertEqual({"train-set", "train-set-2"}, {row["dataset_id"] for row in rows})
            second = next(row for row in rows if row["dataset_id"] == "train-set-2")
            self.assertEqual("palm-v2", second["proposal_variant"])
            self.assertEqual(3.0, second["sampling_weight"])

    def test_partial_variant_skips_unpublished_sources_but_requires_split_rows(self):
        with tempfile.TemporaryDirectory() as temp:
            warehouse = SyntheticWarehouse(Path(temp) / "dataset")
            manifest_path = (
                warehouse.root / "EValSource" / "eval-set" / "dataset_manifest.json"
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["quality_gate_counting_policy"] = (
                "exclusive_by_publish_routing_priority"
            )
            manifest["quality_gate_rejections"] = {
                "hand_presence": 0,
                "boundary_coordinate": 0,
                "connection_length": 0,
                "handedness": 0,
            }
            manifest["capture_sources"].append(
                {
                    "capture_source_id": "room-far-dim-normal-val-s02-dave",
                    "split": "val",
                    "published_variants": [
                        {
                            "proposal_variant": "eos-1.0",
                            "published_labels": 1,
                            "labels_relpath": "not-read-for-unselected-variant.jsonl",
                        }
                    ],
                }
            )
            write_json(manifest_path, manifest)

            rows = WarehouseReader(warehouse.root).source_rows(
                "eval", "eval-set", warehouse.variant, expected_split="val"
            )
            self.assertEqual(["val-roi-1"], [row["roi_id"] for row in rows])
            selected_rows = WarehouseReader(warehouse.root).source_rows(
                "eval",
                "eval-set",
                warehouse.variant,
                expected_split="val",
                capture_source_ids=[warehouse.val_capture],
            )
            self.assertEqual(["val-roi-1"], [row["roi_id"] for row in selected_rows])
            with self.assertRaisesRegex(
                WarehouseContractError, "does not publish selected variant"
            ):
                WarehouseReader(warehouse.root).source_rows(
                    "eval",
                    "eval-set",
                    warehouse.variant,
                    expected_split="val",
                    capture_source_ids=["room-far-dim-normal-val-s02-dave"],
                )
            with self.assertRaisesRegex(WarehouseContractError, "manifest is missing"):
                WarehouseReader(warehouse.root).source_rows(
                    "eval",
                    "eval-set",
                    warehouse.variant,
                    expected_split="val",
                    capture_source_ids=["studio-mid-daylight-normal-val-s03-erin"],
                )
            with self.assertRaisesRegex(WarehouseContractError, "belongs to split test"):
                WarehouseReader(warehouse.root).source_rows(
                    "eval",
                    "eval-set",
                    warehouse.variant,
                    expected_split="val",
                    capture_source_ids=[warehouse.test_capture],
                )
            with self.assertRaisesRegex(WarehouseContractError, "has no published rows"):
                WarehouseReader(warehouse.root).source_rows(
                    "eval", "eval-set", "missing-variant", expected_split="val"
                )

    def test_evaluation_selects_exact_capture_sources_across_variants(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            warehouse = SyntheticWarehouse(root / "dataset")
            second_capture = "studio-mid-daylight-normal-val-s02-dave"
            second_variant = "eos-2.0-hcf0813"
            second_row = warehouse._source_row(
                "val-roi-2",
                "val-raw-2",
                second_capture,
                "EValSource/eval-set",
                120,
                dataset_id="eval-set",
                proposal_variant=second_variant,
            )
            labels = (
                warehouse.root
                / "EValSource"
                / "eval-set"
                / second_capture
                / "05_labels"
                / second_variant
                / "hand_evaluation_labels.jsonl"
            )
            write_jsonl(labels, [second_row])
            manifest_path = (
                warehouse.root / "EValSource" / "eval-set" / "dataset_manifest.json"
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["capture_sources"].append(
                {
                    "capture_source_id": second_capture,
                    "split": "val",
                    "published_variants": [
                        {
                            "proposal_variant": second_variant,
                            "published_labels": 1,
                            "labels_relpath": str(labels.relative_to(warehouse.root)).replace(
                                "\\", "/"
                            ),
                        }
                    ],
                }
            )
            write_json(manifest_path, manifest)

            config = warehouse.config()
            config["evaluation"]["val"] = [
                {
                    "dataset_id": "eval-set",
                    "proposal_variant": warehouse.variant,
                    "capture_source_ids": [warehouse.val_capture],
                },
                {
                    "dataset_id": "eval-set",
                    "proposal_variant": second_variant,
                    "capture_source_ids": [second_capture],
                },
            ]
            config["evaluation"]["test"][0]["capture_source_ids"] = [
                warehouse.test_capture
            ]
            train_root = root / "train"
            report = build_snapshot(config, warehouse.root, train_root, "selected", "geometry")
            self.assertEqual({"train": 2, "val": 2, "test": 1}, report["records"])
            val_rows = read_jsonl(
                train_root / "snapshots" / "selected" / "geometry" / "val.jsonl"
            )
            self.assertEqual(
                {warehouse.val_capture, second_capture},
                {row["capture_source_id"] for row in val_rows},
            )
            self.assertEqual(
                {warehouse.variant, second_variant},
                {row["proposal_variant"] for row in val_rows},
            )

    def test_latest_hlmf_teacher_and_geometry_rescue_provenance_is_preserved(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            warehouse = SyntheticWarehouse(root / "dataset")
            manifest_path = warehouse.root / "PretrainSource" / "train-set" / "dataset_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            labels_path = warehouse.root / manifest["capture_sources"][0]["published_variants"][0][
                "labels_relpath"
            ]
            rows = read_jsonl(labels_path)
            rows[0].update(
                {
                    "source": "mediapipe_hand_landmarker_full_tflite_rtmpose_rescue",
                    "label_origin": "mediapipe",
                    "annotation_style": "mediapipe_tflite_rescue_v1",
                    "teacher_model_id": "mediapipe-hand-landmark-full-tflite",
                    "handedness_teacher_model_id": "hand-classifier-handedness-handpresence-0813",
                    "hand_presence_teacher_model_id": "hand-classifier-handedness-handpresence-0813",
                    "rtmpose_geometry_rescue": {
                        "attempted": True,
                        "accepted": True,
                        "trigger_errors": ["rtmpose_boundary_coordinate_count>=2"],
                        "result_errors": [],
                        "model_id": "mediapipe-hand-landmark-full-tflite",
                    },
                }
            )
            rows[0]["landmarks_crop_px"] = [
                {
                    "id": point["id"],
                    "x": point["x"] * 256.0,
                    "y": point["y"] * 256.0,
                }
                for point in rows[0]["landmarks_crop_norm"]
            ]
            write_jsonl(labels_path, rows)

            build_snapshot(warehouse.config(), warehouse.root, root / "train", "rescue", "geometry")
            snapshot_rows = read_jsonl(root / "train" / "snapshots" / "rescue" / "geometry" / "train.jsonl")
            rescued = next(row for row in snapshot_rows if row["roi_id"] == rows[0]["roi_id"])
            self.assertEqual("mediapipe_tflite_rescue_v1", rescued["annotation_style"])
            self.assertEqual(
                "hand-classifier-handedness-handpresence-0813",
                rescued["hand_presence_teacher_model_id"],
            )
            self.assertTrue(rescued["rtmpose_geometry_rescue"]["accepted"])
            self.assertEqual(
                "crop_extent_0_to_size", rescued["warehouse_crop_pixel_convention"]
            )
            self.assertAlmostEqual(
                rescued["landmarks_crop_norm"][0]["x"] * 255.0,
                rescued["landmarks_crop_px"][0]["x"],
            )

    def test_hlmf_crop_pixel_coordinates_must_match_a_supported_convention(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            warehouse = SyntheticWarehouse(root / "dataset")
            manifest_path = (
                warehouse.root / "PretrainSource" / "train-set" / "dataset_manifest.json"
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            labels_path = warehouse.root / manifest["capture_sources"][0][
                "published_variants"
            ][0]["labels_relpath"]
            rows = read_jsonl(labels_path)
            rows[0]["landmarks_crop_px"] = [
                {
                    "id": point["id"],
                    "x": point["x"] * 255.0,
                    "y": point["y"] * 255.0,
                }
                for point in rows[0]["landmarks_crop_norm"]
            ]
            rows[0]["landmarks_crop_px"][0]["x"] += 5.0
            write_jsonl(labels_path, rows)
            with self.assertRaisesRegex(
                WarehouseContractError, "disagree under both HLMF coordinate conventions"
            ):
                build_snapshot(
                    warehouse.config(), warehouse.root, root / "train", "bad-pixels", "geometry"
                )

    def test_geometry_rejects_negative_dataset(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            warehouse = SyntheticWarehouse(root / "dataset")
            config = warehouse.config()
            config["stages"]["geometry"]["negative_datasets"] = [{"negative_dataset_id": "neg-set"}]
            with self.assertRaisesRegex(WarehouseContractError, "geometry forbids"):
                build_snapshot(config, warehouse.root, root / "train", "run", "geometry")

    def test_path_boundary_and_registry_are_enforced(self):
        with tempfile.TemporaryDirectory() as temp:
            warehouse = SyntheticWarehouse(Path(temp) / "dataset")
            manifest_path = warehouse.root / "PretrainSource" / "train-set" / "dataset_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            labels = warehouse.root / manifest["capture_sources"][0]["published_variants"][0]["labels_relpath"]
            rows = read_jsonl(labels)
            rows[0]["crop_relpath"] = "../escape.png"
            rows[0]["crop_path"] = "../escape.png"
            write_jsonl(labels, rows)
            with self.assertRaisesRegex(WarehouseContractError, "disagrees with registry|escapes"):
                WarehouseReader(warehouse.root).source_rows("pretrain", "train-set", warehouse.variant)

    def test_split_and_single_variant_gates(self):
        base = {
            "capture_source_id": "room-near-day-normal-train-s01-alice",
            "raw_image_id": "raw",
            "roi_id": "roi-a",
            "split": "train",
            "proposal_variant": "palm-a",
        }
        leaked = dict(base, roi_id="roi-b", split="val")
        warnings, report = _assert_membership([base, leaked])
        self.assertTrue(report["errors"])
        other_variant = dict(base, roi_id="roi-c", raw_image_id="raw-c", proposal_variant="palm-b")
        _, report = _assert_membership([base, other_variant])
        self.assertIn("multiple proposal variants", report["errors"][0])

    def test_mining_is_train_only_and_reports_sources(self):
        with tempfile.TemporaryDirectory() as temp:
            warehouse = SyntheticWarehouse(Path(temp) / "dataset")
            label = dict(warehouse.train_rows[0])
            label["crop_path"] = str(warehouse.root / label["crop_relpath"])
            prediction = {
                "roi_id": label["roi_id"],
                "student_landmarks_crop_norm": _points(0.02),
                "student_hand_flag": 0.9,
            }
            reports, request = aggregate_sources([label], [prediction])
            self.assertEqual(1, reports[0]["sample_count"])
            self.assertIn("p90_error_px", reports[0])
            self.assertEqual(label["crop_relpath"], request[0]["crop_path"])
            label["split"] = "test"
            with self.assertRaisesRegex(ValueError, "never read Val/Test"):
                aggregate_sources([label], [prediction])

    def test_winner_and_locked_test_are_immutable(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            checkpoint = root / "best.weights.h5"
            checkpoint.write_bytes(b"weights")
            metrics = root / "metrics.json"
            write_json(
                metrics,
                {
                    "split": "val",
                    "scope": "hand_landmarker_on_provided_hand_roi",
                    "presence_threshold": 0.6,
                },
            )
            winner = freeze_winner(root, "release", metrics, checkpoint, "pretrain", "snapshot")
            self.assertEqual(0.6, winner["presence_threshold"])
            config = {
                "split": "test",
                "evaluation": {"mode": "roi", "tune_thresholds": False},
                "output": {"dir": str(root / "test-result"), "overwrite": False},
                "hand": {},
                "model": {},
            }
            locked = locked_test_config(config, root, "release")
            self.assertEqual(str(checkpoint.resolve()), locked["hand"]["model_path"])
            self.assertEqual(0.6, locked["evaluation"]["hand_flag_threshold"])
            with self.assertRaises(FileExistsError):
                freeze_winner(root, "release", metrics, checkpoint, "pretrain", "snapshot")

    def test_fixed_roi_evaluation_has_required_groups_and_never_needs_palm(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            warehouse = SyntheticWarehouse(root / "dataset")
            train = root / "train"
            build_snapshot(warehouse.config(), warehouse.root, train, "run", "geometry")
            rows = read_jsonl(train / "snapshots" / "run" / "geometry" / "val.jsonl")
            for row in rows:
                row["_resolved_crop_path"] = row["crop_path"]

            class Predictor:
                def predict(self, images, batch_size=64):
                    points = [(point["x"], point["y"]) for point in _points()]
                    return [
                        SimpleNamespace(
                            landmarks_norm=points,
                            hand_flag_score=0.9,
                            handedness="Left",
                            handedness_score=0.9,
                            landmark_raw_max_abs=max(max(point) for point in points),
                            normalized_out_of_range_coordinate_count=0,
                        )
                        for _ in images
                    ]

            config = {
                "model": {"version": "v2", "num_iterations": [2, 2, 3, 4, 4, 6, 6]},
                "hand": {"backend": "keras", "model_path": "unused"},
                "evaluation": {"mode": "roi", "pck_thresholds": [0.05, 0.10, 0.15]},
                "inference": {"batch_size": 4},
            }
            with mock.patch("hand_landmarker.evaluation.create_hand_predictor", return_value=Predictor()):
                report = evaluate_hand_rois(config, rows, model_checkpoint_stage="pretrain")
            self.assertEqual("hand_landmarker_on_provided_hand_roi", report["scope"])
            self.assertEqual(0.0, report["metrics"]["landmarks"]["mean_pixel_error"])
            self.assertEqual(
                {"dataset_id", "capture_source_id", "label_origin", "annotation_style", "distance", "lighting"},
                set(report["grouped_metrics"]),
            )


if __name__ == "__main__":
    unittest.main()
