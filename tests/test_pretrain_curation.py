import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from hand_landmarker.inspect import DatasetContractError, audit_canonical_dataset
from hand_landmarker.io_utils import read_jsonl, sha256_file, write_jsonl
from hand_landmarker.pretrain_curation import (
    _select_teacher_holdout_datasets,
    curate_pretrain_from_config,
    verify_curation_manifest,
)


def _points(normalized=True, out_of_range=False):
    values = []
    for point_id in range(21):
        if normalized:
            x_value = 1.2 if out_of_range and point_id == 20 else 0.2 + point_id * 0.01
            y_value = 0.3 + point_id * 0.005
        else:
            x_value = 10.0 + point_id
            y_value = 12.0 + point_id
        values.append({"id": point_id, "x": x_value, "y": y_value})
    return values


def _positive(crop_id, image, sample_type="POS_RUNTIME", group="frame-1", out_of_range=False):
    return {
        "crop_id": crop_id,
        "dataset_id": "dataset-a",
        "source_group_id": group,
        "sample_type": sample_type,
        "quality_tier": "HIGH",
        "quality_flags": [],
        "hand_presence": {"present": True},
        "crop_path": str(image),
        "landmarks_crop_norm": _points(True, out_of_range),
        "landmarks_crop_px": _points(False),
        "landmarks_image_px": _points(False),
    }


def _negative(crop_id, image, group, sample_type="NEG_RUNTIME_CANDIDATE"):
    return {
        "crop_id": crop_id,
        "dataset_id": "dataset-a",
        "source_group_id": group,
        "sample_type": sample_type,
        "quality_tier": "HIGH",
        "quality_flags": [],
        "hand_presence": {"present": False},
        "crop_path": str(image),
        "landmarks_crop_norm": [],
        "landmarks_crop_px": [],
        "landmarks_image_px": [],
        "roi_corners_px": [[0.0, 0.0], [40.0, 0.0], [40.0, 40.0], [0.0, 40.0]],
    }


class PretrainCurationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.images = self.root / "source_images"
        self.images.mkdir()
        self.rows = []
        for index in range(5):
            path = self.images / "{}.png".format(index)
            path.write_bytes(("image-{}".format(index)).encode("ascii"))
        self.rows.extend(
            [
                _positive("pos-runtime", self.images / "0.png"),
                _positive(
                    "pos-low",
                    self.images / "1.png",
                    sample_type="POS_LOW_PALM",
                    group="frame-2",
                ),
                _positive(
                    "pos-invalid",
                    self.images / "2.png",
                    group="frame-3",
                    out_of_range=True,
                ),
                _negative("neg-overlap", self.images / "3.png", group="frame-1"),
                _negative("neg-clean", self.images / "4.png", group="frame-4"),
            ]
        )
        self.labels = self.root / "source.jsonl"
        write_jsonl(self.labels, self.rows)
        self.source_hash = sha256_file(self.labels)

    def tearDown(self):
        self.temporary.cleanup()

    def _config(self, output):
        review_root = self.root / "reviews" / Path(output).name
        return {
            "task": "curate_pretrain",
            "source": {
                "labels": str(self.labels),
                "crop_root": str(self.images),
            },
            "curation": {
                "allowed_positive_quality_tiers": ["HIGH", "MEDIUM"],
                "normalized_coordinate_range": [0.0, 1.0],
                "overlap_core_landmark_ids": [0, 1, 5, 9, 13, 17],
                "overlap_minimum_core_points": 3,
                "smoke_subset_size": 2,
                "smoke_selection_salt": "test",
            },
            "review": {
                "decisions_file": str(review_root / "negative_review_decisions.jsonl"),
                "candidates_subdir": "negative_candidates",
                "reviewed_subdir": "negative_reviewed",
                "removed_subdir": "negative_removed",
                "quarantine_subdir": "negative_quarantine",
                "removed_manifest_file": "negative_removed_manifest.jsonl",
                "quarantine_manifest_file": "negative_quarantine_manifest.jsonl",
                "cleanup_candidates_after_success": True,
                "retain_reviewed_evidence": True,
                "reviewer": "unit-test-team",
            },
            "output": {
                "dir": str(output),
                "overwrite": False,
            },
        }

    def _review_candidates(self, output, crop_ids):
        review_root = self.root / "reviews" / Path(output).name
        manifest = read_jsonl(review_root / "review_manifest.jsonl")
        selected = set(crop_ids)
        for row in manifest:
            if row["crop_id"] not in selected:
                continue
            relative = Path(row["candidate_relative_path"])
            source = review_root / "negative_candidates" / relative
            destination = review_root / "negative_reviewed" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        return manifest

    def test_teacher_holdout_selects_complete_datasets_deterministically(self):
        rows = []
        for dataset_id, count in (("old", 8), ("new-a", 3), ("new-b", 4), ("new-c", 6)):
            rows.extend({"dataset_id": dataset_id} for _ in range(count))
        config = {
            "enabled": True,
            "eligible_dataset_pattern": "new-.*",
            "minimum_positive_records": 5,
            "target_positive_records": 7,
            "maximum_positive_records": 8,
            "selection_salt": "unit-test",
        }
        first, report = _select_teacher_holdout_datasets(rows, config)
        second, _ = _select_teacher_holdout_datasets(rows, config)
        self.assertEqual(first, second)
        self.assertEqual({"new-a", "new-b"}, first)
        self.assertEqual(7, report["selected_positive_records"])
        self.assertEqual(14, report["remaining_training_positive_records"])

    def test_teacher_holdout_fails_when_whole_sources_cannot_meet_bounds(self):
        rows = [{"dataset_id": "new-a"} for _ in range(11)]
        with self.assertRaisesRegex(ValueError, "no eligible dataset"):
            _select_teacher_holdout_datasets(
                rows,
                {
                    "enabled": True,
                    "eligible_dataset_pattern": "new-.*",
                    "minimum_positive_records": 5,
                    "target_positive_records": 7,
                    "maximum_positive_records": 8,
                },
            )

    def test_curated_teacher_holdout_is_disjoint_and_authenticated(self):
        extra_paths = []
        for index in range(5, 9):
            path = self.images / "{}.png".format(index)
            path.write_bytes(("image-{}".format(index)).encode("ascii"))
            extra_paths.append(path)
        rows = [
            _positive("train-1", extra_paths[0], group="train-frame-1"),
            _positive("train-2", extra_paths[1], group="train-frame-2"),
            _positive("holdout-1", extra_paths[2], group="holdout-frame-1"),
            _positive("holdout-2", extra_paths[3], group="holdout-frame-2"),
        ]
        rows[0]["dataset_id"] = "train-source"
        rows[1]["dataset_id"] = "train-source"
        rows[2]["dataset_id"] = "holdout-source"
        rows[3]["dataset_id"] = "holdout-source"
        write_jsonl(self.labels, rows)
        output = self.root / "automatic-holdout"
        config = self._config(output)
        config["curation"]["teacher_holdout"] = {
            "enabled": True,
            "eligible_dataset_pattern": "holdout-.*",
            "minimum_positive_records": 2,
            "target_positive_records": 2,
            "maximum_positive_records": 2,
            "selection_salt": "integration-test",
        }
        report = curate_pretrain_from_config(config)
        train = read_jsonl(
            output / "05_labels" / "hand_training_labels_pretrain_landmarks.jsonl"
        )
        holdout = read_jsonl(
            output / "05_labels" / "hand_teacher_holdout_labels.jsonl"
        )
        self.assertEqual({"train-source"}, {row["dataset_id"] for row in train})
        self.assertEqual({"holdout-source"}, {row["dataset_id"] for row in holdout})
        self.assertEqual(2, report["counts"]["teacher_holdout_positives"])
        self.assertEqual(
            ["holdout-source"], report["teacher_holdout"]["selected_dataset_ids"]
        )
        with (output / "qc" / "sha256_manifest.json").open(encoding="utf-8") as handle:
            manifest = json.load(handle)
        relative = "05_labels/hand_teacher_holdout_labels.jsonl"
        self.assertEqual(
            sha256_file(output / relative), manifest["artifacts"][relative]["sha256"]
        )

    def test_review_finalize_excludes_negatives_from_teacher_holdout_sources(self):
        extra_paths = []
        for index in range(5, 11):
            path = self.images / "{}.png".format(index)
            path.write_bytes(("image-{}".format(index)).encode("ascii"))
            extra_paths.append(path)
        rows = [
            _positive("train-1", extra_paths[0], group="train-frame-1"),
            _positive("train-2", extra_paths[1], group="train-frame-2"),
            _positive("holdout-1", extra_paths[2], group="holdout-frame-1"),
            _positive("holdout-2", extra_paths[3], group="holdout-frame-2"),
            _negative("train-negative", extra_paths[4], group="train-negative-frame"),
            _negative("holdout-negative", extra_paths[5], group="holdout-negative-frame"),
        ]
        for row in rows[:2] + [rows[4]]:
            row["dataset_id"] = "train-source"
        for row in rows[2:4] + [rows[5]]:
            row["dataset_id"] = "holdout-source"
        write_jsonl(self.labels, rows)

        output = self.root / "automatic-holdout-with-negative"
        config = self._config(output)
        config["curation"]["teacher_holdout"] = {
            "enabled": True,
            "eligible_dataset_pattern": "holdout-.*",
            "minimum_positive_records": 2,
            "target_positive_records": 2,
            "maximum_positive_records": 2,
            "selection_salt": "integration-test",
        }
        curate_pretrain_from_config(config)
        review_manifest = self._review_candidates(output, {"train-negative"})
        self.assertEqual(
            {"train-negative"}, {row["crop_id"] for row in review_manifest}
        )

        report = curate_pretrain_from_config(
            config, overwrite=True, finalize_review=True
        )
        self.assertEqual(1, report["counts"]["negative_review_expected"])
        self.assertEqual(1, report["counts"]["included_confirmed_negatives"])
        multitask = read_jsonl(
            output / "05_labels" / "hand_training_labels_pretrain_multitask.jsonl"
        )
        self.assertEqual(
            {"train-1", "train-2", "train-negative"},
            {row["crop_id"] for row in multitask},
        )

    def test_positive_snapshot_and_negative_quarantine_are_persisted(self):
        output = self.root / "curated"
        report = curate_pretrain_from_config(self._config(output))
        self.assertEqual("ok", report["status"])
        self.assertEqual(2, report["counts"]["included_landmark_positives"])
        self.assertEqual(0, report["counts"]["included_confirmed_negatives"])
        self.assertEqual(3, report["counts"]["excluded_or_held"])
        self.assertEqual(2, report["counts"]["negative_review_queue"])
        self.assertEqual(1, report["counts"]["negative_overlap_confirmed_hand"])

        landmarks = read_jsonl(
            output / "05_labels" / "hand_training_labels_pretrain_landmarks.jsonl"
        )
        self.assertEqual({"pos-runtime", "pos-low"}, {row["crop_id"] for row in landmarks})
        for row in landmarks:
            source = Path(row["crop_path"])
            self.assertTrue(source.is_file())
            self.assertTrue(str(source).startswith(str(self.images)))
            self.assertEqual(
                "external_source_reference",
                row["pretrain_curation"]["image_storage"],
            )
            self.assertEqual(self.source_hash, row["pretrain_curation"]["source_labels_sha256"])
        smoke = read_jsonl(
            output / "05_labels" / "hand_training_labels_pretrain_smoke.jsonl"
        )
        self.assertTrue(all(row["sampling_weight"] == 1.0 for row in smoke))
        self.assertTrue(
            all(row["pretrain_curation"]["smoke_sampling_weight"] == 1.0 for row in smoke)
        )

        queue = read_jsonl(output / "audit" / "negative_review_queue.jsonl")
        by_id = {row["crop_id"]: row for row in queue}
        for row in queue:
            review_crop = Path(row["crop_path"])
            self.assertTrue(review_crop.is_file())
            self.assertTrue(str(review_crop).startswith(str(self.images)))
            self.assertEqual(
                sha256_file(review_crop), row["pretrain_curation"]["review_image_sha256"]
            )
        self.assertIn(
            "NEGATIVE_OVERLAPS_CONFIRMED_HAND",
            by_id["neg-overlap"]["pretrain_curation"]["reasons"],
        )
        self.assertIn(
            "UNVERIFIED_TEACHER_NEGATIVE",
            by_id["neg-clean"]["pretrain_curation"]["reasons"],
        )
        self.assertEqual(5, len(read_jsonl(output / "audit" / "pretrain_curation_catalog.jsonl")))
        self.assertTrue((output / "qc" / "curation_report.json").is_file())
        self.assertTrue((output / "qc" / "sha256_manifest.json").is_file())
        with (output / "qc" / "sha256_manifest.json").open(encoding="utf-8") as handle:
            manifest = json.load(handle)
            self.assertEqual("external_source_reference", manifest["images"]["storage"])
        self.assertEqual(2, manifest["review_candidates"]["count"])
        self.assertFalse((output / "images").exists())
        self.assertFalse((output / "review_images").exists())
        review_root = self.root / "reviews" / output.name
        review_manifest = read_jsonl(review_root / "review_manifest.jsonl")
        self.assertEqual(2, len(review_manifest))
        self.assertEqual(
            2,
            len(list((review_root / "negative_candidates").rglob("*.png"))),
        )
        self.assertFalse((review_root / "negative_review_decisions.jsonl").exists())
        dataset = {
            "labels": str(output / "05_labels" / "hand_training_labels_pretrain_landmarks.jsonl"),
            "curation_manifest": str(output / "qc" / "sha256_manifest.json"),
            "require_curation_schema": "pretrain_curation_v1",
        }
        verified = verify_curation_manifest({}, dataset, error_type=DatasetContractError)
        self.assertEqual(2, verified["image_count"])
        with Path(dataset["labels"]).open("a", encoding="utf-8") as handle:
            handle.write("{}\n")
        with self.assertRaisesRegex(DatasetContractError, "hash mismatch"):
            verify_curation_manifest({}, dataset, error_type=DatasetContractError)
        self.assertEqual(self.source_hash, sha256_file(self.labels))
        with self.assertRaises(FileExistsError):
            curate_pretrain_from_config(self._config(output))

    def test_snapshot_references_train_sources_and_only_copies_review_candidates(self):
        output = self.root / "referenced"
        curate_pretrain_from_config(self._config(output))

        landmarks = read_jsonl(
            output / "05_labels" / "hand_training_labels_pretrain_landmarks.jsonl"
        )
        source_by_id = {row["crop_id"]: Path(row["crop_path"]) for row in self.rows}
        for row in landmarks:
            self.assertEqual(source_by_id[row["crop_id"]].resolve(), Path(row["crop_path"]))

        self.assertFalse((output / "images").exists())
        self.assertFalse((output / "review_images").exists())
        review_root = self.root / "reviews" / output.name
        review_manifest = read_jsonl(review_root / "review_manifest.jsonl")
        for row in review_manifest:
            candidate = review_root / "negative_candidates" / row["candidate_relative_path"]
            source = Path(row["source_crop_path"])
            self.assertTrue(candidate.is_file())
            self.assertEqual(source.read_bytes(), candidate.read_bytes())
            self.assertFalse(source.samefile(candidate))

    def test_source_crop_must_be_inside_configured_train_sources_root(self):
        outside = self.root / "outside.png"
        outside.write_bytes(Path(self.rows[0]["crop_path"]).read_bytes())
        self.rows[0]["crop_path"] = str(outside)
        write_jsonl(self.labels, self.rows)
        with self.assertRaisesRegex(ValueError, "under source.crop_root"):
            curate_pretrain_from_config(self._config(self.root / "outside-source"))

    def test_only_retained_non_conflicting_negative_enters_multitask(self):
        output = self.root / "reviewed"
        config = self._config(output)
        curate_pretrain_from_config(config)
        review_root = self.root / "reviews" / output.name
        self._review_candidates(output, {"neg-clean"})

        report = curate_pretrain_from_config(
            config, overwrite=True, finalize_review=True
        )
        self.assertEqual(1, report["counts"]["included_confirmed_negatives"])
        self.assertEqual(2, report["counts"]["negative_review_expected"])
        self.assertEqual(1, report["counts"]["negative_reviewed"])
        self.assertEqual(1, report["counts"]["negative_admitted"])
        self.assertEqual(1, report["counts"]["negative_removed"])
        self.assertEqual(0, report["counts"]["negative_quarantine"])
        self.assertFalse((review_root / "negative_candidates").exists())
        decisions = review_root / "negative_review_decisions.jsonl"
        self.assertEqual(sha256_file(decisions), report["negative_review_decisions"]["sha256"])
        decision_rows = read_jsonl(decisions)
        self.assertEqual(["neg-clean"], [row["crop_id"] for row in decision_rows])
        self.assertEqual("unit-test-team", decision_rows[0]["reviewer"])
        self.assertEqual(
            "retained_after_visual_deletion_review", decision_rows[0]["review_method"]
        )
        self.assertEqual(
            1,
            report["negative_review_workspace"]["deleted_or_rejected_count"],
        )
        multitask = read_jsonl(
            output / "05_labels" / "hand_training_labels_pretrain_multitask.jsonl"
        )
        self.assertEqual(
            {"pos-runtime", "pos-low", "neg-clean"},
            {row["crop_id"] for row in multitask},
        )
        queue = {row["crop_id"]: row for row in read_jsonl(output / "audit" / "negative_review_queue.jsonl")}
        self.assertIn("neg-overlap", queue)
        self.assertIn(
            "NEGATIVE_OVERLAPS_CONFIRMED_HAND",
            queue["neg-overlap"]["pretrain_curation"]["reasons"],
        )

    def test_retained_overlap_candidate_is_still_blocked_by_automatic_safety(self):
        output = self.root / "retained-overlap"
        config = self._config(output)
        curate_pretrain_from_config(config)
        self._review_candidates(output, {"neg-overlap", "neg-clean"})
        report = curate_pretrain_from_config(
            config, overwrite=True, finalize_review=True
        )
        self.assertEqual(2, report["negative_review_workspace"]["retained_confirmed_count"])
        self.assertEqual(1, report["negative_review_workspace"]["quarantine_count"])
        self.assertEqual(1, report["negative_review_workspace"]["admitted_confirmed_count"])
        self.assertEqual(1, report["counts"]["included_confirmed_negatives"])
        queue = {
            row["crop_id"]: row
            for row in read_jsonl(output / "audit" / "negative_review_queue.jsonl")
        }
        self.assertIn(
            "REVIEW_CONFLICTS_CONFIRMED_HAND_OVERLAP",
            queue["neg-overlap"]["pretrain_curation"]["reasons"],
        )

    def test_review_workspace_rejects_changed_source_snapshot(self):
        output = self.root / "changed-source"
        config = self._config(output)
        curate_pretrain_from_config(config)
        with self.labels.open("a", encoding="utf-8") as handle:
            handle.write("\n")
        with self.assertRaisesRegex(ValueError, "different source-label snapshot"):
            curate_pretrain_from_config(
                config, overwrite=True, finalize_review=True
            )

    def test_review_workspace_rejects_modified_retained_image(self):
        output = self.root / "modified-review-image"
        config = self._config(output)
        curate_pretrain_from_config(config)
        review_root = self.root / "reviews" / output.name
        manifest = self._review_candidates(output, {"neg-clean"})
        row = next(row for row in manifest if row["crop_id"] == "neg-clean")
        candidate = review_root / "negative_reviewed" / row["candidate_relative_path"]
        candidate.write_bytes(b"modified")
        with self.assertRaisesRegex(ValueError, "was modified"):
            curate_pretrain_from_config(
                config, overwrite=True, finalize_review=True
            )

    def test_review_finalize_rejects_changed_train_source_roi(self):
        output = self.root / "modified-source-image"
        config = self._config(output)
        curate_pretrain_from_config(config)
        positive = next(row for row in self.rows if row["crop_id"] == "pos-runtime")
        Path(positive["crop_path"]).write_bytes(b"modified-source")
        with self.assertRaisesRegex(RuntimeError, "Source ROI bytes changed"):
            curate_pretrain_from_config(
                config, overwrite=True, finalize_review=True
            )

    def test_overlap_is_scoped_to_dataset_and_source_hash_is_checked(self):
        for row in self.rows:
            if row["crop_id"] == "neg-overlap":
                row["dataset_id"] = "dataset-b"
        write_jsonl(self.labels, self.rows)
        output = self.root / "dataset-scoped"
        report = curate_pretrain_from_config(self._config(output))
        self.assertEqual(0, report["counts"]["negative_overlap_confirmed_hand"])

        labels = output / "05_labels" / "hand_training_labels_pretrain_landmarks.jsonl"
        included = read_jsonl(labels)
        Path(included[0]["crop_path"]).write_bytes(b"tampered")
        _, audit = audit_canonical_dataset(
            {},
            dataset={
                "labels": str(labels),
                "crop_path_key": "crop_path",
                "path_policy": "canonical_crop_path_only",
            },
            check_images=False,
            hash_images=True,
            raise_on_error=False,
        )
        messages = [
            message
            for item in audit["errors"]
            for message in item.get("errors", [])
        ]
        self.assertIn(
            "source image hash does not match pretrain_curation.image_sha256",
            messages,
        )

    def test_overwrite_refuses_unknown_or_source_containing_directory(self):
        unknown = self.root / "not-a-snapshot"
        unknown.mkdir()
        marker = unknown / "keep.txt"
        marker.write_text("keep", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "without a curation manifest"):
            curate_pretrain_from_config(self._config(unknown), overwrite=True)
        self.assertEqual("keep", marker.read_text(encoding="utf-8"))

        with self.assertRaisesRegex(ValueError, "contains source data"):
            curate_pretrain_from_config(self._config(self.root), overwrite=True)
        self.assertTrue(self.labels.is_file())

    def test_review_folder_rejects_unmanifested_images(self):
        output = self.root / "invalid-review-output"
        config = self._config(output)
        curate_pretrain_from_config(config)
        reviewed = self.root / "reviews" / output.name / "negative_reviewed"
        (reviewed / "unmanifested.png").write_bytes(b"not-reviewed")
        with self.assertRaisesRegex(ValueError, "not present in review_manifest"):
            curate_pretrain_from_config(
                config, overwrite=True, finalize_review=True
            )

    def test_review_folder_rejects_archives_and_preserves_candidates_on_failure(self):
        output = self.root / "archive-review-output"
        config = self._config(output)
        curate_pretrain_from_config(config)
        review_root = self.root / "reviews" / output.name
        (review_root / "negative_reviewed" / "review-copy.zip").write_bytes(b"zip")
        with self.assertRaisesRegex(ValueError, "unsupported file or archive"):
            curate_pretrain_from_config(
                config, overwrite=True, finalize_review=True
            )
        self.assertEqual(
            2, len(list((review_root / "negative_candidates").rglob("*.png")))
        )
        self.assertFalse((review_root / "negative_removed_manifest.jsonl").exists())

    def test_removed_quarantine_manifests_and_idempotent_recovery(self):
        output = self.root / "transactional-review"
        config = self._config(output)
        curate_pretrain_from_config(config)
        review_root = self.root / "reviews" / output.name
        self._review_candidates(output, {"neg-overlap", "neg-clean"})

        first = curate_pretrain_from_config(
            config, overwrite=True, finalize_review=True
        )
        removed = read_jsonl(review_root / "negative_removed_manifest.jsonl")
        quarantine = read_jsonl(review_root / "negative_quarantine_manifest.jsonl")
        self.assertEqual([], removed)
        self.assertEqual(["neg-overlap"], [row["crop_id"] for row in quarantine])
        self.assertEqual(
            "negative_quarantine_manifest_v1", quarantine[0]["schema_version"]
        )
        self.assertTrue(
            (
                review_root
                / "negative_quarantine"
                / quarantine[0]["candidate_relative_path"]
            ).is_file()
        )
        with (review_root / "review_transaction.json").open(encoding="utf-8") as handle:
            transaction = json.load(handle)
        self.assertEqual("committed", transaction["status"])
        self.assertEqual("complete", transaction["candidate_cleanup"])
        self.assertEqual(
            {"expected": 2, "reviewed": 2, "admitted": 1, "quarantine": 1, "removed": 0},
            transaction["partitions"],
        )

        second = curate_pretrain_from_config(
            config, overwrite=True, finalize_review=True
        )
        self.assertEqual(first["counts"], second["counts"])
        dataset = {
            "labels": str(
                output / "05_labels" / "hand_training_labels_pretrain_multitask.jsonl"
            ),
            "curation_manifest": str(output / "qc" / "sha256_manifest.json"),
            "require_curation_schema": "pretrain_curation_v1",
        }
        verified = verify_curation_manifest({}, dataset)
        self.assertEqual(
            "committed", verified["negative_review_transaction"]["status"]
        )

    def test_prepared_transaction_recovers_without_early_candidate_cleanup(self):
        output = self.root / "prepared-recovery"
        config = self._config(output)
        curate_pretrain_from_config(config)
        review_root = self.root / "reviews" / output.name
        self._review_candidates(output, {"neg-clean"})
        with mock.patch(
            "hand_landmarker.pretrain_curation._complete_retained_negative_review_transaction",
            side_effect=RuntimeError("injected-after-snapshot-commit"),
        ):
            with self.assertRaisesRegex(RuntimeError, "injected"):
                curate_pretrain_from_config(
                    config, overwrite=True, finalize_review=True
                )
        self.assertTrue((review_root / "negative_candidates").is_dir())
        with (review_root / "review_transaction.json").open(encoding="utf-8") as handle:
            self.assertEqual("prepared", json.load(handle)["status"])

        recovered = curate_pretrain_from_config(
            config, overwrite=True, finalize_review=True
        )
        self.assertEqual(1, recovered["counts"]["included_confirmed_negatives"])
        self.assertFalse((review_root / "negative_candidates").exists())


if __name__ == "__main__":
    unittest.main()
