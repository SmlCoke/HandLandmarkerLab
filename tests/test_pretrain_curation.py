import json
import tempfile
import unittest
from pathlib import Path

from hand_landmarker.inspect import DatasetContractError, audit_canonical_dataset
from hand_landmarker.io_utils import read_jsonl, sha256_file, write_jsonl
from hand_landmarker.pretrain_curation import (
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
                "reviewer": "unit-test-team",
            },
            "output": {
                "dir": str(output),
                "overwrite": False,
            },
        }

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
                "train_sources_reference",
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
        self.assertEqual("train_sources_reference", manifest["images"]["storage"])
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
        review_manifest = read_jsonl(review_root / "review_manifest.jsonl")
        overlap = next(row for row in review_manifest if row["crop_id"] == "neg-overlap")
        (review_root / "negative_candidates" / overlap["candidate_relative_path"]).unlink()

        report = curate_pretrain_from_config(
            config, overwrite=True, finalize_review=True
        )
        self.assertEqual(1, report["counts"]["included_confirmed_negatives"])
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
        report = curate_pretrain_from_config(
            config, overwrite=True, finalize_review=True
        )
        self.assertEqual(2, report["negative_review_workspace"]["retained_confirmed_count"])
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
        manifest = read_jsonl(review_root / "review_manifest.jsonl")
        candidate = review_root / "negative_candidates" / manifest[0]["candidate_relative_path"]
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
        candidates = self.root / "reviews" / output.name / "negative_candidates"
        (candidates / "unmanifested.png").write_bytes(b"not-reviewed")
        with self.assertRaisesRegex(ValueError, "not present in review_manifest"):
            curate_pretrain_from_config(
                config, overwrite=True, finalize_review=True
            )


if __name__ == "__main__":
    unittest.main()
