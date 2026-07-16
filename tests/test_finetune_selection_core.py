import hashlib
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from hand_landmarker.finetune_curation import build_smoke_snapshot
from hand_landmarker.finetune_replay import select_replay_rows
from hand_landmarker.finetune_selection import (
    disagreement_metrics,
    largest_remainder,
    select_negative_removed,
)
from hand_landmarker.io_utils import sha256_file, write_json, write_jsonl
from hand_landmarker.io_utils import write_image
from hand_landmarker.runtime import HandPrediction
from hand_landmarker.train_prediction import predict_rows
from scripts.check_finetune_smoke import validate_smoke_snapshot
from scripts.prepare_finetune_sources import _restore_rows, _validate_registry_report


def points(scale=1.0):
    return [
        {"id": index, "x": 0.1 + scale * (index % 5) * 0.03, "y": 0.2 + scale * (index // 5) * 0.04}
        for index in range(21)
    ]


def row(identity, tier, sample_type, handedness="Left", present=True):
    return {
        "crop_id": identity,
        "global_crop_id": identity,
        "dataset_id": identity.split(":")[0],
        "source_crop_id": identity.split(":")[-1],
        "source_group_id": "group:" + identity,
        "source_sequence_id": "seq:" + identity,
        "supervision_tier": tier,
        "sample_type": sample_type,
        "sampling_bucket": tier + ":" + sample_type,
        "sampling_weight": 1.0,
        "hand_presence": {"present": present},
        "handedness": {"label": handedness if present else "unknown"},
        "landmarks_crop_norm": points() if present else [],
        "image_sha256": hashlib.sha256(identity.encode()).hexdigest(),
    }


class FinetuneSelectionCoreTest(unittest.TestCase):
    def test_selection_rejects_symlink_ancestor(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            real = root / "real"
            real.mkdir()
            for name in ("manifest.jsonl", "draft.jsonl", "crop.png"):
                (real / name).write_bytes(name.encode())
            link = root / "linked"
            try:
                os.symlink(str(real), str(link), target_is_directory=True)
            except (OSError, NotImplementedError) as exc:
                self.skipTest("Directory symlinks are unavailable: {}".format(exc))
            identity = "d:1"
            catalog = [{
                "crop_id": identity,
                "global_crop_id": identity,
                "dataset_id": "d",
                "source_crop_id": "1",
                "sample_type": "NEG_RUNTIME_CANDIDATE",
                "parent_manifest_path": str((link / "manifest.jsonl").absolute()),
                "parent_manifest_sha256": sha256_file(real / "manifest.jsonl"),
                "parent_draft_path": str((link / "draft.jsonl").absolute()),
                "parent_draft_sha256": sha256_file(real / "draft.jsonl"),
                "crop_path": str((link / "crop.png").absolute()),
                "image_sha256": sha256_file(real / "crop.png"),
            }]
            removed = [{
                "schema_version": "negative_removed_manifest_v1",
                "crop_id": identity,
                "partition": "negative_removed",
                "sample_type": "NEG_RUNTIME_CANDIDATE",
            }]
            with self.assertRaisesRegex(ValueError, "symlink"):
                select_negative_removed(
                    removed,
                    catalog,
                    {
                        "enabled": True,
                        "max_items": 1,
                        "per_dataset_max": 1,
                        "sample_type_fractions": {"NEG_RUNTIME_CANDIDATE": 1.0},
                    },
                )

    def test_hlmf_registry_sibling_report_is_authenticated(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry = root / "pretrain_source_registry.jsonl"
            write_jsonl(registry, [{"schema_version": "pretrain_source_registry_v1", "global_crop_id": "d:1"}])
            report = root / "pretrain_source_registry_report.json"
            write_json(
                report,
                {
                    "schema_version": "pretrain_source_registry_v1",
                    "status": "ok",
                    "rows": 1,
                    "registry": {"path": str(registry.resolve()), "sha256": sha256_file(registry), "count": 1},
                },
            )
            self.assertEqual(_validate_registry_report(registry)["count"], 1)
            registry.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "binding mismatch"):
                _validate_registry_report(registry)

    def test_registry_restore_hashes_each_physical_file_once(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.jsonl"
            draft = root / "draft.jsonl"
            manifest.write_text("{}\n", encoding="utf-8")
            draft.write_text("{}\n", encoding="utf-8")
            registry = []
            canonical = []
            for index in range(2):
                crop = root / "crop{}.png".format(index)
                crop.write_bytes(str(index).encode("ascii"))
                identity = "d:{}".format(index)
                canonical.append(
                    {
                        "global_crop_id": identity,
                        "dataset_id": "d",
                        "source_crop_id": str(index),
                    }
                )
                registry.append(
                    {
                        "schema_version": "pretrain_source_registry_v1",
                        "global_crop_id": identity,
                        "dataset_id": "d",
                        "source_crop_id": str(index),
                        "parent_manifest_path": str(manifest.resolve()),
                        "parent_manifest_sha256": sha256_file(manifest),
                        "parent_draft_path": str(draft.resolve()),
                        "parent_draft_sha256": sha256_file(draft),
                        "parent_crop_path": str(crop.resolve()),
                        "image_sha256": sha256_file(crop),
                    }
                )
            cache = {}
            real_sha256 = sha256_file
            with mock.patch(
                "scripts.prepare_finetune_sources.sha256_file",
                wraps=real_sha256,
            ) as hashed:
                _restore_rows(canonical, registry, cache)
                _restore_rows(canonical, registry, cache)
            self.assertEqual(hashed.call_count, 4)
            self.assertEqual(len(cache), 4)

    def test_largest_remainder_is_exact_and_stable(self):
        self.assertEqual(largest_remainder(7, {"b": 1, "a": 1}), {"b": 3, "a": 4})
        self.assertEqual(sum(largest_remainder(301, {"x": 0.6, "y": 0.4}).values()), 301)

    def test_negative_removed_request_and_global_dataset_cap(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.jsonl"
            draft = root / "draft.jsonl"
            manifest.write_text("{}\n", encoding="utf-8")
            draft.write_text("{}\n", encoding="utf-8")
            catalog = []
            removed = []
            for dataset in ("a", "b"):
                for index, sample_type in enumerate(("NEG_RUNTIME_CANDIDATE", "NEG_LOW_PALM_CANDIDATE")):
                    crop = root / (dataset + str(index) + ".png")
                    crop.write_bytes((dataset + str(index)).encode())
                    identity = dataset + ":" + str(index)
                    catalog.append(
                        {
                            "crop_id": identity,
                            "global_crop_id": identity,
                            "dataset_id": dataset,
                            "source_crop_id": str(index),
                            "sample_type": sample_type,
                            "source_group_id": identity,
                            "parent_manifest_path": str(manifest.resolve()),
                            "parent_manifest_sha256": sha256_file(manifest),
                            "parent_draft_path": str(draft.resolve()),
                            "parent_draft_sha256": sha256_file(draft),
                            "crop_path": str(crop.resolve()),
                            "image_sha256": sha256_file(crop),
                        }
                    )
                    removed.append(
                        {
                            "schema_version": "negative_removed_manifest_v1",
                            "crop_id": identity,
                            "dataset_id": dataset,
                            "source_crop_id": str(index),
                            "sample_type": sample_type,
                            "partition": "negative_removed",
                            "source_image_sha256": sha256_file(crop),
                        }
                    )
            real_sha256 = sha256_file
            with mock.patch(
                "hand_landmarker.finetune_selection.sha256_file",
                wraps=real_sha256,
            ) as hashed:
                selected, _ = select_negative_removed(
                    removed,
                    catalog,
                    {
                        "enabled": True,
                        "max_items": 4,
                        "per_dataset_max": 1,
                        "sample_type_fractions": {
                            "NEG_RUNTIME_CANDIDATE": 0.5,
                            "NEG_LOW_PALM_CANDIDATE": 0.5,
                        },
                        "salt": "test",
                    },
                )
            self.assertEqual(len(selected), 2)
            self.assertEqual({item["parent_dataset_id"] for item in selected}, {"a", "b"})
            self.assertTrue(all(Path(item["parent_crop_path"]).is_absolute() for item in selected))
            hashed_paths = [Path(call.args[0]).resolve() for call in hashed.call_args_list]
            self.assertEqual(hashed_paths.count(manifest.resolve()), 1)
            self.assertEqual(hashed_paths.count(draft.resolve()), 1)

    def test_disagreement_identity_is_zero(self):
        teacher = {"landmarks_crop_norm": points()}
        prediction = {
            "student_landmarks_crop_norm": points(),
            "student_hand_flag": 1.0,
        }
        metrics = disagreement_metrics(teacher, prediction)
        self.assertAlmostEqual(metrics["mean_nme"], 0.0)
        self.assertAlmostEqual(metrics["collapse_log_ratio"], 0.0)

    def test_train_prediction_batch_matches_single_row(self):
        import numpy as np

        class Predictor:
            def predict(self, images, batch_size=64):
                result = []
                for image in images:
                    value = float(np.asarray(image).mean()) / 255.0
                    result.append(
                        HandPrediction(
                            landmarks_norm=tuple((value, value) for _ in range(21)),
                            hand_flag_score=0.9,
                            handedness_score=0.4,
                        )
                    )
                return result

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = []
            for index, value in enumerate((32, 192)):
                path = root / (str(index) + ".bmp")
                write_image(path, np.full((4, 4), value, dtype=np.uint8))
                rows.append(
                    {
                        "global_crop_id": "d:{}".format(index),
                        "dataset_id": "d",
                        "source_crop_id": str(index),
                        "crop_path": str(path.resolve()),
                        "image_sha256": sha256_file(path),
                    }
                )
            self.assertEqual(predict_rows(rows, Predictor(), 1), predict_rows(rows, Predictor(), 2))

    def test_replay_keeps_all_confirmed_negatives(self):
        rows = [row("d:p{}".format(i), "pseudo", "POS_RUNTIME") for i in range(10)]
        for index, sample_type in enumerate(("NEG_RUNTIME_CANDIDATE", "NEG_LOW_PALM_CANDIDATE")):
            value = row("d:n{}".format(index), "pseudo", sample_type, present=False)
            value["pretrain_curation"] = {
                "action": "INCLUDE_CONFIRMED_NEGATIVE",
                "review": {
                    "decision": "CONFIRMED_NEGATIVE",
                    "reviewer": "r",
                    "reviewed_at": "2026-01-01T00:00:00+00:00",
                    "review_method": "visual",
                    "review_image_sha256": "a" * 64,
                },
            }
            rows.append(value)
        selected, report = select_replay_rows(
            rows,
            {
                "enabled": True,
                "max_records": 8,
                "include_all_confirmed_negatives": True,
                "positive_fractions": {"POS_RUNTIME": 0.75, "POS_LOW_PALM": 0.25},
                "salt": "test",
            },
        )
        self.assertEqual(len(selected), 8)
        self.assertEqual(report["selected_confirmed_negative"], 2)
        with self.assertRaisesRegex(ValueError, "must be true"):
            select_replay_rows(
                rows,
                {
                    "enabled": True,
                    "max_records": 8,
                    "include_all_confirmed_negatives": False,
                    "positive_fractions": {"POS_RUNTIME": 0.75, "POS_LOW_PALM": 0.25},
                },
            )

    def test_smoke_snapshot_exact_contract(self):
        rows = []
        handedness = ("Left", "Right", "unknown")
        for index in range(96):
            rows.append(row("g:p{}".format(index), "gold", "POS_RUNTIME", handedness[index % 3]))
        for index in range(96):
            rows.append(row("p:p{}".format(index), "pseudo", "POS_LOW_PALM", handedness[index % 3]))
        for index in range(32):
            rows.append(row("p:r{}".format(index), "pseudo", "NEG_RUNTIME_CANDIDATE", present=False))
            rows.append(row("p:l{}".format(index), "pseudo", "NEG_LOW_PALM_CANDIDATE", present=False))
        smoke, selection, report = build_smoke_snapshot(rows, "fixed")
        self.assertEqual(len(smoke), 256)
        self.assertEqual(len(selection), 256)
        self.assertEqual(report["quotas"]["gold_positive"], 96)
        self.assertEqual(report["gold_negative_policy"], "redistributed_to_gold_positive")
        self.assertEqual(smoke, build_smoke_snapshot(rows, "fixed")[0])
        validated = validate_smoke_snapshot(rows, smoke, selection)
        self.assertEqual(validated["record_count"], 256)


if __name__ == "__main__":
    unittest.main()
