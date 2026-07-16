import json
import hashlib
import os
import tempfile
import unittest
from pathlib import Path

from hand_landmarker.finetune_replay import build_replay_source
from hand_landmarker.finetune_source import validate_finetune_source, validate_gold_aggregate
from hand_landmarker.finetune_source import _safe_dir, _validate_hash_manifest
from hand_landmarker.finetune_curation import _discover_descriptors
from hand_landmarker.io_utils import sha256_file, write_json, write_jsonl


def confirmed_negative(path):
    return {
        "crop_id": "d:n",
        "global_crop_id": "d:n",
        "dataset_id": "d",
        "source_crop_id": "n",
        "crop_path": str(path.resolve()),
        "image_sha256": sha256_file(path),
        "sample_type": "NEG_RUNTIME_CANDIDATE",
        "sampling_bucket": "pseudo:NEG_RUNTIME_CANDIDATE",
        "sampling_weight": 1.0,
        "supervision_tier": "pseudo",
        "hand_presence": {"present": False},
        "handedness": {"label": "unknown"},
        "landmarks_crop_norm": [],
        "pretrain_curation": {
            "action": "INCLUDE_CONFIRMED_NEGATIVE",
            "review": {
                "decision": "CONFIRMED_NEGATIVE",
                "reviewer": "human",
                "reviewed_at": "2026-01-01T00:00:00+00:00",
                "review_method": "visual",
                "review_image_sha256": sha256_file(path),
            },
        },
    }


class FinetuneSourceCoreTest(unittest.TestCase):
    def test_gold_aggregate_rejects_overlapping_semantic_partitions(self):
        with tempfile.TemporaryDirectory() as directory:
            finetune = Path(directory) / "finetune-id"
            source_descriptor = finetune / "sources" / "gold" / "s" / "finetune_source.json"
            source_descriptor.parent.mkdir(parents=True)
            source_descriptor.write_text("{}\n", encoding="utf-8")
            merged = finetune / "hmlf_gold_merged"
            row = {"global_crop_id": "g:1", "selection_action": "include"}
            paths = {}
            for name, rows in (
                ("catalog", [row]),
                ("included", [row]),
                ("excluded", [row]),
            ):
                path = merged / (name + ".jsonl")
                write_jsonl(path, rows)
                paths[name] = {
                    "path": path.relative_to(merged).as_posix(),
                    "sha256": sha256_file(path),
                    "count": len(rows),
                }
            report = merged / "report.json"
            report.write_text("{}\n", encoding="utf-8")
            paths["report"] = {"path": "report.json", "sha256": sha256_file(report), "count": 1}
            descriptor = merged / "hmlf_gold_aggregate.json"
            write_json(
                descriptor,
                {
                    "schema_version": "hmlf_gold_aggregate_v1",
                    "finetune_id": "finetune-id",
                    "source_descriptors": [
                        {
                            "source_id": "s",
                            "path": source_descriptor.relative_to(finetune).as_posix(),
                            "sha256": sha256_file(source_descriptor),
                        }
                    ],
                    "artifacts": paths,
                    "counts": {"catalog": 1, "included": 1, "excluded": 1},
                    "duplicate_count": 0,
                    "conflict_count": 0,
                },
            )
            sources = [
                {
                    "source_id": "s",
                    "path": str(source_descriptor.resolve()),
                    "sha256": sha256_file(source_descriptor),
                }
            ]
            with self.assertRaisesRegex(ValueError, "overlap"):
                validate_gold_aggregate(descriptor, finetune, sources)

    def test_hash_manifest_authenticates_physical_file_and_rejects_traversal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            physical = root / "roi.bin"
            physical.write_bytes(b"original")
            manifest = root / "hashes.jsonl"

            def entry_for(row):
                write_jsonl(manifest, [row])
                aggregate = hashlib.sha256(
                    "{}:{}\n".format(row["path"], row["sha256"]).encode("utf-8")
                ).hexdigest()
                return {
                    "sha256_manifest": "hashes.jsonl",
                    "manifest_sha256": sha256_file(manifest),
                    "aggregate_sha256": aggregate,
                    "count": 1,
                }

            row = {"path": "roi.bin", "sha256": sha256_file(physical)}
            _validate_hash_manifest(root, entry_for(row), "test", root)
            physical.write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "physical file SHA mismatch"):
                _validate_hash_manifest(root, entry_for(row), "test", root)
            outside = root.parent / (root.name + "-outside.bin")
            outside.write_bytes(b"outside")
            try:
                traversal = {"path": "../" + outside.name, "sha256": sha256_file(outside)}
                with self.assertRaisesRegex(ValueError, "non-normalized relative path"):
                    _validate_hash_manifest(root, entry_for(traversal), "test", root)
            finally:
                outside.unlink()

    def test_nested_symlink_is_rejected_by_directory_and_discovery_gates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            real = root / "real"
            real.mkdir()
            (real / "finetune_source.json").write_text("{}", encoding="utf-8")
            link = root / "linked"
            try:
                os.symlink(str(real), str(link), target_is_directory=True)
            except (OSError, NotImplementedError) as exc:
                self.skipTest("Directory symlinks are unavailable: {}".format(exc))
            with self.assertRaisesRegex(ValueError, "symlink"):
                _safe_dir(root, "linked", "test")
            discovered = list(root.rglob("finetune_source.json"))
            if any("linked" in path.parts for path in discovered):
                with self.assertRaisesRegex(ValueError, "symlink"):
                    _discover_descriptors(root)

    def test_replay_round_trip_and_tamper_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            crops = root / "train_sources"
            crops.mkdir()
            image = crops / "n.png"
            image.write_bytes(b"roi")
            parent = root / "parent.json"
            parent.write_text(json.dumps({"schema_version": "pretrain_curation_v1"}), encoding="utf-8")
            output = root / "pretrain_replay"
            build_replay_source(
                [confirmed_negative(image)],
                {
                    "enabled": True,
                    "max_records": 10,
                    "include_all_confirmed_negatives": True,
                    "positive_fractions": {"POS_RUNTIME": 0.75, "POS_LOW_PALM": 0.25},
                    "salt": "test",
                },
                output,
                parent,
                crops,
                "v2-pretrain-r3",
                "test-version",
            )
            validated = validate_finetune_source(output / "finetune_source.json", [crops])
            self.assertEqual(validated["source_kind"], "pretrain_replay")
            self.assertEqual(len(validated["rows"]), 1)
            image.write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "SHA mismatch"):
                validate_finetune_source(output / "finetune_source.json", [crops])


if __name__ == "__main__":
    unittest.main()
