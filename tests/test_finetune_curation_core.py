import tempfile
import unittest
from pathlib import Path

from hand_landmarker.finetune_curation import (
    _check_leakage,
    _gold_weights,
    merge_gold_over_replay,
    verify_finetune_curation_manifest,
)
from hand_landmarker.io_utils import sha256_file, write_json, write_jsonl


def sample(identity, dataset, tier="gold", sample_type="POS_RUNTIME", image_sha=None, sequence=None):
    return {
        "crop_id": identity,
        "global_crop_id": identity,
        "dataset_id": dataset,
        "source_crop_id": identity,
        "supervision_tier": tier,
        "sample_type": sample_type,
        "sampling_bucket": tier + ":" + sample_type,
        "sampling_weight": 1.0,
        "source_sequence_id": sequence or identity,
        "image_sha256": image_sha or (identity[0] * 64),
    }


class FinetuneCurationCoreTest(unittest.TestCase):
    def test_training_manifest_verifier_authenticates_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            labels = root / "05_labels" / "hand_training_labels_finetune.jsonl"
            manifest = root / "qc" / "sha256_manifest.json"
            curator = root / "curate.yaml"
            curator.write_text("task: curate_finetune\n", encoding="utf-8")
            write_jsonl(labels, [{"crop_id": "x"}])
            write_json(
                manifest,
                {
                    "schema_version": "finetune_curation_v1",
                    "output_dir": str(root.resolve()),
                    "config_path": str(curator.resolve()),
                    "config_sha256": sha256_file(curator),
                    "artifacts": {
                        "05_labels/hand_training_labels_finetune.jsonl": {
                            "sha256": sha256_file(labels),
                            "count": 1,
                        }
                    },
                    "gold_aggregate": {"sha256": "a" * 64},
                    "images": {"count": 1, "aggregate_sha256": "b" * 64},
                },
            )
            config = {"_meta": {"repo_root": str(root)}}
            dataset = {
                "labels": str(labels),
                "curation_manifest": str(manifest),
                "require_curation_schema": "finetune_curation_v1",
            }
            verified = verify_finetune_curation_manifest(config, dataset)
            self.assertEqual(verified["training_labels_sha256"], sha256_file(labels))
            labels.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SHA mismatch"):
                verify_finetune_curation_manifest(config, dataset)

    def test_gold_supersedes_replay_by_parent_identity(self):
        gold = sample("gold:new", "gold")
        gold["parent_global_crop_id"] = "old:1"
        replay = sample("old:1", "replay", tier="pseudo")
        other = sample("old:2", "replay", tier="pseudo")
        kept, excluded = merge_gold_over_replay([gold], [replay, other])
        self.assertEqual([row["global_crop_id"] for row in kept], ["old:2"])
        self.assertEqual(excluded[0]["finetune_curation_action"], "SUPERSEDED_BY_GOLD")
        self.assertEqual(excluded[0]["superseded_by_gold"], ["gold:new"])

    def test_leakage_detects_image_and_source_group(self):
        train = sample("train:1", "train", image_sha="f" * 64)
        evaluation = sample("test:1", "test", image_sha="f" * 64)
        with self.assertRaisesRegex(ValueError, "leakage"):
            _check_leakage([train], [evaluation])
        evaluation["image_sha256"] = "e" * 64
        train["source_group_id"] = "session:1"
        evaluation["source_group_id"] = "session:1"
        with self.assertRaisesRegex(ValueError, "leakage"):
            _check_leakage([train], [evaluation])

    def test_source_weights_renormalize_present_roles_and_balance_sequences(self):
        sources = [
            {
                "dataset_id": "dragon",
                "source_id": "dragon",
                "source_kind": "external_gold",
                "role": "dragon_gold",
                "descriptor": {},
            },
            {
                "dataset_id": "hard",
                "source_id": "hard",
                "source_kind": "reviewed_hard_gold",
                "role": "negative_removed_gold",
                "descriptor": {},
            },
        ]
        roles = {
            "dragon_gold": {"target_gold_weight": 0.60},
            "negative_removed_gold": {"target_gold_weight": 0.20},
            "disagreement_gold": {"target_gold_weight": 0.15},
        }
        rows = [
            sample("d:1", "dragon", sequence="clip-a"),
            sample("d:2", "dragon", sequence="clip-a"),
            sample("d:3", "dragon", sequence="clip-b"),
            sample("h:1", "hard", sequence="hard-1"),
        ]
        report = _gold_weights(rows, sources, roles)
        self.assertAlmostEqual(report["effective_role_weights"]["dragon_gold"], 0.75)
        self.assertAlmostEqual(report["effective_role_weights"]["negative_removed_gold"], 0.25)
        self.assertAlmostEqual(sum(row["sampling_weight"] for row in rows), 1.0)
        self.assertAlmostEqual(rows[0]["sampling_weight"], rows[1]["sampling_weight"])
        self.assertAlmostEqual(rows[0]["sampling_weight"] + rows[1]["sampling_weight"], rows[2]["sampling_weight"])


if __name__ == "__main__":
    unittest.main()
