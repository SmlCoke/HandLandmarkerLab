import tempfile
import unittest
from pathlib import Path
from unittest import mock

from hand_landmarker.finetune_curation import (
    _check_leakage,
    _gold_source_selection,
    _gold_weights,
    _load_sources,
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
    def _source_config(self):
        return {
            "gold_source_descriptor_root": "/gold",
            "replay_source_descriptor_root": "/replay",
            "sources": {
                "dragon_gold": {
                    "discover_kind": "external_gold",
                    "enabled": "auto",
                    "required": False,
                },
                "new_recorded_gold": {
                    "discover_kind": "new_recorded_gold",
                    "enabled": "auto",
                    "required": False,
                },
                "pretrain_replay": {
                    "discover_kind": "pretrain_replay",
                    "enabled": True,
                    "required": True,
                },
            },
            "source_selection": {
                "default_gold_enabled": True,
                "gold": {"dragon_batch": False},
            },
        }

    def test_load_sources_authenticates_disabled_gold_but_excludes_it_from_training(self):
        config = self._source_config()
        dragon_path = Path("/gold/dragon/finetune_source.json")
        recorded_path = Path("/gold/recorded/finetune_source.json")
        replay_path = Path("/replay/pretrain/finetune_source.json")
        values = {
            dragon_path: {
                "source_id": "dragon_batch",
                "dataset_id": "dragon",
                "source_kind": "external_gold",
            },
            recorded_path: {
                "source_id": "new_recorded_r01",
                "dataset_id": "recorded",
                "source_kind": "new_recorded_gold",
            },
            replay_path: {
                "source_id": "pretrain_replay",
                "dataset_id": "replay",
                "source_kind": "pretrain_replay",
            },
        }
        with mock.patch(
            "hand_landmarker.finetune_curation._discover_descriptors",
            side_effect=[[dragon_path, recorded_path], [replay_path]],
        ), mock.patch(
            "hand_landmarker.finetune_curation.validate_finetune_source",
            side_effect=lambda path, *_: dict(values[path]),
        ) as validate, mock.patch(
            "hand_landmarker.finetune_curation.validate_source_set"
        ):
            loaded = _load_sources(config, [Path("/")])
        self.assertEqual(3, validate.call_count)
        self.assertEqual(
            ["dragon_batch", "new_recorded_r01"],
            [source["source_id"] for source in loaded["gold_all"]],
        )
        self.assertEqual(
            ["new_recorded_r01"],
            [source["source_id"] for source in loaded["gold"]],
        )
        decisions = {row["source_id"]: row for row in loaded["source_selection"]}
        self.assertEqual("source_disabled", decisions["dragon_batch"]["reason"])
        self.assertFalse(decisions["dragon_batch"]["enabled_for_training"])

    def test_replay_role_cannot_be_disabled(self):
        config = self._source_config()
        config["sources"]["pretrain_replay"]["enabled"] = False
        config["source_selection"]["gold"] = {}
        with mock.patch(
            "hand_landmarker.finetune_curation._discover_descriptors",
            side_effect=[[], []],
        ), self.assertRaisesRegex(ValueError, "must be enabled=true"):
            _load_sources(config, [Path("/")])

    def test_gold_sources_can_be_enabled_individually(self):
        sources = [
            {"source_id": "dragon_batch", "dataset_id": "dragon", "source_kind": "external_gold"},
            {"source_id": "new_recorded_r01", "dataset_id": "recorded", "source_kind": "new_recorded_gold"},
        ]
        selected, report = _gold_source_selection(
            {
                "source_selection": {
                    "default_gold_enabled": True,
                    "gold": {"dragon_batch": False},
                }
            },
            sources,
        )
        self.assertEqual(
            {"dragon_batch": False, "new_recorded_r01": True}, selected
        )
        self.assertEqual(
            {"dragon_batch": "explicit", "new_recorded_r01": "default"},
            {row["source_id"]: row["selection_origin"] for row in report},
        )

    def test_gold_source_selection_rejects_unknown_ids_and_non_boolean_values(self):
        sources = [{"source_id": "known", "dataset_id": "known", "source_kind": "external_gold"}]
        with self.assertRaisesRegex(ValueError, "undiscovered"):
            _gold_source_selection(
                {"source_selection": {"gold": {"typo": False}}}, sources
            )
        with self.assertRaisesRegex(ValueError, "must be boolean"):
            _gold_source_selection(
                {"source_selection": {"gold": {"known": "false"}}}, sources
            )

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
