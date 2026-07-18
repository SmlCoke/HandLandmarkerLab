import json
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
    prepare_gold_selection_from_config,
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
    def _source_config(self, root, decisions=None):
        gold = root / "GoldSource"
        replay = root / "replay"
        selection = root / "gold_selection.yaml"
        gold.mkdir(parents=True, exist_ok=True)
        selection.write_text(
            json.dumps(
                {
                    "schema_version": "hlml_gold_selection_v1",
                    "finetune_id": "ft-test",
                    "gold_repository_root": str(gold.resolve()),
                    "sources": decisions or {},
                }
            ),
            encoding="utf-8",
        )
        return {
            "finetune_id": "ft-test",
            "gold_source_descriptor_root": str(gold),
            "replay_source_descriptor_root": str(replay),
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
            "source_selection": {"manifest": str(selection)},
        }

    def _decision(self, source_id, kind, domain, enabled, digest):
        return {
            "enabled": enabled,
            "source_kind": kind,
            "domain": domain,
            "descriptor": f"{domain}/{source_id}/published/finetune_source.json",
            "descriptor_sha256": digest,
        }

    def test_load_sources_authenticates_disabled_gold_but_excludes_it_from_training(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        dragon_path = root / "GoldSource/dragon/dragon_batch/published/finetune_source.json"
        recorded_path = root / "GoldSource/new_recorded_gold/new_recorded_r01/published/finetune_source.json"
        replay_path = root / "replay/pretrain/finetune_source.json"
        config = self._source_config(
            root,
            {
                "dragon_batch": self._decision("dragon_batch", "external_gold", "dragon", False, "d" * 64),
                "new_recorded_r01": self._decision("new_recorded_r01", "new_recorded_gold", "new_recorded_gold", True, "n" * 64),
            },
        )
        values = {
            dragon_path: {
                "source_id": "dragon_batch",
                "dataset_id": "dragon",
                "source_kind": "external_gold",
                "path": str(dragon_path),
                "sha256": "d" * 64,
                "descriptor": {"sampling": {"source_weight": 0.5}},
            },
            recorded_path: {
                "source_id": "new_recorded_r01",
                "dataset_id": "recorded",
                "source_kind": "new_recorded_gold",
                "path": str(recorded_path),
                "sha256": "n" * 64,
                "descriptor": {"sampling": {"source_weight": 1.5}},
            },
            replay_path: {
                "source_id": "pretrain_replay",
                "dataset_id": "replay",
                "source_kind": "pretrain_replay",
                "path": str(replay_path),
                "sha256": "r" * 64,
                "descriptor": {},
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
        self.assertEqual(
            {"sampling": {"source_weight": 1.5}},
            loaded["gold"][0]["descriptor"],
        )
        decisions = {row["source_id"]: row for row in loaded["source_selection"]}
        self.assertEqual("source_disabled", decisions["dragon_batch"]["reason"])
        self.assertFalse(decisions["dragon_batch"]["enabled_for_training"])

    def test_replay_role_cannot_be_disabled(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self._source_config(Path(directory))
            config["sources"]["pretrain_replay"]["enabled"] = False
            with mock.patch(
                "hand_landmarker.finetune_curation._discover_descriptors",
                side_effect=[[], []],
            ), self.assertRaisesRegex(ValueError, "must be enabled=true"):
                _load_sources(config, [Path("/")])

    def test_gold_sources_can_be_enabled_individually(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = {
                "dragon_batch": root / "GoldSource/dragon/dragon_batch/published/finetune_source.json",
                "new_recorded_r01": root / "GoldSource/new_recorded_gold/new_recorded_r01/published/finetune_source.json",
            }
            sources = [
                {"source_id": "dragon_batch", "dataset_id": "dragon", "source_kind": "external_gold", "path": str(paths["dragon_batch"]), "sha256": "d" * 64},
                {"source_id": "new_recorded_r01", "dataset_id": "recorded", "source_kind": "new_recorded_gold", "path": str(paths["new_recorded_r01"]), "sha256": "n" * 64},
            ]
            config = self._source_config(
                root,
                {
                    "dragon_batch": self._decision("dragon_batch", "external_gold", "dragon", False, "d" * 64),
                    "new_recorded_r01": self._decision("new_recorded_r01", "new_recorded_gold", "new_recorded_gold", True, "n" * 64),
                },
            )
            selected, report, provenance = _gold_source_selection(config, sources)
            self.assertEqual({"dragon_batch": False, "new_recorded_r01": True}, selected)
            self.assertTrue(provenance["sha256"])
            self.assertEqual(
                {"dragon_batch": "explicit_manifest", "new_recorded_r01": "explicit_manifest"},
                {row["source_id"]: row["selection_origin"] for row in report},
            )

    def test_gold_source_selection_rejects_missing_and_non_boolean_decisions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "GoldSource/dragon/known/published/finetune_source.json"
            source = {"source_id": "known", "dataset_id": "known", "source_kind": "external_gold", "path": str(path), "sha256": "a" * 64}
            config = self._source_config(root, {})
            with self.assertRaisesRegex(ValueError, "decide every published batch"):
                _gold_source_selection(config, [source])
            config = self._source_config(
                root,
                {"known": self._decision("known", "external_gold", "dragon", False, "a" * 64)},
            )
            manifest = Path(config["source_selection"]["manifest"])
            value = json.loads(manifest.read_text(encoding="utf-8"))
            value["sources"]["known"]["enabled"] = "false"
            manifest.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "must be boolean"):
                _gold_source_selection(config, [source])

    def test_prepare_gold_selection_freezes_every_published_batch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gold = root / "GoldSource"
            dragon = gold / "dragon/dragon_a/published/finetune_source.json"
            recorded = gold / "new_recorded_gold/recorded_r01/published/finetune_source.json"
            for path in (dragon, recorded):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("{}\n", encoding="utf-8")
            selection = root / "finetune/ft-test/gold_selection.yaml"
            config = {
                "task": "curate_finetune",
                "finetune_id": "ft-test",
                "gold_source_descriptor_root": str(gold),
                "source_selection": {"manifest": str(selection)},
            }
            values = {
                dragon: {"source_id": "dragon_a", "dataset_id": "dragon_a", "source_kind": "external_gold", "path": str(dragon.resolve()), "sha256": "d" * 64},
                recorded: {"source_id": "recorded_r01", "dataset_id": "recorded_r01", "source_kind": "new_recorded_gold", "path": str(recorded.resolve()), "sha256": "r" * 64},
            }
            with mock.patch(
                "hand_landmarker.finetune_curation.validate_finetune_source",
                side_effect=lambda path, *_: dict(values[path]),
            ):
                report = prepare_gold_selection_from_config(config, ["recorded_r01"])
            self.assertEqual(["recorded_r01"], report["enabled_source_ids"])
            self.assertEqual(["dragon_a"], report["disabled_source_ids"])
            import yaml

            manifest = yaml.safe_load(selection.read_text(encoding="utf-8"))
            self.assertFalse(manifest["sources"]["dragon_a"]["enabled"])
            self.assertTrue(manifest["sources"]["recorded_r01"]["enabled"])
            with self.assertRaisesRegex(FileExistsError, "immutable"):
                prepare_gold_selection_from_config(config, [])

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
