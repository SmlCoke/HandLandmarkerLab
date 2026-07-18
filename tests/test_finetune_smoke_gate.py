import copy
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from hand_landmarker.io_utils import sha256_file, write_json, write_jsonl
from scripts.check_finetune_smoke import (
    FIXED_GATE,
    FIXED_SMOKE_TYPE_FRACTIONS,
    _EXPECTED_INTERFACE,
    _jsonable,
    _metric_head_weights,
    compare_smoke_and_full_configs,
    validate_epoch_plan_coverage,
    validate_smoke_snapshot,
    verify_curation_binding,
    verify_smoke_run_provenance,
)


class SmokeRuntimeWeightTests(unittest.TestCase):
    def test_metric_weights_follow_semantic_mapping_order(self):
        weights = {
            "hand_flag": "flag",
            "structure": "structure",
            "landmarks": "points",
            "handedness": "side",
        }
        self.assertEqual(
            _metric_head_weights(weights), ("points", "flag", "side")
        )
        with self.assertRaisesRegex(ValueError, "missing"):
            _metric_head_weights({"landmarks": 1, "hand_flag": 1})


def _full_config(root: Path):
    return {
        "schema_version": 1,
        "task": "train",
        "stage": "finetune",
        "experiment": {"name": "full", "seed": 7},
        "data": {
            "data_root": str(root / "snapshot"),
            "labels": str(root / "snapshot" / "05_labels" / "full.jsonl"),
            "curation_manifest": str(root / "snapshot" / "qc" / "sha256_manifest.json"),
            "require_curation_schema": "finetune_curation_v1",
            "require_training_stage": "finetune",
            "image_size": [256, 256],
            "channels": 1,
            "color_mode": "grayscale",
            "input_layout": "NCHW",
            "input_dtype": "float32",
            "input_scale": 1.0 / 255.0,
            "input_offset": 0.0,
        },
        "model": {
            "version": "v2",
            "checkpoint_stage": "finetune",
            "input_shape": [1, 256, 256],
            "input_layout": "NCHW",
            "num_iterations": [2, 2, 3, 4, 4, 6, 6],
            "output_order": ["landmarks", "hand_flag", "handedness"],
            "output_sizes": {"landmarks": 42, "hand_flag": 1, "handedness": 1},
        },
        "targets": {
            "num_landmarks": 21,
            "landmark_field": "landmarks_crop_norm",
            "landmark_space": "normalized_crop_xy",
            "landmark_order": "id_0_to_20_interleaved_xy",
            "presence_field": "hand_presence.present",
            "handedness_field": "handedness.label",
            "handedness_encoding": {"Left": 0, "Right": 1, "unknown": None},
        },
        "training": {
            "epochs": 40,
            "batch_size": 64,
            "initial_checkpoint": str(root / "initial.weights.h5"),
            "resume_checkpoint": None,
            "optimizer": {"name": "adam", "learning_rate": 1e-5},
            "checkpoint": {"monitor": "val_landmark_mae", "mode": "min"},
            "learning_rate_schedule": {
                "name": "reduce_on_plateau",
                "monitor": "val_landmark_mae",
                "mode": "min",
                "patience": 3,
            },
            "early_stopping": {
                "enabled": True,
                "monitor": "val_landmark_mae",
                "mode": "min",
                "patience": 8,
            },
        },
        "sampling": {
            "epoch_size": 12000,
            "sample_type_fractions_by_tier": {
                "gold": {
                    "POS_RUNTIME": 0.70,
                    "POS_LOW_PALM": 0.20,
                    "NEG_RUNTIME_CANDIDATE": 0.07,
                    "NEG_LOW_PALM_CANDIDATE": 0.03,
                },
                "pseudo": {
                    "POS_RUNTIME": 0.72,
                    "POS_LOW_PALM": 0.18,
                    "NEG_RUNTIME_CANDIDATE": 0.06,
                    "NEG_LOW_PALM_CANDIDATE": 0.04,
                },
            },
        },
        "losses": {
            "landmarks": {"name": "huber", "delta": 0.05, "coefficient": 20.0},
            "hand_flag": {
                "name": "binary_crossentropy",
                "from_logits": False,
                "coefficient": 0.25,
            },
            "handedness": {
                "name": "binary_crossentropy",
                "from_logits": False,
                "coefficient": 0.02,
            },
            "honor_record_loss_weights": True,
        },
        "augmentation": {"enabled": True},
        "validation": {"enabled": True},
        "outputs": {"run_dir": str(root / "full_run")},
    }


def _smoke_config(root: Path):
    value = copy.deepcopy(_full_config(root))
    value["experiment"]["name"] = "smoke"
    value["data"]["labels"] = str(root / "snapshot" / "05_labels" / "smoke.jsonl")
    value["training"]["epochs"] = 300
    value["training"]["batch_size"] = 32
    value["training"]["optimizer"]["learning_rate"] = 1.0e-3
    for component in ("checkpoint", "learning_rate_schedule", "early_stopping"):
        value["training"][component]["monitor"] = "total_loss"
        value["training"][component]["mode"] = "min"
    value["sampling"]["epoch_size"] = 256
    value["sampling"]["sample_type_fractions_by_tier"] = copy.deepcopy(
        FIXED_SMOKE_TYPE_FRACTIONS
    )
    value["augmentation"]["enabled"] = False
    value["validation"]["enabled"] = False
    value["outputs"]["run_dir"] = str(root / "smoke_run")
    value["smoke_gate"] = dict(FIXED_GATE)
    return value


def _snapshot_rows():
    specifications = (
        ("gold", True, "POS_RUNTIME", 48),
        ("gold", True, "POS_LOW_PALM", 48),
        ("pseudo", True, "POS_RUNTIME", 48),
        ("pseudo", True, "POS_LOW_PALM", 48),
        ("pseudo", False, "NEG_RUNTIME_CANDIDATE", 32),
        ("pseudo", False, "NEG_LOW_PALM_CANDIDATE", 32),
    )
    full = []
    for tier, present, sample_type, count in specifications:
        for index in range(count):
            identity = "{}-{}-{:03d}".format(tier, sample_type, index)
            if not present:
                handedness = "unknown"
            elif index == 0:
                handedness = "Left"
            elif index == 1:
                handedness = "Right"
            elif index == 2:
                handedness = "unknown"
            else:
                handedness = "Left" if index % 2 == 0 else "Right"
            full.append(
                {
                    "global_crop_id": identity,
                    "dataset_id": "dataset-{}".format(tier),
                    "supervision_tier": tier,
                    "sample_type": sample_type,
                    "hand_presence": {"present": present},
                    "handedness": {"label": handedness},
                    "sampling_weight": 0.5,
                }
            )
    smoke = []
    selection = []
    for row in full:
        value = copy.deepcopy(row)
        value["sampling_weight"] = 1.0
        value["finetune_curation"] = {
            "smoke_original_sampling_weight": 0.5,
            "smoke_sampling_weight": 1.0,
        }
        smoke.append(value)
        if row["supervision_tier"] == "gold":
            category = "gold_positive"
        elif row["hand_presence"]["present"]:
            category = "pseudo_positive"
        elif row["sample_type"] == "NEG_RUNTIME_CANDIDATE":
            category = "pseudo_neg_runtime"
        else:
            category = "pseudo_neg_low"
        selection.append(
            {
                "schema_version": "finetune_smoke_selection_v1",
                "global_crop_id": row["global_crop_id"],
                "category": category,
                "dataset_id": row["dataset_id"],
                "source_sequence_id": None,
                "selection_hash": hashlib.sha256(row["global_crop_id"].encode()).hexdigest(),
            }
        )
    return full, smoke, selection


class FinetuneSmokeConfigDiffTests(unittest.TestCase):
    def test_only_documented_smoke_overrides_are_allowed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report = compare_smoke_and_full_configs(
                _smoke_config(root), _full_config(root)
            )
            self.assertEqual("ok", report["status"])
            self.assertIn(
                "training.checkpoint.monitor",
                {item["path"] for item in report["differences"]},
            )

    def test_patience_or_optimizer_changes_are_fatal(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            smoke = _smoke_config(root)
            smoke["training"]["early_stopping"]["patience"] = 60
            with self.assertRaisesRegex(ValueError, "non-permitted"):
                compare_smoke_and_full_configs(smoke, _full_config(root))

            smoke = _smoke_config(root)
            smoke["sampling"]["sample_type_fractions_by_tier"]["gold"]["POS_RUNTIME"] = 0.34
            with self.assertRaisesRegex(ValueError, "fixed balanced"):
                compare_smoke_and_full_configs(smoke, _full_config(root))
            smoke = _smoke_config(root)
            smoke["training"]["optimizer"]["learning_rate"] = 2e-5
            with self.assertRaisesRegex(ValueError, "learning rate"):
                compare_smoke_and_full_configs(smoke, _full_config(root))


class FinetuneSmokeSnapshotTests(unittest.TestCase):
    def test_persisted_snapshot_contract_and_masks(self):
        full, smoke, selection = _snapshot_rows()
        report = validate_smoke_snapshot(full, smoke, selection)
        self.assertEqual(256, report["record_count"])
        self.assertEqual("not_applicable_redistributed_to_gold_positive", report["gold_negative_runtime_check"])
        self.assertEqual("required", report["unknown_handedness_runtime_check"])
        self.assertEqual(32, report["categories"]["pseudo_neg_runtime"])

    def test_required_cell_and_mask_errors_fail_closed(self):
        full, smoke, selection = _snapshot_rows()
        full[0]["handedness_loss_weight"] = 0.0
        smoke[0]["handedness_loss_weight"] = 0.0
        with self.assertRaisesRegex(ValueError, "Known handedness"):
            validate_smoke_snapshot(full, smoke, selection)

        full, smoke, selection = _snapshot_rows()
        full[-1]["presence_quality_weight"] = float("nan")
        smoke[-1]["presence_quality_weight"] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            validate_smoke_snapshot(full, smoke, selection)

    def test_epoch_plan_requires_exact_positive_quota_coverage(self):
        _, smoke, _ = _snapshot_rows()
        quotas = {
            "gold": {
                "POS_RUNTIME": 70,
                "POS_LOW_PALM": 18,
                "NEG_RUNTIME_CANDIDATE": 0,
                "NEG_LOW_PALM_CANDIDATE": 0,
            },
            "pseudo": {
                "POS_RUNTIME": 120,
                "POS_LOW_PALM": 30,
                "NEG_RUNTIME_CANDIDATE": 10,
                "NEG_LOW_PALM_CANDIDATE": 8,
            },
        }
        cells = {
            "{}:{}".format(tier, sample_type): count
            for tier, counts in quotas.items()
            for sample_type, count in counts.items()
            if count
        }
        report = {
            "data_report": {
                "sampler": {
                    "mode": "weighted_stratified",
                    "draws_per_epoch": 256,
                    "epoch_type_plan": {
                        "epoch_draw_quota_by_tier_type": quotas,
                        "batch_cell_quotas": [cells],
                        "batch_type_schedule_sha256": "a" * 64,
                    },
                }
            }
        }
        checked = validate_epoch_plan_coverage(report, smoke)
        self.assertEqual("pass", checked["status"])
        report["data_report"]["sampler"]["epoch_type_plan"]["batch_cell_quotas"][0][
            "pseudo:NEG_RUNTIME_CANDIDATE"
        ] = 0
        with self.assertRaisesRegex(ValueError, "not fully drawn"):
            validate_epoch_plan_coverage(report, smoke)


class FinetuneSmokeProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.full_rows, self.smoke_rows, self.selection_rows = _snapshot_rows()
        self.full_config = _full_config(self.root)
        self.smoke_config = _smoke_config(self.root)
        self.snapshot = self.root / "snapshot"
        (self.snapshot / "05_labels").mkdir(parents=True)
        (self.snapshot / "audit").mkdir(parents=True)
        (self.snapshot / "qc").mkdir(parents=True)
        self.full_labels = self.snapshot / "05_labels" / "full.jsonl"
        self.smoke_labels = self.snapshot / "05_labels" / "smoke.jsonl"
        self.selection = self.snapshot / "audit" / "selection.jsonl"
        write_jsonl(self.full_labels, self.full_rows)
        write_jsonl(self.smoke_labels, self.smoke_rows)
        write_jsonl(self.selection, self.selection_rows)
        self.curator_config = self.root / "curate.yaml"
        self.curator_config.write_text("schema_version: 1\n", encoding="utf-8")
        self.manifest = self.snapshot / "qc" / "sha256_manifest.json"
        artifacts = {}
        for path, count in (
            (self.full_labels, 256),
            (self.smoke_labels, 256),
            (self.selection, 256),
        ):
            artifacts[path.relative_to(self.snapshot).as_posix()] = {
                "sha256": sha256_file(path),
                "count": count,
            }
        write_json(
            self.manifest,
            {
                "schema_version": "finetune_curation_v1",
                "output_dir": str(self.snapshot),
                "config_path": str(self.curator_config),
                "config_sha256": sha256_file(self.curator_config),
                "artifacts": artifacts,
                "smoke": {
                    "labels": self.smoke_labels.relative_to(self.snapshot).as_posix(),
                    "selection": self.selection.relative_to(self.snapshot).as_posix(),
                    "selection_config_sha256": hashlib.sha256(b"{}").hexdigest(),
                    "count": 256,
                },
            },
        )
        self.initial = self.root / "initial.weights.h5"
        self.initial.write_bytes(b"initial")

    def tearDown(self):
        self.temporary.cleanup()

    def test_manifest_binds_full_smoke_and_selection_hashes(self):
        binding = verify_curation_binding(self.smoke_config, self.full_config)
        self.assertEqual(sha256_file(self.selection), binding["selection"]["sha256"])
        self.selection.write_text("{}\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            verify_curation_binding(self.smoke_config, self.full_config)

    def test_training_report_provenance_is_authenticated_without_tensorflow(self):
        binding = verify_curation_binding(self.smoke_config, self.full_config)
        run_dir = Path(self.smoke_config["outputs"]["run_dir"])
        checkpoints = run_dir / "checkpoints"
        checkpoints.mkdir(parents=True)
        history_path = run_dir / "history.json"
        best_path = checkpoints / "best.weights.h5"
        write_json(
            history_path,
            {"epochs": [1, 2], "initial_epoch": 0, "history": {"total_loss": [1.0, 0.5]}},
        )
        best_path.write_bytes(b"best")
        smoke_yaml = self.root / "train_finetune_smoke.yaml"
        smoke_yaml.write_text("schema_version: 1\n", encoding="utf-8")
        self.smoke_config["_meta"] = {
            "config_path": str(smoke_yaml),
            "config_dir": str(self.root),
            "repo_root": str(self.root),
        }
        git = {"commit": "abc", "dirty": False, "status_short": []}
        selection = {
            "monitor": "total_loss",
            "mode": "min",
            "completed_epoch": 2,
            "value": 0.5,
            "verified_against_current_history": True,
            "history_best_epoch": 2,
            "history_best_value": 0.5,
        }
        artifact_records = {
            "history": {"path": str(history_path), "sha256": sha256_file(history_path)},
            "best_checkpoint": {"path": str(best_path), "sha256": sha256_file(best_path)},
        }
        data_report = {
            "curation_manifest": {
                "path": binding["manifest"],
                "sha256": binding["manifest_sha256"],
            }
        }
        metadata_path = run_dir / "experiment_metadata.json"
        interface = dict(_EXPECTED_INTERFACE)
        interface["parameter_count"] = 123
        label_hashes = {
            "train": [
                {
                    "path": str(self.smoke_labels),
                    "exists": True,
                    "sha256": sha256_file(self.smoke_labels),
                }
            ]
        }
        write_json(
            metadata_path,
            {
                "status": "complete",
                "stage": "finetune",
                "model_version": "v2",
                "resolved_config": _jsonable(self.smoke_config),
                "config_path": str(smoke_yaml),
                "config_sha256": sha256_file(smoke_yaml),
                "git": git,
                "starting_state": {
                    "mode": "initial_weights",
                    "path": str(self.initial),
                    "sha256": sha256_file(self.initial),
                    "initial_epoch": 0,
                },
                "data_report": data_report,
                "label_hashes": label_hashes,
                "model_interface": interface,
                "checkpoint_selection": selection,
                "artifacts": artifact_records,
                "completed_epochs": [1, 2],
            },
        )
        write_json(
            run_dir / "training_report.json",
            {
                "status": "complete",
                "stage": "finetune",
                "model_version": "v2",
                "metadata_path": str(metadata_path),
                "completed_epochs": [1, 2],
                "label_hashes": label_hashes,
                "data_report": data_report,
                "checkpoint_selection": selection,
                "artifacts": artifact_records,
            },
        )
        with mock.patch("hand_landmarker.training._git_metadata", return_value=git):
            report = verify_smoke_run_provenance(self.smoke_config, binding)
        self.assertEqual(sha256_file(best_path), report["best_checkpoint"]["sha256"])

        history_path.write_text("{}\n", encoding="utf-8")
        with mock.patch("hand_landmarker.training._git_metadata", return_value=git):
            with self.assertRaisesRegex(ValueError, "path/SHA mismatch"):
                verify_smoke_run_provenance(self.smoke_config, binding)


if __name__ == "__main__":
    unittest.main()
