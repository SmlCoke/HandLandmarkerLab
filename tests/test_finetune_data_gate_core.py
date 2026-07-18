import tempfile
import unittest
from pathlib import Path

from hand_landmarker.io_utils import sha256_file, write_json
from scripts.check_finetune_data import (
    _aggregate_repository_root,
    _checkpoint_gate,
    _sampling_gate,
)


class FinetuneDataGateCoreTest(unittest.TestCase):
    def test_aggregate_repository_root_comes_from_authenticated_descriptor(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gold = root / "GoldSource"
            gold.mkdir()
            aggregate = root / "workspace" / "hmlf_gold_aggregate.json"
            write_json(aggregate, {"gold_repository_root": str(gold.resolve())})
            self.assertEqual(_aggregate_repository_root(aggregate), gold.resolve())

            write_json(aggregate, {"gold_repository_root": "GoldSource"})
            with self.assertRaisesRegex(ValueError, "missing or invalid"):
                _aggregate_repository_root(aggregate)

    def test_checkpoint_gate_authenticates_complete_multitask_best(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "pretrain" / "multitask"
            checkpoint = run / "checkpoints" / "best.weights.h5"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"weights")
            artifact = {
                "path": str(checkpoint.resolve()),
                "sha256": sha256_file(checkpoint),
                "size_bytes": checkpoint.stat().st_size,
            }
            metadata = {
                "status": "complete",
                "stage": "pretrain",
                "model_version": "v2",
                "experiment": {"name": "hand_landmarker_v2_multitask"},
                "resolved_config": {
                    "experiment": {"name": "hand_landmarker_v2_multitask"}
                },
                "artifacts": {"best_checkpoint": artifact},
            }
            metadata_path = run / "experiment_metadata.json"
            write_json(metadata_path, metadata)
            report = {
                "status": "complete",
                "stage": "pretrain",
                "model_version": "v2",
                "experiment": "hand_landmarker_v2_multitask",
                "metadata_path": str(metadata_path.resolve()),
                "artifacts": {"best_checkpoint": artifact},
            }
            write_json(run / "training_report.json", report)
            config = {
                "outputs": {"run_dir": str(root / "finetune" / "run")},
                "_meta": {"repo_root": str(root)},
            }
            checked = _checkpoint_gate(config, checkpoint, resume=False)
            self.assertEqual(checked["stage"], "pretrain")
            report["experiment"] = "hand_landmarker_v2_geometry"
            metadata["experiment"]["name"] = "hand_landmarker_v2_geometry"
            metadata["resolved_config"]["experiment"]["name"] = "hand_landmarker_v2_geometry"
            write_json(run / "training_report.json", report)
            write_json(metadata_path, metadata)
            with self.assertRaisesRegex(ValueError, "multitask best"):
                _checkpoint_gate(config, checkpoint, resume=False)
            report["experiment"] = "hand_landmarker_v2_multitask"
            metadata["experiment"]["name"] = "hand_landmarker_v2_multitask"
            metadata["resolved_config"]["experiment"]["name"] = "hand_landmarker_v2_multitask"
            write_json(run / "training_report.json", report)
            metadata["status"] = "running"
            write_json(metadata_path, metadata)
            with self.assertRaisesRegex(ValueError, "not complete"):
                _checkpoint_gate(config, checkpoint, resume=False)

    def test_sampling_gate_persists_exact_epoch_plan(self):
        sample_types = (
            "POS_RUNTIME",
            "POS_LOW_PALM",
            "NEG_RUNTIME_CANDIDATE",
            "NEG_LOW_PALM_CANDIDATE",
        )
        rows = []
        for tier in ("gold", "pseudo"):
            for sample_type in sample_types:
                for index in range(4):
                    rows.append(
                        {
                            "crop_id": "{}:{}:{}".format(tier, sample_type, index),
                            "global_crop_id": "{}:{}:{}".format(tier, sample_type, index),
                            "supervision_tier": tier,
                            "sample_type": sample_type,
                            "sampling_bucket": tier + ":" + sample_type,
                            "sampling_weight": 1.0,
                        }
                    )
        fractions = {name: 0.25 for name in sample_types}
        config = {
            "stage": "finetune",
            "experiment": {"seed": 7},
            "training": {"batch_size": 8, "gold_fraction": 0.5},
            "sampling": {
                "epoch_size": 16,
                "sample_type_fractions_by_tier": {"gold": fractions, "pseudo": fractions},
                "missing_cell_policy": {"gold": "redistribute_within_tier", "pseudo": "fail"},
                "rare_cell_policy": {"gold": "fail", "pseudo": "fail"},
                "replacement": True,
                "honor_record_sampling_weight": True,
                "quota_tie_break": list(sample_types),
            },
        }
        report = _sampling_gate(config, rows)
        self.assertEqual(sum(sum(cell.values()) for cell in report["epoch0_plan"]["epoch_draw_quota_by_tier_type"].values()), 16)
        self.assertEqual(len(report["epoch0_plan"]["batch_type_schedule_sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
