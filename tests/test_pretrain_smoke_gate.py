import tempfile
import unittest
from pathlib import Path
from unittest import mock

from hand_landmarker.io_utils import sha256_file, write_json
from scripts.check_pretrain_smoke import (
    _jsonable,
    _semantic_batch_value,
    _verify_run_provenance,
    check_smoke_run,
)


class PretrainSmokeGateTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.run_dir = self.root / "run"
        self.run_dir.mkdir()
        self.labels = self.root / "smoke.jsonl"
        self.labels.write_text("{}\n", encoding="utf-8")
        self.history = self.run_dir / "history.json"
        self.best = self.run_dir / "best.weights.h5"
        self.history.write_text('{"epochs":[1]}\n', encoding="utf-8")
        self.best.write_bytes(b"weights")
        self.metadata = self.run_dir / "experiment_metadata.json"
        self.config = {
            "data": {"labels": str(self.labels)},
            "outputs": {"run_dir": str(self.run_dir)},
            "smoke_gate": {
                "expected_records": 128,
                "maximum_mean_mae": 0.01,
                "maximum_p90_mae": 0.02,
                "maximum_max_mae": 0.05,
            },
        }
        write_json(
            self.metadata,
            {"status": "complete", "resolved_config": _jsonable(self.config)},
        )
        write_json(
            self.run_dir / "training_report.json",
            {
                "status": "complete",
                "metadata_path": str(self.metadata),
                "checkpoint_selection": {
                    "monitor": "landmark_mae",
                    "mode": "min",
                    "verified_against_current_history": True,
                },
                "label_hashes": {
                    "train": [
                        {
                            "path": str(self.labels),
                            "exists": True,
                            "sha256": sha256_file(self.labels),
                        }
                    ]
                },
                "artifacts": {
                    "history": {
                        "path": str(self.history),
                        "sha256": sha256_file(self.history),
                    },
                    "best_checkpoint": {
                        "path": str(self.best),
                        "sha256": sha256_file(self.best),
                    },
                },
            },
        )

    def tearDown(self):
        self.temporary.cleanup()

    def test_gate_uses_full_subset_metrics_after_provenance_check(self):
        provenance = _verify_run_provenance(self.config, self.run_dir)
        self.assertEqual(sha256_file(self.labels), provenance["labels_sha256"])
        metrics = {
            "mean_landmark_mae": 0.009,
            "p90_sample_landmark_mae": 0.019,
            "max_sample_landmark_mae": 0.049,
        }
        with mock.patch(
            "scripts.check_pretrain_smoke._full_smoke_metrics",
            return_value=metrics,
        ) as full_evaluation:
            report = check_smoke_run(self.config)
        self.assertEqual("pass", report["status"])
        full_evaluation.assert_called_once_with(self.config, self.best, 128)

    def test_tampered_labels_or_stale_config_are_rejected(self):
        self.labels.write_text('{"tampered":true}\n', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "labels path/hash"):
            _verify_run_provenance(self.config, self.run_dir)

        self.labels.write_text("{}\n", encoding="utf-8")
        changed = dict(self.config)
        changed["smoke_gate"] = dict(self.config["smoke_gate"])
        changed["smoke_gate"]["expected_records"] = 64
        with self.assertRaisesRegex(ValueError, "resolved_config differs"):
            _verify_run_provenance(changed, self.run_dir)

    def test_semantic_batch_value_accepts_new_mapping_and_legacy_sequence(self):
        landmarks = object()
        self.assertIs(
            landmarks,
            _semantic_batch_value({"landmarks": landmarks}, "landmarks", 0, "weights"),
        )
        self.assertIs(
            landmarks,
            _semantic_batch_value([landmarks], "landmarks", 0, "weights"),
        )
        with self.assertRaisesRegex(KeyError, "available keys"):
            _semantic_batch_value({"hand_flag": landmarks}, "landmarks", 0, "weights")


if __name__ == "__main__":
    unittest.main()
