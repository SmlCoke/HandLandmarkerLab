import unittest
import tempfile
from pathlib import Path

from hand_landmarker.config import load_config
from hand_landmarker.io_utils import write_json
from hand_landmarker.training import (
    _monitor_mode,
    _multitask_monitor_value,
    _state_json_path,
    _verify_best_checkpoint_selection,
)


ROOT = Path(__file__).resolve().parents[1]


class MonitorDirectionTests(unittest.TestCase):
    def test_multitask_score_is_geometry_first_and_deterministic(self):
        value = _multitask_monitor_value(
            {
                "val_landmark_mae": 0.03,
                "val_hand_flag_accuracy": 0.80,
                "val_handedness_accuracy": 0.60,
            },
            {
                "landmark_mae_weight": 1.0,
                "hand_flag_error_weight": 0.02,
                "handedness_error_weight": 0.005,
            },
        )
        self.assertAlmostEqual(0.036, value)
        self.assertIsNone(_multitask_monitor_value({}, {}))

    def test_error_metrics_are_minimized(self):
        for name in (
            "val_total_loss",
            "val_landmark_mae",
            "mean_pixel_error",
            "mean_nme",
            "rmse",
        ):
            with self.subTest(name=name):
                self.assertEqual("min", _monitor_mode(name))

    def test_quality_metrics_are_maximized(self):
        for name in ("val_accuracy", "pck_0_10", "f1", "recall", "auc"):
            with self.subTest(name=name):
                self.assertEqual("max", _monitor_mode(name))

    def test_explicit_mode_wins_and_invalid_values_fail(self):
        self.assertEqual("min", _monitor_mode("custom_score", "min"))
        self.assertEqual("max", _monitor_mode("custom_loss", "max"))
        with self.assertRaisesRegex(ValueError, "mode must be min or max"):
            _monitor_mode("val_landmark_mae", "auto")
        with self.assertRaisesRegex(ValueError, "Could not infer"):
            _monitor_mode("custom_score")

    def test_pretrain_config_is_explicit_and_lr_can_run_before_early_stop(self):
        config = load_config(ROOT / "configs" / "training.yaml")
        training = config["training"]
        self.assertEqual(
            {"monitor": "val_landmark_mae", "mode": "min"},
            training["checkpoint"],
        )
        self.assertEqual("min", training["learning_rate_schedule"]["mode"])
        self.assertEqual("min", training["early_stopping"]["mode"])
        self.assertLess(
            training["learning_rate_schedule"]["patience"],
            training["early_stopping"]["patience"],
        )

    def test_persisted_best_must_equal_minimum_history_mae(self):
        with tempfile.TemporaryDirectory() as directory:
            best = Path(directory) / "best.weights.h5"
            write_json(
                _state_json_path(best),
                {
                    "monitor": "val_landmark_mae",
                    "mode": "min",
                    "completed_epoch": 2,
                    "value": 0.2,
                },
            )
            training = {
                "checkpoint": {"monitor": "val_landmark_mae", "mode": "min"},
                "learning_rate_schedule": {"monitor": "val_landmark_mae", "mode": "min"},
                "early_stopping": {"monitor": "val_landmark_mae", "mode": "min"},
            }
            summary = _verify_best_checkpoint_selection(
                best,
                {
                    "epochs": [1, 2, 3],
                    "history": {"val_landmark_mae": [0.3, 0.2, 0.25]},
                },
                training,
                resumed=False,
            )
            self.assertEqual(2, summary["history_best_epoch"])
            state_path = _state_json_path(best)
            write_json(
                state_path,
                {
                    "monitor": "val_landmark_mae",
                    "mode": "max",
                    "completed_epoch": 1,
                    "value": 0.3,
                },
            )
            with self.assertRaisesRegex(RuntimeError, "conflicts with config"):
                _verify_best_checkpoint_selection(
                    best,
                    {
                        "epochs": [1, 2, 3],
                        "history": {"val_landmark_mae": [0.3, 0.2, 0.25]},
                    },
                    training,
                    resumed=False,
                )

    def test_best_verification_supports_sparse_validation_history(self):
        with tempfile.TemporaryDirectory() as directory:
            best = Path(directory) / "best.weights.h5"
            write_json(
                _state_json_path(best),
                {
                    "monitor": "val_landmark_mae",
                    "mode": "min",
                    "completed_epoch": 4,
                    "value": 0.1,
                },
            )
            summary = _verify_best_checkpoint_selection(
                best,
                {
                    "epochs": [1, 2, 3, 4, 5],
                    "history": {"val_landmark_mae": [0.3, 0.1]},
                },
                {
                    "checkpoint": {"monitor": "val_landmark_mae", "mode": "min"},
                    "learning_rate_schedule": {
                        "monitor": "val_landmark_mae",
                        "mode": "min",
                    },
                    "early_stopping": {
                        "monitor": "val_landmark_mae",
                        "mode": "min",
                    },
                },
                resumed=False,
                validation={"every_epochs": 2},
            )
            self.assertEqual(4, summary["history_best_epoch"])


if __name__ == "__main__":
    unittest.main()
