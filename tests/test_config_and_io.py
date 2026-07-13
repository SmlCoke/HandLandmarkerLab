import os
import tempfile
import unittest
from pathlib import Path

from hand_landmarker.config import load_config
from hand_landmarker.contracts import effective_head_weights
from hand_landmarker.io_utils import resolve_record_image
from hand_landmarker.runtime import decode_hand_outputs, normalize_runtime_config
from hand_landmarker.training import _guard_training_outputs


class ConfigTests(unittest.TestCase):
    def test_inheritance_and_environment_default(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "base.yaml").write_text("a:\n  b: 1\n  c: old\n", encoding="utf-8")
            (root / "child.yaml").write_text(
                "extends: base.yaml\na:\n  c: new\npath: '${UNSET_HAND_TEST:-fallback}'\n",
                encoding="utf-8",
            )
            config = load_config(root / "child.yaml")
            self.assertEqual(config["a"], {"b": 1, "c": "new"})
            self.assertEqual(config["path"], "fallback")


class IoContractTests(unittest.TestCase):
    def test_source_crop_path_is_not_a_silent_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            legacy = Path(directory) / "legacy.png"
            legacy.write_bytes(b"not-an-image")
            row = {"crop_id": "sample", "crop_path": "/missing/legacy.png", "source_crop_path": str(legacy)}
            with self.assertRaises(FileNotFoundError):
                resolve_record_image(row, [])
            self.assertEqual(resolve_record_image(row, [directory]), legacy.resolve())

    def test_effective_weights_do_not_include_sampling_weight(self):
        row = {
            "hand_presence": {"present": True},
            "handedness": {"label": "Right"},
            "hand_presence_loss_weight": 0.5,
            "landmark_loss_weight": 0.8,
            "handedness_loss_weight": 0.4,
            "supervision_loss_weight": 0.25,
            "presence_quality_weight": 0.2,
            "landmark_quality_weight": 0.5,
            "handedness_quality_weight": 1.0,
            "sampling_weight": 999.0,
        }
        self.assertEqual(effective_head_weights(row), (0.025, 0.1, 0.1))


class RuntimeConfigContractTests(unittest.TestCase):
    def test_fixed_hand_interface_cannot_drift_in_yaml(self):
        with self.assertRaisesRegex(ValueError, "input_shape"):
            normalize_runtime_config({"model": {"input_shape": [3, 224, 224]}})

    def test_fixed_roi_output_size_cannot_drift_in_yaml(self):
        with self.assertRaisesRegex(ValueError, "output_size"):
            normalize_runtime_config({"hand_roi": {"output_size": 224}})

    def test_board_palm_and_roi_parameters_cannot_drift(self):
        with self.assertRaisesRegex(ValueError, "score_threshold"):
            normalize_runtime_config({"palm": {"score_threshold": 0.6}})
        with self.assertRaisesRegex(ValueError, "scale_x"):
            normalize_runtime_config({"hand_roi": {"scale_x": 2.0}})

    def test_duplicate_hand_model_paths_cannot_conflict(self):
        with self.assertRaisesRegex(ValueError, "different files"):
            normalize_runtime_config(
                {
                    "model": {"checkpoint": "first.weights.h5"},
                    "hand": {"model_path": "second.weights.h5"},
                }
            )

    def test_hand_decode_reports_board_scaling_and_out_of_range_values(self):
        import numpy as np

        landmarks = np.zeros((1, 1, 1, 42), dtype=np.float32)
        landmarks.reshape(-1)[0] = 256.0
        prediction = decode_hand_outputs(
            [
                landmarks,
                np.asarray([[[[0.8]]]], dtype=np.float32),
                np.asarray([[[[0.7]]]], dtype=np.float32),
            ],
            1,
        )[0]
        self.assertEqual(256.0, prediction.board_landmark_scale_divisor)
        self.assertEqual(256.0, prediction.landmark_raw_max_abs)
        self.assertEqual((1.0, 0.0), prediction.landmarks_norm[0])
        self.assertEqual(0, prediction.normalized_out_of_range_coordinate_count)

    def test_hand_decode_rejects_non_finite_output(self):
        import numpy as np

        with self.assertRaisesRegex(ValueError, "NaN or Inf"):
            decode_hand_outputs(
                [
                    np.full((1, 1, 1, 42), np.nan, dtype=np.float32),
                    np.zeros((1, 1, 1, 1), dtype=np.float32),
                    np.zeros((1, 1, 1, 1), dtype=np.float32),
                ],
                1,
            )


class TrainingOutputContractTests(unittest.TestCase):
    def test_fresh_training_does_not_silently_overwrite_run(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            run_dir.mkdir()
            (run_dir / "experiment_metadata.json").write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "Training output already exists"):
                _guard_training_outputs(
                    run_dir,
                    [run_dir / "checkpoints" / "best.weights.h5"],
                    resume_requested=False,
                    overwrite=False,
                )
            _guard_training_outputs(
                run_dir,
                [run_dir / "checkpoints" / "best.weights.h5"],
                resume_requested=True,
                overwrite=False,
            )


if __name__ == "__main__":
    unittest.main()
