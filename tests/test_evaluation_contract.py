import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from hand_landmarker.evaluation import evaluate_from_config, evaluate_hand_rois


def _positive_row():
    return {
        "crop_id": "roi-1",
        "dataset_id": "unit",
        "source_image": "source.png",
        "crop_path": "roi.png",
        "_resolved_crop_path": "roi.png",
        "hand_presence": {"present": True},
        "handedness": {"label": "Right"},
        "landmarks_crop_norm": [
            {"id": index, "x": 0.25, "y": 0.75} for index in range(21)
        ],
    }


class _GrayRoi:
    ndim = 2
    shape = (256, 256)
    dtype = "uint8"


class _WrongSizeRoi:
    ndim = 2
    shape = (224, 224)
    dtype = "uint8"


class EvaluationBoundaryTests(unittest.TestCase):
    def _predictor(self):
        hand = SimpleNamespace(
            landmarks_norm=[(0.25, 0.75)] * 21,
            hand_flag_score=0.9,
            handedness="Right",
            handedness_score=0.8,
        )
        predictor = mock.Mock()
        predictor.predict.return_value = [hand]
        return predictor

    @mock.patch("hand_landmarker.evaluation.read_image", return_value=_GrayRoi())
    @mock.patch("hand_landmarker.evaluation.create_hand_predictor")
    def test_roi_evaluation_runs_hand_only_and_test_does_not_sweep(
        self, create_hand, _read
    ):
        with mock.patch.object(Path, "is_file", return_value=True):
            create_hand.return_value = self._predictor()
            report = evaluate_hand_rois(
                {
                    "data": {},
                    "evaluation": {"tune_thresholds": False, "hand_flag_threshold": 0.5},
                    "inference": {"batch_size": 8},
                },
                [_positive_row()],
            )
        self.assertEqual(report["scope"], "hand_landmarker_on_provided_hand_roi")
        self.assertNotIn("presence_threshold_sweep", report)
        self.assertEqual(report["details"][0]["landmarks_roi_norm"], [[0.25, 0.75]] * 21)
        create_hand.assert_called_once()

    @mock.patch("hand_landmarker.evaluation.read_image", return_value=_WrongSizeRoi())
    @mock.patch("hand_landmarker.evaluation.create_hand_predictor")
    def test_roi_evaluation_rejects_wrong_image_shape(self, create_hand, _read):
        with mock.patch.object(Path, "is_file", return_value=True):
            create_hand.return_value = self._predictor()
            with self.assertRaisesRegex(ValueError, "256x256 one-channel"):
                evaluate_hand_rois({"data": {}}, [_positive_row()])

    @mock.patch("hand_landmarker.evaluation.create_hand_predictor")
    def test_roi_evaluation_requires_audited_path(self, create_hand):
        row = _positive_row()
        row.pop("_resolved_crop_path")
        create_hand.return_value = self._predictor()
        with self.assertRaisesRegex(ValueError, "audited _resolved_crop_path"):
            evaluate_hand_rois({"data": {}}, [row])

    def test_evaluation_rejects_palm_cascade_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            labels = Path(directory) / "labels.jsonl"
            labels.write_text("{}\n", encoding="utf-8")
            with mock.patch("hand_landmarker.evaluation._validate_rows"):
                with self.assertRaisesRegex(ValueError, "provided Hand ROIs only"):
                    evaluate_from_config(
                        {
                            "split": "val",
                            "data": {"labels": str(labels)},
                            "evaluation": {"mode": "cascade_replay"},
                        }
                    )

    def test_test_split_cannot_enable_threshold_tuning(self):
        with self.assertRaisesRegex(ValueError, "must not tune"):
            evaluate_from_config(
                {
                    "split": "test",
                    "evaluation": {"mode": "roi", "tune_thresholds": True},
                }
            )

    def test_evaluation_does_not_silently_overwrite_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "metrics.json").write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "output already exists"):
                evaluate_from_config(
                    {
                        "split": "val",
                        "evaluation": {"mode": "roi", "tune_thresholds": False},
                        "output": {"dir": str(root), "overwrite": False},
                    }
                )


if __name__ == "__main__":
    unittest.main()
