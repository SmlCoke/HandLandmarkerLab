from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from hand_landmarker.config import load_config
from hand_landmarker.finetune_analysis import analyze_finetune_runs
from hand_landmarker.finetune_selection import identity_tokens
from hand_landmarker.io_utils import write_jsonl


ROOT = Path(__file__).resolve().parents[1]


class Hlml3FeatureTests(unittest.TestCase):
    def test_v4_stage_profiles_resolve_one_yaml(self) -> None:
        with mock.patch.dict(
            os.environ,
            {
                "HLML_STAGE": "multi_finetune",
                "HLML_EXPERIMENT_ID": "experiment-test",
            },
            clear=False,
        ):
            config = load_config(ROOT / "configs" / "training.yaml")
        self.assertEqual("multi_finetune", config["resolved_profile"])
        self.assertEqual("finetune", config["stage"])
        self.assertEqual(0.55, config["training"]["gold_fraction"])
        self.assertEqual(0.0, config["losses"]["bone_vector"]["coefficient"])
        self.assertIn("/experiment-test/multi_finetune", config["outputs"]["run_dir"])

    def test_identity_union_covers_all_required_namespaces(self) -> None:
        tokens = identity_tokens(
            {
                "dataset_id": "dataset-a",
                "parent_global_crop_id": "global-a",
                "source_image": "session/frame.tiff",
                "image_sha256": "a" * 64,
                "normalized_pixel_sha256": "b" * 64,
                "native_source_root": "/datasets/session-a",
                "native_source_crop_id": "crop-42",
            }
        )
        self.assertIn("parent:global-a", tokens)
        self.assertIn("source_image:dataset-a:session/frame.tiff", tokens)
        self.assertIn("roi_sha256:" + "a" * 64, tokens)
        self.assertIn("pixel_sha256:" + "b" * 64, tokens)
        self.assertIn("native:/datasets/session-a:crop-42", tokens)

    def test_analysis_produces_paired_machine_readable_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            labels = root / "labels.jsonl"
            points = [
                {"id": index, "x": 0.2 + (index % 5) * 0.1, "y": 0.2 + (index // 5) * 0.1}
                for index in range(21)
            ]
            write_jsonl(
                labels,
                [
                    {
                        "global_crop_id": "roi-1",
                        "dataset_id": "dataset-a",
                        "hand_presence": {"present": True},
                        "handedness": {"label": "Left"},
                        "landmarks_crop_norm": points,
                    }
                ],
            )
            prediction = {
                "crop_id": "roi-1",
                "dataset_id": "dataset-a",
                "expected_presence": True,
                "predicted_presence": True,
                "expected_handedness": "Left",
                "predicted_handedness": "Left",
                "landmarks_roi_norm": [[point["x"], point["y"]] for point in points],
                "mean_landmark_error_px": 0.0,
                "nme": 0.0,
                "landmark_errors_px_by_id": [0.0] * 21,
            }
            for run in ("baseline", "candidate"):
                path = root / "hand_landmarker_runs" / run / "eval" / "finetune" / "val" / "predictions.jsonl"
                write_jsonl(path, [prediction])
            output = root / "analysis"
            report = analyze_finetune_runs(
                work_root=root,
                baseline_id="baseline",
                candidate_id="candidate",
                labels_path=labels,
                output_dir=output,
                overlay_limit=0,
            )
            self.assertEqual("ok", report["status"])
            self.assertEqual(1, report["paired"]["matched_records"])
            self.assertTrue((output / "summary.json").is_file())
            self.assertTrue((output / "paired_comparison.json").is_file())
            self.assertTrue((output / "per_roi_metrics.jsonl").is_file())


if __name__ == "__main__":
    unittest.main()
