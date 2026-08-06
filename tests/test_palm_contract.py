import tempfile
import unittest
from pathlib import Path

import numpy as np

from hand_landmarker.palm import PalmDetection, decode_outputs, select_detections
from hand_landmarker.runtime import normalize_runtime_config


class PalmContractTests(unittest.TestCase):
    def test_output_mapping_and_nchw_decode(self):
        reg14 = np.zeros((1, 16, 14, 14), dtype=np.float32)
        cls14 = np.zeros((1, 2, 14, 14), dtype=np.float32)
        reg7 = np.zeros((1, 16, 7, 7), dtype=np.float32)
        cls7 = np.zeros((1, 2, 7, 7), dtype=np.float32)
        cls14[0, 0, 0, 0] = 0.9
        # Deliberately scramble output order; mapping is by element count.
        detections = decode_outputs([cls7, reg14, reg7, cls14])
        self.assertEqual(len(detections), 1)
        detection = detections[0]
        self.assertEqual(detection.head_size, 14)
        self.assertAlmostEqual(detection.score, 0.9, places=6)
        center = 0.5 / 14.0
        self.assertAlmostEqual(detection.bbox_norm[0], 0.0, places=6)
        self.assertAlmostEqual(detection.wrist_norm[0], center, places=6)

    def test_board_does_not_apply_intra_head_nms_to_7x7(self):
        # IoU is 1/3: annotation-side 0.30 NMS would suppress it, while the
        # board's 0.35 selected-candidate suppression keeps it.
        first = PalmDetection(0.9, (0.0, 0.0, 0.5, 0.5), (0.2, 0.5), (0.3, 0.2), 7, 0)
        second = PalmDetection(0.8, (0.25, 0.0, 0.75, 0.5), (0.2, 0.5), (0.3, 0.2), 7, 1)
        selected = select_detections([first, second], max_detections=2)
        self.assertEqual(selected, [first, second])

    def test_board_stops_7x7_scan_when_selected_reaches_limit(self):
        head14_first = PalmDetection(
            0.20, (0.00, 0.00, 0.10, 0.10), (0.02, 0.02), (0.08, 0.08), 14, 0
        )
        head14_second = PalmDetection(
            0.10, (0.20, 0.00, 0.30, 0.10), (0.22, 0.02), (0.28, 0.08), 14, 1
        )
        head7_first = PalmDetection(
            0.90, (0.40, 0.00, 0.50, 0.10), (0.42, 0.02), (0.48, 0.08), 7, 0
        )
        head7_second = PalmDetection(
            0.80, (0.60, 0.00, 0.70, 0.10), (0.62, 0.02), (0.68, 0.08), 7, 1
        )

        # The board starts with both 14x14 NMS results, appends the first
        # eligible 7x7 candidate, then breaks before inspecting head7_second.
        selected = select_detections(
            [head14_first, head14_second, head7_first, head7_second],
            max_detections=2,
        )
        self.assertEqual(selected, [head7_first, head14_first])

    def test_runtime_resolves_selected_eos_below_palm_detector(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = {
                "_meta": {"repo_root": str(root)},
                "palm": {
                    "models_root": "palm_detector",
                    "model_id": "eos-2.0",
                    "model_filename": "model_opt.onnx",
                },
            }
            resolved = normalize_runtime_config(config)
            self.assertEqual(
                str(root / "palm_detector" / "eos-2.0" / "model_opt.onnx"),
                resolved["palm"]["model_path"],
            )

    def test_runtime_rejects_palm_model_directory_traversal(self):
        config = {
            "palm": {
                "models_root": "palm_detector",
                "model_id": "../outside",
                "model_filename": "model_opt.onnx",
            }
        }
        with self.assertRaisesRegex(ValueError, "safe single directory"):
            normalize_runtime_config(config)

    def test_pixel_mapping_uses_dimension_minus_one_and_cpp_rounding(self):
        detection = PalmDetection(1.0, (0.5, 0.5, 1.0, 1.0), (0.5, 0.5), (1.0, 1.0), 14, 0)
        self.assertEqual(detection.bbox_px(1280, 720), (640.0, 360.0, 1279.0, 719.0))
        self.assertEqual(detection.wrist_px(1280, 720), (640.0, 360.0))


if __name__ == "__main__":
    unittest.main()
