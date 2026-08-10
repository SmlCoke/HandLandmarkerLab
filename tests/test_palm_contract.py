from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from hand_landmarker.contracts import BOARD_CONTRACT
from hand_landmarker.palm import (
    PalmDetection,
    decode_outputs,
    feature_level_anchor_count,
    generate_anchors,
    normalize_feature_levels,
    palm_model_contract,
    preprocess_for_onnx,
)
from hand_landmarker.runtime import normalize_runtime_config


def eos2_levels():
    return normalize_feature_levels(
        {"feature_levels": BOARD_CONTRACT["palm_feature_levels"]}
    )


class _Meta:
    def __init__(self, name, shape, tensor_type="tensor(float)"):
        self.name = name
        self.shape = shape
        self.type = tensor_type


class _Session:
    def __init__(self, input_shape=None):
        self._inputs = [_Meta("inputs", input_shape or [1, 1, 224, 384])]
        self._outputs = [
            _Meta("reg14", [1, 16, 14, 24]),
            _Meta("cls14", [1, 2, 14, 24]),
            _Meta("reg7", [1, 16, 7, 12]),
            _Meta("cls7", [1, 2, 7, 12]),
        ]

    def get_inputs(self):
        return self._inputs

    def get_outputs(self):
        return self._outputs


class PalmContractTests(unittest.TestCase):
    def test_rectangular_preprocess_matches_hlmf_area_resize(self):
        image = np.arange(720 * 1280, dtype=np.uint32).reshape(720, 1280) % 256
        image = image.astype(np.uint8)
        actual = preprocess_for_onnx(image, 384, 224)
        expected = (
            cv2.resize(image, (384, 224), interpolation=cv2.INTER_AREA).astype(np.float32)
            / 255.0
        )
        self.assertEqual((1, 1, 224, 384), actual.shape)
        self.assertEqual(np.float32, actual.dtype)
        np.testing.assert_array_equal(expected, actual[0, 0])

    def test_rectangular_anchors_have_expected_count_and_centers(self):
        levels = eos2_levels()
        self.assertEqual(840, feature_level_anchor_count(levels))
        anchors = generate_anchors(14, 24, levels[0]["anchor_sizes"])
        self.assertEqual((672, 4), anchors.shape)
        np.testing.assert_allclose(
            [1.0 / 48.0, 1.0 / 28.0], anchors[0, :2], rtol=0.0, atol=1e-7
        )
        np.testing.assert_allclose(
            [23.5 / 24.0, 13.5 / 14.0], anchors[-1, :2], rtol=0.0, atol=1e-7
        )

    def test_shuffled_outputs_decode_and_apply_global_nms(self):
        levels = eos2_levels()
        reg14 = np.zeros((1, 16, 14, 24), dtype=np.float32)
        cls14 = np.zeros((1, 2, 14, 24), dtype=np.float32)
        reg7 = np.zeros((1, 16, 7, 12), dtype=np.float32)
        cls7 = np.zeros((1, 2, 7, 12), dtype=np.float32)
        cls14[0, 0, 0, 0] = 0.90
        cls7[0, 0, 0, 0] = 0.80
        detections = decode_outputs(
            [cls7, reg14, cls14, reg7],
            levels,
            score_threshold=0.25,
            nms_iou_threshold=0.10,
            max_detections=2,
            output_layout="nchw",
        )
        self.assertEqual(1, len(detections))
        self.assertEqual("14x24", detections[0].feature_level)
        self.assertAlmostEqual(0.90, detections[0].score, places=6)

    def test_score_threshold_equality_passes(self):
        levels = eos2_levels()
        outputs = [
            np.zeros((1, 16, 14, 24), dtype=np.float32),
            np.zeros((1, 2, 14, 24), dtype=np.float32),
            np.zeros((1, 16, 7, 12), dtype=np.float32),
            np.zeros((1, 2, 7, 12), dtype=np.float32),
        ]
        outputs[1][0, 0, 2, 3] = 0.25
        detections = decode_outputs(outputs, levels)
        self.assertEqual(1, len(detections))

    def test_model_contract_reports_eos2_geometry(self):
        contract = palm_model_contract(_Session(), 384, 224, eos2_levels())
        self.assertEqual([1, 1, 224, 384], contract["input_shape"])
        self.assertEqual(840, contract["anchor_count"])
        with self.assertRaisesRegex(ValueError, "does not match configured"):
            palm_model_contract(_Session([1, 1, 224, 224]), 384, 224, eos2_levels())

    def test_invalid_feature_level_config_is_rejected(self):
        levels = [dict(level) for level in BOARD_CONTRACT["palm_feature_levels"]]
        levels[1] = dict(levels[1], height=14, width=24)
        with self.assertRaisesRegex(ValueError, "shapes must be unique"):
            normalize_feature_levels({"feature_levels": levels})

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
            self.assertEqual(384, resolved["palm"]["input_width"])
            self.assertEqual(224, resolved["palm"]["input_height"])

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

    def test_runtime_rejects_eos1_decoder_keys(self):
        with self.assertRaisesRegex(ValueError, "input_size is obsolete"):
            normalize_runtime_config({"palm": {"input_size": 224}})
        with self.assertRaisesRegex(ValueError, "global NMS"):
            normalize_runtime_config({"palm": {"cross_head_suppress_iou": 0.35}})

    def test_pixel_mapping_uses_dimension_minus_one_and_cpp_rounding(self):
        detection = PalmDetection(
            1.0,
            (0.5, 0.5, 1.0, 1.0),
            (0.5, 0.5),
            (1.0, 1.0),
            "14x24",
            0,
        )
        self.assertEqual(detection.bbox_px(1280, 720), (640.0, 360.0, 1279.0, 719.0))
        self.assertEqual(detection.wrist_px(1280, 720), (640.0, 360.0))


if __name__ == "__main__":
    unittest.main()
