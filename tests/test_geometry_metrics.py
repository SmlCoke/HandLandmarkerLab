import unittest

import numpy as np

from hand_landmarker.board_ops import resize_gray_uint8, sample_rotated_roi_gray_uint8
from hand_landmarker.geometry import (
    RoiRect,
    build_roi_rect,
    project_normalized_points,
    rotated_rect_iou,
    unproject_image_points,
)
from hand_landmarker.metrics import EvaluationMetrics, LandmarkMetrics


class BoardGeometryTests(unittest.TestCase):
    def test_identity_resize_and_roi_sample(self):
        image = np.arange(256 * 256, dtype=np.uint32).reshape(256, 256).astype(np.uint8)
        np.testing.assert_array_equal(resize_gray_uint8(image, 256, 256), image)
        corners = ((0.0, 0.0), (255.0, 0.0), (255.0, 255.0), (0.0, 255.0))
        np.testing.assert_array_equal(sample_rotated_roi_gray_uint8(image, corners), image)

    def test_roi_contract_and_projection_roundtrip(self):
        rect = build_roi_rect(
            [500.0, 300.0, 620.0, 420.0],
            [540.0, 410.0],
            [560.0, 330.0],
            1280,
            720,
        )
        self.assertAlmostEqual(rect.width, 216.0)
        self.assertAlmostEqual(rect.height, 216.0)
        points = [(0.0, 0.0), (1.0, 1.0), (0.2, 0.7)]
        projected = project_normalized_points(points, rect)
        restored = unproject_image_points(projected, rect)
        for expected, actual in zip(points, restored):
            self.assertAlmostEqual(expected[0], actual[0], places=6)
            self.assertAlmostEqual(expected[1], actual[1], places=6)
        self.assertAlmostEqual(rotated_rect_iou(rect, rect), 1.0, places=6)


class MetricsTests(unittest.TestCase):
    def test_missing_landmark_prediction_counts_as_zero_pck_coverage(self):
        metric = LandmarkMetrics((0.05,))
        expected = [(float(index), float(index)) for index in range(21)]
        metric.update(expected, None)
        report = metric.report()
        self.assertEqual(report["gt_positive_count"], 1)
        self.assertEqual(report["prediction_count"], 0)
        self.assertEqual(report["prediction_coverage"], 0.0)
        self.assertEqual(report["pck"]["0.05"], 0.0)

    def test_landmarks_are_scored_even_when_presence_is_false(self):
        metric = EvaluationMetrics((0.05,))
        points = [(float(index), 0.0) for index in range(21)]
        metric.update(True, False, points, points, "Left", "Left")
        report = metric.report()
        self.assertEqual(report["presence"]["fn"], 1)
        self.assertEqual(report["landmarks"]["prediction_coverage"], 1.0)
        self.assertEqual(report["landmarks"]["mean_pixel_error"], 0.0)


if __name__ == "__main__":
    unittest.main()

