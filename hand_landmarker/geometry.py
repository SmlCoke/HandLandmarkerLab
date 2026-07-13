"""Palm-to-hand ROI geometry mirrored from the A1 board implementation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


@dataclass(frozen=True)
class RoiRect:
    x_center: float
    y_center: float
    width: float
    height: float
    rotation_rad: float
    corners: Tuple[Tuple[float, float], ...]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "x_center": self.x_center,
            "y_center": self.y_center,
            "width": self.width,
            "height": self.height,
            "rotation_rad": self.rotation_rad,
            "roi_corners_px": [list(point) for point in self.corners],
        }


def normalize_radians(angle: float) -> float:
    return float(angle - 2.0 * math.pi * math.floor((angle + math.pi) / (2.0 * math.pi)))


def roi_corners(
    x_center: float, y_center: float, width: float, height: float, rotation_rad: float
) -> Tuple[Tuple[float, float], ...]:
    cos_r = math.cos(rotation_rad)
    sin_r = math.sin(rotation_rad)
    vx_x = cos_r * width * 0.5
    vx_y = sin_r * width * 0.5
    vy_x = -sin_r * height * 0.5
    vy_y = cos_r * height * 0.5
    return (
        (x_center - vx_x - vy_x, y_center - vx_y - vy_y),
        (x_center + vx_x - vy_x, y_center + vx_y - vy_y),
        (x_center + vx_x + vy_x, y_center + vx_y + vy_y),
        (x_center - vx_x + vy_x, y_center - vx_y + vy_y),
    )


def build_roi_rect(
    bbox_px: Sequence[float],
    wrist_px: Sequence[float],
    middle_mcp_px: Sequence[float],
    image_width: int,
    image_height: int,
    scale_x: float = 1.8,
    scale_y: float = 1.8,
    shift_x: float = 0.0,
    shift_y: float = -0.1,
) -> RoiRect:
    """Construct the exact rotated ROI used by ``hand_landmarker.cpp``."""

    if len(bbox_px) != 4 or len(wrist_px) != 2 or len(middle_mcp_px) != 2:
        raise ValueError("bbox must have 4 values and Palm keypoints must have 2 values")
    x1 = max(0.0, min(float(image_width - 1), min(float(bbox_px[0]), float(bbox_px[2]))))
    y1 = max(0.0, min(float(image_height - 1), min(float(bbox_px[1]), float(bbox_px[3]))))
    x2 = max(0.0, min(float(image_width - 1), max(float(bbox_px[0]), float(bbox_px[2]))))
    y2 = max(0.0, min(float(image_height - 1), max(float(bbox_px[1]), float(bbox_px[3]))))
    raw_width = max(1.0, x2 - x1)
    raw_height = max(1.0, y2 - y1)
    center_x = 0.5 * (x1 + x2)
    center_y = 0.5 * (y1 + y2)

    dx = float(middle_mcp_px[0]) - float(wrist_px[0])
    dy = float(middle_mcp_px[1]) - float(wrist_px[1])
    rotation = normalize_radians((math.pi * 0.5) - math.atan2(-dy, dx))
    cos_r = math.cos(rotation)
    sin_r = math.sin(rotation)
    center_x += raw_width * float(shift_x) * cos_r - raw_height * float(shift_y) * sin_r
    center_y += raw_width * float(shift_x) * sin_r + raw_height * float(shift_y) * cos_r

    long_side = max(raw_width, raw_height)
    width = long_side * float(scale_x)
    height = long_side * float(scale_y)
    return RoiRect(
        x_center=float(center_x),
        y_center=float(center_y),
        width=float(width),
        height=float(height),
        rotation_rad=float(rotation),
        corners=roi_corners(center_x, center_y, width, height, rotation),
    )


def crop_hand_roi(image, rect: RoiRect, output_width: int = 256, output_height: int = 256):
    from .board_ops import sample_rotated_roi_gray_uint8

    return sample_rotated_roi_gray_uint8(image, rect.corners, output_width, output_height)


def project_normalized_points(
    points: Iterable[Tuple[float, float]], rect: RoiRect
) -> List[Tuple[float, float]]:
    top_left = rect.corners[0]
    top_right = rect.corners[1]
    bottom_left = rect.corners[3]
    projected = []
    for x_value, y_value in points:
        x_value = float(x_value)
        y_value = float(y_value)
        x = top_left[0] + x_value * (top_right[0] - top_left[0]) + y_value * (bottom_left[0] - top_left[0])
        y = top_left[1] + x_value * (top_right[1] - top_left[1]) + y_value * (bottom_left[1] - top_left[1])
        projected.append((float(x), float(y)))
    return projected


def unproject_image_points(
    points: Iterable[Tuple[float, float]], rect: RoiRect
) -> List[Tuple[float, float]]:
    """Map source-image points back into normalized coordinates of ``rect``."""

    top_left = rect.corners[0]
    axis_x = (rect.corners[1][0] - top_left[0], rect.corners[1][1] - top_left[1])
    axis_y = (rect.corners[3][0] - top_left[0], rect.corners[3][1] - top_left[1])
    determinant = axis_x[0] * axis_y[1] - axis_x[1] * axis_y[0]
    if abs(determinant) < 1e-12:
        raise ValueError("Cannot invert degenerate ROI rectangle")
    result = []
    for point_x, point_y in points:
        delta_x = float(point_x) - top_left[0]
        delta_y = float(point_y) - top_left[1]
        x_value = (delta_x * axis_y[1] - delta_y * axis_y[0]) / determinant
        y_value = (axis_x[0] * delta_y - axis_x[1] * delta_x) / determinant
        result.append((float(x_value), float(y_value)))
    return result


def rect_from_record(row: Mapping[str, Any]) -> RoiRect:
    value = row.get("roi_rect") or {}
    corners_value = row.get("roi_corners_px") or value.get("roi_corners_px")
    if corners_value and len(corners_value) == 4:
        corners = tuple((float(point[0]), float(point[1])) for point in corners_value)
    else:
        corners = roi_corners(
            float(value["x_center"]),
            float(value["y_center"]),
            float(value["width"]),
            float(value["height"]),
            float(value.get("rotation_rad", value.get("rotation", 0.0))),
        )
    return RoiRect(
        x_center=float(value["x_center"]),
        y_center=float(value["y_center"]),
        width=float(value["width"]),
        height=float(value["height"]),
        rotation_rad=float(value.get("rotation_rad", value.get("rotation", 0.0))),
        corners=corners,
    )


def rotated_rect_iou(first: RoiRect, second: RoiRect) -> float:
    import cv2
    import numpy as np

    polygon_a = np.asarray(first.corners, dtype=np.float32)
    polygon_b = np.asarray(second.corners, dtype=np.float32)
    area_a = abs(float(cv2.contourArea(polygon_a)))
    area_b = abs(float(cv2.contourArea(polygon_b)))
    intersection, _ = cv2.intersectConvexConvex(polygon_a, polygon_b)
    union = area_a + area_b - float(intersection)
    return max(0.0, min(1.0, float(intersection) / union)) if union > 0.0 else 0.0
