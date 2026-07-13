"""Vectorized references for the A1 scheduler's uint8 bilinear samplers."""

from __future__ import annotations

from typing import Tuple


def resize_gray_uint8(image, output_width: int, output_height: int):
    """Half-pixel bilinear resize with edge replication and C++-style rounding."""

    import numpy as np

    from .io_utils import to_uint8_gray

    source = to_uint8_gray(image)
    source_height, source_width = source.shape[:2]
    x_float = (np.arange(output_width, dtype=np.float64) + 0.5) * source_width / float(output_width) - 0.5
    y_float = (np.arange(output_height, dtype=np.float64) + 0.5) * source_height / float(output_height) - 0.5
    x0 = np.floor(x_float).astype(np.int64)
    y0 = np.floor(y_float).astype(np.int64)
    wx = x_float - x0
    wy = y_float - y0
    low_x = x0 < 0
    low_y = y0 < 0
    x0[low_x] = 0
    y0[low_y] = 0
    wx[low_x] = 0.0
    wy[low_y] = 0.0
    x1 = x0 + 1
    y1 = y0 + 1
    high_x = x1 >= source_width
    high_y = y1 >= source_height
    x1[high_x] = source_width - 1
    x0[high_x] = x1[high_x]
    wx[high_x] = 0.0
    y1[high_y] = source_height - 1
    y0[high_y] = y1[high_y]
    wy[high_y] = 0.0

    v00 = source[y0[:, None], x0[None, :]].astype(np.float64)
    v01 = source[y0[:, None], x1[None, :]].astype(np.float64)
    v10 = source[y1[:, None], x0[None, :]].astype(np.float64)
    v11 = source[y1[:, None], x1[None, :]].astype(np.float64)
    top = v00 * (1.0 - wx[None, :]) + v01 * wx[None, :]
    bottom = v10 * (1.0 - wx[None, :]) + v11 * wx[None, :]
    values = top * (1.0 - wy[:, None]) + bottom * wy[:, None]
    return np.clip(np.floor(values + 0.5), 0, 255).astype(np.uint8)


def sample_rotated_roi_gray_uint8(image, corners, output_width: int = 256, output_height: int = 256):
    """Endpoint-mapped affine ROI sampler used by ``hand_landmarker.cpp``."""

    import numpy as np

    from .io_utils import to_uint8_gray

    source = to_uint8_gray(image)
    source_height, source_width = source.shape[:2]
    top_left = np.asarray(corners[0], dtype=np.float32)
    top_right = np.asarray(corners[1], dtype=np.float32)
    bottom_left = np.asarray(corners[3], dtype=np.float32)
    tx = np.arange(output_width, dtype=np.float32) / float(max(1, output_width - 1))
    ty = np.arange(output_height, dtype=np.float32) / float(max(1, output_height - 1))
    source_x = (
        top_left[0]
        + tx[None, :] * (top_right[0] - top_left[0])
        + ty[:, None] * (bottom_left[0] - top_left[0])
    )
    source_y = (
        top_left[1]
        + tx[None, :] * (top_right[1] - top_left[1])
        + ty[:, None] * (bottom_left[1] - top_left[1])
    )
    outside = (
        (source_x < -1.0)
        | (source_y < -1.0)
        | (source_x > float(source_width))
        | (source_y > float(source_height))
    )
    x0 = np.floor(source_x).astype(np.int64)
    y0 = np.floor(source_y).astype(np.int64)
    x1 = x0 + 1
    y1 = y0 + 1
    wx = source_x - x0
    wy = source_y - y0

    def gather(xs, ys):
        valid = (xs >= 0) & (ys >= 0) & (xs < source_width) & (ys < source_height)
        clipped_x = np.clip(xs, 0, max(0, source_width - 1))
        clipped_y = np.clip(ys, 0, max(0, source_height - 1))
        values = source[clipped_y, clipped_x].astype(np.float32)
        return np.where(valid, values, 0.0)

    v00 = gather(x0, y0)
    v01 = gather(x1, y0)
    v10 = gather(x0, y1)
    v11 = gather(x1, y1)
    top = v00 * (1.0 - wx) + v01 * wx
    bottom = v10 * (1.0 - wx) + v11 * wx
    values = top * (1.0 - wy) + bottom * wy
    values = np.where(outside, 0.0, values)
    return np.clip(np.floor(values + 0.5), 0, 255).astype(np.uint8)

