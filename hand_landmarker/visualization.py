"""Arbitrary full-image Palm -> ROI -> Hand inference and review rendering."""

from __future__ import annotations

from pathlib import Path
import math
from typing import Any, Dict, List, Mapping

from .config import resolve_path
from .contracts import HAND_CONNECTIONS
from .evaluation import _runtime_config
from .io_utils import (
    ensure_bgr,
    image_files,
    read_image,
    sha256_file,
    write_image,
    write_json,
    write_jsonl,
)
from .runtime import CascadeRunner


def draw_detections(
    image,
    detections,
    hand_flag_threshold: float,
    draw_options: Mapping[str, Any] = None,
):
    import cv2
    import numpy as np

    output = ensure_bgr(image)
    options = dict(draw_options or {})
    draw_palm_box = bool(options.get("draw_palm_box", True))
    draw_roi = bool(options.get("draw_roi", True))
    draw_landmarks = bool(options.get("draw_landmarks", True))
    draw_scores = bool(options.get("draw_scores", True))
    for index, detection in enumerate(detections):
        bbox = detection.palm.bbox_px(output.shape[1], output.shape[0])
        if draw_palm_box:
            cv2.rectangle(
                output,
                (int(round(bbox[0])), int(round(bbox[1]))),
                (int(round(bbox[2])), int(round(bbox[3]))),
                (0, 200, 255),
                1,
                cv2.LINE_AA,
            )
        if draw_roi:
            polygon = np.rint(detection.roi.corners).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(output, [polygon], True, (0, 255, 0), 2, cv2.LINE_AA)
        visible = detection.hand.hand_flag_score >= float(hand_flag_threshold)
        if visible and draw_landmarks:
            points = [
                (
                    int(round(max(0.0, min(float(output.shape[1] - 1), point[0])))),
                    int(round(max(0.0, min(float(output.shape[0] - 1), point[1])))),
                )
                for point in detection.landmarks_image
            ]
            for start, end in HAND_CONNECTIONS:
                cv2.line(output, points[start], points[end], (0, 255, 255), 2, cv2.LINE_AA)
            for point in points:
                cv2.circle(output, point, 3, (0, 0, 255), -1, cv2.LINE_AA)
        if draw_scores:
            label = "#{} palm={:.2f} hand={:.2f} {}={:.2f}".format(
                index,
                detection.palm.score,
                detection.hand.hand_flag_score,
                detection.hand.handedness,
                max(detection.hand.handedness_score, 1.0 - detection.hand.handedness_score),
            )
            anchor_x = max(0, int(round(min(point[0] for point in detection.roi.corners))))
            anchor_y = max(18, int(round(min(point[1] for point in detection.roi.corners))) - 6)
            cv2.putText(output, label, (anchor_x, anchor_y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(output, label, (anchor_x, anchor_y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 0), 1, cv2.LINE_AA)
    summary = "Palm ROIs={} Hand>={:.2f}:{}".format(
        len(detections),
        float(hand_flag_threshold),
        sum(item.hand.hand_flag_score >= float(hand_flag_threshold) for item in detections),
    )
    cv2.putText(output, summary, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(output, summary, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)
    return output


def infer_folder_from_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    if config.get("schema_version", 1) != 1:
        raise ValueError("Unsupported configuration schema_version")
    if str(config.get("task", "infer_folder")).lower() != "infer_folder":
        raise ValueError("Folder inference entry point requires task: infer_folder")
    input_value = config.get("input", {}).get("images_dir") or config.get("paths", {}).get("input_dir")
    output_value = (
        config.get("output", {}).get("dir")
        or config.get("output", {}).get("directory")
        or config.get("paths", {}).get("output_dir")
    )
    if not input_value or not output_value:
        raise KeyError("Folder inference requires input.images_dir and output.dir")
    input_dir = resolve_path(str(input_value), config)
    output_dir = resolve_path(str(output_value), config)
    if not input_dir.is_dir():
        raise FileNotFoundError("Input image directory not found: {}".format(input_dir))
    try:
        output_dir.relative_to(input_dir)
    except ValueError:
        pass
    else:
        raise ValueError("output.dir must not be the input directory or one of its descendants")
    output_dir.mkdir(parents=True, exist_ok=True)
    rendered_dir = output_dir / "rendered"
    threshold = float(
        config.get("inference", {}).get(
            "hand_flag_threshold",
            config.get("pipeline", {}).get("hand", {}).get("presence_threshold", 0.5),
        )
    )
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("inference.hand_flag_threshold must be finite and in [0,1]")
    recursive = bool(config.get("input", {}).get("recursive", True))
    orientation = str(config.get("input", {}).get("source_orientation", "")).lower()
    allowed_orientations = {
        "",
        "as_stored_no_auto_rotate",
        "sensor_portrait_rotate_clockwise",
        "portrait_rotate_clockwise",
    }
    if orientation not in allowed_orientations:
        raise ValueError(
            "input.source_orientation must be one of {}".format(
                sorted(allowed_orientations)
            )
        )
    rotate_clockwise = bool(config.get("input", {}).get("rotate_clockwise", False)) or orientation in {
        "sensor_portrait_rotate_clockwise",
        "portrait_rotate_clockwise",
    }
    fail_fast = bool(config.get("inference", {}).get("fail_fast", False))
    paths = image_files(input_dir, recursive=recursive)
    configured_extensions = config.get("input", {}).get("extensions")
    if configured_extensions is not None:
        if not isinstance(configured_extensions, list) or not configured_extensions:
            raise ValueError("input.extensions must be a non-empty list when configured")
        extensions = {
            value if str(value).startswith(".") else ".{}".format(value)
            for value in (str(item).lower() for item in configured_extensions)
        }
        paths = [path for path in paths if path.suffix.lower() in extensions]
    overwrite = bool(config.get("output", {}).get("overwrite", False))
    predictions_path = resolve_path(
        str(config.get("output", {}).get("jsonl") or (output_dir / "predictions.jsonl")), config
    )
    summary_path = output_dir / "summary.json"
    if not overwrite and (predictions_path.exists() or summary_path.exists()):
        raise FileExistsError("Inference output already exists; set output.overwrite=true to replace it")
    runtime_config = _runtime_config(config)
    runner = CascadeRunner(runtime_config)
    rows: List[Dict[str, Any]] = []
    failed = 0
    for path in paths:
        try:
            image = read_image(path)
            if image is None:
                raise ValueError("unreadable image")
            if rotate_clockwise:
                import cv2

                image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            detections = runner.predict(image)
            relative = path.relative_to(input_dir)
            # Preserve the original suffix in the output filename so foo.jpg
            # and foo.png can never overwrite one another.
            rendered_relative = relative.with_name(relative.name + ".annotated.png")
            output_path = rendered_dir / rendered_relative
            if output_path.exists() and not overwrite:
                raise FileExistsError("Rendered output already exists: {}".format(output_path))
            write_annotated = bool(config.get("output", {}).get("write_annotated_images", True))
            if write_annotated:
                write_image(
                    output_path,
                    draw_detections(image, detections, threshold, config.get("output", {})),
                )
            row = {
                "image": str(path),
                "rendered": str(output_path) if write_annotated else None,
                "width": int(image.shape[1]),
                "height": int(image.shape[0]),
                "detections": [],
            }
            for detection in detections:
                row["detections"].append(
                    {
                        "palm": detection.palm.as_dict(image.shape[1], image.shape[0]),
                        "roi": detection.roi.as_dict(),
                        "hand_flag_score": detection.hand.hand_flag_score,
                        "hand_flag_raw_score": detection.hand.hand_flag_raw_score,
                        "handedness_score": detection.hand.handedness_score,
                        "handedness_raw_score": detection.hand.handedness_raw_score,
                        "handedness": detection.hand.handedness,
                        "landmarks_roi_norm": [list(point) for point in detection.hand.landmarks_norm],
                        "landmarks_image_px": [list(point) for point in detection.landmarks_image],
                        "landmark_raw_max_abs": detection.hand.landmark_raw_max_abs,
                        "board_landmark_scale_divisor": detection.hand.board_landmark_scale_divisor,
                        "normalized_out_of_range_coordinate_count": (
                            detection.hand.normalized_out_of_range_coordinate_count
                        ),
                    }
                )
            rows.append(row)
        except Exception as exc:
            failed += 1
            rows.append({"image": str(path), "error": str(exc), "detections": []})
            if fail_fast:
                raise
    summary = {
        "status": "failed" if failed else "ok",
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "image_count": len(paths),
        "failed_count": failed,
        "palm_detection_count": sum(len(row.get("detections", [])) for row in rows),
        "predicted_hand_count": sum(
            detection["hand_flag_score"] >= threshold
            for row in rows
            for detection in row.get("detections", [])
        ),
        "hand_flag_threshold": threshold,
        "rotated_clockwise": rotate_clockwise,
        "models": {},
    }
    hand_rows = [
        detection
        for row in rows
        for detection in row.get("detections", [])
    ]
    summary["landmark_output_health"] = {
        "prediction_count": len(hand_rows),
        "non_finite_outputs_rejected": True,
        "maximum_raw_absolute_value": max(
            (float(item["landmark_raw_max_abs"]) for item in hand_rows),
            default=None,
        ),
        "board_scale_divisor_256_count": sum(
            float(item["board_landmark_scale_divisor"]) == 256.0 for item in hand_rows
        ),
        "normalized_out_of_range_hand_count": sum(
            int(item["normalized_out_of_range_coordinate_count"]) > 0 for item in hand_rows
        ),
        "normalized_out_of_range_coordinate_count": sum(
            int(item["normalized_out_of_range_coordinate_count"]) for item in hand_rows
        ),
    }
    for name in ("palm", "hand"):
        model_path = Path(str(runtime_config.get(name, {}).get("model_path", "")))
        if not model_path.is_file():
            raise FileNotFoundError("{} model is not a file: {}".format(name, model_path))
        summary["models"][name] = {
            "path": str(model_path),
            "sha256": sha256_file(model_path),
        }
    config_path_value = config.get("_meta", {}).get("config_path")
    config_path = Path(str(config_path_value)) if config_path_value else None
    summary["config_path"] = str(config_path) if config_path else None
    summary["config_sha256"] = (
        sha256_file(config_path) if config_path is not None and config_path.is_file() else None
    )
    if bool(config.get("output", {}).get("write_jsonl", True)):
        write_jsonl(predictions_path, rows)
    write_json(summary_path, summary)
    return summary
