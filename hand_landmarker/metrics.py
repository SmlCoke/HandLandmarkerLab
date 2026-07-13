"""Deterministic metrics shared by validation and tests."""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


def safe_div(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def percentile(values: Sequence[float], quantile: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = min(1.0, max(0.0, float(quantile))) * float(len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - float(lower)
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


class BinaryConfusion:
    def __init__(self) -> None:
        self.tp = 0
        self.fp = 0
        self.tn = 0
        self.fn = 0

    def update(self, expected: bool, predicted: bool) -> None:
        if expected and predicted:
            self.tp += 1
        elif expected:
            self.fn += 1
        elif predicted:
            self.fp += 1
        else:
            self.tn += 1

    def report(self) -> Dict[str, Any]:
        total = self.tp + self.fp + self.tn + self.fn
        precision = safe_div(self.tp, self.tp + self.fp)
        recall = safe_div(self.tp, self.tp + self.fn)
        return {
            "count": total,
            "tp": self.tp,
            "fp": self.fp,
            "tn": self.tn,
            "fn": self.fn,
            "accuracy": safe_div(self.tp + self.tn, total),
            "precision": precision,
            "recall": recall,
            "f1": safe_div(2.0 * precision * recall, precision + recall),
            "false_positive_rate": safe_div(self.fp, self.fp + self.tn),
            "false_negative_rate": safe_div(self.fn, self.fn + self.tp),
        }


class LandmarkMetrics:
    def __init__(self, pck_thresholds: Sequence[float] = (0.05, 0.10, 0.15)) -> None:
        self.thresholds = tuple(float(value) for value in pck_thresholds)
        self.gt_positive = 0
        self.available = 0
        self.roi_mean_errors: List[float] = []
        self.nmes: List[float] = []
        self.per_landmark_errors: List[List[float]] = [[] for _ in range(21)]
        self.pck_hits = {threshold: 0 for threshold in self.thresholds}
        self.pck_total = {threshold: 0 for threshold in self.thresholds}

    @staticmethod
    def _normalizer(points: Sequence[Tuple[float, float]]) -> float:
        xs = [float(point[0]) for point in points]
        ys = [float(point[1]) for point in points]
        return max(math.hypot(max(xs) - min(xs), max(ys) - min(ys)), 1e-6)

    def update(
        self,
        expected: Sequence[Tuple[float, float]],
        predicted: Optional[Sequence[Tuple[float, float]]],
    ) -> None:
        if len(expected) != 21:
            raise ValueError("Expected landmark count must be 21")
        self.gt_positive += 1
        if predicted is None:
            for threshold in self.thresholds:
                self.pck_total[threshold] += 21
            return
        if len(predicted) != 21:
            raise ValueError("Predicted landmark count must be 21")
        self.available += 1
        normalizer = self._normalizer(expected)
        errors = []
        for index, (target, estimate) in enumerate(zip(expected, predicted)):
            error = math.hypot(float(estimate[0]) - float(target[0]), float(estimate[1]) - float(target[1]))
            errors.append(error)
            self.per_landmark_errors[index].append(error)
            normalized = error / normalizer
            for threshold in self.thresholds:
                self.pck_hits[threshold] += int(normalized <= threshold)
                self.pck_total[threshold] += 1
        roi_mean = sum(errors) / 21.0
        self.roi_mean_errors.append(roi_mean)
        self.nmes.append(roi_mean / normalizer)

    def report(self) -> Dict[str, Any]:
        mean_error = safe_div(sum(self.roi_mean_errors), len(self.roi_mean_errors)) if self.roi_mean_errors else None
        mean_nme = safe_div(sum(self.nmes), len(self.nmes)) if self.nmes else None
        return {
            "gt_positive_count": self.gt_positive,
            "prediction_count": self.available,
            "prediction_coverage": safe_div(self.available, self.gt_positive),
            "mean_pixel_error": mean_error,
            "median_pixel_error": percentile(self.roi_mean_errors, 0.50),
            "p90_pixel_error": percentile(self.roi_mean_errors, 0.90),
            "p95_pixel_error": percentile(self.roi_mean_errors, 0.95),
            "mean_nme": mean_nme,
            "median_nme": percentile(self.nmes, 0.50),
            "pck": {
                "{:.2f}".format(threshold): safe_div(self.pck_hits[threshold], self.pck_total[threshold])
                for threshold in self.thresholds
            },
            "per_landmark_mean_pixel_error": [
                safe_div(sum(values), len(values)) if values else None for values in self.per_landmark_errors
            ],
        }


class EvaluationMetrics:
    def __init__(self, pck_thresholds: Sequence[float] = (0.05, 0.10, 0.15)) -> None:
        self.presence = BinaryConfusion()
        self.handedness = BinaryConfusion()  # Right is the positive class.
        self.landmarks = LandmarkMetrics(pck_thresholds)
        self.handedness_count = 0

    def update(
        self,
        expected_presence: bool,
        predicted_presence: bool,
        expected_landmarks: Optional[Sequence[Tuple[float, float]]] = None,
        predicted_landmarks: Optional[Sequence[Tuple[float, float]]] = None,
        expected_handedness: Optional[str] = None,
        predicted_handedness: Optional[str] = None,
    ) -> None:
        self.presence.update(expected_presence, predicted_presence)
        if expected_presence and expected_landmarks is not None:
            self.landmarks.update(expected_landmarks, predicted_landmarks)
        if expected_presence and expected_handedness in {"Left", "Right"} and predicted_handedness in {"Left", "Right"}:
            self.handedness.update(expected_handedness == "Right", predicted_handedness == "Right")
            self.handedness_count += 1

    def report(self) -> Dict[str, Any]:
        handedness = self.handedness.report()
        handedness["eligible_count"] = self.handedness_count
        handedness["left_recall"] = safe_div(self.handedness.tn, self.handedness.tn + self.handedness.fp)
        handedness["right_recall"] = safe_div(self.handedness.tp, self.handedness.tp + self.handedness.fn)
        return {
            "presence": self.presence.report(),
            "landmarks": self.landmarks.report(),
            "handedness": handedness,
        }


def threshold_sweep(
    labels_and_scores: Iterable[Tuple[bool, float]], thresholds: Sequence[float]
) -> List[Dict[str, Any]]:
    pairs = [(bool(label), float(score)) for label, score in labels_and_scores]
    reports = []
    for threshold in thresholds:
        metric = BinaryConfusion()
        for label, score in pairs:
            metric.update(label, score >= float(threshold))
        report = metric.report()
        report["threshold"] = float(threshold)
        reports.append(report)
    return reports

