"""Deterministic Gold-review candidate selection for Hand Landmarker finetuning.

The selectors publish *requests*, not labels.  HLMF restores the authenticated
parent ROI/draft/manifest and is the only component allowed to turn a request
into human Gold.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple

from .contracts import HAND_CONNECTIONS, STRUCTURAL_BONES, ordered_landmarks
from .io_utils import sha256_file


SELECTION_SCHEMA = "finetune_selection_v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
POSITIVE_SAMPLE_TYPES = ("POS_RUNTIME", "POS_LOW_PALM")
NEGATIVE_SAMPLE_TYPES = ("NEG_RUNTIME_CANDIDATE", "NEG_LOW_PALM_CANDIDATE")


def clean_row(row: Mapping[str, Any]) -> Dict[str, Any]:
    return {str(key): value for key, value in row.items() if not str(key).startswith("_")}


def stable_hex(salt: str, *parts: Any) -> str:
    payload = "\x1f".join([str(salt), *(str(part) for part in parts)])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def identity_value(row: Mapping[str, Any]) -> str:
    return str(
        row.get("parent_global_crop_id")
        or row.get("global_crop_id")
        or row.get("crop_id")
        or ""
    )


def identity_tokens(row: Mapping[str, Any]) -> Set[str]:
    """Return all stable identities available across Gold, selection and eval rows."""

    tokens: Set[str] = set()
    fields = {
        "parent": ("parent_global_crop_id", "global_crop_id", "crop_id", "source_crop_id"),
        "source_image": ("source_image_identity", "source_image", "image"),
        "roi_sha256": ("image_sha256", "crop_image_sha256", "source_image_sha256"),
        "pixel_sha256": ("normalized_pixel_sha256",),
    }
    for namespace, keys in fields.items():
        for key in keys:
            value = str(row.get(key) or "").strip().lower()
            if value:
                if namespace == "source_image":
                    dataset = str(
                        row.get("parent_dataset_id") or row.get("dataset_id") or ""
                    ).strip().lower()
                    value = "{}:{}".format(dataset, value)
                tokens.add("{}:{}".format(namespace, value))
    native_crop_id = str(row.get("native_source_crop_id") or "").strip().lower()
    if native_crop_id:
        native_root = str(row.get("native_source_root") or "").strip().lower()
        tokens.add("native:{}:{}".format(native_root, native_crop_id))
    return tokens


def overlaps_identity(row: Mapping[str, Any], occupied_tokens: Set[str]) -> bool:
    return bool(identity_tokens(row) & occupied_tokens)


def largest_remainder(total: int, weights: Mapping[str, float]) -> Dict[str, int]:
    """Allocate an integer total with deterministic largest-remainder rounding."""

    if isinstance(total, bool) or int(total) < 0:
        raise ValueError("quota total must be a non-negative integer")
    total = int(total)
    normalized = {str(key): float(value) for key, value in weights.items()}
    if any(not math.isfinite(value) or value < 0.0 for value in normalized.values()):
        raise ValueError("quota weights must be finite and non-negative")
    weight_sum = sum(normalized.values())
    if total and weight_sum <= 0.0:
        raise ValueError("positive quota requires at least one positive weight")
    raw = {
        key: (total * value / weight_sum if weight_sum else 0.0)
        for key, value in normalized.items()
    }
    result = {key: int(math.floor(value)) for key, value in raw.items()}
    remainder = total - sum(result.values())
    order = sorted(raw, key=lambda key: (-(raw[key] - result[key]), key))
    for key in order[:remainder]:
        result[key] += 1
    return result


def _bounded_quota(
    total: int,
    weights: Mapping[str, float],
    capacities: Mapping[str, int],
) -> Tuple[Dict[str, int], Dict[str, Any]]:
    """Allocate a quota, repeatedly redistributing shortfall to available cells."""

    keys = sorted(set(weights) | set(capacities))
    capacity = {key: max(0, int(capacities.get(key, 0))) for key in keys}
    target = min(max(0, int(total)), sum(capacity.values()))
    result = {key: 0 for key in keys}
    rounds: List[Dict[str, Any]] = []
    remaining = target
    while remaining:
        active = [key for key in keys if result[key] < capacity[key]]
        if not active:
            break
        active_weights = {key: max(0.0, float(weights.get(key, 0.0))) for key in active}
        if sum(active_weights.values()) <= 0.0:
            active_weights = {key: 1.0 for key in active}
        proposed = largest_remainder(remaining, active_weights)
        progress = 0
        applied: Dict[str, int] = {}
        for key in active:
            value = min(proposed.get(key, 0), capacity[key] - result[key])
            if value:
                result[key] += value
                progress += value
                applied[key] = value
        rounds.append({"remaining_before": remaining, "applied": applied})
        if not progress:
            # This is reachable only through a pathological all-zero rounding
            # case; allocate one to the lexicographically first active cell.
            key = active[0]
            result[key] += 1
            progress = 1
            rounds[-1]["applied"] = {key: 1}
        remaining -= progress
    return result, {"requested": int(total), "bounded_total": target, "rounds": rounds}


def _dataset_quotas(
    rows: Sequence[Mapping[str, Any]],
    total: int,
    per_dataset_max: Optional[int],
    user_weights: Optional[Mapping[str, float]] = None,
) -> Tuple[Dict[str, int], Dict[str, Any]]:
    counts = Counter(str(row.get("dataset_id") or "") for row in rows)
    if "" in counts:
        raise ValueError("Every selection candidate must have dataset_id")
    capacities = {
        dataset_id: min(count, int(per_dataset_max)) if per_dataset_max is not None else count
        for dataset_id, count in counts.items()
    }
    configured = dict(user_weights or {})
    weights = {
        dataset_id: float(configured.get(dataset_id, 1.0)) * math.sqrt(count)
        for dataset_id, count in counts.items()
    }
    quotas, redistribution = _bounded_quota(total, weights, capacities)
    return quotas, {
        "eligible_count": dict(sorted(counts.items())),
        "configured_weights": dict(sorted(configured.items())),
        "effective_sqrt_weights": dict(sorted(weights.items())),
        "capacities": dict(sorted(capacities.items())),
        "quotas": dict(sorted(quotas.items())),
        "redistribution": redistribution,
    }


def _diverse_take(
    rows: Sequence[Mapping[str, Any]],
    count: int,
    salt: str,
    score_key: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Prefer one row per group/sequence, then fill by deterministic rank."""

    def rank(row: Mapping[str, Any]) -> Tuple[Any, ...]:
        score = float(row.get(score_key, 0.0)) if score_key else 0.0
        return (-score, stable_hex(salt, identity_value(row)))

    ordered = sorted(rows, key=rank)
    first: List[Mapping[str, Any]] = []
    later: List[Mapping[str, Any]] = []
    seen: Set[str] = set()
    for row in ordered:
        group = str(
            row.get("source_sequence_id")
            or row.get("source_group_id")
            or identity_value(row)
        )
        if group not in seen:
            seen.add(group)
            first.append(row)
        else:
            later.append(row)
    return [clean_row(row) for row in (first + later)[: max(0, int(count))]]


def _enforce_global_dataset_cap(
    selected: Sequence[Mapping[str, Any]],
    eligible: Sequence[Mapping[str, Any]],
    per_dataset_max: Optional[int],
    salt: str,
    score_key: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Enforce ``per_dataset_max`` across all sample-type cells, then refill."""

    if per_dataset_max is None:
        return [clean_row(row) for row in selected]
    cap = int(per_dataset_max)
    if cap < 0:
        raise ValueError("per_dataset_max must be non-negative")
    kept = [clean_row(row) for row in selected]

    def desirability(row: Mapping[str, Any]) -> Tuple[float, str]:
        score = float(row.get(score_key, 0.0)) if score_key else 0.0
        return score, stable_hex(salt, identity_value(row))

    shortfalls: Counter = Counter()
    counts = Counter(str(row.get("dataset_id")) for row in kept)
    for dataset_id in sorted(counts):
        excess = max(0, counts[dataset_id] - cap)
        if not excess:
            continue
        candidates = sorted(
            [row for row in kept if str(row.get("dataset_id")) == dataset_id],
            key=desirability,
        )
        remove_ids = {identity_value(row) for row in candidates[:excess]}
        for row in candidates[:excess]:
            shortfalls[str(row.get("sample_type"))] += 1
        kept = [row for row in kept if identity_value(row) not in remove_ids]
        counts[dataset_id] -= excess
    selected_ids = {identity_value(row) for row in kept}
    for sample_type in sorted(shortfalls):
        pool = sorted(
            [
                row for row in eligible
                if str(row.get("sample_type")) == sample_type
                and identity_value(row) not in selected_ids
            ],
            key=desirability,
            reverse=True,
        )
        for row in pool:
            if shortfalls[sample_type] <= 0:
                break
            dataset_id = str(row.get("dataset_id"))
            if counts[dataset_id] >= cap:
                continue
            value = clean_row(row)
            kept.append(value)
            selected_ids.add(identity_value(value))
            counts[dataset_id] += 1
            shortfalls[sample_type] -= 1
    return kept


def _selection_request(
    row: Mapping[str, Any],
    selector_role: str,
    selection_score: Optional[float],
    salt: str,
    provenance_hash_cache: MutableMapping[Path, str],
) -> Dict[str, Any]:
    """Build the exact request contract consumed by the HLMF materializer."""

    required = (
        "dataset_id",
        "source_crop_id",
        "global_crop_id",
        "parent_manifest_path",
        "parent_draft_path",
        "crop_path",
    )
    missing = [key for key in required if not str(row.get(key) or "").strip()]
    if missing:
        raise ValueError(
            "Selection candidate {} is missing HLMF restore fields {}".format(
                identity_value(row), missing
            )
        )
    paths = {
        "parent_manifest_path": Path(str(row["parent_manifest_path"])),
        "parent_draft_path": Path(str(row["parent_draft_path"])),
        "parent_crop_path": Path(str(row["crop_path"])),
    }
    for key, path in paths.items():
        if not path.is_absolute() or not path.is_file():
            raise ValueError("{} must be an existing absolute file: {}".format(key, path))
        symlink = next(
            (candidate for candidate in (path, *path.parents) if candidate.is_symlink()),
            None,
        )
        if symlink is not None:
            raise ValueError("Selection source paths may not traverse symlinks: {}".format(symlink))
    provenance_hashes = {
        "parent_manifest_sha256": str(row.get("parent_manifest_sha256") or "").lower(),
        "parent_draft_sha256": str(row.get("parent_draft_sha256") or "").lower(),
        "image_sha256": str(
            row.get("image_sha256")
            or row.get("crop_image_sha256")
            or row.get("source_image_sha256")
            or ""
        ).lower(),
    }
    if any(not _SHA256_RE.fullmatch(value) for value in provenance_hashes.values()):
        raise ValueError("Selection request requires three explicit SHA256 provenance fields")
    for path_key, sha_key in (
        ("parent_manifest_path", "parent_manifest_sha256"),
        ("parent_draft_path", "parent_draft_sha256"),
        ("parent_crop_path", "image_sha256"),
    ):
        resolved = paths[path_key].resolve(strict=True)
        actual = provenance_hash_cache.get(resolved)
        if actual is None:
            actual = sha256_file(resolved)
            provenance_hash_cache[resolved] = actual
        if actual != provenance_hashes[sha_key]:
            raise ValueError("Selection provenance SHA mismatch: {}".format(paths[path_key]))
    parent_global = str(row.get("parent_global_crop_id") or row["global_crop_id"])
    result = {
        "schema_version": SELECTION_SCHEMA,
        "selector_role": selector_role,
        "source_kind": {
            "negative_removed_gold": "reviewed_hard_gold",
            "disagreement_gold": "disagreement_gold",
        }[selector_role],
        "parent_dataset_id": str(row["dataset_id"]),
        "parent_source_crop_id": str(row["source_crop_id"]),
        "source_crop_id": str(row["source_crop_id"]),
        "parent_global_crop_id": parent_global,
        "global_crop_id": str(row["global_crop_id"]),
        "parent_manifest_path": str(paths["parent_manifest_path"].resolve()),
        "parent_manifest_sha256": provenance_hashes["parent_manifest_sha256"],
        "parent_draft_path": str(paths["parent_draft_path"].resolve()),
        "parent_draft_sha256": provenance_hashes["parent_draft_sha256"],
        "parent_crop_path": str(paths["parent_crop_path"].resolve()),
        "image_sha256": provenance_hashes["image_sha256"],
        "dataset_id": str(row["dataset_id"]),
        "sample_type": str(row.get("sample_type") or ""),
        "source_group_id": row.get("source_group_id"),
        "source_sequence_id": row.get("source_sequence_id"),
        "selection_score": selection_score,
        "selection_tiebreak": stable_hex(salt, parent_global),
    }
    if selector_role == "negative_removed_gold":
        result["selection_evidence"] = clean_row(row.get("negative_removed_evidence") or {})
    elif selector_role == "disagreement_gold":
        result["selection_evidence"] = {
            key: row.get(key)
            for key in (
                "mean_l2",
                "mean_nme",
                "p90_nme",
                "max_nme",
                "teacher_palm_width",
                "teacher_edge_length",
                "student_edge_length",
                "collapse_log_ratio",
                "bone_vector_nme",
                "teacher_bbox_area",
                "student_bbox_area",
                "hand_flag_error",
                "mean_nme_percentile",
                "p90_nme_percentile",
                "collapse_log_ratio_percentile",
                "bone_vector_nme_percentile",
                "hand_flag_error_percentile",
            )
        }
        prediction = row.get("student_prediction") or {}
        result["selection_evidence"]["checkpoint_sha256"] = prediction.get("checkpoint_sha256")
        result["selection_evidence"]["source_labels_sha256"] = prediction.get("source_labels_sha256")
        result["selection_evidence"]["checkpoint_stage"] = prediction.get("checkpoint_stage")
    return result


def select_negative_removed(
    removed_rows: Sequence[Mapping[str, Any]],
    catalog_rows: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    occupied_parent_ids: Optional[Set[str]] = None,
    occupied_identity_tokens: Optional[Set[str]] = None,
    provenance_hash_cache: Optional[MutableMapping[Path, str]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Select hard negatives from authenticated ``negative_removed`` evidence."""

    if not bool(config.get("enabled", True)) or int(config.get("max_items", 0)) == 0:
        return [], {"status": "disabled", "input_count": len(removed_rows), "actual_selected": 0}
    occupied = set(occupied_parent_ids or set())
    occupied_tokens = set(occupied_identity_tokens or set())
    catalog = {identity_value(row): row for row in catalog_rows}
    if "" in catalog:
        raise ValueError("Pretrain catalog rows require global/crop identity")
    eligible: List[Dict[str, Any]] = []
    for evidence in removed_rows:
        if str(evidence.get("schema_version")) != "negative_removed_manifest_v1":
            raise ValueError("negative_removed selector accepts only negative_removed_manifest_v1")
        if str(evidence.get("partition")) != "negative_removed":
            raise ValueError("Quarantine/non-removed evidence may not enter selector b")
        crop_id = identity_value(evidence)
        source = catalog.get(crop_id)
        if source is None:
            raise ValueError("Removed evidence has no pretrain catalog row: {}".format(crop_id))
        if crop_id in occupied or overlaps_identity(source, occupied_tokens):
            continue
        sample_type = str(source.get("sample_type") or evidence.get("sample_type") or "")
        if sample_type not in NEGATIVE_SAMPLE_TYPES:
            raise ValueError("Removed candidate is not a negative sample type: {}".format(crop_id))
        source_hash = str(evidence.get("source_image_sha256") or "")
        row = clean_row(source)
        if source_hash:
            row["image_sha256"] = source_hash
        row["sample_type"] = sample_type
        row["negative_removed_evidence"] = clean_row(evidence)
        eligible.append(row)

    fractions = {
        key: float(value)
        for key, value in dict(config.get("sample_type_fractions") or {}).items()
    }
    unknown = sorted(set(fractions) - set(NEGATIVE_SAMPLE_TYPES))
    if unknown:
        raise ValueError("Unknown negative_removed sample types: {}".format(unknown))
    total = min(int(config.get("max_items", 0)), len(eligible))
    capacities = Counter(str(row["sample_type"]) for row in eligible)
    cell_quotas, cell_report = _bounded_quota(total, fractions, capacities)
    salt = str(config.get("salt", "negative_removed_gold_v1"))
    per_dataset_max = config.get("per_dataset_max")
    selected: List[Dict[str, Any]] = []
    allocation: Dict[str, Any] = {}
    for sample_type in NEGATIVE_SAMPLE_TYPES:
        cell = [row for row in eligible if row["sample_type"] == sample_type]
        quota = int(cell_quotas.get(sample_type, 0))
        dataset_quotas, dataset_report = _dataset_quotas(
            cell,
            quota,
            int(per_dataset_max) if per_dataset_max is not None else None,
            config.get("dataset_weights"),
        ) if cell else ({}, {})
        allocation[sample_type] = dataset_report
        for dataset_id, dataset_quota in sorted(dataset_quotas.items()):
            rows = [row for row in cell if str(row["dataset_id"]) == dataset_id]
            selected.extend(_diverse_take(rows, dataset_quota, salt + ":" + sample_type))
    selected = _enforce_global_dataset_cap(
        selected,
        eligible,
        int(per_dataset_max) if per_dataset_max is not None else None,
        salt,
    )
    hash_cache = provenance_hash_cache if provenance_hash_cache is not None else {}
    requests = [
        _selection_request(row, "negative_removed_gold", None, salt, hash_cache)
        for row in selected
    ]
    requests.sort(key=lambda row: (str(row["sample_type"]), str(row["selection_tiebreak"])))
    report = {
        "status": "ok",
        "selector_role": "negative_removed_gold",
        "input_count": len(removed_rows),
        "eligible_count": len(eligible),
        "configured_budget": int(config.get("max_items", 0)),
        "actual_selected": len(requests),
        "selected_by_sample_type": dict(sorted(Counter(row["sample_type"] for row in requests).items())),
        "cell_allocation": cell_report,
        "dataset_allocation": allocation,
        "salt": salt,
    }
    return requests, report


def _edge_length(points: Sequence[Tuple[float, float]]) -> float:
    return sum(
        math.hypot(points[a][0] - points[b][0], points[a][1] - points[b][1])
        for a, b in HAND_CONNECTIONS[:20]
    )


def _bbox_area(points: Sequence[Tuple[float, float]]) -> float:
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return max(0.0, max(xs) - min(xs)) * max(0.0, max(ys) - min(ys))


def disagreement_metrics(
    teacher_row: Mapping[str, Any],
    prediction: Mapping[str, Any],
) -> Dict[str, float]:
    """Return interpretable geometry disagreement metrics for one ROI."""

    teacher = ordered_landmarks(teacher_row)
    raw_student = prediction.get("student_landmarks_crop_norm")
    if not isinstance(raw_student, list) or len(raw_student) != 21:
        raise ValueError("student_landmarks_crop_norm must contain 21 points")
    student = [(float(point["x"]), float(point["y"])) for point in raw_student]
    if any(not math.isfinite(value) for point in student for value in point):
        raise ValueError("Student landmarks contain NaN/Inf")
    palm_width = math.hypot(teacher[5][0] - teacher[17][0], teacher[5][1] - teacher[17][1])
    scale = max(palm_width, 0.05)
    distances = sorted(
        math.hypot(tx - sx, ty - sy) for (tx, ty), (sx, sy) in zip(teacher, student)
    )
    normalized = [value / scale for value in distances]
    p90_index = max(0, int(math.ceil(0.90 * len(normalized))) - 1)
    teacher_edge = _edge_length(teacher)
    student_edge = _edge_length(student)
    collapse = math.log(max(student_edge, 1e-8) / max(teacher_edge, 1e-8))
    bone_vector_error = sum(
        math.hypot(
            (student[end][0] - student[start][0])
            - (teacher[end][0] - teacher[start][0]),
            (student[end][1] - student[start][1])
            - (teacher[end][1] - teacher[start][1]),
        )
        for start, end in STRUCTURAL_BONES
    ) / (len(STRUCTURAL_BONES) * scale)
    return {
        "mean_l2": sum(distances) / len(distances),
        "mean_nme": sum(normalized) / len(normalized),
        "p90_nme": normalized[p90_index],
        "max_nme": normalized[-1],
        "teacher_palm_width": palm_width,
        "teacher_edge_length": teacher_edge,
        "student_edge_length": student_edge,
        "collapse_log_ratio": collapse,
        "bone_vector_nme": bone_vector_error,
        "teacher_bbox_area": _bbox_area(teacher),
        "student_bbox_area": _bbox_area(student),
        "hand_flag_error": abs(1.0 - float(prediction.get("student_hand_flag", 0.0))),
    }


def _percentile_ranks(values: Sequence[float]) -> List[float]:
    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    result = [0.0] * len(values)
    denominator = max(1, len(values) - 1)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = ((start + end - 1) / 2.0) / denominator
        for position in range(start, end):
            result[order[position]] = rank
        start = end
    return result


def select_teacher_student(
    teacher_rows: Sequence[Mapping[str, Any]],
    prediction_rows: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    occupied_parent_ids: Optional[Set[str]] = None,
    occupied_identity_tokens: Optional[Set[str]] = None,
    provenance_hash_cache: Optional[MutableMapping[Path, str]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], List[Dict[str, Any]]]:
    """Rank geometry positives by teacher/student disagreement."""

    if not bool(config.get("enabled", True)):
        return [], {"status": "disabled", "actual_selected": 0}, []
    occupied = set(occupied_parent_ids or set())
    occupied_tokens = set(occupied_identity_tokens or set())
    teachers = {identity_value(row): row for row in teacher_rows}
    predictions = {identity_value(row): row for row in prediction_rows}
    if set(teachers) != set(predictions):
        missing = sorted(set(teachers) - set(predictions))
        extra = sorted(set(predictions) - set(teachers))
        raise ValueError("Prediction coverage mismatch: missing={} extra={}".format(missing[:10], extra[:10]))
    scored: List[Dict[str, Any]] = []
    for crop_id in sorted(teachers):
        teacher = teachers[crop_id]
        if crop_id in occupied or overlaps_identity(teacher, occupied_tokens):
            continue
        if str(teacher.get("sample_type")) not in POSITIVE_SAMPLE_TYPES:
            continue
        metrics = disagreement_metrics(teacher, predictions[crop_id])
        row = clean_row(teacher)
        row.update(metrics)
        row["student_prediction"] = clean_row(predictions[crop_id])
        scored.append(row)
    weights = {
        "mean_nme": 1.0,
        "p90_nme": 0.5,
        "collapse_log_ratio": 0.5,
        "bone_vector_nme": 0.5,
        "hand_flag_error": 0.0,
    }
    weights.update({key: float(value) for key, value in dict(config.get("score_weights") or {}).items()})
    unknown = sorted(
        set(weights)
        - {"mean_nme", "p90_nme", "collapse_log_ratio", "bone_vector_nme", "hand_flag_error"}
    )
    if unknown:
        raise ValueError("Unknown disagreement score components: {}".format(unknown))
    if str(config.get("checkpoint_stage", "geometry")) == "geometry" and weights["hand_flag_error"] != 0.0:
        raise ValueError("Geometry disagreement must keep hand_flag_error weight at 0")
    for metric in weights:
        values = [abs(float(row[metric])) if metric == "collapse_log_ratio" else float(row[metric]) for row in scored]
        ranks = _percentile_ranks(values)
        for row, rank in zip(scored, ranks):
            row[metric + "_percentile"] = rank
    for row in scored:
        row["disagreement_score"] = sum(
            weights[metric] * float(row[metric + "_percentile"])
            for metric in weights
        )

    fractions = {key: float(value) for key, value in dict(config.get("sample_type_fractions") or {}).items()}
    total = min(int(config.get("max_items", 0)), len(scored))
    capacities = Counter(str(row["sample_type"]) for row in scored)
    cell_quotas, cell_report = _bounded_quota(total, fractions, capacities)
    salt = str(config.get("salt", "geometry_disagreement_v1"))
    per_dataset_max = config.get("per_dataset_max")
    selected: List[Dict[str, Any]] = []
    allocation: Dict[str, Any] = {}
    for sample_type in POSITIVE_SAMPLE_TYPES:
        cell = [row for row in scored if row["sample_type"] == sample_type]
        quota = int(cell_quotas.get(sample_type, 0))
        dataset_quotas, dataset_report = _dataset_quotas(
            cell,
            quota,
            int(per_dataset_max) if per_dataset_max is not None else None,
            config.get("dataset_weights"),
        ) if cell else ({}, {})
        allocation[sample_type] = dataset_report
        for dataset_id, dataset_quota in sorted(dataset_quotas.items()):
            rows = [row for row in cell if str(row["dataset_id"]) == dataset_id]
            selected.extend(
                _diverse_take(rows, dataset_quota, salt + ":" + sample_type, "disagreement_score")
            )
    selected = _enforce_global_dataset_cap(
        selected,
        scored,
        int(per_dataset_max) if per_dataset_max is not None else None,
        salt,
        "disagreement_score",
    )
    hash_cache = provenance_hash_cache if provenance_hash_cache is not None else {}
    requests = [
        _selection_request(
            row,
            "disagreement_gold",
            float(row["disagreement_score"]),
            salt,
            hash_cache,
        )
        for row in selected
    ]
    requests.sort(key=lambda row: (-float(row["selection_score"]), str(row["selection_tiebreak"])))
    report = {
        "status": "ok",
        "selector_role": "disagreement_gold",
        "input_count": len(teacher_rows),
        "eligible_count": len(scored),
        "configured_budget": int(config.get("max_items", 0)),
        "actual_selected": len(requests),
        "score_weights": weights,
        "selected_by_sample_type": dict(sorted(Counter(row["sample_type"] for row in requests).items())),
        "cell_allocation": cell_report,
        "dataset_allocation": allocation,
        "salt": salt,
    }
    return requests, report, sorted(scored, key=lambda row: (-float(row["disagreement_score"]), identity_value(row)))
