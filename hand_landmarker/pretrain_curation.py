"""Persistent, auditable curation for pseudo-labelled pretrain data.

Teacher abstentions are candidates, not verified background. This module
persists curated labels and image hashes while keeping canonical ROI paths in
``train_sources``. Every rejected or unverified row remains in an on-disk
audit catalog instead of being filtered only in training memory.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

from .config import load_config, resolve_path
from .io_utils import image_files, read_jsonl, sha256_file, write_json, write_jsonl


CURATION_SCHEMA = "pretrain_curation_v1"
POSITIVE_SAMPLE_TYPES = {"POS_RUNTIME", "POS_LOW_PALM"}
NEGATIVE_SAMPLE_TYPES = {"NEG_RUNTIME_CANDIDATE", "NEG_LOW_PALM_CANDIDATE"}
REVIEW_DECISIONS = {"CONFIRMED_NEGATIVE", "FALSE_NEGATIVE_HAND_VISIBLE", "HOLD"}


def _config_mapping(config: Union[Mapping[str, Any], str, Path]) -> Dict[str, Any]:
    return load_config(config) if isinstance(config, (str, Path)) else dict(config)


def verify_curation_manifest(
    config: Mapping[str, Any],
    dataset: Mapping[str, Any],
    error_type=ValueError,
) -> Optional[Dict[str, Any]]:
    """Authenticate a configured curated label snapshot before it is consumed."""

    manifest_value = dataset.get("curation_manifest")
    if not manifest_value:
        return None
    manifest_path = resolve_path(str(manifest_value), config)
    if not manifest_path.is_file():
        raise error_type("Configured curation manifest does not exist: {}".format(manifest_path))
    try:
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (OSError, ValueError) as exc:
        raise error_type(
            "Could not read curation manifest {}: {}".format(manifest_path, exc)
        ) from exc
    if not isinstance(manifest, Mapping):
        raise error_type("Curation manifest root must be an object: {}".format(manifest_path))
    required_schema = dataset.get("require_curation_schema")
    if required_schema and str(manifest.get("schema_version")) != str(required_schema):
        raise error_type(
            "Curation manifest schema mismatch: expected {}, got {}".format(
                required_schema, manifest.get("schema_version")
            )
        )
    labels_path = resolve_path(str(dataset.get("labels", "")), config)
    output_dir_value = manifest.get("output_dir")
    if not output_dir_value:
        raise error_type("Curation manifest has no output_dir: {}".format(manifest_path))
    output_root = Path(str(output_dir_value)).resolve()
    try:
        relative_labels = labels_path.resolve().relative_to(output_root).as_posix()
    except (ValueError, OSError) as exc:
        raise error_type(
            "Training labels {} are outside curation snapshot {}".format(
                labels_path, output_root
            )
        ) from exc
    artifact = (manifest.get("artifacts") or {}).get(relative_labels)
    if not isinstance(artifact, Mapping) or not artifact.get("sha256"):
        raise error_type(
            "Curation manifest does not authenticate training labels {}".format(relative_labels)
        )
    actual_hash = sha256_file(labels_path)
    if actual_hash != str(artifact["sha256"]):
        raise error_type(
            "Curated training labels hash mismatch for {}: expected {}, got {}".format(
                labels_path, artifact["sha256"], actual_hash
            )
        )
    return {
        "path": str(manifest_path),
        "sha256": sha256_file(manifest_path),
        "schema_version": manifest.get("schema_version"),
        "source_labels_sha256": (manifest.get("source") or {}).get("labels_sha256"),
        "training_labels_relative_path": relative_labels,
        "training_labels_sha256": actual_hash,
        "image_count": (manifest.get("images") or {}).get("count"),
        "image_aggregate_sha256": (manifest.get("images") or {}).get(
            "aggregate_sha256"
        ),
    }


def _is_within(path: Path, directory: Path) -> bool:
    try:
        path.resolve().relative_to(directory.resolve())
        return True
    except (OSError, ValueError):
        return False


def _guard_curation_output(
    output_dir: Path,
    source_labels: Path,
    source_crops: Sequence[Path],
    allow_overwrite: bool,
) -> None:
    """Refuse dangerous recursive replacement targets and unknown directories."""

    resolved = output_dir.resolve()
    filesystem_root = Path(resolved.anchor).resolve()
    home = Path.home().resolve()
    if resolved in {filesystem_root, home}:
        raise ValueError("Refusing dangerous curation output directory: {}".format(resolved))
    protected_sources = [source_labels, *source_crops]
    conflict = next(
        (source for source in protected_sources if _is_within(source, resolved)),
        None,
    )
    if conflict is not None:
        raise ValueError(
            "Refusing curation output directory {} because it contains source data {}".format(
                resolved, conflict.resolve()
            )
        )
    if not output_dir.exists():
        return
    if output_dir.is_symlink() or not output_dir.is_dir():
        raise ValueError("Refusing to replace non-directory or symlink output: {}".format(output_dir))
    if not allow_overwrite:
        raise FileExistsError(
            "Curated pretrain snapshot already exists; choose a new version or pass --overwrite: {}".format(
                output_dir
            )
        )
    sentinel_path = output_dir / "qc" / "sha256_manifest.json"
    if not sentinel_path.is_file():
        raise ValueError(
            "Refusing to overwrite a directory without a curation manifest: {}".format(output_dir)
        )
    try:
        with sentinel_path.open("r", encoding="utf-8") as handle:
            sentinel = json.load(handle)
    except (OSError, ValueError) as exc:
        raise ValueError(
            "Refusing to overwrite a directory with an unreadable curation manifest: {}".format(
                sentinel_path
            )
        ) from exc
    sentinel_output = sentinel.get("output_dir") if isinstance(sentinel, Mapping) else None
    if (
        not isinstance(sentinel, Mapping)
        or sentinel.get("schema_version") != CURATION_SCHEMA
        or not sentinel_output
        or Path(str(sentinel_output)).resolve() != resolved
    ):
        raise ValueError(
            "Refusing to overwrite a directory whose curation manifest does not identify it: {}".format(
                output_dir
            )
        )


def _clean_row(row: Mapping[str, Any]) -> Dict[str, Any]:
    return {str(key): value for key, value in row.items() if not str(key).startswith("_")}


def _source_group_key(row: Mapping[str, Any]) -> Tuple[str, str]:
    return str(row.get("dataset_id") or ""), str(row.get("source_group_id") or "")


def _ordered_points(row: Mapping[str, Any], field: str) -> List[Tuple[float, float]]:
    raw = row.get(field)
    if not isinstance(raw, list) or len(raw) != 21:
        raise ValueError("{} must contain exactly 21 points".format(field))
    by_id: Dict[int, Tuple[float, float]] = {}
    for offset, point in enumerate(raw):
        if not isinstance(point, Mapping):
            raise ValueError("{} point {} is not an object".format(field, offset))
        point_id = int(point.get("id"))
        x_value, y_value = float(point["x"]), float(point["y"])
        if point_id in by_id:
            raise ValueError("{} has duplicate landmark id {}".format(field, point_id))
        if not math.isfinite(x_value) or not math.isfinite(y_value):
            raise ValueError("{} has a non-finite coordinate".format(field))
        by_id[point_id] = (x_value, y_value)
    if set(by_id) != set(range(21)):
        raise ValueError("{} ids must be exactly 0..20".format(field))
    return [by_id[index] for index in range(21)]


def _positive_reasons(
    row: Mapping[str, Any],
    allowed_quality_tiers: Sequence[str],
    minimum_coordinate: float,
    maximum_coordinate: float,
) -> List[str]:
    reasons: List[str] = []
    if str(row.get("sample_type")) not in POSITIVE_SAMPLE_TYPES:
        reasons.append("INVALID_POSITIVE_SAMPLE_TYPE")
    if allowed_quality_tiers and str(row.get("quality_tier")) not in set(allowed_quality_tiers):
        reasons.append("DISALLOWED_QUALITY_TIER")
    if row.get("needs_review") is True:
        reasons.append("SOURCE_MARKED_NEEDS_REVIEW")
    try:
        normalized = _ordered_points(row, "landmarks_crop_norm")
        _ordered_points(row, "landmarks_crop_px")
        _ordered_points(row, "landmarks_image_px")
    except (KeyError, TypeError, ValueError) as exc:
        normalized = []
        reasons.append("INVALID_LANDMARKS:{}".format(str(exc)))
    if normalized and any(
        x < minimum_coordinate
        or x > maximum_coordinate
        or y < minimum_coordinate
        or y > maximum_coordinate
        for x, y in normalized
    ):
        reasons.append("LANDMARK_OUT_OF_ALLOWED_RANGE")
    crop_path = Path(str(row.get("crop_path") or ""))
    if not crop_path.is_file():
        reasons.append("MISSING_CROP_IMAGE")
    return sorted(set(reasons))


def _point_on_segment(
    point: Tuple[float, float],
    start: Tuple[float, float],
    end: Tuple[float, float],
    epsilon: float = 1.0e-6,
) -> bool:
    px, py = point
    ax, ay = start
    bx, by = end
    cross = (px - ax) * (by - ay) - (py - ay) * (bx - ax)
    if abs(cross) > epsilon:
        return False
    dot = (px - ax) * (px - bx) + (py - ay) * (py - by)
    return dot <= epsilon


def _point_in_polygon(point: Tuple[float, float], polygon: Sequence[Tuple[float, float]]) -> bool:
    if len(polygon) < 3:
        return False
    inside = False
    px, py = point
    previous = polygon[-1]
    for current in polygon:
        if _point_on_segment(point, previous, current):
            return True
        ax, ay = previous
        bx, by = current
        if (ay > py) != (by > py):
            intersection_x = (bx - ax) * (py - ay) / (by - ay) + ax
            if px < intersection_x:
                inside = not inside
        previous = current
    return inside


def _negative_overlap_count(
    row: Mapping[str, Any],
    positives: Sequence[Mapping[str, Any]],
    core_landmark_ids: Sequence[int],
) -> int:
    raw_polygon = row.get("roi_corners_px")
    if not isinstance(raw_polygon, list) or len(raw_polygon) != 4:
        return 0
    try:
        polygon = [(float(point[0]), float(point[1])) for point in raw_polygon]
    except (TypeError, ValueError, IndexError):
        return 0
    wanted = set(int(value) for value in core_landmark_ids)
    maximum = 0
    for positive in positives:
        try:
            points = _ordered_points(positive, "landmarks_image_px")
        except (KeyError, TypeError, ValueError):
            continue
        maximum = max(
            maximum,
            sum(
                _point_in_polygon(points[point_id], polygon)
                for point_id in wanted
                if 0 <= point_id < len(points)
            ),
        )
    return maximum


def _load_review_decisions(
    path_value: Optional[Any], config: Mapping[str, Any]
) -> Dict[str, Dict[str, Any]]:
    if not path_value:
        return {}
    path = resolve_path(str(path_value), config)
    if not path.is_file():
        raise FileNotFoundError("Negative review decisions file not found: {}".format(path))
    decisions: Dict[str, Dict[str, Any]] = {}
    for raw in read_jsonl(path):
        row = _clean_row(raw)
        crop_id = str(row.get("crop_id") or "")
        decision = str(row.get("decision") or "")
        if not crop_id:
            raise ValueError("Every review decision must provide crop_id")
        if crop_id in decisions:
            raise ValueError("Duplicate review decision for {}".format(crop_id))
        if decision not in REVIEW_DECISIONS:
            raise ValueError(
                "Unsupported review decision {!r} for {}; expected one of {}".format(
                    decision, crop_id, sorted(REVIEW_DECISIONS)
                )
            )
        for field in ("reviewer", "reviewed_at"):
            if not str(row.get(field) or "").strip():
                raise ValueError(
                    "Every human review decision requires non-empty {} for {}".format(
                        field, crop_id
                    )
                )
        decisions[crop_id] = row
    return decisions


def _safe_path_component(value: Any) -> str:
    text = "".join(
        character if character.isalnum() or character in {"-", "_"} else "_"
        for character in str(value or "unknown")
    ).strip("_")
    return text or "unknown"


def _negative_review_paths(
    config: Mapping[str, Any], review: Mapping[str, Any]
) -> Dict[str, Path]:
    decisions_value = review.get("decisions_file")
    if not str(decisions_value or "").strip():
        raise ValueError("review.decisions_file is required")
    decisions_file = resolve_path(str(decisions_value), config)
    workspace_dir = decisions_file.parent
    subdir = Path(str(review.get("candidates_subdir") or "negative_candidates"))
    if subdir.is_absolute() or not subdir.parts or any(part in {"", ".", ".."} for part in subdir.parts):
        raise ValueError("review.candidates_subdir must be a safe relative path")
    return {
        "workspace_dir": workspace_dir,
        "candidates_dir": workspace_dir / subdir,
        "decisions_file": decisions_file,
        "manifest_file": workspace_dir / "review_manifest.jsonl",
        "report_file": workspace_dir / "review_report.json",
        "instructions_file": workspace_dir / "REVIEW_INSTRUCTIONS.md",
    }


def _copy_review_candidate(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(source), str(destination))


def _prepare_negative_review_workspace(
    config: Mapping[str, Any],
    review: Mapping[str, Any],
    snapshot_root: Path,
    source_labels_sha256: str,
) -> Dict[str, Any]:
    paths = _negative_review_paths(config, review)
    workspace = paths["workspace_dir"]
    if workspace.exists():
        raise FileExistsError(
            "Negative review workspace already exists; use a new HAND_PRETRAIN_ID or finish its review: {}".format(
                workspace
            )
        )
    workspace.parent.mkdir(parents=True, exist_ok=True)
    temporary: Optional[Path] = Path(
        tempfile.mkdtemp(prefix=workspace.name + ".tmp.", dir=str(workspace.parent))
    )
    try:
        assert temporary is not None
        candidates_subdir = paths["candidates_dir"].relative_to(workspace)
        queue = read_jsonl(snapshot_root / "audit" / "negative_review_queue.jsonl")
        manifest_rows: List[Dict[str, Any]] = []
        seen_relative = set()
        for row in queue:
            source = Path(str(row.get("crop_path") or ""))
            if not source.is_file():
                raise FileNotFoundError("Negative review source crop is missing: {}".format(source))
            sample_type = _safe_path_component(row.get("sample_type"))
            dataset_id = _safe_path_component(row.get("dataset_id"))
            digest = hashlib.sha256(
                str(row.get("crop_id")).encode("utf-8")
            ).hexdigest()[:20]
            candidate_relative = (
                Path(sample_type) / dataset_id / (digest + "__" + source.name)
            )
            if candidate_relative.as_posix() in seen_relative:
                raise ValueError("Duplicate review candidate path: {}".format(candidate_relative))
            seen_relative.add(candidate_relative.as_posix())
            destination = temporary / candidates_subdir / candidate_relative
            _copy_review_candidate(source, destination)
            expected_hash = str(
                ((row.get("pretrain_curation") or {}).get("review_image_sha256") or "")
            )
            actual_hash = sha256_file(destination)
            if expected_hash and actual_hash != expected_hash:
                raise RuntimeError(
                    "Review candidate hash mismatch for {}".format(row.get("crop_id"))
                )
            manifest_rows.append(
                {
                    "crop_id": str(row.get("crop_id")),
                    "dataset_id": str(row.get("dataset_id")),
                    "sample_type": str(row.get("sample_type")),
                    "candidate_relative_path": candidate_relative.as_posix(),
                    "candidate_path": str(paths["candidates_dir"] / candidate_relative),
                    "sha256": actual_hash,
                    "source_labels_sha256": source_labels_sha256,
                    "automatic_reasons": list(
                        (row.get("pretrain_curation") or {}).get("reasons") or []
                    ),
                    "source_crop_path": str(source.resolve()),
                }
            )
        manifest_rows.sort(key=lambda row: str(row["crop_id"]))
        write_jsonl(temporary / paths["manifest_file"].name, manifest_rows)
        instructions = (
            "# Negative candidate visual review\n\n"
            "Open `negative_candidates/` recursively. Delete every image that contains "
            "a hand, fingers, wrist, or uncertain hand-like content. Retain only clear "
            "background. Do not add, rename, move, or edit images. When all reviewers "
            "finish, run `make pretrain-curate-reviewed`.\n"
        )
        with (temporary / paths["instructions_file"].name).open(
            "w", encoding="utf-8", newline="\n"
        ) as handle:
            handle.write(instructions)
        report = {
            "status": "awaiting_visual_review",
            "review_method": "delete_hand_images_retain_background",
            "workspace_dir": str(workspace),
            "candidates_dir": str(paths["candidates_dir"]),
            "manifest_file": str(paths["manifest_file"]),
            "decisions_file": str(paths["decisions_file"]),
            "candidate_count": len(manifest_rows),
            "source_labels_sha256": source_labels_sha256,
        }
        write_json(temporary / paths["report_file"].name, report)
        os.replace(str(temporary), str(workspace))
        temporary = None
        return report
    finally:
        if temporary is not None and temporary.exists():
            shutil.rmtree(str(temporary), ignore_errors=True)


def _finalize_retained_negative_review(
    config: Mapping[str, Any],
    review: Mapping[str, Any],
    source_labels_sha256: str,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    paths = _negative_review_paths(config, review)
    if not paths["workspace_dir"].is_dir():
        raise FileNotFoundError(
            "Negative review workspace does not exist: {}".format(paths["workspace_dir"])
        )
    if not paths["candidates_dir"].is_dir():
        raise FileNotFoundError(
            "Negative review candidates directory does not exist: {}".format(
                paths["candidates_dir"]
            )
        )
    if not paths["manifest_file"].is_file():
        raise FileNotFoundError(
            "Negative review manifest does not exist: {}".format(paths["manifest_file"])
        )
    manifest_rows = [_clean_row(row) for row in read_jsonl(paths["manifest_file"])]
    expected: Dict[str, Dict[str, Any]] = {}
    for row in manifest_rows:
        relative = str(row.get("candidate_relative_path") or "")
        if not relative:
            raise ValueError("Every negative review manifest row requires candidate_relative_path")
        if relative in expected:
            raise ValueError("Duplicate negative review candidate path: {}".format(relative))
        if str(row.get("source_labels_sha256") or "") != source_labels_sha256:
            raise ValueError(
                "Negative review workspace belongs to a different source-label snapshot"
            )
        expected[relative] = row

    actual: Dict[str, Path] = {}
    for path in image_files(paths["candidates_dir"], recursive=True):
        relative = path.relative_to(paths["candidates_dir"]).as_posix()
        if relative in actual:
            raise ValueError("Duplicate retained negative path: {}".format(relative))
        actual[relative] = path
    unknown = sorted(set(actual) - set(expected))
    if unknown:
        raise ValueError(
            "Review folder contains images not present in review_manifest.jsonl: {}".format(
                unknown[:10]
            )
        )
    reviewer = str(review.get("reviewer") or "").strip()
    if not reviewer:
        raise ValueError("review.reviewer is required")
    reviewed_at = datetime.now(timezone.utc).astimezone().isoformat()
    decisions_rows: List[Dict[str, Any]] = []
    for relative in sorted(actual):
        row = expected[relative]
        actual_hash = sha256_file(actual[relative])
        if actual_hash != str(row.get("sha256") or ""):
            raise ValueError(
                "Retained review image was modified: {}".format(actual[relative])
            )
        decisions_rows.append(
            {
                "crop_id": str(row.get("crop_id")),
                "decision": "CONFIRMED_NEGATIVE",
                "reviewer": reviewer,
                "reviewed_at": reviewed_at,
                "review_method": "retained_after_visual_deletion_review",
                "review_image_path": str(actual[relative].resolve()),
                "review_image_sha256": actual_hash,
            }
        )
    write_jsonl(paths["decisions_file"], decisions_rows)
    report = {
        "status": "visual_review_finalized",
        "review_method": "delete_hand_images_retain_background",
        "reviewer": reviewer,
        "reviewed_at": reviewed_at,
        "workspace_dir": str(paths["workspace_dir"]),
        "candidates_dir": str(paths["candidates_dir"]),
        "manifest_file": str(paths["manifest_file"]),
        "decisions_file": str(paths["decisions_file"]),
        "original_candidate_count": len(expected),
        "retained_confirmed_count": len(decisions_rows),
        "deleted_or_rejected_count": len(expected) - len(decisions_rows),
        "decisions_sha256": sha256_file(paths["decisions_file"]),
        "source_labels_sha256": source_labels_sha256,
    }
    write_json(paths["report_file"], report)
    return _load_review_decisions(paths["decisions_file"], config), report


def _decision_metadata(
    action: str,
    reasons: Sequence[str],
    source_labels_sha256: str,
    overlap_core_points: int = 0,
    review: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "schema_version": CURATION_SCHEMA,
        "action": str(action),
        "reasons": sorted(set(str(value) for value in reasons)),
        "source_labels_sha256": source_labels_sha256,
        "overlap_confirmed_hand_core_points": int(overlap_core_points),
        "negative_evidence": "human_confirmed" if action == "INCLUDE_CONFIRMED_NEGATIVE" else None,
        "review": dict(review) if review else None,
    }


def _stable_smoke_subset(
    rows: Sequence[Mapping[str, Any]], count: int, salt: str
) -> List[Dict[str, Any]]:
    if count <= 0:
        return []
    groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row.get("dataset_id")), str(row.get("sample_type")))].append(dict(row))
    for key in groups:
        groups[key].sort(
            key=lambda row: hashlib.sha256(
                (salt + "\0" + str(row.get("crop_id"))).encode("utf-8")
            ).hexdigest()
        )
    selected: List[Dict[str, Any]] = []
    keys = sorted(groups)
    offset = 0
    while len(selected) < min(int(count), len(rows)):
        progressed = False
        for key in keys:
            if offset < len(groups[key]) and len(selected) < int(count):
                selected.append(groups[key][offset])
                progressed = True
        if not progressed:
            break
        offset += 1
    return selected


def _reference_crop(
    row: Mapping[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    source = Path(str(row.get("crop_path") or ""))
    if not source.is_file():
        raise FileNotFoundError("Canonical crop does not exist: {}".format(source))
    source = source.resolve()
    image_sha256 = sha256_file(source)
    value = dict(row)
    value["crop_path"] = str(source)
    value["pretrain_curation"] = dict(value["pretrain_curation"])
    value["pretrain_curation"].update(
        {
            "image_storage": "train_sources_reference",
            "image_sha256": image_sha256,
        }
    )
    review = value["pretrain_curation"].get("review") or {}
    reviewed_hash = str(review.get("review_image_sha256") or "")
    if reviewed_hash and image_sha256 != reviewed_hash:
        raise RuntimeError(
            "Reviewed source crop changed after visual review: {}".format(source)
        )
    manifest = {
        "crop_id": str(row.get("crop_id")),
        "path": str(source),
        "sha256": image_sha256,
        "size_bytes": int(source.stat().st_size),
        "storage": "train_sources_reference",
    }
    return value, manifest


def _reference_review_crop(
    row: Mapping[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Record a held source crop for later workspace generation."""
    source = Path(str(row.get("crop_path") or ""))
    if not source.is_file():
        raise FileNotFoundError("Negative review crop does not exist: {}".format(source))
    source = source.resolve()
    digest_value = sha256_file(source)
    value = dict(row)
    value["crop_path"] = str(source)
    value["pretrain_curation"] = dict(value["pretrain_curation"])
    value["pretrain_curation"].update(
        {
            "review_image_storage": "train_sources_reference",
            "review_image_sha256": digest_value,
        }
    )
    return value, {
        "crop_id": str(row.get("crop_id")),
        "path": str(source),
        "sha256": digest_value,
        "size_bytes": int(source.stat().st_size),
        "storage": "train_sources_reference",
    }


def _verify_existing_source_snapshot(
    output_dir: Path,
    image_manifest: Sequence[Mapping[str, Any]],
    review_manifest: Sequence[Mapping[str, Any]],
) -> None:
    """Fail if source ROI bytes changed between curation and review finalization."""

    if not output_dir.exists():
        return
    old_rows: List[Mapping[str, Any]] = []
    for relative in (
        Path("audit") / "image_manifest.jsonl",
        Path("audit") / "review_image_manifest.jsonl",
    ):
        path = output_dir / relative
        if not path.is_file():
            raise ValueError("Existing curation snapshot lacks {}".format(path))
        old_rows.extend(read_jsonl(path))
    old_hashes = {str(row.get("crop_id")): str(row.get("sha256")) for row in old_rows}
    new_hashes = {
        str(row.get("crop_id")): str(row.get("sha256"))
        for row in [*image_manifest, *review_manifest]
    }
    if old_hashes.keys() != new_hashes.keys():
        raise RuntimeError("Source ROI membership changed after the initial curation snapshot")
    changed = sorted(
        crop_id
        for crop_id, digest in old_hashes.items()
        if new_hashes.get(crop_id) != digest
    )
    if changed:
        raise RuntimeError(
            "Source ROI bytes changed after the initial curation snapshot: {}".format(
                changed[:10]
            )
        )


def _counter_dict(values: Iterable[Any]) -> Dict[str, int]:
    return dict(sorted(Counter(str(value) for value in values).items()))


def curate_pretrain_from_config(
    config: Union[Mapping[str, Any], str, Path],
    overwrite: Optional[bool] = None,
    finalize_review: bool = False,
) -> Dict[str, Any]:
    """Create an authenticated curated pretrain index on disk."""

    cfg = _config_mapping(config)
    if str(cfg.get("task")) != "curate_pretrain":
        raise ValueError("Curation config task must be curate_pretrain")
    source_cfg = dict(cfg.get("source", {}))
    output_cfg = dict(cfg.get("output", {}))
    rules = dict(cfg.get("curation", {}))
    review_cfg = dict(cfg.get("review", {}))
    source_labels = resolve_path(source_cfg.get("labels", ""), cfg)
    if not source_labels.is_file():
        raise FileNotFoundError("Source pretrain labels not found: {}".format(source_labels))
    source_crop_root_value = source_cfg.get("crop_root")
    if not str(source_crop_root_value or "").strip():
        raise ValueError("source.crop_root is required")
    source_crop_root = resolve_path(str(source_crop_root_value), cfg)
    if not source_crop_root.is_dir():
        raise FileNotFoundError("Configured train source ROI root not found: {}".format(source_crop_root))
    output_dir = resolve_path(output_cfg.get("dir", ""), cfg)
    if not str(output_cfg.get("dir", "")):
        raise ValueError("output.dir is required")
    allow_overwrite = bool(output_cfg.get("overwrite", False) if overwrite is None else overwrite)
    review_paths = _negative_review_paths(cfg, review_cfg)
    if finalize_review and not allow_overwrite:
        raise ValueError("Finalizing retained negative review requires overwrite=true")
    if not finalize_review and review_paths["workspace_dir"].exists():
        raise FileExistsError(
            "Negative review workspace already exists; use a new HAND_PRETRAIN_ID or finish its review: {}".format(
                review_paths["workspace_dir"]
            )
        )

    source_hash = sha256_file(source_labels)
    records = [_clean_row(row) for row in read_jsonl(source_labels)]
    if not records:
        raise ValueError("Source pretrain labels are empty")
    records.sort(key=lambda row: str(row.get("crop_id") or ""))
    crop_ids = [str(row.get("crop_id") or "") for row in records]
    if any(not value for value in crop_ids):
        raise ValueError("Every source record must have crop_id")
    duplicates = [value for value, count in Counter(crop_ids).items() if count > 1]
    if duplicates:
        raise ValueError("Duplicate crop_id values in source labels: {}".format(duplicates[:10]))
    source_crops = [Path(str(row.get("crop_path") or "")) for row in records]
    outside_source_root = [
        str(path)
        for path in source_crops
        if not path.is_file() or not _is_within(path, source_crop_root)
    ]
    if outside_source_root:
        raise ValueError(
            "Every source crop must be an existing file under source.crop_root {}: {}".format(
                source_crop_root, outside_source_root[:10]
            )
        )
    _guard_curation_output(
        output_dir, source_labels, source_crops, allow_overwrite
    )

    review_finalize_report: Optional[Dict[str, Any]] = None
    if finalize_review:
        decisions, review_finalize_report = _finalize_retained_negative_review(
            cfg, review_cfg, source_hash
        )
        review_path = review_paths["decisions_file"]
    else:
        decisions = {}
        review_path = None
    review_source = {
        "path": str(review_paths["decisions_file"]),
        "sha256": sha256_file(review_path) if review_path else None,
        "decision_count": len(decisions),
        "method": "retained_after_visual_deletion_review" if finalize_review else None,
    }
    unknown_decisions = sorted(set(decisions) - set(crop_ids))
    if unknown_decisions:
        raise ValueError("Review decisions reference unknown crop_id values: {}".format(unknown_decisions[:10]))
    rows_by_id = {str(row["crop_id"]): row for row in records}
    invalid_decision_targets = sorted(
        crop_id
        for crop_id in decisions
        if bool((rows_by_id[crop_id].get("hand_presence") or {}).get("present", False))
        or str(rows_by_id[crop_id].get("sample_type")) not in NEGATIVE_SAMPLE_TYPES
    )
    if invalid_decision_targets:
        raise ValueError(
            "Negative review decisions may reference only negative candidates: {}".format(
                invalid_decision_targets[:10]
            )
        )
    allowed_tiers = [str(value) for value in rules.get("allowed_positive_quality_tiers", ["HIGH", "MEDIUM"])]
    coordinate_range = list(rules.get("normalized_coordinate_range", [0.0, 1.0]))
    if len(coordinate_range) != 2 or float(coordinate_range[0]) > float(coordinate_range[1]):
        raise ValueError("curation.normalized_coordinate_range must be [minimum, maximum]")
    minimum_coordinate, maximum_coordinate = float(coordinate_range[0]), float(coordinate_range[1])
    core_ids = [int(value) for value in rules.get("overlap_core_landmark_ids", [0, 1, 5, 9, 13, 17])]
    overlap_minimum = int(rules.get("overlap_minimum_core_points", 3))
    if overlap_minimum < 1 or overlap_minimum > len(set(core_ids)):
        raise ValueError("curation.overlap_minimum_core_points is invalid")

    positives_by_group: Dict[Tuple[str, str], List[Mapping[str, Any]]] = defaultdict(list)
    for row in records:
        if bool((row.get("hand_presence") or {}).get("present", False)):
            positives_by_group[_source_group_key(row)].append(row)

    catalog: List[Dict[str, Any]] = []
    included_positive_source: List[Dict[str, Any]] = []
    confirmed_negative_source: List[Dict[str, Any]] = []
    excluded: List[Dict[str, Any]] = []
    negative_review_queue: List[Dict[str, Any]] = []
    reason_counts: Counter = Counter()
    overlap_count = 0
    for row in records:
        present = bool((row.get("hand_presence") or {}).get("present", False))
        crop_id = str(row["crop_id"])
        if present:
            reasons = _positive_reasons(
                row, allowed_tiers, minimum_coordinate, maximum_coordinate
            )
            action = "INCLUDE_LANDMARKS" if not reasons else "QUARANTINE_POSITIVE"
            metadata = _decision_metadata(action, reasons, source_hash)
            value = dict(row)
            value["pretrain_curation"] = metadata
            if reasons:
                excluded.append(value)
                reason_counts.update(reasons)
            else:
                included_positive_source.append(value)
            catalog.append(value)
            continue

        sample_type = str(row.get("sample_type"))
        if sample_type not in NEGATIVE_SAMPLE_TYPES:
            reasons = ["INVALID_NEGATIVE_SAMPLE_TYPE"]
            overlap = 0
        else:
            overlap = _negative_overlap_count(
                row,
                positives_by_group.get(_source_group_key(row), []),
                core_ids,
            )
            reasons = []
            if overlap >= overlap_minimum:
                reasons.append("NEGATIVE_OVERLAPS_CONFIRMED_HAND")
                overlap_count += 1
        review = decisions.get(crop_id)
        review_decision = str((review or {}).get("decision") or "")
        if review_decision == "CONFIRMED_NEGATIVE" and not reasons:
            action = "INCLUDE_CONFIRMED_NEGATIVE"
            evidence_reasons = ["HUMAN_CONFIRMED_NEGATIVE"]
            value = dict(row)
            value["pretrain_curation"] = _decision_metadata(
                action, evidence_reasons, source_hash, overlap, review
            )
            confirmed_negative_source.append(value)
        else:
            action = "HOLD_NEGATIVE_CANDIDATE"
            if review_decision == "CONFIRMED_NEGATIVE" and reasons:
                reasons.append("REVIEW_CONFLICTS_CONFIRMED_HAND_OVERLAP")
            elif review_decision == "FALSE_NEGATIVE_HAND_VISIBLE":
                reasons.append("HUMAN_CONFIRMED_FALSE_NEGATIVE")
            elif review_decision == "HOLD":
                reasons.append("HUMAN_REQUESTED_HOLD")
            else:
                reasons.append("UNVERIFIED_TEACHER_NEGATIVE")
            value = dict(row)
            value["pretrain_curation"] = _decision_metadata(
                action, reasons, source_hash, overlap, review
            )
            excluded.append(value)
            negative_review_queue.append(value)
            reason_counts.update(reasons)
        catalog.append(value)

    if not included_positive_source:
        raise ValueError("Curation produced no valid positive landmark records")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir: Optional[Path] = Path(
        tempfile.mkdtemp(prefix=output_dir.name + ".tmp.", dir=str(output_dir.parent))
    )
    try:
        assert temporary_dir is not None
        referenced_by_id: Dict[str, Dict[str, Any]] = {}
        image_manifest: List[Dict[str, Any]] = []
        for row in included_positive_source + confirmed_negative_source:
            value, image_entry = _reference_crop(row)
            referenced_by_id[str(row["crop_id"])] = value
            image_manifest.append(image_entry)
        review_referenced: List[Dict[str, Any]] = []
        review_image_manifest: List[Dict[str, Any]] = []
        for row in negative_review_queue:
            value, image_entry = _reference_review_crop(row)
            review_referenced.append(value)
            review_image_manifest.append(image_entry)
        negative_review_queue = review_referenced
        ending_source_hash = sha256_file(source_labels)
        if ending_source_hash != source_hash:
            raise RuntimeError(
                "Source labels changed while the curation snapshot was being built"
            )
        included_landmarks = [
            referenced_by_id[str(row["crop_id"])] for row in included_positive_source
        ]
        multitask = included_landmarks + [
            referenced_by_id[str(row["crop_id"])] for row in confirmed_negative_source
        ]
        included_landmarks.sort(key=lambda row: str(row["crop_id"]))
        multitask.sort(key=lambda row: str(row["crop_id"]))
        image_manifest.sort(key=lambda row: str(row["crop_id"]))
        review_image_manifest.sort(key=lambda row: str(row["crop_id"]))
        _verify_existing_source_snapshot(
            output_dir, image_manifest, review_image_manifest
        )
        smoke_count = int(rules.get("smoke_subset_size", 128))
        smoke = _stable_smoke_subset(
            included_landmarks,
            smoke_count,
            str(rules.get("smoke_selection_salt", "pretrain_landmark_smoke_v1")),
        )
        for row in smoke:
            original_sampling_weight = row.get("sampling_weight")
            row["sampling_weight"] = 1.0
            row["pretrain_curation"] = dict(row["pretrain_curation"])
            row["pretrain_curation"][
                "smoke_original_sampling_weight"
            ] = original_sampling_weight
            row["pretrain_curation"]["smoke_sampling_weight"] = 1.0

        labels_dir = temporary_dir / "05_labels"
        audit_dir = temporary_dir / "audit"
        qc_dir = temporary_dir / "qc"
        labels_dir.mkdir(parents=True, exist_ok=True)
        audit_dir.mkdir(parents=True, exist_ok=True)
        qc_dir.mkdir(parents=True, exist_ok=True)
        landmarks_path = labels_dir / "hand_training_labels_pretrain_landmarks.jsonl"
        multitask_path = labels_dir / "hand_training_labels_pretrain_multitask.jsonl"
        smoke_path = labels_dir / "hand_training_labels_pretrain_smoke.jsonl"
        catalog_path = audit_dir / "pretrain_curation_catalog.jsonl"
        included_path = audit_dir / "included_landmarks.jsonl"
        excluded_path = audit_dir / "excluded_and_held.jsonl"
        review_queue_path = audit_dir / "negative_review_queue.jsonl"
        image_manifest_path = audit_dir / "image_manifest.jsonl"
        review_image_manifest_path = audit_dir / "review_image_manifest.jsonl"
        write_jsonl(landmarks_path, included_landmarks)
        write_jsonl(multitask_path, multitask)
        write_jsonl(smoke_path, smoke)
        write_jsonl(catalog_path, catalog)
        write_jsonl(included_path, included_landmarks)
        write_jsonl(excluded_path, excluded)
        write_jsonl(review_queue_path, negative_review_queue)
        write_jsonl(image_manifest_path, image_manifest)
        write_jsonl(review_image_manifest_path, review_image_manifest)

        artifact_paths = [
            landmarks_path,
            multitask_path,
            smoke_path,
            catalog_path,
            included_path,
            excluded_path,
            review_queue_path,
            image_manifest_path,
            review_image_manifest_path,
        ]
        artifacts = {
            path.relative_to(temporary_dir).as_posix(): {
                "sha256": sha256_file(path),
                "size_bytes": int(path.stat().st_size),
            }
            for path in artifact_paths
        }
        image_aggregate = hashlib.sha256(
            "".join(
                "{}:{}\n".format(row["path"], row["sha256"])
                for row in image_manifest
            ).encode("utf-8")
        ).hexdigest()
        config_path_value = cfg.get("_meta", {}).get("config_path")
        config_hash = (
            sha256_file(config_path_value)
            if config_path_value and Path(str(config_path_value)).is_file()
            else None
        )
        manifest = {
            "schema_version": CURATION_SCHEMA,
            "source": {
                "labels": str(source_labels),
                "labels_sha256": source_hash,
                "crop_root": str(source_crop_root.resolve()),
                "record_count": len(records),
            },
            "negative_review_decisions": review_source,
            "negative_review_workspace": {
                "workspace_dir": str(review_paths["workspace_dir"]),
                "candidates_dir": str(review_paths["candidates_dir"]),
                "manifest_file": str(review_paths["manifest_file"]),
                "decisions_file": str(review_paths["decisions_file"]),
            },
            "config_path": str(config_path_value) if config_path_value else None,
            "config_sha256": config_hash,
            "output_dir": str(output_dir),
            "artifacts": artifacts,
            "images": {
                "count": len(image_manifest),
                "aggregate_sha256": image_aggregate,
                "manifest": "audit/image_manifest.jsonl",
                "storage": "train_sources_reference",
            },
            "review_candidates": {
                "count": len(review_image_manifest),
                "manifest": "audit/review_image_manifest.jsonl",
                "aggregate_sha256": hashlib.sha256(
                    "".join(
                        "{}:{}\n".format(row["path"], row["sha256"])
                        for row in review_image_manifest
                    ).encode("utf-8")
                ).hexdigest(),
                "storage": "train_sources_reference",
            },
        }
        manifest_path = qc_dir / "sha256_manifest.json"
        write_json(manifest_path, manifest)
        report = {
            "status": "ok",
            "schema_version": CURATION_SCHEMA,
            "source_labels": str(source_labels),
            "source_labels_sha256": source_hash,
            "source_crop_root": str(source_crop_root.resolve()),
            "negative_review_decisions": review_source,
            "negative_review_workspace": None,
            "output_dir": str(output_dir),
            "counts": {
                "source_records": len(records),
                "included_landmark_positives": len(included_landmarks),
                "included_confirmed_negatives": len(confirmed_negative_source),
                "multitask_records": len(multitask),
                "excluded_or_held": len(excluded),
                "negative_review_queue": len(negative_review_queue),
                "negative_overlap_confirmed_hand": overlap_count,
                "smoke_records": len(smoke),
            },
            "included_by_dataset": _counter_dict(
                row.get("dataset_id") for row in included_landmarks
            ),
            "included_by_sample_type": _counter_dict(
                row.get("sample_type") for row in included_landmarks
            ),
            "reason_counts": dict(sorted(reason_counts.items())),
            "negative_policy": "unverified teacher negatives are held; only non-conflicting human-confirmed negatives enter multitask",
            "artifacts": {
                "landmark_labels": str(output_dir / landmarks_path.relative_to(temporary_dir)),
                "multitask_labels": str(output_dir / multitask_path.relative_to(temporary_dir)),
                "smoke_labels": str(output_dir / smoke_path.relative_to(temporary_dir)),
                "catalog": str(output_dir / catalog_path.relative_to(temporary_dir)),
                "excluded": str(output_dir / excluded_path.relative_to(temporary_dir)),
                "negative_review_queue": str(output_dir / review_queue_path.relative_to(temporary_dir)),
                "review_image_manifest": str(output_dir / review_image_manifest_path.relative_to(temporary_dir)),
                "manifest": str(output_dir / manifest_path.relative_to(temporary_dir)),
            },
        }
        report_path = qc_dir / "curation_report.json"

        if finalize_review:
            report["negative_review_workspace"] = review_finalize_report
        else:
            report["negative_review_workspace"] = _prepare_negative_review_workspace(
                cfg,
                review_cfg,
                temporary_dir,
                source_hash,
            )
        write_json(report_path, report)

        if output_dir.exists():
            _guard_curation_output(
                output_dir, source_labels, source_crops, allow_overwrite
            )
            shutil.rmtree(str(output_dir))
        os.replace(str(temporary_dir), str(output_dir))
        temporary_dir = None
        return report
    finally:
        if temporary_dir is not None and temporary_dir.exists():
            shutil.rmtree(str(temporary_dir), ignore_errors=True)
