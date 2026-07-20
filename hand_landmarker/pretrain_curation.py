"""Persistent, auditable curation for pseudo-labelled pretrain data.

Teacher abstentions are candidates, not verified background. This module
persists curated labels and image hashes while keeping canonical ROI paths in
the read-only dataset warehouse. Every rejected or unverified row remains in an on-disk
audit catalog instead of being filtered only in training memory.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import tempfile
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

from .config import load_config, resolve_path
from .io_utils import (
    VALID_IMAGE_EXTENSIONS,
    read_jsonl,
    sha256_file,
    write_json,
    write_jsonl,
)


CURATION_SCHEMA = "pretrain_curation_v1"
POSITIVE_SAMPLE_TYPES = {"POS_RUNTIME", "POS_LOW_PALM"}
NEGATIVE_SAMPLE_TYPES = {"NEG_RUNTIME_CANDIDATE", "NEG_LOW_PALM_CANDIDATE"}
REVIEW_DECISIONS = {"CONFIRMED_NEGATIVE", "FALSE_NEGATIVE_HAND_VISIBLE", "HOLD"}
REVIEW_TRANSACTION_SCHEMA = "negative_review_transaction_v1"
REMOVED_MANIFEST_SCHEMA = "negative_removed_manifest_v1"
QUARANTINE_MANIFEST_SCHEMA = "negative_quarantine_manifest_v1"
TEACHER_HOLDOUT_SCHEMA = "pretrain_teacher_holdout_v1"


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
    review_transaction = manifest.get("negative_review_transaction")
    verified_review_transaction = None
    if review_transaction is not None:
        if not isinstance(review_transaction, Mapping):
            raise error_type("Curation negative_review_transaction must be an object")
        report_value = review_transaction.get("report")
        if not str(report_value or "").strip():
            raise error_type("Curation negative_review_transaction has no report")
        report_path = Path(str(report_value))
        if not report_path.is_file():
            raise error_type(
                "Negative review transaction report does not exist: {}".format(report_path)
            )
        try:
            with report_path.open("r", encoding="utf-8") as handle:
                transaction_report = json.load(handle)
        except (OSError, ValueError) as exc:
            raise error_type(
                "Could not read negative review transaction report {}: {}".format(
                    report_path, exc
                )
            ) from exc
        if (
            not isinstance(transaction_report, Mapping)
            or transaction_report.get("schema_version") != REVIEW_TRANSACTION_SCHEMA
            or transaction_report.get("status") != "committed"
        ):
            raise error_type(
                "Negative review transaction is not committed: {}".format(report_path)
            )
        if transaction_report.get("candidate_cleanup") not in {
            "complete",
            "skipped_by_config",
        }:
            raise error_type(
                "Negative review transaction cleanup is incomplete: {}".format(report_path)
            )
        if str(transaction_report.get("transaction_id") or "") != str(
            review_transaction.get("transaction_id") or ""
        ):
            raise error_type("Negative review transaction id mismatch")
        if str(transaction_report.get("source_labels_sha256") or "") != str(
            (manifest.get("source") or {}).get("labels_sha256") or ""
        ):
            raise error_type("Negative review transaction source-label hash mismatch")
        report_partitions = dict(transaction_report.get("partitions") or {})
        manifest_partitions = dict(review_transaction.get("partitions") or {})
        if report_partitions != manifest_partitions:
            raise error_type("Negative review transaction partition counts changed")
        try:
            expected_count = int(report_partitions["expected"])
            reviewed_count = int(report_partitions["reviewed"])
            admitted_count = int(report_partitions["admitted"])
            quarantine_count = int(report_partitions["quarantine"])
            removed_count = int(report_partitions["removed"])
        except (KeyError, TypeError, ValueError) as exc:
            raise error_type("Negative review transaction partition counts are invalid") from exc
        if (
            reviewed_count != admitted_count + quarantine_count
            or expected_count != admitted_count + quarantine_count + removed_count
        ):
            raise error_type("Negative review transaction partition conservation failed")
        authenticated_artifacts = dict(review_transaction.get("artifacts") or {})
        required_review_artifacts = {
            "review_manifest",
            "decisions",
            "removed_manifest",
            "quarantine_manifest",
            "review_report",
        }
        if set(authenticated_artifacts) != required_review_artifacts:
            raise error_type(
                "Negative review transaction must authenticate exactly {}".format(
                    sorted(required_review_artifacts)
                )
            )
        for name, expected in authenticated_artifacts.items():
            if not isinstance(expected, Mapping):
                raise error_type(
                    "Negative review artifact metadata must be an object: {}".format(name)
                )
            path = Path(str(expected.get("path") or ""))
            digest = str(expected.get("sha256") or "")
            if not path.is_file() or not digest:
                raise error_type(
                    "Negative review artifact is missing: {} ({})".format(name, path)
                )
            actual = sha256_file(path)
            if actual != digest:
                raise error_type(
                    "Negative review artifact hash mismatch for {}: expected {}, got {}".format(
                        path, digest, actual
                    )
                )
        verified_review_transaction = {
            "path": str(report_path),
            "transaction_id": transaction_report.get("transaction_id"),
            "status": transaction_report.get("status"),
            "candidate_cleanup": transaction_report.get("candidate_cleanup"),
            "partitions": transaction_report.get("partitions"),
            "artifact_count": len(authenticated_artifacts),
        }
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
        "negative_review_transaction": verified_review_transaction,
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


def _recover_curation_backup(output_dir: Path, source_labels_sha256: str) -> None:
    """Recover the single internal backup left by an interrupted directory swap."""

    if not output_dir.parent.is_dir():
        return
    backups = sorted(output_dir.parent.glob(".{}.review-backup.*".format(output_dir.name)))
    if not backups:
        return
    if len(backups) != 1 or backups[0].is_symlink() or not backups[0].is_dir():
        raise RuntimeError("Ambiguous interrupted curation backups: {}".format(backups))
    backup = backups[0]

    def authenticated_snapshot(path: Path) -> bool:
        manifest_path = path / "qc" / "sha256_manifest.json"
        if not manifest_path.is_file():
            return False
        try:
            with manifest_path.open("r", encoding="utf-8") as handle:
                manifest = json.load(handle)
        except (OSError, ValueError):
            return False
        return bool(
            isinstance(manifest, Mapping)
            and manifest.get("schema_version") == CURATION_SCHEMA
            and Path(str(manifest.get("output_dir") or "")).resolve()
            == output_dir.resolve()
            and str((manifest.get("source") or {}).get("labels_sha256") or "")
            == source_labels_sha256
        )

    if not authenticated_snapshot(backup):
        raise RuntimeError("Interrupted curation backup is not authenticated: {}".format(backup))
    if not output_dir.exists():
        os.replace(str(backup), str(output_dir))
        return
    if not authenticated_snapshot(output_dir):
        raise RuntimeError(
            "Both curation output and backup exist, but output is not authenticated"
        )
    shutil.rmtree(str(backup))


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


def _safe_relative_path(value: Any, field: str) -> Path:
    text = str(value or "")
    path = Path(text)
    if (
        "\\" in text
        or (len(text) >= 2 and text[1] == ":")
        or path.is_absolute()
        or not path.parts
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError("{} must be a safe relative path".format(field))
    return path


def _negative_review_paths(
    config: Mapping[str, Any], review: Mapping[str, Any]
) -> Dict[str, Path]:
    decisions_value = review.get("decisions_file")
    if not str(decisions_value or "").strip():
        raise ValueError("review.decisions_file is required")
    decisions_file = resolve_path(str(decisions_value), config)
    workspace_dir = decisions_file.parent
    relative_values = {
        "candidates_dir": _safe_relative_path(
            review.get("candidates_subdir") or "negative_candidates",
            "review.candidates_subdir",
        ),
        "reviewed_dir": _safe_relative_path(
            review.get("reviewed_subdir") or "negative_reviewed",
            "review.reviewed_subdir",
        ),
        "removed_dir": _safe_relative_path(
            review.get("removed_subdir") or "negative_removed",
            "review.removed_subdir",
        ),
        "quarantine_dir": _safe_relative_path(
            review.get("quarantine_subdir") or "negative_quarantine",
            "review.quarantine_subdir",
        ),
        "removed_manifest_file": _safe_relative_path(
            review.get("removed_manifest_file") or "negative_removed_manifest.jsonl",
            "review.removed_manifest_file",
        ),
        "quarantine_manifest_file": _safe_relative_path(
            review.get("quarantine_manifest_file")
            or "negative_quarantine_manifest.jsonl",
            "review.quarantine_manifest_file",
        ),
    }
    normalized = [value.as_posix().casefold() for value in relative_values.values()]
    if len(normalized) != len(set(normalized)):
        raise ValueError("Negative review directory and manifest paths must be distinct")
    for left_name, left in relative_values.items():
        for right_name, right in relative_values.items():
            if left_name == right_name:
                continue
            if left in right.parents:
                raise ValueError(
                    "Negative review paths must not contain one another: {} and {}".format(
                        left, right
                    )
                )
    paths = {
        "workspace_dir": workspace_dir,
        "decisions_file": decisions_file,
        "manifest_file": workspace_dir / "review_manifest.jsonl",
        "report_file": workspace_dir / "review_report.json",
        "instructions_file": workspace_dir / "REVIEW_INSTRUCTIONS.md",
        "transaction_file": workspace_dir / "review_transaction.json",
        "lock_file": workspace_dir / ".review-finalize.lock",
    }
    paths.update(
        {name: workspace_dir / relative for name, relative in relative_values.items()}
    )
    return paths


def _acquire_review_lock(path: Path):
    """Acquire an OS-backed, single-process lock without leaving stale lock state."""

    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+b")
    try:
        if path.stat().st_size == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (OSError, IOError) as exc:
        handle.close()
        raise RuntimeError(
            "Another negative-review finalization is already running: {}".format(path)
        ) from exc
    return handle


def _release_review_lock(handle) -> None:
    if handle is None:
        return
    try:
        handle.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    finally:
        handle.close()


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
        reviewed_subdir = paths["reviewed_dir"].relative_to(workspace)
        (temporary / reviewed_subdir).mkdir(parents=True, exist_ok=True)
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
                    "candidate_image_sha256": actual_hash,
                    "source_labels_sha256": source_labels_sha256,
                    "automatic_reasons": list(
                        (row.get("pretrain_curation") or {}).get("reasons") or []
                    ),
                    "source_crop_path": str(source.resolve()),
                    "source_image_sha256": actual_hash,
                }
            )
        manifest_rows.sort(key=lambda row: str(row["crop_id"]))
        write_jsonl(temporary / paths["manifest_file"].name, manifest_rows)
        instructions = (
            "# Negative candidate visual review\n\n"
            "Copy `negative_candidates/` to `negative_reviewed/` while preserving every "
            "relative path. Review only `negative_reviewed/`: delete every image that "
            "contains a hand, fingers, wrist, or uncertain hand-like content, and retain "
            "only clear background. Do not add, rename, move, or edit images. Keep the "
            "original `negative_candidates/` complete. When review finishes, run "
            "`make pretrain-curate-reviewed`.\n"
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
            "reviewed_dir": str(paths["reviewed_dir"]),
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


def _scan_review_images(root: Path, field: str) -> Dict[str, Path]:
    if root.is_symlink():
        raise ValueError("{} must not be a symbolic link: {}".format(field, root))
    if not root.is_dir():
        raise FileNotFoundError("{} directory does not exist: {}".format(field, root))
    discovered: Dict[str, Path] = {}
    casefolded = set()
    for path in sorted(root.rglob("*"), key=lambda value: str(value).lower()):
        if path.is_symlink():
            raise ValueError("{} must not contain symbolic links: {}".format(field, path))
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if path.suffix.lower() not in VALID_IMAGE_EXTENSIONS:
            raise ValueError(
                "{} contains an unsupported file or archive: {}".format(field, relative)
            )
        folded = relative.casefold()
        if folded in casefolded:
            raise ValueError("{} contains a duplicate path: {}".format(field, relative))
        casefolded.add(folded)
        discovered[relative] = path
    return discovered


def _link_or_copy_verified(source: Path, destination: Path, expected_sha256: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(str(source), str(destination))
    except OSError:
        shutil.copy2(str(source), str(destination))
    actual = sha256_file(destination)
    if actual != expected_sha256:
        raise RuntimeError(
            "Materialized review partition image hash mismatch: {}".format(destination)
        )


def _partition_manifest_row(
    row: Mapping[str, Any],
    source_row: Mapping[str, Any],
    schema_version: str,
    partition: str,
    reason: str,
    source_labels_sha256: str,
) -> Dict[str, Any]:
    candidate_sha256 = str(
        row.get("candidate_image_sha256") or row.get("sha256") or ""
    )
    source_sha256 = str(row.get("source_image_sha256") or candidate_sha256)
    return {
        "schema_version": schema_version,
        "crop_id": str(row.get("crop_id")),
        "dataset_id": str(source_row.get("dataset_id") or row.get("dataset_id") or ""),
        "source_crop_id": str(source_row.get("source_crop_id") or ""),
        "sample_type": str(source_row.get("sample_type") or row.get("sample_type") or ""),
        "candidate_relative_path": str(row.get("candidate_relative_path")),
        "source_crop_path": str(Path(str(source_row.get("crop_path"))).resolve()),
        "candidate_image_sha256": candidate_sha256,
        "source_image_sha256": source_sha256,
        "source_labels_sha256": source_labels_sha256,
        "partition": partition,
        "partition_reason": reason,
        "review_decision_basis": reason,
    }


def _verify_partition_directory(directory: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    actual = _scan_review_images(directory, directory.name)
    expected = {
        str(row.get("candidate_relative_path")): str(
            row.get("candidate_image_sha256") or ""
        )
        for row in rows
    }
    if set(actual) != set(expected):
        raise RuntimeError(
            "Review partition membership mismatch for {}".format(directory)
        )
    for relative, path in actual.items():
        if sha256_file(path) != expected[relative]:
            raise RuntimeError("Review partition image hash mismatch: {}".format(path))


def _commit_review_transaction_artifacts(transaction: Mapping[str, Any]) -> None:
    artifacts = dict(transaction.get("artifacts") or {})
    for name in ("decisions", "removed_manifest", "quarantine_manifest"):
        artifact = dict(artifacts.get(name) or {})
        target = Path(str(artifact.get("path") or ""))
        staged = Path(str(artifact.get("staged_path") or ""))
        expected_sha256 = str(artifact.get("sha256") or "")
        if target.is_file():
            if sha256_file(target) != expected_sha256:
                raise RuntimeError(
                    "Existing review transaction artifact changed: {}".format(target)
                )
            continue
        if not staged.is_file():
            raise RuntimeError(
                "Review transaction cannot recover missing staged artifact: {}".format(staged)
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(str(staged), str(target))
        if sha256_file(target) != expected_sha256:
            raise RuntimeError("Committed review artifact hash mismatch: {}".format(target))

    for name, manifest_name in (
        ("removed_dir", "removed_manifest"),
        ("quarantine_dir", "quarantine_manifest"),
    ):
        artifact = dict(artifacts.get(name) or {})
        target = Path(str(artifact.get("path") or ""))
        staged = Path(str(artifact.get("staged_path") or ""))
        manifest_path = Path(str((artifacts.get(manifest_name) or {}).get("path") or ""))
        rows = read_jsonl(manifest_path)
        if target.is_dir():
            _verify_partition_directory(target, rows)
            continue
        if target.exists():
            raise RuntimeError("Review partition target is not a directory: {}".format(target))
        if not staged.is_dir():
            raise RuntimeError(
                "Review transaction cannot recover missing staged directory: {}".format(staged)
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(str(staged), str(target))
        _verify_partition_directory(target, rows)


def _review_transaction_report(
    transaction: Mapping[str, Any], paths: Mapping[str, Path]
) -> Dict[str, Any]:
    partitions = dict(transaction.get("partitions") or {})
    artifacts = dict(transaction.get("artifacts") or {})
    return {
        "status": str(transaction.get("status")),
        "schema_version": REVIEW_TRANSACTION_SCHEMA,
        "transaction_id": str(transaction.get("transaction_id")),
        "review_method": "delete_hand_images_retain_background",
        "reviewer": str(transaction.get("reviewer")),
        "reviewed_at": str(transaction.get("reviewed_at")),
        "workspace_dir": str(paths["workspace_dir"]),
        "candidates_dir": str(paths["candidates_dir"]),
        "reviewed_dir": str(paths["reviewed_dir"]),
        "removed_dir": str(paths["removed_dir"]),
        "quarantine_dir": str(paths["quarantine_dir"]),
        "manifest_file": str(paths["manifest_file"]),
        "decisions_file": str(paths["decisions_file"]),
        "removed_manifest_file": str(paths["removed_manifest_file"]),
        "quarantine_manifest_file": str(paths["quarantine_manifest_file"]),
        "transaction_file": str(paths["transaction_file"]),
        "original_candidate_count": int(partitions.get("expected", 0)),
        "retained_confirmed_count": int(partitions.get("reviewed", 0)),
        "admitted_confirmed_count": int(partitions.get("admitted", 0)),
        "quarantine_count": int(partitions.get("quarantine", 0)),
        "removed_count": int(partitions.get("removed", 0)),
        "deleted_or_rejected_count": int(partitions.get("removed", 0)),
        "candidate_cleanup": str(transaction.get("candidate_cleanup")),
        "decisions_sha256": str((artifacts.get("decisions") or {}).get("sha256")),
        "source_labels_sha256": str(transaction.get("source_labels_sha256")),
    }


def _validate_review_transaction(
    transaction: Mapping[str, Any],
    paths: Mapping[str, Path],
    source_labels_sha256: str,
) -> None:
    if transaction.get("schema_version") != REVIEW_TRANSACTION_SCHEMA:
        raise ValueError("Unsupported negative review transaction schema")
    if str(transaction.get("source_labels_sha256") or "") != source_labels_sha256:
        raise ValueError("Negative review transaction belongs to a different source-label snapshot")
    if transaction.get("status") not in {"prepared", "committed"}:
        raise ValueError("Negative review transaction has invalid status")
    review_manifest = dict(transaction.get("review_manifest") or {})
    if Path(str(review_manifest.get("path") or "")).resolve() != paths[
        "manifest_file"
    ].resolve():
        raise ValueError("Negative review transaction manifest path mismatch")
    if (
        not paths["manifest_file"].is_file()
        or sha256_file(paths["manifest_file"])
        != str(review_manifest.get("sha256") or "")
    ):
        raise ValueError("Negative review manifest changed after transaction preparation")
    partitions = dict(transaction.get("partitions") or {})
    try:
        expected_count = int(partitions["expected"])
        reviewed_count = int(partitions["reviewed"])
        admitted_count = int(partitions["admitted"])
        quarantine_count = int(partitions["quarantine"])
        removed_count = int(partitions["removed"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Negative review transaction partition counts are invalid") from exc
    if (
        reviewed_count != admitted_count + quarantine_count
        or expected_count != admitted_count + quarantine_count + removed_count
    ):
        raise ValueError("Negative review transaction partition conservation failed")
    expected_targets = {
        "decisions": paths["decisions_file"],
        "removed_manifest": paths["removed_manifest_file"],
        "quarantine_manifest": paths["quarantine_manifest_file"],
        "removed_dir": paths["removed_dir"],
        "quarantine_dir": paths["quarantine_dir"],
    }
    artifacts = dict(transaction.get("artifacts") or {})
    staging_dir = Path(str(transaction.get("staging_dir") or ""))
    if (
        not staging_dir.name.startswith(".review-finalize.")
        or not staging_dir.name.endswith(".tmp")
        or not _is_within(staging_dir, paths["workspace_dir"])
        or staging_dir.resolve() == paths["workspace_dir"].resolve()
    ):
        raise ValueError("Negative review transaction staging path is invalid")
    for name, expected_path in expected_targets.items():
        artifact = dict(artifacts.get(name) or {})
        actual = Path(str(artifact.get("path") or ""))
        if actual.resolve() != expected_path.resolve():
            raise ValueError("Negative review transaction artifact path mismatch: {}".format(name))
        staged_path = Path(str(artifact.get("staged_path") or ""))
        if not _is_within(staged_path, staging_dir):
            raise ValueError(
                "Negative review transaction staged artifact escapes staging: {}".format(name)
            )


def _begin_retained_negative_review_transaction(
    config: Mapping[str, Any],
    review: Mapping[str, Any],
    source_labels_sha256: str,
    rows_by_id: Mapping[str, Mapping[str, Any]],
    review_candidate_ids: Sequence[str],
    quarantine_ids: Sequence[str],
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
    paths = _negative_review_paths(config, review)
    if not paths["workspace_dir"].is_dir():
        raise FileNotFoundError(
            "Negative review workspace does not exist: {}".format(paths["workspace_dir"])
        )
    if not paths["manifest_file"].is_file():
        raise FileNotFoundError(
            "Negative review manifest does not exist: {}".format(paths["manifest_file"])
        )

    if paths["transaction_file"].is_file():
        with paths["transaction_file"].open("r", encoding="utf-8") as handle:
            transaction = json.load(handle)
        if not isinstance(transaction, Mapping):
            raise ValueError("Negative review transaction root must be an object")
        transaction = dict(transaction)
        _validate_review_transaction(transaction, paths, source_labels_sha256)
        _commit_review_transaction_artifacts(transaction)
        decisions = _load_review_decisions(paths["decisions_file"], config)
        persisted_quarantine_ids = {
            str(row.get("crop_id"))
            for row in read_jsonl(paths["quarantine_manifest_file"])
        }
        expected_quarantine_ids = set(decisions) & set(quarantine_ids)
        if persisted_quarantine_ids != expected_quarantine_ids:
            raise RuntimeError(
                "Current overlap safety policy disagrees with the committed quarantine partition"
            )
        if bool(transaction.get("retain_reviewed_evidence", True)):
            reviewed = _scan_review_images(paths["reviewed_dir"], "negative_reviewed")
            expected_reviewed = {
                str(row.get("candidate_relative_path")): str(row.get("review_image_sha256"))
                for row in decisions.values()
            }
            if set(reviewed) != set(expected_reviewed):
                raise RuntimeError("negative_reviewed membership changed after finalization")
            for relative, path in reviewed.items():
                if sha256_file(path) != expected_reviewed[relative]:
                    raise RuntimeError("negative_reviewed image changed: {}".format(path))
        report = _review_transaction_report(transaction, paths)
        write_json(paths["report_file"], report)
        return decisions, report, transaction

    for orphan in paths["workspace_dir"].glob(".review-finalize.*.tmp"):
        if orphan.is_dir() and not orphan.is_symlink():
            shutil.rmtree(str(orphan), ignore_errors=True)
    protected_targets = [
        paths["decisions_file"],
        paths["removed_manifest_file"],
        paths["quarantine_manifest_file"],
        paths["removed_dir"],
        paths["quarantine_dir"],
    ]
    existing_targets = [str(path) for path in protected_targets if path.exists()]
    if existing_targets:
        raise ValueError(
            "Review artifacts exist without a transaction report: {}".format(existing_targets)
        )
    if not paths["candidates_dir"].is_dir():
        raise FileNotFoundError(
            "The complete negative_candidates directory is required for first finalization: {}".format(
                paths["candidates_dir"]
            )
        )

    manifest_rows = [_clean_row(row) for row in read_jsonl(paths["manifest_file"])]
    expected: Dict[str, Dict[str, Any]] = {}
    seen_crop_ids = set()
    seen_relative_paths = set()
    for row in manifest_rows:
        relative = str(row.get("candidate_relative_path") or "")
        relative_path = _safe_relative_path(relative, "candidate_relative_path")
        if relative != relative_path.as_posix():
            raise ValueError("candidate_relative_path must use normalized forward slashes")
        folded = relative.casefold()
        if folded in seen_relative_paths:
            raise ValueError("Duplicate negative review candidate path: {}".format(relative))
        seen_relative_paths.add(folded)
        crop_id = str(row.get("crop_id") or "")
        if not crop_id or crop_id in seen_crop_ids:
            raise ValueError("Every review manifest row requires a unique crop_id")
        seen_crop_ids.add(crop_id)
        if str(row.get("source_labels_sha256") or "") != source_labels_sha256:
            raise ValueError(
                "Negative review workspace belongs to a different source-label snapshot"
            )
        source_row = rows_by_id.get(crop_id)
        if source_row is None:
            raise ValueError("Review manifest references unknown crop_id: {}".format(crop_id))
        if (
            bool((source_row.get("hand_presence") or {}).get("present", False))
            or str(source_row.get("sample_type")) not in NEGATIVE_SAMPLE_TYPES
        ):
            raise ValueError("Review manifest references a non-negative candidate: {}".format(crop_id))
        source_path = Path(str(source_row.get("crop_path") or ""))
        recorded_source = Path(str(row.get("source_crop_path") or ""))
        if source_path.resolve() != recorded_source.resolve():
            raise ValueError("Review manifest source path mismatch for {}".format(crop_id))
        expected_sha256 = str(
            row.get("candidate_image_sha256") or row.get("sha256") or ""
        )
        source_sha256 = sha256_file(source_path)
        if not expected_sha256 or source_sha256 != expected_sha256:
            raise ValueError("Review manifest/source image hash mismatch for {}".format(crop_id))
        recorded_source_sha256 = str(row.get("source_image_sha256") or expected_sha256)
        if recorded_source_sha256 != source_sha256:
            raise ValueError("Review manifest source image hash mismatch for {}".format(crop_id))
        row["candidate_image_sha256"] = expected_sha256
        row["source_image_sha256"] = source_sha256
        expected[relative] = row

    expected_review_ids = {str(crop_id) for crop_id in review_candidate_ids}
    if seen_crop_ids != expected_review_ids:
        raise ValueError("Review manifest does not exactly cover source negative candidates")

    candidates = _scan_review_images(paths["candidates_dir"], "negative_candidates")
    if set(candidates) != set(expected):
        raise ValueError("negative_candidates no longer matches the complete review manifest")
    for relative, path in candidates.items():
        if sha256_file(path) != str(expected[relative]["candidate_image_sha256"]):
            raise ValueError("Original negative candidate was modified: {}".format(path))

    reviewed = _scan_review_images(paths["reviewed_dir"], "negative_reviewed")
    unknown = sorted(set(reviewed) - set(expected))
    if unknown:
        raise ValueError(
            "negative_reviewed contains images not present in review_manifest.jsonl: {}".format(
                unknown[:10]
            )
        )
    for relative, path in reviewed.items():
        expected_sha256 = str(expected[relative]["candidate_image_sha256"])
        if sha256_file(path) != expected_sha256:
            raise ValueError("Reviewed negative image was modified: {}".format(path))

    quarantine_set = set(quarantine_ids)
    reviewed_crop_ids = {str(expected[relative]["crop_id"]) for relative in reviewed}
    quarantine_crop_ids = reviewed_crop_ids & quarantine_set
    admitted_crop_ids = reviewed_crop_ids - quarantine_crop_ids
    removed_relatives = sorted(set(expected) - set(reviewed))
    quarantine_relatives = sorted(
        relative
        for relative in reviewed
        if str(expected[relative]["crop_id"]) in quarantine_crop_ids
    )
    if (
        len(expected)
        != len(removed_relatives) + len(quarantine_relatives) + len(admitted_crop_ids)
    ):
        raise RuntimeError("Negative review partition conservation failed")

    reviewer = str(review.get("reviewer") or "").strip()
    if not reviewer:
        raise ValueError("review.reviewer is required")
    reviewed_at = datetime.now(timezone.utc).astimezone().isoformat()
    decisions_rows: List[Dict[str, Any]] = []
    for relative in sorted(reviewed):
        row = expected[relative]
        decisions_rows.append(
            {
                "crop_id": str(row.get("crop_id")),
                "candidate_relative_path": relative,
                "decision": "CONFIRMED_NEGATIVE",
                "reviewer": reviewer,
                "reviewed_at": reviewed_at,
                "review_method": "retained_after_visual_deletion_review",
                "review_image_path": str(reviewed[relative].resolve()),
                "review_image_sha256": str(row["candidate_image_sha256"]),
            }
        )

    removed_rows = [
        _partition_manifest_row(
            expected[relative],
            rows_by_id[str(expected[relative]["crop_id"])],
            REMOVED_MANIFEST_SCHEMA,
            "negative_removed",
            "absent_from_negative_reviewed",
            source_labels_sha256,
        )
        for relative in removed_relatives
    ]
    quarantine_rows = [
        _partition_manifest_row(
            expected[relative],
            rows_by_id[str(expected[relative]["crop_id"])],
            QUARANTINE_MANIFEST_SCHEMA,
            "negative_quarantine",
            "confirmed_hand_overlap_conflict",
            source_labels_sha256,
        )
        for relative in quarantine_relatives
    ]

    staging = Path(
        tempfile.mkdtemp(
            prefix=".review-finalize.", suffix=".tmp", dir=str(paths["workspace_dir"])
        )
    )
    transaction_published = False
    try:
        staged_targets = {
            name: staging / "artifacts" / path.relative_to(paths["workspace_dir"])
            for name, path in {
                "decisions": paths["decisions_file"],
                "removed_manifest": paths["removed_manifest_file"],
                "quarantine_manifest": paths["quarantine_manifest_file"],
                "removed_dir": paths["removed_dir"],
                "quarantine_dir": paths["quarantine_dir"],
            }.items()
        }
        write_jsonl(staged_targets["decisions"], decisions_rows)
        write_jsonl(staged_targets["removed_manifest"], removed_rows)
        write_jsonl(staged_targets["quarantine_manifest"], quarantine_rows)
        staged_targets["removed_dir"].mkdir(parents=True, exist_ok=True)
        staged_targets["quarantine_dir"].mkdir(parents=True, exist_ok=True)
        for row in removed_rows:
            relative = str(row["candidate_relative_path"])
            _link_or_copy_verified(
                candidates[relative],
                staged_targets["removed_dir"] / Path(relative),
                str(row["candidate_image_sha256"]),
            )
        for row in quarantine_rows:
            relative = str(row["candidate_relative_path"])
            _link_or_copy_verified(
                reviewed[relative],
                staged_targets["quarantine_dir"] / Path(relative),
                str(row["candidate_image_sha256"]),
            )

        artifact_targets = {
            "decisions": paths["decisions_file"],
            "removed_manifest": paths["removed_manifest_file"],
            "quarantine_manifest": paths["quarantine_manifest_file"],
            "removed_dir": paths["removed_dir"],
            "quarantine_dir": paths["quarantine_dir"],
        }
        artifacts: Dict[str, Dict[str, Any]] = {}
        for name, target in artifact_targets.items():
            value = {
                "path": str(target),
                "staged_path": str(staged_targets[name]),
                "kind": "directory" if name.endswith("_dir") else "file",
            }
            if value["kind"] == "file":
                value["sha256"] = sha256_file(staged_targets[name])
            artifacts[name] = value
        transaction = {
            "schema_version": REVIEW_TRANSACTION_SCHEMA,
            "transaction_id": str(uuid.uuid4()),
            "status": "prepared",
            "candidate_cleanup": "pending",
            "source_labels_sha256": source_labels_sha256,
            "reviewer": reviewer,
            "reviewed_at": reviewed_at,
            "review_manifest": {
                "path": str(paths["manifest_file"]),
                "sha256": sha256_file(paths["manifest_file"]),
                "count": len(expected),
            },
            "partitions": {
                "expected": len(expected),
                "reviewed": len(reviewed),
                "admitted": len(admitted_crop_ids),
                "quarantine": len(quarantine_rows),
                "removed": len(removed_rows),
            },
            "artifacts": artifacts,
            "staging_dir": str(staging),
            "cleanup_candidates_after_success": bool(
                review.get("cleanup_candidates_after_success", True)
            ),
            "retain_reviewed_evidence": bool(review.get("retain_reviewed_evidence", True)),
        }
        staged_transaction = staging / paths["transaction_file"].name
        write_json(staged_transaction, transaction)
        os.replace(str(staged_transaction), str(paths["transaction_file"]))
        transaction_published = True
        _commit_review_transaction_artifacts(transaction)
        decisions = _load_review_decisions(paths["decisions_file"], config)
        report = _review_transaction_report(transaction, paths)
        write_json(paths["report_file"], report)
        return decisions, report, transaction
    finally:
        if not transaction_published and staging.exists():
            shutil.rmtree(str(staging), ignore_errors=True)


def _complete_retained_negative_review_transaction(
    transaction: Mapping[str, Any],
    paths: Mapping[str, Path],
    output_dir: Path,
) -> Dict[str, Any]:
    manifest_path = output_dir / "qc" / "sha256_manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError("Cannot commit review transaction without curated manifest")
    value = dict(transaction)
    value["status"] = "committed"
    value["committed_at"] = datetime.now(timezone.utc).astimezone().isoformat()
    value["curated_snapshot"] = {
        "output_dir": str(output_dir),
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
    }
    write_json(paths["transaction_file"], value)

    if bool(value.get("cleanup_candidates_after_success", True)):
        if paths["candidates_dir"].exists():
            if paths["candidates_dir"].is_symlink() or not paths["candidates_dir"].is_dir():
                raise RuntimeError(
                    "Refusing to clean non-directory negative_candidates: {}".format(
                        paths["candidates_dir"]
                    )
                )
            shutil.rmtree(str(paths["candidates_dir"]))
        value["candidate_cleanup"] = "complete"
    else:
        value["candidate_cleanup"] = "skipped_by_config"
    if not bool(value.get("retain_reviewed_evidence", True)) and paths["reviewed_dir"].exists():
        shutil.rmtree(str(paths["reviewed_dir"]))
    write_json(paths["transaction_file"], value)
    staging = Path(str(value.get("staging_dir") or ""))
    if staging.is_dir() and _is_within(staging, paths["workspace_dir"]):
        shutil.rmtree(str(staging), ignore_errors=True)
    report = _review_transaction_report(value, paths)
    write_json(paths["report_file"], report)
    return report


def _finalize_review_authentication(output_dir: Path, paths: Mapping[str, Path]) -> None:
    """Add final immutable reports to the curated manifest after commit."""

    manifest_path = output_dir / "qc" / "sha256_manifest.json"
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    transaction = dict(manifest.get("negative_review_transaction") or {})
    authenticated = dict(transaction.get("artifacts") or {})
    authenticated["review_report"] = {
        "path": str(paths["report_file"]),
        "sha256": sha256_file(paths["report_file"]),
    }
    transaction["artifacts"] = authenticated
    manifest["negative_review_transaction"] = transaction
    curation_report = output_dir / "qc" / "curation_report.json"
    manifest_artifacts = dict(manifest.get("artifacts") or {})
    manifest_artifacts["qc/curation_report.json"] = {
        "sha256": sha256_file(curation_report),
        "size_bytes": int(curation_report.stat().st_size),
    }
    manifest["artifacts"] = manifest_artifacts
    write_json(manifest_path, manifest)

    with paths["transaction_file"].open("r", encoding="utf-8") as handle:
        transaction_report = json.load(handle)
    transaction_report["curated_snapshot"] = {
        "output_dir": str(output_dir),
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
    }
    write_json(paths["transaction_file"], transaction_report)


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
            "image_storage": "external_source_reference",
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
        "storage": "external_source_reference",
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
            "review_image_storage": "external_source_reference",
            "review_image_sha256": digest_value,
        }
    )
    return value, {
        "crop_id": str(row.get("crop_id")),
        "path": str(source),
        "sha256": digest_value,
        "size_bytes": int(source.stat().st_size),
        "storage": "external_source_reference",
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


def _select_teacher_holdout_datasets(
    valid_positive_rows: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> Tuple[set, Dict[str, Any]]:
    """Select whole datasets for a deterministic teacher-labelled holdout.

    Dataset IDs are the isolation unit because every row from a selected ID,
    including teacher abstentions, must stay out of all pretrain stages.  A
    bounded subset-sum search chooses a combination close to the configured
    target without splitting a source or requiring an operator to pick files.
    """

    holdout = dict(config or {})
    enabled = bool(holdout.get("enabled", False))
    base_report: Dict[str, Any] = {
        "schema_version": TEACHER_HOLDOUT_SCHEMA,
        "enabled": enabled,
        "selection_unit": "dataset_id",
    }
    if not enabled:
        return set(), dict(base_report, status="disabled", selected_dataset_ids=[])
    if str(holdout.get("selection_unit", "dataset_id")) != "dataset_id":
        raise ValueError("curation.teacher_holdout.selection_unit must be dataset_id")

    minimum = int(holdout.get("minimum_positive_records", 5000))
    target = int(holdout.get("target_positive_records", 8192))
    maximum = int(holdout.get("maximum_positive_records", 10000))
    if minimum <= 0 or not minimum <= target <= maximum:
        raise ValueError(
            "curation.teacher_holdout requires 0 < minimum <= target <= maximum"
        )
    salt = str(holdout.get("selection_salt", "pretrain_teacher_holdout_v1"))
    pattern_text = str(holdout.get("eligible_dataset_pattern", ".*"))
    try:
        pattern = re.compile(pattern_text)
    except re.error as exc:
        raise ValueError(
            "curation.teacher_holdout.eligible_dataset_pattern is invalid: {}".format(exc)
        ) from exc

    counts = Counter(str(row.get("dataset_id") or "") for row in valid_positive_rows)
    if "" in counts:
        raise ValueError("Teacher holdout requires dataset_id on every valid positive")
    eligible = {
        dataset_id: int(count)
        for dataset_id, count in counts.items()
        if pattern.fullmatch(dataset_id) and int(count) <= maximum
    }
    if not eligible:
        raise ValueError(
            "Teacher holdout has no eligible dataset matching {!r} with at most {} positives".format(
                pattern_text, maximum
            )
        )

    ordered = sorted(
        eligible,
        key=lambda dataset_id: hashlib.sha256(
            (salt + "\0" + dataset_id).encode("utf-8")
        ).hexdigest(),
    )
    combinations: Dict[int, Tuple[str, ...]] = {0: ()}
    for dataset_id in ordered:
        count = eligible[dataset_id]
        for total, selected in sorted(list(combinations.items()), reverse=True):
            candidate_total = total + count
            if candidate_total > maximum or candidate_total in combinations:
                continue
            combinations[candidate_total] = selected + (dataset_id,)
    feasible = [total for total in combinations if minimum <= total <= maximum]
    if not feasible:
        raise ValueError(
            "No whole-dataset teacher holdout can satisfy [{}, {}]; eligible positive counts: {}".format(
                minimum, maximum, dict(sorted(eligible.items()))
            )
        )

    def candidate_key(total: int) -> Tuple[Any, ...]:
        selected = combinations[total]
        digest = hashlib.sha256(
            (salt + "\0" + "\0".join(sorted(selected))).encode("utf-8")
        ).hexdigest()
        return abs(total - target), 0 if total >= target else 1, digest

    selected_total = min(feasible, key=candidate_key)
    selected_ids = set(combinations[selected_total])
    remaining = len(valid_positive_rows) - selected_total
    if remaining <= 0:
        raise ValueError("Teacher holdout selection would remove every valid positive")
    report = dict(
        base_report,
        status="ok",
        eligible_dataset_pattern=pattern_text,
        selection_salt=salt,
        minimum_positive_records=minimum,
        target_positive_records=target,
        maximum_positive_records=maximum,
        selected_positive_records=int(selected_total),
        remaining_training_positive_records=int(remaining),
        selected_dataset_ids=sorted(selected_ids),
        selected_positive_records_by_dataset={
            dataset_id: eligible[dataset_id] for dataset_id in sorted(selected_ids)
        },
        eligible_positive_records_by_dataset=dict(sorted(eligible.items())),
        all_valid_positive_records_by_dataset=dict(sorted(counts.items())),
    )
    return selected_ids, report


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
    _recover_curation_backup(output_dir, source_hash)
    _guard_curation_output(
        output_dir, source_labels, source_crops, allow_overwrite
    )

    rows_by_id = {str(row["crop_id"]): row for row in records}
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
    positive_reasons_by_id: Dict[str, List[str]] = {}
    for row in records:
        if bool((row.get("hand_presence") or {}).get("present", False)):
            positives_by_group[_source_group_key(row)].append(row)
            positive_reasons_by_id[str(row["crop_id"])] = _positive_reasons(
                row, allowed_tiers, minimum_coordinate, maximum_coordinate
            )
    if not any(not reasons for reasons in positive_reasons_by_id.values()):
        raise ValueError("Curation produced no valid positive landmark records")
    valid_positive_rows = [
        row
        for row in records
        if bool((row.get("hand_presence") or {}).get("present", False))
        and not positive_reasons_by_id[str(row["crop_id"])]
    ]
    holdout_dataset_ids, teacher_holdout_report = _select_teacher_holdout_datasets(
        valid_positive_rows,
        dict(rules.get("teacher_holdout") or {}),
    )

    overlap_by_id: Dict[str, int] = {}
    for row in records:
        if bool((row.get("hand_presence") or {}).get("present", False)):
            continue
        if str(row.get("sample_type")) not in NEGATIVE_SAMPLE_TYPES:
            continue
        overlap_by_id[str(row["crop_id"])] = _negative_overlap_count(
            row,
            positives_by_group.get(_source_group_key(row), []),
            core_ids,
        )

    review_finalize_report: Optional[Dict[str, Any]] = None
    review_transaction: Optional[Dict[str, Any]] = None
    review_lock = None
    if finalize_review:
        review_lock = _acquire_review_lock(review_paths["lock_file"])
        try:
            review_candidate_ids = [
                crop_id
                for crop_id, row in rows_by_id.items()
                if not bool((row.get("hand_presence") or {}).get("present", False))
                and str(row.get("sample_type")) in NEGATIVE_SAMPLE_TYPES
                and str(row.get("dataset_id") or "") not in holdout_dataset_ids
            ]
            decisions, review_finalize_report, review_transaction = (
                _begin_retained_negative_review_transaction(
                    cfg,
                    review_cfg,
                    source_hash,
                    rows_by_id,
                    review_candidate_ids,
                    [
                        crop_id
                        for crop_id, overlap in overlap_by_id.items()
                        if overlap >= overlap_minimum
                    ],
                )
            )
        except Exception:
            _release_review_lock(review_lock)
            raise
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

    catalog: List[Dict[str, Any]] = []
    included_positive_source: List[Dict[str, Any]] = []
    teacher_holdout_source: List[Dict[str, Any]] = []
    confirmed_negative_source: List[Dict[str, Any]] = []
    excluded: List[Dict[str, Any]] = []
    negative_review_queue: List[Dict[str, Any]] = []
    reason_counts: Counter = Counter()
    overlap_count = 0
    for row in records:
        present = bool((row.get("hand_presence") or {}).get("present", False))
        crop_id = str(row["crop_id"])
        if str(row.get("dataset_id") or "") in holdout_dataset_ids:
            positive_reasons = list(positive_reasons_by_id.get(crop_id, [])) if present else []
            reasons = ["AUTOMATIC_TEACHER_HOLDOUT"] + positive_reasons
            if present and not positive_reasons:
                action = "HOLDOUT_TEACHER_EVAL"
            elif present:
                action = "QUARANTINE_HOLDOUT_POSITIVE"
            else:
                action = "EXCLUDE_HOLDOUT_DATASET"
            value = dict(row)
            value["pretrain_curation"] = _decision_metadata(action, reasons, source_hash)
            if action == "HOLDOUT_TEACHER_EVAL":
                teacher_holdout_source.append(value)
            excluded.append(value)
            catalog.append(value)
            reason_counts.update(reasons)
            continue
        if present:
            reasons = list(positive_reasons_by_id[crop_id])
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
            overlap = overlap_by_id.get(crop_id, 0)
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
        _release_review_lock(review_lock)
        raise ValueError("Curation produced no valid positive landmark records")

    try:
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        temporary_dir: Optional[Path] = Path(
            tempfile.mkdtemp(prefix=output_dir.name + ".tmp.", dir=str(output_dir.parent))
        )
    except Exception:
        _release_review_lock(review_lock)
        raise
    try:
        assert temporary_dir is not None
        referenced_by_id: Dict[str, Dict[str, Any]] = {}
        image_manifest: List[Dict[str, Any]] = []
        for row in included_positive_source + confirmed_negative_source:
            value, image_entry = _reference_crop(row)
            image_entry["partition"] = "train"
            referenced_by_id[str(row["crop_id"])] = value
            image_manifest.append(image_entry)
        teacher_holdout: List[Dict[str, Any]] = []
        for row in teacher_holdout_source:
            value, image_entry = _reference_crop(row)
            image_entry["partition"] = "teacher_holdout"
            teacher_holdout.append(value)
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
        teacher_holdout.sort(key=lambda row: str(row["crop_id"]))
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
        holdout_path = labels_dir / "hand_teacher_holdout_labels.jsonl"
        holdout_report_path = qc_dir / "teacher_holdout_report.json"
        catalog_path = audit_dir / "pretrain_curation_catalog.jsonl"
        included_path = audit_dir / "included_landmarks.jsonl"
        excluded_path = audit_dir / "excluded_and_held.jsonl"
        review_queue_path = audit_dir / "negative_review_queue.jsonl"
        image_manifest_path = audit_dir / "image_manifest.jsonl"
        review_image_manifest_path = audit_dir / "review_image_manifest.jsonl"
        write_jsonl(landmarks_path, included_landmarks)
        write_jsonl(multitask_path, multitask)
        write_jsonl(smoke_path, smoke)
        write_jsonl(holdout_path, teacher_holdout)
        write_json(holdout_report_path, teacher_holdout_report)
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
            holdout_path,
            holdout_report_path,
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
        authenticated_review_transaction = None
        if review_transaction is not None:
            transaction_artifacts = dict(review_transaction.get("artifacts") or {})
            authenticated_artifacts = {
                "review_manifest": {
                    "path": str((review_transaction.get("review_manifest") or {}).get("path")),
                    "sha256": str(
                        (review_transaction.get("review_manifest") or {}).get("sha256")
                    ),
                }
            }
            for name in ("decisions", "removed_manifest", "quarantine_manifest"):
                authenticated_artifacts[name] = {
                    "path": str((transaction_artifacts.get(name) or {}).get("path")),
                    "sha256": str((transaction_artifacts.get(name) or {}).get("sha256")),
                }
            authenticated_review_transaction = {
                "report": str(review_paths["transaction_file"]),
                "transaction_id": str(review_transaction.get("transaction_id")),
                "partitions": dict(review_transaction.get("partitions") or {}),
                "artifacts": authenticated_artifacts,
            }
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
                "reviewed_dir": str(review_paths["reviewed_dir"]),
                "removed_dir": str(review_paths["removed_dir"]),
                "quarantine_dir": str(review_paths["quarantine_dir"]),
                "manifest_file": str(review_paths["manifest_file"]),
                "decisions_file": str(review_paths["decisions_file"]),
            },
            "negative_review_transaction": authenticated_review_transaction,
            "teacher_holdout": {
                "enabled": bool(teacher_holdout_report.get("enabled")),
                "labels": holdout_path.relative_to(temporary_dir).as_posix(),
                "positive_records": len(teacher_holdout),
                "selected_dataset_ids": list(
                    teacher_holdout_report.get("selected_dataset_ids") or []
                ),
                "report": holdout_report_path.relative_to(temporary_dir).as_posix(),
            },
            "config_path": str(config_path_value) if config_path_value else None,
            "config_sha256": config_hash,
            "output_dir": str(output_dir),
            "artifacts": artifacts,
            "images": {
                "count": len(image_manifest),
                "aggregate_sha256": image_aggregate,
                "manifest": "audit/image_manifest.jsonl",
                "storage": "external_source_reference",
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
                "storage": "external_source_reference",
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
                "teacher_holdout_positives": len(teacher_holdout),
                "teacher_holdout_datasets": len(holdout_dataset_ids),
                "included_confirmed_negatives": len(confirmed_negative_source),
                "multitask_records": len(multitask),
                "excluded_or_held": len(excluded),
                "negative_review_queue": len(negative_review_queue),
                "negative_overlap_confirmed_hand": overlap_count,
                "smoke_records": len(smoke),
                "negative_review_expected": int(
                    ((review_transaction or {}).get("partitions") or {}).get("expected", 0)
                ),
                "negative_reviewed": int(
                    ((review_transaction or {}).get("partitions") or {}).get("reviewed", 0)
                ),
                "negative_admitted": int(
                    ((review_transaction or {}).get("partitions") or {}).get("admitted", 0)
                ),
                "negative_removed": int(
                    ((review_transaction or {}).get("partitions") or {}).get("removed", 0)
                ),
                "negative_quarantine": int(
                    ((review_transaction or {}).get("partitions") or {}).get("quarantine", 0)
                ),
            },
            "included_by_dataset": _counter_dict(
                row.get("dataset_id") for row in included_landmarks
            ),
            "included_by_sample_type": _counter_dict(
                row.get("sample_type") for row in included_landmarks
            ),
            "teacher_holdout": teacher_holdout_report,
            "reason_counts": dict(sorted(reason_counts.items())),
            "negative_policy": "unverified teacher negatives are held; only non-conflicting human-confirmed negatives enter multitask",
            "artifacts": {
                "landmark_labels": str(output_dir / landmarks_path.relative_to(temporary_dir)),
                "multitask_labels": str(output_dir / multitask_path.relative_to(temporary_dir)),
                "smoke_labels": str(output_dir / smoke_path.relative_to(temporary_dir)),
                "teacher_holdout_labels": str(
                    output_dir / holdout_path.relative_to(temporary_dir)
                ),
                "teacher_holdout_report": str(
                    output_dir / holdout_report_path.relative_to(temporary_dir)
                ),
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

        backup_dir: Optional[Path] = None
        if output_dir.exists():
            _guard_curation_output(output_dir, source_labels, source_crops, allow_overwrite)
            backup_dir = output_dir.parent / (
                ".{}.review-backup.{}".format(output_dir.name, uuid.uuid4().hex)
            )
            os.replace(str(output_dir), str(backup_dir))
        try:
            os.replace(str(temporary_dir), str(output_dir))
            temporary_dir = None
        except Exception:
            if backup_dir is not None and backup_dir.exists() and not output_dir.exists():
                os.replace(str(backup_dir), str(output_dir))
            raise
        if backup_dir is not None and backup_dir.exists():
            shutil.rmtree(str(backup_dir), ignore_errors=True)
        if finalize_review:
            assert review_transaction is not None
            report["negative_review_workspace"] = (
                _complete_retained_negative_review_transaction(
                    review_transaction, review_paths, output_dir
                )
            )
            write_json(output_dir / "qc" / "curation_report.json", report)
            _finalize_review_authentication(output_dir, review_paths)
        return report
    finally:
        if temporary_dir is not None and temporary_dir.exists():
            shutil.rmtree(str(temporary_dir), ignore_errors=True)
        _release_review_lock(review_lock)
