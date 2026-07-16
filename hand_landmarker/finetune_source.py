"""Strict ``finetune_source_v1`` and HLMF Gold-aggregate validation."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from .contracts import validate_label_record
from .finetune_selection import NEGATIVE_SAMPLE_TYPES, clean_row, identity_value
from .io_utils import read_jsonl, sha256_file


SOURCE_SCHEMA = "finetune_source_v1"
AGGREGATE_SCHEMA = "hmlf_gold_aggregate_v1"
GOLD_KINDS = {
    "external_gold",
    "reviewed_hard_gold",
    "disagreement_gold",
    "new_recorded_gold",
}
SOURCE_KINDS = GOLD_KINDS | {"pretrain_replay"}
GOLD_MODES = {"gold_only", "reviewed_gold"}


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("JSON descriptor root must be an object: {}".format(path))
    return value


def _safe_file(root: Path, relative: Any, field: str) -> Path:
    path = Path(str(relative or ""))
    if not str(path) or path.is_absolute() or ".." in path.parts:
        raise ValueError("{} must be a safe path relative to {}".format(field, root))
    candidate = root / path
    if candidate.is_symlink() or not candidate.is_file():
        raise ValueError("{} is missing, not a file, or a symlink: {}".format(field, candidate))
    resolved = candidate.resolve(strict=True)
    try:
        resolved.relative_to(root.resolve(strict=True))
    except ValueError as exc:
        raise ValueError("{} escapes source root: {}".format(field, candidate)) from exc
    # Reject symlinked parent components as well as a symlink leaf.
    current = root.resolve(strict=True)
    for part in path.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError("{} traverses a symlink: {}".format(field, current))
    return resolved


def _safe_dir(root: Path, relative: Any, field: str) -> Path:
    path = Path(str(relative or ""))
    if not str(path) or path.is_absolute() or ".." in path.parts:
        raise ValueError("{} must be a safe directory relative to {}".format(field, root))
    candidate = root / path
    if candidate.is_symlink() or not candidate.is_dir():
        raise ValueError("{} is missing, not a directory, or a symlink: {}".format(field, candidate))
    resolved = candidate.resolve(strict=True)
    try:
        resolved.relative_to(root.resolve(strict=True))
    except ValueError as exc:
        raise ValueError("{} escapes source root: {}".format(field, candidate)) from exc
    current = root.resolve(strict=True)
    for part in path.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError("{} traverses a symlink: {}".format(field, current))
    return resolved


def _artifact_file(
    source_root: Path,
    artifacts: Mapping[str, Any],
    name: str,
    count_jsonl: bool = False,
) -> Tuple[Path, Mapping[str, Any], Optional[List[Dict[str, Any]]]]:
    entry = artifacts.get(name)
    if not isinstance(entry, Mapping):
        raise ValueError("finetune source is missing artifact {}".format(name))
    path = _safe_file(source_root, entry.get("path"), "artifacts.{}.path".format(name))
    expected = str(entry.get("sha256") or "")
    if not expected or sha256_file(path) != expected:
        raise ValueError("Artifact SHA mismatch: {}".format(path))
    rows: Optional[List[Dict[str, Any]]] = None
    if count_jsonl:
        rows = [clean_row(row) for row in read_jsonl(path)]
        if "count" in entry and int(entry["count"]) != len(rows):
            raise ValueError("Artifact count mismatch: {}".format(path))
    return path, entry, rows


def _manifest_aggregate(rows: Sequence[Mapping[str, Any]]) -> str:
    entries: List[Tuple[str, str]] = []
    for row in rows:
        relative = str(row.get("path") or row.get("relative_path") or "")
        digest = str(row.get("sha256") or "")
        if not relative or not digest:
            raise ValueError("Hash manifest rows require path and sha256")
        entries.append((relative, digest))
    if len({path for path, _ in entries}) != len(entries):
        raise ValueError("Hash manifest contains duplicate paths")
    return hashlib.sha256(
        "".join("{}:{}\n".format(path, digest) for path, digest in sorted(entries)).encode("utf-8")
    ).hexdigest()


def _validate_hash_manifest(
    source_root: Path,
    entry: Mapping[str, Any],
    field: str,
    row_root: Path,
) -> List[Dict[str, Any]]:
    path = _safe_file(source_root, entry.get("sha256_manifest"), field + ".sha256_manifest")
    rows = [clean_row(row) for row in read_jsonl(path)]
    expected_manifest = str(entry.get("manifest_sha256") or "")
    if not expected_manifest or sha256_file(path) != expected_manifest:
        raise ValueError("{} manifest SHA mismatch".format(field))
    expected_aggregate = str(entry.get("aggregate_sha256") or "")
    if not expected_aggregate or _manifest_aggregate(rows) != expected_aggregate:
        raise ValueError("{} aggregate SHA mismatch".format(field))
    if "count" in entry and int(entry["count"]) != len(rows):
        raise ValueError("{} count mismatch".format(field))
    for row in rows:
        relative = str(row.get("path") or "")
        normalized = Path(relative)
        if not relative or normalized.is_absolute() or ".." in normalized.parts:
            raise ValueError("{} contains a non-normalized relative path".format(field))
        physical = _safe_file(row_root, relative, field + ".row.path")
        expected = str(row.get("sha256") or "")
        if not expected or sha256_file(physical) != expected:
            raise ValueError("{} physical file SHA mismatch: {}".format(field, physical))
    return rows


def _row_id(row: Mapping[str, Any]) -> str:
    return str(row.get("source_crop_id") or row.get("crop_id") or row.get("global_crop_id") or "")


def _validate_gold_rows(
    descriptor: Mapping[str, Any],
    manifest_rows: Sequence[Mapping[str, Any]],
    label_rows: Sequence[Mapping[str, Any]],
) -> None:
    manifest_ids = [_row_id(row) for row in manifest_rows]
    label_ids = [_row_id(row) for row in label_rows]
    if any(not value for value in manifest_ids + label_ids):
        raise ValueError("Gold manifest/labels require crop identity")
    if len(set(manifest_ids)) != len(manifest_ids) or len(set(label_ids)) != len(label_ids):
        raise ValueError("Gold manifest/labels contain duplicate crop identities")
    if set(manifest_ids) != set(label_ids):
        raise ValueError("Gold labels must fully cover the HLMF ROI manifest")
    included = 0
    ignored = 0
    handedness_policy = str(descriptor.get("handedness_policy"))
    for row in label_rows:
        if bool(row.get("ignore_for_training")):
            ignored += 1
            continue
        included += 1
        errors = validate_label_record(row, split="train")
        if errors:
            raise ValueError("Invalid Gold label {}: {}".format(_row_id(row), errors))
        if str(row.get("annotation_provenance")) != "human_gold":
            raise ValueError("Gold labels require annotation_provenance=human_gold")
        if str(row.get("supervision_tier")) != "gold":
            raise ValueError("Gold labels require supervision_tier=gold")
        handedness = str((row.get("handedness") or {}).get("label", "unknown")).lower()
        if handedness_policy == "unavailable" and handedness != "unknown":
            raise ValueError("handedness_policy=unavailable requires unknown labels")
    counts = descriptor.get("counts") or {}
    expected = {
        "manifest": len(manifest_rows),
        "gold_labels": len(label_rows),
        "included": included,
        "ignored": ignored,
    }
    for key, value in expected.items():
        if int(counts.get(key, -1)) != value:
            raise ValueError("Gold descriptor counts.{} mismatch".format(key))


def _validate_all_artifact_hashes(source_root: Path, artifacts: Mapping[str, Any]) -> None:
    """Authenticate every declared artifact, including optional audit sidecars."""

    for name, raw in artifacts.items():
        if not isinstance(raw, Mapping):
            raise ValueError("artifacts.{} must be an object".format(name))
        if "path" not in raw:
            # Directory/hash-manifest artifacts are validated by their
            # source-kind branch below.
            if name not in {"crop_images", "source_images"}:
                raise ValueError("artifacts.{} requires path".format(name))
            continue
        path = _safe_file(source_root, raw.get("path"), "artifacts.{}.path".format(name))
        expected = str(raw.get("sha256") or "")
        if not expected or sha256_file(path) != expected:
            raise ValueError("Artifact SHA mismatch: {}".format(path))
        if "count" in raw and path.suffix.lower() == ".jsonl":
            if len(read_jsonl(path)) != int(raw["count"]):
                raise ValueError("Artifact count mismatch: {}".format(path))


def _validate_replay_rows(descriptor: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> List[str]:
    root_entries = descriptor.get("external_crop_roots")
    if not isinstance(root_entries, list) or not root_entries:
        raise ValueError("Replay source requires external_crop_roots")
    roots: List[Path] = []
    for entry in root_entries:
        if not isinstance(entry, Mapping):
            raise ValueError("external_crop_roots rows must be objects")
        root = Path(str(entry.get("root") or ""))
        if not root.is_absolute() or not root.is_dir() or root.is_symlink():
            raise ValueError("Replay external root must be an existing absolute non-symlink directory")
        if entry.get("read_only") is not True or entry.get("row_image_sha256_required") is not True:
            raise ValueError("Replay external roots must be read_only and require row image SHA")
        roots.append(root.resolve(strict=True))
    resolved_roots = [str(path) for path in roots]
    seen: Set[str] = set()
    confirmed_negative = 0
    for row in rows:
        identity = identity_value(row)
        if not identity or identity in seen:
            raise ValueError("Replay identities must be present and unique")
        seen.add(identity)
        if str(row.get("supervision_tier")) != "pseudo" or str(row.get("training_stage")) != "finetune":
            raise ValueError("Replay rows require pseudo tier and finetune stage")
        path = Path(str(row.get("crop_path") or ""))
        if not path.is_absolute() or not path.is_file() or path.is_symlink():
            raise ValueError("Replay crop_path must be an existing absolute non-symlink file")
        resolved = path.resolve(strict=True)
        lexical_roots = []
        for root in roots:
            try:
                path.relative_to(root)
                lexical_roots.append(root)
            except ValueError:
                pass
        if not lexical_roots or not any(_is_within(resolved, root) for root in lexical_roots):
            raise ValueError("Replay crop is outside authenticated roots: {}".format(path))
        for root in lexical_roots:
            current = root
            for part in path.relative_to(root).parts:
                current = current / part
                if current.is_symlink():
                    raise ValueError("Replay crop traverses a symlink: {}".format(current))
        expected = str(row.get("image_sha256") or "")
        if not expected or sha256_file(resolved) != expected:
            raise ValueError("Replay row image SHA mismatch: {}".format(path))
        present = bool((row.get("hand_presence") or {}).get("present", False))
        if not present:
            if str(row.get("sample_type")) not in NEGATIVE_SAMPLE_TYPES:
                raise ValueError("Replay negative has invalid sample_type")
            curation = row.get("pretrain_curation") or {}
            review = curation.get("review") or {}
            if str(curation.get("action")) != "INCLUDE_CONFIRMED_NEGATIVE":
                raise ValueError("Replay negative lacks INCLUDE_CONFIRMED_NEGATIVE evidence")
            if str(review.get("decision")) != "CONFIRMED_NEGATIVE" or any(
                not str(review.get(key) or "").strip()
                for key in ("reviewer", "reviewed_at", "review_method", "review_image_sha256")
            ):
                raise ValueError("Replay negative has incomplete human review evidence")
            confirmed_negative += 1
    counts = descriptor.get("counts") or {}
    if int(counts.get("canonical_labels", -1)) != len(rows):
        raise ValueError("Replay canonical_labels count mismatch")
    if int(counts.get("confirmed_negative", -1)) != confirmed_negative:
        raise ValueError("Replay confirmed_negative count mismatch")
    return resolved_roots


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def validate_finetune_source(
    descriptor_path: Path,
    allowed_external_roots: Optional[Sequence[Path]] = None,
) -> Dict[str, Any]:
    """Validate one immutable source package and return its canonical rows."""

    descriptor_path = Path(descriptor_path)
    if descriptor_path.is_symlink() or not descriptor_path.is_file():
        raise ValueError("finetune source descriptor is missing or a symlink: {}".format(descriptor_path))
    source_root = descriptor_path.parent.resolve(strict=True)
    descriptor = _read_json(descriptor_path)
    required_text = (
        "source_id", "dataset_id", "source_kind", "source_mode", "producer",
        "producer_version", "created_at", "supervision_tier", "handedness_policy",
    )
    if str(descriptor.get("schema_version")) != SOURCE_SCHEMA:
        raise ValueError("Unsupported finetune source schema")
    missing = [key for key in required_text if not str(descriptor.get(key) or "").strip()]
    if missing:
        raise ValueError("finetune source is missing fields {}".format(missing))
    if list(descriptor.get("enabled_stages") or []) != ["finetune"]:
        raise ValueError("finetune source enabled_stages must be exactly ['finetune']")
    source_kind = str(descriptor["source_kind"])
    source_mode = str(descriptor["source_mode"])
    if source_kind not in SOURCE_KINDS:
        raise ValueError("Unsupported finetune source kind: {}".format(source_kind))
    artifacts = descriptor.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("finetune source artifacts must be an object")
    try:
        datetime.fromisoformat(str(descriptor["created_at"]).replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("finetune source created_at must be ISO-8601") from exc
    if str(descriptor.get("handedness_policy")) not in {"unavailable", "optional_per_row", "required"}:
        raise ValueError("Unsupported handedness_policy")
    _validate_all_artifact_hashes(source_root, artifacts)

    result: Dict[str, Any] = {
        "path": str(descriptor_path.resolve()),
        "sha256": sha256_file(descriptor_path),
        "root": str(source_root),
        "descriptor": descriptor,
        "source_id": str(descriptor["source_id"]),
        "dataset_id": str(descriptor["dataset_id"]),
        "source_kind": source_kind,
    }
    if source_kind in GOLD_KINDS:
        expected_mode = "gold_only" if source_kind == "external_gold" else "reviewed_gold"
        if source_mode != expected_mode or str(descriptor.get("supervision_tier")) != "gold":
            raise ValueError("Gold source has invalid source_mode/supervision_tier")
        if not isinstance(descriptor.get("input_sha256"), Mapping) or not descriptor.get("input_sha256"):
            raise ValueError("Gold source requires non-empty input_sha256 provenance")
        _, _, manifest_rows = _artifact_file(source_root, artifacts, "manifest", count_jsonl=True)
        _, _, gold_rows = _artifact_file(source_root, artifacts, "gold_labels", count_jsonl=True)
        _artifact_file(source_root, artifacts, "qc_report", count_jsonl=False)
        crop_entry = artifacts.get("crop_images")
        if not isinstance(crop_entry, Mapping):
            raise ValueError("Gold source requires crop_images artifact")
        _safe_dir(source_root, crop_entry.get("root"), "artifacts.crop_images.root")
        crop_hash_rows = _validate_hash_manifest(
            source_root, crop_entry, "artifacts.crop_images", source_root
        )
        if len(crop_hash_rows) != len(manifest_rows or []):
            raise ValueError("Gold crop image hash manifest must cover every ROI")
        if source_kind == "external_gold":
            source_images = artifacts.get("source_images")
            if not isinstance(source_images, Mapping):
                raise ValueError("external_gold requires source_images artifact")
            if source_images.get("read_only") is not True:
                raise ValueError(
                    "external_gold source_images must be declared read-only"
                )
            physical_root = _safe_dir(
                source_root,
                source_images.get("root"),
                "artifacts.source_images.root",
            )
            _validate_hash_manifest(
                source_root,
                source_images,
                "artifacts.source_images",
                physical_root,
            )
        assert manifest_rows is not None and gold_rows is not None
        _validate_gold_rows(descriptor, manifest_rows, gold_rows)
        ignored_entry = artifacts.get("ignored_sidecar")
        if ignored_entry is not None:
            _, _, ignored_rows = _artifact_file(
                source_root, artifacts, "ignored_sidecar", count_jsonl=True
            )
            ignored_ids = {_row_id(row) for row in (ignored_rows or [])}
            expected_ignored = {_row_id(row) for row in gold_rows if bool(row.get("ignore_for_training"))}
            if ignored_ids != expected_ignored:
                raise ValueError("ignored_sidecar must be derived exactly from Gold labels")
        result["rows"] = gold_rows
        result["manifest_rows"] = manifest_rows
    else:
        if source_mode != "canonical_replay_index" or str(descriptor.get("supervision_tier")) != "pseudo":
            raise ValueError("Replay source has invalid source_mode/supervision_tier")
        if not str(descriptor.get("parent_pretrain_id") or ""):
            raise ValueError("Replay source requires parent_pretrain_id")
        _, _, replay_rows = _artifact_file(source_root, artifacts, "canonical_labels", count_jsonl=True)
        _artifact_file(source_root, artifacts, "parent_curation_manifest", count_jsonl=False)
        _artifact_file(source_root, artifacts, "qc_report", count_jsonl=False)
        assert replay_rows is not None
        resolved_roots = _validate_replay_rows(descriptor, replay_rows)
        allowed = [Path(path).resolve(strict=True) for path in (allowed_external_roots or [])]
        if allowed and any(not any(_is_within(Path(root), parent) for parent in allowed) for root in resolved_roots):
            raise ValueError("Replay descriptor references a crop root outside allowed_crop_roots")
        result["rows"] = replay_rows
        result["external_crop_roots"] = resolved_roots
    return result


def validate_source_set(sources: Sequence[Mapping[str, Any]]) -> None:
    for field in ("source_id", "dataset_id"):
        values = [str(source[field]) for source in sources]
        duplicates = sorted(value for value, count in Counter(values).items() if count > 1)
        if duplicates:
            raise ValueError("Duplicate {} values across finetune sources: {}".format(field, duplicates))


def validate_gold_aggregate(
    descriptor_path: Path,
    finetune_root: Path,
    gold_sources: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Authenticate the one HLMF Gold aggregate consumed by HLML."""

    descriptor_path = Path(descriptor_path)
    finetune_root = Path(finetune_root).resolve(strict=True)
    if descriptor_path.is_symlink() or not descriptor_path.is_file():
        raise ValueError("Gold aggregate descriptor is missing or a symlink")
    aggregate_root = descriptor_path.parent.resolve(strict=True)
    try:
        aggregate_root.relative_to(finetune_root)
    except ValueError as exc:
        raise ValueError("Gold aggregate is outside finetune root") from exc
    descriptor = _read_json(descriptor_path)
    if str(descriptor.get("schema_version")) != AGGREGATE_SCHEMA:
        raise ValueError("Unsupported Gold aggregate schema")
    if not str(descriptor.get("finetune_id") or ""):
        raise ValueError("Gold aggregate requires finetune_id")
    artifacts = descriptor.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("Gold aggregate artifacts must be an object")
    artifact_rows: Dict[str, Any] = {}
    for name in ("catalog", "included", "excluded"):
        _, _, rows = _artifact_file(aggregate_root, artifacts, name, count_jsonl=True)
        artifact_rows[name] = rows or []
    _artifact_file(aggregate_root, artifacts, "report", count_jsonl=False)
    counts = descriptor.get("counts") or {}
    for name in ("catalog", "included", "excluded"):
        if int(counts.get(name, -1)) != len(artifact_rows[name]):
            raise ValueError("Gold aggregate counts.{} mismatch".format(name))
    if int(descriptor.get("conflict_count", -1)) != 0:
        raise ValueError("Gold aggregate conflict_count must be exactly 0")
    if int(descriptor.get("duplicate_count", -1)) < 0:
        raise ValueError("Gold aggregate duplicate_count must be non-negative")

    indexed_partitions: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for name in ("catalog", "included", "excluded"):
        index: Dict[str, Dict[str, Any]] = {}
        for row in artifact_rows[name]:
            identity = str(row.get("global_crop_id") or "")
            if not identity or identity in index:
                raise ValueError("Gold aggregate {} global IDs must be present and unique".format(name))
            index[identity] = row
        indexed_partitions[name] = index
    included_ids = set(indexed_partitions["included"])
    excluded_ids = set(indexed_partitions["excluded"])
    catalog_ids = set(indexed_partitions["catalog"])
    if included_ids & excluded_ids:
        raise ValueError("Gold aggregate included/excluded partitions overlap")
    if included_ids | excluded_ids != catalog_ids:
        raise ValueError("Gold aggregate included/excluded union must equal catalog")
    for name in ("included", "excluded"):
        for identity, row in indexed_partitions[name].items():
            if row != indexed_partitions["catalog"][identity]:
                raise ValueError("Gold aggregate partition row differs from catalog: {}".format(identity))

    configured = {
        str(source["source_id"]): (str(source["path"]), str(source["sha256"]))
        for source in gold_sources
    }
    listed: Dict[str, Tuple[str, str]] = {}
    for item in descriptor.get("source_descriptors") or []:
        if not isinstance(item, Mapping):
            raise ValueError("Gold aggregate source_descriptors rows must be objects")
        source_id = str(item.get("source_id") or "")
        relative = Path(str(item.get("path") or ""))
        if not source_id or relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Invalid aggregate source descriptor reference")
        if source_id in listed:
            raise ValueError("Gold aggregate contains duplicate source_id: {}".format(source_id))
        path = (finetune_root / relative)
        if path.is_symlink() or not path.is_file():
            raise ValueError("Aggregate source descriptor is missing or a symlink: {}".format(path))
        digest = str(item.get("sha256") or "")
        if not digest or sha256_file(path) != digest:
            raise ValueError("Aggregate source descriptor SHA mismatch: {}".format(path))
        listed[source_id] = (str(path.resolve()), digest)
    if listed != configured:
        raise ValueError("Gold aggregate source descriptor set does not match discovered Gold sources")
    return {
        "path": str(descriptor_path.resolve()),
        "sha256": sha256_file(descriptor_path),
        "descriptor": descriptor,
        "catalog": artifact_rows["catalog"],
        "included": artifact_rows["included"],
        "excluded": artifact_rows["excluded"],
        "artifact_paths": {
            name: str(_safe_file(aggregate_root, artifacts[name]["path"], "aggregate." + name))
            for name in ("catalog", "included", "excluded", "report")
        },
    }
