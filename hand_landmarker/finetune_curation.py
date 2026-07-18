"""Merge HLMF Gold with one authenticated replay source for finetuning."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple, Union

from .config import load_config, resolve_path
from .contracts import effective_head_weights, validate_label_record
from .finetune_selection import (
    NEGATIVE_SAMPLE_TYPES,
    POSITIVE_SAMPLE_TYPES,
    clean_row,
    identity_value,
    stable_hex,
)
from .finetune_source import (
    GOLD_KINDS,
    validate_finetune_source,
    validate_gold_aggregate,
    validate_source_set,
)
from .io_utils import read_jsonl, sha256_file, write_json, write_jsonl


CURATION_SCHEMA = "finetune_curation_v1"
GOLD_SELECTION_SCHEMA = "hlml_gold_selection_v1"
GOLD_DOMAIN_BY_KIND = {
    "external_gold": "dragon",
    "reviewed_hard_gold": "negative_removed_gold",
    "disagreement_gold": "disagreement_gold",
    "new_recorded_gold": "new_recorded_gold",
}


def _config_mapping(config: Union[Mapping[str, Any], str, Path]) -> Dict[str, Any]:
    return load_config(config) if isinstance(config, (str, Path)) else dict(config)


def _canonical_id(row: Mapping[str, Any]) -> str:
    return str(row.get("global_crop_id") or row.get("crop_id") or identity_value(row))


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("JSON root must be an object: {}".format(path))
    return value


def _within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _resolved_allowed_roots(config: Mapping[str, Any]) -> List[Path]:
    roots: List[Path] = []
    for value in config.get("allowed_crop_roots") or []:
        root = resolve_path(str(value), config)
        if root.is_symlink() or not root.is_dir():
            raise ValueError("allowed_crop_roots entry is missing or a symlink: {}".format(root))
        roots.append(root.resolve(strict=True))
    if not roots:
        raise ValueError("curate_finetune requires at least one allowed_crop_roots entry")
    return roots


def _check_row_crop(row: Mapping[str, Any], allowed_roots: Sequence[Path]) -> Tuple[str, str]:
    path = Path(str(row.get("crop_path") or ""))
    if not path.is_absolute() or not path.is_file() or path.is_symlink():
        raise ValueError("Canonical crop_path must be an existing absolute non-symlink file")
    lexical_roots = []
    for root in allowed_roots:
        try:
            path.relative_to(root)
            lexical_roots.append(root)
        except ValueError:
            pass
    if not lexical_roots:
        raise ValueError("Canonical crop_path is outside allowed roots: {}".format(path))
    for root in lexical_roots:
        current = root
        for part in path.relative_to(root).parts:
            current = current / part
            if current.is_symlink():
                raise ValueError("Canonical crop_path traverses a symlink: {}".format(current))
    resolved = path.resolve(strict=True)
    if not any(_within(resolved, root) for root in lexical_roots):
        raise ValueError("Canonical crop_path resolves outside allowed roots: {}".format(path))
    expected = str(row.get("image_sha256") or row.get("crop_image_sha256") or row.get("roi_image_sha256") or "")
    actual = sha256_file(resolved)
    if not expected or expected != actual:
        raise ValueError("Canonical row image SHA mismatch: {}".format(path))
    return str(resolved), actual


def _identity_tokens(row: Mapping[str, Any]) -> Set[Tuple[str, str]]:
    tokens: Set[Tuple[str, str]] = set()
    for key in ("parent_global_crop_id", "global_crop_id"):
        value = str(row.get(key) or "")
        if value:
            tokens.add(("global_crop_lineage", value))
    dataset_id = str(row.get("parent_dataset_id") or row.get("dataset_id") or "")
    source_crop_id = str(row.get("parent_source_crop_id") or row.get("source_crop_id") or "")
    if dataset_id and source_crop_id:
        tokens.add(("dataset_source_crop", dataset_id + "\x1f" + source_crop_id))
    for key in ("roi_image_sha256", "image_sha256", "crop_image_sha256", "normalized_pixel_sha256"):
        value = str(row.get(key) or "")
        if value:
            canonical = "roi_image_sha256" if key in {"roi_image_sha256", "image_sha256", "crop_image_sha256"} else key
            tokens.add((canonical, value))
    return tokens


def _leakage_tokens(row: Mapping[str, Any]) -> Set[Tuple[str, str]]:
    tokens = _identity_tokens(row)
    for key in ("source_group_id", "source_session_id", "session_id"):
        value = str(row.get(key) or "")
        if value:
            tokens.add((key, value))
    return tokens


def _assert_unique_tokens(rows: Sequence[Mapping[str, Any]], label: str) -> None:
    owners: Dict[Tuple[str, str], str] = {}
    for row in rows:
        identity = _canonical_id(row)
        if not identity:
            raise ValueError("{} row has no identity".format(label))
        for token in _identity_tokens(row):
            previous = owners.get(token)
            if previous is not None and previous != identity:
                raise ValueError(
                    "{} contains unresolved duplicate identity {} shared by {} and {}".format(
                        label, token, previous, identity
                    )
                )
            owners[token] = identity


def _discover_descriptors(root: Path) -> List[Path]:
    if not root.exists():
        return []
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Source descriptor root is not a normal directory: {}".format(root))
    paths = sorted(root.rglob("finetune_source.json"), key=lambda path: str(path))
    for path in paths:
        if path.is_symlink():
            raise ValueError("Source descriptor discovery does not accept symlinks")
        current = root.resolve(strict=True)
        for part in path.relative_to(root).parts:
            current = current / part
            if current.is_symlink():
                raise ValueError("Source descriptor discovery traverses a symlink: {}".format(current))
    return paths


def _gold_repository_entry(
    root: Path, path: Path, source: Mapping[str, Any]
) -> Dict[str, str]:
    relative = path.resolve().relative_to(root.resolve())
    expected = Path(
        GOLD_DOMAIN_BY_KIND[str(source["source_kind"])]
    ) / str(source["source_id"]) / "published" / "finetune_source.json"
    if relative != expected:
        raise ValueError(
            "Gold source is outside its canonical domain/batch: {} (expected {})".format(
                path, expected
            )
        )
    return {
        "domain": expected.parts[0],
        "descriptor": expected.as_posix(),
    }


def _read_gold_selection(path: Path) -> Dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("Gold selection manifest is missing or a symlink: {}".format(path))
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to read Gold selection manifests") from exc
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle) or {}
    if not isinstance(value, dict):
        raise ValueError("Gold selection manifest root must be an object")
    return value


def _role_configuration(config: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    roles: Dict[str, Dict[str, Any]] = {}
    for role, raw in dict(config.get("sources") or {}).items():
        if not isinstance(raw, Mapping):
            raise ValueError("sources.{} must be an object".format(role))
        value = dict(raw)
        kind = str(value.get("discover_kind") or "")
        if not kind:
            raise ValueError("sources.{}.discover_kind is required".format(role))
        roles[str(role)] = value
    return roles


def _gold_source_selection(
    config: Mapping[str, Any], sources: Sequence[Mapping[str, Any]]
) -> Tuple[Dict[str, bool], List[Dict[str, Any]], Dict[str, Any]]:
    raw = config.get("source_selection") or {}
    if not isinstance(raw, Mapping):
        raise ValueError("source_selection must be an object")
    unknown_options = sorted(set(raw) - {"manifest"})
    if unknown_options:
        raise ValueError(
            "Unknown source_selection options: {}".format(unknown_options)
        )
    manifest_value = str(raw.get("manifest") or "")
    if not manifest_value:
        raise ValueError("source_selection.manifest is required")
    manifest_path = resolve_path(manifest_value, config)
    manifest = _read_gold_selection(manifest_path)
    if str(manifest.get("schema_version")) != GOLD_SELECTION_SCHEMA:
        raise ValueError("Unsupported Gold selection manifest schema")
    if str(manifest.get("finetune_id") or "") != str(config.get("finetune_id") or ""):
        raise ValueError("Gold selection finetune_id does not match configuration")
    gold_root = resolve_path(str(config.get("gold_source_descriptor_root") or ""), config)
    if Path(str(manifest.get("gold_repository_root") or "")).resolve() != gold_root.resolve():
        raise ValueError("Gold selection repository root does not match configuration")
    decisions = manifest.get("sources") or {}
    if not isinstance(decisions, Mapping):
        raise ValueError("Gold selection sources must be an object")
    discovered = {str(source["source_id"]): source for source in sources}
    if set(str(key) for key in decisions) != set(discovered):
        missing = sorted(set(discovered) - set(str(key) for key in decisions))
        unknown = sorted(set(str(key) for key in decisions) - set(discovered))
        raise ValueError(
            "Gold selection must decide every published batch: missing={} unknown={}".format(
                missing, unknown
            )
        )
    selected: Dict[str, bool] = {}
    for source_id, source in discovered.items():
        decision = decisions[source_id]
        if not isinstance(decision, Mapping):
            raise ValueError("Gold selection {} must be an object".format(source_id))
        unknown_fields = sorted(
            set(decision) - {"enabled", "source_kind", "domain", "descriptor", "descriptor_sha256"}
        )
        if unknown_fields:
            raise ValueError(
                "Gold selection {} has unknown fields {}".format(source_id, unknown_fields)
            )
        if not isinstance(decision.get("enabled"), bool):
            raise ValueError("Gold selection {}.enabled must be boolean".format(source_id))
        expected_entry = _gold_repository_entry(
            gold_root, Path(str(source["path"])), source
        )
        expected = {
            "source_kind": str(source["source_kind"]),
            "domain": expected_entry["domain"],
            "descriptor": expected_entry["descriptor"],
            "descriptor_sha256": str(source["sha256"]),
        }
        observed = {key: str(decision.get(key) or "") for key in expected}
        if observed != expected:
            raise ValueError(
                "Gold selection provenance mismatch for {}".format(source_id)
            )
        selected[source_id] = bool(decision["enabled"])
    report = [
        {
            "source_id": str(source["source_id"]),
            "dataset_id": str(source["dataset_id"]),
            "source_kind": str(source["source_kind"]),
            "enabled_by_source": selected[str(source["source_id"])],
            "selection_origin": "explicit_manifest",
        }
        for source in sorted(sources, key=lambda value: str(value["source_id"]))
    ]
    return selected, report, {
        "path": str(manifest_path.resolve()),
        "sha256": sha256_file(manifest_path),
        "finetune_id": str(manifest["finetune_id"]),
        "source_count": len(decisions),
    }


def prepare_gold_selection_from_config(
    config: Union[Mapping[str, Any], str, Path],
    enabled_source_ids: Sequence[str],
) -> Dict[str, Any]:
    """Freeze one explicit per-batch Gold decision manifest for a finetune run."""

    cfg = _config_mapping(config)
    if str(cfg.get("task")) != "curate_finetune":
        raise ValueError("Finetune curation config task must be curate_finetune")
    gold_root = resolve_path(str(cfg.get("gold_source_descriptor_root") or ""), cfg)
    selection_cfg = cfg.get("source_selection") or {}
    manifest_path = resolve_path(str(selection_cfg.get("manifest") or ""), cfg)
    if manifest_path.exists():
        raise FileExistsError(
            "Gold selection is immutable and already exists: {}".format(manifest_path)
        )
    descriptor_paths = _discover_descriptors(gold_root)
    if not descriptor_paths:
        raise ValueError("Gold repository has no published source descriptors")
    sources = [validate_finetune_source(path) for path in descriptor_paths]
    if any(str(source["source_kind"]) not in GOLD_KINDS for source in sources):
        raise ValueError("Gold repository contains a non-Gold source")
    validate_source_set(sources)
    discovered = {str(source["source_id"]): source for source in sources}
    requested = {str(value).strip() for value in enabled_source_ids if str(value).strip()}
    unknown = sorted(requested - set(discovered))
    if unknown:
        raise ValueError("Cannot enable undiscovered Gold source IDs: {}".format(unknown))
    decisions: Dict[str, Any] = {}
    for source_id, source in sorted(discovered.items()):
        entry = _gold_repository_entry(gold_root, Path(str(source["path"])), source)
        decisions[source_id] = {
            "enabled": source_id in requested,
            "source_kind": str(source["source_kind"]),
            "domain": entry["domain"],
            "descriptor": entry["descriptor"],
            "descriptor_sha256": str(source["sha256"]),
        }
    manifest = {
        "schema_version": GOLD_SELECTION_SCHEMA,
        "finetune_id": str(cfg.get("finetune_id") or ""),
        "gold_repository_root": str(gold_root.resolve()),
        "sources": decisions,
    }
    if not manifest["finetune_id"]:
        raise ValueError("finetune_id is required")
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to write Gold selection manifests") from exc
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=".gold_selection.", suffix=".yaml", dir=str(manifest_path.parent)
    )
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(
            yaml.safe_dump(manifest, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        os.replace(str(temporary), str(manifest_path))
    finally:
        if temporary.exists():
            temporary.unlink()
    return {
        "status": "ok",
        "path": str(manifest_path.resolve()),
        "sha256": sha256_file(manifest_path),
        "source_count": len(decisions),
        "enabled_source_ids": sorted(requested),
        "disabled_source_ids": sorted(set(discovered) - requested),
    }


def _load_sources(config: Mapping[str, Any], allowed_roots: Sequence[Path]) -> Dict[str, Any]:
    gold_root = resolve_path(str(config.get("gold_source_descriptor_root") or ""), config)
    replay_root = resolve_path(str(config.get("replay_source_descriptor_root") or ""), config)
    roles = _role_configuration(config)
    kind_to_role: Dict[str, str] = {}
    for role, role_cfg in roles.items():
        kind = str(role_cfg["discover_kind"])
        if kind in kind_to_role:
            raise ValueError("Each source kind may be configured by only one role")
        kind_to_role[kind] = role

    gold: List[Dict[str, Any]] = []
    gold_all: List[Dict[str, Any]] = []
    role_status: Dict[str, Dict[str, Any]] = {}
    gold_paths = _discover_descriptors(gold_root)
    # A malformed descriptor under the enabled discovery root is always fatal:
    # it must not become an accidentally ignored source package.
    for path in gold_paths:
        source = validate_finetune_source(path)
        if source["source_kind"] not in GOLD_KINDS:
            raise ValueError("Non-Gold source found in Gold repository: {}".format(path))
        role = kind_to_role.get(str(source["source_kind"]))
        if role is None:
            raise ValueError("Discovered Gold kind has no configured role: {}".format(source["source_kind"]))
        source.update(_gold_repository_entry(gold_root, path, source))
        source["role"] = role
        gold_all.append(source)
    source_enabled, source_selection, selection_manifest = _gold_source_selection(
        config, gold_all
    )
    selection_by_id = {row["source_id"]: row for row in source_selection}
    for source in gold_all:
        role_enabled = roles[str(source["role"])].get("enabled", "auto") is not False
        enabled = role_enabled and source_enabled[str(source["source_id"])]
        selection_by_id[str(source["source_id"])].update(
            {
                "role": str(source["role"]),
                "enabled_by_role": role_enabled,
                "enabled_for_training": enabled,
                "reason": (
                    "enabled"
                    if enabled
                    else "role_disabled"
                    if not role_enabled
                    else "source_disabled"
                ),
            }
        )
        if enabled:
            gold.append(source)
    for role, role_cfg in roles.items():
        if str(role_cfg["discover_kind"]) == "pretrain_replay":
            continue
        present = [source for source in gold if source.get("role") == role]
        enabled = role_cfg.get("enabled", "auto")
        required = bool(role_cfg.get("required", False))
        if enabled is False:
            status = "disabled"
        elif not present:
            status = "absent_optional"
            if required:
                raise ValueError("Required Gold source role is absent: {}".format(role))
        else:
            status = "present_valid"
        all_for_role = [source for source in gold_all if source.get("role") == role]
        role_status[role] = {
            "status": status,
            "source_count": len(present),
            "discovered_source_count": len(all_for_role),
            "disabled_source_count": len(all_for_role) - len(present),
        }

    replay_paths = _discover_descriptors(replay_root)
    replay_cfgs = [
        (role, role_cfg)
        for role, role_cfg in roles.items()
        if str(role_cfg["discover_kind"]) == "pretrain_replay"
    ]
    if len(replay_cfgs) != 1:
        raise ValueError("Exactly one pretrain_replay role must be configured")
    replay_role, replay_cfg = replay_cfgs[0]
    if replay_cfg.get("enabled") is not True or replay_cfg.get("required") is not True:
        raise ValueError("pretrain_replay must be enabled=true and required=true")
    replay = [validate_finetune_source(path, allowed_roots) for path in replay_paths]
    if any(source["source_kind"] != "pretrain_replay" for source in replay):
        raise ValueError("Only pretrain_replay may appear under sources/replay")
    if bool(replay_cfg.get("required", True)) and len(replay) != 1:
        raise ValueError("Exactly one required pretrain replay source must exist")
    if len(replay) > 1:
        raise ValueError("Multiple replay sources are not supported")
    role_status[replay_role] = {
        "status": "present_valid" if replay else "absent_optional",
        "source_count": len(replay),
    }
    validate_source_set([*gold_all, *replay])
    return {
        "gold": gold,
        "gold_all": gold_all,
        "replay": replay,
        "roles": roles,
        "role_status": role_status,
        "source_selection": source_selection,
        "source_selection_manifest": selection_manifest,
    }


def _source_by_dataset(sources: Sequence[Mapping[str, Any]]) -> Dict[str, Mapping[str, Any]]:
    result = {str(source["dataset_id"]): source for source in sources}
    if len(result) != len(sources):
        raise ValueError("Gold dataset IDs must be unique")
    return result


def _validate_gold_row(row: Mapping[str, Any], source: Mapping[str, Any]) -> None:
    errors = validate_label_record(row, split="train")
    if errors:
        raise ValueError("Invalid aggregate Gold row {}: {}".format(_canonical_id(row), errors))
    if str(row.get("annotation_provenance")) != "human_gold" or str(row.get("supervision_tier")) != "gold":
        raise ValueError("Aggregate Gold rows require human_gold provenance and gold tier")
    review = row.get("finetune_review") or {}
    present = bool((row.get("hand_presence") or {}).get("present", False))
    expected_presence = "hand" if present else "no_hand"
    if str(review.get("presence_decision")) != expected_presence:
        raise ValueError("Gold row lacks explicit presence decision: {}".format(_canonical_id(row)))
    handedness = str((row.get("handedness") or {}).get("label", "unknown"))
    if str(review.get("handedness_decision", "")).lower() != handedness.lower():
        raise ValueError("Gold row handedness decision does not match canonical label")
    source_sha = str(review.get("source_descriptor_sha256") or "")
    if source_sha != str(source["sha256"]):
        raise ValueError("Gold row is not bound to its source descriptor SHA")
    if source["source_kind"] != "external_gold" and not str(review.get("task_descriptor_sha256") or ""):
        raise ValueError("Reviewed Gold row lacks task_descriptor_sha256")
    if not present and str(review.get("presence_decision")) != "no_hand":
        raise ValueError("Gold negative must come from strict CVAT no_hand")
    presence_weight, landmark_weight, handedness_weight = effective_head_weights(row)
    if presence_weight <= 0.0:
        raise ValueError("Every included Gold row must supervise presence")
    if present and landmark_weight <= 0.0:
        raise ValueError("Gold positive must supervise landmarks")
    if not present and (landmark_weight != 0.0 or handedness_weight != 0.0):
        raise ValueError("Gold negative may supervise only presence")
    if handedness.lower() in {"left", "right"} and handedness_weight <= 0.0:
        raise ValueError("Known Gold handedness must have non-zero head weight")
    if handedness.lower() == "unknown" and handedness_weight != 0.0:
        raise ValueError("Unknown Gold handedness must have zero head weight")
    if str(row.get("sample_type") or "") not in POSITIVE_SAMPLE_TYPES + NEGATIVE_SAMPLE_TYPES:
        raise ValueError("Gold row has an invalid sample_type")
    if str(row.get("sampling_bucket") or "") != "gold:" + str(row.get("sample_type")):
        raise ValueError("Gold row sampling_bucket does not match tier/sample_type")


def _ignored_rows(
    aggregate: Mapping[str, Any], active_dataset_ids: Set[str]
) -> List[Dict[str, Any]]:
    result = []
    for row in aggregate["excluded"]:
        if str(row.get("dataset_id") or "") not in active_dataset_ids:
            continue
        if bool(row.get("ignore_for_training")) or str(row.get("selection_action")) == "drop_ignore":
            value = clean_row(row)
            value["finetune_curation_action"] = "EXCLUDE_IGNORE"
            result.append(value)
    result.sort(key=identity_value)
    return result


def _gold_weights(
    rows: List[Dict[str, Any]],
    sources: Sequence[Mapping[str, Any]],
    roles: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    source_map = _source_by_dataset(sources)
    present_roles = sorted({str(source["role"]) for source in sources})
    configured = {
        role: float(roles[role].get("target_gold_weight", 0.0))
        for role in present_roles
    }
    if any(not math.isfinite(value) or value < 0.0 for value in configured.values()) or sum(configured.values()) <= 0.0:
        raise ValueError("Present Gold roles require finite positive target weights")
    effective = {role: value / sum(configured.values()) for role, value in configured.items()}
    source_counts = Counter(str(row.get("dataset_id") or "") for row in rows)
    source_base: Dict[str, float] = {}
    for role in present_roles:
        role_sources = [source for source in sources if str(source["role"]) == role]
        raw = {
            str(source["dataset_id"]): math.sqrt(source_counts[str(source["dataset_id"])])
            * float((source["descriptor"].get("sampling") or {}).get("source_weight", 1.0))
            for source in role_sources
        }
        denominator = sum(raw.values())
        if denominator <= 0.0:
            raise ValueError("Gold source role has no included rows: {}".format(role))
        for dataset_id, value in raw.items():
            source_base[dataset_id] = value / denominator

    cell_reports: Dict[str, Any] = {}
    for sample_type in sorted({str(row.get("sample_type")) for row in rows}):
        cell = [row for row in rows if str(row.get("sample_type")) == sample_type]
        cell_roles = sorted({str(source_map[str(row["dataset_id"])]["role"]) for row in cell})
        role_denominator = sum(effective[role] for role in cell_roles)
        role_cell = {role: effective[role] / role_denominator for role in cell_roles}
        source_cell: Dict[str, float] = {}
        for role in cell_roles:
            datasets = sorted({str(row["dataset_id"]) for row in cell if str(source_map[str(row["dataset_id"])]["role"]) == role})
            denominator = sum(source_base[dataset_id] for dataset_id in datasets)
            for dataset_id in datasets:
                source_cell[dataset_id] = role_cell[role] * source_base[dataset_id] / denominator
        row_weights: Dict[str, float] = {}
        for dataset_id, target in source_cell.items():
            source_rows = [row for row in cell if str(row["dataset_id"]) == dataset_id]
            sequences = Counter(
                str(row.get("source_sequence_id") or row.get("source_group_id") or _canonical_id(row))
                for row in source_rows
            )
            raw_weights = {
                _canonical_id(row): 1.0 / sequences[
                    str(row.get("source_sequence_id") or row.get("source_group_id") or _canonical_id(row))
                ]
                for row in source_rows
            }
            denominator = sum(raw_weights.values())
            for row in source_rows:
                weight = target * raw_weights[_canonical_id(row)] / denominator
                row["sampling_weight"] = weight
                row_weights[_canonical_id(row)] = weight
        cell_reports[sample_type] = {
            "role_effective_weights": role_cell,
            "source_effective_weights": source_cell,
            "row_weight_sum": sum(row_weights.values()),
            "record_count": len(cell),
        }
    return {
        "configured_role_weights": configured,
        "effective_role_weights": effective,
        "source_sqrt_allocation": source_base,
        "cells": cell_reports,
    }


def _diverse_stable(rows: Sequence[Mapping[str, Any]], count: int, salt: str) -> List[Dict[str, Any]]:
    ordered = sorted(rows, key=lambda row: stable_hex(salt, _canonical_id(row)))
    first: List[Mapping[str, Any]] = []
    later: List[Mapping[str, Any]] = []
    seen: Set[Tuple[str, str]] = set()
    for row in ordered:
        key = (
            str(row.get("dataset_id") or ""),
            str(row.get("source_sequence_id") or row.get("source_group_id") or _canonical_id(row)),
        )
        if key in seen:
            later.append(row)
        else:
            seen.add(key)
            first.append(row)
    return [clean_row(row) for row in (first + later)[:count]]


def _handedness(row: Mapping[str, Any]) -> str:
    return str((row.get("handedness") or {}).get("label", "unknown")).lower()


def _ensure_smoke_handedness(
    selected: Dict[str, List[Dict[str, Any]]],
    pools: Dict[str, Sequence[Mapping[str, Any]]],
    salt: str,
) -> None:
    positive_categories = ("gold_positive", "pseudo_positive")
    all_positive = [row for category in positive_categories for row in pools[category]]
    available = {_handedness(row) for row in all_positive}
    if not {"left", "right"}.issubset(available):
        raise ValueError("Finetune smoke requires at least one known Left and Right positive")
    desired = ["left", "right"] + (["unknown"] if "unknown" in available else [])
    for label in desired:
        if any(_handedness(row) == label for category in positive_categories for row in selected[category]):
            continue
        replacement_done = False
        for category in positive_categories:
            chosen_ids = {_canonical_id(row) for row in selected[category]}
            candidates = [
                row for row in pools[category]
                if _handedness(row) == label and _canonical_id(row) not in chosen_ids
            ]
            if not candidates:
                continue
            candidate = min(candidates, key=lambda row: stable_hex(salt + ":force:" + label, _canonical_id(row)))
            counts = Counter(_handedness(row) for row in selected[category])
            replaceable = [
                (index, row) for index, row in enumerate(selected[category])
                if counts[_handedness(row)] > 1 and _handedness(row) != label
            ]
            if replaceable:
                index, _ = max(replaceable, key=lambda item: stable_hex(salt + ":replace", _canonical_id(item[1])))
                selected[category][index] = clean_row(candidate)
                replacement_done = True
                break
        if not replacement_done:
            raise ValueError("Could not place {} handedness in fixed smoke quotas".format(label))


def build_smoke_snapshot(
    rows: Sequence[Mapping[str, Any]],
    salt: str = "finetune_smoke_v1",
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """Build the persisted, deterministic 256-row finetune smoke snapshot."""

    pools = {
        "gold_positive": [row for row in rows if str(row.get("supervision_tier")) == "gold" and bool((row.get("hand_presence") or {}).get("present", False))],
        "gold_negative": [row for row in rows if str(row.get("supervision_tier")) == "gold" and not bool((row.get("hand_presence") or {}).get("present", False))],
        "pseudo_positive": [row for row in rows if str(row.get("supervision_tier")) == "pseudo" and bool((row.get("hand_presence") or {}).get("present", False))],
        "pseudo_neg_runtime": [row for row in rows if str(row.get("supervision_tier")) == "pseudo" and str(row.get("sample_type")) == "NEG_RUNTIME_CANDIDATE"],
        "pseudo_neg_low": [row for row in rows if str(row.get("supervision_tier")) == "pseudo" and str(row.get("sample_type")) == "NEG_LOW_PALM_CANDIDATE"],
    }
    gold_negative_count = min(16, len(pools["gold_negative"]))
    quotas = {
        "gold_positive": 80 + (16 - gold_negative_count),
        "gold_negative": gold_negative_count,
        "pseudo_positive": 96,
        "pseudo_neg_runtime": 32,
        "pseudo_neg_low": 32,
    }
    for category, quota in quotas.items():
        if len(pools[category]) < quota:
            raise ValueError(
                "Finetune smoke category {} needs {}, has {}".format(category, quota, len(pools[category]))
            )
    selected = {
        category: _diverse_stable(pools[category], quota, salt + ":" + category)
        for category, quota in quotas.items()
    }
    _ensure_smoke_handedness(selected, pools, salt)
    smoke: List[Dict[str, Any]] = []
    selection: List[Dict[str, Any]] = []
    for category in quotas:
        for row in selected[category]:
            value = clean_row(row)
            value["finetune_curation"] = dict(value.get("finetune_curation") or {})
            value["finetune_curation"]["smoke_original_sampling_weight"] = value.get("sampling_weight")
            value["sampling_weight"] = 1.0
            value["finetune_curation"]["smoke_sampling_weight"] = 1.0
            smoke.append(value)
            selection.append(
                {
                    "schema_version": "finetune_smoke_selection_v1",
                    "global_crop_id": _canonical_id(row),
                    "category": category,
                    "dataset_id": str(row.get("dataset_id") or ""),
                    "source_sequence_id": row.get("source_sequence_id"),
                    "selection_hash": stable_hex(salt + ":" + category, _canonical_id(row)),
                }
            )
    smoke.sort(key=_canonical_id)
    selection.sort(key=lambda row: (str(row["category"]), str(row["selection_hash"])))
    if len(smoke) != 256 or len({_canonical_id(row) for row in smoke}) != 256:
        raise ValueError("Finetune smoke snapshot must contain 256 unique rows")
    report = {
        "record_count": 256,
        "salt": salt,
        "quotas": quotas,
        "selected_by_handedness": dict(sorted(Counter(_handedness(row) for row in smoke if bool((row.get("hand_presence") or {}).get("present", False))).items())),
        "unknown_handedness_runtime_check": "required" if any(_handedness(row) == "unknown" for row in smoke) else "not_applicable",
        "gold_negative_policy": "selected" if gold_negative_count else "redistributed_to_gold_positive",
    }
    return smoke, selection, report


def _load_leakage_rows(config: Mapping[str, Any]) -> List[Dict[str, Any]]:
    leakage = config.get("leakage") or {}
    paths: List[Path] = []
    for key in ("validation_labels", "validation_ignored", "test_labels", "test_ignored"):
        value = leakage.get(key)
        if value:
            path = resolve_path(str(value), config)
            if not path.is_file():
                raise FileNotFoundError("Leakage reference does not exist: {}".format(path))
            paths.append(path)
    return [clean_row(row) for path in paths for row in read_jsonl(path)]


def _check_leakage(train_rows: Sequence[Mapping[str, Any]], eval_rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    eval_tokens: Dict[Tuple[str, str], str] = {}
    for row in eval_rows:
        for token in _leakage_tokens(row):
            eval_tokens[token] = _canonical_id(row)
    conflicts = []
    for row in train_rows:
        for token in sorted(_leakage_tokens(row)):
            if token in eval_tokens:
                conflicts.append({"train": _canonical_id(row), "eval": eval_tokens[token], "token": list(token)})
    if conflicts:
        raise ValueError("Train/Val/Test leakage detected: {}".format(conflicts[:10]))
    return {"eval_records": len(eval_rows), "checked_token_count": len(eval_tokens), "conflicts": 0}


def merge_gold_over_replay(
    gold_rows: Sequence[Mapping[str, Any]],
    replay_rows: Sequence[Mapping[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Keep Gold for every shared identity and emit an explicit replay audit."""

    gold_tokens: Dict[Tuple[str, str], str] = {
        token: _canonical_id(row)
        for row in gold_rows
        for token in _identity_tokens(row)
    }
    kept: List[Dict[str, Any]] = []
    superseded: List[Dict[str, Any]] = []
    for row in replay_rows:
        overlaps = sorted(_identity_tokens(row) & set(gold_tokens))
        if overlaps:
            value = clean_row(row)
            value["finetune_curation_action"] = "SUPERSEDED_BY_GOLD"
            value["superseded_by_gold"] = sorted({gold_tokens[token] for token in overlaps})
            value["matched_identity_tokens"] = [list(token) for token in overlaps]
            superseded.append(value)
        else:
            kept.append(clean_row(row))
    return kept, superseded


def check_finetune_sources_from_config(config: Union[Mapping[str, Any], str, Path]) -> Dict[str, Any]:
    """Run the P3 source/data gate without publishing curation outputs."""

    cfg = _config_mapping(config)
    if str(cfg.get("task")) != "curate_finetune":
        raise ValueError("Finetune curation config task must be curate_finetune")
    allowed_roots = _resolved_allowed_roots(cfg)
    loaded = _load_sources(cfg, allowed_roots)
    gate = cfg.get("gate") or {}
    if gate.get("require_replay", True) is not True:
        raise ValueError("gate.require_replay must remain true")
    if len(loaded["replay"]) != 1:
        raise ValueError("Finetune curation requires exactly one replay source")
    if bool(gate.get("require_at_least_one_gold_source", True)) and not loaded["gold"]:
        raise ValueError("Finetune curation requires at least one Gold source")
    aggregate_path = resolve_path(str((cfg.get("gold_aggregate") or {}).get("descriptor") or ""), cfg)
    if not loaded["gold"]:
        raise ValueError("Gold aggregate cannot be validated without Gold source descriptors")
    gold_repository_root = resolve_path(
        str(cfg.get("gold_source_descriptor_root")), cfg
    )
    aggregate = validate_gold_aggregate(
        aggregate_path, gold_repository_root, loaded["gold_all"]
    )
    dataset_sources = _source_by_dataset(loaded["gold"])
    all_dataset_sources = _source_by_dataset(loaded["gold_all"])
    active_dataset_ids = set(dataset_sources)
    gold_rows: List[Dict[str, Any]] = []
    disabled_rows: List[Dict[str, Any]] = []
    for raw in aggregate["included"]:
        row = clean_row(raw)
        dataset_id = str(row.get("dataset_id") or "")
        if dataset_id not in all_dataset_sources:
            raise ValueError(
                "Aggregate Gold row references an undiscovered dataset: {}".format(dataset_id)
            )
        if dataset_id not in active_dataset_ids:
            row["finetune_curation_action"] = "EXCLUDE_SOURCE_DISABLED"
            row["disabled_source_id"] = str(all_dataset_sources[dataset_id]["source_id"])
            disabled_rows.append(row)
            continue
        source = dataset_sources.get(dataset_id)
        _validate_gold_row(row, source)
        crop_path, digest = _check_row_crop(row, allowed_roots)
        row["crop_path"] = crop_path
        row["image_sha256"] = digest
        row["training_stage"] = "finetune"
        row["source_role"] = str(source["role"])
        row["source_kind"] = str(source["source_kind"])
        row["finetune_source_id"] = str(source["source_id"])
        gold_rows.append(row)
    minimum = int(gate.get("minimum_gold_positive", 256))
    gold_positive = sum(bool((row.get("hand_presence") or {}).get("present", False)) for row in gold_rows)
    if gold_positive < minimum:
        raise ValueError("Gold positives {} are below configured minimum {}".format(gold_positive, minimum))
    _assert_unique_tokens(gold_rows, "Gold aggregate")

    ignored = _ignored_rows(aggregate, active_dataset_ids)
    for raw in aggregate["excluded"]:
        dataset_id = str(raw.get("dataset_id") or "")
        if dataset_id in all_dataset_sources and dataset_id not in active_dataset_ids:
            row = clean_row(raw)
            row["finetune_curation_action"] = "EXCLUDE_SOURCE_DISABLED"
            row["disabled_source_id"] = str(all_dataset_sources[dataset_id]["source_id"])
            disabled_rows.append(row)
    _assert_unique_tokens(ignored, "Gold ignored")
    included_tokens = {token for row in gold_rows for token in _identity_tokens(row)}
    ignored_overlap = [_canonical_id(row) for row in ignored if included_tokens & _identity_tokens(row)]
    if ignored_overlap:
        raise ValueError("Gold included and ignored identities overlap: {}".format(ignored_overlap[:10]))

    replay_rows = [clean_row(row) for row in loaded["replay"][0]["rows"]]
    for row in replay_rows:
        crop_path, digest = _check_row_crop(row, allowed_roots)
        row["crop_path"] = crop_path
        row["image_sha256"] = digest
        row["training_stage"] = "finetune"
        row["supervision_tier"] = "pseudo"
        row["source_role"] = "pretrain_replay"
        row["source_kind"] = "pretrain_replay"
        row["finetune_source_id"] = str(loaded["replay"][0]["source_id"])
        if str(row.get("sampling_bucket") or "") != "pseudo:" + str(row.get("sample_type")):
            raise ValueError("Replay row sampling_bucket does not match tier/sample_type")
    _assert_unique_tokens(replay_rows, "Replay")

    kept_replay, superseded = merge_gold_over_replay(gold_rows, replay_rows)

    weight_report = _gold_weights(gold_rows, loaded["gold"], loaded["roles"])
    final_rows = sorted([*gold_rows, *kept_replay], key=lambda row: (str(row.get("supervision_tier")), _canonical_id(row)))
    eval_rows = _load_leakage_rows(cfg)
    leakage_report = _check_leakage(final_rows, eval_rows)
    smoke_cfg = cfg.get("smoke") or {}
    smoke, smoke_selection, smoke_report = build_smoke_snapshot(
        final_rows, str(smoke_cfg.get("salt", "finetune_smoke_v1"))
    )
    return {
        "status": "ok",
        "config": cfg,
        "allowed_crop_roots": [str(path) for path in allowed_roots],
        "sources": loaded,
        "aggregate": aggregate,
        "gold_rows": gold_rows,
        "replay_rows": kept_replay,
        "ignored_rows": ignored,
        "disabled_rows": disabled_rows,
        "superseded_rows": superseded,
        "final_rows": final_rows,
        "smoke_rows": smoke,
        "smoke_selection": smoke_selection,
        "reports": {
            "source_roles": loaded["role_status"],
            "source_selection": loaded["source_selection"],
            "source_selection_manifest": loaded["source_selection_manifest"],
            "source_weights": weight_report,
            "leakage": leakage_report,
            "smoke": smoke_report,
        },
    }


def _artifact_entry(path: Path, root: Path, count: Optional[int] = None) -> Dict[str, Any]:
    value: Dict[str, Any] = {
        "sha256": sha256_file(path),
        "size_bytes": int(path.stat().st_size),
    }
    if count is not None:
        value["count"] = int(count)
    return value


def curate_finetune_from_config(
    config: Union[Mapping[str, Any], str, Path],
    overwrite: Optional[bool] = None,
) -> Dict[str, Any]:
    """Run all P3 gates and atomically publish the final canonical snapshot."""

    checked = check_finetune_sources_from_config(config)
    cfg = checked["config"]
    output_cfg = cfg.get("output") or {}
    output_dir = resolve_path(str(output_cfg.get("dir") or ""), cfg)
    if not str(output_cfg.get("dir") or ""):
        raise ValueError("output.dir is required")
    allow_overwrite = bool(output_cfg.get("overwrite", False) if overwrite is None else overwrite)
    if output_dir.exists() and not allow_overwrite:
        raise FileExistsError("Finetune curation output already exists: {}".format(output_dir))
    if output_dir.exists():
        sentinel = output_dir / "qc" / "sha256_manifest.json"
        if output_dir.is_symlink() or not sentinel.is_file():
            raise ValueError("Refusing to overwrite a non-curation directory")
        previous = _read_json(sentinel)
        if str(previous.get("schema_version")) != CURATION_SCHEMA or Path(str(previous.get("output_dir") or "")).resolve() != output_dir.resolve():
            raise ValueError("Refusing to overwrite a directory not owned by this curation snapshot")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=output_dir.name + ".tmp.", dir=str(output_dir.parent)))
    try:
        labels_dir = temporary / "05_labels"
        audit_dir = temporary / "audit"
        qc_dir = temporary / "qc"
        labels_path = labels_dir / "hand_training_labels_finetune.jsonl"
        ignored_path = labels_dir / "hand_training_ignored_finetune.jsonl"
        smoke_path = labels_dir / "hand_training_labels_finetune_smoke.jsonl"
        source_catalog_path = audit_dir / "source_catalog.jsonl"
        selection_catalog_path = audit_dir / "selection_catalog.jsonl"
        excluded_path = audit_dir / "excluded_and_superseded.jsonl"
        smoke_selection_path = audit_dir / "finetune_smoke_selection.jsonl"
        report_path = qc_dir / "curation_report.json"
        manifest_path = qc_dir / "sha256_manifest.json"

        source_catalog = [
            {
                "source_id": source["source_id"],
                "dataset_id": source["dataset_id"],
                "source_kind": source["source_kind"],
                "role": source.get("role", "pretrain_replay"),
                "descriptor_path": source["path"],
                "descriptor_sha256": source["sha256"],
            }
            for source in [*checked["sources"]["gold"], *checked["sources"]["replay"]]
        ]
        source_catalog.sort(key=lambda row: str(row["source_id"]))
        selection_catalog = [
            {
                "global_crop_id": _canonical_id(row),
                "dataset_id": str(row.get("dataset_id") or ""),
                "supervision_tier": str(row.get("supervision_tier") or ""),
                "sample_type": str(row.get("sample_type") or ""),
                "source_role": str(row.get("source_role") or ""),
                "source_id": str(row.get("finetune_source_id") or ""),
                "sampling_weight": float(row.get("sampling_weight", 1.0)),
                "action": "INCLUDE",
            }
            for row in checked["final_rows"]
        ]
        excluded = sorted(
            [
                *checked["ignored_rows"],
                *checked["disabled_rows"],
                *checked["superseded_rows"],
            ],
            key=_canonical_id,
        )
        write_jsonl(labels_path, checked["final_rows"])
        write_jsonl(ignored_path, checked["ignored_rows"])
        write_jsonl(smoke_path, checked["smoke_rows"])
        write_jsonl(source_catalog_path, source_catalog)
        write_jsonl(selection_catalog_path, selection_catalog)
        write_jsonl(excluded_path, excluded)
        write_jsonl(smoke_selection_path, checked["smoke_selection"])

        report = {
            "status": "ok",
            "schema_version": CURATION_SCHEMA,
            "counts": {
                "gold": len(checked["gold_rows"]),
                "replay_after_gold_override": len(checked["replay_rows"]),
                "included": len(checked["final_rows"]),
                "ignored": len(checked["ignored_rows"]),
                "disabled_source_rows": len(checked["disabled_rows"]),
                "superseded_by_gold": len(checked["superseded_rows"]),
                "smoke": len(checked["smoke_rows"]),
            },
            "source_roles": checked["reports"]["source_roles"],
            "source_selection": checked["reports"]["source_selection"],
            "source_selection_manifest": checked["reports"]["source_selection_manifest"],
            "source_weights": checked["reports"]["source_weights"],
            "leakage": checked["reports"]["leakage"],
            "smoke": checked["reports"]["smoke"],
            "aggregate": {"path": checked["aggregate"]["path"], "sha256": checked["aggregate"]["sha256"]},
            "allowed_crop_roots": checked["allowed_crop_roots"],
        }
        write_json(report_path, report)
        artifacts = {
            path.relative_to(temporary).as_posix(): _artifact_entry(path, temporary, count)
            for path, count in (
                (labels_path, len(checked["final_rows"])),
                (ignored_path, len(checked["ignored_rows"])),
                (smoke_path, len(checked["smoke_rows"])),
                (source_catalog_path, len(source_catalog)),
                (selection_catalog_path, len(selection_catalog)),
                (excluded_path, len(excluded)),
                (smoke_selection_path, len(checked["smoke_selection"])),
                (report_path, None),
            )
        }
        image_entries = sorted(
            (str(row["crop_path"]), str(row["image_sha256"])) for row in checked["final_rows"]
        )
        config_path = cfg.get("_meta", {}).get("config_path")
        manifest = {
            "schema_version": CURATION_SCHEMA,
            "output_dir": str(output_dir.resolve()),
            "config_path": str(config_path) if config_path else None,
            "config_sha256": sha256_file(config_path) if config_path and Path(str(config_path)).is_file() else None,
            "gold_aggregate": {"path": checked["aggregate"]["path"], "sha256": checked["aggregate"]["sha256"]},
            "gold_selection": checked["reports"]["source_selection_manifest"],
            "source_descriptors": source_catalog,
            "allowed_crop_roots": checked["allowed_crop_roots"],
            "artifacts": artifacts,
            "images": {
                "count": len(image_entries),
                "aggregate_sha256": hashlib.sha256(
                    "".join("{}:{}\n".format(path, digest) for path, digest in image_entries).encode("utf-8")
                ).hexdigest(),
            },
            "smoke": {
                "labels": "05_labels/hand_training_labels_finetune_smoke.jsonl",
                "selection": "audit/finetune_smoke_selection.jsonl",
                "selection_config_sha256": hashlib.sha256(
                    json.dumps(cfg.get("smoke") or {}, sort_keys=True, separators=(",", ":")).encode("utf-8")
                ).hexdigest(),
                "count": len(checked["smoke_rows"]),
            },
        }
        write_json(manifest_path, manifest)
        if output_dir.exists():
            shutil.rmtree(str(output_dir))
        os.replace(str(temporary), str(output_dir))
        temporary = None  # type: ignore[assignment]
        return {
            "status": "ok",
            "output_dir": str(output_dir.resolve()),
            "manifest": str((output_dir / "qc" / "sha256_manifest.json").resolve()),
            "manifest_sha256": sha256_file(output_dir / "qc" / "sha256_manifest.json"),
            "counts": report["counts"],
        }
    finally:
        if temporary is not None and temporary.exists():
            shutil.rmtree(str(temporary), ignore_errors=True)


def verify_finetune_curation_manifest(
    config: Mapping[str, Any],
    dataset: Mapping[str, Any],
    error_type=ValueError,
) -> Dict[str, Any]:
    """Authenticate a finetune labels snapshot before ``data.py`` consumes it."""

    manifest_value = dataset.get("curation_manifest")
    if not manifest_value:
        raise error_type("Finetune data requires data.curation_manifest")
    manifest_path = resolve_path(str(manifest_value), config)
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise error_type("Finetune curation manifest is missing or a symlink: {}".format(manifest_path))
    try:
        manifest = _read_json(manifest_path)
    except (OSError, ValueError) as exc:
        raise error_type("Could not read finetune curation manifest: {}".format(exc)) from exc
    required_schema = str(dataset.get("require_curation_schema") or CURATION_SCHEMA)
    if str(manifest.get("schema_version")) != required_schema:
        raise error_type("Finetune curation schema mismatch")
    output_root = Path(str(manifest.get("output_dir") or ""))
    curator_config_value = str(manifest.get("config_path") or "")
    curator_config_sha = str(manifest.get("config_sha256") or "")
    curator_config = Path(curator_config_value)
    if (
        not curator_config_value
        or not curator_config.is_absolute()
        or curator_config.is_symlink()
        or not curator_config.is_file()
        or not curator_config_sha
        or sha256_file(curator_config) != curator_config_sha
    ):
        raise error_type("Finetune curation manifest does not authenticate its curator config")
    labels_path = resolve_path(str(dataset.get("labels") or ""), config)
    try:
        relative = labels_path.resolve(strict=True).relative_to(output_root.resolve(strict=True)).as_posix()
    except (OSError, ValueError) as exc:
        raise error_type("Finetune labels are outside the authenticated snapshot") from exc
    artifact = (manifest.get("artifacts") or {}).get(relative)
    if not isinstance(artifact, Mapping) or not str(artifact.get("sha256") or ""):
        raise error_type("Finetune manifest does not authenticate configured labels")
    actual = sha256_file(labels_path)
    if actual != str(artifact["sha256"]):
        raise error_type("Finetune labels SHA mismatch")
    return {
        "path": str(manifest_path.resolve()),
        "sha256": sha256_file(manifest_path),
        "schema_version": manifest.get("schema_version"),
        "training_labels_relative_path": relative,
        "training_labels_sha256": actual,
        "curator_config": str(curator_config.resolve()),
        "curator_config_sha256": curator_config_sha,
        "gold_aggregate_sha256": (manifest.get("gold_aggregate") or {}).get("sha256"),
        "image_count": (manifest.get("images") or {}).get("count"),
        "image_aggregate_sha256": (manifest.get("images") or {}).get("aggregate_sha256"),
    }
