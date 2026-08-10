"""HLML 4.0 reader for HLMF 3.0 published warehouse manifests.

Images remain under ``HAND_DATASET_ROOT``.  A snapshot written below
``HAND_TRAIN_ROOT`` contains JSONL indexes and an audit report only; it never
copies or authenticates image bytes with repeated cryptographic hashes.
"""

from __future__ import annotations

import json
import math
import os
import re
import sqlite3
from collections import Counter, defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

from .config import load_config
from .io_utils import read_image, read_jsonl, write_json, write_jsonl


HLMF_SCHEMA = "hlmf_dataset_v1"
SNAPSHOT_SCHEMA = "hlml_warehouse_snapshot_v1"
TRAIN_SCHEMA = "hlml_warehouse_train_v1"
EVALUATION_SCHEMA = "hlml_fixed_roi_evaluation_v1"
STAGES = ("geometry", "multitask", "multi_finetune")


class WarehouseContractError(ValueError):
    """Raised when an HLMF publication cannot be consumed safely."""


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise WarehouseContractError("required manifest does not exist: {}".format(path))
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise WarehouseContractError("manifest root must be an object: {}".format(path))
    return value


def _clean_rows(path: Path) -> List[Dict[str, Any]]:
    rows = read_jsonl(path)
    for row in rows:
        row.pop("_jsonl_line", None)
    return rows


def _positive(row: Mapping[str, Any]) -> bool:
    return bool((row.get("hand_presence") or {}).get("present", False))


def _capture_fields(capture_source_id: str) -> Dict[str, str]:
    parts = str(capture_source_id).split("-")
    fields = ("background", "distance", "lighting", "condition", "split", "session", "performer")
    parsed = dict(zip(fields, parts))
    valid_slugs = all(re.fullmatch(r"[a-z0-9_]+", part) for part in parts)
    if (
        len(parts) != 7
        or parsed.get("split") not in {"train", "val", "test"}
        or re.fullmatch(r"s[0-9]+", parsed.get("session", "")) is None
        or not valid_slugs
    ):
        raise WarehouseContractError(
            "invalid capture_source_id; expected background-distance-lighting-condition-"
            "split-session-performer: {!r}".format(capture_source_id)
        )
    return parsed


def _dataset_entries(config: Mapping[str, Any], key: str) -> List[Dict[str, Any]]:
    raw = config.get(key) or []
    if not isinstance(raw, list):
        raise WarehouseContractError("{} must be a list".format(key))
    output = []
    for value in raw:
        if not isinstance(value, Mapping):
            raise WarehouseContractError("{} entries must be objects".format(key))
        entry = dict(value)
        if not entry.get("dataset_id") or not entry.get("proposal_variant"):
            raise WarehouseContractError(
                "{} entries require dataset_id and proposal_variant".format(key)
            )
        weight = float(entry.get("weight", 1.0))
        if not math.isfinite(weight) or weight <= 0.0:
            raise WarehouseContractError("dataset weight must be finite and positive")
        entry["weight"] = weight
        output.append(entry)
    return output


def _selected_ids(config: Mapping[str, Any], key: str) -> List[Dict[str, Any]]:
    raw = config.get(key) or []
    if not isinstance(raw, list):
        raise WarehouseContractError("{} must be a list".format(key))
    output = []
    for value in raw:
        if isinstance(value, str):
            item = {key[:-1] + "_id": value, "weight": 1.0}
        elif isinstance(value, Mapping):
            item = dict(value)
        else:
            raise WarehouseContractError("{} entries must be IDs or objects".format(key))
        weight = float(item.get("weight", 1.0))
        if not math.isfinite(weight) or weight <= 0.0:
            raise WarehouseContractError("{} weight must be finite and positive".format(key))
        item["weight"] = weight
        output.append(item)
    return output


class WarehouseReader:
    """Read and validate one HLMF warehouse without mutating it."""

    def __init__(self, dataset_root: Path):
        self.root = Path(dataset_root).resolve()
        self.registry_path = self.root / "Registry" / "registry.sqlite3"
        if not self.root.is_dir():
            raise WarehouseContractError("HAND_DATASET_ROOT does not exist: {}".format(self.root))
        if not self.registry_path.is_file():
            raise WarehouseContractError("HLMF registry does not exist: {}".format(self.registry_path))

    @contextmanager
    def _registry(self):
        connection = sqlite3.connect("file:{}?mode=ro".format(self.registry_path.as_posix()), uri=True)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
        finally:
            connection.close()

    def _check_image(self, relative: str) -> Path:
        path = (self.root / str(relative)).resolve()
        if not _inside(path, self.root):
            raise WarehouseContractError("ROI path escapes HAND_DATASET_ROOT: {}".format(relative))
        if not path.is_file():
            raise WarehouseContractError("ROI image is missing: {}".format(path))
        image = read_image(path)
        if image is None:
            raise WarehouseContractError("ROI image cannot be decoded: {}".format(path))
        if image.ndim == 3 and image.shape[2] == 1:
            image = image[:, :, 0]
        if image.ndim != 2 or tuple(image.shape) != (256, 256):
            raise WarehouseContractError(
                "ROI must decode as one-channel 256x256: {} ({})".format(path, image.shape)
            )
        return path

    def _check_registry_roi(self, row: Mapping[str, Any]) -> None:
        with self._registry() as db:
            stored = db.execute(
                "SELECT raw_image_id,capture_source_id,proposal_variant,crop_relpath "
                "FROM rois WHERE roi_id=?",
                (str(row.get("roi_id") or row.get("crop_id")),),
            ).fetchone()
        if stored is None:
            raise WarehouseContractError("roi_id is not registered: {}".format(row.get("roi_id")))
        expected = (
            str(row.get("raw_image_id")),
            str(row.get("capture_source_id")),
            str(row.get("proposal_variant")),
            str(row.get("crop_relpath") or row.get("crop_path")),
        )
        if tuple(stored) != expected:
            raise WarehouseContractError(
                "ROI manifest disagrees with registry for {}: {} != {}".format(
                    row.get("roi_id"), tuple(stored), expected
                )
            )

    def source_rows(
        self,
        scope: str,
        dataset_id: str,
        proposal_variant: str,
        expected_split: str | None = None,
        weight: float = 1.0,
    ) -> List[Dict[str, Any]]:
        bucket = "PretrainSource" if scope == "pretrain" else "EValSource"
        manifest_path = self.root / bucket / dataset_id / "dataset_manifest.json"
        manifest = _read_json(manifest_path)
        if str(manifest.get("schema_version")) != HLMF_SCHEMA:
            raise WarehouseContractError(
                "unsupported HLMF dataset manifest schema: {}".format(manifest_path)
            )
        if str(manifest.get("dataset_id")) != dataset_id or str(manifest.get("scope")) != scope:
            raise WarehouseContractError("dataset manifest identity mismatch: {}".format(manifest_path))
        rows: List[Dict[str, Any]] = []
        for source in manifest.get("capture_sources") or []:
            capture_id = str(source.get("capture_source_id"))
            capture = _capture_fields(capture_id)
            if expected_split and capture["split"] != expected_split:
                continue
            variants = [
                item
                for item in source.get("published_variants") or []
                if str(item.get("proposal_variant")) == proposal_variant
            ]
            if len(variants) != 1:
                raise WarehouseContractError(
                    "capture source {} must publish exactly one selected variant {}; got {}".format(
                        capture_id, proposal_variant, len(variants)
                    )
                )
            report = variants[0]
            labels_path = (self.root / str(report.get("labels_relpath", ""))).resolve()
            if not _inside(labels_path, self.root):
                raise WarehouseContractError("labels path escapes HAND_DATASET_ROOT: {}".format(labels_path))
            source_rows = _clean_rows(labels_path)
            if len(source_rows) != int(report.get("published_labels", -1)):
                raise WarehouseContractError("published label count mismatch: {}".format(labels_path))
            for row in source_rows:
                item = dict(row)
                if str(item.get("dataset_id")) != dataset_id:
                    raise WarehouseContractError("row dataset_id mismatch in {}".format(labels_path))
                if str(item.get("capture_source_id")) != capture_id:
                    raise WarehouseContractError("row capture_source_id mismatch in {}".format(labels_path))
                if str(item.get("proposal_variant")) != proposal_variant:
                    raise WarehouseContractError("row proposal_variant mismatch in {}".format(labels_path))
                if str(item.get("split")) != capture["split"]:
                    raise WarehouseContractError("row split mismatch in {}".format(labels_path))
                item["dataset_weight"] = float(weight)
                item.update({key: item.get(key, value) for key, value in capture.items()})
                self._check_registry_roi(item)
                relative = str(item.get("crop_relpath") or item.get("crop_path") or "")
                item["crop_relpath"] = relative
                item["_absolute_crop_path"] = str(self._check_image(relative))
                rows.append(item)
        if not rows:
            raise WarehouseContractError(
                "dataset {} has no published rows for variant {}{}".format(
                    dataset_id,
                    proposal_variant,
                    " and split {}".format(expected_split) if expected_split else "",
                )
            )
        return rows

    def negative_rows(self, negative_dataset_id: str, weight: float = 1.0) -> List[Dict[str, Any]]:
        root = self.root / "GoldSource" / "NegativeSamples" / negative_dataset_id / "published"
        manifest = _read_json(root / "manifest.json")
        if str(manifest.get("schema_version")) != HLMF_SCHEMA:
            raise WarehouseContractError("unsupported HLMF negative manifest schema")
        if str(manifest.get("image_policy")) != "copied_review_and_published_images":
            raise WarehouseContractError(
                "negative dataset must use independent HLMF published images"
            )
        if str(manifest.get("negative_dataset_id")) != negative_dataset_id:
            raise WarehouseContractError("negative dataset manifest identity mismatch")
        rows = _clean_rows(root / str(manifest.get("labels", "negative_labels.jsonl")))
        if len(rows) != int(manifest.get("records", -1)):
            raise WarehouseContractError("negative dataset record count mismatch")
        with self._registry() as db:
            status = db.execute(
                "SELECT status FROM negative_datasets WHERE negative_dataset_id=?",
                (negative_dataset_id,),
            ).fetchone()
            registered = {
                str(row[0]): str(row[1])
                for row in db.execute(
                    "SELECT roi_id,published_relpath FROM published_negatives WHERE negative_dataset_id=?",
                    (negative_dataset_id,),
                )
            }
        if status is None or status["status"] != "published":
            raise WarehouseContractError("negative dataset is not published in registry")
        expected_registry = {
            str(row.get("roi_id")): str(row.get("published_relpath")) for row in rows
        }
        if registered != expected_registry:
            raise WarehouseContractError("negative dataset rows disagree with registry")
        for row in rows:
            if str(row.get("split")) != "train":
                raise WarehouseContractError("published negative datasets must contain Train ROIs only")
            self._check_registry_roi(row)
            row["negative_dataset_id"] = negative_dataset_id
            row["dataset_weight"] = float(weight)
            row["hand_presence"] = {"present": False}
            row["landmarks_crop_norm"] = []
            source_relative = str(row.get("crop_relpath") or row.get("crop_path") or "")
            published_relative = str(row.get("published_relpath") or "")
            if not published_relative:
                raise WarehouseContractError("negative row is missing published_relpath")
            if published_relative == source_relative:
                raise WarehouseContractError(
                    "negative published image must be independent from its source ROI"
                )
            row["crop_relpath"] = published_relative
            row["crop_path"] = published_relative
            row["_absolute_crop_path"] = str(self._check_image(row["crop_relpath"]))
        return rows

    def selection_rows(self, selection_id: str, weight: float = 1.0) -> List[Dict[str, Any]]:
        root = self.root / "Selections" / selection_id / "published"
        manifest = _read_json(root / "manifest.json")
        if str(manifest.get("schema_version")) != HLMF_SCHEMA:
            raise WarehouseContractError("unsupported HLMF selection manifest schema")
        if str(manifest.get("selection_id")) != selection_id:
            raise WarehouseContractError("selection manifest identity mismatch")
        if str(manifest.get("image_policy")) != "copied_review_and_published_images":
            raise WarehouseContractError(
                "selection must use independent HLMF published images"
            )
        rows = _clean_rows(root / str(manifest.get("selection", "selection.jsonl")))
        if len(rows) != int(manifest.get("records", -1)):
            raise WarehouseContractError("selection record count mismatch")
        with self._registry() as db:
            status = db.execute(
                "SELECT status FROM selections WHERE selection_id=?", (selection_id,)
            ).fetchone()
        if status is None or status["status"] != "published":
            raise WarehouseContractError("selection is not published in registry")
        for row in rows:
            if str(row.get("split")) != "train" or not _positive(row):
                raise WarehouseContractError("hard selection accepts Train positives only")
            source_relative = str(row.get("source_crop_relpath") or "")
            published_relative = str(row.get("published_relpath") or "")
            original_relative = str(row.get("crop_relpath") or row.get("crop_path") or "")
            if not source_relative or not published_relative:
                raise WarehouseContractError(
                    "selection row requires source_crop_relpath and published_relpath"
                )
            if original_relative != source_relative:
                raise WarehouseContractError(
                    "selection source_crop_relpath disagrees with its source ROI path"
                )
            if published_relative == source_relative:
                raise WarehouseContractError(
                    "selection published image must be independent from its source ROI"
                )
            self._check_registry_roi(row)
            row["selection_id"] = selection_id
            row["dataset_weight"] = float(weight)
            row["crop_relpath"] = published_relative
            row["crop_path"] = published_relative
            row["_absolute_crop_path"] = str(self._check_image(published_relative))
        return rows


def _head_points(row: MutableMapping[str, Any]) -> None:
    present = _positive(row)
    if not present:
        row["landmarks_crop_norm"] = []
        row["landmarks_crop_px"] = []
        row["landmarks_image_px"] = []
        row["handedness"] = {"label": "unknown", "score": None}
        row["hand_id"] = None
        return
    points = list(row.get("landmarks_crop_norm") or [])
    if len(points) != 21:
        raise WarehouseContractError("positive ROI requires exactly 21 landmarks: {}".format(row.get("roi_id")))
    by_id = {int(point.get("id", index)): point for index, point in enumerate(points)}
    if set(by_id) != set(range(21)):
        raise WarehouseContractError("landmark IDs must be exactly 0..20: {}".format(row.get("roi_id")))
    normalized = [
        {"id": index, "x": float(by_id[index]["x"]), "y": float(by_id[index]["y"])}
        for index in range(21)
    ]
    row["landmarks_crop_norm"] = normalized
    if len(row.get("landmarks_crop_px") or []) != 21:
        row["landmarks_crop_px"] = [
            {"id": point["id"], "x": point["x"] * 255.0, "y": point["y"] * 255.0}
            for point in normalized
        ]
    if len(row.get("landmarks_image_px") or []) != 21:
        # Training and fixed-ROI evaluation never consume image-space points.
        # Preserve the field contract without reconstructing an original-image ROI.
        row["landmarks_image_px"] = [dict(point) for point in row["landmarks_crop_px"]]
    handedness = dict(row.get("handedness") or {})
    label = str(handedness.get("label", "unknown")).title()
    handedness["label"] = label if label in {"Left", "Right"} else "unknown"
    handedness.setdefault("score", None)
    row["handedness"] = handedness


def canonical_record(
    raw: Mapping[str, Any],
    training_stage: str | None,
    mix_role: str | None = None,
) -> Dict[str, Any]:
    row = {key: value for key, value in dict(raw).items() if not str(key).startswith("_")}
    roi_id = str(row.get("roi_id") or row.get("crop_id") or "")
    if not roi_id:
        raise WarehouseContractError("published row is missing roi_id")
    capture_id = str(row.get("capture_source_id") or "")
    capture = _capture_fields(capture_id)
    present = _positive(row)
    palm_valid = str(row.get("proposal_kind", "runtime")) == "runtime"
    row.update(
        {
            "schema_version": TRAIN_SCHEMA if training_stage else EVALUATION_SCHEMA,
            "global_crop_id": roi_id,
            "crop_id": roi_id,
            "source_crop_id": roi_id,
            "source_group_id": capture_id,
            "crop_path": str(raw.get("_absolute_crop_path")),
            "warehouse_crop_relpath": str(row.get("crop_relpath") or row.get("crop_path")),
            "width": 256,
            "height": 256,
            "palm_valid": bool(palm_valid),
            "ignore_for_training": False,
            "selection_action": "include",
            "quality_tier": "HIGH",
            "quality_flags": [],
            "hand_presence_loss_weight": 1.0,
            "landmark_loss_weight": 1.0 if present else 0.0,
            "handedness_loss_weight": 1.0 if present else 0.0,
            "supervision_loss_weight": 1.0,
            "presence_quality_weight": 1.0,
            "landmark_quality_weight": 1.0,
            "handedness_quality_weight": 1.0,
            "sampling_weight": float(row.get("dataset_weight", 1.0)),
            **capture,
        }
    )
    _head_points(row)
    if training_stage:
        origin = str(row.get("label_origin", "mediapipe"))
        tier = "gold" if mix_role == "hard" else "pseudo"
        row.update(
            {
                "training_stage": training_stage,
                "mix_role": mix_role,
                "supervision_tier": tier,
                "annotation_provenance": (
                    "human_gold"
                    if origin in {"human", "mediapipe_human_corrected"}
                    else "mediapipe_pseudo"
                ),
                "sample_type": (
                    "POS_RUNTIME"
                    if present and palm_valid
                    else "POS_LOW_PALM"
                    if present
                    else "NEG_RUNTIME_CANDIDATE"
                    if palm_valid
                    else "NEG_LOW_PALM_CANDIDATE"
                ),
            }
        )
        row["sampling_bucket"] = "{}:{}".format(row["supervision_tier"], row["sample_type"])
    else:
        row["ground_truth_valid"] = True
        row["palm_valid"] = True
    return row


def _assert_membership(rows: Sequence[Mapping[str, Any]]) -> Tuple[List[str], Dict[str, Any]]:
    split_by_capture: Dict[str, set[str]] = defaultdict(set)
    split_by_raw: Dict[str, set[str]] = defaultdict(set)
    variants_by_capture: Dict[str, set[str]] = defaultdict(set)
    performer_splits: Dict[str, set[str]] = defaultdict(set)
    roi_ids: Counter[str] = Counter()
    for row in rows:
        split = str(row.get("split"))
        capture = str(row.get("capture_source_id"))
        raw_id = str(row.get("raw_image_id"))
        variant = str(row.get("proposal_variant"))
        performer = _capture_fields(capture)["performer"]
        split_by_capture[capture].add(split)
        split_by_raw[raw_id].add(split)
        variants_by_capture[capture].add(variant)
        performer_splits[performer].add(split)
        roi_ids[str(row.get("roi_id") or row.get("crop_id"))] += 1
    errors = []
    for label, mapping in (("capture source", split_by_capture), ("raw image", split_by_raw)):
        for identity, splits in mapping.items():
            if len(splits) > 1:
                errors.append("{} {} crosses splits: {}".format(label, identity, sorted(splits)))
    for capture, variants in variants_by_capture.items():
        if len(variants) > 1:
            errors.append(
                "capture source {} selects multiple proposal variants: {}".format(
                    capture, sorted(variants)
                )
            )
    duplicates = sorted(identity for identity, count in roi_ids.items() if count > 1)
    if duplicates:
        errors.append("duplicate roi_id values in run: {}".format(duplicates[:10]))
    warnings = [
        "performer {} crosses splits: {}".format(performer, sorted(splits))
        for performer, splits in sorted(performer_splits.items())
        if len(splits) > 1
    ]
    return warnings, {"errors": errors, "performer_cross_split": warnings}


def _load_stage_rows(
    reader: WarehouseReader, config: Mapping[str, Any], stage: str
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    stage_cfg = dict((config.get("stages") or {}).get(stage) or {})
    datasets = _dataset_entries(stage_cfg, "datasets")
    base_rows: List[Dict[str, Any]] = []
    for entry in datasets:
        base_rows.extend(
            reader.source_rows(
                "pretrain",
                str(entry["dataset_id"]),
                str(entry["proposal_variant"]),
                expected_split="train",
                weight=float(entry["weight"]),
            )
        )
    if any(not _positive(row) for row in base_rows):
        raise WarehouseContractError("PretrainSource training publications must contain positives only")
    if stage == "geometry":
        if stage_cfg.get("negative_datasets"):
            raise WarehouseContractError("geometry forbids negative datasets")
        return [canonical_record(row, "pretrain") for row in base_rows], {"mix": {"positive": len(base_rows)}}
    if stage == "multitask":
        negatives: List[Dict[str, Any]] = []
        for entry in _selected_ids(stage_cfg, "negative_datasets"):
            negative_id = str(entry.get("negative_dataset_id") or "")
            if not negative_id:
                raise WarehouseContractError("negative_datasets entry requires negative_dataset_id")
            negatives.extend(reader.negative_rows(negative_id, float(entry["weight"])))
        if not negatives:
            raise WarehouseContractError("multitask requires at least one published negative dataset")
        rows = [canonical_record(row, "pretrain") for row in base_rows + negatives]
        return rows, {"mix": {"positive": len(base_rows), "negative": len(negatives)}}
    if stage != "multi_finetune":
        raise WarehouseContractError("unsupported stage: {}".format(stage))
    new_recorded: List[Dict[str, Any]] = []
    for entry in _dataset_entries(stage_cfg, "new_datasets"):
        new_recorded.extend(
            reader.source_rows(
                "pretrain",
                str(entry["dataset_id"]),
                str(entry["proposal_variant"]),
                expected_split="train",
                weight=float(entry["weight"]),
            )
        )
    if any(not _positive(row) for row in new_recorded):
        raise WarehouseContractError("multi_finetune new datasets must publish positives only")
    selections: List[Dict[str, Any]] = []
    for entry in _selected_ids(stage_cfg, "selections"):
        selection_id = str(entry.get("selection_id") or "")
        if not selection_id:
            raise WarehouseContractError("selections entry requires selection_id")
        selections.extend(reader.selection_rows(selection_id, float(entry["weight"])))
    if not selections:
        raise WarehouseContractError("multi_finetune requires a published hard-positive selection")
    finetune_negatives: List[Dict[str, Any]] = []
    for entry in _selected_ids(stage_cfg, "negative_datasets"):
        negative_id = str(entry.get("negative_dataset_id") or "")
        if not negative_id:
            raise WarehouseContractError("negative_datasets entry requires negative_dataset_id")
        finetune_negatives.extend(
            reader.negative_rows(negative_id, float(entry["weight"]))
        )
    replay_fraction = float(stage_cfg.get("replay_fraction", 0.45))
    hard_fraction = float(stage_cfg.get("hard_fraction", 0.55))
    if replay_fraction <= 0.0 or hard_fraction <= 0.0 or not math.isclose(
        replay_fraction + hard_fraction, 1.0, abs_tol=1e-9
    ):
        raise WarehouseContractError("multi_finetune hard/replay fractions must be positive and sum to 1")
    hard_source = selections + new_recorded + finetune_negatives
    selected_roi_ids = {
        str(row.get("roi_id") or row.get("crop_id")) for row in hard_source
    }
    replay_source = [
        row
        for row in base_rows
        if str(row.get("roi_id") or row.get("crop_id")) not in selected_roi_ids
    ]
    if not replay_source:
        raise WarehouseContractError(
            "multi_finetune mandatory replay is empty after excluding hard-selection ROIs"
        )
    hard_rows = [canonical_record(row, "finetune", mix_role="hard") for row in hard_source]
    replay_rows = [canonical_record(row, "finetune", mix_role="replay") for row in replay_source]
    for row in hard_rows:
        row["sampling_weight"] *= hard_fraction / float(len(hard_rows))
        row["sampling_bucket"] = "gold:{}".format(row["sample_type"])
    for row in replay_rows:
        row["sampling_weight"] *= replay_fraction / float(len(replay_rows))
    return hard_rows + replay_rows, {
        "mix": {
            "hard": len(hard_rows),
            "hard_selection": len(selections),
            "new_recorded": len(new_recorded),
            "true_negative": len(finetune_negatives),
            "replay": len(replay_rows),
            "hard_fraction": hard_fraction,
            "replay_fraction": replay_fraction,
        }
    }


def _load_evaluation_rows(
    reader: WarehouseReader, config: Mapping[str, Any], split: str
) -> List[Dict[str, Any]]:
    evaluation = dict(config.get("evaluation") or {})
    entries = _dataset_entries(evaluation, split)
    rows: List[Dict[str, Any]] = []
    for entry in entries:
        rows.extend(
            reader.source_rows(
                "eval",
                str(entry["dataset_id"]),
                str(entry["proposal_variant"]),
                expected_split=split,
                weight=float(entry["weight"]),
            )
        )
    return [canonical_record(row, None) for row in rows]


def build_snapshot(
    datasets_config: Mapping[str, Any] | Path | str,
    dataset_root: Path,
    train_root: Path,
    snapshot_id: str,
    stage: str,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Audit selected IDs and write immutable zero-copy JSONL indexes."""

    if stage not in STAGES:
        raise WarehouseContractError("stage must be one of {}".format(STAGES))
    config = load_config(datasets_config) if isinstance(datasets_config, (str, Path)) else dict(datasets_config)
    reader = WarehouseReader(dataset_root)
    train_rows, stage_report = _load_stage_rows(reader, config, stage)
    val_rows = _load_evaluation_rows(reader, config, "val")
    test_rows = _load_evaluation_rows(reader, config, "test")
    all_rows = train_rows + val_rows + test_rows
    warnings, membership = _assert_membership(all_rows)
    if membership["errors"]:
        raise WarehouseContractError("; ".join(membership["errors"]))
    performer_policy = str((config.get("policies") or {}).get("performer_cross_split", "warn"))
    if warnings and performer_policy == "error":
        raise WarehouseContractError("; ".join(warnings))
    if performer_policy not in {"warn", "error"}:
        raise WarehouseContractError("policies.performer_cross_split must be warn or error")
    output = Path(train_root).resolve() / "snapshots" / str(snapshot_id) / stage
    existing = [path for path in (output / "train.jsonl", output / "val.jsonl", output / "test.jsonl") if path.exists()]
    if existing and not overwrite:
        raise FileExistsError("snapshot outputs already exist: {}".format(existing))
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "train.jsonl", train_rows)
    write_jsonl(output / "val.jsonl", val_rows)
    write_jsonl(output / "test.jsonl", test_rows)
    report = {
        "schema_version": SNAPSHOT_SCHEMA,
        "snapshot_id": str(snapshot_id),
        "stage": stage,
        "dataset_root": str(reader.root),
        "image_policy": "zero_copy_read_from_hand_dataset_root",
        "image_content_sha256": "not_computed",
        "records": {"train": len(train_rows), "val": len(val_rows), "test": len(test_rows)},
        "files": {"train": "train.jsonl", "val": "val.jsonl", "test": "test.jsonl"},
        "membership": membership,
        "warnings": warnings,
        **stage_report,
    }
    write_json(output / "snapshot.json", report)
    return report


def verify_snapshot_manifest(
    config: Mapping[str, Any], dataset: Mapping[str, Any], error_type=WarehouseContractError
) -> Dict[str, Any]:
    """Training-loader gate for an HLML 4.0 snapshot (no image hashing)."""

    value = dataset.get("curation_manifest")
    if not value:
        raise error_type("data.curation_manifest is required for warehouse snapshots")
    path = Path(str(value))
    if not path.is_absolute():
        path = Path(str(config.get("_meta", {}).get("repo_root", Path.cwd()))) / path
    try:
        report = _read_json(path.resolve())
    except (OSError, ValueError) as exc:
        raise error_type(str(exc)) from exc
    if report.get("schema_version") != SNAPSHOT_SCHEMA:
        raise error_type("warehouse snapshot schema mismatch: {}".format(path))
    labels = Path(str(dataset.get("labels", ""))).resolve()
    expected = (path.parent / str((report.get("files") or {}).get("train", "train.jsonl"))).resolve()
    if labels != expected:
        raise error_type("data.labels does not match snapshot train index")
    if report.get("image_policy") != "zero_copy_read_from_hand_dataset_root":
        raise error_type("snapshot image policy must remain zero-copy")
    return report


def stage_paths(train_root: Path, snapshot_id: str, stage: str) -> Dict[str, Path]:
    root = Path(train_root).resolve() / "snapshots" / str(snapshot_id) / stage
    return {
        "root": root,
        "manifest": root / "snapshot.json",
        "train": root / "train.jsonl",
        "val": root / "val.jsonl",
        "test": root / "test.jsonl",
    }
