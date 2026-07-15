"""Build deterministic Hand ROI inputs for the model-conversion toolchain.

The canonical JSONL files remain the sole source index.  This module only reads
their already-cropped 256x256 grayscale Hand ROIs and writes independent NPY
artifacts below the configured export directory.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
import uuid
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .config import load_config, resolve_path
from .inspect import audit_canonical_dataset
from .io_utils import read_image, sha256_file, write_json


DATASET_NAMES = ("calibrate_datasets", "evaluate_datasets")
DOCUMENTED_MINIMUMS = {"calibrate_datasets": 20, "evaluate_datasets": 10}
INPUT_SHAPE = (1, 1, 256, 256)
INPUT_DTYPE = "float32"
INPUT_NORMALIZATION = "uint8_div_255"


class ConversionDatasetError(ValueError):
    """Raised when conversion inputs cannot be produced without ambiguity."""


def conversion_datasets_enabled(config: Mapping[str, Any]) -> bool:
    value = config.get("export", {}).get("conversion_datasets", {})
    return isinstance(value, Mapping) and bool(value.get("enabled", False))


def conversion_dataset_output_root(config: Mapping[str, Any]) -> Optional[Path]:
    """Return the independent conversion artifact directory when enabled."""

    if not conversion_datasets_enabled(config):
        return None
    value = config.get("export", {}).get("conversion_datasets", {}).get("output_dir")
    if not value:
        raise ConversionDatasetError(
            "export.conversion_datasets.output_dir is required when generation is enabled"
        )
    output_root = resolve_path(str(value), config).resolve()
    export_config = config.get("export", {})
    model_value = export_config.get("model_path")
    artifact_paths: List[Path] = []
    if model_value:
        model_path = resolve_path(str(model_value), config).resolve()
        artifact_paths.append(model_path)
        contract_value = export_config.get("contract_path")
        artifact_paths.append(
            resolve_path(str(contract_value), config).resolve()
            if contract_value
            else model_path.with_suffix(".contract.json")
        )
    for artifact_path in artifact_paths:
        if (
            output_root == artifact_path
            or output_root in artifact_path.parents
            or artifact_path in output_root.parents
        ):
            raise ConversionDatasetError(
                "export.conversion_datasets.output_dir must be independent of export "
                "model/contract paths: {} conflicts with {}".format(
                    output_root, artifact_path
                )
            )
    return output_root


def guard_conversion_dataset_output(
    config: Mapping[str, Any], overwrite: Optional[bool] = None
) -> None:
    """Fail before ONNX export when an independent dataset output is protected."""

    output_root = conversion_dataset_output_root(config)
    if output_root is None:
        return
    allow_overwrite = (
        bool(config.get("export", {}).get("overwrite", False))
        if overwrite is None
        else bool(overwrite)
    )
    if output_root.exists() and not allow_overwrite:
        raise FileExistsError(
            "Conversion dataset output already exists; set export.overwrite=true "
            "or choose a new export.conversion_datasets.output_dir: {}".format(output_root)
        )


def _field_value(row: Mapping[str, Any], dotted_key: str) -> Any:
    current: Any = row
    for part in dotted_key.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return "<missing>"
        current = current[part]
    if isinstance(current, (Mapping, list, tuple)):
        return json.dumps(current, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return current


def _record_id(row: Mapping[str, Any]) -> str:
    value = row.get("global_crop_id") or row.get("crop_id") or row.get("_resolved_crop_path")
    if value in (None, ""):
        raise ConversionDatasetError("A canonical record has no stable identity")
    return str(value)


def _selection_hash(row: Mapping[str, Any], salt: str) -> str:
    return hashlib.sha256((salt + "\0" + _record_id(row)).encode("utf-8")).hexdigest()


def deterministic_stratified_sample(
    records: Sequence[Mapping[str, Any]],
    count: int,
    stratify_by: Sequence[str],
    salt: str,
) -> List[Dict[str, Any]]:
    """Select a proportional, order-independent sample without an RNG."""

    count = int(count)
    if count < 1:
        raise ConversionDatasetError("Each conversion source count must be positive")
    if count > len(records):
        raise ConversionDatasetError(
            "Requested {} records, but the canonical source contains only {}".format(
                count, len(records)
            )
        )
    fields = [str(value) for value in stratify_by]
    grouped: Dict[Tuple[str, ...], List[Dict[str, Any]]] = defaultdict(list)
    seen_ids = set()
    for source_row in records:
        row = dict(source_row)
        identity = _record_id(row)
        if identity in seen_ids:
            raise ConversionDatasetError(
                "Canonical source contains a duplicate stable identity: {}".format(identity)
            )
        seen_ids.add(identity)
        key = tuple(str(_field_value(row, field)) for field in fields) or ("<all>",)
        grouped[key].append(row)

    total = len(records)
    quotas: Dict[Tuple[str, ...], int] = {}
    remainders: List[Tuple[float, Tuple[str, ...]]] = []
    for key in sorted(grouped):
        ideal = count * len(grouped[key]) / float(total)
        quotas[key] = int(math.floor(ideal))
        remainders.append((ideal - quotas[key], key))
    remaining = count - sum(quotas.values())
    for _fraction, key in sorted(remainders, key=lambda item: (-item[0], item[1]))[:remaining]:
        quotas[key] += 1

    selected: List[Dict[str, Any]] = []
    for key in sorted(grouped):
        ranked = sorted(
            grouped[key],
            key=lambda row: (_selection_hash(row, salt), _record_id(row)),
        )
        selected.extend(ranked[: quotas[key]])
    selected.sort(key=lambda row: (_selection_hash(row, salt), _record_id(row)))
    if len(selected) != count:
        raise AssertionError("Deterministic sampler produced an invalid count")
    return selected


def _validate_input_contract(config: Mapping[str, Any]) -> None:
    input_config = config.get("input", {})
    if not isinstance(input_config, Mapping):
        raise ConversionDatasetError("export.conversion_datasets.input must be a mapping")
    shape = tuple(int(value) for value in input_config.get("shape", INPUT_SHAPE))
    layout = str(input_config.get("layout", "NCHW")).upper()
    dtype = str(input_config.get("dtype", INPUT_DTYPE)).lower()
    normalization = str(input_config.get("normalization", INPUT_NORMALIZATION)).lower()
    if shape != INPUT_SHAPE:
        raise ConversionDatasetError(
            "Conversion input shape must remain NCHW (1,1,256,256); got {}".format(shape)
        )
    if layout != "NCHW":
        raise ConversionDatasetError("Conversion input layout must remain NCHW")
    if dtype != INPUT_DTYPE:
        raise ConversionDatasetError("Conversion input dtype must remain float32")
    if normalization != INPUT_NORMALIZATION:
        raise ConversionDatasetError(
            "Conversion input normalization must remain uint8_div_255"
        )


def _load_and_audit_sources(
    config: Mapping[str, Any],
    conversion_config: Mapping[str, Any],
    output_root: Path,
) -> Tuple[Dict[str, List[Dict[str, Any]]], List[Dict[str, Any]]]:
    sets_config = conversion_config.get("sets", {})
    if not isinstance(sets_config, Mapping):
        raise ConversionDatasetError("export.conversion_datasets.sets must be a mapping")
    export_stage = str(config.get("model", {}).get("checkpoint_stage", ""))
    if export_stage not in {"pretrain", "finetune"}:
        raise ConversionDatasetError(
            "model.checkpoint_stage must be pretrain or finetune for conversion datasets"
        )
    selected_by_set: Dict[str, List[Dict[str, Any]]] = {}
    source_reports: List[Dict[str, Any]] = []
    observed_eval_splits = set()

    for dataset_name in DATASET_NAMES:
        dataset_config = sets_config.get(dataset_name, {})
        if not isinstance(dataset_config, Mapping):
            raise ConversionDatasetError(
                "export.conversion_datasets.sets.{} must be a mapping".format(dataset_name)
            )
        minimum = int(dataset_config.get("minimum_count", DOCUMENTED_MINIMUMS[dataset_name]))
        documented_minimum = DOCUMENTED_MINIMUMS[dataset_name]
        if minimum < documented_minimum:
            raise ConversionDatasetError(
                "{} minimum_count cannot be below the documented minimum {}".format(
                    dataset_name, documented_minimum
                )
            )
        sources = dataset_config.get("sources", {})
        if not isinstance(sources, Mapping) or not sources:
            raise ConversionDatasetError("{} requires one or more named sources".format(dataset_name))

        set_rows: List[Dict[str, Any]] = []
        for source_name, source_value in sources.items():
            if not isinstance(source_value, Mapping):
                raise ConversionDatasetError(
                    "Conversion source {} must be a mapping".format(source_name)
                )
            source_path_value = source_value.get("config_path")
            if not source_path_value:
                raise ConversionDatasetError(
                    "Conversion source {} requires config_path".format(source_name)
                )
            source_path = resolve_path(str(source_path_value), config)
            source_config = load_config(source_path)
            source_task = str(source_config.get("task", "")).lower()
            expected_stage: Optional[str] = None
            expected_split: Optional[str] = None
            if dataset_name == "calibrate_datasets":
                if source_task != "train":
                    raise ConversionDatasetError(
                        "Calibration inputs must come from a train config; {} uses task {}".format(
                            source_name, source_task
                        )
                    )
                expected_stage = str(source_config.get("stage", ""))
                if expected_stage != export_stage:
                    raise ConversionDatasetError(
                        "Calibration source {} stage {} does not match export stage {}".format(
                            source_name, expected_stage, export_stage
                        )
                    )
                source_model_stage = str(
                    source_config.get("model", {}).get("checkpoint_stage", "")
                )
                if source_model_stage != expected_stage:
                    raise ConversionDatasetError(
                        "Calibration source {} model checkpoint stage {} does not match "
                        "its train stage {}".format(
                            source_name, source_model_stage, expected_stage
                        )
                    )
            else:
                if source_task != "evaluate":
                    raise ConversionDatasetError(
                        "Evaluation inputs must come from an evaluate config; {} uses task {}".format(
                            source_name, source_task
                        )
                    )
                expected_split = str(source_config.get("split", ""))
                if expected_split not in {"val", "test"}:
                    raise ConversionDatasetError(
                        "Evaluation source {} must declare split val or test".format(source_name)
                    )
                observed_eval_splits.add(expected_split)
                source_model_stage = str(
                    source_config.get("model", {}).get("checkpoint_stage", "")
                )
                if source_model_stage != export_stage:
                    raise ConversionDatasetError(
                        "Evaluation source {} checkpoint stage {} does not match export stage {}".format(
                            source_name, source_model_stage, export_stage
                        )
                    )

            rows, audit = audit_canonical_dataset(
                source_config,
                dataset=source_config.get("data", {}),
                expected_stage=expected_stage,
                expected_split=expected_split,
                # Validate every canonical row and resolve every crop path, but
                # avoid decoding the entire training corpus during export. The
                # selected ROIs are decoded and checked strictly before np.save.
                check_images=False,
                hash_images=False,
                raise_on_error=True,
            )
            protected_directories = set()
            labels_value = audit.get("labels")
            if labels_value:
                protected_directories.add(Path(str(labels_value)).resolve().parent)
            protected_directories.update(
                Path(str(row["_resolved_crop_path"])).resolve().parent for row in rows
            )
            for protected in protected_directories:
                if (
                    output_root == protected
                    or output_root in protected.parents
                    or protected in output_root.parents
                ):
                    raise ConversionDatasetError(
                        "Conversion output must not overlap a canonical label/ROI directory: "
                        "{} conflicts with {}".format(output_root, protected)
                    )
            source_count = int(source_value.get("count", 0))
            stratify_by = source_value.get("stratify_by", [])
            if not isinstance(stratify_by, (list, tuple)):
                raise ConversionDatasetError(
                    "Conversion source {} stratify_by must be a list".format(source_name)
                )
            salt = "{}\0{}\0{}".format(
                conversion_config.get("selection_salt", "hand-landmarker-conversion-v1"),
                dataset_name,
                source_name,
            )
            selected = deterministic_stratified_sample(rows, source_count, stratify_by, salt)
            for row in selected:
                row["_conversion_source"] = str(source_name)
                row["_conversion_source_config"] = str(source_path)
                row["_conversion_stratify_by"] = [str(value) for value in stratify_by]
            set_rows.extend(selected)
            source_reports.append(
                {
                    "dataset": dataset_name,
                    "source": str(source_name),
                    "task": source_task,
                    "stage": expected_stage or source_config.get("model", {}).get("checkpoint_stage"),
                    "split": expected_split,
                    "config_path": str(source_path),
                    "labels_path": audit.get("labels"),
                    "labels_sha256": audit.get("labels_sha256"),
                    "canonical_records": len(rows),
                    "selected_records": len(selected),
                    "stratify_by": [str(value) for value in stratify_by],
                }
            )
        if len(set_rows) < minimum:
            raise ConversionDatasetError(
                "{} requires at least {} files; configured sources produce {}".format(
                    dataset_name, minimum, len(set_rows)
                )
            )
        selected_by_set[dataset_name] = set_rows

    if observed_eval_splits != {"val", "test"}:
        raise ConversionDatasetError(
            "evaluate_datasets must sample both canonical Val and Test; got {}".format(
                sorted(observed_eval_splits)
            )
        )
    return selected_by_set, source_reports


def _model_input_tensor(path: Path):
    import numpy as np

    image = read_image(path)
    if image is None:
        raise ConversionDatasetError("Canonical Hand ROI is unreadable: {}".format(path))
    if image.ndim == 3 and image.shape[2] == 1:
        image = image[:, :, 0]
    if image.ndim != 2:
        raise ConversionDatasetError(
            "Canonical Hand ROI must already be single-channel grayscale: {} has shape {}".format(
                path, image.shape
            )
        )
    if tuple(image.shape) != (256, 256):
        raise ConversionDatasetError(
            "Canonical Hand ROI must already be 256x256; {} has shape {}".format(
                path, image.shape
            )
        )
    if image.dtype != np.uint8:
        raise ConversionDatasetError(
            "Canonical Hand ROI must be uint8 before /255 normalization: {} is {}".format(
                path, image.dtype
            )
        )
    return (image.astype(np.float32) / np.float32(255.0))[None, None, :, :]


def _write_deterministic_archive(datasets_dir: Path, archive_path: Path) -> None:
    with zipfile.ZipFile(
        str(archive_path), "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for path in sorted(datasets_dir.rglob("*.npy"), key=lambda value: value.as_posix()):
            relative = Path("datasets") / path.relative_to(datasets_dir)
            info = zipfile.ZipInfo(relative.as_posix(), date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes(), compress_type=zipfile.ZIP_DEFLATED)


def _publish_directory(staging: Path, output_root: Path, overwrite: bool) -> None:
    backup: Optional[Path] = None
    if output_root.exists():
        if not overwrite:
            raise FileExistsError("Conversion dataset output already exists: {}".format(output_root))
        backup = output_root.with_name(
            ".{}.backup.{}".format(output_root.name, uuid.uuid4().hex)
        )
        os.replace(str(output_root), str(backup))
    try:
        os.replace(str(staging), str(output_root))
    except Exception:
        if backup is not None and backup.exists() and not output_root.exists():
            os.replace(str(backup), str(output_root))
        raise
    if backup is not None and backup.exists():
        shutil.rmtree(str(backup))


def generate_conversion_datasets(config: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """Generate, verify, package, and atomically publish conversion inputs."""

    if not conversion_datasets_enabled(config):
        return None
    conversion_config = config.get("export", {}).get("conversion_datasets", {})
    assert isinstance(conversion_config, Mapping)
    _validate_input_contract(conversion_config)
    output_root = conversion_dataset_output_root(config)
    assert output_root is not None
    overwrite = bool(config.get("export", {}).get("overwrite", False))
    guard_conversion_dataset_output(config, overwrite=overwrite)
    selected_by_set, source_reports = _load_and_audit_sources(
        config, conversion_config, output_root
    )

    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=".{}.tmp.".format(output_root.name), dir=str(output_root.parent))
    )
    try:
        datasets_dir = staging / "datasets"
        manifest_files: List[Dict[str, Any]] = []
        for dataset_name in DATASET_NAMES:
            target_dir = datasets_dir / dataset_name
            target_dir.mkdir(parents=True, exist_ok=False)
            for index, row in enumerate(selected_by_set[dataset_name], start=1):
                source_image = Path(str(row["_resolved_crop_path"])).resolve()
                tensor = _model_input_tensor(source_image)
                output_file = target_dir / "img_{:04d}.npy".format(index)
                import numpy as np

                np.save(str(output_file), tensor, allow_pickle=False)
                reloaded = np.load(str(output_file), allow_pickle=False)
                if reloaded.shape != INPUT_SHAPE or reloaded.dtype != np.float32:
                    raise ConversionDatasetError(
                        "Generated NPY contract check failed: {}".format(output_file)
                    )
                relative = output_file.relative_to(staging).as_posix()
                stratify_by = row.get("_conversion_stratify_by", [])
                manifest_files.append(
                    {
                        "dataset": dataset_name,
                        "relative_path": relative,
                        "source": row.get("_conversion_source"),
                        "source_config": row.get("_conversion_source_config"),
                        "record_id": _record_id(row),
                        "dataset_id": row.get("dataset_id"),
                        "source_crop_path": str(source_image),
                        "source_image_sha256": sha256_file(source_image),
                        "selection_stratum": {
                            str(field): _field_value(row, str(field)) for field in stratify_by
                        },
                        "npy_sha256": sha256_file(output_file),
                        "shape": list(INPUT_SHAPE),
                        "dtype": INPUT_DTYPE,
                    }
                )

        actual_entries = sorted(
            path.relative_to(datasets_dir).as_posix()
            for path in datasets_dir.rglob("*")
            if path.is_file()
        )
        if any(not entry.endswith(".npy") for entry in actual_entries):
            raise ConversionDatasetError("datasets/ contains a non-NPY file")
        expected_count = sum(len(rows) for rows in selected_by_set.values())
        if len(actual_entries) != expected_count:
            raise ConversionDatasetError("Generated datasets/ file count is inconsistent")

        manifest = {
            "schema_version": 1,
            "selection_method": "proportional_strata_then_sha256_rank_no_rng",
            "selection_salt": conversion_config.get(
                "selection_salt", "hand-landmarker-conversion-v1"
            ),
            "input_contract": {
                "shape": list(INPUT_SHAPE),
                "layout": "NCHW",
                "dtype": INPUT_DTYPE,
                "normalization": INPUT_NORMALIZATION,
                "source": "canonical 256x256 uint8 grayscale Hand ROI",
                "contains_model_outputs": False,
            },
            "files": manifest_files,
        }
        write_json(staging / "datasets_manifest.json", manifest)
        archive_path = staging / "datasets.zip"
        _write_deterministic_archive(datasets_dir, archive_path)

        counts = {
            name: len(selected_by_set[name]) for name in DATASET_NAMES
        }
        final_archive = output_root / "datasets.zip"
        report: Dict[str, Any] = {
            "status": "ok",
            "model_checkpoint_stage": config.get("model", {}).get("checkpoint_stage"),
            "output_root": str(output_root),
            "datasets_dir": str(output_root / "datasets"),
            "archive_path": str(final_archive),
            "archive_sha256": sha256_file(archive_path),
            "manifest_path": str(output_root / "datasets_manifest.json"),
            "counts": counts,
            "minimum_counts": dict(DOCUMENTED_MINIMUMS),
            "source_reports": source_reports,
            "input_contract": manifest["input_contract"],
        }
        write_json(staging / "datasets_report.json", report)
        _publish_directory(staging, output_root, overwrite=overwrite)
        report["report_path"] = str(output_root / "datasets_report.json")
        return report
    finally:
        if staging.exists():
            shutil.rmtree(str(staging))
